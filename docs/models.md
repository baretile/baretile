# Models, serving and adapters

The code in [models/](../models) runs pretrained models on baretile and serves
them over HTTP. [adapters/](../adapters) lets tinygrad scripts, and PyTorch
through tinygrad, run on baretile. These are layer 4: they use `Tensor` and add
the kernels that only they need.

## Running a model

`models/gpt2.py` and `models/llama.py` can run the same model operations with
baretile tensors or NumPy arrays. The generation scripts compare the two at
each step. They feed NumPy's chosen token to both versions on the next step,
so one disagreement won't make the rest of the comparison useless.

`llama.load` reads a local Hugging Face config and safetensors weights. It
handles dense Llama-family models, several mixtures of experts, and the text
models of Qwen3.5 and Qwen3.6. The tokenizer reads byte-level BPE data from
`tokenizer.json`. It uses the supplied merge order and `ignore_merges` setting;
files without a merge list use token IDs as ranks, as in tiktoken. Unsupported
BPE dropout, subword affixes and byte fallback are rejected when loading.
The standalone GPT-2 command also needs `tiktoken`.
From Python in a checkout, import these helpers with `from models import llama`
or `from models import serve`. Models and adapters are outside the PyPI package.

Choose the precision as the fourth argument after the script name:

```sh
python3 models/llama.py /path/to/SmolLM-135M "Hello" 6 bfp8
python3 models/gpt2.py /path/to/gpt2/model.safetensors "Hello" 6 bf16
```

`bf16` is the default. `bfp8` stores weight matrices in bfp8. The Llama `bfp4`
mode puts gate and up projections in bfp4 and the other matrices in bfp8.
`tt` follows Tenstorrent's performance configuration, including its fidelity
settings. `experts` puts the MLP matrices in bfp4, the other matrices in bfp8,
and the KV cache in bfp8. `llama.precision` has the settings for each operation.
We haven't measured Qwen3.6 token accuracy with all matrix weights in bfp8 or
with a bfp4 KV cache.

The loader fuses q, k and v projections, and gate and up projections, where
appropriate. Matmul weights are stored by tile columns. The KV cache is
allocated once and zeroed on the chip, and each token writes its keys and values
at its position.

## Experts and Gated DeltaNet

`models/qwen.py` contains the operations that extend a Llama block. Expert
routing uses `topk` on the chip. During decode, `touched` lists the selected
experts so tokens can share their weight reads. For longer prompts, `group`
sorts tokens by expert.

Qwen3.5 and Qwen3.6 combine attention with Gated DeltaNet. A DeltaNet block keeps
convolution history and a fixed-size recurrent state. Its decode kernel performs
the convolution, gates and state update in fp32. Prompt processing works in
pieces of 64 tokens. State updates are written in place so later replays see
the new state.

Models with a multi-token prediction layer can draft a token and verify it with
the next token. Verification keeps state after the first token and after both
tokens. The next step reads whichever state matches the accepted tokens.
Free server slots leave their recurrent states unchanged. DeltaNet decode requires
key and value head dimensions divisible by 32, and validates speculative state
shapes before launching. Truncating a hybrid to only its recurrent blocks is
supported; these stacks use the RoPE tables for position capacity, or the
server's explicit slot capacity.

Expert routing rejects chunks whose choices and expert list do not fit worker
L1. Reduce the prefill chunk size for these layouts.

## Serving

The server accepts a folder or Hugging Face model ID. Its positional arguments
are model, port, slots, positions, precision and optional KV format:

```sh
python3 models/serve.py HuggingFaceTB/SmolLM-135M 8000 4 512 bf16
```

Here, four requests can run at once, with 512 positions per slot. The slot
capacity must fit both loaded RoPE tables; construction rejects a larger value.
Choose a multiple of 32 for that size, and leave room for the answer as well as the
prompt. The engine also reserves positions at the end for inactive slots.

Downloads go to `~/.cache/baretile/hub`. `HF_TOKEN` supplies access to a gated
model, and `HF_ENDPOINT` changes the download endpoint. Redirects retain that
credential only within the same origin, including its scheme and port. Completed
downloads are reused. The server binds to all interfaces on the selected port and has no API
key check.

It serves `GET /v1/models`, `POST /v1/completions` and
`POST /v1/chat/completions`. For the example above:

```sh
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"SmolLM-135M","prompt":"The meaning of life is","max_tokens":6,"temperature":0}'
```

Set `stream` to true in the request to receive tokens as they're generated.
You can set temperature, top_k, top_p, min_p, seed, stop strings and repetition
penalties. Defaults come from the model's `generation_config.json` when present.
Prompts must contain at least one token. `max_tokens` must be a positive integer;
`top_k=0` keeps all tokens. Temperature must be nonnegative, `top_p` in `(0, 1]`,
`min_p` in `[0, 1]`, and repetition penalties positive. Nonzero temperatures
must have an inverse representable as a normal fp32 value. Penalties must fit
fp32, and repetition penalties must also be normal. Invalid requests get HTTP
400 before they take a slot.

Each connection has 10 seconds to send its headers and body. Bodies require one
nonnegative `Content-Length`, may contain at most 8 MiB, and cannot use chunked
transfer encoding. Expired bodies get HTTP 408; oversized bodies get HTTP 413.
At most 32 connections have active handlers; additional connections get HTTP
503. `serve.server` exposes these limits as `input_wait`, `body_limit` and
`connections`.

EOS tokens and sampling defaults are resolved per field from the generation
config, then the selected text-model config, then the outer model config. An
explicit null EOS disables automatic end tokens.

The engine gives each request a slot in a shared KV cache and decodes occupied
slots together. Prompts are processed in chunks between decode steps. A request
gets HTTP 429 when no slot is free. A non-streaming request that receives no text
for 600 seconds gets HTTP 504. A stream that times out releases its slot.
A sampling failure ends the affected request with an error and frees its slot;
other requests continue. If the engine fails, pending requests end with an error
and new requests get HTTP 503.

## Reusing a conversation

When a prompt arrives, the engine looks for a free slot that has already
processed a prefix of it. It can also restore a saved session with its tokens,
KV cache and recurrent state. DeltaNet must resume from the exact tokens that
produced its state.

Sessions are saved in spare DRAM on the chip. The server evicts old sessions
before allocating a new snapshot. When device space is insufficient, it reads
the snapshot directly into host memory. The host cache has a size limit too; a
snapshot that cannot fit is dropped without failing the request. Prompt snapshots
include the prediction layer's final cache update when speculation is enabled.
The response's usage field counts the prompt tokens that were reused.

## Sampling and API limits

For supported vocabulary shapes, the device returns the largest 64 logits and
the normalization sum. Greedy selection and top_k up to 64 can use that result.
Penalties, logprobs and constrained output need the full row. Some other
sampling requests do too, so they can read much more data back to the host.

Chat templates are written in Python, so a model with a new template may need
code in `serve.py`. Qwen tool calls are returned in `tool_calls`, and reasoning
goes in `reasoning_content`. Streams buffer partial reasoning markers so delimiter
characters do not leak into either field. `tool_choice` and JSON response formats constrain
which tokens can be sampled. `test_chat_tools` and `test_serve` have example
requests to start from when adding a model or client.
JSON schemas support types, properties, required fields, additional properties,
arrays, enums, constants, alternatives and nonrecursive local references.
Sibling constraints intersect, including fields beside `$ref`, following the
[JSON Schema 2020-12 reference rules](https://json-schema.org/draft/2020-12/json-schema-core#section-8.2.3.1).
Enums and constants match decoded JSON values; strings accept literal Unicode
and JSON escapes. A token may contain only part of a UTF-8 character. Generation
keeps those bytes pending and reports an error if no token can complete the
constraint. It does not label an incomplete constrained answer as a successful
stop.

`serve.formatted` documents the supported subset. String patterns, formats and
numeric bounds are not enforced. Generation allows at most one whitespace
character between tokens and 32 characters per number; integer schemas use
spellings without fractions or exponents.

## tinygrad and PyTorch

Import `adapters.tinygrad_baretile` before using tinygrad's `BARETILE` device.
The adapter translates tinygrad's graph into baretile tensors before tinygrad
schedules kernels. It recognizes a broadcast multiply followed by a reduction
as a matmul. `TinyJit` uses baretile recordings when the call can be recorded.

You can run an existing tinygrad script through the adapter:

```sh
python3 adapters/tinygrad_baretile.py /path/to/script.py
```

The adapter calls tinygrad's internal APIs, so a tinygrad update can break it.
CI tests it on the tinygrad commit pinned in
[.github/workflows/test.yml](../.github/workflows/test.yml). Put a checkout of
that commit on `PYTHONPATH`, or in `TINYGRAD` for the tests.

Unsupported operations fall back to tinygrad's CPU device. These include
64-bit integers, bit operations, random-number generation and symbolic shapes.
The adapter preserves the logical dtype through fallback; uint16 values and
bfloat16 storage use different conversions, and boolean results stay 0 or 1.

`TinyJit` captures the second eligible call into private input buffers, so
swapping inputs between calls does not overwrite the caller's tensors. Results
reuse their recorded storage on subsequent calls. Assigning a captured BARETILE
weight or other buffer invalidates the recording; the next eligible call records
again. A mixed-device call also discards the recording before running normally.
Calls that materialize data on the host, bind symbolic variables, pass repeated
or partial input buffers, or store into input buffers execute normally. Discarded
recordings release their allocations after queued work completes. Keep captured
CPU tensors constant, or pass changing values as BARETILE inputs.

PyTorch uses tinygrad's `extra/torch_backend`, which needs a tinygrad checkout,
PyTorch, a C++ compiler and ninja. `test_torch_frontend` exercises that path.

## Estimating cost

`models/cost.py` estimates memory use and serving cost from a model config:

```sh
python3 models/cost.py /path/to/Qwen3.6-35B-A3B experts context=16384 accept=0.55
```

Add `probe=probe.json` once you've measured a card. Until then, the model assumes
that decode gets 70% of the specified 512 GB/s DRAM bandwidth and that a launch
takes one microsecond. Those two assumptions are guesses. Compute uses the
fidelity phases selected for each operation by the precision mode, including
mixed TT/experts modes and attention's separate decode and prefill settings.
`phases=number` overrides all operations with that uniform assumption; the
report identifies this override. Prompt chunks charge both the main and draft
vocabulary projections once per chunk, including the final short chunk.
Memory estimates reserve the full context for each attention cache, including
sliding attention, matching the current allocator. If the weights leave no room
for a session, the report says so and omits throughput estimates.

The model counts the weights, caches, states and activations that a step reads.
It estimates how long those reads and the matrix operations take, uses the
longer time, then adds the cost of launches and RISC-V control code. The number
of different experts selected and the fraction of drafts accepted will change
with your workload.
