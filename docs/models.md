# Models, serving and adapters

The code in [baretile/models/](../baretile/models) runs pretrained models on
baretile and serves them over HTTP. [baretile/adapters/](../baretile/adapters)
lets tinygrad scripts, and PyTorch through tinygrad, run on baretile. These are
layer 4: they use `Tensor` and add the kernels that only they need.

The [first P150a workload](workload.md) fixes the initial release target,
checkpoint revisions and acceptance checks. The capabilities described here
are broader than that target; hardware qualification is still pending.

## Running a model

`baretile/models/gpt2.py` and `baretile/models/llama.py` can run the same model
operations with baretile tensors or NumPy arrays. The generation scripts compare
the two at each step. They feed NumPy's chosen token to both versions on the next
step, so one disagreement won't make the rest of the comparison useless.

`llama.load` reads a local Hugging Face config and safetensors weights. It runs
these families, by the `model_type` in `config.json`, or in its `text_config` for
a model that also sees images:

| Family | `model_type` | For example | Checked on ttsim |
| --- | --- | --- | --- |
| Llama 3, SmolLM | `llama` | Llama-3.1-8B-Instruct, SmolLM-135M | Against transformers |
| Mistral | `mistral` | Mistral-7B | Against transformers |
| Mixtral | `mixtral` | Mixtral-8x7B | Against transformers |
| Qwen2, Qwen2.5 | `qwen2` | Qwen2.5-0.5B | Against transformers |
| Qwen2-MoE | `qwen2_moe` | Qwen1.5-MoE-A2.7B | Against transformers |
| Qwen3 | `qwen3` | Qwen3-0.6B | Against transformers |
| Qwen3-MoE | `qwen3_moe` | Qwen3-30B-A3B | Against transformers |
| Qwen3.5, Qwen3.6 | `qwen3_5_moe_text`, `qwen3_5_text` | Qwen3.6-35B-A3B | The mixtures of experts against transformers; the dense models not compared |
| gpt-oss | `gpt_oss` | gpt-oss-20b | Against transformers |

The comparisons use small models with random weights, in `test_hf_families`,
`test_qwen3_moe` and `test_qwen3_5_moe`. They need PyTorch and transformers,
and skip without them. SmolLM-135M's real checkpoint is checked against NumPy
when `LLAMA` names it (`test_llama_model`). None of these models has run on a
card yet. Another `model_type` fails at load with a `ValueError` that names it,
since a family such as Gemma has weights of the same names but computes
differently.

The tokenizer reads byte-level BPE data from
`tokenizer.json`. It uses the file's merge order and `ignore_merges` setting;
files without a merge list use token IDs as ranks, as in tiktoken. Split, Digits
and ByteLevel pre-tokenizers run in their configured order. Bytes missing from
the alphabet are skipped before merging, as Hugging Face BPE does with no unknown
token. Loading raises `ValueError` for BPE dropout, subword affixes, byte
fallback, an unknown token or ByteLevel's prefix space, which the tokenizer
doesn't implement. The standalone GPT-2 command also needs `tiktoken`.
Import them as `from baretile.models import llama` or `from baretile.models import
serve`. They need NumPy, which `pip install 'baretile[models]'` installs.

Choose the precision as the fourth argument after the script name:

```sh
python3 -m baretile.models.llama /path/to/SmolLM-135M "Hello" 6 bfp8
python3 -m baretile.models.gpt2 /path/to/gpt2/model.safetensors "Hello" 6 bf16
```

`bf16` is the default. `bfp8` stores weight matrices in bfp8. The Llama `bfp4`
mode puts gate and up projections in bfp4 and the other matrices in bfp8.
`tt` follows Tenstorrent's performance configuration, including its fidelity
settings. `experts` puts the MLP matrices in bfp4, the other matrices in bfp8,
and the KV cache in bfp8. `llama.precision` has the settings for each operation.
We haven't measured Qwen3.6 token accuracy with all matrix weights in bfp8 or
with a bfp4 KV cache.

In every mode, the Llama families keep the residual stream between blocks in
fp32. Each half of a block adds its last matmul's fp32 sum to it, and each norm
reads it rounded to bf16. In bf16, a large activation that a later block cancels
would lose the low bits of what's left: SmolLM-135M has one of about 31,000,
which a bf16 copy holds only to within 128. GPT-2's stream is still bf16.

Qwen3 and Qwen3.5 RMS-norm each head of q and k before RoPE. The norm reads the
qkv matmul's fp32 sum, runs in fp32 and rounds once to bf16 for RoPE. In bf16,
the norm's scale would round once for a whole head, and a score doesn't average
that error out over its products, as it does the roundings of single values.
Qwen3-0.6B's norm weights make scores near 48, where a rounding of 2^-8 is 0.19
([#70](https://github.com/baretile/baretile/issues/70)).

The loader fuses q, k and v projections, and gate and up projections, where
appropriate. Matmul weights are stored by tile columns. The KV cache is
allocated once and zeroed on the chip, and each token writes its keys and values
at its position.

## Experts and Gated DeltaNet

`baretile/models/qwen.py` contains the operations that extend a Llama block. Expert
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
Free server slots leave their recurrent states unchanged. DeltaNet needs key and
value head dimensions that are multiples of 32. A hybrid cut down to its
recurrent blocks also runs. With no attention cache to give the positions per
sequence, `llama.forward` takes them from its `capacity` argument, which the
server passes, or else from the RoPE tables.

Expert routing holds a prompt chunk's choices in one worker's L1, so it raises
`ValueError` for a chunk too large for that, several thousand tokens. The
server's chunks of 2,048 tokens fit. From Python, `serve.Engine(chunk=...)` and
`llama.prefill`'s `tokens` set the chunk.

## Serving

The server accepts a folder or Hugging Face model ID. Its positional arguments
are model, port, slots, positions, precision, optional KV format and optional
connection limit:

```sh
python3 -m baretile.models.serve HuggingFaceTB/SmolLM-135M 8000 4 512 bf16
```

Here, four requests can run at once, with 512 positions per slot. Choose a
multiple of 32 that the model's RoPE tables cover, and leave room for the answer
as well as the prompt. The engine also reserves positions at the end for
inactive slots.

Downloads go to `$XDG_CACHE_HOME/baretile/hub`, `~/.cache/baretile/hub` by default. `HF_TOKEN` supplies access to a gated
model, and `HF_ENDPOINT` changes the download endpoint. A redirect to another
origin (scheme, host or port) doesn't get the token. Completed downloads are
reused. The server binds to all interfaces on the
selected port and has no API key check.

It serves `GET /v1/models`, `POST /v1/completions` and
`POST /v1/chat/completions`. For the example above:

```sh
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"SmolLM-135M","prompt":"The meaning of life is","max_tokens":6,"temperature":0}'
```

Set `stream` to true in the request to receive tokens as they're generated, and
`stream_options.include_usage` to also get the token counts in a last chunk
whose `choices` is empty. You can set temperature, top_k, top_p, min_p, seed,
stop strings and repetition
penalties. Defaults come from the model's `generation_config.json` when present.
Prompts must contain at least one token. `max_tokens` must be a positive integer;
a chat request can name it `max_completion_tokens`, which takes precedence over
`max_tokens`. A message's content is a string or a list of text parts.
`top_k=0` keeps all tokens. Temperature must be nonnegative, `top_p` in `(0, 1]`
and `min_p` in `[0, 1]`. The chip takes the inverse of a nonzero temperature as
fp32, so it must be a normal fp32 value. Presence and frequency penalties must
fit fp32, and a repetition penalty must be a positive normal fp32 value, so
penalized logits stay finite. Invalid requests get HTTP 400 before they take a
slot.

A connection has 10 seconds to send its headers and body, or it gets HTTP 408.
The body needs one `Content-Length` and no chunked transfer encoding (HTTP 400),
and at most 8 MiB (HTTP 413). Each connection has a handler thread. By default
at most 4 × (slots + 8) connections have handlers, 48 for four slots, and one
client address may hold a quarter of them, which is more than the slots. So one
client can use every slot, but its idle sockets can't take every handler. Past
the total a connection gets HTTP 503, and past its address's share HTTP 429,
each with a JSON error that names the limit. The seventh argument sets the
total, for example
`python3 -m baretile.models.serve HuggingFaceTB/SmolLM-135M 8000 4 512 bf16 "" 64`, where
`""` keeps the precision's KV format. Behind a reverse proxy every client has the
proxy's address, so they share a quarter of the handlers. To give that address
all of them, start the server from Python as `serve.main` does, and pass
`per_client` equal to `connections` to `serve.server`. It also takes
`input_wait` and `body_limit`.

EOS tokens and sampling defaults come from `generation_config.json`, then from
`text_config` in `config.json` (a multimodal model's text model), then from the
rest of `config.json`, field by field. An explicit null EOS means no automatic
end token.

The engine gives each request a slot in a shared KV cache and decodes occupied
slots together. Prompts are processed in chunks between decode steps. A request
gets HTTP 429 when no slot is free. A non-streaming request that receives no text
for 600 seconds gets HTTP 504. A request that times out, or whose client hangs
up, is cancelled. The handler checks for hangups at least once a second; the
engine releases the slot when it next handles that request.
A failure in a request's own work on the host, such as sampling or checking its
output constraint, ends that request with an error and frees its slot; other
requests continue. A failure on the chip, including reading logits back, stops
the engine: pending requests end with an error and new requests get HTTP 503.

## Reusing a conversation

When a prompt arrives, the engine looks for a free slot that has already
processed a prefix of it. It can also restore a saved session with its tokens,
KV cache and recurrent state. DeltaNet must resume from the exact tokens that
produced its state.

Sessions are saved in spare DRAM on the chip. To make room for a new one, the
server moves the least recently used sessions to host memory. A session that
doesn't fit in DRAM is saved straight to the host. The host cache has a size
limit too, past which the oldest sessions are dropped; a session too large for
it isn't saved, and its request still succeeds. Restoring a session from host
memory uploads all of it before it changes the slot. If DRAM can't hold it, the
server frees what it uploaded, drops the session and prefills the prompt from
the start. With speculation, a session saved after its prompt includes the
prediction layer's cache rows for that prompt. The response's usage field counts
the prompt tokens that were reused.

## Sampling and API limits

For supported vocabulary shapes, the device returns the largest 64 logits and
the normalization sum. Greedy selection and top_k up to 64 can use that result.
Penalties, logprobs and constrained output need the full row. Some other
sampling requests do too, so they can read much more data back to the host.

Chat templates are written in Python, so a model with a new template may need
code in `serve.py`. Qwen tool calls are returned in `tool_calls`, and reasoning
goes in `reasoning_content`. Each tool needs a function with a nonempty name.
Its `parameters`, if not null, must be an object, with `properties` mapping
names to schemas. Other tool shapes get HTTP 400, as do bodies nested too deeply
for Python's JSON parser. A stream holds back text that may begin a reasoning or
tool-call marker until the marker is complete or ruled out. Once generation ends,
the whole answer stays in the content if no tool call parses. Otherwise, content
contains the text before the first tool-call marker. `tool_choice` and
JSON response formats constrain which tokens can be sampled. `test_chat_tools`,
`test_serve_http`, `test_serve_schema` and `test_serve_chat` have example
requests to start from when adding a model or client.
JSON schemas support types, properties, required fields, additional properties,
arrays, enums, constants, alternatives and nonrecursive local references.
Sibling constraints intersect, including fields beside `$ref`, following the
[JSON Schema 2020-12 reference rules](https://json-schema.org/draft/2020-12/json-schema-core#section-8.2.3.1).
Enums and constants match decoded JSON values; strings accept literal Unicode
and JSON escapes. A token may end partway through a UTF-8 character, and the
constraint checks that the character can still complete. If no token can
continue a constrained answer, or it stops before it's complete (at a stop
string, say), the request ends with an error: HTTP 500, or `finish_reason`
`error` in a stream.

`serve.formatted` documents the supported subset. String patterns, formats and
numeric bounds are not enforced. Generation allows at most one whitespace
character between tokens and 32 characters per number; integer schemas use
spellings without fractions or exponents. Arrays and objects nest at most 32
deep (`serve.NESTING`). A deeper bracket is never sampled, and if no other token
fits, the request ends with an error. Schemas nest at most 32 levels,
counting each subschema and reference, and an enum or constant value only as
deep as its level leaves; deeper ones get HTTP 400.

## tinygrad and PyTorch

Import `baretile.adapters.tinygrad_device` before using tinygrad's `BARETILE` device.
The adapter translates tinygrad's graph into baretile tensors before tinygrad
schedules kernels. It recognizes a broadcast multiply followed by a reduction
as a matmul, also when the products are cast to fp32 before the sum, as in
tinygrad's bf16 matmul. That matmul casts its sum back to bf16, so it's
baretile's bf16 matmul, and the elementwise ops after it, like a bias and an
activation, join its kernel. A matmul that more than one kernel reads is
computed once, on its own. A matmul that baretile cuts into parts along K, as
it can for a single row of tiles, keeps the fp32 sum, since in bf16 each part
would round. Its tensors have three dimensions, with tinygrad's leading axes
folded into the rows. A view that an operation needs as a tensor, like a permute
or padding, is one move on the chip when its axes merge into at most five runs.
A permute with more runs takes a move for each pair of axis groups it swaps,
and anything else is copied through the host. `TinyJit` uses baretile
recordings when the call can be recorded.

You can run an existing tinygrad script through the adapter:

```sh
python3 -m baretile.adapters.tinygrad_device /path/to/script.py
```

The adapter calls tinygrad's internal APIs, so a tinygrad update can break it.
CI tests it on the tinygrad commit pinned in
[.github/workflows/test.yml](../.github/workflows/test.yml). Put a checkout of
that commit on `PYTHONPATH`, or in `TINYGRAD` for the tests.

Unsupported operations fall back to tinygrad's CPU device. These include
64-bit integers, bit operations, random-number generation and symbolic shapes.
Int32 multiplication and reductions also use the CPU to preserve exact integer
results; baretile's tensor API computes those operations through fp32.
The adapter preserves the logical dtype through fallback; uint16 values and
bfloat16 storage use different conversions, and boolean results stay 0 or 1.

`TinyJit` records the second eligible call and replays its launches from the
third. A replay reads each input, and each BARETILE buffer the call captured,
where it is at that call, so new inputs, a weight or learning rate assigned
between calls, and buffers that another `TinyJit` updates all replay without
recording again. What the call stores into a captured buffer is copied back
into that buffer's storage, and an input that is a result of the recording is
copied before the replay reads it. Results reuse their recorded storage on
subsequent calls. A call records again when a buffer the recording reads now
has another layout, shape or dtype, or when a buffer the recording made or
uploaded has changed. A mixed-device call discards the recording before running
normally. Calls that materialize data on the host, bind symbolic
variables, pass repeated or partial input buffers, store into input buffers, or
return an input or a view of one execute normally, including aliases inside
returned containers. Discarded recordings release their allocations after
queued work completes. Reading another device's buffers executes the call
normally, so CPU-generated values are read again on each call. Python and NumPy
literals can be recorded. A captured buffer read from host bytes and then
written needs another recording once its current value is on the chip.

PyTorch uses tinygrad's `extra/torch_backend`, which needs a tinygrad checkout,
PyTorch, a C++ compiler and ninja. `test_torch_frontend` exercises that path.

## Training

`python3 -m baretile.models.train` trains GPT-2 from scratch on GPT-2 tokens.
Its defaults are GPT-2 124M's shape and nanoGPT's recipe:

- 12 blocks 768 wide, with 12 heads, on sequences of 1,024 tokens, and the
  vocabulary padded to 50,304.
- Weights, gradients and AdamW's state in fp32. AdamW uses betas 0.9 and 0.95,
  and decays the matrices and embeddings by 0.1, but not the biases or the
  layer norms. Gradients are clipped to a norm of 1.
- 524,288 tokens a step, as the gradients of 512 sequences added up. The
  learning rate warms up to 6e-4 over 700 steps, then follows a cosine down to
  a tenth of that at the last of 19,073 steps, which read 10B tokens once.

Each sequence's forward and backward pass, and each update, runs as a recorded
replay (`jit`), so the host only sends the sequence's tokens. The chip compares
the targets with each column of the logits to pick them out, so no one-hot
matrix crosses PCIe.

The tokens come from llm.c's `.bin` shards: a 1 KiB header whose first three
int32s are 20240520, 1 and the token count, then the tokens as uint16s. FineWeb's
10B-token sample comes in that form, in shards of 100M tokens:

```sh
hf download kjj0/fineweb10B-gpt2 --repo-type dataset --local-dir fineweb fineweb_train_000001.bin fineweb_val_000000.bin
python3 -m baretile.models.train fineweb/fineweb_train_000001.bin fineweb/fineweb_val_000000.bin log=train.json
```

Settings go after the files as `key=value`. Run the command with no arguments to
list them. Every 250 steps it reports the validation loss over the same 20
sequences of the second file, and `log` writes each step's numbers as a line of
JSON. On ttsim, try a small model, such as `layers=2 dim=64 heads=2 ctx=32
accum=2 steps=10 warmup=2`, with a short warmup so the rate rises within its
steps. A step of that model takes about half a minute on ttsim v1.11.5, most of
it the output layer over the 50,304-token vocabulary.

`test_train_gpt2` trains a small GPT-2 this way on a counting sequence. At each
step, the loss and gradients must match PyTorch's on the same weights and
sequences, and the update must be AdamW's, computed in float64 from the
gradients the chip added up. Nothing has trained on a card yet. llm.c and the
modded-nanogpt speedrun train GPT-2 124M on FineWeb to a validation loss of
3.28, and [#45](https://github.com/baretile/baretile/issues/45) tracks doing
that on a P150a. fp32 matmuls take three bf16 products each, so training speed
is work for the card.

[test/train.py](../test/train.py) trains a small character-level GPT-2 on the
README and these docs, with the same model code and Adam, which is AdamW without decay. It has 2 blocks
64 wide, and reads 64 characters at a time. Every step runs as one recording
(`jit`), so the host only uploads each step's inputs. At each step PyTorch
computes the loss and gradients on baretile's current weights, and a separate
PyTorch run trains its own copy from the same starting weights:

```sh
python3 test/train.py 300
```

It needs PyTorch. On ttsim v1.11.5 a step takes about 1.3 seconds. In this run
the loss stayed within 5e-4 of PyTorch's on the same weights, and the gradients
within 3e-2 of PyTorch's largest. The mean loss over each 50 steps:

| Steps | baretile | PyTorch, same weights | PyTorch, own run |
| --- | --- | --- | --- |
| 1-50 | 3.55 | 3.55 | 3.56 |
| 51-100 | 3.34 | 3.34 | 3.36 |
| 101-150 | 3.25 | 3.25 | 3.35 |
| 151-200 | 3.18 | 3.18 | 3.31 |
| 201-250 | 3.12 | 3.12 | 3.21 |
| 251-300 | 3.05 | 3.05 | 3.12 |

PyTorch's own run drifts away from baretile's. Adam's first updates follow each
gradient's sign, which a matmul's rounding can flip for a gradient near zero, so
the two runs take different paths to similar losses.

To train with baretile, write the model with `Tensor`. Its operations have
gradients (`backward`, `gradient`), `assign` updates a weight in place, and
`jit` records a whole step, as `train.py` does with `adam`. tinygrad's
optimizers and `TinyJit` also train through the [adapter](#tinygrad-and-pytorch),
and so does PyTorch through tinygrad's torch frontend; `test_tinygrad` and
`test_torch_frontend` train small networks both ways. The adapter runs what it
can't translate on the host, so check how much of a step stays on the chip with
`baretile.viz`.

## Estimating cost

`baretile/models/cost.py` estimates memory use and serving cost from a model config:

```sh
python3 -m baretile.models.cost /path/to/Qwen3.6-35B-A3B experts context=16384 accept=0.55
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
