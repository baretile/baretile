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
`tokenizer.json`. The standalone GPT-2 command also needs `tiktoken`.

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
allocated once, and each token writes its keys and values at its position.

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
Free server slots leave their recurrent states unchanged.

## Serving

The server accepts a folder or Hugging Face model ID. Its positional arguments
are model, port, slots, positions, precision and optional KV format:

```sh
python3 models/serve.py HuggingFaceTB/SmolLM-135M 8000 4 512 bf16
```

Here, four requests can run at once, with 512 positions per slot. Choose a
multiple of 32 for that size, and leave room for the answer as well as the
prompt. The engine also reserves positions at the end for inactive slots.

Downloads go to `~/.cache/baretile/hub`. `HF_TOKEN` supplies access to a gated
model, and `HF_ENDPOINT` changes the download endpoint. Completed downloads are
reused. The server binds to all interfaces on the selected port and has no API
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

The engine gives each request a slot in a shared KV cache and decodes occupied
slots together. Prompts are processed in chunks between decode steps. A request
gets HTTP 429 when no slot is free. A non-streaming request that receives no text
for 600 seconds gets HTTP 504. A stream that times out releases its slot.

## Reusing a conversation

When a prompt arrives, the engine looks for a free slot that has already
processed a prefix of it. It can also restore a saved session with its tokens,
KV cache and recurrent state. DeltaNet must resume from the exact tokens that
produced its state.

Sessions are saved in spare DRAM on the chip. When that fills, the server copies
the least recently used sessions to host memory. The host cache has a size limit
too, after which sessions are dropped. The response's usage field counts the
prompt tokens that were reused.

## Sampling and API limits

For supported vocabulary shapes, the device returns the largest 64 logits and
the normalization sum. Greedy selection and top_k up to 64 can use that result.
Penalties, logprobs and constrained output need the full row. Some other
sampling requests do too, so they can read much more data back to the host.

Chat templates are written in Python, so a model with a new template may need
code in `serve.py`. Qwen tool calls are returned in `tool_calls`, and reasoning
goes in `reasoning_content`. `tool_choice` and JSON response formats constrain
which tokens can be sampled. `test_chat_tools` and `test_serve` have example
requests to start from when adding a model or client.

## tinygrad and PyTorch

Import `adapters.tinygrad_baretile` before using tinygrad's `BARETILE` device.
The adapter translates tinygrad's graph into baretile tensors before tinygrad
schedules kernels. It recognizes a broadcast multiply followed by a reduction
as a matmul. `TinyJit` uses baretile recordings when the call can be recorded.

You can run an existing tinygrad script through the adapter:

```sh
python3 adapters/tinygrad_baretile.py /path/to/script.py
```

Install tinygrad or put its checkout on `PYTHONPATH`. The tests also accept a
checkout in `TINYGRAD`. The adapter calls tinygrad's internal APIs, so a tinygrad
update can break it.

Unsupported operations fall back to tinygrad's CPU device. These include
64-bit integers, bit operations, random-number generation and symbolic shapes.

PyTorch uses tinygrad's `extra/torch_backend`, which needs a tinygrad checkout,
PyTorch, a C++ compiler and ninja. `test_torch_frontend` exercises that path.

## Estimating cost

`models/cost.py` estimates memory use and serving cost from a model config:

```sh
python3 models/cost.py /path/to/Qwen3.6-35B-A3B experts context=16384 accept=0.55
```

Add `probe=probe.json` once you've measured a card. Until then, the model assumes
that decode gets 70% of the specified 512 GB/s DRAM bandwidth and that a launch
takes one microsecond. Those two assumptions are guesses.

The model counts the weights, caches, states and activations that a step reads.
It estimates how long those reads and the matrix operations take, uses the
longer time, then adds the cost of launches and RISC-V control code. The number
of different experts selected and the fraction of drafts accepted will change
with your workload.
