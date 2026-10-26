# baretile

baretile is the software stack for Tenstorrent's Blackhole, from PCIe up to a tensor compiler, in about 2,400 lines
of Python.

The only Tenstorrent code it runs is the firmware on the card. There's no kernel driver, no tt-umd, no tt-metal and
no LLK. baretile talks to the card through its PCIe BARs from userspace, loads its own firmware onto the Tensix cores,
and compiles tensor code to Tensix instructions that we wrote from the ISA docs. It's the stack George Hotz described
in [Advice to Tenstorrent](https://github.com/geohot/tt-tiny): a compiler and a runtime.

It hasn't been on a card yet. Everything below runs on ttsim, Tenstorrent's Blackhole simulator.

```python
from baretile import pci
from baretile.chip import Chip
from baretile.runtime import Runtime
from baretile.tensor import Tensor

chip = Chip(pci.open_device())              # the card, or the simulator if TTSIM is set
rt = Runtime(chip)                          # loads firmware onto the Tensix cores and picks one to dispatch
x = Tensor.of(rt, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
w = Tensor.of(rt, [[0.5, -1.0], [2.0, 0.25], [-1.0, 1.0]])
print((x @ w).relu().softmax(-1).tolist())  # lazy, so it runs here, as one fused kernel
```

## Running

You need Python 3, clang, lld, and ttsim built for Blackhole (`libttsim.so`). The tests also need numpy. Without
`TTSIM`, baretile looks for a Blackhole on the PCI bus, which needs root.

    TTSIM=/path/to/libttsim.so python3 -m baretile.chip      # what the card says about itself
    TTSIM=/path/to/libttsim.so python3 -m baretile.hello     # fib(30) on all the workers
    TTSIM=/path/to/libttsim.so python3 -m pytest test        # the tests, on the simulator
    TTSIM=/path/to/libttsim.so python3 models/llama.py SmolLM-135M "The meaning of life is" 6
    TTSIM=/path/to/libttsim.so python3 models/serve.py SmolLM-135M 8000

## What runs

These load from their Hugging Face checkpoints:

- GPT-2
- Llama 3.1, Mistral, Qwen2 and Qwen3
- Mixtral, Qwen2-MoE, Qwen3-MoE and gpt-oss, with the experts picked on the chip
- Qwen3.5 and Qwen3.6, with their Gated DeltaNet blocks

Each model is checked against a numpy version of itself, and the ones that aren't Llamas are checked against
transformers too. Weights can be bf16, bfp8 or bfp4. A decode step is recorded once and replayed after that.

`models/serve.py` is an OpenAI-compatible server. Requests decode together in a batch. An agent's session stays on
the card between turns, so the next turn doesn't prefill the conversation again. Qwen3.6 speculates with its
multi-token prediction layer. Tool calls, structured outputs and cached-token usage work with OpenAI and OpenRouter
clients.

tinygrad runs on baretile as a device (`adapters/`), and PyTorch runs through tinygrad's torch frontend.

## Code

- `baretile/` is the stack, from the bottom up:
  - `bh.py`: facts about Blackhole, kept as data
  - `pci.py` and `chip.py`: talking to the card over PCIe
  - `runtime.py`: memory, loading, launching and the command queue
  - `tensix.py`: Tensix instructions
  - `kernels.py`: kernels written with them
  - `tensor.py`: lazy tensors
- `models/` has the models, the server, and an estimate of what serving costs on a card.
- `adapters/` has tinygrad on baretile.
- `test/` has tests for each layer, a training run checked against PyTorch, and benchmarks against TT-NN.

`NOTES.md` explains how it works, where the facts about the chip come from, how to bring up a card, and what we don't
know yet.

## License

Apache-2.0 (`LICENSE`).
