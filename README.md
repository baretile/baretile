# baretile

[![test](https://github.com/baretile/baretile/actions/workflows/test.yml/badge.svg)](https://github.com/baretile/baretile/actions/workflows/test.yml)

baretile is a compiler and runtime for Tenstorrent Blackhole, written in Python.
You write tensor operations, and it compiles them into Tensix instructions. It
also handles the PCIe access needed to load and run those programs on the chip.

It runs on Tenstorrent's ttsim simulator. We haven't tested it on a card yet.

The code in `baretile/` uses only the Python standard library. The driver runs
in userspace, and baretile loads its own firmware onto the workers. It needs
Tenstorrent's board firmware for ARC and Ethernet. There's no dependency on
tt-metal, tt-umd or LLK.

## Try it

You'll need Python 3.12 or later, clang with a RISC-V target, lld, and the
Blackhole library from [ttsim](https://github.com/tenstorrent/ttsim). ttsim's
[releases](https://github.com/tenstorrent/ttsim/releases) include it as
`libttsim_bh.so`. You can also build it in a ttsim checkout:

```sh
./make.py src/_out/release_bh/libttsim.so
```

We test with ttsim v1.11.5. CI uses its prebuilt `libttsim_bh.so`, and we also
build it from source.

Clone baretile and run its first two commands:

```sh
git clone https://github.com/baretile/baretile
cd baretile
export TTSIM=/absolute/path/to/libttsim.so
python3 -m baretile.chip
python3 -m baretile.hello
```

With 120 Tensix tiles, `baretile.hello` prints:

```text
119 Tensix workers computed fib(30) = {832040}, and one tile dispatched the work
```

Keep `TTSIM` set for the examples below. Without it, baretile tries to open a
Blackhole on the PCI bus. That needs root and takes the card away from its kernel
driver. See [bring-up](https://github.com/baretile/baretile/blob/main/docs/bring-up.md)
before running on a card.

`pip install baretile` installs the `baretile` package from PyPI. That's enough
for the commands above and the tensor example below. The ramps, models and
tests are in the repository.

## Tensors

Run this in Python:

```python
from baretile import pci
from baretile.chip import Chip
from baretile.runtime import Runtime
from baretile.tensor import Tensor

rt = Runtime(Chip(pci.open_device()))
x = Tensor.of(rt, [[1.0, 2.0], [3.0, 4.0]])
w = Tensor.of(rt, [[2.0, 0.0], [0.0, 3.0]])
print((x @ w).relu().tolist())
# [[2.0, 6.0], [6.0, 12.0]]
```

The matmul and ReLU run when you call `tolist()`. Until then, they're an
expression waiting to be compiled. They run in one kernel here, with the matmul's
result kept in local memory for the ReLU.

Tensors support broadcasting, reductions, views and autograd. Storage formats are
bf16, fp32, int32, bfp8 and bfp4. The
[compiler docs](https://github.com/baretile/baretile/blob/main/docs/compiler.md)
explain the layout and precision limits.

To see the kernels a script launched and what each one computes, run it with
`python3 -m baretile.viz script.py` and open the `viz.html` it writes.

## Learn baretile

The [ramps](https://github.com/baretile/baretile/tree/main/ramps) teach baretile
from the top down. Each one is a script that explains its steps and checks
their results:

```sh
python3 ramps/1_tensors.py
python3 ramps/2_runtime.py
python3 ramps/3_tensix.py
```

The first covers tensor operations, fusion, formats, gradients and `jit`. The
second runs C kernels on the workers and moves data over the NoC. The third
programs the unpack, math and pack threads of a Tensix tile by hand.

The [documentation](https://github.com/baretile/baretile/blob/main/docs/README.md)
explains the runtime, the compiler, the models and the hardware, and includes a
map of the code.

## Models

Install NumPy for the model code and pytest for the tests:

```sh
python3 -m pip install numpy pytest
python3 -m pytest -s test
```

The tests run small GPT-2, Llama, mixture-of-experts and Gated DeltaNet models
and compare their results with NumPy. Some comparisons need PyTorch and
transformers, and pytest skips them if those packages aren't installed. Tests
that need checkpoint files are opt-in.

To generate text from a local SmolLM-135M checkpoint:

```sh
python3 models/llama.py /path/to/SmolLM-135M "The meaning of life is" 6
```

You'll need the model's config, tokenizer and safetensors weights in that folder.
The command runs baretile and NumPy side by side. It prints when they choose
different tokens, then continues with NumPy's choice.

The loader also handles Llama, Mistral, Qwen2, Qwen3, Mixtral, gpt-oss and the
Qwen3.5/3.6 text models. Most tests use small models. Full checkpoints need
separate checks, which
[testing](https://github.com/baretile/baretile/blob/main/docs/testing.md#optional-checks)
lists.

To serve a model through an OpenAI-compatible HTTP API:

```sh
python3 models/serve.py HuggingFaceTB/SmolLM-135M 8000
```

This downloads the model if you don't already have it. You can also pass a local
folder. Requests decode together, and the server can reuse a cached prompt when
you send the next turn of a conversation.
[Serving](https://github.com/baretile/baretile/blob/main/docs/models.md#serving)
has an example request.

tinygrad scripts run on baretile through
[adapters/tinygrad_baretile.py](https://github.com/baretile/baretile/blob/main/adapters/tinygrad_baretile.py),
and PyTorch runs through tinygrad's torch frontend.

## Contributing

[CONTRIBUTING.md](https://github.com/baretile/baretile/blob/main/CONTRIBUTING.md)
explains how to run the checks, what the code should look like and how to
report a bug.

Licensed under [Apache-2.0](https://github.com/baretile/baretile/blob/main/LICENSE).
