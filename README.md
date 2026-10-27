# baretile

baretile is a compiler and runtime for Tenstorrent Blackhole, written in Python.
You write tensor operations, and it compiles them into Tensix instructions. It
also handles the PCIe access needed to load and run those programs on the chip.

It runs on Tenstorrent's ttsim simulator. We haven't tested it on a card yet.

The code in `baretile/` uses only the Python standard library. The driver runs
in userspace, and baretile loads its own firmware onto the workers. It needs
Tenstorrent's board firmware for ARC and Ethernet. There's no dependency on
tt-metal, tt-umd or LLK.

## Try it

You'll need Python 3, clang with a RISC-V target, lld, and a Blackhole build of
[ttsim](https://github.com/tenstorrent/ttsim). In a ttsim checkout, build the library:

```sh
./make.py src/_out/release_bh/libttsim.so
```

Then, from the baretile checkout:

```sh
export TTSIM=/absolute/path/to/ttsim/src/_out/release_bh/libttsim.so
python3 -m baretile.chip
python3 -m baretile.hello
```

With 120 Tensix cores, `baretile.hello` prints:

```text
119 Tensix cores computed fib(30) = {832040}, and one more core dispatched the work
```

Keep `TTSIM` set for the examples below. Without it, baretile tries to open a
Blackhole on the PCI bus. That needs root and takes the card away from its kernel
driver. See [card bring-up](NOTES.md#bring-up-on-a-p150) before running on a card.

## Tensors

Run this in Python from the checkout:

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
bf16, fp32, int32, bfp8 and bfp4. [NOTES.md](NOTES.md#tensors-and-kernels) explains
the layout and precision limits.

To see the kernels a script launched and what each one computes, run it with
`python3 -m baretile.viz script.py` and open the `viz.html` it writes.
[Profiling](NOTES.md#profiling) describes the page.

## Models

Install NumPy for the model code and pytest for the tests:

```sh
python3 -m pip install numpy pytest
python3 -m pytest -s test
```

The tests include small GPT-2, Llama, mixture-of-experts and Gated DeltaNet models.
They compare results with NumPy. Some comparisons need PyTorch and transformers,
and pytest skips them if those packages aren't installed. Tests that need
checkpoint files are opt-in.

To generate text from a local SmolLM-135M checkpoint:

```sh
python3 models/llama.py /path/to/SmolLM-135M "The meaning of life is" 6
```

You'll need the model's config, tokenizer and safetensors weights in that folder.
The command runs baretile and NumPy side by side. It prints when they choose
different tokens, then continues with NumPy's choice.

The loader also handles Llama, Mistral, Qwen2, Qwen3, Mixtral, gpt-oss and the
Qwen3.5/3.6 text models. Most tests use small models. Full checkpoints need
separate checks. [Models and tests](NOTES.md#models-and-tests) has the commands.

To serve a model through an OpenAI-compatible HTTP API:

```sh
python3 models/serve.py HuggingFaceTB/SmolLM-135M 8000
```

This downloads the model if you don't already have it. You can also pass a local
folder. Requests decode together, and the server can reuse a cached prompt when
you send the next turn of a conversation. See [Serving](NOTES.md#serving) for an
example request.

## Reading the code

Start with [hello.py](baretile/hello.py) to see a kernel launched and read back.
[pci.py](baretile/pci.py) opens the device and [chip.py](baretile/chip.py) reads and
writes its memory. [runtime.py](baretile/runtime.py) loads programs, allocates
memory and queues work.

[tensix.py](baretile/tensix.py) emits instructions.
[kernels.py](baretile/kernels.py) uses those instructions to build kernels, and
[tensor.py](baretile/tensor.py) compiles tensor expressions into launches.
Addresses and instruction fields live in [bh.py](baretile/bh.py).

[models/](models/) contains the model code and HTTP server.
[adapters/](adapters/) connects tinygrad to baretile. PyTorch uses tinygrad's torch
frontend. There are more details in [NOTES.md](NOTES.md).

Licensed under [Apache-2.0](LICENSE).
