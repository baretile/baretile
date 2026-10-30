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

You'll need Linux, Python 3.12 or later, and clang with a RISC-V target and lld.
On Debian and Ubuntu, `sudo apt update && sudo apt install python3-venv clang lld
curl git` installs them, with curl and git for the later sections. Install
baretile in a virtual environment, download the simulator and check the setup:

```sh
python3 -m venv baretile-env
. baretile-env/bin/activate
python3 -m pip install baretile
python3 -m baretile.chip ttsim
python3 -m baretile.chip check
```

In fish, activate the environment with `. baretile-env/bin/activate.fish`.

`chip ttsim` downloads the Blackhole library of
[ttsim](https://github.com/tenstorrent/ttsim) v1.11.5, the release baretile is
tested with, into baretile's cache. With no card on the PCI bus, baretile uses it.
`chip check` says what's missing and how to fix it. Then run the first two
commands. `chip` prints the firmware, the clock, the Tensix tiles and the DRAM
banks, then the raw telemetry:

```sh
python3 -m baretile.chip
python3 -m baretile.hello
```

On ttsim, which has 120 Tensix tiles, `baretile.hello` prints:

```text
119 Tensix workers computed fib(30) = {832040}, and one tile dispatched the work
```

baretile needs only the standard library. The models and the server also need
NumPy, so install `'baretile[models]'` for those. The ramps and the tests are in
the repository.

To use another ttsim build, such as
[ttsim-chaos](https://github.com/baretile/baretile/blob/main/docs/testing.md#ttsim-chaos),
set `TTSIM=/absolute/path/to/libttsim.so`. It comes before a card and before the
downloaded release. On a machine with a Blackhole and no `TTSIM`, baretile opens
the card through tt-kmd, Tenstorrent's kernel driver, without root. Without
tt-kmd, it opens the card through sysfs, which needs root. See
[bring-up](https://github.com/baretile/baretile/blob/main/docs/bring-up.md)
before running on a card.

## Tensors

Save this as `tensors.py` and run it with `python3 tensors.py`:

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
expression waiting to be compiled. They run in one kernel here: the ReLU works on
the matmul's result in the math unit's registers, before it's written out.

Tensors support broadcasting, reductions, views and autograd. Storage formats are
bf16, fp32, int32, bfp8 and bfp4. The
[compiler docs](https://github.com/baretile/baretile/blob/main/docs/compiler.md)
explain the layout and precision limits.

To see the kernels a script launched and what each one computes, run it with
`python3 -m baretile.viz tensors.py` and open the `viz.html` it writes.

## Run a model

Install baretile with NumPy for the models, then serve a model by its Hugging
Face ID:

```sh
python3 -m pip install 'baretile[models]'
python3 -m baretile.models.serve HuggingFaceTB/SmolLM-135M 8000
```

The first time, the server downloads the model, about 540 MB, which takes a few
minutes. It can take a local folder instead. When it prints `serving
SmolLM-135M on port 8000`, ask it for a completion from another terminal:

```sh
curl http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt": "The meaning of life is", "max_tokens": 4, "temperature": 0}'
```

On ttsim the answer's text is `" a question that has"`. It takes about two
minutes, half a minute a token, since the simulator runs every core of the chip,
and curl prints nothing until it's done. The server speaks OpenAI's completions
and chat completions API, so OpenAI's clients work with it too.

[Models](https://github.com/baretile/baretile/blob/main/docs/models.md#running-a-model)
lists the families baretile runs and how each one is checked, and
[serving](https://github.com/baretile/baretile/blob/main/docs/models.md#serving)
covers the server's options. To generate from a local checkpoint and compare
each token with NumPy's choice, pass its folder. The server keeps what it
downloads in `~/.cache/baretile/hub`:

```sh
python3 -m baretile.models.llama ~/.cache/baretile/hub/HuggingFaceTB/SmolLM-135M "The meaning of life is" 6
```

tinygrad scripts run on baretile through
[baretile/adapters/tinygrad_device.py](https://github.com/baretile/baretile/blob/main/baretile/adapters/tinygrad_device.py),
and PyTorch runs through tinygrad's torch frontend.

To train, `python3 test/train.py 300` in a checkout trains a small GPT-2 on
baretile's docs and checks every step against PyTorch.
[Training](https://github.com/baretile/baretile/blob/main/docs/models.md#training)
shows its loss curve and how to train your own model.

## Learn baretile

The [ramps](https://github.com/baretile/baretile/tree/main/ramps) teach baretile
from the top down. Each one is a script that checks its results as it goes, and
its comments explain each step, so read it beside its output. They're in the
repository:

```sh
git clone https://github.com/baretile/baretile
cd baretile
python3 ramps/1_tensors.py
python3 ramps/2_runtime.py
python3 ramps/3_tensix.py
python3 ramps/4_kernel.py
```

The first covers tensor operations, fusion, formats, gradients and `jit`. The
second runs C kernels on the workers and moves data over the NoC. The third
programs the unpack, math and pack threads of a Tensix tile by hand, and the
fourth puts all four programs together into a tensor operation of your own.

The [documentation](https://github.com/baretile/baretile/blob/main/docs/README.md)
explains the runtime, the compiler, the models and the hardware, and includes a
map of the code.

## Contributing

The tests run on ttsim. They need NumPy and pytest, and some comparisons also
need PyTorch and transformers, which pytest skips without:

```sh
python3 -m pip install numpy pytest
python3 -m pytest -s test
```

Tests that need checkpoint files are opt-in, and
[testing](https://github.com/baretile/baretile/blob/main/docs/testing.md#optional-checks)
lists them.

[CONTRIBUTING.md](https://github.com/baretile/baretile/blob/main/CONTRIBUTING.md)
explains how to run the checks, what the code should look like and how to
report a bug.

Licensed under [Apache-2.0](https://github.com/baretile/baretile/blob/main/LICENSE).
