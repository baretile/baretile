# Contributing to baretile

baretile is small on purpose. One person should be able to read all of it, from
the PCIe driver to the model server. Before you add something, check whether
you can remove or reuse something instead.

## Setting up

Follow [Try it](README.md#try-it) to get ttsim, then install the test
dependencies:

```sh
python3 -m pip install numpy pytest ruff
export TTSIM=/absolute/path/to/libttsim.so
python3 -m pytest -s test
ruff check
```

[Testing and debugging](docs/testing.md) lists the optional checks that need
model files, PyTorch or transformers, and explains what to do when a launch
hangs.

## What a change needs

- A test that runs on ttsim. Compare kernels against NumPy.
- The whole suite passing before you send it.
- For a new or changed kernel, a passing run under [ttsim-chaos](docs/testing.md#ttsim-chaos),
  which adds the NoC latency, FPU and SFPU latency, and core jitter that ttsim leaves out.
  CI runs the kernel tests under it with a new seed each time, and the whole suite on all 140 Tensix tiles.
- For a refactor of a kernel, the same compiled bytes. Run the suite once before
  and once after with an empty cache (for example `XDG_CACHE_HOME=$(mktemp -d)`)
  and compare the images it leaves in `baretile/` under that cache. Images are
  named by the hash of their source.
- Docs that stay true. If your change alters how something works, update the page
  in [docs/](docs) that describes it.

## How the code is organized

The code has five layers, and each one imports only the standard library and the
layers below it. `test_layers` checks this.

- Layer 0 is the device (`bh.py`, `pci.py`, `chip.py`) and layer 1 is the runtime
  (`runtime.py`).
- Layer 2 is Tensix codegen (`tensix.py`) and layer 3 is the kernels and `Tensor`
  (`kernels.py`, `tensor.py`).
- Layer 4 is everything outside `baretile/`: the models, the server and the adapters.

If only one workload uses a kernel, that kernel lives in layer 4 next to the
workload. It moves down when a second workload needs it. No file uses another
file's private names.

## Style

- `baretile/` uses only the standard library. The tests, `adapters/` and `models/`
  can use NumPy.
- Keep it flat: few files, no class hierarchies and no config systems. Don't add
  an abstraction until it has a second user.
- Facts about the chip live as data in [bh.py](baretile/bh.py). An instruction's
  modifier values stay next to the compiler code that uses them.
- Every wait has a timeout. If the chip hangs, we report it.
- A caller's mistake raises an exception (`ValueError` for a bad argument), and so
  does a card in the wrong state (`RuntimeError`). Use `assert` only for baretile's
  own invariants, and never put work in one: `python -O` drops asserts.
  `test_checks_without_asserts` runs bad calls under `-O`.
- Comments explain a constraint or a reason that the code doesn't make clear.
  Docstrings start with what the function does and mention anything surprising.
  Document kernel arguments and layouts when a caller needs them.

Docs, comments and errors use short, direct sentences. Name the operation and
say which buffer or worker you mean. Be specific about what's untested, and
include numbers only with the command that reproduces them. Simulator cycle
counts, cost estimates and measurements from a card are different things, so
say which one a number is.

We use the same terms everywhere. A worker is a Tensix tile running a kernel, and
BRISC and the TRISCs are its RISC-V cores. A tensor tile holds 32 by 32 values.
"Tile column" refers to `Tensor.columns`.

Keep commit subjects under 70 characters, for example "matmul: read B by
columns" or "fix hang when the ring wraps". Add a short body when the reason
needs explaining.

## Reporting a bug

Open an [issue](https://github.com/baretile/baretile/issues) with:

- whether you ran on ttsim (and which version) or a card, and for a card, the
  output of `python3 -m baretile.chip check` and `python3 -m baretile.chip`
- the command you ran and its full output, including the `TimeoutError` report
  if a launch hung
- your Python and clang versions

## Releasing

The version is in [pyproject.toml](pyproject.toml). The package on PyPI holds
`baretile/` alone. To publish a new version, raise the version, then build and
upload it:

```sh
uv build
uv publish
```

CI builds a wheel from the source distribution and checks its imports and
visualization template on Python 3.12, 3.13 and 3.14. Before publishing, test the
built wheel in a clean environment so imports cannot fall back to the checkout.
