# Changelog

baretile is pre-alpha. Everything below is tested on Tenstorrent's ttsim
simulator; nothing has run on a P150a yet.

## 0.0.5

The first release whose tensors have more than three dimensions, and whose Qwen3-0.6B passes the generation check:

```python
from baretile import pci
from baretile.chip import Chip
from baretile.runtime import Runtime
from baretile.tensor import Tensor

rt = Runtime(Chip(pci.open_device()))
x = Tensor.of(rt, [[[[1.0, 2.0]], [[3.0, 4.0]]], [[[5.0, 6.0]], [[7.0, 8.0]]]])  # (2, 2, 1, 2)
print(x.sum(1).tolist())
# [[[[4.0, 6.0]]], [[[12.0, 14.0]]]]
```

### Added

- A tensor has up to five dimensions: up to three leading ones, which fold into the batch, then the matrices' rows and
  columns. Elementwise ops and matmuls broadcast over them as NumPy does, reading an operand in place where its leading
  sizes are one run of the result's, and `sum`, `max`, `mean` and `softmax` reduce over any of them in one kernel.
  Movement and slicing take every dimension, a reshape of the leading sizes is a view, and gradients come back through
  all of them. [The public API](docs/api.md#tensors-and-memory) lists the ops that still take three.
- A kernel's programs can `print(fmt, ...)` a string literal and up to six words. The host shows each line after
  `wait()`, with the worker, core and launch that printed it, and a hang's report ends with the last three lines of
  each of the stuck worker's cores. Each core keeps its last 127 lines.
- The tinygrad adapter makes a view a tensor in one move over up to five axes, where a permute or padding on two axes
  of a group took several moves or a copy through the host, so a `TinyJit` replays such calls too.

### Changed

- Qwen3 and Qwen3.5 run the q and k norm in fp32 and round once for RoPE. Qwen3-0.6B now passes the first workload's
  bf16 logit gate, at 2.95% at worst where it was 4.40%, in the same launches.
- A reshape that regroups whole tile rows into other matrices, like `(2, 64, cols)` into `(128, cols)`, is a view.
- Each kernel image holds `kmain` once: 153 of the suite's programs held its body twice.
- Loaded kernels get 16 KiB less room in L1, which the print rings take.

### Fixed

- An operand broadcast into a result whose gradient was broadcast too, as a sum's is, got that gradient once instead
  of once for each copy: `(a + c).sum()` gave `c` a gradient of 1 where it's the number of copies.

## 0.0.4

The first release that can open a card without root, and that trains a model from scratch:

```sh
python3 -m pip install 'baretile[models]'
python3 -m baretile.models.train train.bin val.bin layers=2 dim=64 heads=2 ctx=32 accum=2 steps=10 warmup=2
```

### Added

- On a card that tt-kmd drives, baretile opens it through kmd's device file, `/dev/tenstorrent/N`. That needs no
  root, needs no `iommu=pt` when the IOMMU translates, and leaves the card to tt-smi too. It needs tt-kmd 2.0 or
  later, and 2.11 for an explicit `Chip.arc_msg`.
- `python3 -m baretile.models.train` trains GPT-2 from scratch on GPT-2 tokens in llm.c's `.bin` shards, such as
  FineWeb's. By default that's GPT-2 124M with nanoGPT's recipe: AdamW, a warmup and cosine schedule, and clipping.
  [Training](docs/models.md#training) shows how to get the tokens.
- `BARETILE_DUMP=folder` writes the C programs of each kernel a runtime launches, one file for each core, and
  `baretile.viz` shows them beside the kernel's launches.

### Changed

- With tt-kmd driving the card, baretile no longer unbinds it. It opens the card through sysfs as root, as before,
  only when tt-kmd doesn't drive it, and `python3 -m baretile.chip check` reports what that path needs.
- The Llama families keep the residual stream between blocks in fp32. SmolLM-135M now passes the first workload's
  bf16 logit gate, at 2.72% at worst where it was 4.16%. A decode step takes more launches, which #57 would fuse back.
- The core, from the PCIe driver to the tensors, has a size budget: under 4,500 lines of code at 120 columns, with
  every line within 120 columns.

## 0.0.3

The first release that runs models from an installed package:

```sh
python3 -m pip install 'baretile[models]'
python3 -m baretile.chip ttsim
python3 -m baretile.models.serve HuggingFaceTB/SmolLM-135M 8000
```

### Added

- The models, the OpenAI-compatible server and the tinygrad adapter are in the
  package, as `baretile.models` and `baretile.adapters`. The `models` extra adds
  NumPy, which they need; baretile itself still needs only the standard library.
- `python3 -m baretile.models.serve` serves a model by its Hugging Face ID.
- `python3 -m baretile.chip ttsim` downloads the tested ttsim release, so
  `TTSIM` is optional.
- `python3 -m baretile.chip check` checks clang, lld, NumPy, and the simulator
  or the card, and says how to fix what's missing.
- `python3 -m baretile.viz -m module` runs a module under the launch viewer.
- The supported model families are listed in
  [docs/models.md](docs/models.md#running-a-model), each compared against
  transformers on ttsim. Other model types are refused at load.

### Changed

- `from models import llama` is now `from baretile.models import llama`, and
  `adapters.tinygrad_baretile` is `baretile.adapters.tinygrad_device`. The
  commands run as `python3 -m baretile.models.serve`, `.llama`, `.gpt2` and
  `.cost`.
- A device has one opener at a time. A second `Chip` on the same chip raises
  `RuntimeError`.
- Closing a runtime lets its running launch finish, for up to two seconds,
  instead of stopping workers inside a Tensix program.
- The server sends a stream's usage chunk only when
  `stream_options.include_usage` asks for it, reads `max_completion_tokens`,
  and takes message content as a list of text parts.
- A jit replay gives each input's gradient back to the caller.

### Fixed

- A matmul split along K dropped its fused scalars, so `(x @ w) * 0.5` was
  wrong for long K.
- fp32 `exp`, `exp2`, `reciprocal` and `where` mishandled NaNs with the sign
  bit set.
- jit replays could read and write the same storage when an input shared it
  with a result or with something the replay writes.
- A runtime stopped in the middle of a Tensix kernel left its workers blocked
  for the next runtime.
- The tokenizer took quadratic time on long words, and its cache grew without
  bound.
- The server could drop a slow client without a 408, kept generating for a
  client that had hung up, failed on padded vocabularies, and lost the text of
  a tool call that was cut short.
- Turning on a card's bus mastering cleared its PCI error bits.
- `baretile.viz` crashed when the script closed or replaced its runtime.

## 0.0.2 and earlier

The device, the runtime, Tensix code generation, the kernels and `Tensor`,
without the models.
