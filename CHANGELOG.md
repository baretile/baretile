# Changelog

baretile is pre-alpha. Everything below is tested on Tenstorrent's ttsim
simulator; nothing has run on a P150a yet.

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
