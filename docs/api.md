# Public API

This page lists what you can build on, and what it promises. Names it doesn't
list, and names that start with an underscore, are internal and change freely.
baretile is pre-alpha, and nothing has run on a card yet, so everything below is
promised on ttsim only.

## Stability

A **stable** name keeps its meaning across 0.x releases. When one has to change,
the release's [changelog](../CHANGELOG.md) says so under "Changed" and says how to
update. An **experimental** name can change in any release, and the changelog
still lists the change. The version stays below 0.1 until baretile is tested on
a P150a.

## What's public

| Name | What it is | Status |
| --- | --- | --- |
| `python3 -m baretile.chip [check \| ttsim \| release \| reset]`, `baretile.hello` | The setup and device commands | stable |
| `baretile.viz`, `baretile.probe`, `baretile.bench` | The launch viewer and the measurements | stable commands, experimental output |
| `pci.open_device`, `pci.simulator`, `pci.fetch_ttsim` | Opening the simulator or a card | stable |
| `chip.Chip`: `read`, `write`, `read32`, `write32`, `telemetry`, `tensix`, `banks` | NoC access, ARC telemetry and the available tiles | stable |
| `chip.Chip`: `links`, `send` | Ethernet links to another chip | experimental |
| `runtime.Runtime`: the constructor, `launch`, `wait`, `close`, `closed`, `profile`, `args`, `dram_alloc`, `dram_free` | Workers, launches and DRAM | stable |
| `runtime.compile_kernel`, `runtime.compile_rv32`, and the C helpers in `runtime.DEVICE` | Building your own kernels | experimental |
| `tensor.Tensor`: its constructors, operations, movement, `cast`, `realize`, `assign`, `tolist`, `tobytes`, `backward` and `gradient` | Tensors | stable |
| `Tensor.to`, `Tensor.columns`, `Tensor.run` | Moving to another chip, storage by tile columns, launching a kernel | experimental |
| `tensor.jit` | Recording and replaying a function's launches | stable |
| `tensor.Recording`, `runtime.Bound` | Recordings by hand | experimental |
| `baretile.tensix`, `baretile.kernels` | Tensix code generation and the kernels | experimental |
| `python3 -m baretile.models.serve` and its HTTP API | The OpenAI-compatible server | stable |
| `baretile.models` from Python: `llama`, `qwen`, `gpt2`, `serve`, `cost` | Loading and running models | experimental |
| `baretile.adapters.tinygrad_device` | The tinygrad device | experimental, since it calls tinygrad's internals |

## What you can rely on

Each promise names the test that checks it.

### Devices and runtimes

- One `Chip` at a time has a device open, in any process. A second opener gets
  `RuntimeError` instead of sharing the first one's TLBs and DRAM.
  [`test_open_device`](../test/test_device.py)
- A `Runtime` owns a rectangle of workers. A new runtime that overlaps it takes
  over, and the old one, its tensors and its recordings then raise
  `RuntimeError`. [`test_takeover`](../test/test_device.py),
  [`test_tensors_and_recordings_reject_closed_runtime`](../test/test_kernels.py)
- Launches are queued, and `wait()` blocks until they're done. If no launch
  completes for `timeout` seconds, it raises `TimeoutError` with a report of
  the unfinished workers. [`test_hung_kernel_times_out`](../test/test_device.py)
- `close()` lets the running launch finish, for up to two seconds, and cancels
  the rest, so the next runtime finds the workers idle.
  [`test_takeover_during_tensix_kernel`](../test/test_kernels.py)
- A runtime dropped without `close()` gives its memory back at the next safe
  point: the next runtime, or an allocation that finds no room.
  [`test_dropped_runtime_gives_memory_back`](../test/test_device.py)
- A bad argument raises `ValueError`, and a card in the wrong state
  `RuntimeError`, also under `python -O`.
  [`test_runtime_inputs`](../test/test_device.py),
  [`test_checks_without_asserts`](../test/test_kernels.py)

### Tensors and memory

- A tensor is a batch of matrices, `(batch, rows, cols)`. Operations build an
  expression, and `realize()`, `tolist()` or `tobytes()` compiles it into fused
  kernels. [`test_fusion`](../test/test_kernels.py),
  [`test_tensor_inputs`](../test/test_kernels.py)
- A tensor's DRAM goes back once nothing refers to it and the launches queued
  so far are done. A tensor held only by a reference cycle, as a parameter is
  after `backward()`, goes back when an allocation needs the room.
  [`test_dram_is_freed`](../test/test_kernels.py),
  [`test_dram_in_reference_cycle`](../test/test_kernels.py)
- `backward()` adds to each leaf's `.grad`, as in PyTorch, and queues the
  gradients before returning, so updating one parameter can't change another's
  gradient. [`test_autograd`](../test/test_kernels.py),
  [`test_backward_before_parameter_updates`](../test/test_kernels.py)

### jit

- `jit(f)` runs the first call normally, records the second, and replays its
  launches from the third, on each call's inputs where they are. An input laid
  out unlike a new tensor, such as a slice, is copied in, and back out if `f`
  writes it. [`test_jit`](../test/test_kernels.py),
  [`test_jit_writes_inputs`](../test/test_kernels.py)
- Results and retained tensors reuse their buffers, so the next call overwrites
  them. An input's `.grad` holds the latest call's gradient instead of adding
  up. [`test_recording_lifetime_and_reuse`](../test/test_kernels.py),
  [`test_jit_input_grads`](../test/test_kernels.py)
- The same tensor can be passed twice, and a returned view of an input is on
  that call's input. Keep inputs that `f` writes apart from each other and from
  captured tensors that `f` only reads.
  [`test_jit_input_aliases`](../test/test_kernels.py),
  [`test_jit_returned_views`](../test/test_kernels.py)

### Numbers

[Matmul precision](compiler.md#matmul-precision) has the limits in full. These
are the ones a caller meets first:

- bf16 is the default. Uploads round to nearest even, and a packed bf16 result
  rounds half up, so the two can differ by one bf16 step.
  [`test_fp32`](../test/test_kernels.py)
- A matmul takes four fidelity phases by default, which makes bf16 products
  exact; fewer phases drop bits. Sums are still rounded.
  [`test_matmul_fidelity`](../test/test_kernels.py)
- int32 addition and subtraction wrap, and comparisons cover the full range.
  An int32 matmul accumulates in fp32.
  [`test_int32`](../test/test_kernels.py),
  [`test_integer_boundaries`](../test/test_kernels.py)
- fp32 comparisons treat NaN as unordered, whatever its sign, and `where` keeps
  the selected value's bits. [`test_selection_boundaries`](../test/test_kernels.py)
- bfp8 and bfp4 are storage formats. `cast` makes them, and a matmul by a
  block-float B is bit for bit the bf16 matmul by its values.
  [`test_bfp`](../test/test_kernels.py)
