# Testing and debugging

Every part of baretile has a test that runs on ttsim. This page covers running
the tests, the optional checks that need models or other packages, and what to
do when a kernel hangs.

## Running the tests

Run these from the checkout with Python 3.12 or later, clang and ld.lld on
`PATH`. Set `TTSIM` to a Blackhole `libttsim.so` (see the
[README](../README.md#try-it) for where to get one). You'll need NumPy and
pytest for the tests.

```sh
export TTSIM=/absolute/path/to/libttsim.so
python3 -m pytest -s test
```

The whole suite takes several minutes on ttsim. To run one layer at a time:

```sh
python3 -m pytest -s test/test_device.py
python3 -m pytest -s test/test_kernels.py
python3 -m pytest -s test/test_models.py
```

The tests follow the layers of the code. `test_device.py` starts at PCIe access
and goes through the runtime. `test_kernels.py` tests the instructions and the
tensor operations built from them. Models, the server, the adapters and the
[ramps](../ramps) are in `test_models.py`. `test_layers` checks that each module
imports only the standard library and the layers below it.

If pytest exits without explaining why, run it with `-s`. ttsim stops the
process on errors such as an invalid address or decrementing an empty semaphore.
pytest can otherwise hide that error message. The tests share a `Chip` because
the simulator can only be initialized once per process.

The runtime compiles device programs with clang for `riscv32-unknown-elf` and
links them with lld. Builds go in `$XDG_CACHE_HOME/baretile`, which is
`~/.cache/baretile` by default, so you won't have to
compile them again on the next run. Entries expire after a week without use.

## Optional checks

Most model tests use small configurations. To compare against transformers,
install PyTorch and a transformers release that includes the model you're
testing. The MXFP4 test also needs safetensors, `test_tokenizer` needs
tiktoken, and the adapter tests need tinygrad at the commit that CI pins (see
[models](models.md#tinygrad-and-pytorch)). With a checkout in `TINYGRAD`, they
fail instead of skipping when tinygrad doesn't import. Check pytest's skipped
tests if you expected one of these comparisons to run.

To include pretrained checkpoint checks:

```sh
GPT2=/path/to/gpt2/model.safetensors python3 -m pytest -s test -k gpt2_model
LLAMA=/path/to/SmolLM-135M python3 -m pytest -s test -k llama_model
```

`LLAMA8B=1` enables a test at Llama-3.1-8B block dimensions. `FUZZ=n` raises the
number of random expressions in the kernel fuzz test above its default of 30. `TTSIM2=/path/to/libttsim.so`
enables the two-chip test. ttsim's releases include that library as
`libttsim_bh_x2.so`, or you can build ttsim's `release_bh_x2` target.

`llama.accuracy` compares generated choices with Tenstorrent reference tokens:

```sh
python3 models/llama.py /path/to/model /path/to/<model>.refpt [steps] [mode] [numpy]
```

`numpy` runs the comparison on the host instead. Reading the reference file
requires PyTorch. The usual reference procedure has a 512-token prompt and
500 following steps. Top-1 counts how often baretile picks the reference's best
token. Top-5 counts how often its pick is among the reference's five best tokens.

## Benchmarks

[test/benchmark.py](../test/benchmark.py) records matmul, decode, prefill and
serving results; its opening comment gives the commands and workload sizes.
`load` also drives an external OpenAI-compatible completion server. It asks
for streamed usage and reports missing token counts or cache counts as `null`.
Per-user rates include the whole request, including first-token latency.
First-text latency measures the first nonempty SSE text event: a chunk may
contain several tokens, so it is not an individual token timestamp. An empty
completion has no first-text latency.

A serving workload and each TT-NN demo subprocess have a one-hour deadline.
Python callers can set `load(..., timeout=seconds)` for shorter runs. A failed
or timed-out workload retains completed turns, reports failures, and includes
the unfinished requests' elapsed time in its overall rate. TT-NN subprocesses
write output as they run to `benchmark-ttnn-<run>.log`; a timeout stops their
process group and retains that partial log. Version and line-count subprocesses
have a separate 60-second limit.

## Where ttsim differs from a card

Tensix instructions take no time in ttsim. NoC requests complete immediately.
`STALLWAIT` and `SEMWAIT` hold up the whole thread, whatever their block masks
name, so a wait that names too few units still works there. Most of the cycles
you see come from the RISC-V code that controls the kernel. A lower count can show that
you've reduced that work, but it won't tell you how fast the kernel is on a card.

ttsim also decodes an unexpected `XMOV` in very long loops of inline Tensix
instructions. We don't know why. The DeltaNet kernel calls SFPU functions to
avoid generating a loop that long.

[Bring-up](bring-up.md#what-still-needs-checking) lists the behavior that we can
only check on a card.

## ttsim-chaos

[test/ttsim-chaos.patch](../test/ttsim-chaos.patch) adds NoC delays, FPU and SFPU
latency, random pauses, a 140-tile option and the NIU counter for read bytes to
ttsim v1.11.5 at commit `464814f`. Apply it to a separate ttsim checkout
and build the Blackhole library there:

```sh
git clone --branch v1.11.5 https://github.com/tenstorrent/ttsim ttsim-chaos
cd ttsim-chaos
git apply /path/to/baretile/test/ttsim-chaos.patch
./make.py src/_out/release_bh/libttsim.so
```

Run with delays enabled to check whether the kernels wait for their data:

```sh
TTSIM=/path/to/ttsim-chaos/src/_out/release_bh/libttsim.so \
  TTSIM_CHAOS_NOC=1,400 TTSIM_CHAOS_RISCV=0.2 TTSIM_CHAOS_TENSIX=0.2 TTSIM_CHAOS_MATH=1,40 \
  python3 -m pytest -s test
```

`TTSIM_CHAOS_NOC=lo,hi` delays a request by a random number of clocks in that
range. The RISC-V and Tensix settings give the probability of skipping a turn.
Set `TTSIM_CHAOS_SEED` to repeat a run. `TTSIM_CHAOS_UNHARVESTED=1` enables all
140 Tensix tiles, so you can check the code with a larger grid.

CI builds ttsim-chaos this way for every push and pull request. It runs `test/test_kernels.py`
and the model-specific DeltaNet and MoE kernel checks (`test_delta_step`,
`test_delta_pieces` and `test_moe`)
with the four delay settings above and the run number as the seed, so each run
tries other timings, and it runs the whole suite with `TTSIM_CHAOS_UNHARVESTED=1`.
The log prints the seed, which repeats a failure.

`TTSIM_CHAOS_MATH=lo,hi` gives the FPU and SFPU time. Each of their instructions
runs a random lo to hi clocks after it leaves its thread, and the thread goes on
meanwhile. `STALLWAIT` and `SEMWAIT` then block only the instructions their
block masks name, as the ISA's `STALLWAIT.md` describes. A configuration write,
a `SEMPOST` or a pack that should wait for the math finds it unfinished unless a
`STALLWAIT` holds it back. FPU and SFPU instructions keep their order, and each
one uses the address modes it saw when it left its thread, since tt-llk relies on
both. The unpackers, the packer and the configuration unit still finish at once,
so this doesn't check the waits for them.

The patch also implements the NIU counter for bytes received by reads.
`pci.Sim.counts` detects the patched library. Released ttsim doesn't support that
counter, so profiles from it report `None` for bytes.

## When a launch hangs

`Runtime.wait()` times out after 60 seconds without a completed launch by
default. Its report identifies the current launch and unfinished workers. For
up to four workers, it prints core status, synchronization counters, flags and
outstanding NoC requests. On a card it also reads the program counters.

If a worker never received `go`, check dispatch and multicast first. If the
workers finished but the host is still waiting, check their atomic reports and
the status write to host memory. A read ID that's still outstanding means a
transfer hasn't finished. Use the [pipeline counters](compiler.md#unpack-math-and-pack) to
see where a kernel is waiting.

Creating a new runtime resets its worker cores, but not a Tensix coprocessor that a
hung kernel left waiting. If the next runtime's kernels hang on those workers, or
the NoC or ARC has hung, see [bring-up](bring-up.md#when-a-card-hangs) for the chip
reset command.
