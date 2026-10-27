# Bring-up on a card

These steps take baretile onto a Blackhole p150 for the first time. We haven't
run them on a card yet. Start at the bottom of the stack and stop at the first
failure, since the later tests need the earlier ones to work.

## Before you start

Check that the card appears in `lspci -d 1e52:b140`, has the required power,
and has working board firmware. Check firmware with Tenstorrent's tools before
opening baretile, because `pci.Card` unbinds tt-kmd. Configure the IOMMU for
passthrough or disable it, and reserve a few 2 MiB hugepages as root:

```sh
echo 4 > /proc/sys/vm/nr_hugepages
```

## Steps

Run the following commands as root from the checkout, with `TTSIM` unset.
Use the Python environment that has the test dependencies.

1. Run `python3 -m baretile.chip`. Check ARC status, the clock, available columns
   and DRAM training before loading worker programs.
2. Run `python3 -m pytest -s test/test_device.py -k 'config_space or arc'`, then
   `python3 -m pytest -s test/test_device.py -k 'dma or l1 or (dram and not freed)'`.
   These test register access and memory transfers.
3. Run `python3 -m baretile.hello`, then the rest of `test/test_device.py`.
   Investigate launch failures before testing Tensix math.
4. Run `python3 -m pytest -s test/test_kernels.py -k 'tile_math or tensor_math'`.
   This checks instruction encoding and basic unpack, math and pack behavior.
5. Run the rest of `test/test_kernels.py`, then the model tests. If a failure
   comes and goes, check the waits that ttsim doesn't model.
6. Run `python3 -m baretile.probe 8 probe.json` and
   `python3 -m baretile.bench`. The probe saves measurements for the cost model.

## When a card hangs

A new runtime resets the cores after a kernel hangs. If ARC or the NoC stops
responding, close processes using the card before trying
`python3 -m baretile.chip reset`. That writes PCIe configuration, then removes
and rescans the device. So far, we've tested the reset code against fake sysfs
files. We don't know yet whether it recovers a hung card.

## Measuring a card

`baretile.probe` measures launch overhead, transfers, memory latency and Tensix
operations. `baretile.bench` runs a matmul. Use a card for performance numbers.

`test/benchmark.py` compares matmul, Llama decode and prefill with TT-NN. Its
fourth workload sends concurrent HTTP requests to baretile. The TT-NN runs need
tt-metal's Python environment. You'll also need PyTorch to load the reference
file, and `cloc` to count lines of code. The script lists its commands at the top.

## What still needs checking

On a card, we need to check that DMA reaches host memory, ARC turns the clocks
on, and the status writes arrive in order. We also need to check that loading
a new kernel invalidates the old instructions and that the Tensix registers
are at the indices we use. The BAR2 offsets and parts of ARC's protocol come
from the drivers and simulator because the ISA doesn't document them.

The host relies on x86 ordering between writes to the ring and the doorbell.
There's no explicit barrier for other host architectures.

There are also ISA and simulator differences: `UNPACR` fields have different
names, `SETDVALID` is unavailable in ttsim, and some address-mode documentation
describes Wormhole. The code uses the encodings accepted by the Blackhole simulator.

