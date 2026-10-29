# Bring-up on a card

These steps take baretile onto a Blackhole p150 for the first time. We haven't
run them on a card yet. Start at the bottom of the stack and stop at the first
failure, since the later tests need the earlier ones to work.

## Before you start

Check that the card appears in `lspci -d 1e52:b140`, has the required power,
and has working board firmware. Check firmware with Tenstorrent's tools before
opening baretile, because `pci.Card` unbinds tt-kmd. Then check what baretile
needs:

```sh
python3 -m baretile.chip check
```

It reads sysfs without touching the card, so it runs without root too. It
checks clang and lld, then reports whether you're root, whether the IOMMU passes DMA addresses through,
how many 2 MiB hugepages are free, whether the PCIe link runs at the card's
best speed and width, and which driver has the card, with tt-kmd's firmware
version. It exits with an error if anything would stop baretile. Configure the
IOMMU for passthrough (`iommu=pt`) or disable it, and reserve a few hugepages as
root:

```sh
echo 4 > /proc/sys/vm/nr_hugepages
```

## Steps

Run the following commands as root from the checkout, with `TTSIM` unset.
Use the Python environment that has the test dependencies.

1. Run `python3 -m baretile.chip`. Check ARC status, the clock, available columns
   and DRAM training before loading worker programs.
2. Run `python3 -m pytest -s test/test_device.py::test_config_space test/test_device.py::test_arc`,
   then `python3 -m pytest -s test/test_device.py::test_dma test/test_device.py::test_l1 test/test_device.py::test_dram`.
   These test register access and memory transfers that the host drives,
   including DRAM at the top of each bank. Nothing runs on the workers yet.
3. Run `python3 -m baretile.hello`, then the rest of `test/test_device.py`.
   `test_host_memory` checks that a worker can read and write host memory, which
   the dispatcher's commands, uploads and readback depend on. Investigate launch
   failures before testing Tensix math.
4. Run `python3 -m pytest -s test/test_kernels.py -k 'tile_math or tensor_math'`.
   This checks instruction encoding and basic unpack, math and pack behavior.
5. Run `python3 -m baretile.probe 8 probe.json`, before anything uploads large
   tensors. Compare `upload over PCIe`, where workers read the data from host
   memory, with `upload over PCIe, through the BAR`. `tensor.DMA_UPLOAD` picks
   between the two by size, and its value is a guess until these measurements exist.
6. Run the rest of `test/test_kernels.py`, then the model tests. If a failure
   comes and goes, check the waits that ttsim doesn't model. ttsim-chaos models
   some of them (see [testing](testing.md#ttsim-chaos)).
7. Run `python3 -m baretile.bench`, and pass `probe=probe.json` to
   `baretile/models/cost.py`.

## Giving the card back

Once no baretile process is running, give the card back to tt-kmd so that
tt-smi and TT-NN can use it:

```sh
python3 -m baretile.chip release
```

`test/benchmark.py` runs both stacks on the same card, so its TT-NN runs need
this first.

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
from the drivers and simulator because the ISA doesn't document them. The SFPU
programs read -1.0 from LReg 11, which leaving the SFPU's soft reset sets and
tt-llk keeps there. baretile doesn't reset the SFPU itself, so check that the
value is there when kernels start.
`python3 -m baretile.chip release` is tested against fake sysfs files only.

The host relies on x86 ordering between writes to the ring and the doorbell.
There's no explicit barrier for other host architectures.

There are also ISA and simulator differences: `UNPACR` fields have different
names, `SETDVALID` is unavailable in ttsim, and some address-mode documentation
describes Wormhole. The code uses the encodings accepted by the Blackhole simulator.
