# Bring-up on a card

These steps take baretile onto a Blackhole p150 for the first time. We haven't
run them on a card yet. Start at the bottom of the stack and stop at the first
failure, since the later tests need the earlier ones to work.

## Before you start

Check that the card appears in `lspci -d 1e52:b140`, has the required power,
and has working board firmware. Install tt-kmd, Tenstorrent's kernel driver, and
check the firmware with Tenstorrent's tools. Then check what baretile needs:

```sh
python3 -m baretile.chip check
```

It reads sysfs without touching the card, so it runs without root. It checks
clang and lld, then how baretile will open the card. It exits with an error if
anything would stop baretile.

With tt-kmd driving the card, baretile opens its device file,
`/dev/tenstorrent/0`, as `pci.Kmd`, which needs no root. The check reports
tt-kmd's version and whether you can open the file. kmd pins the memory that the
card reaches by DMA. With an IOMMU translating, that's ordinary memory;
otherwise it's a hugepage, so the check also counts the free ones. tt-smi can
use the card at the same time.

Without tt-kmd, baretile opens the card through sysfs as `pci.Card`, which
unbinds any other driver. The check reports whether you're root, whether the
IOMMU passes DMA addresses through, and how many hugepages are free. Configure
the IOMMU for passthrough (`iommu=pt`) or disable it.

Either way, it reports whether the PCIe link runs at the card's best speed and
width, and the firmware version that tt-kmd reads. To reserve hugepages, as
root:

```sh
echo 4 > /proc/sys/vm/nr_hugepages
```

## Steps

Run bring-up from the checkout, with `TTSIM` unset, in the Python environment
that has the test dependencies. Through sysfs, run it as root:

```sh
python3 test/bringup.py record
```

It runs the stages below in order and stops at the first failure, since the
later stages need the earlier ones to work. It says where the failing stage's
output is and what to look at. Each stage's output goes in `record/<stage>.log`.
`record/record.json` holds the device, the host, the commit, and each stage's
result and time, so keep the folder as the record of the bring-up. After a fix,
`--from <stage>` starts again at that stage, and `--only` runs the stages you
name, separated by commas.

1. **preflight** runs `python3 -m baretile.chip check`, as above.
2. **chip** runs `python3 -m baretile.chip`: ARC status, the clock, available
   columns and DRAM training, before any worker program loads.
3. **transfers** runs `test_config_space`, `test_arc`, `test_dma`, `test_l1` and
   `test_dram` from `test/test_device.py`. They test register access and memory
   transfers that the host drives, including DRAM at the top of each bank.
   Nothing runs on the workers yet.
4. **launches** runs `python3 -m baretile.hello`, then all of
   `test/test_device.py`. `test_host_memory` checks that a worker can read and
   write host memory, which the dispatcher's commands, uploads and readback
   depend on.
5. **math** runs `test_tile_math` and `test_tensor_math` from
   `test/test_kernels.py`, which check instruction encoding and basic unpack,
   math and pack.
6. **probe** runs `python3 -m baretile.probe 8 record/probe.json` before anything
   uploads large tensors. Compare `upload over PCIe`, where workers read the data
   from host memory, with `upload over PCIe, through the BAR`. `tensor.DMA_UPLOAD`
   picks between the two by size, and its value is a guess until these
   measurements exist.
7. **kernels** runs all of `test/test_kernels.py`. If a failure comes and goes,
   check the waits that ttsim doesn't model. ttsim-chaos models some of them (see
   [testing](testing.md#ttsim-chaos)).
8. **models** runs `test/test_models.py`.
9. **smollm** runs `test_llama_model` on SmolLM-135M's checkpoint, given by
   `--smollm path/to/SmolLM-135M`.
10. **llama** runs the [first workload](workload.md)'s token accuracy, given
    `--llama path/to/Llama-3.1-8B-Instruct` and `--reference
    path/to/Llama-3.1-8B-Instruct.refpt`. It passes at top-1 90% and top-5 98%
    over 500 steps.

The stages run on ttsim too, which is how they're tested. Once they pass on a
card, run `python3 -m baretile.bench`, and pass `probe=record/probe.json` to
`python3 -m baretile.models.cost`.

## Giving the card back

Through tt-kmd, baretile never takes the card away, and TT-NN can use it once no
baretile process has it open. Through sysfs, `pci.Card` unbound tt-kmd, so once no
baretile process is running, give the card back to it so that tt-smi and TT-NN can
use it:

```sh
python3 -m baretile.chip release
```

`test/benchmark.py` runs both stacks on the same card, so through sysfs, its
TT-NN runs need this first.

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
