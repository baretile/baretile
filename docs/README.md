# baretile documentation

These pages explain how baretile works and how to run it on the simulator or a
card. The [README](../README.md) has the setup. They describe the code as it is
now, and each page links to the files it covers.

## Learning baretile

The ramps are scripts that teach baretile in order, from tensors down to Tensix
instructions. Each one explains its steps in comments and checks its results,
also under `python -O`, so you can change a step and see what breaks. Run them
from the checkout with `TTSIM` set:

```sh
python3 ramps/1_tensors.py
python3 ramps/2_runtime.py
python3 ramps/3_tensix.py
python3 ramps/4_kernel.py
```

1. [Tensors](../ramps/1_tensors.py): operations, fusion, shapes, formats, views,
   weight storage, gradients and `jit`.
2. [Runtime](../ramps/2_runtime.py): C kernels on the workers' RISC-V cores,
   DRAM and the NoC, the command queue, a tensor's tiles, and timeouts.
3. [Tensix](../ramps/3_tensix.py): unpack, math and pack programs written by
   hand, including a matmul and the SFPU, compared with what `Tensor` compiles.
4. [A kernel](../ramps/4_kernel.py): a tensor operation of your own, from its
   BRISC and Tensix programs to a function tested against NumPy.
   [Writing a kernel](kernels.md) walks through it.

After the ramps, [baretile/models/gpt2.py](../baretile/models/gpt2.py) is a whole
model in under 200 lines. It runs the same code on baretile tensors and NumPy arrays.

## Pages

| Page | What it covers |
| --- | --- |
| [Writing a kernel](kernels.md) | A tensor operation of your own, from its four programs to a tested function |
| [Runtime](runtime.md) | Workers and the dispatcher, L1 layout, the command ring, DRAM allocation, `jit` and profiling |
| [Compiler](compiler.md) | Tile layout and storage formats, fusion and reductions, unpack, math and pack, matmul and attention |
| [Models, serving and adapters](models.md) | The model loaders, precision modes, the OpenAI-compatible server, tinygrad and PyTorch, and cost estimates |
| [First P150a workload](workload.md) | Pinned models, correctness gates, single-user limits and the measurements needed for the first release |
| [Testing and debugging](testing.md) | Running the tests, optional checks, ttsim-chaos and hung launches |
| [Hardware reference](hardware.md) | PCIe, TLB windows, coordinates, DRAM, ARC, the RISC-V cores, NoC transfers and Ethernet, with sources |
| [Bring-up on a card](bring-up.md) | Running on a p150 for the first time, measuring it, and what we still need to check |

## How the pieces fit

Each module imports only the layers below it, and layers 0 to 3 need only the
standard library. `bh.py` holds hardware constants, `pci.py` opens the device,
and `chip.py` reads and writes through the network on chip (NoC). The runtime uses these
modules to load and launch programs. The compiler builds kernels from Tensix
instructions, and `Tensor` compiles expressions into those kernels. The models and
adapters are layer 4, in `baretile/models/` and `baretile/adapters/`, and also use
NumPy.

`Tensor` builds an expression when you add, multiply or reduce. `realize()` turns
that expression into kernels and queues them. `tolist()` also waits and copies
the result back to Python. A matmul computes its input matrices first. It can
combine operations on its output into the matmul kernel, as with the ReLU in
the README.

A Tensix tile has local memory and RISC-V cores that control data movement and
computation. A kernel can supply a program for BRISC and each of the three
TRISCs. In tensor kernels, BRISC moves data, TRISC0 feeds the unpacker, TRISC1
feeds the math unit, and TRISC2 feeds the packer. NCRISC stays in reset.
The three TRISCs submit Tensix instructions to separate hardware queues.

We also use "tile" for a 32 by 32 block of tensor values. A worker is a Tensix
tile running a kernel; a tensor tile is the data it processes.

## Where the code is

| Layer | File | What it does |
| --- | --- | --- |
| 0, device | [bh.py](../baretile/bh.py) | Blackhole's addresses, sizes and instruction encodings, as data |
| | [pci.py](../baretile/pci.py) | The device: `Sim` wraps libttsim.so, and `Card` opens a PCIe card through sysfs; checking a card first, and giving it back to tt-kmd |
| | [chip.py](../baretile/chip.py) | NoC reads and writes through TLB windows, ARC messages, telemetry, core reset and Ethernet links; `python3 -m baretile.chip [check \| release \| reset]` |
| 1, runtime | [runtime.py](../baretile/runtime.py) | Builds device C with clang, loads firmware and kernels, queues launches, allocates DRAM and profiles |
| 2, codegen | [tensix.py](../baretile/tensix.py) | Tensix instructions from the ISA, and the BRISC, unpack, math and pack programs of a kernel |
| 3, kernels and tensors | [kernels.py](../baretile/kernels.py) | Fused elementwise, matmul and reduction kernels, attention, data movement and `topk` |
| | [tensor.py](../baretile/tensor.py) | `Tensor`: lazy expressions, broadcasting, views, storage formats, autograd and `jit` |
| Commands | [hello.py](../baretile/hello.py), [bench.py](../baretile/bench.py), [probe.py](../baretile/probe.py), [viz.py](../baretile/viz.py) | The fib demo, a matmul benchmark, card measurements and the launch viewer |
| 4, on top | [baretile/models/](../baretile/models) | GPT-2, the Llama loader and its families, the server and the cost model |
| | [baretile/adapters/](../baretile/adapters) | The BARETILE device for tinygrad |

A kernel that only one workload uses stays in layer 4 next to that workload.
For example, Gated DeltaNet's decode kernel is in `baretile/models/qwen.py`.
