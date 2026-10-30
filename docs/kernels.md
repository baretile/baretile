# Writing a kernel

This page writes a tensor operation of your own, from its four programs to a
tested function. Its code is [ramps/4_kernel.py](../ramps/4_kernel.py), which
runs as a test, so the page and the code can't drift apart. Read ramps 2 and 3
first: they cover BRISC and the NoC, and unpack, math and pack.

The operation is `relu(a + b)` on two bf16 tensors. `Tensor` builds the same
thing, which lets the ramp check the result against NumPy and against
`Tensor`'s kernel, bit for bit. The helpers it uses come from `tensix.py` and
`runtime.py`. They're experimental, so they can change between releases.

## The shape of a kernel

A kernel has four programs that run together on each worker:

- BRISC fetches input tiles from DRAM into L1 and writes results back.
- TRISC0 feeds the unpacker, which moves tiles from L1 into SrcA and SrcB.
- TRISC1 feeds the math unit: the FPU and the SFPU, which compute into Dst.
- TRISC2 feeds the packer, which moves Dst back into L1.

`compile_kernel(brisc, unpacks, maths, packs)` builds the four into one image,
cached by its source. `Runtime.launch(kernel, args, units)` divides the units
between the workers, and each worker gets `[first unit, units, *args]`. Here a
unit is a result tile. The arguments are the DRAM addresses of the result and of
`a` and `b`, so the C reads them as `a[2]`, `a[3]` and `a[4]`.

All four programs run the same loop over the worker's tiles. Tile `t` of a
tensor is in DRAM bank `t % 8`, and BRISC fetches it with `get(base, t, slot)`.
Two frames of L1 slots take turns, so that BRISC can fetch the next tile while
the TRISCs work on this one. Frame `f` holds `a` in slot `FRAMES + 3f`, `b`
after it, and the result after `b`.

## The four programs

The programs pass each frame along through counters in L1. The helpers that
`brisc_program`, `unpack_program`, `math_program` and `pack_program` put around
the loop wait on those counters:

| Program | For each tile | Built with |
| --- | --- | --- |
| BRISC | `reached(frame_done[f])` waits until the frame's last tiles are unpacked, writing finished results to DRAM meanwhile. Two `get`s fetch the inputs, `ready()` waits for them to land and tells TRISC0, and `frame_done[f] = ++events` names the unpack that will free the frame. | `brisc_program("a[1]", ...)`, which writes the remaining results after the loop |
| TRISC0 | `fetched()` waits for BRISC. `unpack(a_slot, b_slot)` fills SrcA and SrcB, and `REPORT` tells BRISC the slots are free. | `unpack_program(..., kernel_config())` |
| TRISC1 | `math_begin` waits for a free half of Dst, `elementwise("add")` adds on the FPU, `sfpu(["relu"])` applies relu in fp32, and `MATH_END` hands the half to pack. `h` moves on to the other half. | `math_program(slots, ...)`, where `allocate({"relu"})` places relu's program in the math thread's replay buffer |
| TRISC2 | `output(f, slot, t)` queues the result for DRAM tile `t`, once BRISC has written what that slot held before. `pack` rounds the half to bf16 into the slot and posts `SEM_PACKED` for BRISC. | `pack_program(...)` |

A missing wait shows up as a wrong tile or a hang, but not always on ttsim,
which finishes most units at once. [ttsim-chaos](testing.md#ttsim-chaos) delays
the NoC and the math units, and that finds most of these. Run the ramp under it
after changing a program. CI runs it there with every kind of delay.

## The operation

`add_relu(a, b)` checks its inputs, realizes them, makes a tensor for the result
and launches the kernel over its tiles. The inputs must be stored the way
`Tensor` stores a new tensor, tile by tile, so a slice or a repeat would have to
be copied first. Then the kernel's result is an ordinary tensor: it can be read
back, used in other operations, or freed like any other.

To test the kernel, compare it with NumPy on inputs whose result is exact in bf16,
as the ramp does with small integers, and on a shape that gives each worker
several tiles, so both frames take turns. The ramp runs on four workers for that.

## Going further

`kernels.py` builds `Tensor`'s kernels the same way, with more around them:
fusion of many operations, reductions over several passes, matmuls that stream
B, and block floats. `transpose_kernel` is the closest to this one.
[Compiler](compiler.md) explains the counters and semaphores they use, and
`tensix.py` has the instructions.
To read the C that `kernels.py` generates for an operation, run it with
[`BARETILE_DUMP`](testing.md#reading-a-kernels-programs) set.
