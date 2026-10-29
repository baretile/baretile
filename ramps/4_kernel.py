# Write a tensor operation of your own, from its four programs. BRISC reads each input tile from DRAM into L1 and
# writes each result back, and the TRISCs unpack, compute and pack it, as in ramp 3. The operation is relu(a + b).
# Tensor builds that too, so the ramp checks the kernel against NumPy and against Tensor's kernel, bit for bit.
#   python3 ramps/4_kernel.py
# docs/kernels.md walks through it. To check its waits, run it under ttsim-chaos (docs/testing.md).
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from baretile import pci
from baretile.chip import Chip
from baretile.runtime import Runtime, compile_kernel
from baretile.tensix import (DST_HALF, FRAMES, MATH_END, REPORT, SEM_PACKED, allocate, brisc_program, elementwise, kernel_config, math_begin,
                             math_program, pack, pack_program, sfpu, tt, unpack, unpack_program)
from baretile.tensor import Tensor

def check(ok):  # an assert that python -O keeps. The traceback shows which check failed
  if not ok: raise AssertionError("the ramp's result is wrong")

chip = Chip(pci.open_device())
rt = Runtime(chip, end=chip.tensix[4])  # four workers and the dispatcher, so each worker takes several tiles

# 1. A launch divides its units between the workers, and each worker gets [first unit, units, *args]. Here a unit is a
# result tile, and the arguments are the DRAM addresses of the result and of a and b: a[2], a[3] and a[4] in the C.
# Tile t of a tensor is in DRAM bank t % 8. get(base, t, slot) fetches it into an L1 slot, and BANK and AT locate it.
# Every program runs the same loop over its worker's tiles. Two frames of L1 slots take turns, so BRISC can fetch the
# next tile while the TRISCs work on this one. Frame f holds a in slot FRAMES + 3f, b after it, and the result after b.
def loop(frame0, frame1):
  return f"""  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned t = a[0] + u;
    if (u & 1) {{ {frame1} }} else {{ {frame0} }}
  }}"""
slot = lambda f, k: FRAMES + 3 * f + k  # k is 0 for a, 1 for b and 2 for the result

# 2. BRISC. reached(n) waits until thread 0 has unpacked what the frame held before, which is n unpacks in, and writes
# finished results to DRAM meanwhile. ready() waits for the reads to land, then tells TRISC0. events counts the unpacks,
# so frame_done[f] is the one that frees frame f. brisc_program writes the results that remain once the loop is done.
brisc = brisc_program("a[1]", loop(*[f"reached(frame_done[{f}]); get(a[3], t, {slot(f, 0)}); get(a[4], t, {slot(f, 1)}); ready(); "
                                     f"frame_done[{f}] = ++events;" for f in (0, 1)]))

# 3. TRISC0. fetched() waits for BRISC. unpack puts a's tile in SrcA and b's in SrcB, and REPORT tells BRISC that the
# frame's slots are free again.
unpacks = unpack_program(loop(*[f"fetched(); {tt(unpack(slot(f, 0), slot(f, 1)) + REPORT)}" for f in (0, 1)]), kernel_config())

# 4. TRISC1. math_begin waits for a free half of Dst, the FPU adds the two tiles into it, and the SFPU applies relu
# there in fp32. MATH_END hands the half to pack, and h moves on to the other half.
slots = allocate({"relu"})  # where relu's program goes in the math thread's replay buffer
maths = math_program(slots, loop(*[tt(math_begin(False) + elementwise("add", {}) + sfpu(["relu"], slots) + MATH_END) + f" h ^= {DST_HALF};"] * 2))

# 5. TRISC2. output(f, slot, t) queues the result in that slot for DRAM tile t, once BRISC has written what the slot
# held before. pack rounds the Dst half to bf16 into the slot and posts SEM_PACKED, which tells BRISC it's there.
packs = pack_program(loop(*[f"output({f}, {slot(f, 2)}, t); {tt(pack('h', slot(f, 2), SEM_PACKED))} h ^= {DST_HALF};" for f in (0, 1)]))

KERNEL = compile_kernel(brisc, unpacks, maths, packs)  # cached by its source, so later runs reuse the image
print(f"1-5. relu(a + b) compiled into a {len(KERNEL)}-byte image")

# 6. The operation launches the kernel over the result's tiles. Its inputs must be realized tensors stored as Tensor
# stores a new one, tile by tile, so a slice or a repeat would have to be copied first.
def add_relu(a, b):
  if (a.shape, a.dtype) != (b.shape, b.dtype) or a.dtype != "bf16": raise ValueError("add_relu takes two bf16 tensors of the same shape")
  a, b = a.realize(), b.realize()
  out = Tensor(rt, a.rows, a.cols, batch=a.batch)
  rt.launch(KERNEL, (out.addr, a.addr, b.addr), a.tiles, name="add_relu")
  return out

rng = np.random.default_rng(4)
x, y = rng.integers(-8, 8, (2, 64, 96)), rng.integers(-8, 8, (2, 64, 96))  # integers, so the sums are exact in bf16
tx, ty = Tensor.of(rt, x.tolist()), Tensor.of(rt, y.tolist())
mine = add_relu(tx, ty)
check(np.array(mine.tolist()).tolist() == np.maximum(x + y, 0).tolist())
print(f"6. add_relu ran over {tx.tiles} tiles on {min(tx.tiles, len(rt.cores))} workers and matched NumPy")

# 7. Tensor fuses (a + b).relu() into the same four programs, with more around them. Both kernels give the same bits.
check(mine.tobytes() == (tx + ty).relu().tobytes())
print("7. it matched Tensor's (a + b).relu() bit for bit")

print(f"the ramp ran {rt.launches} launches")
