# Program the math of a Tensix tile by hand. We put tensor tiles in one worker's L1 from the host, then run three
# programs on its TRISCs that unpack, compute and pack them. Ramp 2 covers BRISC, which moves the tiles in a tensor kernel.
#   TTSIM=/path/to/libttsim.so python3 ramps/3_tensix.py
# docs/compiler.md explains the semaphores and counters that larger kernels use.
import math, os, struct, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baretile import bh, pci, tensor
from baretile.chip import Chip
from baretile.runtime import Runtime, compile_kernel
from baretile.tensix import (DST_HALF, MATH_END, MUL_PHASE, allocate, elementwise, from_bf16, ins, kernel_config, math_begin, math_program,
                             matmul, pack, pack_program, sfpu, slot_base, to_bf16, tt, unpack, unpack_program)
from baretile.tensor import Tensor

chip = Chip(pci.open_device())
rt = Runtime(chip)
x, y = rt.cores[0]  # a launch over one unit runs on the first worker

def check(value, message="ramp result did not match"):
  if not value: raise AssertionError(message)

# 1. A Tensix instruction is a 32-bit word with the opcode in its top byte and fields below it. bh.TENSIX_OPS has each
# instruction's opcode and field layout from the ISA. A TRISC pushes instructions to its Tensix thread by executing
# them: tt() turns a list of words into C that a TRISC runs, with each word rotated left by 2 bits.
add = ins("ELWADD", dst=0)
check(add >> 24 == bh.TENSIX_OPS["ELWADD"][0] == 0x28)
print(f"1. ELWADD is {add:#010x}, and a TRISC runs it as: {tt([add])}")

# 2. Unpack, math and pack. Thread 0 unpacks tiles from L1 into the source registers SrcA and SrcB. Thread 1 computes
# into Dst. Thread 2 packs Dst back into L1 in bf16. TRISC0, TRISC1 and TRISC2 feed those threads, and BRISC stays idle
# here because the host writes the inputs into L1 itself. L1 slot i is a tile at slot_base + i * 2048; slot 0 holds
# zeros.
def to_l1(slot, m):  # writes a 32 x 32 matrix into an L1 slot as bf16, in four 16 x 16 faces, each stored row by row
  words = [0] * 1024
  for r in range(32):
    for c in range(32): words[(r // 16 * 2 + c // 16) * 256 + r % 16 * 16 + c % 16] = to_bf16(m[r][c])
  chip.write(x, y, slot_base("bf16") + slot * 2048, struct.pack("<1024H", *words))
def from_l1(slot):  # reads a bf16 tile back from an L1 slot as a 32 x 32 matrix
  words = struct.unpack("<1024H", chip.read(x, y, slot_base("bf16") + slot * 2048, 2048))
  return [[from_bf16(words[(r // 16 * 2 + c // 16) * 256 + r % 16 * 16 + c % 16]) for c in range(32)] for r in range(32)]
def run(unpacks, maths, packs, slots=None):  # builds the three TRISC programs from lists of words and runs them on one worker
  rt.launch(compile_kernel(None, unpack_program(tt(unpacks), kernel_config()), math_program(slots or {}, tt(maths)), pack_program(tt(packs))), (), 1)
  rt.wait()

a = [[float((r * 32 + c) % 7 - 3) for c in range(32)] for r in range(32)]
b = [[float((r + 2 * c) % 5 - 2) for c in range(32)] for r in range(32)]
to_l1(1, a), to_l1(2, b)
# unpack(1, 2) puts slot 1 in SrcA and slot 2 in SrcB. math_begin waits for a free half of Dst, elementwise issues the
# ELWADDs, and MATH_END tells pack that the half is full. pack copies 64 rows of Dst, a whole tile, into slot 3.
run(unpack(1, 2), math_begin(False) + elementwise("add", {}) + MATH_END, pack("h", 3))
check(from_l1(3) == [[p + q for p, q in zip(ra, rb)] for ra, rb in zip(a, b)])
print("2. unpacked two tiles, added them in Dst and packed the sum")

# 3. Multiplication runs in fidelity phases. Each phase multiplies some of the significand bits, and the four phases
# together give the exact product. ELWMUL adds to Dst, so math_begin(True) zeroes the half first. The phases are
# recorded in the math thread's replay buffer once, then each replay costs the TRISC one word: allocate() chooses
# where each sequence goes in that buffer.
slots = allocate({"mul"})
run(unpack(1, 2), math_begin(True) + elementwise("mul", slots) + MATH_END, pack("h", 3), slots)
check(from_l1(3) == [[p * q for p, q in zip(ra, rb)] for ra, rb in zip(a, b)])
print(f"3. multiplied in 4 phases of {len(MUL_PHASE) - 1} ELWMULs each")

# 4. The matrix unit computes Dst += SrcB @ SrcA, so the left matrix goes in SrcB: unpack(2, 1) puts B in SrcA and A in
# SrcB to compute A @ B. The sums of these small integers are exact in bf16.
slots = allocate({"mm"})
run(unpack(2, 1), math_begin(True) + matmul(slots) + MATH_END, pack("h", 3), slots)
check(from_l1(3) == [[sum(a[r][k] * b[k][c] for k in range(32)) for c in range(32)] for r in range(32)])
print("4. A @ B with A in SrcB and B in SrcA")

# 5. The SFPU is a 32-lane vector unit that works on Dst. After the FPU adds, sfpu() runs the exp program over the Dst
# half, before pack rounds the result to bf16. Dst holds fp32, so the sum isn't rounded between the two operations.
c = [[(r - k) / 16 for k in range(32)] for r in range(32)]  # exact in bf16, from -1.9375 to 1.9375
to_l1(4, c)
slots = allocate({"exp"})
run(unpack(1, 4), math_begin(False) + elementwise("add", {}) + sfpu(["exp"], slots) + MATH_END, pack("h", 5), slots)
by_hand = from_l1(5)
check(all(abs(got / math.exp(p + q) - 1) < 2**-7 for rg, ra, rc in zip(by_hand, a, c) for got, p, q in zip(rg, ra, rc)))
print(f"5. exp(a + c) in Dst: exp({a[0][1] + c[0][1]}) = {by_hand[0][1]}")

# 6. Dst has two halves of 64 rows, so math can fill one half while pack drains the other. h is the first row of the
# half that the next tile gets, and the math and pack programs each move it on after a tile. The words are pushed
# from C, so we write the programs' loop bodies as C here.
maths = " ".join(f"{tt(math_begin(False) + elementwise('add', {}) + MATH_END)} h ^= {DST_HALF};" for _ in range(2))
packs = " ".join(f"{tt(pack('h', slot))} h ^= {DST_HALF};" for slot in (3, 6))
rt.launch(compile_kernel(None, unpack_program(tt(unpack(1, 2) + unpack(1, 4)), kernel_config()), math_program({}, maths), pack_program(packs)),
          (), 1)
rt.wait()
check(from_l1(3) == [[p + q for p, q in zip(ra, rb)] for ra, rb in zip(a, b)])
check(from_l1(6) == [[p + q for p, q in zip(ra, rc)] for ra, rc in zip(a, c)])
print("6. two sums went through both halves of Dst")

# 7. Tensor builds the same three programs for (A + C).exp(), plus a BRISC program that reads each tile from DRAM and
# writes the result back. tensor.kernels caches the kernel by its expression, where 0 and 1 are the inputs A and C.
A, C = Tensor.of(rt, a), Tensor.of(rt, c)
out = (A + C).exp().realize()
check(list(tensor.kernels)[-1][0] == ("exp", ("add", 0, 1)) and out.tolist() == by_hand)
print("7. Tensor's kernel for (A + C).exp() gave the same tile, bit for bit")

# See tensix.py for the instructions and kernels.py for the kernels that Tensor builds from them.
print(f"the ramp ran {rt.launches} launches")
