# Run C kernels on the workers' RISC-V cores, and move data between DRAM and L1 over the NoC. This is the runtime
# that tensor kernels use. Ramp 3 programs the Tensix math that those kernels feed.
#   TTSIM=/path/to/libttsim.so python3 ramps/2_runtime.py
# docs/runtime.md explains the dispatcher and the command ring.
import array, os, random, struct, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baretile import bh, pci
from baretile.chip import Chip
from baretile.runtime import DATA, Runtime, compile_kernel
from baretile.tensix import BANKS
from baretile.tensor import Tensor

# 1. The host reads and writes any tile's memory through windows in PCIe BAR0, set up by chip.py. A Tensix tile has
# 1.5 MiB of L1, which its RISC-V cores and Tensix share. Coordinates are (x, y) on the NoC, the network on chip.
chip = Chip(pci.open_device())
rt = Runtime(chip)
(x, y), (dx, dy) = rt.cores[0], rt.dispatcher
chip.write32(x, y, DATA, 1234)
assert chip.read32(x, y, DATA) == 1234
print(f"1. {len(chip.tensix)} Tensix tiles and {len(bh.DRAM)} DRAM banks. The first worker is at {(x, y)}, and the dispatcher at {(dx, dy)}")

# 2. compile_kernel uses clang to build programs for a worker's RISC-V cores: BRISC and three TRISCs. You can omit
# programs for unused cores, and this kernel uses only BRISC. kmain receives a pointer to its arguments in L1.
# noc_me() returns the worker's coordinates as x | y << 6. rt.args() is where the last launch's arguments are.
ME = "void kmain(volatile unsigned *a) { a[2] = noc_me(); }"
rt.launch(compile_kernel(ME))
rt.wait()
assert [chip.read32(cx, cy, rt.args() + 8) for cx, cy in rt.cores] == [cx | cy << 6 for cx, cy in rt.cores]
print(f"2. {len(rt.cores)} workers wrote their own coordinates")

# 3. A launch divides its work units among the workers, one each by default. Each worker receives [first unit, unit
# count, *args]. Here 1,000 units are split into ranges that cover them in order, and every worker gets one.
# Passing (100, 0) puts 100 in a[2] and reserves a[3] for the answer. Wait for completion before reading it.
SQUARE = "void kmain(volatile unsigned *a) { a[3] = a[0] * a[0] + a[2]; }"
rt.launch(compile_kernel(SQUARE), (100, 0), units=1000, name="square")
rt.wait()
first, count, _, answer = zip(*(struct.unpack("<4I", chip.read(cx, cy, rt.args(), 16)) for cx, cy in rt.cores))
assert first[0] == 0 and all(f + n == g for f, n, g in zip(first, count, first[1:] + (1000,))) and min(count) > 0
assert list(answer) == [f * f + 100 for f in first]
print(f"3. workers took {min(count)} or {max(count)} of the 1000 units")

# 4. DRAM has eight banks, each a tile on the NoC. dram_alloc reserves the same address range in all eight. Each of the
# first eight workers reads 4 KiB from its own bank into L1, adds its index to every word, and writes the words back.
# noc_read and noc_write start transfers, and noc_wait waits for them to finish.
n, addr = 4096, rt.dram_alloc(4096)
data = [random.randbytes(n) for _ in bh.DRAM]
for (bx, by), d in zip(bh.DRAM, data): chip.write(bx, by, addr, d)
ADD = BANKS + """
void kmain(volatile unsigned *a) {  // a: [first unit, units, address, n]
  unsigned i = a[0], *buf = (unsigned *)L1_DATA;
  noc_read(BANK(i), a[2], L1_DATA, a[3]);  // BANK(i) is the x, y of bank i % 8
  noc_wait();
  for (unsigned k = 0; k < a[3] / 4; k++) buf[k] += i;
  noc_write(L1_DATA, BANK(i), a[2], a[3]);
  noc_wait();
}"""
rt.launch(compile_kernel(ADD), (addr, n), units=len(bh.DRAM), name="add")
rt.wait()
for i, ((bx, by), d) in enumerate(zip(bh.DRAM, data)):
  assert array.array("I", chip.read(bx, by, addr, n)) == array.array("I", [(w + i) & 0xFFFFFFFF for w in array.array("I", d)])
print(f"4. eight workers read, changed and wrote back 4 KiB of their bank at {addr:#x}")

# 5. A worker can write into another worker's L1 the same way. Each worker sends its index to the next worker.
# a[3...] lists the workers' coordinates, which the kernel needs because the NoC addresses tiles by position.
PASS = """
void kmain(volatile unsigned *a) {  // a: [index, 1, workers, each worker's x | y << 6]
  unsigned i = a[0], next = a[3 + (i + 1) % a[2]];
  *(volatile unsigned *)L1_DATA = i;
  noc_write(L1_DATA, next & 63, next >> 6, L1_DATA + 64, 4);
  noc_wait();
}"""
rt.launch(compile_kernel(PASS), (len(rt.cores), *[cx | cy << 6 for cx, cy in rt.cores]), name="pass")
rt.wait()
assert [chip.read32(cx, cy, DATA + 64) for cx, cy in rt.cores] == [(i - 1) % len(rt.cores) for i in range(len(rt.cores))]
print("5. each worker wrote its index into the next worker's L1")

# 6. Commands go into a ring in host memory, and the dispatcher reads them from there. A launch costs the host one
# PCIe write, the doorbell, and launches run in order. rt.profile groups the launches since a count by name, with the
# cycles from each launch's go to its last worker's report. ttsim's cycles don't tell you how fast a card would be.
count = compile_kernel("void kmain(volatile unsigned *a) { *(volatile unsigned *)L1_DATA += a[2]; }")
chip.write32(x, y, DATA, 0)
since = rt.launches
for i in range(100): rt.launch(count, (i,), units=1, name="count")
[(name, launches, cycles, _)] = rt.profile(since)
assert chip.read32(x, y, DATA) == sum(range(100)) and (name, launches) == ("count", 100)
print(f"6. 100 launches in a row added up to {sum(range(100))}, in {cycles} simulated cycles")

# 7. A tensor's tiles live in DRAM. In the default layout, tile i is in bank i % 8, at i // 8 tiles past the tensor's
# address. Each 32 x 32 tile holds four 16 x 16 faces: top left, top right, bottom left and bottom right. The values
# within each face are stored row by row, and an int32 tile occupies 4096 bytes.
m = Tensor.of(rt, [[r * 64 + c for c in range(64)] for r in range(32)], "int32")
for i in range(2):  # tile i holds columns 32i to 32i + 31
  tile = array.array("i", chip.read(*bh.DRAM[i], m.addr, 4096))
  assert all(tile[(r // 16 * 2 + c // 16) * 256 + r % 16 * 16 + c % 16] == r * 64 + 32 * i + c for r in range(32) for c in range(32))
print(f"7. read both tiles of m from banks 0 and 1 at {m.addr:#x}")

# 8. Tensor.run launches your own kernel over a tensor's tiles, one unit per tile. For each tensor argument it passes
# the address, the batch and row strides in tiles, and the offset of the first tile, so the kernel works on views too.
# run caches the compiled kernel by its key. This kernel squares every value of an int32 tensor.
SQUARES = BANKS + """
void kmain(volatile unsigned *a) {  // a: [first unit, units, out's address, strides and offset, then x's, tile columns]
  for (unsigned u = 0; u < a[1]; u++) {
    unsigned t = a[0] + u, i = t / a[10], j = t % a[10], from = i * a[8] + j + a[9], to = i * a[4] + j + a[5];
    int *v = (int *)L1_DATA;
    noc_read(BANK(from), AT(a[6], from, 4096), L1_DATA, 4096);
    noc_wait();
    for (unsigned k = 0; k < 1024; k++) v[k] *= v[k];
    noc_write(L1_DATA, BANK(to), AT(a[2], to, 4096), 4096);
    noc_wait();
  }
}"""
out = Tensor(rt, m.rows, m.cols, dtype="int32")
out.run(("squares",), lambda: compile_kernel(SQUARES), out, m, 2, name="squares")
assert out.tolist() == [[(r * 64 + c) ** 2 for c in range(64)] for r in range(32)]
print("8. squared m's tiles with our own kernel")

# 9. Every wait has a timeout. A kernel that never finishes makes rt.wait raise TimeoutError, with a report of the
# workers that are still running. A new Runtime resets the workers' cores.
rt.launch(compile_kernel("void kmain(volatile unsigned *a) { for (;;); }"), units=1, name="hang")
try: rt.wait(timeout=0.5)
except TimeoutError as e: report = str(e)
assert "still running on" in report and "not finished by BRISC" in report
launches, rt = rt.launches, Runtime(chip)
rt.launch(compile_kernel(ME))
rt.wait()
print("9. a hung kernel timed out, and a new runtime reset the workers:\n  " + report.splitlines()[1].strip())

# See runtime.py for the firmware and the dispatcher, and chip.py for the TLB windows.
print(f"the ramp ran {launches + rt.launches} launches")
