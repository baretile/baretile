# Measure launch cost, DRAM and NoC latency, bandwidth, Tensix throughput, BRISC topk and PCIe
# transfers on a card. See NOTES.md, "Bring-up on a p150". Use these measurements to choose
# buffer depths, request sizes, bank assignments and matmul fidelity.
# The probes use small kernels built with tensix.py and timed by the tile's wall clock.
# ttsim can check that they run, but its timings don't predict card performance.
#   TTSIM=/path/to/libttsim.so python3 -m baretile.probe [MiB] [out.json]
# MiB is the upload and readback size (default 8). models/cost.py can read out.json.
import functools, json, sys, time
from baretile import bh, pci
from baretile.chip import Chip
from baretile.tensix import (BANKS, CHUNKS, DEPTH, DST_HALF, FRAMES, MATH_END, allocate, chunk_tiles, ins, kernel_config, math_begin, math_program,
                            matmul, pack, pack_program, run, sfpu, table, tile_bytes, tt, unpack, unpack_program)
from baretile.runtime import DATA, Runtime, compile_kernel
from baretile.tensor import Tensor

EMPTY = "void kmain(volatile unsigned *a) {}"

# BRISC measures DRAM reads from base address a[2]. Each worker issues a[3] reads of a[6] bytes per round,
# for a[4] rounds, and stores elapsed cycles in a[5]. a[0] selects the worker's share of the data.
# With a[9], use the worker's bank and port from the `reader` table prepended from Runtime.readers.
# Otherwise, a nonzero a[7] selects one bank by NoC coordinates x | y << 6. With neither, interleave reads
# across banks as in ordinary tensor storage. A nonzero a[8] splits each read into that many bytes followed
# by the remainder, matching getb's separate requests for block-float exponents and values.
DRAM_READS = BANKS + f"""
void kmain(volatile unsigned *a) {{
  unsigned d = a[3], n = a[6], s = a[8], t0 = CLOCK;
  for (unsigned r = 0; r < a[4]; r++) {{
    for (unsigned i = 0; i < d; i++) {{
      unsigned t = a[0] * d + i, xy = a[9] ? reader[a[0]] : a[7] ? a[7] : bank[t % {len(bh.DRAM)}], l1 = {DATA:#x} + i * n;
      unsigned at = a[7] || a[9] ? a[2] + t * n : AT(a[2], t, n);
      if (s) noc_get(xy & 63, xy >> 6, at, l1, s);
      noc_get(xy & 63, xy >> 6, at + s, l1 + s, n - s);
    }}
    noc_wait();
  }}
  a[5] = CLOCK - t0;
}}"""

# BRISC alone. Workers a[2] and a[3], at NoC a[4] and a[5], bounce a word back and forth a[6] times. Worker a[2]
# times it and writes the cycles into a[8].
PING = f"""
#define FLAG ((volatile unsigned *){DATA:#x})
#define WORD ((volatile unsigned *){DATA + 64:#x})
void kmain(volatile unsigned *a) {{
  unsigned me = a[0], first = me == a[2], other = first ? a[5] : a[4], t0 = CLOCK;
  if (!first && me != a[3]) return;
  // a[7] is this launch's first value, so a value left over from another launch can't match
  for (unsigned r = 0, v = a[7]; r < a[6]; r++, v += 2) {{
    if (!first) while (*FLAG != v + 1) asm volatile("fence" ::: "memory");
    *WORD = v + 1 + !first;
    noc_write({DATA + 64:#x}, other & 63, other >> 6, {DATA:#x}, 4);
    if (first) while (*FLAG != v + 2) asm volatile("fence" ::: "memory");
  }}
  if (first) a[8] = CLOCK - t0;
}}"""

def tensix(unpacks="", maths="", packs="", slots=None):
  """Build a kernel from C bodies for the three TRISCs. Each body runs a[2] times, on the half of Dst in h. The last
  TRISC that has work times its loop, up to the last instruction of its thread, and writes the cycles into a[3]."""
  timed = [i for i, body in enumerate((unpacks, maths, packs)) if body][-1]
  loop = lambda i, body: (f"unsigned t0 = CLOCK; for (unsigned i = 0; i < a[2]; i++) {{ {body} }}" +
                          (" tensix_idle(); args[3] = CLOCK - t0;" if i == timed else "")) if body else ""
  return compile_kernel(None, unpack_program(loop(0, unpacks), kernel_config()), math_program(slots or {}, loop(1, maths)),
                        pack_program(loop(2, packs)))

@functools.cache
def kernels():
  mm = allocate({"mm"})
  steps = {p: tensix(tt(unpack(1, 2)), tt(matmul(mm, p)), slots=mm) for p in (1, 2, 4)}  # a matmul step: two tiles in, p phases
  # MVMULs alone. The Src banks are filled once and kept, since the replayed phases don't give them back, and they're
  # released at the end.
  mvmul = compile_kernel(None, unpack_program(tt(unpack(1, 2)), kernel_config()),
                         math_program(mm, "unsigned t0 = CLOCK; for (unsigned i = 0; i < a[2]; i++) { " +
                                       tt([ins("SETRWC", bit_mask=0b1111)] + run(mm, "mm") * 4) +
                                       " } " + tt(matmul(mm, 1)) + " tensix_idle(); args[3] = CLOCK - t0;"), pack_program(""))
  packs = tensix(maths=tt(math_begin(False) + MATH_END) + f" h ^= {DST_HALF};", packs=tt(pack("h", FRAMES)) + f" h ^= {DST_HALF};")
  ex = allocate({"exp"})
  return dict(steps=steps, mvmul=mvmul, pack=packs, exp=tensix(maths=tt(sfpu(["exp"], ex)), slots=ex),
              empty=compile_kernel(EMPTY), ping=compile_kernel(PING))

def probe(chip, n=64, rounds=32, upload=8 << 20):
  """Run the probes and return rows of (what, measured, bound, unit). Tensix probes loop n times, and NoC probes run
  `rounds` rounds."""
  k, rows, hz = kernels(), [], chip.telemetry()[bh.TAG_AICLK] * 1e6
  rt = Runtime(chip)
  first, workers, grid = rt.cores[0], len(rt.cores), len(rt.readers)
  reading = compile_kernel(table("reader", [bh.noc_xy(*port) for _, port in rt.readers]) + "\n" + DRAM_READS)
  result = lambda core, i: chip.read32(*core, rt.args() + 4 * i)
  def timed(kernel, args, units, at=3):  # launch once. the first worker's cycles from a[at], or the launch's if at is None
    since = rt.launches
    rt.launch(kernel, args, units)
    rt.wait()
    return rt.profile(since)[0][2] if at is None else result(first, at)
  # A launch alone, timed by the host's round trip and by the chip from go to last report. Then 100 queued back to back.
  start = time.perf_counter()
  cycles = timed(k["empty"], (), workers, at=None)
  rows += [("launch, alone: host round trip", (time.perf_counter() - start) * 1e6, None, "us"),
           ("launch, alone: on the chip", cycles, None, "cycles")]
  since, start = rt.launches, time.perf_counter()
  rt.send(*rt.block([(k["empty"], (), workers, "")] * 100))
  rt.wait()
  rows += [("launch, 100 queued: host each", (time.perf_counter() - start) * 1e4, None, "us"),
           ("launch, 100 queued: chip each", sum(c for _, _, c, _ in rt.profile(since)) / 100, None, "cycles")]
  # Tensix, in cycles per tile or step: a matmul step with its unpacks, MVMULs alone, packs, and an SFPU program
  for p, kernel in k["steps"].items():
    rows.append((f"matmul step, {p} phase{'s' * (p > 1)}, unpacks included", timed(kernel, (n,), 1) / n, 16 * p, "cycles"))
  rows.append(("MVMUL alone", 64 * n / max(timed(k["mvmul"], (n,), 1), 1), 1, "per cycle"))
  rows.append(("pack a bf16 tile", timed(k["pack"], (n,), 1) / n, None, "cycles"))
  rows.append(("SFPU exp of a tile", timed(k["exp"], (n,), 1) / n, None, "cycles"))
  # BRISC's scalar topk. A mixture of experts picks a token's best 8 of 256 experts, and sampling takes the 64 largest
  # fp32 logits from a 2,560-wide part of the vocabulary (serve's heads).
  for what, make in (("topk, a token's 8 of 256 experts", lambda: Tensor.of(rt, [[float(i * 37 % 256) for i in range(256)]]).topk(8)),
                     ("topk, a row's 64 of 2,560 logits", lambda: Tensor.of(rt, [[float(i * 37 % 2560) for i in range(2560)]], "fp32").topk(64))):
    since = rt.launches
    make()
    rt.wait()
    rows.append((what, rt.profile(since)[0][2], None, "cycles"))
  # DRAM. The bandwidth comes from the cycles of the slowest worker. The probes:
  # - all workers, by request size and reads in flight, interleaved over the banks the way tiles are
  # - block-float tiles the way the matmuls read them, each in two requests: 1,088 bytes for bfp8, 576 for bfp4
  # - one worker reading from the DRAM port whose data reaches it in the fewest NoC0 links, and in the most. On a
  #   torus the request goes the rest of the way round.
  # - the grid's workers reading B the way a token's matmul does: as it lies, with its tiles interleaved, or by
  #   columns, where workers read chunks of a column's run (tensix.chunk_tiles) from their own banks, through the
  #   ports in Runtime.readers
  # - one 64-byte read alone, for the latency, and 64 in flight, as gather does
  def dram(size, depth, units, bank=0, split=0, own=0):  # the slowest worker's cycles for all its rounds, and GB/s
    at = rt.dram_alloc(-(-units * depth // len(bh.DRAM)) * size if not (bank or own) else units * depth * size)
    timed(reading, (at, depth, rounds, 0, size, bank, split, own), units, at=5)
    rt.dram_free(at)
    cycles = max(max(result(c, 5) for c in rt.cores[:units]), 1)
    return cycles, size * depth * rounds * units / cycles * hz / 1e9
  for size in (2048, 8192, 16384):
    rows += [(f"DRAM reads of {size >> 10} KiB, {d} in flight: all workers", dram(size, d, workers)[1], bh.DRAM_GBPS, "GB/s")
             for d in (1, 2, 4, 8, 16)]
  for name, size in (("bfp8", 1088), ("bfp4", 576)):
    rows += [(f"DRAM reads of {name} tiles in two requests, {d} in flight: all workers", dram(size, d, workers, split=64)[1], bh.DRAM_GBPS, "GB/s")
             for d in (4, 8)]
  ports = [(x, y + p) for x, y in bh.DRAM for p in range(3)]
  near, far = (bh.noc_xy(*f(ports, key=lambda xy: bh.hops(chip.raw(*xy), chip.raw(*first)))) for f in (min, max))
  for size in (2048, 16384):
    rows += [(f"DRAM reads of {size >> 10} KiB, 16 in flight: one worker, its {which} DRAM port", dram(size, 16, 1, bank)[1], None, "GB/s")
             for which, bank in (("nearest", near), ("farthest", far))]
  # the grid's workers, reading B as it lies, the way a token's matmul does
  for name, size, split in (("bf16", 2048, 0), ("bfp8", 1088, 64)):
    rows.append((f"DRAM reads of {name} tiles, {DEPTH} in flight: grid workers, interleaved", dram(size, DEPTH, grid, split=split)[1],
                 bh.DRAM_GBPS, "GB/s"))
  # and B by columns: chunks of a run from the worker's own bank, with a block float's exponents in a separate read
  for name in ("bf16", "bfp8", "bfp4"):
    n = chunk_tiles(None if name == "bf16" else name)
    rows.append((f"DRAM reads of {name} chunks, {CHUNKS} in flight: grid workers, own banks",
                 dram(n * tile_bytes(name), CHUNKS, grid, split=0 if name == "bf16" else 64 * n, own=1)[1], bh.DRAM_GBPS, "GB/s"))
  rows += [("DRAM read of 64 bytes, alone: one worker", dram(64, 1, 1)[0] / rounds, None, "cycles"),
           ("DRAM reads of 64 bytes, 64 in flight: all workers", dram(64, 64, workers)[1], bh.DRAM_GBPS, "GB/s")]
  # NoC: a word there and back, between neighbours and across the grid
  for name, j in (("neighbours", 1), ("far apart", workers - 1)):
    cycles = timed(k["ping"], (0, j, bh.noc_xy(*first), bh.noc_xy(*rt.cores[j]), rounds, rt.launches << 16), workers, at=8)
    rows.append((f"NoC round trip, {name}", cycles / rounds, None, "cycles"))
  # PCIe: upload into DRAM through the BAR, and read back by the chip writing into host memory
  start = time.perf_counter()
  t = Tensor.frombytes(rt, upload // 4096, 2048, bytes(upload))
  rows.append(("upload over PCIe", upload / (time.perf_counter() - start) / 1e9, None, "GB/s"))
  start = time.perf_counter()
  t.tobytes()
  rows.append(("read back over PCIe", upload / (time.perf_counter() - start) / 1e9, None, "GB/s"))
  # Launch cost by the size of the rectangle. The launches above used all the workers, and these use one worker, then
  # a row. A launch is one multicast and one count for any rectangle.
  for end in (chip.tensix[1], chip.tensix[len(chip.tensix) // len(bh.TENSIX_Y) - 1]):
    small = Runtime(chip, end=end)  # takes these cores over from rt, which is done
    since = small.launches
    small.launch(k["empty"])
    small.wait()
    workers = f"{len(small.cores)} worker{'s' * (len(small.cores) > 1)}"
    rows.append((f"launch on {workers}: on the chip", small.profile(since)[0][2], None, "cycles"))
  return rows

if __name__ == "__main__":
  chip = Chip(pci.open_device())
  sim = isinstance(chip.pci, pci.Sim)
  print("ttsim: probes completed; timings do not predict card performance" if sim else f"{chip.pci.sys}: the card")
  rows = probe(chip, upload=(int(sys.argv[1]) if len(sys.argv) > 1 else 8) << 20)
  for what, value, against, unit in rows: print(f"  {what:72s} {value:12.2f} {unit:9s}" + (f" (bound: {against})" if against is not None else ""))
  if len(sys.argv) > 2: json.dump({what: value for what, value, _, _ in rows}, open(sys.argv[2], "w"), indent=1)
