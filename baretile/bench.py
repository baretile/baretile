# Times a bf16 matmul across the workers. B is the weights, in bf16 or a block float, laid out by columns the way the
# models upload them. On a card the time is wall time. On ttsim it's simulated cycles of the RISC-V work on the
# critical path, which isn't silicon's speed: queued Tensix instructions cost ttsim nothing, and its DRAM reads take no
# time. bench also counts the bytes that the reads brought in, where the NIUs count them (a card or ttsim-chaos), and
# compares them with reading each input once. On silicon a big matmul waits on DRAM more than on its MVMULs, and decode
# (M of 1) is bound by DRAM, so on a card bench also reports the GB/s of reading each input once.
#   TTSIM=/path/to/libttsim.so python3 -m baretile.bench [M K N [fidelity phases [bf16|bfp8|bfp4]]]
import sys
from baretile import bh, pci, tensor
from baretile.chip import Chip
from baretile.tensix import tile_bytes
from baretile.tensor import Tensor
from baretile.runtime import Runtime

def bench(chip, m, k, n, weights="bf16", repeat=3):
  """Time (m x k) @ (k x n) with B in the dtype `weights`, laid out by columns. Returns seconds, TFLOPS, the GB/s of
  reading each input once, and how many times over the inputs were read, which is None where the NIUs don't count
  bytes. The inputs are 1.0, so the result is easy to check."""
  rt = Runtime(chip)
  a, b = (Tensor.frombytes(rt, r, c, b"\x80\x3f" * (r * c)) for r, c in ((m, k), (k, n)))  # bf16 1.0
  b = b.cast(weights).realize().columns()
  assert (a @ b).tolist()[0][0] == k  # compiles it, and checks the answer
  since, start = rt.launches, chip.pci.now()
  for _ in range(repeat): (a @ b).realize()  # queued back to back
  rt.wait()
  seconds, read = (chip.pci.now() - start) / repeat, sum(n for *_, n in rt.profile(since)) / repeat if rt.counts else None
  once = -(-m // 32) * -(-k // 32) * tile_bytes("bf16") + -(-k // 32) * -(-n // 32) * tile_bytes(weights)  # each input's tiles
  return seconds, 2 * m * k * n / seconds / 1e12, once / seconds / 1e9, read and read / once

if __name__ == "__main__":
  chip = Chip(pci.open_device())
  m, k, n = map(int, sys.argv[1:4]) if len(sys.argv) > 3 else (512, 768, 3072)
  if len(sys.argv) > 4: tensor.FIDELITY = int(sys.argv[4])
  weights = sys.argv[5] if len(sys.argv) > 5 else "bf16"
  seconds, tflops, gbs, reread = bench(chip, m, k, n, weights)
  peak = len(chip.tensix) * bh.MVMUL_FLOPS * bh.AICLK / tensor.FIDELITY / 1e12  # a product takes FIDELITY phases
  speed = (f"{seconds * bh.AICLK / 1e3:.1f}k ttsim cycles (RISC-V work on the critical path, not silicon time)"
           if isinstance(chip.pci, pci.Sim) else
           f"{seconds * 1e3:.3f} ms, {tflops:.2f} TFLOPS of {peak:.0f} peak at {tensor.FIDELITY} fidelity phases, {gbs:.0f} GB/s")
  print(f"{m}x{k} @ {k}x{n} ({weights} B): {speed}" + (f"; DRAM reads {reread:.2f}x the inputs" if reread is not None else ""))
