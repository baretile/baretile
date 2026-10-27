# Time a matmul with B stored by columns, as model weights are. Report throughput and the bytes read
# relative to reading each input once. Byte counts need a card or ttsim-chaos. On ttsim, timings reflect
# RISC-V work; Tensix instructions and DRAM reads take no time. Use a card for performance measurements.
#   TTSIM=/path/to/libttsim.so python3 -m baretile.bench [M K N [fidelity phases [bf16|bfp8|bfp4]]]
import sys
from baretile import bh, pci, tensor
from baretile.chip import Chip
from baretile.tensix import tile_bytes
from baretile.tensor import Tensor
from baretile.runtime import Runtime

def bench(chip, m, k, n, weights="bf16", repeat=3):
  """Time (m x k) @ (k x n), with B stored by columns in weights format. Return (seconds, TFLOPS, input GB/s,
  read ratio, peak TFLOPS). Input GB/s assumes each input is read once. The read ratio compares counted
  bytes with that minimum, or is None without NIU counters. Peak TFLOPS uses the grid workers and reported
  AI clock, excluding the dispatcher and workers outside the grid. Inputs are 1.0 so the result is easy to
  check."""
  rt = Runtime(chip)
  a, b = (Tensor.frombytes(rt, r, c, b"\x80\x3f" * (r * c)) for r, c in ((m, k), (k, n)))  # bf16 1.0
  b = b.cast(weights).realize().columns()
  assert (a @ b).tolist()[0][0] == k  # compiles it, and checks the answer
  since, start = rt.launches, chip.pci.now()
  for _ in range(repeat): (a @ b).realize()  # queued back to back
  rt.wait()
  seconds, read = (chip.pci.now() - start) / repeat, sum(n for *_, n in rt.profile(since)) / repeat if rt.counts else None
  once = -(-m // 32) * -(-k // 32) * tile_bytes("bf16") + -(-k // 32) * -(-n // 32) * tile_bytes(weights)  # each input's tiles
  peak = len(rt.grid[0]) * len(rt.grid[1]) * bh.MVMUL_FLOPS * chip.aiclk * 1e6 / tensor.FIDELITY / 1e12  # a product takes FIDELITY phases
  return seconds, 2 * m * k * n / seconds / 1e12, once / seconds / 1e9, read and read / once, peak

if __name__ == "__main__":
  chip = Chip(pci.open_device())
  m, k, n = map(int, sys.argv[1:4]) if len(sys.argv) > 3 else (512, 768, 3072)
  if len(sys.argv) > 4: tensor.FIDELITY = int(sys.argv[4])
  weights = sys.argv[5] if len(sys.argv) > 5 else "bf16"
  seconds, tflops, gbs, reread, peak = bench(chip, m, k, n, weights)
  speed = (f"{seconds * bh.AICLK / 1e3:.1f}k ttsim cycles (simulator timing; use a card to measure performance)"
           if isinstance(chip.pci, pci.Sim) else
           f"{seconds * 1e3:.3f} ms, {tflops:.2f} TFLOPS of {peak:.0f} peak at {tensor.FIDELITY} fidelity phases, {gbs:.0f} GB/s")
  print(f"{m}x{k} @ {k}x{n} ({weights} B): {speed}" + (f"; DRAM reads {reread:.2f}x the inputs" if reread is not None else ""))
