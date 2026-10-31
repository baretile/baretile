# Tests for Tensix code generation, kernels and Tensor (layers 2 and 3). See conftest.py for simulator setup.
import gc, json, math, os, pathlib, random, re, struct, subprocess, sys, pytest
import numpy as np
from baretile import bh, pci, tensix, tensor, viz
from baretile.kernels import grid_parts
from baretile.tensor import Tensor
from baretile.runtime import TIMES, Runtime, compile_kernel
from conftest import bf16, hold, value

def kernels(rt, since):  # counts launches after launch `since`, leaving out the transfers of tobytes and frombytes
  return sum(rt.sent[n % TIMES][0] not in ("readback", "upload") for n in range(since + 1, rt.launches + 1))

def row_end(chip):  # where a Runtime ends so that its grid is one row of 12, however the chip is harvested
  return chip.tensix[len(chip.tensix) // len(bh.TENSIX_Y) + 11]

def page_data(path):  # what viz.save put in a page
  return json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', path.read_text(), re.S)[1])

def close(t, want, tol):  # checks t against numpy's answer, with tol as both the relative and absolute tolerance
  got = np.array(t.tolist())
  assert got.shape == want.shape and np.allclose(got, want, rtol=tol, atol=tol), f"off by up to {np.abs(got - want).max()}"

def bf(v): return float(bf16(v))  # rounds to bf16 the way uploads do

def packed(v): return float(bf16(v, True))  # rounds to bf16 the way the packer does, half up

def gelu(a): return 0.5 * a * (1 + np.tanh(math.sqrt(2 / math.pi) * (a + 0.044715 * a**3)))  # GPT-2's, in numpy

def conv(x, w, padding=0):  # numpy convolution, a tap at a time. x is (channels, height, width), w is (out, in, k * k)
  k = math.isqrt(w.shape[2])
  x = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))
  ho, wo = x.shape[1] - k + 1, x.shape[2] - k + 1
  return sum(np.einsum("oc,chw->ohw", w[:, :, ky * k + kx], x[:, ky:ky + ho, kx:kx + wo]) for ky in range(k) for kx in range(k))

def test_probe(chip):
  # The 50 probes run and report positive numbers. The sizes are small, and on ttsim the numbers mean nothing for
  # silicon. The NIU read counters should count at least as many reads as the DRAM probes say they made.
  from baretile import probe
  reads = lambda: sum(chip.read32(x, y, bh.NIU_READS) for x, y in chip.tensix)
  before = reads()
  rows = probe.probe(chip, n=4, rounds=2, upload=1 << 16)
  names = [what for what, *_ in rows]
  assert len(names) == len(set(names)) == 50 and all(value > 0 for _, value, *_ in rows), rows
  workers = len(chip.tensix) - 1
  assert reads() - before >= 2 * workers * 3 * (1 + 2 + 4 + 8 + 16), "fewer reads than the DRAM probes made, at three sizes on the workers"

def test_bench(chip):
  # bench runs and checks its result. On a card or ttsim-chaos, where the NIUs count bytes, a token's row and a grid's
  # blocks read each input once. B is laid out by columns, like a model's weights.
  from baretile import bench
  for m, k, n, weights in ((1, 1024, 4096, "bfp8"), (320, 1024, 416, "bf16")):
    *_, reread, peak = bench.bench(chip, m, k, n, weights, repeat=1)
    assert reread == (1 if chip.pci.counts else None), (m, k, n, weights, reread)
  # The peak counts the grid's workers, where matmuls run, at the clock the card reported. The dispatcher doesn't
  # multiply, and neither do the workers off the grid.
  grid, core = Runtime(chip).grid, bh.MVMUL_FLOPS * chip.aiclk * 1e6 / tensor.FIDELITY / 1e12
  assert round(peak / core) == len(grid[0]) * len(grid[1]) < len(chip.tensix) - 1, (peak, core, grid)

def test_profile(chip):
  # Launches write their start and last report times to host memory, on the dispatcher's clock, and Runtime.profile
  # sums them by name. Three kernels queued back to back four times take no longer than the host saw. On ttsim the
  # host's clock is the chip's, and they take most of it; the rest is the gaps between launches. On a card the host's
  # clock is its own, and Python's time between launches counts too.
  rt = Runtime(chip)
  x, y = (Tensor.of(rt, [[float((r + c * k) % 7) for c in range(256)] for r in range(256)]) for k in (1, 2))
  run = lambda: ((x + y).realize(), (x @ y).realize(), x.softmax().realize())
  run(), rt.wait()  # compile them first
  since, start = rt.launches, chip.pci.now()
  for _ in range(4): run()
  sim = isinstance(chip.pci, pci.Sim)
  rows, elapsed = rt.profile(since), (chip.pci.now() - start) * (bh.AICLK if sim else chip.aiclk * 1e6)
  assert sorted((name, n) for name, n, *_ in rows) == [("add", 4), ("matmul", 4), ("max2 sub exp sum2 recip mul", 4)], rows
  cycles = sum(c for _, _, c, _ in rows)
  assert all(c > 0 for _, _, c, _ in rows) and cycles <= elapsed and (0.5 * elapsed < cycles or not sim), (cycles, elapsed)

def test_bytes(chip):
  # On a card or ttsim-chaos, where the NIUs count read data, Runtime.profile has the bytes that each launch read, in
  # 64-byte flits. An add reads the tiles of its two inputs. A token's bfp4 matmul reads A's row once, shared by
  # multicast, and each tile of B in two reads, exponents and values. For each result tile, a gather reads the tile of
  # its ids, the zero tile, and two runs for each row, in 64-byte blocks. Its copy of the zero tile is a read within L1.
  if not chip.pci.counts: pytest.skip("needs ttsim-chaos's libttsim.so: released ttsim doesn't count read data")
  rt, rng = Runtime(chip), np.random.default_rng(0)
  x, y = (Tensor.of(rt, rng.normal(0, 1, (64, 64)).tolist()) for _ in range(2))
  a, b = Tensor.of(rt, rng.normal(0, 1, (32, 128)).tolist()), Tensor.of(rt, rng.normal(0, 1, (128, 64)).tolist()).cast("bfp4").realize()
  table, ids = Tensor.of(rt, rng.normal(0, 1, (64, 64)).tolist()), Tensor.of(rt, [[3, 40, 7]], "int32")
  for run, want in ((lambda: (x + y).realize(), 8 * 2048), (lambda: (a @ b).realize(), 4 * 2048 + 8 * (64 + 512)),
                    (lambda: table.gather(ids), 2 * (4096 + 2048 + 3 * 2 * 64))):
    run(), rt.wait()  # compile it first
    since = rt.launches
    run()
    assert [r[3] for r in rt.profile(since)] == [want]

def test_gather_shared_indices(chip):
  # A worker keeps its index tile across the result tiles that share it, and zeros only partial tiles. A block float
  # can't be gathered, and the refusal comes before any allocation or launch.
  rt = Runtime(chip, end=chip.tensix[1])
  rows = np.arange(65 * 96).reshape(65, 96) % 7
  selected = list(range(34, -1, -1))
  x = Tensor.of(rt, rows.tolist())
  ids = Tensor.of(rt, [selected], "int32")
  since = rt.launches
  gathered = x.gather(ids)
  profile = rt.profile(since)
  assert np.array_equal(value(gathered)[0], rows[selected])
  if chip.pci.counts:
    # Two index tiles, zero padding for three partial output tiles, and two 64-byte runs per selected row and tile column.
    assert [r[3] for r in profile] == [2 * 4096 + 3 * 2048 + 35 * 3 * 2 * 64]
  for dtype in ("bfp8", "bfp4"):
    packed = x.cast(dtype).realize()
    before = dict(rt.used), rt.launches
    with pytest.raises(ValueError, match="block-float"): packed.gather([0])
    assert (rt.used, rt.launches) == before

def test_viz(chip, tmp_path):
  # viz.save records completed launches and compiler details. A shared subexpression gets one graph node,
  # even when a reduction recomputes it on several passes. Matmul's B node describes its read mode.
  # A custom kernel absent from tensor.kernels has launch statistics but no expression graph. Each kernel has its C programs.
  rt = Runtime(chip)
  x = Tensor.of(rt, [[float(c % 5) for c in range(64)] for _ in range(32)])
  since = rt.launches
  x.softmax().realize(), (x @ x.T).realize()
  rt.launch(compile_kernel("void kmain(volatile unsigned *a) {}"), name="empty")
  rt.wait()
  viz.save(tmp_path / "viz.html", rt, since=since)
  [run] = page_data(tmp_path / "viz.html")
  assert (tmp_path / "viz.html").read_text().count('"title"') == 1, "the page holds its data once"
  assert [(n, run["kernels"][k]["name"]) for k, n, *_ in run["launches"]] == \
    [(since + 1, "max2 sub exp sum2 recip mul"), (since + 2, "T matmul"), (since + 3, "empty")]
  assert all(cycles > 0 and workers > 0 for *_, cycles, _, workers in run["launches"])
  softmax, matmul, empty = run["kernels"]
  assert softmax["graph"] == [["in 0", "1x32x64", []], ["max2", "", [0]], ["sub", "", [0, 1]], ["exp", "", [2]], ["sum2", "", [3]],
                              ["recip", "", [4]], ["mul", "", [3, 5]]]
  assert matmul["graph"] == [["in 0", "1x32x64", []], ["read B", "transposed", [0]], ["matmul", "", [0, 1]]]
  assert "graph" not in empty and "text" not in empty and empty["launches"] == 1
  assert [core for core, _ in softmax["programs"]] == ["brisc", "trisc0", "trisc1", "trisc2"]
  assert empty["programs"] == [["brisc", "void kmain(volatile unsigned *a) {}"]], "each core's C, as compile_kernel was given it"

@pytest.mark.skipif(pci.simulator() is None, reason="the script opens the device again, and a card can't be open in two processes")
def test_viz_script(tmp_path):
  # The command runs a script, or with -m a module, and writes one viz.html containing the runtimes still open at the
  # end. The second runtime replaces the first, whose records went with it.
  (tmp_path / "script.py").write_text("from baretile import pci\nfrom baretile.chip import Chip\nfrom baretile.runtime import Runtime\n"
                                      "from baretile.tensor import Tensor\nchip = Chip(pci.open_device())\nRuntime(chip)\nrt = Runtime(chip)\n"
                                      "print((Tensor.of(rt, [[1.0, 2.0]]) * 3).tolist())\n")


def test_jit(chip, monkeypatch):
  # tensor.jit records the second call of a function and replays its launches after that. Later calls send the same
  # launches on their inputs where they are, without building anything or realizing another expression. The function
  # is a block of matmuls, a softmax and a layer norm. The DRAM that the recording took stays pinned for the replays.
  rt = Runtime(chip)
  rng = np.random.default_rng(7)
  w, v = (Tensor.of(rt, bf16(rng.normal(0, 0.1, (96, 96))).tolist()) for _ in range(2))
  f = lambda x: ((x @ w).relu() @ v).softmax().layernorm() * 2.0
  step, realize, realized = tensor.jit(f), Tensor.realize, []
  for call in range(4):
    x = Tensor.of(rt, bf16(rng.normal(0, 1, (64, 96))).tolist())
    if call == 3: monkeypatch.setattr(Tensor, "realize", lambda t, into=None: (realized.append(t), realize(t, into))[1])
    y = step(x)
    monkeypatch.setattr(Tensor, "realize", realize)
    assert y.tobytes() == f(x).tobytes(), call
  assert not realized and rt.pinned, realized

def test_jit_input_aliases(chip):
  rt = Runtime(chip)
  a, b = (Tensor.full(rt, 1, 1, v, "fp32") for v in (2, 3))
  step = tensor.jit(lambda x, y: x * 10 + y)
  for xs, want in (((a, b), 23), ((a, b), 23), ((b, a), 32), ((a, b), 23)):
    assert step(*xs).tolist() == [[want]]
    assert (a.tolist(), b.tolist()) == ([[2]], [[3]])
  repeated = tensor.jit(lambda x, y: x * 10 + y)
  repeated(a, a), repeated(a, a)
  assert repeated(a, b).tolist() == [[23]]
  view = a.repeat_interleave(2)
  varied = tensor.jit(lambda x: x + 1)
  varied(view), varied(view)
  assert varied(Tensor.of(rt, [[[4]], [[5]]], "fp32")).tolist() == [[[5]], [[6]]]
  exposed = []
  def capture(x, y):
    exposed[:] = [x, y]
    return x * 10 + y
  aliased = tensor.jit(capture)
  aliased(a, b), aliased(a, b)
  assert aliased(exposed[1], exposed[0]).tolist() == [[32]], "captured inputs can be exposed by user code"
  # An input that f only reads is copied when it shares storage with one that the replay writes
  x, ids = Tensor.of(rt, np.arange(512 * 32, dtype=np.float32).reshape(512, 32).tolist(), "fp32"), np.random.default_rng(3).permutation(512)
  write = tensor.jit(lambda c, at, row: c.put(at, row))
  for _ in range(2): write(Tensor.full(rt, 512, 32, 0, "fp32"), Tensor.of(rt, [ids.tolist()], "int32"), Tensor.full(rt, 512, 32, 1, "fp32"))
  want = value(x).reshape(512, 32).copy()
  want[ids] = want.copy()
  write(x, Tensor.of(rt, [ids.tolist()], "int32"), x)
  assert np.array_equal(value(x).reshape(512, 32), want)

def test_jit_writes_inputs(chip):
  # Replays read and write the inputs where they are, so an input that the function writes in place ends as a normal
  # call would leave it, on every call, also when it is passed twice. A slice is copied in and back out. A tensor by
  # columns is read in place too, so any with the recorded layout will do.
  rt = Runtime(chip)
  x, y = Tensor.full(rt, 1, 1, 0, "fp32"), Tensor.full(rt, 1, 1, 10, "fp32")
  step = tensor.jit(lambda t: t.assign(t + 1))
  assert [step(x).tolist() for _ in range(4)] == [[[1]], [[2]], [[3]], [[4]]] and x.tolist() == [[4]]
  assert step(y).tolist() == [[11]] and (x.tolist(), y.tolist()) == ([[4]], [[11]])
  rt.wait()
  launches = rt.launches
  step(x)
  assert rt.launches == launches + 1, "a replay on a tensor of its own sends just the recorded launch"
  add = tensor.jit(lambda t, u: t.assign(t + u))
  add(x, y), add(x, y)
  assert add(x, x).tolist() == [[54]] and add(x, y).tolist() == [[65]] and (x.tolist(), y.tolist()) == ([[65]], [[11]])
  rows = Tensor.full(rt, 64, 32, 0, "fp32")
  bump = tensor.jit(lambda t: t.assign(t + 1))
  for _ in range(3): bump(rows[32:64, :])
  assert value(rows).reshape(64, 32)[[0, 32], 0].tolist() == [0, 3]
  cache = Tensor.full(rt, 64, 32, 0, "fp32")
  write = tensor.jit(lambda c, ids, row: c.put(ids, row))
  for i in range(4): write(cache, Tensor.of(rt, [[i]], "int32"), Tensor.full(rt, 1, 32, i + 1, "fp32"))
  assert value(cache).reshape(64, 32)[:5, 0].tolist() == [1, 2, 3, 4, 0]
  a = Tensor.full(rt, 1, 64, 1)
  w = Tensor.full(rt, 64, 64, 1).columns()
  times = tensor.jit(lambda w: a @ w)
  assert [times(w).tolist()[0][0] for _ in range(3)] == [64] * 3
  assert times(Tensor.full(rt, 64, 64, 2).columns()).tolist()[0][0] == 128

def test_jit_input_grads(chip):
  # An input's .grad is the gradient of the latest call, also on a replay, where backward ran on the recording's
  # stand-in for the input. A gradient that is another input is that input on this call.
  rt = Runtime(chip)
  w = Tensor.full(rt, 1, 1, 3, "fp32")
  square = tensor.jit(lambda x: (x * x * w).sum().backward())
  passed = tensor.jit(lambda x, gy: (x + w).backward(gy))
  for v in (1, 2, 5, 7):
    x = Tensor.full(rt, 1, 1, v, "fp32").requires_grad_()
    square(x)
    assert x.grad.tolist() == [[6 * v]]
    x.grad = None
    passed(x, Tensor.full(rt, 1, 1, v, "fp32"))
    assert x.grad.tolist() == [[v]]

def test_recording_lifetime_and_reuse(chip):
  rt = Runtime(chip, end=chip.tensix[1])
  x = Tensor.full(rt, 1, 1, 2, "fp32")
  step = tensor.jit(lambda t: t + 1)
  step(x).tolist()
  baseline = dict(rt.used)  # x and the runtime's cached scalar 1
  for _ in range(3):
    step = tensor.jit(lambda t: t + 1)
    step(x), step(x), step(x)
    del step
    gc.collect()
    rt.wait()
    assert not rt.pinned and not rt.pinned_freed and rt.used == baseline
  first, second = tensor.Recording(rt), tensor.Recording(rt)
  with first: out = (x + 1).realize()
  with second: other = (out * 2).realize()
  addr = out.addr
  del out
  first.close()
  rt.wait()
  assert addr in rt.used and addr in rt.pinned, "another recording still reads this allocation"
  second.send()
  assert other.tolist() == [[6]]
  second.close()
  rt.wait()
  assert addr not in rt.used and other.tolist() == [[6]], "live output storage survives closing a recording"
  with first: out = (x + 1).realize()
  first.send()
  assert out.tolist() == [[3]]
  with first:  # a launch that fails its checks isn't recorded, so the recording still replays
    out = (x * 2).realize()
    with pytest.raises(ValueError, match="arguments"): rt.launch(bytes(16), (-1,))
  first.send()
  assert out.tolist() == [[4]]
  go = hold(rt)
  first.send()
  addr = out.addr
  del out
  first.close()
  assert addr in rt.used, "queued replay keeps storage after the recording is closed"
  go()
  rt.wait()
  assert addr not in rt.used and not rt.pinned and not rt.pinned_freed
  with pytest.raises(RuntimeError, match="not ready"): first.send()

def test_jit_formats(chip):
  # Copy replay inputs in their storage format. Adding zero cannot copy block floats and changes negative zero.
  rt = Runtime(chip)
  same = tensor.jit(lambda x: x)
  for n in (1, 2, 3, 4):
    raw = struct.pack("<2f", n, -0.0)
    assert same(Tensor.frombytes(rt, 1, 2, raw, dtype="fp32")).tobytes() == raw
  a = Tensor.full(rt, 32, 32, 1)
  for dtype in ("bfp8", "bfp4"):
    step = tensor.jit(lambda x: a @ x)
    for n in (1, 2, 3, 4):
      x = Tensor.full(rt, 32, 32, n).cast(dtype).realize()
      assert step(x).tolist() == [[32 * n] * 32] * 32

def test_jit_returned_views(chip):
  # A returned view of an input is on each call's input, keeps a repeat's rep, and has a plain address, so another
  # jit that captures it doesn't take it for one of its own inputs.
  rt = Runtime(chip)
  repeat = tensor.jit(lambda x: x.repeat_interleave(2))
  for n in range(1, 5):
    x = Tensor.of(rt, [[[float(n)] * 32] * 32, [[float(n + 1)] * 32] * 32])
    assert np.array_equal(value(repeat(x)), np.repeat(value(x), 2, axis=0))
  x = Tensor.of(rt, [[[4.0] * 32] * 32, [[5.0] * 32] * 32])
  pick = tensor.jit(lambda x: x[1])
  pick(x)
  captured = pick(x)  # returned by the recording call
  add = tensor.jit(lambda y: y + captured)
  for n in (10, 20, 30, 40):
    assert add(Tensor.full(rt, 32, 32, n)).tolist() == [[n + 5] * 32] * 32

def test_repeat_copies_preserve_storage(chip):
  # Assigning a repeat copies its bits, -0 and block floats included, instead of computing x + 0.
  rt = Runtime(chip)
  raw = struct.pack("<2f", -0.0, 3.0)
  x = Tensor.frombytes(rt, 1, 2, raw, dtype="fp32").repeat_interleave(3)
  out = Tensor(rt, 1, 2, batch=3, dtype="fp32").assign(x)
  assert out.tobytes() == raw * 3
  for dtype in ("bfp8", "bfp4"):
    x = Tensor.full(rt, 32, 32, 1.5).cast(dtype).realize().repeat_interleave(3)
    out = Tensor(rt, 32, 32, batch=3, dtype=dtype).assign(x)
    assert out.tobytes() == x.tobytes()

def test_upload(chip, monkeypatch):
  # frombytes writes through the BAR below tensor.DMA_UPLOAD bytes, and through upload_kernel from there, in chunks
  # staged in Runtime.back. Both leave the same tiles in every bank, also when the tiles don't fill the banks evenly.
  # 1100 x 1024 takes several chunks. A recording leaves the upload's launches out: they'd read whatever the host
  # staged since, and the uploaded tensor keeps its values.
  rt, rng = Runtime(chip), np.random.default_rng(11)
  def banks(t):  # the tiles of t in each bank, read through the BAR
    tb = tensix.tile_bytes(t.dtype)
    return [chip.read(*bh.DRAM[b], t.addr, len(range(b, t.tiles, len(bh.DRAM))) * tb) for b in range(len(bh.DRAM))]
  for shape, dtype, chunks in (((1, 1), "bf16", 1), ((3, 40, 40), "int32", 1), ((70, 100), "fp32", 1), ((1100, 1024), "bf16", 5)):
    a = rng.integers(-2**31, 2**31, shape, np.int32) if dtype == "int32" else rng.standard_normal(shape).astype(np.float32)
    data, made = (a.view(np.uint32) >> 16).astype(np.uint16).tobytes() if dtype == "bf16" else a.tobytes(), []
    for dma in (False, True):
      monkeypatch.setattr(tensor, "DMA_UPLOAD", 0 if dma else 1 << 62)
      since = rt.launches
      made.append(Tensor.frombytes(rt, *shape[-2:], data, shape[0] if len(shape) == 3 else 1, dtype))
      assert rt.launches - since == dma * chunks and kernels(rt, since) == 0, (shape, dma)
    assert banks(made[0]) == banks(made[1]) and made[1].tobytes() == data, (shape, dtype)
  monkeypatch.setattr(tensor, "DMA_UPLOAD", 0)  # the recording below must exercise host staging
  ones = struct.pack("<1024H", *[0x3F80] * 1024)  # 1.0 in bf16
  step, x = tensor.jit(lambda x: x + Tensor.frombytes(rt, 32, 32, ones)), Tensor.of(rt, [[1.0] * 32] * 32)
  for call in range(4):
    Tensor.frombytes(rt, 32, 32, bytes(2048))  # stages zeros where the call before staged ones
    assert step(x).tolist() == [[2.0] * 32] * 32, call

def test_zeros(chip):
  # Tensor.zeros writes every byte of a tensor's storage from the zero tile in the workers' L1, in one launch and
  # with nothing over PCIe. All-zero bytes are zeros in each format, block floats included, so it's the same storage
  # as casting uploaded zeros. It works on freed memory that still holds another tensor's values.
  rt = Runtime(chip)
  for dtype in ("bf16", "fp32", "int32", "bfp8", "bfp4"):
    junk = Tensor.of(rt, np.random.default_rng(3).integers(1, 9, (2, 70, 100)).tolist(), "int32")
    at = junk.addr
    del junk
    rt.wait()
    since, z = rt.launches, Tensor.zeros(rt, 70, 100, 2, dtype)
    assert z.addr == at, dtype  # where the integers were
    cast = Tensor.frombytes(rt, 70, 100, bytes(2 * 2 * 70 * 100), 2).cast(dtype).realize()
    rt.wait()
    each = -(-z.tiles // len(bh.DRAM)) * tensix.tile_bytes(dtype)
    raw = lambda t: [chip.read(*bh.DRAM[b], t.addr, each) for b in range(len(bh.DRAM))]
    assert rt.sent[(since + 1) % TIMES][0] == "zeros" and raw(z) == raw(cast) == [bytes(each)] * len(bh.DRAM), dtype
    assert not np.array(z.tolist()).any(), dtype

def test_checks_without_asserts():
  # A bad call raises an exception, so it still stops under python -O, which drops assert statements. The checks run
  # before the tensors touch DRAM, so a stand-in runtime that hands out address 0 is enough, and no chip is needed.
  code = """
from baretile.tensor import Tensor
class Runtime:
  def check(self): pass
  def dram_alloc(self, size): return 0
  def dram_free(self, addr): pass
rt = Runtime()
x, y = Tensor(rt, 2, 3), Tensor(rt, 2, 3)
for bad in (lambda: x @ y, lambda: x.reshape(5, 5), lambda: x.permute(0, 0, 1), lambda: x + Tensor(rt, 3, 2),
            lambda: Tensor.frombytes(rt, 1, 1, bytes(2), dtype="bfp8"), lambda: x.cast("int32") @ x.T.cast("int32")):
  try: bad()
  except ValueError: continue
  raise SystemExit("no ValueError")
print("all raised")
"""
  run = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True, text=True, timeout=60, cwd=pathlib.Path(__file__).parent.parent)
  assert run.returncode == 0 and run.stdout.strip() == "all raised", run.stdout + run.stderr

def test_tensor_inputs(chip):
  # Reject malformed inputs before allocating DRAM. Ragged rows can have the right total byte count.
  rt = Runtime(chip, end=chip.tensix[1])
  used = dict(rt.used)
  for data in ([], [[]], [[1, 2], [3], [4, 5, 6]], [[[1]], [[2], [3]]]):
    with pytest.raises(ValueError, match="nonempty|rectangular"): Tensor.of(rt, data)
  for rows, cols, batch in ((0, 1, 1), (1, -1, 1), (1, 1, 0)):
    with pytest.raises(ValueError, match="positive"): Tensor(rt, rows, cols, batch=batch)
  with pytest.raises(ValueError, match="dtype"): Tensor(rt, 1, 1, dtype="float64")
  with pytest.raises(ValueError, match="bytes"): Tensor.frombytes(rt, 1, 2, bytes(2))
  assert rt.used == used
  x = Tensor.of(rt, [[[1, 2]], [[3, 4]]], "fp32")
  assert x[-1].tolist() == [[3, 4]]
  for index in (-3, 2):
    with pytest.raises(IndexError, match="batch"): x[index]
  for shape in ((), (1, 1, 1, 1, 2, 2)):
    with pytest.raises(ValueError, match="dimensions"): x.reshape(*shape)
    with pytest.raises(ValueError, match="dimensions"): x.expand(*shape)
  for axis in (-4, 3):
    with pytest.raises(ValueError, match="axis"): x.sum(axis)
    with pytest.raises(ValueError, match="axis"): x.cat(x, axis=axis)
  for k, parts in ((0, 1), (-1, 1), (1, 0)):
    with pytest.raises(ValueError, match="positive"): x.topk(k, parts)
  for op in (lambda: x.sum(1.5), lambda: x.cat(x, axis=1.5), lambda: x.split(1.5), lambda: x.topk(1.5)):
    with pytest.raises(TypeError): op()
  # realize writes into a plain destination only from an unrealized tensor, and a block float moves only as a matmul's
  # B. merge needs whole tiles, and a matmul by index has no gradient.
  b, w = Tensor.of(rt, [[1.0] * 32] * 32), Tensor.zeros(rt, 32, 32, 2)
  with pytest.raises(ValueError, match="unrealized"): b.realize(Tensor.zeros(rt, 32, 32))
  with pytest.raises(ValueError, match="repeat"): (Tensor.zeros(rt, 32, 32, 4) + 1).realize(Tensor.zeros(rt, 32, 32).repeat_interleave(4))
  with pytest.raises(ValueError, match="block float"): b.cast("bfp8").T.realize()
  with pytest.raises(ValueError, match="tiles wide"): Tensor.zeros(rt, 2, 3, 2).merge()
  with pytest.raises(ValueError, match="block-float B"): b.matmul(w.float(), index=Tensor.of(rt, [[1]], "int32"))
  picked = b.requires_grad_().matmul(w, index=Tensor.of(rt, [[1]], "int32"))
  with pytest.raises(NotImplementedError, match="by index"): picked.sum().backward()

def test_movement_allocation_retry(chip, monkeypatch):
  # A transpose or move whose allocation fails stays unrealized, so it can be realized again once there's room.
  rt = Runtime(chip)
  source = Tensor.of(rt, [[1.0, 2.0], [3.0, 4.0]])
  allocate = rt.dram_alloc
  def full(size): raise MemoryError("test allocation limit")
  for moved, want in ((source.T, [[1, 3], [2, 4]]), (source.reshape(1, 1, 4), [[1, 2, 3, 4]])):
    monkeypatch.setattr(rt, "dram_alloc", full)
    with pytest.raises(MemoryError, match="allocation limit"): moved.realize()
    monkeypatch.setattr(rt, "dram_alloc", allocate)
    assert moved.tolist() == want

def test_move_bounds_and_tensor_truth(chip):
  rt = Runtime(chip)
  x = Tensor.of(rt, [[[1, 2]], [[3, 4]]], "fp32")
  neighbor = Tensor.of(rt, [[123, 456]], "fp32")
  before = rt.launches, dict(rt.used)
  bad = [((1, 1, 2), 16, (0, 0, 1)), ((1, 1, 2), -1, (0, 0, 1)), ((1, 1, 2), 0, (0, 1)),
         ((1, 2), 0, (0, 0, 1)), ((1, 1, 2), 2**31, (0, 0, 1)), ((1, 1, 2), 0, (0, 0, 2**31))]
  for args in bad:
    with pytest.raises(ValueError, match="move"): x.move(*args)
  for lo, hi in (((0, 0), None), ((0, 0, 0), (1, 1, 3)), ((0, 0, 2), (1, 1, 1))):
    with pytest.raises(ValueError, match="move"): x.move((1, 1, 2), 0, (0, 0, 1), lo, hi)
  assert (rt.launches, rt.used) == before
  assert neighbor.tolist() == [[123, 456]]
  assert x.move((1, 1, 4), 3, (0, 0, -1)).tolist() == [[4, 3, 2, 1]]
  assert x.move((1, 1, 4), -1, (0, 0, 1), lo=(0, 0, 1)).tolist() == [[0, 1, 2, 3]]
  assert x.move((1, 1, 2), 9, (0, 0, 1), hi=(0, 0, 0)).tolist() == [[0, 0]]
  for v in (0, -0.0, 2, -3, float("nan")):
    assert bool(Tensor.full(rt, 1, 1, v, "fp32")) == bool(v)
  a, b = Tensor.full(rt, 1, 1, 2), Tensor.full(rt, 1, 1, 3)
  assert not bool(a == b) and bool(a != b) and a not in [b]
  with pytest.raises(ValueError, match="ambiguous"): bool(x)
  with pytest.raises(ValueError, match="ambiguous"): bool(x == x)

def test_padding_crops_and_validation(chip):
  rt = Runtime(chip)
  data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
  for widths in (((0, 0), (0, 0), (-1, 0)), ((-1, 1), (1, -1), (-1, 2)), ((0, 0), (-1, 0), (0, -2))):
    x = Tensor.of(rt, data.tolist(), "fp32").requires_grad_()
    kept = tuple(slice(max(-a, 0), n - max(-b, 0)) for (a, b), n in zip(widths, data.shape))
    want = np.pad(data[kept], [(max(a, 0), max(b, 0)) for a, b in widths])
    y = x.pad(widths)
    assert np.array_equal(value(y), want)
    y.sum().backward()
    grad = np.zeros_like(data)
    grad[kept] = 1
    assert np.array_equal(value(x.grad), grad)
  for widths in (((0, 0),) * 4, ((1,),), ((1, 2, 3),), ((0.5, 1),), (1,)):
    with pytest.raises(ValueError, match="pad"): x.pad(widths)

def test_bf16_special_values(chip):
  # Host conversion must keep NaNs as NaNs, including payloads that would round to infinity or overflow 16 bits.
  bits = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7FFFFFFF, 0xFFFFFFFF, 0x7F800001, 0x3F808000, 0x3F818000]
  floats = struct.unpack(f"<{len(bits)}f", struct.pack(f"<{len(bits)}I", *bits))
  want = [0, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0x7FFF, 0xFFFF, 0x7FC0, 0x3F80, 0x3F82]
  assert list(map(tensix.to_bf16, floats)) == want
  rt = Runtime(chip, end=chip.tensix[1])
  assert Tensor.of(rt, [floats]).tobytes() == struct.pack(f"<{len(want)}H", *want)

def test_jit_inputs_and_retry(chip):
  rt = Runtime(chip, end=chip.tensix[1])
  x, y = Tensor.full(rt, 1, 1, 2), Tensor.full(rt, 1, 1, 3)
  calls = []
  def add(x, y):
    calls.append(None)
    if len(calls) == 2:
      (x + y).realize()  # an abandoned intermediate must not stay pinned after the failed recording
      raise ValueError("recording failed")
    return x + y
  step = tensor.jit(add)
  assert step(x, y).tolist() == [[5]]
  with pytest.raises(ValueError, match="recording failed"): step(x, y)
  rt.wait()
  assert rt.log is None and not rt.pinned and not rt.pinned_freed and len(rt.used) == 2
  assert step(x, y).tolist() == [[5]]  # retry records, and the following call replays
  assert step(x, y).tolist() == [[5]] and len(calls) == 3
  for xs in ((), (x,), (x, y, x)):
    with pytest.raises(ValueError, match="inputs"): step(*xs)
  # Disjoint runtimes can run together, but their tensors still cannot mix in an expression.
  other = Runtime(chip, start=chip.tensix[2], end=chip.tensix[3])
  z = Tensor.full(other, 1, 1, 7)
  with pytest.raises(ValueError, match="runtime"): step(z, z)
  with pytest.raises(ValueError, match="runtime"): x + z
  with pytest.raises(ValueError, match="runtime"): x.assign(z)
  with pytest.raises(ValueError, match="runtime"): x.gather(z.cast("int32"))
  with pytest.raises(ValueError, match="runtime"): x.put(x.cast("int32"), z)
  with pytest.raises(ValueError, match="runtime"): (x + 1).realize(z)
  with pytest.raises(ValueError, match="runtime"): x.attend(z, z, z)
  with pytest.raises(ValueError, match="runtime"): x.topk(1, labels=z.cast("int32"))
  with pytest.raises(ValueError, match="runtime"): x.cat(z)

def test_gradient_targets(chip):
  rt = Runtime(chip, end=chip.tensix[1])
  x = Tensor.of(rt, [[2, 3]], "fp32").requires_grad_()
  middle = x * x
  loss = (middle * 3).sum()
  gx, gm, gl = loss.gradient(x, middle, loss)
  assert gx.tolist() == [[12, 18]] and gm.tolist() == [[3, 3]] and gl.tolist() == [[1]]
  x.sum().backward()
  (x > 0).sum().backward()  # a nondifferentiable path leaves an existing gradient alone
  assert x.grad.tolist() == [[1, 1]]

def test_disjoint_runtime_tensors(chip):
  with Runtime(chip, end=chip.tensix[1]) as a, Runtime(chip, start=chip.tensix[2], end=chip.tensix[3]) as b:
    x, y = Tensor.of(a, [[1, 2]]), Tensor.of(b, [[7, 8]])
    assert x.tolist() == [[1, 2]] and y.tolist() == [[7, 8]]
    left, right = (x + 10).realize(), (y * 2).realize()
    assert left.tolist() == [[11, 12]] and right.tolist() == [[14, 16]]

def test_dram_in_reference_cycle(chip, no_gc):
  # A tensor in a reference cycle holds its DRAM until Python collects it, as a parameter does after backward(). An
  # allocation that finds no room collects and tries again.
  with Runtime(chip, end=chip.tensix[1]) as rt:
    t = Tensor(rt, 32, 256)  # 2 KiB in each bank
    t.cycle = t
    for _, n in list(rt.holes): rt.dram_alloc(n)  # take the rest of DRAM
    del t
    assert rt.dram_alloc(2048) is not None

def test_takeover_during_tensix_kernel(chip):
  # A runtime taken over in the middle of a Tensix kernel lets that launch finish first. A worker stopped inside a
  # Tensix program would leave its coprocessor blocked, and the next runtime's kernels would queue behind it.
  rt, rng = Runtime(chip, end=chip.tensix[1]), np.random.default_rng(1)
  q, k, v = (Tensor.of(rt, rng.normal(0, 1, (1, n, 64)).tolist()) for n in (32, 768, 768))
  places = Tensor.of(rt, [[[767]] * 32], "int32")
  rt.wait()
  q.attend(k, v, places).realize()
  for _ in range(170): chip.pci.tick()  # on ttsim, about 60% of the attention launch
  rt = Runtime(chip, end=chip.tensix[1])
  a = Tensor.of(rt, [[1.0, 2.0]])
  assert (a + a).tolist() == [[2.0, 4.0]]

def test_tensors_and_recordings_reject_closed_runtime(chip):
  rt = Runtime(chip, end=chip.tensix[1])
  x = Tensor.of(rt, [[1, 2]])
  lazy = x + 1
  with tensor.Recording(rt) as recording: (x * 2).realize()
  replacement = Runtime(chip, end=chip.tensix[1])
  for work in (x.tolist, x.realize, lazy.realize, lambda: x + 1, lambda: x.assign(x), recording.send,
               lambda: Tensor.of(rt, [[3, 4]])):
    with pytest.raises(RuntimeError, match="closed"): work()
  assert Tensor.of(replacement, [[7, 8]]).tolist() == [[7, 8]]
  replacement.close()

def test_dram_is_freed(chip):
  # DRAM is allocated first fit, 64-byte aligned, and a freed block merges with the holes on both sides. A tensor gives
  # its DRAM back once nothing refers to it, intermediates included, so a loop of launches leaves DRAM as it found it.
  # A runtime on the whole chip first closes every other one, so DRAM starts empty.
  Runtime(chip).close()
  rt = Runtime(chip, end=chip.tensix[1])
  a, b, c = (rt.dram_alloc(n) for n in (100, 4096, 1))
  assert (a, b, c) == (0, 128, 128 + 4096)
  rt.dram_free(b)
  assert rt.dram_alloc(1000) == 128 and rt.dram_alloc(4096) == 128 + 4096 + 64, "first fit"
  for addr in (128, a, c, 128 + 4096 + 64): rt.dram_free(addr)
  assert rt.holes == [(0, bh.DRAM_BANK_SIZE)] and not rt.used
  with pytest.raises(MemoryError): rt.dram_alloc(bh.DRAM_BANK_SIZE + 1)
  x = Tensor.of(rt, [[float(r - c) for c in range(64)] for r in range(64)])
  for _ in range(3):
    y = (x + x) * x
    assert y.tolist()[1][0] == 2.0
  assert len(rt.used) == 2, "x and the last y"
  del x, y
  assert rt.holes == [(0, bh.DRAM_BANK_SIZE)] and not rt.used
  # Launches are queued. DRAM freed while a queued launch may still read it isn't reused until that launch is done. A
  # launch that waits for the host keeps the multiply queued.
  x = Tensor.of(rt, [[float(r - c) for c in range(64)] for r in range(64)])
  go = hold(rt)
  y, at = (x * 3.0).realize(), x.addr
  del x
  z = Tensor.of(rt, [[1.0] * 64] * 64)
  go()
  assert z.addr != at and y.tolist()[1][0] == 3.0

def test_tile_math(chip):
  # a tile of small integers through unpack, FPU and pack. add, sub and mul must be exact
  rt = Runtime(chip, end=chip.tensix[1])
  a = [[float((r * 32 + c) % 17 - 8) for c in range(32)] for r in range(32)]
  b = [[float((r + 3 * c) % 11 - 5) for c in range(32)] for r in range(32)]
  ta, tb = Tensor.of(rt, a), Tensor.of(rt, b)
  for got, f in ((ta + tb, float.__add__), (ta - tb, float.__sub__), (ta * tb, float.__mul__)):
    assert got.tolist() == [[f(x, y) for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]

def test_tensor_math(chip):
  # Random bf16 over 8 tiles on 8 workers. mul runs the four fidelity phases into fp32 Dst, so it's the exact product
  # rounded once by the packer. add and sub align mantissas at 11 bits first, so allow a bf16 step of the larger
  # input.
  rt, rows, cols = Runtime(chip), 64, 128
  a, b = ([[bf(random.uniform(-4, 4)) for _ in range(cols)] for _ in range(rows)] for _ in range(2))
  ta, tb = Tensor.of(rt, a), Tensor.of(rt, b)
  assert (ta * tb).tolist() == [[packed(x * y) for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]
  for got, f in (((ta + tb).tolist(), float.__add__), ((ta - tb).tolist(), float.__sub__)):
    for r in range(rows):
      for c in range(cols):
        x, y = a[r][c], b[r][c]
        assert abs(got[r][c] - f(x, y)) <= max(abs(x), abs(y)) * 2**-7, (r, c, x, y, got[r][c])

def test_numbers_once(chip):
  # A number in an elementwise kernel is a tile. Workers read it once into a slot of their own, so x * 0.5 + 1.5 over
  # 768 tiles reads x plus two tiles per worker.
  rt = Runtime(chip)
  reads = lambda: sum(chip.read32(x, y, bh.NIU_READS) for x, y in rt.cores)
  a = bf16(np.random.default_rng(6).normal(0, 1, (1024, 768)))
  x = Tensor.of(rt, a.tolist())
  y, before = x * 0.5 + 1.5, reads()
  y.realize(), rt.wait()
  assert reads() - before == x.tiles + 2 * len(rt.cores), reads() - before
  close(y, a * 0.5 + 1.5, 1e-2)

def test_matmul_exact(chip):
  # Small integers with K = 2 tiles, so Dst accumulates across them. Then A @ I == A bit for bit, which pins the tile
  # layout.
  rt = Runtime(chip)
  a = [[float((r * 7 + c * 3) % 3 - 1) for c in range(64)] for r in range(32)]
  b = [[float((r * 5 + c) % 3 - 1) for c in range(32)] for r in range(64)]
  assert (Tensor.of(rt, a) @ Tensor.of(rt, b)).tolist() == [[sum(a[i][k] * b[k][j] for k in range(64)) for j in range(32)] for i in range(32)]
  x = [[bf(random.uniform(-8, 8)) for _ in range(64)] for _ in range(96)]
  eye = [[float(r == c) for c in range(64)] for r in range(64)]
  assert (Tensor.of(rt, x) @ Tensor.of(rt, eye)).tolist() == x
  # In X @ diag(d), d has full mantissas and an output is the product x * d. It's exact in fp32 Dst only if all four
  # fidelity phases ran, since the phases cover different slices of the mantissas. The packer then rounds it once.
  d = [bf(random.uniform(1, 2)) for _ in range(64)]
  diag = [[d[c] if r == c else 0.0 for c in range(64)] for r in range(64)]
  assert (Tensor.of(rt, x) @ Tensor.of(rt, diag)).tolist() == [[packed(v * d[c]) for c, v in enumerate(row)] for row in x]

def test_matmul_random(chip):
  # 64x96 @ 96x128 is 8 output tiles on 8 workers, 3 K tiles each. Four fidelity phases make the products exact, Dst
  # sums in fp32, and the packer rounds once to bf16, so the error stays under a bf16 step of sum |a*b|.
  rt, (m, k, n) = Runtime(chip), (64, 96, 128)
  a = [[bf(random.uniform(-1, 1)) for _ in range(k)] for _ in range(m)]
  b = [[bf(random.uniform(-1, 1)) for _ in range(n)] for _ in range(k)]
  got = (Tensor.of(rt, a) @ Tensor.of(rt, b)).tolist()
  for i in range(m):
    for j in range(n):
      terms = [a[i][x] * b[x][j] for x in range(k)]
      assert abs(got[i][j] - sum(terms)) <= 2**-8 * sum(map(abs, terms)), (i, j, got[i][j], sum(terms))

def test_matmul_blocks(chip):
  # On one worker, result tiles go in blocks of up to 4 x 4, with the block's rows of A and column of B kept in L1. On
  # the full grid, each worker takes a block. 5 x 6 tiles leaves partial blocks at the edges either way. Their missing
  # tiles must not be computed, or they'd land past the result in DRAM, on the tensor allocated after it. That tensor
  # is the canary, written while a launch that waits for the host keeps the matmul queued.
  a = [[float((r * 7 + c * 3) % 5 - 2) for c in range(96)] for r in range(160)]
  b = [[float((r + 2 * c) % 3 - 1) for c in range(192)] for r in range(96)]
  for workers in (chip.tensix[1], chip.tensix[-1]):
    rt = Runtime(chip, end=workers)
    ta, tb = Tensor.of(rt, a), Tensor.of(rt, b)
    go = hold(rt)
    y = (ta @ tb).realize()
    canary = Tensor.of(rt, [[1.0] * 192] * 160)
    go()
    assert y.tolist() == [[sum(a[i][k] * b[k][j] for k in range(96)) for j in range(192)] for i in range(160)]
    assert canary.tolist() == [[1.0] * 192] * 160
  # Next, a row of result tiles and a long K, like a token's MLP. K is cut into 8 parts on 8 times the workers, then
  # summed. Sums stay under 256, so partials and totals are exact in bf16. Slicing the sum by tiles slices each part
  # the same way, and the kernel that reads the slice adds up the parts in one launch, like the halves of a GLU.
  a, b = [[float((r + c) % 3 - 1) for c in range(1024)] for r in range(5)], [[float(r % 16 == c % 16) for c in range(64)] for r in range(1024)]
  want = [[sum(a[i][k] * b[k][j] for k in range(1024)) for j in range(64)] for i in range(5)]
  y = Tensor.of(rt, a) @ Tensor.of(rt, b)
  assert y.op == "add" and y.tolist() == want
  half = (Tensor.of(rt, a) @ Tensor.of(rt, b))[:, 32:]
  launches = rt.launches
  assert half.op == "add" and (half * 2).tolist() == [[2 * v for v in row[32:]] for row in want] and kernels(rt, launches) == 1

def test_matmul_fidelity(chip):
  # Fewer fidelity phases drop the bits that tensix.matmul says they do: A's last significand bit below three phases,
  # and B's last three below two. 1 + 2^-7 is 1 plus bf16's last bit, so X @ I keeps it from three phases up and I @ X
  # from two. On random values, three phases stay within bf16 rounding of four, and two within that plus 2^-7.
  rt = Runtime(chip)
  x, eye = Tensor.full(rt, 64, 64, 1 + 2**-7), Tensor.of(rt, np.eye(64).tolist())
  for phases in (0, -1, 5):
    with pytest.raises(ValueError, match="fidelity"): x.matmul(eye, fidelity=phases)
  for phases in (4, 3, 2, 1):
    assert x.matmul(eye, fidelity=phases).tolist()[0][0] == (1 + 2**-7 if phases >= 3 else 1.0), phases
    assert eye.matmul(x, fidelity=phases).tolist()[0][0] == (1 + 2**-7 if phases >= 2 else 1.0), phases
  rng = np.random.default_rng(4)
  a, b = bf16(rng.normal(0, 1, (96, 256))), bf16(rng.normal(0, 1, (256, 64)))
  ta, tb = Tensor.of(rt, a.tolist()), Tensor.of(rt, b.tolist())
  for phases, bound in ((3, 2**-8), (2, 2**-8 + 2**-7)):
    assert np.all(np.abs(value(ta.matmul(tb, fidelity=phases))[0] - a @ b) <= bound * (np.abs(a) @ np.abs(b))), phases

@pytest.mark.parametrize("dtype", ["bfp8", "bfp4"])
def test_bfp(chip, dtype):
  # A matmul by a block-float B is bit for bit the bf16 matmul by its values. B reads 1088 bytes a tile for bfp8 and
  # 576 for bfp4, against 2048 for bf16. The loop covers the ways B comes in: streamed beside A when K is over 256
  # tiles, a token's row of A kept and shared, multicast in three buffers, a batch with a block's column kept, and K
  # split. Then it's fused with what follows and with a bf16 matmul, so unpacker 0 switches to the block float and
  # back.
  rt = Runtime(chip)
  rng = np.random.default_rng(5)
  on = lambda *shape: Tensor.of(rt, rng.normal(0, 1, shape).tolist())
  for b, m, k, n in ((1, 64, 9000, 40), (1, 1, 768, 2304), (1, 100, 256, 200), (2, 100, 256, 96), (1, 32, 2048, 64)):
    x, w = on(b, m, k), on(b, k, n).cast(dtype)
    assert (x @ w).tolist() == (x @ Tensor.of(rt, w.tolist())).tolist(), (b, m, k, n)
  x, w, v, bias = on(64, 192), on(192, 96).cast(dtype).realize(), on(192, 96), on(1, 96)  # K too short to split, so one kernel
  launches, y = rt.launches, (x @ w + x @ v + bias).relu().realize()
  assert rt.launches == launches + 1 and y.tolist() == (x @ Tensor.of(rt, w.tolist()) + x @ v + bias).relu().tolist()

def test_matmul_multicast(chip):
  # On a grid of workers (Runtime.grid), a matmul reads each input tile from DRAM once. With several rows of tiles,
  # worker (gy, gx) computes a block. It reads its share of A's rows and multicasts it along its grid row, and its
  # share of B's columns down its grid column. With one row, like a token's, workers take a block of columns each, and
  # the first grid row reads A's row and multicasts it to the grid, from 2 workers to 96.
  # The cases add odd edges, a K so long that B's three buffers nearly fill the slots below 256, and few result tiles
  # of more than one row, where K isn't split because split K's batch would read its own tiles. B is also read
  # transposed, a tile per read: the unpacker transposes each face and the math walks the faces in transposed order.
  rt = Runtime(chip)
  reads = lambda: sum(chip.read32(x, y, bh.NIU_READS) for x, y in rt.cores)
  rng = np.random.default_rng(3)
  for m, k, n, bt in ((320, 96, 416, False), (288, 1920, 224, False), (64, 512, 256, False), (352, 64, 200, True),
                      (1, 768, 3072, False), (5, 96, 40, True)):
    a, b = bf16(rng.uniform(-1, 1, (m, k))), bf16(rng.uniform(-1, 1, (k, n)))
    tb = Tensor.of(rt, b.T.tolist()).T if bt else Tensor.of(rt, b.tolist())
    y, before = Tensor.of(rt, a.tolist()) @ tb, reads()
    y.realize(), rt.wait()
    tiles = -(-m // 32) * (k // 32) + (k // 32) * -(-n // 32)
    assert reads() - before == tiles, (m, k, n, reads() - before, tiles)
    assert np.all(np.abs(value(y)[0] - a @ b) <= 2**-8 * (np.abs(a) @ np.abs(b))), (m, k, n)
  x, w, d, bias = (Tensor.of(rt, bf16(rng.uniform(-1, 1, shape)).tolist()) for shape in ((96, 64), (64, 200), (96, 200), (1, 200)))
  assert (x @ w * d + bias).tolist() == unfused(x @ w * d + bias).tolist()

def test_matmul_parts(chip):
  # A grid matmul whose K is too long for A's rows and B's buffers below slot 256 takes K in parts, in one kernel (see
  # kernels.grid_parts). Workers sum their result tiles over the parts in fp32 accumulators in L1, and the last part
  # starts from that sum. When the accumulators don't fit, a worker does its block a group of columns at a time and
  # reads A again for each group. B is read once; A is read once per group.
  # The cases split K into these parts, in tiles: 22, 22, 20. Then 64, 64 with a bfp8 B in regions past the slots, 22
  # with B transposed, and 32 four times, so A's buffers go round. Then 22 with the last workers of the grid taking 2
  # of their 3 columns. Last is a grid of one row, with blocks of 32 tile rows in two groups of 4 columns, where the
  # last worker has none in the second group.
  base = tensix.FRAMES + 2 * 3 + 1  # a lone matmul: two frames of 3 slots, plus one slot for its op
  rng = np.random.default_rng(41)
  for end, m, k, n, how in ((None, 320, 2048, 416, "bf16"), (None, 288, 4096, 224, "bfp8"), (None, 352, 2048, 200, "T"),
                            (None, 96, 4096, 200, "bf16"), (None, 320, 2048, 832, "bf16"), (row_end(chip), 1024, 256, 2944, "bf16")):
    rt = Runtime(chip, end=end)
    reads = lambda: sum(chip.read32(x, y, bh.NIU_READS) for x, y in rt.cores)
    kp, q, _ = grid_parts(-(-m // 32), -(-n // 32), k // 32, rt.grid, base, 2)  # this case goes in parts
    w = -(-n // 32)
    bq = -(-w // min(len(rt.grid[0]), w))  # columns of result tiles in each worker's block, as in grid_parts
    groups = -(-bq // q)  # and the groups they're taken in
    a, b = bf16(rng.uniform(-1, 1, (m, k))), bf16(rng.uniform(-1, 1, (k, n)))
    tb = Tensor.of(rt, b.T.tolist()).T if how == "T" else Tensor.of(rt, b.tolist()).cast(how).realize()
    x = Tensor.of(rt, a.tolist())
    rt.wait()
    y, launches, before = x @ tb, rt.launches, reads()
    y.realize(), rt.wait()
    tiles = -(-m // 32) * (k // 32) * groups + (k // 32) * -(-n // 32) * (2 if how == "bfp8" else 1)  # a block float reads exponents and values
    assert kernels(rt, launches) == 1 and reads() - before == tiles, (m, k, n, how, reads() - before, tiles)
    v = value(tb)[0]
    assert np.all(np.abs(value(y)[0] - a @ v) <= 2**-8 * (np.abs(a) @ np.abs(v))), (m, k, n, how)
    if end: assert (kp, q, groups) == (2, 4, 2), (kp, q, groups)
  rt = Runtime(chip)
  x, w, d, bias = (Tensor.of(rt, bf16(rng.uniform(-1, 1, shape)).tolist()) for shape in ((96, 4096), (4096, 200), (96, 200), (1, 200)))
  assert (x @ w * d + bias).tolist() == unfused(x @ w * d + bias).tolist()

@pytest.mark.parametrize("layout", ["bf16", "bfp8", "columns"])
def test_matmul_parts_scalars(chip, layout):
  # K spans several parts; the last part's epilogue still needs the scalars fetched with the first tile.
  rt = Runtime(chip)
  x, w = Tensor.full(rt, 64, 2048, 1 / 32), Tensor.full(rt, 2048, 128, 1 / 32)
  w = w.columns() if layout == "columns" else w.cast(layout).realize()
  result = (x @ w * 0.5 + 3).realize()
  assert result.tolist() == [[4.0] * 128] * 64

@pytest.mark.parametrize("dtype", ["bf16", "bfp8", "bfp4"])
def test_matmul_split_transposed_weights(chip, dtype):
  # A one-row matmul split along K reads a transposed B in place, block floats included, with no transpose kernel.
  rt = Runtime(chip)
  x = Tensor.full(rt, 1, 1024, 1 / 32)
  w = Tensor.full(rt, 64, 1024, 1 / 32).cast(dtype).realize()
  since = rt.launches
  assert (x @ w.T).tolist() == [[1.0] * 64]
  assert all(rt.sent[n % TIMES][0] != "transpose" for n in range(since + 1, rt.launches + 1))

def test_matmul_long_row(chip):
  # A token's row of A can be too long for the slots below 256, like a decode step's down projection with K of 448
  # tiles. Then it stays in L1 past those slots, and unpacker 1 steps through it in regions of 256 slots. The grid's
  # first row reads A once and multicasts it, instead of reading it once per output tile. A grid row of 12 workers is
  # too few to cut K into a batch. K is 300 tiles, B is bf16, bfp4 or transposed, and a residual is added in the same
  # kernel.
  rt = Runtime(chip, end=row_end(chip))
  reads = lambda: sum(chip.read32(x, y, bh.NIU_READS) for x, y in rt.cores)
  rng = np.random.default_rng(31)
  a, b, r = bf16(rng.uniform(-1, 1, (1, 9600))), bf16(rng.uniform(-1, 1, (9600, 384))), bf16(rng.uniform(-1, 1, (1, 384)))
  x, res = Tensor.of(rt, a.tolist()), Tensor.of(rt, r.tolist())
  for how in ("bf16", "bfp4", "T"):
    w = Tensor.of(rt, b.T.tolist()).T if how == "T" else Tensor.of(rt, b.tolist()).cast(how).realize()
    rt.wait()  # so the cast's reads aren't counted
    y, launches, before = x @ w + res, rt.launches, reads()
    y.realize(), rt.wait()
    assert kernels(rt, launches) == 1 and reads() - before == 300 + 12 * (300 * (2 if how == "bfp4" else 1) + 1), (how, reads() - before)
    v = value(w)[0]
    assert np.all(np.abs(value(y)[0] - (a @ v + r)) <= 2**-8 * (np.abs(a) @ np.abs(v) + np.abs(r))), how

def test_matmul_bank_local(chip):
  # A token's matmul whose B keeps each column of tiles in one bank. Grid workers take the columns of the bank that
  # Runtime.readers gives them, read through their port, and take turns with the bank's other readers. B is bf16,
  # bfp8, bfp4, or a view whose first tile is in bank 3. Then a row of A too long for the slots below 256 runs on a
  # grid row of 12, with K of 300 tiles. A residual is added in the same kernel. Each input tile is read once, A's by
  # the first grid row.
  rng = np.random.default_rng(41)
  for end, k, n, hows in ((None, 1024, 4096, ("bf16", "bfp8", "bfp4", "view")), (row_end(chip), 9600, 512, ("bf16", "bfp4"))):
    rt = Runtime(chip, end=end)
    reads = lambda: sum(chip.read32(x, y, bh.NIU_READS) for x, y in rt.cores)
    a, b, r = bf16(rng.uniform(-1, 1, (1, k))), bf16(rng.uniform(-1, 1, (k, n + 256))), bf16(rng.uniform(-1, 1, (1, n)))
    x, res = Tensor.of(rt, a.tolist()), Tensor.of(rt, r.tolist())
    for how in hows:
      w = Tensor.of(rt, b.tolist())[:, 96:n + 96] if how == "view" else Tensor.of(rt, b[:, :n].tolist()).cast(how).realize()
      rt.wait()  # so the cast's reads aren't counted
      y, launches, before, rt.log = x @ w + res, rt.launches, reads(), []
      y.realize(), rt.wait()
      (_, _, units, _), rt.log = rt.log[-1], None
      assert kernels(rt, launches) == 1 and units == len(rt.readers), (how, units)
      assert reads() - before == k // 32 + n // 32 * (k // 32 * (2 if how in ("bfp8", "bfp4") else 1) + 1), (how, reads() - before)
      v = value(w)[0]
      assert np.all(np.abs(value(y)[0] - (a @ v + r)) <= 2**-8 * (np.abs(a) @ np.abs(v) + np.abs(r))), how

def test_matmul_by_columns(chip):
  # Tensor.columns must preserve matmul results across all read paths. Tile column j forms one run along K
  # in bank j % 8. Block floats store the run's exponents before its values.
  # The cases cover a token's row: bank-local, with K too long for the slots, and with K cut into a batch. They cover a
  # grid's blocks with K in one part and in several, passes over a row as in a softmax, blocks on one worker, A
  # streamed beside B, and the experts that tokens pick with index=. B by columns reads back as it was; ordinary
  # arithmetic and movement require casting it back to ordinary tiles.
  rng = np.random.default_rng(43)
  cases = [(None, 1, 1024, 4000, "bf16"), (None, 1, 2048, 1024, "bfp8"),
    (None, 288, 4096, 224, "bfp8"), (None, 320, 2048, 416, "bf16"), (None, 64, 256, 512, "softmax"), (1, 96, 256, 128, "bf16"),
    (1, 32, 9600, 64, "bf16")]
  for end, m, k, n, how in cases:
    rt = Runtime(chip, end=row_end(chip) if end == "row" else end and chip.tensix[end])
    x, b = Tensor.of(rt, bf16(rng.uniform(-1, 1, (m, k))).tolist()), Tensor.of(rt, bf16(rng.uniform(-1, 1, (k, n))).tolist())
    w = b.cast(how).realize() if how in ("bfp8", "bfp4") else b
    f = (lambda w: (x @ w).softmax(-1)) if how == "softmax" else (lambda w: x @ w)
    wc = w.columns()
    assert np.array_equal(value(f(wc)), value(f(w))) and wc.tobytes() == w.tobytes(), (end, m, k, n, how)
  experts = Tensor.of(rt := Runtime(chip), rng.normal(0, 0.1, (16, 96, 160)).tolist())
  for w in (experts, experts.cast("bfp4").realize()):
    for t, e in ((1, 8), (4, 3)):  # one token with 8 experts sharing A's row, and 4 tokens with 3 each
      ids, x = Tensor.of(rt, rng.integers(0, 16, (t, e)).tolist(), "int32"), Tensor.of(rt, rng.normal(0, 1, (t, 1, 96)).tolist())
      x = x.repeat_interleave(e) if t > 1 else x
      assert np.array_equal(value(x.matmul(w.columns(), index=ids)), value(x.matmul(w, index=ids))), (w.dtype, t, e)
  x, wc = Tensor.of(rt, rng.normal(0, 1, (1, 64)).tolist()), Tensor.of(rt, rng.normal(0, 1, (64, 64)).tolist()).columns()
  for misuse in (lambda: wc + 1, lambda: x @ wc.T, lambda: wc[:, :32], lambda: wc.gather([0])):
    with pytest.raises(ValueError, match="by columns"): misuse().realize()
  invalid = wc.T
  for _ in range(2):
    with pytest.raises(ValueError, match="by columns"): invalid.realize()
    assert invalid.op == "T" and invalid.addr is None, "a failed move stays unrealized"

def test_matmul_chunks(chip):
  # B laid out by columns comes in long reads. A token's row streams each column in chunks with CHUNKS reads in flight,
  # also when K is too long for the slots. A chunk is 8 tiles in bf16 or 16 in a block float, whose exponents are a
  # read of their own (tensix.chunk_tiles). A grid worker reads its share of a column in one read. The read count must
  # be exact, with A read a tile at a time, and the results must match bit for bit what B as it lies gives.
  rng = np.random.default_rng(47)
  for end, m, k, n, how in ((None, 1, 1024, 4096, "bf16"), (None, 1, 1024, 4096, "bfp8"), (None, 1, 1024, 4096, "bfp4"),
                            (row_end(chip), 1, 9600, 512, "bfp4"), (None, 320, 1024, 416, "bf16"), (None, 320, 1024, 416, "bfp8")):
    rt = Runtime(chip, end=end)
    reads = lambda: sum(chip.read32(x, y, bh.NIU_READS) for x, y in rt.cores)
    x, b = Tensor.of(rt, bf16(rng.uniform(-1, 1, (m, k))).tolist()), Tensor.of(rt, bf16(rng.uniform(-1, 1, (k, n))).tolist())
    w = b if how == "bf16" else b.cast(how).realize()
    wc, tr, kt, wt, pieces = w.columns(), -(-m // 32), k // 32, -(-n // 32), 1 if how == "bf16" else 2  # exponents and values
    rt.wait()
    before = reads()
    y = (x @ wc).realize()
    rt.wait()
    if m == 1: want = kt + wt * -(-kt // tensix.chunk_tiles(None if how == "bf16" else how)) * pieces
    else: want = tr * kt + wt * -(-tr // -(-tr // len(rt.grid[1]))) * pieces  # each grid row's share of a column, up to 16 KiB
    assert reads() - before == want and np.array_equal(value(y), value(x @ w)), (m, k, n, how, reads() - before, want)
    assert wc.tobytes() == w.tobytes(), "column storage reads back unchanged"

def test_accumulate_parts(chip):
  # Build a one-worker matmul with six K tiles split into three parts. The first two parts accumulate in fp32
  # in L1. The last waits on SEM_L1, reloads the sum through SrcA as TF32, then adds its products in Dst.
  # Compare with accumulating all six tiles in Dst. The allowed difference is TF32's 2^-10 on the reloaded sum.
  c, rt, rng = tensix, Runtime(chip, end=chip.tensix[1]), np.random.default_rng(37)
  parts, kp = 3, 2  # parts of K, and tiles in each part
  kt, acc, slots, cfg = parts * kp, tensix.B8, tensix.allocate({"mm"}), tensix.kernel_config()
  a, b = bf16(rng.normal(0, 1, (32, 32 * kt))), bf16(rng.normal(0, 1, (32 * kt, 32)))
  tile = lambda m: m.reshape(2, 16, 2, 16).transpose(0, 2, 1, 3).reshape(-1)  # a 32 x 32 matrix in tile order, face by face
  x, y = rt.cores[0]
  for m in range(kt):  # B's tile m goes to slot 1 + m for SrcA, A's to 1 + kt + m for SrcB
    for at, t in ((1 + m, b[32 * m:32 * m + 32]), (1 + kt + m, a[:, 32 * m:32 * m + 32])):
      chip.write(x, y, c.slot_base("bf16") + at * 2048, (tile(t).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes())
  last = parts - 1
  steps = lambda p: [w for m in range(p * kp, p * kp + kp) for w in c.unpack(1 + m, 1 + kt + m)]
  unpacks = c.tt([w for p in range(parts) for w in (c.TAKE_L1 + c.reload(acc, 0) if p == last else []) + steps(p)])
  maths = " ".join(c.tt(c.math_begin(p < last) + c.elementwise("add", slots) * (p == last) + c.matmul(slots) * kp + c.MATH_END) +
                   f" h ^= {c.DST_HALF};" for p in range(parts))
  packer = [cfg] + [c.accumulating(acc, 0 < p < last) for p in range(parts)] + [cfg]  # the packer config for each part, and after the last
  packs = " ".join(c.tt(c.repack(packer[p], packer[p + 1]) + c.pack("h", int(p == last), *[c.SEM_L1] * (p == last - 1))) +
                   f" h ^= {c.DST_HALF};" for p in range(parts)) + " " + c.tt(c.repack(packer[parts], cfg))
  rt.launch(compile_kernel(None, c.unpack_program(unpacks, cfg), c.math_program(slots, maths), c.pack_program(packs)), (), 1)
  rt.wait()
  got = np.frombuffer(chip.read(x, y, acc + 4096, 4096), np.float32).reshape(2, 2, 16, 16).transpose(0, 2, 1, 3).reshape(32, 32)
  want, first = value(Tensor.of(rt, a.tolist()).matmul(Tensor.of(rt, b.tolist()), "fp32"))[0], 32 * kp * last
  bound = 2**-10 * (np.abs(a[:, :first]) @ np.abs(b[:first])) + 2**-20 * (np.abs(a) @ np.abs(b))
  assert np.all(np.abs(got - want) <= bound), np.abs(got - want).max()

def test_matmul_indexed(chip):
  # x.matmul(w, index=ids) is the matmul of a mixture of experts. Entry b of the result is row b // k of x times the
  # matrix of w that ids[b // k, b % k] picks. BRISC reads ids on the chip, so a replay uses the experts that its tokens
  # chose. The first case is one token with 8 experts sharing A's row, with a grid row for each expert. Then come 4
  # tokens with 3 experts each, more entries than the grid has rows, and 40 tokens with 2 each, so the indices take two
  # tiles. Entries match the matmul by their expert bit for bit, and an id of 16 picks no expert and gives zeros.
  rt, rng = Runtime(chip), np.random.default_rng(29)
  experts = Tensor.of(rt, rng.normal(0, 0.1, (16, 96, 160)).tolist())
  for wt in ("bf16", "bfp4"):
    w = experts if wt == "bf16" else experts.cast(wt).realize()
    for n, k in ((1, 8), (4, 3), (40, 2)):
      ids, x = rng.integers(0, 16, (n, k)), Tensor.of(rt, rng.normal(0, 1, (n, 1, 96)).tolist())
      ids[-1, 0] = 16
      got = value((x.repeat_interleave(k) if n > 1 else x).matmul(w, index=Tensor.of(rt, ids.tolist(), "int32")))
      want = lambda b, e: value(x[b // k] @ w[e])[0] if e < 16 else np.zeros((1, 160))
      assert all(np.array_equal(got[b], want(b, int(ids[b // k, b % k]))) for b in range(n * k)), (wt, n, k)

def test_indexed_matmul_rows_and_passes(chip):
  # A matmul by index over several tile rows, and one under a softmax's passes. An index past B's batch picks zeros.
  rt, rng = Runtime(chip), np.random.default_rng(71)
  a, b = bf16(rng.normal(0, 0.1, (64, 96))), bf16(rng.normal(0, 0.1, (3, 96, 160)))
  x, w = Tensor.of(rt, a.tolist()), Tensor.of(rt, b.tolist())
  for index in (1, 3):
    ids = Tensor.of(rt, [[index]], "int32")
    want = a @ b[index] if index < 3 else np.zeros((64, 160))
    close(x.matmul(w, index=ids), want, 0.004)
  a = bf16(rng.normal(0, 0.1, (2, 1, 96)))
  x = Tensor.of(rt, a.tolist()).repeat_interleave(2)
  ids = Tensor.of(rt, [[0, 1], [2, 7]], "int32")
  scores = np.stack([a[i // 2] @ b[j] if j < 3 else np.zeros((1, 160)) for i, j in enumerate((0, 1, 2, 7))])
  expected = np.exp(scores - scores.max(axis=-1, keepdims=True))
  expected /= expected.sum(axis=-1, keepdims=True)
  close(x.matmul(w, index=ids).softmax(), expected, 0.0001)

def test_reduction_l1_boundary(chip):
  # fp32 softmaxes at the widths where a row's tiles stop fitting in the slots below the outputs. One of them raised.
  rt, rng = Runtime(chip), np.random.default_rng(73)
  for width in (7424, 7456, 7488):
    a = rng.normal(0, 1, (2, width)).astype(np.float32)
    x = Tensor.frombytes(rt, 2, width, a.tobytes(), dtype="fp32")
    expected = np.exp(a - a.max(axis=-1, keepdims=True))
    expected /= expected.sum(axis=-1, keepdims=True)
    close(x.softmax(), expected, 1e-6)

def test_stream_on_one_worker(chip):
  # On one worker the pipeline cycles through many tiles: L1 slots and Dst halves alternate, the threads hand tiles
  # along by semaphore, and BRISC fetches and writes back around them. The test runs 32 elementwise tiles, then a
  # 128x96 @ 96x128 matmul of 16 output tiles with 3 steps each.
  rt = Runtime(chip, end=chip.tensix[1])
  a, b = ([[bf(random.uniform(-2, 2)) for _ in range(128)] for _ in range(256)] for _ in range(2))
  assert (Tensor.of(rt, a) * Tensor.of(rt, b)).tolist() == [[packed(x * y) for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]
  x = [[float((r + 2 * c) % 5 - 2) for c in range(96)] for r in range(128)]
  y = [[float((3 * r + c) % 3 - 1) for c in range(128)] for r in range(96)]
  assert (Tensor.of(rt, x) @ Tensor.of(rt, y)).tolist() == [[sum(x[i][k] * y[k][j] for k in range(96)) for j in range(128)] for i in range(128)]

def unfused(t):  # realizes the same tree of ops with one launch per op
  if t.op is None: return t
  srcs = [unfused(s) for s in t.srcs]
  return (srcs[0] @ srcs[1] if t.op == "matmul" else Tensor(t.rt, t.rows, t.cols, t.op, tuple(srcs))).realize()

def test_fusion(chip):
  # A fused chain is one launch, and its intermediates don't get DRAM. It gives the same bits as one launch per op,
  # because intermediates are rounded to bf16 on their way to L1, as they would be on their way to DRAM. Both trees
  # have ops that read two intermediates, and in the second an op reads an intermediate made two ops earlier.
  # On one worker, only the semaphore waits hold a thread back, so this tests them. The TRISCs feed their threads as
  # fast as they can, and ttsim runs an instruction the moment it can. The pack TRISC has the least to push, so it
  # queues each pack before its math is done (SEM_DST). Thread 0 reaches the ops that read intermediates before the
  # pack thread has written them (SEM_L1). The second tree's math has three ops in a row that read only inputs, so the
  # third waits for the pack thread to drain a Dst half (SEM_DST).
  for workers in (chip.tensix[1], chip.tensix[-1]):  # one worker cycling through 12 tiles, then a tile per worker
    rt = Runtime(chip, end=workers)
    a, b, c, d = (Tensor.of(rt, [[bf(random.uniform(-3, 3)) for _ in range(96)] for _ in range(128)]) for _ in range(4))
    for build in (lambda a, b, c, d: (a + b) * c - d * a, lambda a, b, c, d: (a + b) * ((c - d) + a * a)):
      fused, launches = build(a, b, c, d), rt.launches
      lazy = [t for t in (fused, *fused.srcs, *fused.srcs[0].srcs, *fused.srcs[1].srcs) if t.op]
      got = fused.tolist()
      assert kernels(rt, launches) == 1 and all(t.addr is None for t in lazy[1:]), "one kernel, intermediates stay on chip"
      assert got == unfused(build(a, b, c, d)).tolist()

def test_fused_cast(chip):
  # A cast at the root of a tree is part of its kernel: the packer switches to the result's format for the root's pack,
  # then back. It's bit for bit what casting the computed value gives, where a bf16 result is rounded first as its pack
  # would round it.
  rt, rng = Runtime(chip), np.random.default_rng(16)
  x, y = (Tensor.of(rt, rng.normal(0, 2, (70, 100)).astype(np.float32).tolist(), "fp32") for _ in range(2))
  a, b, c = (Tensor.of(rt, rng.normal(0, 2, shape).tolist()) for shape in ((70, 100), (70, 100), (100, 40)))
  for build in (lambda: (x * 3 + y).bfloat16(), lambda: (x * 50 + y).cast("int32"), lambda: (x * y).sum(1).bfloat16(),
                lambda: (x * y - x).cast("bfp8"), lambda: (a * b + a).float(), lambda: a.softmax(-1).float(), lambda: (a @ c).float(),
                lambda: (a * b + a).cast("bfp8")):
    launches, fused = rt.launches, build().realize()
    assert kernels(rt, launches) == 1
    t = build()
    t.srcs[0].realize()
    assert fused.tobytes() == t.realize().tobytes()

def test_assign(chip):
  # assign writes a new value into the tensor's DRAM, so views of it see the value and a jit call can update weights.
  # It goes tile by tile when the computation reads the tensor elementwise. When a matmul reads it, it goes through a
  # copy, since each result tile of a matmul reads a row of A and a column of B, which other result tiles may already
  # have overwritten.
  rt, rng = Runtime(chip), np.random.default_rng(17)
  w0, m = (bf16(rng.uniform(-1, 1, shape)) for shape in ((64, 96), (96, 96)))
  w, tm = Tensor.of(rt, w0.tolist()), Tensor.of(rt, m.tolist())
  addr, view = w.addr, w[:, 32:64]
  w.assign(w * 2 + 1)
  assert w.addr == addr and np.allclose(value(view)[0], (w0 * 2 + 1)[:, 32:64], rtol=1e-2), "in place: a view sees it"
  fresh = value(Tensor.of(rt, value(w)[0].tolist()) @ tm)
  w.assign(w @ tm)
  assert w.addr == addr and np.array_equal(value(w), fresh), "w @ tm in place should match w @ tm on a fresh copy"
  step = tensor.jit(lambda x: w.assign(w + x))
  for _ in range(4): step(Tensor.of(rt, np.ones((64, 96)).tolist()))
  assert np.allclose(value(w), fresh + 4, rtol=1e-2, atol=0.05), "four updates, two of them replayed"

def test_assign_overlapping_views(chip):
  # A copy must read the original source even if a different tile of the destination aliases it. Cover both
  # directions, raw copies and fused expressions, on one worker and across the grid.
  a = np.arange(64 * 640, dtype=np.float32).reshape(64, 640)
  for end in (chip.tensix[1], chip.tensix[-1]):
    rt = Runtime(chip, end=end)
    for left, right in ((slice(32, None), slice(None, -32)), (slice(None, -32), slice(32, None))):
      for add in (False, True):
        x = Tensor.of(rt, a.tolist(), "fp32")
        source, want = x[:, right], a.copy()
        x[:, left].assign(source + 1 if add else source)
        want[:, left] = a[:, right] + int(add)
        assert np.array_equal(value(x)[0], want), (end, left, add)
    x = Tensor.of(rt, a.tolist()).cast("bfp8").realize()
    want = value(x)[0]
    x[:, 32:].assign(x[:, :-32])
    want[:, 32:] = want[:, :-32].copy()
    assert np.array_equal(value(x)[0], want), "overlapping block floats copy without elementwise arithmetic"

def test_assign_matmul_without_copy(chip):
  # assign computes a matmul that doesn't read its destination straight into it, in one kernel.
  rt = Runtime(chip)
  x, w, out = (Tensor.full(rt, 32, 32, n) for n in (1, 2, 0))
  since = rt.launches
  out.assign(x @ w + 3)
  assert kernels(rt, since) == 1, "an independent destination is written by the fused matmul"
  assert out.tolist() == [[67.0] * 32] * 32

def test_realize_overlapping_destination(chip):
  for end in (chip.tensix[1], chip.tensix[-1]):
    rt = Runtime(chip, end=end)
    a, b = (Tensor.full(rt, 256, 256, 1) for _ in range(2))
    out = (a @ b).realize(b)
    assert out.addr == b.addr and np.all(value(b) == 256) and np.all(value(out) == 256)
    data = np.arange(64 * 96, dtype=np.float32).reshape(64, 96)
    x = Tensor.of(rt, data.tolist(), "fp32")
    (x[:, :-32] + 1).realize(x[:, 32:])
    want = data.copy()
    want[:, 32:] = data[:, :-32] + 1
    assert np.array_equal(value(x)[0], want)

def test_matmul_fusion(chip):
  # Matmuls inside elementwise trees run as one kernel: the matmul's result tile is packed into L1 and read from there.
  # It gives the same bits as one launch per op, since the tile is rounded to bf16 on its way to L1 as it would be on
  # its way to DRAM.
  rnd = lambda r, c: [[bf(random.uniform(-1, 1)) for _ in range(c)] for _ in range(r)]
  for workers in (chip.tensix[1], chip.tensix[-1]):  # one worker cycling through 8 tiles, then a tile per worker
    rt = Runtime(chip, end=workers)
    a, b, c, d, e, f = (Tensor.of(rt, rnd(*shape)) for shape in ((64, 96), (96, 128), (64, 128), (64, 128), (64, 64), (64, 128)))
    for build in (lambda: (a @ b) + c, lambda: (a @ b) * d - c, lambda: (a @ b) + (e @ f)):
      launches, got = rt.launches, build().tolist()
      assert kernels(rt, launches) == 1, "one kernel"
      assert got == unfused(build()).tolist()

def test_sfpu_scheduling():
  # An SFPMAD's result takes two cycles, as do SFPMUL's and SFPADD's. Blackhole's stall logic holds back most readers of
  # it, but not SFPSHFT2's rotations, SFPSWAP's compares, or SFPSHFT and SFPIADD reading vd (ISA SFPMAD.md, "Instruction
  # scheduling"). So no SFPU program may follow one of these products with such a read of it. A body runs again and
  # again, so it's checked across its wrap too.
  names = {op: name for name, (op, _) in bh.TENSIX_OPS.items()}
  def get(w, f):
    first, width = bh.TENSIX_OPS[names[w >> 24]][1][f]
    return w >> first & ((1 << width) - 1)
  def unseen(w):  # the LRegs that w reads where the stall logic doesn't look
    name = names.get(w >> 24)
    if name == "SFPSHFT2" and get(w, "mod1") == 3 or name == "SFPSWAP" and get(w, "mod1"): return {get(w, "vc"), get(w, "vd")}
    return {get(w, "vd")} if name in ("SFPSHFT", "SFPIADD") else set()
  programs = [setup + body + body for setup, body in tensix.SFPU.values()] + [body + body for body in tensix.COMBINE.values()]
  programs += [tensix.ACROSS[ax](kind) for kind, ax in tensix.FOLDS.items()]
  for program in programs:
    words = [w for w in program if isinstance(w, int)]
    for a, b in zip(words, words[1:]):
      assert not (names.get(a >> 24) in ("SFPMAD", "SFPMUL", "SFPADD") and get(a, "vd") in unseen(b)), (f"{a:#x}", f"{b:#x}")

def test_replay_keeps_pushes_small():
  # The TRISCs push one Tensix instruction a cycle, or four when they come in a row, so words pushed per tile cost
  # fetches and time. The repeating runs (a matmul phase, a mul phase, 15 PACRs) are recorded once per kernel in the
  # replay buffers of the threads, and replaying a run takes one word. An SFPU body's 32 runs per tile are a loop
  # around one REPLAY word. Aiming the FPU at a Dst half takes a STALLWAIT too, since a reduction's SFPU work may still
  # be running when the next tile begins.
  c, slots = tensix, tensix.allocate({"mm", "mul"})
  tile = lambda op: len(c.unpack(0, 1)) + len(c.math_begin(op == "mul") + c.elementwise(op, slots) + c.MATH_END) + len(c.pack(0, 0, c.SEM_L1))
  assert tile("add") <= 27 and tile("mul") <= 28, "a tile's repeating runs replayed, a word each"
  assert len(c.step_unpack(0) + c.matmul(slots)) <= 12, "a matmul step's phases replayed"
  assert len(c.sfpu(["recip"], c.allocate({"recip"}))) <= 3, "32 passes of 10 words: one REPLAY word, in a loop"
  for names in ({"mm", "mul"}, {"exp", "recip", "relu", "mm", "mul"}):  # what is recorded fits the 32-entry buffer
    assert sum(n for _, n in c.allocate(names).values()) <= 32
  # Only the transcendentals, with their long range reductions, don't fit on their own. A run left out records what fits.
  assert {n for n, (_, body) in c.SFPU.items() if len(body) > 32} == {"exp", "exp2", "fexp", "fexp2", "log", "log2", "sin", "cos"}
  assert c.run(c.allocate({"sin"}), "sin") == [c.replay(0, 32)] + c.MATH_RUNS["sin"][32:]
  # The programs that a kernel runs together for rsqrt, a row's rsqrt, where and int32 minimum are all recorded whole
  for names in ({"sqrt", "recip"}, {"sum2", "sqrt"}, {"fcmplt", "keep", "drop", "join"}, {"icmplt", "keep", "drop", "join"}, {"isub", "imax"}):
    assert {name: n for name, (_, n) in c.allocate(names).items()} == {name: len(c.MATH_RUNS[name]) for name in names}, names

def test_sfpu(chip):
  # relu is exact, and exp and 1/x are within a bf16 step: they run in fp32 on the SFPU and the packer rounds once. The
  # Tensix flushes denormals to zero on the way in and out, so the reference does too: 1/denormal is inf and exp(-100)
  # is 0. They run on an input (through x + 0), and in place after an add. There exp sees the add's fp32 result with no
  # bf16 rounding in between, so the reference is exp of the exact sum.
  rt = Runtime(chip, end=chip.tensix[2])
  flush = lambda v: 0.0 if abs(v) < 2**-126 else v
  x = [[bf(random.uniform(-6, 6)) for _ in range(64)] for _ in range(64)]
  x[0][:10] = map(bf, [0.0, -0.0, 1.0, -1.0, -100.0, 3.0e-39, 1e30, -1e30, math.inf, -math.inf])
  y = [[bf(random.uniform(-1, 1)) for _ in range(64)] for _ in range(64)]
  tx, ty = Tensor.of(rt, x), Tensor.of(rt, y)
  def check(got_rows, ins, f):  # within a bf16 step, but 1/0 only has to be infinite
    for got, row in zip(got_rows, ins):
      for g, v in zip(got, row):
        want = f(flush(v))
        assert (math.isinf(g) if flush(v) == 0 and f is recip else g == want or abs(g - want) <= abs(want) * 2**-7), (v, g, want)
  exp = lambda v: math.inf if v > 88.7 else 0.0 if v < -104 else flush(bf(math.exp(v)))
  recip = lambda v: math.inf if v == 0 else flush(bf(1 / v))  # 1/inf is 0
  assert tx.relu().tolist() == [[max(flush(v), 0.0) for v in row] for row in x]
  check(tx.exp().tolist(), x, exp)
  check(tx.reciprocal().tolist(), x, recip)
  check((tx + ty).exp().tolist(), [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(x, y)], exp)
  # relu fused onto a matmul plus an add is one launch. Rounding commutes with relu, so it matches op by op bit for bit.
  a, b, c = (Tensor.of(rt, [[bf(random.uniform(-1, 1)) for _ in range(64)] for _ in range(64)]) for _ in range(3))
  launches, fused = rt.launches, ((a @ b) + c).relu().tolist()
  assert kernels(rt, launches) == 1 and fused == unfused(((a @ b) + c).relu()).tolist()
  # Next, the three SFPU programs run in one kernel. recip and relu are replayed. exp's 38 words don't fit the 32-entry
  # replay buffer with them, so the 18 that do are replayed and the rest pushed.
  u = [[bf(random.uniform(-3, 3)) for _ in range(64)] for _ in range(64)]
  w = [[bf(random.choice((-1, 1)) * random.uniform(0.5, 2)) for _ in range(64)] for _ in range(64)]
  tu, tw = Tensor.of(rt, u), Tensor.of(rt, w)
  launches, got = rt.launches, (tu.exp().relu() + tw.reciprocal()).tolist()
  assert kernels(rt, launches) == 1, "one kernel"
  for g, ru, rw in zip(got, u, w):
    for gv, uv, wv in zip(g, ru, rw):
      e, r = exp(uv), recip(wv)
      assert abs(gv - (e + r)) <= 2**-6 * max(abs(e), abs(r)), (uv, wv, gv, e + r)

def test_elementwise(chip):
  # This checks tinygrad's primitives on the SFPU, and what's built from them, against numpy. exp2, log2, sin and sqrt
  # are within a bf16 step. max, cmplt, cmpne and where (two exact picks and an add) are exact. The inputs include
  # edges: 0, -0, a denormal that the Tensix flushes to 0, inf, -inf, and negatives where only positives make sense.
  # Those give NaN, which the packer turns into inf.
  rt, rng = Runtime(chip), np.random.default_rng(8)
  edges = [0.0, -0.0, 1.0, -1.0, 3e-39, np.inf, -np.inf, 0.5, 2.0, 100.0, -100.0]
  def make(lo, hi, extra=()):
    a = bf16(np.concatenate([edges, extra, rng.uniform(lo, hi, 40 * 64 - len(edges) - len(extra))]).reshape(40, 64))
    return a, np.where(np.abs(a) < 2.0**-126, 0.0, a), Tensor.of(rt, a.tolist())
  def check(t, want, rel, atol=0.0):  # rounds want like the packer: past bf16's range is inf, and so is NaN
    want, got = np.nan_to_num(bf16(want, True), nan=np.inf, posinf=np.inf, neginf=-np.inf), np.array(t.tolist())
    assert ((got == want) | (np.abs(got - want) <= rel * np.abs(want) + atol)).all(), np.argwhere(got != want)[:5]
  with np.errstate(all="ignore"):
    _, fx, tx = make(-8, 8, [1e30, -1e30])
    for f, ref in [(tx.exp2, np.exp2), (tx.exp, np.exp), (tx.log2, np.log2), (tx.log, np.log), (tx.sqrt, np.sqrt)]: check(f(), ref(fx), 2**-7)
    _, fa, ta = make(-20, 20, [np.pi / 2, np.pi])  # absolute tolerance, since sine is small near its zeros
    check(ta.sin(), np.sin(fa), 0, 2**-7)
    check(ta.cos(), np.cos(fa), 0, 2**-7)
  pos = bf16(rng.uniform(0.1, 50, (40, 64)))
  check(Tensor.of(rt, pos.tolist()).rsqrt(), 1 / np.sqrt(pos), 2**-7)
  # comparisons, max, min and where, on small integers so that ties happen, plus signed zeros and infinities
  ints = lambda: bf16(np.concatenate([[0.0, -0.0, np.inf, -np.inf, 3.0], rng.integers(-3, 4, 40 * 64 - 5)]).reshape(40, 64))
  a, b = ints(), ints()
  ta, tb = Tensor.of(rt, a.tolist()), Tensor.of(rt, b.tolist())
  for got, want in [(ta < tb, a < b), (ta > tb, a > b), (ta == tb, a == b), (ta != tb, a != b), (ta <= tb, a <= b), (ta >= tb, a >= b),
                    (ta.maximum(tb), np.maximum(a, b)), (ta.minimum(tb), np.minimum(a, b)), (ta < 1.0, a < 1), (2.0 > ta, 2 > a),
                    ((ta < tb).where(tx, -np.inf), np.where(a < b, fx, -np.inf)), (ta.where(tb, 7.0), np.where(a != 0, b, 7))]:
    assert np.array(got.tolist()).tolist() == np.asarray(want, float).tolist()
  # A comparison, a where and two unary ops fuse into one kernel. Then a masked softmax with a mask of -inf, not -1e9.
  launches, got = rt.launches, (tx > 0).where(tx.log2(), tx.exp2())
  with np.errstate(all="ignore"): check(got, np.where(fx > 0, np.log2(fx), np.exp2(fx)), 2**-7)
  assert kernels(rt, launches) == 1
  s, m = bf16(rng.uniform(-4, 4, (40, 64))), np.tril(np.ones((40, 64)))
  masked = np.where(m > 0, s, -np.inf)
  e = np.exp(masked - masked.max(1, keepdims=True))
  close((Tensor.of(rt, m.tolist()) > 0).where(Tensor.of(rt, s.tolist()), -math.inf).softmax(), e / e.sum(1, keepdims=True), 0.02)

def test_reductions(chip):
  # Row max is exact and row sum is within a bf16 step of the sum of magnitudes. Both come out as a column. Then they
  # broadcast back, as x - max(x) along rows and as a column times a number. Python computes the reference.
  rt, rows, cols = Runtime(chip), 64, 96
  x = [[bf(random.uniform(-4, 4)) for _ in range(cols)] for _ in range(rows)]
  tx = Tensor.of(rt, x)
  assert tx.max(2).tolist() == [[max(row)] for row in x]
  for (got,), row in zip(tx.sum(2).tolist(), x):
    assert abs(got - sum(row)) <= 2**-8 * sum(map(abs, row)), (got, sum(row))
  for got, row in zip((tx - tx.max(2)).tolist(), x):
    assert all(abs(g - (v - max(row))) <= 2**-7 * max(abs(v), abs(max(row))) for g, v in zip(got, row))
  assert (tx.max(2) * 0.5).tolist() == [[max(row) / 2] for row in x]
  # A kernel sweeps its rows at one width and has two accumulators, so reductions of a wider tensor, and a third
  # reduction, run first in kernels of their own.
  y = [[bf(random.uniform(-4, 4)) for _ in range(200)] for _ in range(rows)]
  ty = Tensor.of(rt, y)
  for (got,), a, b in zip((tx.max(2) + ty.max(2)).tolist(), x, y): assert abs(got - (max(a) + max(b))) <= 2**-6 * 8, (got, a, b)
  for (got,), a in zip((tx.max(2) + tx.sum(2) * 0.0 + (tx * 2.0).max(2)).tolist(), x): assert abs(got - 3 * max(a)) <= 2**-6 * 12

def test_softmax_and_layernorm(chip):
  # Both are one launch. A row stays on one worker, which sweeps it in passes, keeping reductions in Dst and their
  # results in L1. softmax(x) = e / sum(e) with e = exp(x - max(x)) along each row, in three passes. layernorm(x) =
  # (x - mean) * rsqrt(var + eps) * gamma + beta, with gamma and beta as rows repeated down. It runs on many workers,
  # and on one, which does 3 tile rows in turn and reuses its slots and accumulators.
  for workers, rows, cols in ((chip.tensix[1], 96, 128), (chip.tensix[-1], 64, 160)):
    rt = Runtime(chip, end=workers)
    x = [[bf(random.gauss(0, 2)) for _ in range(cols)] for _ in range(rows)]
    tx = Tensor.of(rt, x)
    launches, got = rt.launches, tx.softmax().tolist()
    assert kernels(rt, launches) == 1
    for g, row in zip(got, x):
      want = [math.exp(v - max(row)) for v in row]
      assert all(abs(gv - w / sum(want)) <= 0.02 * w / sum(want) + 1e-4 for gv, w in zip(g, want))
    gamma, beta = [bf(random.uniform(0.5, 1.5)) for _ in range(cols)], [bf(random.uniform(-1, 1)) for _ in range(cols)]
    tg, tb = Tensor.of(rt, [gamma]), Tensor.of(rt, [beta])
    launches, got = rt.launches, (tx.layernorm() * tg + tb).tolist()
    assert kernels(rt, launches) == 1
    for g, row in zip(got, x):
      mean = sum(row) / cols
      sd = math.sqrt(sum((v - mean) ** 2 for v in row) / cols + 1e-5)
      assert all(abs(gv - ((v - mean) / sd * gm + bt)) <= 0.05 + 0.02 * abs(v) for gv, v, gm, bt in zip(g, row, gamma, beta))

def test_attend(chip):
  # Tensor.attend computes softmax(q k^T + mask) v, checked against numpy within 1.1%. FPU product alignment and
  # rounding probabilities to bf16 contribute error even though maxima and subtraction stay fp32. The kernel produces
  # weighted sums; a separate division normalizes them. The mask excludes future keys and, with a window, keys
  # at least window positions before the query.
  # The cases cover several tile rows of queries, and a key and value head serving two query heads, with the repeat
  # read in place. They cover 256-wide heads that share their positions, a KV cache in bfp8, and keys that end partway
  # through a tile, with three tile rows of queries. Then come a sliding window, and ten tile rows in units of two.
  # After those, the keys of each unit are shared out among parts, like a decode step does. Its rows share a position,
  # and a 1,024-key bfp8 cache goes in four parts of 8 key tiles, the fewest that attention_parts gives a part. Some
  # rows see none of a part's keys, and with positions below 32, some parts have no keys at all.
  # The same kernels serve other lengths and row counts in the same tiles. Without parts it's two launches, the
  # attention and the division, and it reads only what the positions reach. Rows at positions 200 to 255 of 256 read
  # the positions tile of each unit, its query tiles, and its 8 tiles of keys and of values once, keeping the keys in
  # L1 between sweeps.
  rt, rng = Runtime(chip), np.random.default_rng(43)
  reads = lambda: sum(chip.read32(x, y, bh.NIU_READS) for x, y in rt.cores)
  for e, rows, hd, keys, rep, fmt, shared, window, lo, hi, parts in (
      (2, 40, 64, 64, 1, "bf16", False, 0, 0, 64, 1), (4, 8, 64, 96, 2, "bf16", False, 0, 0, 96, 1), (2, 33, 256, 128, 1, "bf16", True, 0, 0, 128, 1),
      (4, 8, 64, 64, 2, "bfp8", False, 0, 0, 64, 1), (3, 70, 64, 40, 1, "bf16", True, 0, 0, 40, 1),
      (2, 40, 128, 200, 1, "bf16", False, 50, 60, 200, 1), (2, 8, 64, 256, 1, "bf16", False, 0, 200, 256, 1),
      (2, 300, 64, 320, 2, "bf16", True, 0, 0, 320, 1), (4, 8, 64, 1024, 2, "bfp8", False, 0, 0, 1024, None),
      (2, 40, 128, 200, 1, "bf16", False, 50, 60, 200, 3), (4, 8, 64, 256, 1, "bf16", False, 0, 0, 30, 8),
      (2, 40, 64, 160, 1, "bf16", False, 0, 100, 160, 1), (2, 50, 64, 72, 1, "bf16", False, 0, 30, 72, 1)):
    q, k, v = (bf16(rng.normal(0, 1, shape)) for shape in ((e, rows, hd), (e // rep, keys, hd), (e // rep, keys, hd)))
    places = rng.integers(lo, hi, (1 if shared else e, rows, 1))
    if parts is None: places = places[:, :1, :].repeat(rows, 1)  # like a decode step: the query heads of a slot share a position
    mask = np.where((np.arange(keys) > places) | ((np.arange(keys) <= places - window) if window else False), np.float32(-1e9), np.float32(0))
    tq, tk, tv = (Tensor.of(rt, x.tolist()) for x in (q, k, v))
    if fmt != "bf16": tk, tv = (x.cast(fmt).realize() for x in (tk, tv)); k, v = value(tk), value(tv)  # as the chip holds them
    if rep > 1: tk, tv = tk.repeat_interleave(rep), tv.repeat_interleave(rep)
    tp = Tensor.of(rt, places.tolist(), "int32")
    rt.wait()
    before, launches, made = reads(), rt.launches, len(tensor.kernels)
    out = tq.attend(tk, tv, tp, window, parts=parts).realize()
    rt.wait()
    assert (rows, keys) not in ((40, 160), (50, 72)) or len(tensor.kernels) == made, "other lengths and rows reuse the first case's kernels"
    read, got = reads() - before, value(out)
    s = q @ np.repeat(k, rep, 0).transpose(0, 2, 1) + mask
    p = np.exp(s - s.max(-1, keepdims=True))
    close(got, p / p.sum(-1, keepdims=True) @ np.repeat(v, rep, 0), 0.011)
    assert parts != 1 or kernels(rt, launches) == 2
    # 2 units of 1 + 2 + 8 x 2 + 8 x 2 tiles. Then the division has 4 result tiles, each reading a tile of p v and
    # one of the sums.
    if lo == 200:
      assert read == 2 * (1 + 2 + 16 + 16) + 8, read

def test_attention_unit_rows(chip):
  # With one worker, a unit takes up to 8 tile rows of queries, which share each key and value tile it reads. The
  # rows' largest scores share Dst's two accumulator tiles, a quarter of a tile each. The cases cover 8 rows, a last
  # unit of 3, a sliding window over a bfp8 cache, a repeat, a unit of 3 rows in parts, and scores whose maximum
  # bf16 can't hold.
  rt, rng = Runtime(chip, end=chip.tensix[1]), np.random.default_rng(44)
  for e, rows, hd, keys, rep, fmt, window, parts, R in ((1, 256, 64, 256, 1, "bf16", 0, None, 8), (2, 330, 64, 320, 2, "bf16", 0, None, 8),
                                                         (1, 100, 128, 200, 1, "bfp8", 50, None, 4), (1, 96, 64, 128, 1, "bf16", 0, 2, 3)):
    q, k, v = (bf16(rng.normal(0, 1, shape)) for shape in ((e, rows, hd), (e // rep, keys, hd), (e // rep, keys, hd)))
    places = np.clip(np.arange(rows) + keys - rows + rng.integers(-8, 8, rows), 0, keys - 1)[None, :, None].repeat(e, 0)
    mask = np.where((np.arange(keys) > places) | ((np.arange(keys) <= places - window) if window else False), np.float32(-1e9), np.float32(0))
    tq, tk, tv = (Tensor.of(rt, x.tolist()) for x in (q, k, v))
    if fmt != "bf16": tk, tv = (x.cast(fmt).realize() for x in (tk, tv)); k, v = value(tk), value(tv)
    if rep > 1: tk, tv = tk.repeat_interleave(rep), tv.repeat_interleave(rep)
    got = value(tq.attend(tk, tv, Tensor.of(rt, places.tolist(), "int32"), window, parts=parts))
    s = q @ np.repeat(k, rep, 0).transpose(0, 2, 1) + mask
    p = np.exp(s - s.max(-1, keepdims=True))
    close(got, p / p.sum(-1, keepdims=True) @ np.repeat(v, rep, 0), 0.011)
    assert any(key[:4] == ("attend", hd // 32, -(-rows // 32), R) for key in tensor.kernels), "a unit of R tile rows"
  q = Tensor.of(rt, [[10000] * 64] * 256)
  k = Tensor.of(rt, [[10000] * 64] * 64)
  v = Tensor.of(rt, [[1] * 64] * 32 + [[5] * 64] * 32)
  assert np.array_equal(value(q.attend(k, v, Tensor.of(rt, [[31]] * 256, "int32"))), np.ones((1, 256, 64)))

def test_attention_extremes(chip):
  rt = Runtime(chip)
  # Both signs exceed the old bf16 maximum's precision. Empty parts and masked keys must contribute no weight,
  # including when a finite score exceeds the old -1e9 mask.
  for rows, query, key, parts in ((1, 100, 100, 1), (1, 100, 100, 2), (1, 100, -100, 2), (70, 10000, 10000, 1), (40, 10000, -10000, 4)):
    q = Tensor.of(rt, [[query] * 64] * rows)
    k = Tensor.of(rt, [[key] * 64] * 64)
    v = Tensor.of(rt, [[1] * 64] * 32 + [[5] * 64] * 32)
    pos = Tensor.of(rt, [[31]] * rows, "int32")
    assert np.array_equal(value(q.attend(k, v, pos, parts=parts)), np.ones((1, rows, 64)))
  q = Tensor.of(rt, [[100] * 64])
  k = Tensor.of(rt, [[100] * 64] * 32 + [[101] * 64] * 32)
  v = Tensor.of(rt, [[1] * 64] * 32 + [[2] * 64] * 32)
  for parts in (1, 2):
    assert np.array_equal(value(q.attend(k, v, Tensor.of(rt, [[63]], "int32"), parts=parts)), np.full((1, 1, 64), 2))

def test_attention_capacity(chip):
  from baretile.kernels import attention_parts, attention_store
  rt = Runtime(chip)
  for fmt in ("bf16", "bfp8", "bfp4"):
    q, k, v = Tensor(rt, 1, 1536), Tensor(rt, 32, 1536, dtype=fmt), Tensor(rt, 32, 1536, dtype=fmt)
    pos = Tensor.of(rt, [[31]], "int32")
    before, used = rt.launches, dict(rt.used)
    for parts in (None, 1, 2):
      with pytest.raises(ValueError, match="fit in L1"): q.attend(k, v, pos, parts=parts)
    assert rt.launches == before and rt.used == used, "reject before allocating or submitting an output"
  assert attention_store(42, 1, None, 1)[1] == 0
  assert attention_parts(42, 1, None, 2, 1, len(rt.cores)) == 1
  q, k = Tensor.of(rt, [[0] * 1344]), Tensor.of(rt, [[0] * 1344] * 32)
  v = Tensor.of(rt, [[1] * 1344] * 32)
  assert np.array_equal(value(q.attend(k, v, Tensor.of(rt, [[31]], "int32"))), np.ones((1, 1, 1344)))

def test_reduce_axes(chip):
  # sum and max run down the rows, through the batch and along each row, and each is one launch. The padding past the
  # last tile row is masked. The values are negative, so an unmasked 0 would win a max, and padding rows would add into
  # a sum. Then the results broadcast back in a column softmax. Last come a mean over everything and a batch sum fused
  # with what reads it. It runs on one worker, where a column of tiles cycles through the slots, and on many.
  rng = np.random.default_rng(6)
  for workers in (chip.tensix[1], chip.tensix[-1]):
    rt = Runtime(chip, end=workers)
    x = bf16(rng.uniform(-6, -1, (3, 45, 70)))
    tx = Tensor.of(rt, x.tolist())
    for axis in (0, 1, 2):
      launches, top, total = rt.launches, value(tx.max(axis)), value(tx.sum(axis))
      assert kernels(rt, launches) == 2 and top.tolist() == x.max(axis, keepdims=True).tolist()
      assert np.allclose(total, x.sum(axis, keepdims=True), rtol=2**-8, atol=2**-8), np.abs(total - x.sum(axis, keepdims=True)).max()
    e = np.exp(x - x.max(1, keepdims=True))
    close((tx - tx.max(1)).exp() / (tx - tx.max(1)).exp().sum(1), e / e.sum(1, keepdims=True), 0.02)
    launches, got = rt.launches, tx.sum() * (1 / x.size)
    assert abs(got.tolist()[0][0] - x.mean()) <= 2**-7 * abs(x.mean()) and kernels(rt, launches) == 3, "one launch per axis"
    launches = rt.launches
    close(tx.sum(0) * tx[1] - 1.0, x.sum(0) * x[1] - 1, 2**-6)  # the batch sum, a view and a number in one kernel
    assert kernels(rt, launches) == 1, "fused"
    y = bf16(rng.uniform(-2, 2, (2, 45, 200)))  # reductions of another length than the result along the axis run first, alone
    ty = Tensor.of(rt, y.tolist())
    close(tx[0:1, :, :].max(1) * ty[:, :, :70] + 1.0, x[0:1].max(1, keepdims=True) * y[:, :, :70] + 1, 2**-6)
    close(tx.max(0) * ty[:, :, :70], x.max(0, keepdims=True) * y[:, :, :70], 2**-7)
    close(tx[0].sum(2) + ty[1], x[0].sum(1, keepdims=True) + y[1], 2**-6)
    narrow = bf16(rng.uniform(-3, -1, (20, 70)))  # one tile tall: reduced in a kernel of its own, and inside a taller kernel
    tnarrow = Tensor.of(rt, narrow.tolist())
    assert tnarrow.max(1).tolist() == narrow.max(0, keepdims=True).tolist()
    close(tnarrow.sum(1) + tx[2], narrow.sum(0, keepdims=True) + x[2], 2**-6)

def test_any_shape(chip):
  # 45 x 70 is two tile rows by three tile columns, the last of each cut short, and a dimension of 1 broadcasts.
  # Results are written back with zero padding, which a matmul of two results needs, or the padding of K adds in. Row
  # reductions mask the padding of their last tile. The values are negative, so an unmasked 0 would be the max and
  # would add up in softmax. A kernel serves any row count within the same tiles, like a prompt's chunks of any
  # length, taking the rows of its last tile as an argument.
  rt, rng = Runtime(chip), np.random.default_rng(0)
  def make(rows, cols, lo=-2.0, hi=2.0):
    a = bf16(rng.uniform(lo, hi, (rows, cols)))
    return a, Tensor.of(rt, a.tolist())
  (x, tx), (y, ty), (col, tcol), (row, trow) = make(45, 70), make(45, 70), make(45, 1), make(1, 70)
  assert (tx * ty).tolist() == bf16(x * y, True).tolist() and (tcol @ trow).tolist() == bf16(col * row, True).tolist()
  close(tx + ty, x + y, 2**-7)
  close(tx * trow + tcol - 2.5, x * row + col - 2.5, 2**-6)
  neg, tneg = make(45, 70, -6, -1)
  assert tneg.max(2).tolist() == neg.max(1, keepdims=True).tolist()
  close(tneg.sum(2), neg.sum(1, keepdims=True), 2**-8)
  e = np.exp(neg - neg.max(1, keepdims=True))
  close((tneg - tneg.max(2)).exp() / (tneg - tneg.max(2)).exp().sum(2), e / e.sum(1, keepdims=True), 0.02)
  narrow, tnarrow = make(45, 20, -3, -1)  # one tile wide: reduced in a kernel of its own, and inside a three-tile-wide kernel
  assert tnarrow.max(2).tolist() == narrow.max(1, keepdims=True).tolist()
  close(tnarrow.sum(2) + tx, narrow.sum(1, keepdims=True) + x, 2**-6)
  (a, ta), (b, tb) = make(45, 50, -1, 1), make(50, 70, -1, 1)
  close(ta.exp() @ tb.exp(), bf16(np.exp(a)) @ bf16(np.exp(b)), 2**-6)
  for n in (37, 50, 64):
    z, tz = make(n, 70)
    made, out = len(tensor.kernels), (tz * 2.0 - 1.0).realize()
    whole = value(out._view(64, 70, 1, 0))  # including the rows of the last tile past n
    assert n == 37 or len(tensor.kernels) == made, "later row counts reuse the first one's kernel"
    close(whole[0, :n], z * 2 - 1, 2**-7)
    assert not whole[0, n:].any(), n

def test_views_transpose_gelu(chip):
  # Views are read in place: tile-aligned slices as inputs to elementwise ops and matmuls, like the 64-column attention
  # heads of a GPT-2 layer. Transpose is exact on odd shapes and views, and turns a filled row into a filled column. A
  # repeat of heads, as in grouped-query attention, reads the right heads when indexed, sliced and repeated again. gelu
  # is checked against numpy within the bf16 rounding of its steps.
  rt, rng = Runtime(chip), np.random.default_rng(1)
  def make(rows, cols, lo=-2.0, hi=2.0):
    a = bf16(rng.uniform(lo, hi, (rows, cols)))
    return a, Tensor.of(rt, a.tolist())
  (x, tx), (y, ty), (row, trow) = make(70, 100), make(70, 192), make(1, 100)
  v = tx[32:70, 64:100]
  assert v.tolist() == x[32:70, 64:100].tolist() and (v * 2.0).tolist() == (x[32:70, 64:100] * 2).tolist()
  assert tx.T.tolist() == x.T.tolist() and v.T.tolist() == x[32:70, 64:100].T.tolist()
  close(tx.T + trow.T, x.T + row.T, 2**-7)
  q, k = ty[:, 64:128], ty[:, 128:192]  # a head's query and key
  close(q @ k.T, y[:, 64:128] @ y[:, 128:192].T, 2**-6)
  assert tx.gather([5, 0, 69, 37, 5]).tolist() == x[[5, 0, 69, 37, 5]].tolist()  # rows from both tile rows, one twice
  close(tx + tx.gather([7]), x + x[7], 2**-7)  # a single row fills its tile, so it broadcasts
  heads = ty[:, 0:128].split(2).repeat_interleave(2)  # heads 0, 0, 1, 1
  assert heads[1].tolist() == y[:, 0:64].tolist() and heads[2].tolist() == y[:, 64:128].tolist()
  assert heads[1:3, :, :].tolist() == [y[:, 0:64].tolist(), y[:, 64:128].tolist()]
  assert heads.repeat_interleave(2)[5].tolist() == y[:, 64:128].tolist()  # heads 0, 0, 0, 0, 1, 1, 1, 1
  close(tx.gelu(), gelu(x), 0.02)

def test_batch(chip):
  # This checks a batch of matrices against numpy. Elementwise ops broadcast over it, and a batch of 1 repeats. Row
  # reductions and softmax work per matrix, and matmul pairs the matrices up or multiplies the batch by one matrix.
  # Views split a matrix into a batch of heads and merge it back. The 40 rows end partway through a tile, on purpose.
  rt, rng = Runtime(chip), np.random.default_rng(4)
  def make(*shape):
    a = bf16(rng.uniform(-2, 2, shape))
    return a, Tensor.of(rt, a.tolist())
  (x, tx), (y, ty), (row, trow) = make(3, 40, 70), make(1, 40, 70), make(1, 70)
  assert tx.shape == (3, 40, 70) and ty.shape == (1, 40, 70) and np.array(tx.tolist()).tolist() == x.tolist()
  assert (tx * ty).tolist() == bf16(x * y, True).tolist() and (tx[1] * 2.0).tolist() == (x[1] * 2).tolist()
  close(tx - trow + 1.0, x - row + 1, 2**-6)
  assert tx.max(2).tolist() == x.max(2, keepdims=True).tolist()
  e = np.exp(x - x.max(2, keepdims=True))
  close((tx - tx.max(2)).exp() / (tx - tx.max(2)).exp().sum(2), e / e.sum(2, keepdims=True), 0.02)
  (a, ta), (b, tb), (w, tw) = make(3, 40, 96), make(3, 96, 70), make(96, 70)
  close(ta @ tb, a @ b, 2**-6)
  close(ta @ tw + trow, a @ w + row, 2**-6)
  (qkv, tqkv), (kt, tkt) = make(40, 192), make(192, 40)  # 3 heads of 64: split columns, split rows
  heads, keys = tqkv.split(3), tkt.split(3, axis=1)
  assert heads.shape == (3, 40, 64) and np.array(heads.tolist()).tolist() == qkv.reshape(40, 3, 64).transpose(1, 0, 2).tolist()
  assert np.array(keys.tolist()).tolist() == kt.reshape(3, 64, 40).tolist()
  close(heads @ keys, qkv.reshape(40, 3, 64).transpose(1, 0, 2) @ kt.reshape(3, 64, 40), 2**-6)
  assert np.array(heads.T.tolist()).tolist() == qkv.reshape(40, 3, 64).transpose(1, 2, 0).tolist()  # each matrix, transposed
  h = qkv.reshape(40, 3, 64).transpose(1, 0, 2)
  close(heads @ heads.T, h @ h.transpose(0, 2, 1), 2**-6)  # B transposed as the matmul reads it, no transpose kernel
  (c, tc), (d, td) = make(96, 64), make(160, 64)
  close(tc @ td.T, c @ d.T, 2**-6)  # and in blocks, with B's column kept in L1
  assert heads.merge().tolist() == qkv.tolist()  # side by side again: here a copy, and below a kernel writing into its columns
  (m, tm) = make(64, 64)
  close((heads @ tm).merge(), (qkv.reshape(40, 3, 64) @ m).reshape(40, 192), 2**-6)
  # This goes last, since it takes two of rt's cores. On one worker, with a tile row per matrix, the row of A kept in
  # L1 changes with the matrix.
  one, a = Runtime(chip, end=chip.tensix[1]), a[:, :20]
  close(Tensor.of(one, a.tolist()) @ Tensor.of(one, b.tolist()), a @ b, 2**-6)

def test_movement(chip):
  # reshape, permute, expand, pad and slices at any offset and step match numpy bit for bit, since a move only copies.
  # Tile-aligned slices stay views, and so do slices of a batch at any step. The other moves are one launch, and so is
  # a chain of them. A moved tensor is written back with clean padding, so a matmul and a reduction can read it. Then
  # come random chains of moves on one worker, whose cache of tiles cycles.
  rng = np.random.default_rng(7)
  rt = Runtime(chip)
  x = bf16(rng.uniform(-2, 2, (3, 45, 70)))
  tx = Tensor.of(rt, x.tolist())
  cases = [(tx.reshape(5, 27, 70), x.reshape(5, 27, 70)), (tx.reshape(63, 150), x.reshape(63, 150)),
           *[(tx.permute(*o), x.transpose(o)) for o in [(1, 0, 2), (2, 1, 0), (1, 2, 0), (2, 0, 1), (0, 2, 1)]],
           (tx[1:3, 5:40:3, ::-1], x[1:3, 5:40:3, ::-1]), (tx[:, 3:4], x[:, :, 3:4]), (tx[2:3, 44:45, :], x[2:3, 44:45, :]),
           (tx[0:1, 7:8, :].expand(3, 45, 70), np.broadcast_to(x[0:1, 7:8, :], (3, 45, 70))),
           (tx.pad(((3, 1), (40, 2))), np.pad(x, ((0, 0), (3, 1), (40, 2))))]
  for t, want in cases: assert value(t).tolist() == want.reshape(t.shape).tolist(), (t.shape, want.shape)
  launches, t = rt.launches, tx.reshape(135, 70)[:, 7:100:2, ::-1].pad(((1, 0), (3, 1), (2, 5))).permute(2, 1, 0)[:, 1:4, 1:2].expand(77, 3, 40)
  want = np.broadcast_to(np.pad(x.reshape(1, 135, 70)[:, 7:100:2, ::-1], ((1, 0), (3, 1), (2, 5))).transpose(2, 1, 0)[:, 1:4, 1:2], (77, 3, 40))
  assert value(t).tolist() == want.tolist() and kernels(rt, launches) == 1, "a chain of moves is one move"
  every = Tensor.of(rt, bf16(rng.uniform(-2, 2, (7, 32, 64))).tolist())  # a batch slice with a step is a view of every 3rd matrix
  launches, t = rt.launches, every[1::3, :, :] * 2
  t.realize()
  assert value(t).tolist() == (value(every)[1::3] * 2).tolist() and kernels(rt, launches) == 1, "read in place"
  # The paths of the move kernel, in fp32 so that a datum is a word. It copies tiles as they are, for a pad on tile
  # boundaries, or transposed. It copies runs along x's rows for a reshape or an unaligned slice, runs down its columns
  # for splitting attention heads, and datum by datum for a permute.
  y = rng.uniform(-2, 2, (2, 64, 96)).astype(np.float32)
  ty = Tensor.of(rt, y.tolist(), "fp32")
  for t, want in [(ty.T, y.transpose(0, 2, 1)), (ty.pad(((0, 0), (32, 0), (0, 64))), np.pad(y, ((0, 0), (32, 0), (0, 64)))),
                  (ty.reshape(4, 48, 64), y.reshape(4, 48, 64)), (ty[:, 3:50, 5:90], y[:, 3:50, 5:90]),
                  (ty[:, :45, :70].T, y[:, :45, :70].transpose(0, 2, 1)),
                  (ty.reshape(1, 128, 96).move((16, 96, 8), 0, (768, 1, 96)), y.reshape(16, 8, 96).transpose(0, 2, 1)),
                  (ty.permute(2, 0, 1), y.transpose(2, 0, 1))]:
    assert value(t).tolist() == want.tolist(), t.shape
  launches, v = rt.launches, tx[1:3, 32:45, 32:64]
  assert rt.launches == launches and v.addr == tx.addr and value(v).tolist() == x[1:3, 32:45, 32:64].tolist(), "a view, no copy"
  w = bf16(rng.uniform(-1, 1, (50, 20)))
  close(tx[0][5:40, 10:60] @ Tensor.of(rt, w.tolist()), x[0, 5:40, 10:60] @ w, 2**-6)
  close(tx[:, 3:50, 5:69].sum(1) * (1 / 47), x[:, 3:50, 5:69].sum(1, keepdims=True) / 47, 2**-7)
  one = Runtime(chip, end=chip.tensix[1])
  def move(t, a):  # applies the same random move to t and to numpy's a, keeping batches small
    b, r, c = a.shape
    kind = rng.choice(["reshape", "permute", "slice", "pad", "expand" if 1 in a.shape else "slice"])
    if kind == "reshape":
      n = int(rng.choice([d for d in range(1, min(b * r * c, 200) + 1) if b * r * c % d == 0]))
      nb = int(rng.choice([d for d in range(1, 4) if b * r * c // n % d == 0]))
      shape = (nb, b * r * c // n // nb, n)
      return t.reshape(*shape), a.reshape(shape)
    if kind == "permute":
      o = tuple(int(k) for k in rng.permutation(3))
      while a.shape[o[0]] > 4: o = tuple(int(k) for k in rng.permutation(3))
      return t.permute(*o), a.transpose(o)
    if kind == "pad":
      p = [(int(rng.integers(0, 2)), int(rng.integers(0, 2)))] + [(int(rng.integers(0, 20)), int(rng.integers(0, 20))) for _ in range(2)]
      return t.pad(p), np.pad(a, p)
    if kind == "expand":
      shape = tuple(n if n > 1 else int(rng.integers(2, 4)) if ax == 0 else int(rng.choice([2, 33])) for ax, n in enumerate(a.shape))
      return t.expand(*shape), np.broadcast_to(a, shape)
    key = []
    for n in a.shape:
      lo = int(rng.integers(0, n))
      hi = int(rng.integers(lo + 1, n + 1))
      step = int(rng.choice([1, 1, 2, 3, -1]))
      key.append(slice(hi - 1, lo - 1 if lo else None, step) if step < 0 else slice(lo, hi, step))
    return t[tuple(key)], a[tuple(key)]
  for _ in range(8):
    shape = (int(rng.choice([1, 2, 3])), int(rng.choice([1, 20, 45])), int(rng.choice([1, 20, 70])))
    a = bf16(rng.uniform(-2, 2, shape))
    t = Tensor.of(one, a.tolist() if shape[0] > 1 else a[0].tolist())
    for _ in range(3):
      t, a = move(t, a)
      assert value(t).tolist() == a.reshape(t.shape).tolist(), a.shape

def test_leading_movement(chip):
  # Moves over leading dimensions match numpy bit for bit: permutes, expands, pads, cat, reshapes that change the
  # matrices, and slices of any leading dimension, in four and five dimensions. Leading slices that pick evenly spaced
  # matrices are views, and a chain of moves is one move. A gradient comes back through the moves.
  rng, rt = np.random.default_rng(39), Runtime(chip)
  x = bf16(rng.uniform(-2, 2, (2, 3, 5, 40)))
  tx, f = Tensor.of(rt, x.tolist()), x.reshape(3, 1, 2, 5, 40)
  tf = tx.reshape(*f.shape)
  cases = [(tx.permute(1, 0, 2, 3), x.transpose(1, 0, 2, 3)), (tx.permute(3, 1, 0, 2), x.transpose(3, 1, 0, 2)),
           (tf.permute(2, 4, 0, 3, 1), f.transpose(2, 4, 0, 3, 1)), (tx[:, 0:2, :, :], x[:, 0:2]),
           (Tensor.of(rt, x.tolist(), "fp32").permute(1, 0, 3, 2), x.transpose(1, 0, 3, 2)),
           (tx[::-1, ::2, 1:4, ::-1], x[::-1, ::2, 1:4, ::-1]), (tf[1:, :, ::-1, :, 8:], f[1:, :, ::-1, :, 8:]),
           (tx[0:1, :, :, :].expand(2, 3, 5, 40), np.broadcast_to(x[0:1], x.shape)),
           (tx[:, 0:1, :, :].expand(2, 3, 5, 40), np.broadcast_to(x[:, 0:1], x.shape)),
           (Tensor.of(rt, x[0].tolist()).expand(2, 3, 5, 40), np.broadcast_to(x[0], x.shape)),
           (tx.pad(((1, 0), (0, 2), (3, 1), (0, 0))), np.pad(x, ((1, 0), (0, 2), (3, 1), (0, 0)))),
           (tx.cat(tx[:, 0:2, :, :], axis=1), np.concatenate((x, x[:, 0:2]), 1)),
           (tx.cat(tx, axis=0), np.concatenate((x, x))), (tx.reshape(2, 5, 3, 40), x.reshape(2, 5, 3, 40))]
  for t, want in cases: assert t.shape == want.shape and value(t).tolist() == want.tolist(), (t.shape, want.shape)
  launches, v = rt.launches, tx[:, 1:2, :, :]
  assert rt.launches == launches and v.addr == tx.addr and value(v).tolist() == x[:, 1:2].tolist(), "a view, no copy"
  launches, t = rt.launches, tx.permute(1, 0, 3, 2)[:, :, 2:9, ::-1].pad(((0, 0), (1, 1), (0, 0), (0, 0)))
  want = np.pad(x.transpose(1, 0, 3, 2)[:, :, 2:9, ::-1], ((0, 0), (1, 1), (0, 0), (0, 0)))
  assert value(t).tolist() == want.tolist() and kernels(rt, launches) == 1, "a chain of moves is one move"
  a = Tensor.of(rt, x.tolist(), "fp32").requires_grad_()
  g, grad = (a.permute(1, 0, 3, 2)[:, :, 2:9, ::-1].pad(((0, 0), (1, 1), (0, 0), (0, 0))) * 2).sum().gradient(a)[0], np.zeros_like(x)
  grad[:, :, :, 2:9] = 2
  assert g.shape == x.shape and value(g).tolist() == grad.tolist()
  with pytest.raises(ValueError, match="permute"): tx.permute(0, 1, 2)
  with pytest.raises(ValueError, match="move"): tx.move((1, 1, 1, 1, 1, 1), 0, (0,) * 6)

def test_leading_dimensions(chip):
  # A tensor takes up to five dimensions, and the leading ones fold into the batch in row-major order, so (2, 3, 5, 40)
  # is stored as (6, 5, 40). A reshape of the leading sizes alone is then a view, with no launch, and an unrealized one
  # fuses into the kernel that reads it. Elementwise ops, reductions, matmuls, transposes and slices keep the leading
  # shape, an operand whose leading sizes are all 1 broadcasts, and a gradient comes back through the views to each
  # input's shape.
  rng, rt = np.random.default_rng(38), Runtime(chip)
  x, y = (bf16(rng.uniform(-2, 2, (2, 3, 5, 40))) for _ in range(2))
  tx, ty = Tensor.of(rt, x.tolist()), Tensor.of(rt, y.tolist())
  assert tx.shape == (2, 3, 5, 40) and tx.batch == 6 and tx.tolist() == x.tolist(), "nested as numpy nests it"
  launches, flat = rt.launches, tx.reshape(6, 5, 40)
  assert flat.shape == (6, 5, 40) and flat.addr == tx.addr and rt.launches == launches, "a view of the same matrices"
  five = flat.reshape(3, 1, 2, 5, 40)
  assert five.shape == (3, 1, 2, 5, 40) and five.tolist() == x.reshape(3, 1, 2, 5, 40).tolist()
  assert value(Tensor.of(rt, x.reshape(1, 2, 3, 5, 40).tolist())).tolist() == x.reshape(1, 2, 3, 5, 40).tolist()
  launches, t = rt.launches, ((tx + ty).reshape(6, 5, 40) * 2).realize()
  assert kernels(rt, launches) == 1 and t.shape == (6, 5, 40), "the reshape reads through to the sum, in one kernel"
  close(t, ((x + y) * 2).reshape(6, 5, 40), 2**-5)
  row, matrix = bf16(rng.uniform(-2, 2, (2, 3, 1, 40))), bf16(rng.uniform(-2, 2, (5, 40)))
  close(tx * ty - Tensor.of(rt, row.tolist()), x * y - row, 2**-5)
  close((tx + Tensor.of(rt, matrix.tolist())).exp(), np.exp(x + matrix), 2**-5)
  close(tx.sum(-1), x.sum(-1, keepdims=True), 2**-6)
  close(tx.max(-2), x.max(-2, keepdims=True), 0)
  close(tx.softmax(-1), np.exp(x) / np.exp(x).sum(-1, keepdims=True), 2**-6)
  close(tx[1:2, :, :, :].sum(1), x[1:2].sum(1, keepdims=True), 2**-6)
  w, v = bf16(rng.uniform(-1, 1, (2, 3, 40, 8))), bf16(rng.uniform(-1, 1, (40, 8)))
  close(tx @ Tensor.of(rt, w.tolist()), x @ w, 2**-6)
  close(tx @ Tensor.of(rt, v.tolist()), x @ v, 2**-6)
  column = x.reshape(6, 1, 5, 40)  # a leading size of 6 over sizes of 1, which a slice can step through
  for t, want in [(tx[1], x[1]), (tx[-2:-1, :, :, :], x[-2:-1]), (tx.reshape(6, 1, 5, 40)[::2, :, :, :], column[::2]),
                  (tx[1:2, :, 1:4, 8:40], x[1:2, :, 1:4, 8:40]), (tx.T, x.transpose(0, 1, 3, 2)),
                  (tx.reshape(3, 2, 8, 25), x.reshape(3, 2, 8, 25))]:
    assert t.shape == want.shape and value(t).tolist() == want.tolist(), (t.shape, want.shape)
  with pytest.raises(ValueError, match="dimensions"): tx.reshape(1, 2, 1, 3, 5, 40)
  with pytest.raises(ValueError, match="broadcasting"): tx + Tensor.of(rt, x[:1].tolist())
  with pytest.raises(ValueError, match="leading"): tx.sum(1)
  with pytest.raises(ValueError, match="matmul"): tx @ Tensor.of(rt, w[:1].tolist())
  with pytest.raises(ValueError, match="three dimensions"): tx.split(2)
  out = Tensor.zeros(rt, 8, 25, 6).reshape(2, 3, 8, 25)
  assert value(out.assign(tx.reshape(2, 3, 8, 25))).tolist() == x.reshape(2, 3, 8, 25).tolist()
  double = tensor.jit(lambda t: t * 2)  # run, recorded, then replayed, each with the leading shape
  for t, want in ((tx, x), (ty, y), (tx, x)): assert value(double(t)).tolist() == (2 * want).tolist()
  a = Tensor.of(rt, x.tolist(), "fp32").requires_grad_()
  b = Tensor.of(rt, matrix.tolist(), "fp32").requires_grad_()
  close(a.sum(), x.sum(keepdims=True), 1e-5)
  assert bool(a.sum()) and not Tensor.of(rt, [[[[0.0]]]])
  assert value(a.sum().gradient(a)[0]).tolist() == np.ones_like(x).tolist()
  ga, gb = (a * a * b).sum().gradient(a, b)  # b is one matrix for all six, so its gradient sums over them
  assert ga.shape == x.shape and value(ga).tolist() == (2 * x * matrix).tolist()
  close(gb, (x * x).sum((0, 1)), 1e-5)

def test_fp32(chip):
  # fp32 tensors have 4 KiB tiles. The math thread unpacks them into Dst, bypassing SrcA and SrcB, and the SFPU does
  # the math. add, sub, mul, comparisons, where, movement and max match numpy's float32 bit for bit, and sums are
  # within its rounding. exp, log, sin and the like are within a step or two of fp32. Casts work both ways, and to
  # bf16 rounds like the packer. bf16 with fp32 gives fp32, and a number takes its tensor's dtype, so a mean of 768
  # values is exact.
  # A matmul sums in fp32, but the FPU adds each group of 8 products at 12 bits (ttsim's model of it). That leaves fp32
  # results good to about 2^-11, and fp32 operands, which go in as bf16x3, reach that too, against bf16's 2^-8. It
  # runs on one worker, where the pipeline cycles, and on many.
  rng = np.random.default_rng(9)
  f32 = lambda *shape: rng.uniform(-2, 2, shape).astype(np.float32)
  for workers in (chip.tensix[1], chip.tensix[-1]):
    rt = Runtime(chip, end=workers)
    x, y = f32(2, 45, 70), f32(2, 45, 70)
    tx, ty = Tensor.of(rt, x.tolist(), "fp32"), Tensor.of(rt, y.tolist(), "fp32")
    launches, got = rt.launches, value((tx + ty) * tx - ty * 1.5)
    assert kernels(rt, launches) == 1 and got.tolist() == ((x + y) * x - y * np.float32(1.5)).tolist(), "one kernel, fp32's rounding"
    for got, want in [(tx, x), ((tx < ty).where(tx, ty * 0.5), np.where(x < y, x, y * np.float32(0.5))), (tx.max(1), x.max(1, keepdims=True)),
                      (tx.T, x.transpose(0, 2, 1)), (tx[1][3:40:2, 5:66], x[1, 3:40:2, 5:66]), (tx[0].gather([44, 0, 7]), x[0, [44, 0, 7]])]:
      assert value(got).tolist() == np.asarray(want, np.float32).reshape(got.shape).tolist()
    assert np.allclose(value(tx.sum(2)), x.sum(2, keepdims=True), rtol=1e-6, atol=1e-5)
    # The transcendentals are within a step or two of fp32 (2^-23 is 1.2e-7). The error is relative, but absolute for
    # sin and cos.
    w, v = np.float64(x * np.float32(4)), np.float64(x * x + np.float32(0.5))
    for got, want, scale in [((tx * 4.0).exp(), np.exp(w), np.exp(w)), ((tx * 4.0).exp2(), np.exp2(w), np.exp2(w)),
                             ((tx * tx + 0.5).log(), np.log(v), np.abs(np.log(v))), ((tx * tx + 0.5).log2(), np.log2(v), np.abs(np.log2(v))),
                             ((tx * tx + 0.5).sqrt(), np.sqrt(v), np.sqrt(v)), ((tx * tx + 0.5).rsqrt(), 1 / np.sqrt(v), 1 / np.sqrt(v)),
                             ((tx * 32.0).sin(), np.sin(w * 8), 1), ((tx * 32.0).cos(), np.cos(w * 8), 1)]:
      assert (np.abs(value(got) - want) / scale).max() <= 3e-7, (np.abs(value(got) - want) / scale).max()
    s = np.array([[np.nan, -np.nan, np.inf, -np.inf, 0, -2, 2]], np.float32)  # NaN of either sign passes through and equals nothing
    ts = Tensor.of(rt, s.tolist(), "fp32")
    with np.errstate(all="ignore"):
      for got, want in [(ts.exp(), np.exp(s)), (ts.exp2(), np.exp2(s)), (ts.log(), np.log(s)), (ts.sqrt(), np.sqrt(s)),
                        (ts.reciprocal(), 1 / s), (ts.sin(), np.sin(s)), (ts.ne(ts), s != s), (ts < ts * 2, s < s * 2)]:
        assert np.allclose(value(got), np.asarray(want, np.float32).reshape(got.shape), rtol=1e-6, atol=0, equal_nan=True), (value(got), want)
    b = bf16(x)
    assert value(tx.bfloat16()).tolist() == bf16(x, True).tolist() and value(Tensor.of(rt, b.tolist()).float()).tolist() == b.tolist()
    assert value(Tensor.of(rt, b.tolist()) + tx).tolist() == (b.astype(np.float32) + x).tolist(), "bf16 with fp32 is fp32"
    assert value(Tensor.of(rt, np.full((1, 768), 3.0).tolist(), "fp32").mean()).tolist() == [[[3.0]]]
    a, w = f32(64, 96), f32(96, 70)
    ta, tw = Tensor.of(rt, a.tolist(), "fp32"), Tensor.of(rt, w.tolist(), "fp32")
    want = bf16(a) @ bf16(w)
    assert np.abs(value(Tensor.of(rt, bf16(a).tolist()).matmul(Tensor.of(rt, bf16(w).tolist()), "fp32")) - want).max() <= 2**-11 * np.abs(want).max()
    want = a.astype(np.float64) @ w
    assert np.abs(value(ta @ tw) - want).max() <= 2**-11 * np.abs(want).max()

def test_int32(chip):
  # int32 tensors stay two's complement through L1, Dst and the SFPU. Values move raw, stored as INT32, since the FP32
  # store flushes denormals, and small positive integers look like denormals. add and sub wrap like int32. max, min,
  # comparisons (1 or 0) and where, whose condition holds where it isn't 0, negatives too, are exact over the full
  # int32 range. Multiplies and sums go through fp32, exact below 2^24. Casts work both ways, and to an integer rounds
  # toward zero. A float number makes fp32, and a float condition picks ints. gather reads its indices from an int32
  # tensor on the chip, so they can be computed there.
  rng = np.random.default_rng(10)
  wrap = lambda a: (a + 2**31) % 2**32 - 2**31
  for workers in (chip.tensix[1], chip.tensix[-1]):
    rt = Runtime(chip, end=workers)
    x, y, s = rng.integers(-2**31 + 1, 2**31, (45, 70)), rng.integers(-2**31 + 1, 2**31, (45, 70)), rng.integers(-99, 100, (45, 70))
    x[0, :5], s[0, :3] = [0, -1, 1, 2**31 - 1, -2**31 + 1], [3, 0, -3]
    tx, ty, ts = (Tensor.of(rt, a.tolist(), "int32") for a in (x, y, s))
    f = rng.uniform(-2**30, 2**30, (45, 70)).astype(np.float32)
    f[0, :6] = [0.0, -0.0, 0.7, -0.7, 2.5, -1e9]
    tf = Tensor.of(rt, f.tolist(), "fp32")
    for got, want in [(tx, x), (tx + ty, wrap(x + y)), (tx - ty, wrap(x - y)), (tx.maximum(ty), np.maximum(x, y)), (tx.minimum(ty), np.minimum(x, y)),
                      (tx < ty, x < y), (tx != ts, x != s), (ts == 3, s == 3), (ts >= 0, s >= 0), ((ts > 0).where(tx, ty), np.where(s > 0, x, y)),
                      (ts.where(tx, ty), np.where(s != 0, x, y)),
                      (ts * ts - 1, s * s - 1), (ts.sum(1), s.sum(0, keepdims=True)), (ts.max(), s.max(keepdims=True)), (ts.relu(), np.maximum(s, 0)),
                      (ts.float(), s), (ts.bfloat16(), s), (tf.cast("int32"), np.trunc(f)), ((tf > 0).where(ts, 0), np.where(f > 0, s, 0)),
                      (tf.gather(Tensor.of(rt, [[44, 0, 3, 44]], "int32")), f[[44, 0, 3, 44]])]:
      assert value(got).tolist() == np.asarray(want, np.float64).reshape(got.shape).tolist(), (got.shape, got.dtype)
    assert (ts * 0.5).dtype == "fp32" and (ts > 0).dtype == "int32" and value(ts * 0.5).tolist() == (s * 0.5).reshape(1, 45, 70).tolist()

def test_trig_range(chip):
  rt, rng = Runtime(chip), np.random.default_rng(22)
  # Cover both signs, all supported binades, and the old reduction's large-argument failures.
  x = np.concatenate([rng.uniform(-1, 1, 64).astype(np.float32) * np.float32(2**e) for e in range(25)])
  turns = (np.array([1, 2, 4095, 4096, 4097, 2**20, 2**22, 5340353]) * (math.pi / 2)).astype(np.float32)
  near = np.concatenate((np.nextafter(turns, -np.inf), turns, np.nextafter(turns, np.inf)))
  x = np.concatenate((x, near, -near, np.array([1e7, -1e7, 16777215, -16777215, 0, -0.0], np.float32))).reshape(1, -1)
  t = Tensor.frombytes(rt, *x.shape, x.tobytes(), dtype="fp32")
  for got, want in ((t.sin(), np.sin(x.astype(np.float64))), (t.cos(), np.cos(x.astype(np.float64)))):
    error = np.abs(value(got)[0] - want)
    assert np.max(error) < 5e-7, (x.ravel()[error.argmax()], error.max())
  edge = np.array([[2**24, -2**24, np.finfo(np.float32).max, np.inf, -np.inf, np.nan]], np.float32)
  t = Tensor.frombytes(rt, *edge.shape, edge.tobytes(), dtype="fp32")
  assert np.isnan(value(t.sin())).all() and np.isnan(value(t.cos())).all()
  zeros = Tensor.frombytes(rt, 1, 2, np.array([0.0, -0.0], np.float32).tobytes(), dtype="fp32")
  assert zeros.sin().tobytes() == zeros.tobytes()

def test_sqrt_range(chip):
  rt = Runtime(chip)
  # Every normal exponent, with exact powers of two and values near each binade's upper edge.
  bits = (np.arange(1, 255, dtype=np.uint32)[:, None] << 23) | np.array([0, 1, 0x123456, 0x400000, 0x7FFFFE, 0x7FFFFF], np.uint32)
  x = bits.view(np.float32)
  t = Tensor.frombytes(rt, *x.shape, x.tobytes(), dtype="fp32")
  got, want = value(t.sqrt())[0], np.sqrt(x.astype(np.float64))
  assert np.all(got > 0) and np.max(np.abs(got / want - 1)) < 3e-7
  edge = np.array([[0.0, -0.0, np.inf, -np.inf, np.nan, -1.0]], np.float32)
  got = np.frombuffer(Tensor.frombytes(rt, 1, edge.size, edge.tobytes(), dtype="fp32").sqrt().tobytes(), np.float32)
  with np.errstate(invalid="ignore"): want = np.sqrt(edge[0])
  assert np.array_equal(got, want, equal_nan=True)
  assert np.array_equal(np.signbit(got[:2]), np.signbit(want[:2]))

def test_integer_boundaries(chip):
  rt = Runtime(chip)
  edge = np.array([-2**31, -2**31 + 1, -2**24 - 1, -1, 0, 1, 2**24 + 1, 2**31 - 1], np.int32)
  x, y = edge[:, None], edge[None, :]
  a, b = (Tensor.of(rt, v.tolist(), "int32") for v in (x, y))
  for got, want in [(a < b, x < y), (a <= b, x <= y), (a > b, x > y), (a >= b, x >= y), (a == b, x == y), (a != b, x != y),
                    (a.maximum(b), np.maximum(x, y)), (a.minimum(b), np.minimum(x, y))]:
    assert np.array_equal(value(got)[0], want)
  assert np.array_equal(value(a.float())[0], x.astype(np.float32))
  assert np.array_equal(value(a.bfloat16())[0], bf16(x.astype(np.float32), True))

def test_selection_boundaries(chip):
  rt = Runtime(chip)
  x = np.array([[np.nan, -np.inf, -1, -0.0, 0.0, 1, np.inf]], np.float32)
  a, b = (Tensor.of(rt, v.tolist(), "fp32") for v in (x, x.T))
  for got, want in [(a <= b, x <= x.T), (a >= b, x >= x.T), (a <= 0, x <= 0), (0 <= a, 0 <= x), (a >= 0, x >= 0), (0 >= a, 0 >= x)]:
    assert np.array_equal(value(got)[0], want)
  x = np.array([[-0.0, 0.0, -np.inf, np.inf, -0.0, 3.0]], np.float32)
  y = np.array([[0.0, -0.0, np.inf, -np.inf, np.nan, -0.0]], np.float32)
  for cond in ([[1, 0, 1, 0, 1, 0]], [[0, 1, 0, 1, 0, 1]]):
    a, b = (Tensor.frombytes(rt, 1, 6, v.tobytes(), dtype="fp32") for v in (x, y))
    got = Tensor.of(rt, cond, "int32").where(a, b)
    want = np.where(cond, x, y)
    assert got.tobytes() == want.tobytes()
    with np.errstate(all="ignore"):
      assert np.array_equal(np.signbit(value(got.reciprocal())[0]), np.signbit(1 / want))
  c = np.array([[-np.nan, np.nan, -0.0, 0.0, -1e-45, np.inf]], np.float32)  # a NaN is true whatever its sign, as in numpy
  assert Tensor.of(rt, c.tolist(), "fp32").where(a, b).tobytes() == np.where(c, x, y).tobytes()

def test_autograd(chip):
  # This checks the ops' backward passes against numpy, on fp32 tensors with odd shapes and broadcasting. The loss is
  # sum(f(inputs) * w), with w random so that the elements of the gradient differ, and the reference gradient comes
  # from central differences in float64. Matmuls are within the FPU's 2^-11, the rest within fp32 rounding. Then come a
  # bf16 leaf, whose gradient comes back through the cast, a tensor read twice, whose gradients add, and .grad adding
  # up over two backward passes.
  rt, rng = Runtime(chip), np.random.default_rng(11)
  softmax = lambda a, ax: np.exp(a - a.max(ax, keepdims=True)) / np.exp(a - a.max(ax, keepdims=True)).sum(ax, keepdims=True)
  norm = lambda a: (a - a.mean(2, keepdims=True)) / np.sqrt(a.var(2, keepdims=True) + 1e-5)
  ew, row, col, pos = ((2, 20, 40), (-1, 1)), ((1, 1, 40), (-1, 1)), ((1, 20, 1), (-1, 1)), ((2, 20, 40), (0.5, 2))
  cases = [  # (baretile, numpy, (shape, range) of each input, optional tolerance that defaults to 2e-4)
    (lambda a, b: a + b, lambda a, b: a + b, [ew, col]), (lambda a, b: a - b, lambda a, b: a - b, [row, ew]),
    (lambda a, b: a * b, lambda a, b: a * b, [ew, row]), (lambda a, b: a / b, lambda a, b: a / b, [ew, pos]),
    (lambda a, b: a.maximum(b), np.maximum, [ew, ew]), (lambda a, b: a.minimum(b * 0.5), lambda a, b: np.minimum(a, b * 0.5), [ew, col]),
    (lambda a, b: (a > b).where(a * 2.0, b), lambda a, b: np.where(a > b, a * 2, b), [ew, row]),
    *[(getattr(Tensor, f), g, [r]) for f, g, r in [
      ("exp", np.exp, ew), ("exp2", np.exp2, ew), ("log", np.log, pos), ("log2", np.log2, pos), ("sin", np.sin, ew), ("cos", np.cos, ew),
      ("sqrt", np.sqrt, pos), ("rsqrt", lambda a: 1 / np.sqrt(a), pos), ("reciprocal", lambda a: 1 / a, pos),
      ("relu", lambda a: np.maximum(a, 0), ew), ("gelu", gelu, ew)]],
    *[(lambda a, ax=ax: a.sum(ax), lambda a, ax=ax: a.sum(ax, keepdims=True), [ew]) for ax in (0, 1, 2)],
    *[(lambda a, ax=ax: a.max(ax), lambda a, ax=ax: a.max(ax, keepdims=True), [ew]) for ax in (0, 1, 2)],
    (lambda a: a.mean(), lambda a: a.mean(keepdims=True), [ew]), (lambda a: a.softmax(), lambda a: softmax(a, 2), [ew]),
    (lambda a: a.softmax(1), lambda a: softmax(a, 1), [ew]), (lambda a: a.layernorm(), norm, [ew]),
    (lambda a, b: a @ b, lambda a, b: a @ b, [((2, 20, 40), (-1, 1)), ((1, 40, 30), (-1, 1))], 2**-11),
    (lambda a, b: a @ b.T, lambda a, b: a @ b.swapaxes(1, 2), [((2, 20, 40), (-1, 1)), ((2, 30, 40), (-1, 1))], 2**-11),
    (lambda a: a.T, lambda a: a.swapaxes(1, 2), [ew]), (lambda a: a.reshape(4, 25, 16), lambda a: a.reshape(4, 25, 16), [ew]),
    (lambda a: a.permute(2, 0, 1), lambda a: a.transpose(2, 0, 1), [ew]),
    (lambda a: a.expand(3, 20, 40), lambda a: np.broadcast_to(a, (3, 20, 40)), [col]),
    (lambda a: a.pad(((1, 2), (3, 0), (0, 5))), lambda a: np.pad(a, ((1, 2), (3, 0), (0, 5))), [ew]),
    (lambda a: a[:, 3:17, ::-1], lambda a: a[:, 3:17, ::-1], [ew]), (lambda a: a[1], lambda a: a[1:2], [ew]),
    (lambda a: a.split(2), lambda a: a.reshape(1, 20, 2, 32).transpose(2, 1, 3, 0).reshape(2, 20, 32), [((1, 20, 64), (-1, 1))]),
    (lambda a: a.merge(), lambda a: a.transpose(1, 0, 2).reshape(1, 20, 128), [((2, 20, 64), (-1, 1))]),
    (lambda a: a.gather([3, 0, 7, 3]), lambda a: a[:, [3, 0, 7, 3]], [((1, 9, 40), (-1, 1))]),
    (lambda a, b: a.cat(b, axis=2), lambda a, b: np.concatenate([a, b], 2), [ew, ((2, 20, 13), (-1, 1))]),
    (lambda a, b: b.cat(a, a, axis=0), lambda a, b: np.concatenate([b, a, a], 0), [ew, ((1, 20, 40), (-1, 1))]),
    (Tensor.silu, lambda a: a / (1 + np.exp(-a)), [ew]),
    (lambda a, b: a.conv2d(b, 1), lambda a, b: conv(a, b, 1), [((3, 9, 10), (-1, 1)), ((4, 3, 9), (-1, 1))], 2**-11),
    (Tensor.max_pool2d, lambda a: a.reshape(2, 10, 2, 20, 2).max((2, 4)), [ew])]
  for n, (f, g, specs, *tol) in enumerate(cases):
    xs = [rng.uniform(lo, hi, shape) for shape, (lo, hi) in specs]
    ts = [Tensor.of(rt, x.astype(np.float32).tolist() if x.shape[0] > 1 else x[0].astype(np.float32).tolist(), "fp32").requires_grad_() for x in xs]
    xs = [np.float64(np.float32(x)) for x in xs]
    out = g(*xs)
    w = rng.uniform(-1, 1, out.shape)
    loss = (f(*ts) * Tensor.of(rt, w.tolist() if w.shape[0] > 1 else w[0].tolist(), "fp32")).sum()
    loss.backward()
    for i, (x, t) in enumerate(zip(xs, ts)):
      want = np.zeros_like(x)
      for at in np.ndindex(x.shape):  # d loss / d x[at] by central difference
        hi_, lo_ = [v.copy() for v in xs], [v.copy() for v in xs]
        hi_[i][at] += 1e-6; lo_[i][at] -= 1e-6
        want[at] = ((g(*hi_) * w).sum() - (g(*lo_) * w).sum()) / 2e-6
      got = value(t.grad)
      assert np.abs(got - want).max() <= (tol or [2e-4])[0] * (np.abs(want).max() + 1), (n, i, np.abs(got - want).max())
  a, b, w = rng.uniform(-1, 1, (20, 40)), bf16(rng.uniform(-1, 1, (20, 40))), rng.uniform(-1, 1, (20, 40))
  ta, tb, tw = Tensor.of(rt, a.tolist(), "fp32").requires_grad_(), Tensor.of(rt, b.tolist()).requires_grad_(), Tensor.of(rt, w.tolist(), "fp32")
  ((ta * tb + ta * ta) * tw).sum().backward()
  assert tb.grad.dtype == "bf16" and np.allclose(value(tb.grad)[0], a * w, rtol=2**-8, atol=2**-8), "a bf16 leaf, through the cast"
  assert np.allclose(value(ta.grad)[0], (b + 2 * a) * w, rtol=1e-6, atol=1e-6), "ta read twice: the gradients add"
  (ta * tw).sum().backward()
  assert np.allclose(value(ta.grad)[0], (b + 2 * a) * w + w, rtol=1e-6, atol=1e-6), ".grad adds up"
  assert np.allclose(value((ta * ta).sum().gradient(ta)[0])[0], 2 * a, rtol=1e-6), "gradient() gives it without touching .grad"
  # a sum's gradient is one value for all of its operand, so an operand broadcast into that operand gets it once a copy
  tc = Tensor.of(rt, w[:1].tolist(), "fp32").requires_grad_()
  assert value((ta + tc).sum().gradient(tc)[0]).tolist() == np.full((1, 1, 40), 20.0).tolist(), "20 rows add to each column"
  assert np.allclose(value((tc - ta).mean().gradient(tc)[0]), 1 / 40, rtol=1e-6), "20 rows of a mean over 800"

def test_long_autograd_history(chip):
  # Graphs deeper than Python's recursion limit realize and differentiate.
  rt = Runtime(chip)
  x = Tensor.full(rt, 1, 1, 2, "fp32").requires_grad_()
  y = x
  for _ in range(1100): y = y + 1
  g, = y.gradient(x)
  assert g.tolist() == [[1.0]]
  assert y.tolist() == [[1102.0]], "a deep lazy expression also realizes without Python recursion"
  # Realized steps still retain their gradient history. backward must traverse that history without recursion too.
  y = x
  for _ in range(1100): y = y.repeat_interleave(1)
  y.backward()
  assert x.grad.tolist() == [[1.0]]

def test_backward_before_parameter_updates(chip):
  rt = Runtime(chip)
  for reverse in (False, True):
    a, b = (Tensor.full(rt, 1, 1, v, "fp32").requires_grad_() for v in (2, 3))
    loss = a * b
    loss.backward()
    for p in ((b, a) if reverse else (a, b)): p.assign(p - p.grad)
    assert (a.tolist(), b.tolist(), loss.tolist()) == ([[-1]], [[1]], [[6]])
    assert (a.grad.tolist(), b.grad.tolist()) == ([[3]], [[2]])
  a, b = (Tensor.full(rt, 1, 1, v, "fp32").requires_grad_() for v in (2, 3))
  @tensor.jit
  def step(rate):
    a.grad = b.grad = None
    loss = a * b
    loss.backward()
    a.assign(a - a.grad * rate)
    b.assign(b - b.grad * rate)
    return loss
  want_a, want_b = 2.0, 3.0
  for _ in range(4):
    got = step(Tensor.full(rt, 1, 1, 0.125, "fp32"))
    assert np.allclose(got.tolist(), [[want_a * want_b]])
    want_a, want_b = want_a - want_b * 0.125, want_b - want_a * 0.125
    assert np.allclose([a.tolist()[0][0], b.tolist()[0][0]], [want_a, want_b])

def test_fuzz(chip, monkeypatch):
  # Random expressions mix elementwise ops with broadcasting, unary ops, reductions along any axis and matmuls, with odd
  # shapes and batches. The kernels they turn into are checked against numpy separately, on their inputs as read back
  # from the chip. This isolates kernel errors from bf16 rounding accumulated by earlier operations.
  rt, checked = Runtime(chip), []
  def numpy(e, xs):
    if isinstance(e, int): return xs[e]
    v = [numpy(s, xs) for s in e[1:]]
    f = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "matmul": np.matmul, "T": lambda a: a.swapaxes(-1, -2),
         "relu": lambda a: np.maximum(a, 0), "exp": np.exp, "exp2": np.exp2, "log": np.log, "log2": np.log2, "sin": np.sin,
         "cos": np.cos, "sqrt": np.sqrt, "recip": lambda a: 1 / a, "max": np.maximum, "cmplt": lambda a, b: (a < b) * 1.0,
         "cmpne": lambda a, b: (a != b) * 1.0, "keep": lambda c, x: np.where(c != 0, x, 0), "drop": lambda c, y: np.where(c != 0, 0, y),
         "join": lambda a, b: (np.float32(a).view(np.uint32) | np.float32(b).view(np.uint32)).view(np.float32),
         "cast": lambda a: a, "ftob": lambda a: ((np.float32(a).view(np.uint32) + 0x8000) & 0xFFFF0000).view(np.float32)}
    f |= {k: lambda a, k=k: getattr(a, k[:3])(int(k[3]), keepdims=True) for k in tensix.FOLDS}  # sum0, max2, ...
    return f[e[0]](*v)
  realize = Tensor.realize
  def checked_realize(t):
    if t.op in (None, "T", "move"): return realize(t)  # test_movement covers data movement
    tensor._fit(t)  # realize may compute parts first (a long matmul, a shared value, a tree too big): check those too
    inputs = []
    e, _ = tensor._tree(t, inputs)
    xs = [value(u) for u in inputs]
    realize(t)
    want, got = numpy(e, xs), value(t)
    assert np.abs(got - want).max() <= 0.03 * (np.abs(want).max() + 1), (e, np.abs(got - want).max())
    checked.append(e)
    return t
  monkeypatch.setattr(Tensor, "realize", checked_realize)
  rng = np.random.default_rng(5)
  dtype = "bf16"
  def leaf(shape):  # some leaves are all negative, so an unmasked 0 in the padding would win a max
    lo = rng.choice([-2.0, -4.0])
    a = rng.uniform(lo, lo + 4 if lo == -2 else lo + 3, shape)
    a = bf16(a) if dtype == "bf16" else a.astype(np.float32)
    return Tensor.of(rt, a.tolist() if shape[0] > 1 else a[0].tolist(), dtype)
  def gen(shape, depth):
    b, r, c = shape
    kind = "leaf" if depth == 0 or rng.random() < 0.2 else rng.choice(["bin", "bin", "unary", "red", "mm", "where"])
    if kind in ("bin", "where"):  # each operand keeps a dimension or has 1 there, but not both have 1
      s1 = tuple(1 if rng.random() < 0.2 else d for d in shape)
      s2 = tuple(d if s1[i] == 1 else (1 if rng.random() < 0.2 else d) for i, d in enumerate(shape))
      # compare only inputs: on computed values a rounding step can flip a comparison, and numpy can't check that
      if kind == "where": return (leaf(shape) > 0).where(gen(s1, depth - 1), gen(s2, depth - 1))
      return getattr(gen(s1, depth - 1), rng.choice(["__add__", "__sub__", "__mul__", "maximum", "minimum"]))(gen(s2, depth - 1))
    if kind == "unary":  # sin and cos get values clamped to +-8. Of values like exp's 4e9, they're noise in fp32 and float64
      x = gen(shape, depth - 1)
      return rng.choice([x.relu, (x * 0.5).exp, (x * 0.5).exp2, x.maximum(-8.0).minimum(8.0).sin, x.maximum(-8.0).minimum(8.0).cos,
                         (x * x + 1.0).reciprocal, (x * x + 1.0).rsqrt, (x * x + 1.0).log2, (x * x + 1.0).sqrt])()
    if kind == "red" and 1 in shape:
      # Reduce along an axis that is 1 here: the batch, rows or columns. Take means, not sums, so values stay near 1, or
      # exp would turn rounding into large errors.
      axis = int(rng.choice([ax for ax, d in enumerate(shape) if d == 1]))
      n = int(rng.choice([2, 3] if axis == 0 else [20, 32, 70, 96, 192]))
      x = gen(tuple(n if ax == axis else d for ax, d in enumerate(shape)), depth - 1)
      return x.sum(axis) * (1 / n) if rng.random() < 0.5 else x.max(axis)
    if kind == "mm" and r > 1 and c > 1:
      k = int(rng.choice([32, 50, 64]))
      bb = b if rng.random() < 0.5 else 1
      return (leaf((b, r, k)) @ (leaf((bb, c, k)).T if rng.random() < 0.3 else leaf((bb, k, c)))) * (1 / k)  # B read transposed, or not
    return leaf(shape)
  # The trees alternate between bf16 and fp32. Sizes past 4 tiles make partial blocks in batched matmuls. Unbatched
  # matmuls share by multicast instead.
  for n in range(max(30, int(os.environ.get("FUZZ", 30)))):
    shape = (int(rng.choice([1, 1, 2])), int(rng.choice([1, 20, 32, 45, 64, 160])), int(rng.choice([1, 20, 32, 70, 96, 192])))
    dtype = ("bf16", "fp32")[n % 2]
    gen(shape, 4).realize()
  assert len(checked) >= 20 and any(e[0] == "matmul" or "sum" in str(e) for e in checked)

def test_mnist_mlp(chip):
  # An MNIST-shaped network, 784 -> 128 -> 10 with relu and biases, runs on a batch of 16 and is checked against numpy
  # in float64. It's two launches: matmul + bias + relu, then matmul + bias. The weights are random since there's no
  # dataset here. The logits must match closely, and so must the predicted classes wherever the top two aren't a near
  # tie. K = 784 is 25 tiles, more than the result is wide.
  rt, rng = Runtime(chip), np.random.default_rng(2)
  x = bf16(rng.uniform(0, 1, (16, 784)))
  w1, b1 = bf16(rng.normal(0, 784**-0.5, (784, 128))), bf16(rng.normal(0, 0.1, (1, 128)))
  w2, b2 = bf16(rng.normal(0, 128**-0.5, (128, 10))), bf16(rng.normal(0, 0.1, (1, 10)))
  tx, tw1, tb1, tw2, tb2 = (Tensor.of(rt, a.tolist()) for a in (x, w1, b1, w2, b2))
  launches = rt.launches
  logits = np.array(((tx @ tw1 + tb1).relu() @ tw2 + tb2).tolist())
  assert kernels(rt, launches) == 2
  want = np.maximum(x @ w1 + b1, 0) @ w2 + b2
  assert np.allclose(logits, want, rtol=0.02, atol=0.02), np.abs(logits - want).max()
  top2 = np.sort(want, 1)[:, -2:]
  clear = top2[:, 1] - top2[:, 0] > 0.05
  assert clear.sum() >= 8 and (logits.argmax(1) == want.argmax(1))[clear].all()

def test_mnist_cnn(chip):
  # A small CNN with random weights runs on MNIST-shaped images and is checked against numpy in float64. It's a padded
  # 3x3 convolution from 1 to 8 channels and relu, a 2x2 max pool, a 3x3 convolution from 8 to 16 channels and relu, a
  # pool, and 784 -> 10. A convolution is one matmul by the stacked patches, and each of its 9 taps is one move.
  rt, rng = Runtime(chip), np.random.default_rng(3)
  w1, b1 = bf16(rng.normal(0, 1 / 3, (8, 1, 9))), bf16(rng.normal(0, 0.1, (8, 1, 1)))
  w2, b2 = bf16(rng.normal(0, 72**-0.5, (16, 8, 9))), bf16(rng.normal(0, 0.1, (16, 1, 1)))
  w3, b3 = bf16(rng.normal(0, 784**-0.5, (784, 10))), bf16(rng.normal(0, 0.1, (1, 10)))
  tw1, tb1, tw2, tb2, tw3, tb3 = (Tensor.of(rt, a.tolist()) for a in (w1, b1, w2, b2, w3, b3))
  pool = lambda a: a.reshape(a.shape[0], a.shape[1] // 2, 2, a.shape[2] // 2, 2).max((2, 4))
  for _ in range(2):
    x = bf16(rng.uniform(0, 1, (1, 28, 28)))
    launches = rt.launches
    h = ((Tensor.of(rt, x[0].tolist()).conv2d(tw1, 1) + tb1).relu().max_pool2d().conv2d(tw2, 1) + tb2).relu().max_pool2d()
    logits = np.array((h.reshape(1, 784) @ tw3 + tb3).tolist())
    assert rt.launches - launches <= 50, rt.launches - launches
    h = pool(np.maximum(conv(pool(np.maximum(conv(x, w1, 1) + b1, 0)), w2, 1) + b2, 0))
    want = h.reshape(1, 784) @ w3 + b3
    assert np.allclose(logits, want, rtol=0.02, atol=0.02), np.abs(logits - want).max()

def test_topk(chip):
  # Tensor.topk is one kernel that runs on BRISC. It gives the k largest of each row and their columns, in the order of
  # numpy's stable argsort: the first of equals first, and -0 equal to 0. The cases start with one row, whose result
  # tiles are full of copies of it, as a broadcast reads them, and a k of 1, copied across. Then come rows past one
  # tile, a batch, and columns that aren't a multiple of 32, or 256 of them. Then fp32 rows go in parts, with the best
  # 64 or 32 of each part side by side, and the best of those found with their indices as labels. Last is the softmax
  # of a matmul that isn't computed yet. The picks and values match the softmax as stored, and none of its DRAM is
  # kept after.
  rt, rng = Runtime(chip), np.random.default_rng(37)
  for batch, rows, cols, k in ((1, 1, 128, 8), (1, 3, 16, 4), (1, 40, 40, 1), (2, 33, 256, 32)):
    x = rng.normal(0, 2, (batch, rows, cols))
    x[0, -1, :3], x[0, -1, 3:] = [-0.0, 0.0, -0.0], -np.abs(x[0, -1, 3:]) - 1  # -0 and 0, equal and the largest
    x[0, 0, 9] = x[0, 0, 3] = x[0, 0].max()  # a tie
    x = bf16(x)
    values, ids = Tensor.of(rt, x.tolist()).topk(k)
    want = np.argsort(-x, -1, kind="stable")[..., :k]
    assert np.array_equal(value(ids), want) and np.array_equal(value(values), np.take_along_axis(x, want, -1)), (batch, rows, cols, k)
    wide = Tensor.of(rt, np.ones((batch, 32 if rows == 1 else rows, 32 if k == 1 else k)).tolist())  # a dimension of 1 broadcasts from its copies
    assert np.array_equal(value(values * wide), np.broadcast_to(value(values), wide.shape)), (rows, k)
  logits = Tensor.of(rt, rng.normal(0, 1, (32, 64)).tolist()).matmul(Tensor.of(rt, rng.normal(0, 0.05, (64, 8)).tolist()))
  probs = logits.softmax()
  values, ids = probs.topk(2)
  got, p = value(ids)[0], value(probs)[0]  # ids first: reading probs first would compute it separately, before topk
  assert np.array_equal(got, np.argsort(-p, -1, kind="stable")[:, :2]) and np.array_equal(value(values)[0], -np.sort(-p, -1)[:, :2])
  for rows, cols, parts, k in ((5, 2560, 4, 64), (40, 1024, 2, 32)):  # fp32 rows in parts side by side, then the best of those
    x = rng.normal(0, 3, (rows, cols)).astype(np.float32)
    x[0, 100] = x[0, cols // 2 + 3] = x[0].max() + 1  # a tie across parts
    values, ids = Tensor.frombytes(rt, rows, cols, x.tobytes(), dtype="fp32").topk(k, parts)
    w = cols // parts
    want = np.concatenate([np.argsort(-x[:, p * w:(p + 1) * w], -1, kind="stable")[:, :k] + p * w for p in range(parts)], 1)
    assert np.array_equal(value(ids)[0], want) and np.array_equal(value(values)[0], np.take_along_axis(x, want, -1)), (rows, cols, parts, k)
    values, ids = values.topk(k, labels=ids)
    want = np.argsort(-x, -1, kind="stable")[:, :k]
    assert np.array_equal(value(ids)[0], want) and np.array_equal(value(values)[0], np.take_along_axis(x, want, -1)), (rows, cols, parts, k)

def test_put(chip):
  # put writes rows into a tensor at positions read on the chip, the inverse of gather. 70 rows of a batch of 3 go into
  # 200 positions, scattered and out of order. The positions of the 70 rows take three tiles, and the 200 positions
  # span seven tile rows. One position comes twice, and the later row wins. Then a prompt's rows go into their
  # positions in order, the way a KV cache takes them.
  rt, rng = Runtime(chip), np.random.default_rng(23)
  for dtype in ("bf16", "fp32"):
    cache, x = rng.normal(0, 1, (3, 200, 96)).astype(np.float32), rng.normal(0, 1, (3, 70, 96)).astype(np.float32)
    if dtype == "bf16": cache, x = bf16(cache), bf16(x)
    places = [int(p) for p in rng.permutation(200)[:70]]
    places[40] = places[7]
    t = Tensor.of(rt, cache.tolist(), dtype)
    t.put(Tensor.of(rt, [places], "int32"), Tensor.of(rt, x.tolist(), dtype))
    for i, p in enumerate(places): cache[:, p] = x[:, i]
    assert np.array_equal(value(t), cache), dtype
  kv, rows = Tensor.of(rt, np.zeros((3, 64, 64)).tolist()), bf16(rng.normal(0, 1, (3, 40, 64)))
  kv.put(Tensor.of(rt, [list(range(5, 45))], "int32"), Tensor.of(rt, rows.tolist()))
  assert np.array_equal(value(kv)[:, 5:45], rows) and not value(kv)[:, :5].any() and not value(kv)[:, 45:].any()

def test_put_aliases_and_singleton(chip):
  for end in (chip.tensix[1], chip.tensix[-1]):
    rt = Runtime(chip, end=end)
    a = np.arange(64, dtype=np.float32).reshape(64, 1)
    x = Tensor.of(rt, a.tolist(), "fp32")
    ids = Tensor.of(rt, [list(range(32, 64)) + list(range(32))], "int32")
    x.put(ids, x)
    assert np.array_equal(value(x)[0], np.concatenate([a[32:], a[:32]]))
    x = Tensor.of(rt, a.tolist(), "fp32")
    x.put(Tensor.of(rt, [list(range(16, 48))], "int32"), x[:32, :])
    want = a.copy()
    want[16:48] = a[:32]
    assert np.array_equal(value(x)[0], want)
    for dtype in ("bf16", "fp32", "int32", "bfp8", "bfp4"):
      x = Tensor.zeros(rt, 1, 2, dtype=dtype)
      x.put(Tensor.of(rt, [[0]], "int32"), Tensor.of(rt, [[2, 3]], "bf16" if dtype in ("bfp8", "bfp4") else dtype))
      if dtype in ("bfp8", "bfp4"):
        rt.wait()
        raw = chip.read(*bh.DRAM[0], x.addr, tensix.tile_bytes(dtype))
        rows = np.frombuffer(tensor._bf16s(raw, dtype), np.uint16).reshape(4, 16, 16)[[0, 2], :, :2]
        assert np.all(rows == [0x4000, 0x4040]), dtype
      else: assert (x + Tensor.zeros(rt, 2, 2, dtype=dtype)).tolist() == [[2, 3], [2, 3]], dtype

def test_block_float_cache(chip):
  # This tests a KV cache in bfp8, the format Tenstorrent's models use, and in bfp4. put writes bf16 rows into a
  # block-float tensor and rounds each run of 16 the way cast does, bit for bit. The puts are one row, rows in two
  # tiles, a prompt filling a tile, and the edges. Attention reads the cache in place, with each key and value head
  # serving two query heads. Keys are read transposed, which needs unpacker 0 to transpose faces of block floats. The
  # matmuls and attention in one kernel are bit for bit the bf16 ones on the same values. Last, assign copies a view of
  # the cache as it is into a separate tensor, and from there into another view of the cache.
  rt, rng = Runtime(chip), np.random.default_rng(16)
  for fmt in ("bfp8", "bfp4"):
    cache = Tensor.of(rt, np.zeros((4, 96, 64)).tolist()).cast(fmt).realize()  # 2 key heads, then 2 value heads
    want = np.zeros((4, 96, 64))
    for places in ([5], [33, 34, 70], list(range(32, 64)), [0, 95]):
      x = bf16(rng.normal(0, 2, (4, len(places), 64)) * 10.0 ** rng.integers(-3, 3, (4, len(places), 1)))
      cache.put(Tensor.of(rt, [places], "int32"), Tensor.of(rt, x.tolist()))
      want[:, places] = value(Tensor.of(rt, x.tolist()).cast(fmt))
      assert np.array_equal(value(cache), want), (fmt, places)
    same = Tensor.of(rt, want.tolist())  # the same values in bf16
    q, s = (Tensor.of(rt, bf16(rng.normal(0, 1, shape)).tolist()) for shape in ((4, 5, 64), (4, 5, 96)))
    for c in (cache, same): c.keys, c.values = c[0:2, :, :].repeat_interleave(2), c[2:4, :, :].repeat_interleave(2)
    assert value(q @ cache.keys.T).tolist() == value(q @ same.keys.T).tolist()
    assert value(s @ cache.values).tolist() == value(s @ same.values).tolist()
    places = Tensor.of(rt, [[[95]] * 5], "int32")
    assert value(q.attend(cache.keys, cache.values, places)).tolist() == value(q.attend(same.keys, same.values, places)).tolist(), fmt
    back = Tensor(rt, 64, 64, batch=2, dtype=fmt).assign(cache[0:2, :64, :])
    cache[2:4, 32:96, :].assign(back)
    want[2:4, 32:96] = want[0:2, :64]
    assert np.array_equal(value(back), want[0:2, :64]) and np.array_equal(value(cache), want), fmt

def test_matmul_composition_and_integer_output(chip):
  rt = Runtime(chip)
  for k in (1, 2, 32):
    a, b = Tensor.full(rt, 1, k, 1), Tensor.full(rt, k, 1, 1.5)
    assert a.matmul(b, "int32").tolist() == [[int(1.5 * k)]]
  a = Tensor.full(rt, 32, 64, 0.5)
  w1, w2 = (Tensor.of(rt, [[[v] * 32] * 64] * 2) for v in (1., 2.))
  ids = Tensor.of(rt, [[1, 0]], "int32")
  assert (a.matmul(w1, index=ids) + a.matmul(w2, index=ids)).tolist() == [[[96.] * 32] * 32] * 2
  assert a.matmul(w1, "int32", index=ids).tolist() == [[[32] * 32] * 32] * 2
  assert (a @ w1.cast("bfp8") + a @ w2.cast("bfp4")).tolist() == [[[96.] * 32] * 32] * 2

def test_storage_conversion_and_gradients(chip):
  rt, rng = Runtime(chip), np.random.default_rng(83)
  raw = bf16(rng.normal(size=(2, 65, 96)))
  x = Tensor.of(rt, raw.tolist())
  for dtype in ("bf16", "bfp8", "bfp4"):
    stored = x.cast(dtype).realize()
    for t in (stored, stored.columns()):
      assert t.cast("bf16").tolist() == t.tolist()
      assert t.cast("fp32").tolist() == t.tolist()
    if dtype != "bf16":
      assert stored[:, 32:64, :].cast("bf16").tolist() == value(stored)[:, 32:64].tolist()
      assert stored.repeat_interleave(2).cast("bf16").tolist() == np.repeat(value(stored), 2, axis=0).tolist()
      assert stored.cast("bf16").T.tolist() == value(stored).transpose(0, 2, 1).tolist()
  split = Tensor.full(rt, 96, 64, 1).cast("bfp4").columns().split(3, 1)
  assert split.cast("bf16").tolist() == split.tolist()
  for dtype in ("bf16", "bfp8", "bfp4"):
    a, w = Tensor.of(rt, [[1., 2.]]), Tensor.of(rt, [[1.25, -2.], [3.5, 4.25]])
    a.requires_grad = w.requires_grad = True
    quant = w.cast(dtype).columns()
    ga, gw = (a @ quant).sum().gradient(a, w)
    close(ga, np.asarray(quant.tolist()).sum(1).reshape(1, 2), 0.02)
    assert gw.tolist() == [[1., 1.], [2., 2.]]
    w = Tensor.of(rt, [[1.25, -2.], [3.5, 4.25], [0.5, 1.]]).requires_grad_()
    stored = w.cast(dtype)
    ga, gw = (a @ stored.T).sum().gradient(a, w)
    close(ga, np.asarray(stored.tolist()).sum(0).reshape(1, 2), 0.02)
    assert gw.tolist() == [[1., 2.]] * 3
