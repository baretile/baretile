# Build tensor expressions and compile them into kernels. Fuse operations where possible; compute inputs
# separately when fusion would exceed a kernel's limits or require a different layout or dtype.
import array, collections, functools, itertools, math, operator, struct
from baretile import bh
from baretile.runtime import Bound
from baretile.tensix import ACCS, BFP, COMBINE, FOLDS, INT32_OPS, ITEMSIZE, MAX_OPS, ext, round_bf16, tile_bytes
from baretile.kernels import attention_plan, link_flags
from baretile.kernels import (attention_kernel, bfp4_kernel, columns_kernel, copy_kernel, fused_kernel, gather_kernel,
                              link_kernel, move_kernel, put_kernel, readback_kernel, storage_kernel, topk_kernel,
                              transpose_kernel, upload_kernel, zeros_kernel)

FIDELITY = 4  # phases a bf16 matmul takes by default. 4 gives exact products, and tensix.matmul says what fewer drop
DIMS = 5  # a tensor's most dimensions: up to three leading ones, then its matrices' rows and columns
# frombytes sends at least this many bytes through upload_kernel, and less through the BAR, which starts sooner but
# takes a PCIe write for each word. A guess until the probe measures both on a card.
DMA_UPLOAD = 1 << 16

def _matrices(f):  # an op on a batch of matrices: a tensor with leading dimensions reshapes to three first
  @functools.wraps(f)
  def op(self, *args, **kwargs):
    if len(self.shape) > 3: raise ValueError(f"{f.__name__} takes three dimensions, not {self.shape}")
    return f(self, *args, **kwargs)
  return op

class Tensor:
  """Represent a batch of matrices with shape (batch, rows, cols), or with up to three leading dimensions in place of
  the batch, which fold into it in row-major order. Supported storage formats are bf16, fp32, int32, bfp8 and bfp4.
  An unrealized tensor holds an operation and its operands.

  Default DRAM storage uses 32 x 32 tiles in row order, with matrices stored consecutively. Tile i is
  in bank i % 8. Partial tiles are padded with zeros. A dimension of size 1 repeats across its tile
  for broadcasting. Kernels preserve this layout, including the padding needed by matmul.

  Batch indexing, tile-aligned slices, split and repeat_interleave can share an owner's storage.
  Tile (b, i, j) of a view is at b // rep * bstride + i * stride + j + offset in that storage.
  Movement that can't use a view is copied in one launch: by move_kernel, or transpose_kernel for a bf16 transpose."""
  def __init__(self, rt, rows, cols, op=None, srcs=(), batch=1, dtype="bf16"):
    self.addr = None  # construction can fail before there is an allocation to release
    batch, rows, cols = _shape((batch, rows, cols))
    if dtype not in (*ITEMSIZE, *BFP): raise ValueError(f"unknown tensor dtype: {dtype}")
    if any(s.rt is not rt for s in srcs): raise ValueError("tensor operands must use the same runtime")
    rt.check()
    self.rt, self.rows, self.cols, self.batch, self.op, self.srcs, self.dtype = rt, rows, cols, batch, op, srcs, dtype
    self.lead = (batch,)  # the leading dimensions, which multiply to the batch
    tr, tc = -(-rows // 32), -(-cols // 32)  # tile rows and columns
    self.tiles, self.bstride, self.stride, self.offset, self.owner, self.rep = batch * tr * tc, tr * tc, tc, 0, None, 1
    # by columns: (runs a batch entry moves on in each bank, tiles a run, tiles a batch entry moves along its runs).
    # columns gives (runs a matrix, tiles a run, 0), and split, cutting K, gives (0, tiles a run, K tiles a part)
    self.runs, self.cyc = None, 0  # cyc, if not 0: kernels read stored matrix (b / rep) % cyc, as a spread's view has
    self.addr = None if op else self._alloc()
    self.requires_grad, self.grad, self._ctx = False, None, None
    if op not in (None, "move", "view"): self._track(op, srcs)  # moves and views are tracked by their callers
  def _track(self, op, srcs, *args):  # remember how this was made, if a source needs a gradient
    if any(s.requires_grad for s in srcs): self.requires_grad, self._ctx = True, (op, srcs, args)
    return self
  def requires_grad_(self, on=True):
    self.requires_grad = on
    return self
  # the address and tile strides a kernel reads it by, or for a tensor by columns, its runs
  def _where(self, runs=False):
    if self.runs and not runs:
      raise ValueError("cast a tensor by columns to bf16 for operations other than matmul's B or readback")
    return (self.addr, *self.runs) if self.runs else (self.addr, self.bstride if self.batch > 1 else 0, self.stride,
                                                      self.offset)
  def _alloc(self): return self.rt.dram_alloc(-(-self.tiles // len(bh.DRAM)) * tile_bytes(self.dtype))
  def _same_runtime(self, *xs):
    if any(x.rt is not self.rt for x in xs): raise ValueError("tensor operands must use the same runtime")
  def _alias(self, into):  # take into's storage as this tensor's
    self.addr, self.owner, self.bstride, self.stride, self.offset = (into.addr, _root(into), into.bstride, into.stride,
                                                                     into.offset)

  def __del__(self):  # free the DRAM. the runtime holds it until the launches queued so far are done
    if self.addr is not None and self.owner is None: self.rt.dram_free(self.addr)

  @classmethod
  def full(cls, rt, rows, cols, value, dtype="bf16"): return cls.of(rt, [[value] * cols for _ in range(rows)], dtype)

  @classmethod
  def zeros(cls, rt, rows, cols, batch=1, dtype="bf16"):
    """Make a tensor of zeros in any storage format, block floats included. The workers write them from the tile of
    zeros in their L1, so nothing crosses PCIe. Model caches start this way."""
    t = cls(rt, rows, cols, batch=batch, dtype=dtype)
    each = -(-t.tiles // len(bh.DRAM)) * tile_bytes(dtype)  # the bytes it has in each bank
    # a unit: 4 KiB of a bank
    return t.run(("zeros",), zeros_kernel, t.addr, each, units=len(bh.DRAM) * -(-each // 4096), name="zeros")

  @classmethod
  def of(cls, rt, data, dtype="bf16"):
    """Make a tensor from rows of numbers, or a list of those for a batch, nested again for each leading dimension
    past the first. bf16 rounds to nearest even."""
    if not len(data) or not len(data[0]): raise ValueError("tensor data must be nonempty")
    lead = []
    while isinstance(data[0][0], (list, tuple)) and len(data[0][0]) and isinstance(data[0][0][0], (list, tuple)):
      if len(lead) == DIMS - 3 or any(not isinstance(d, (list, tuple)) or len(d) != len(data[0]) for d in data):
        raise ValueError(f"tensor data must be rectangular, with at most {DIMS} dimensions")
      lead, data = lead + [len(data) // math.prod(lead)], [m for d in data for m in d]  # an entry per index along lead
    matrices = data if isinstance(data[0][0], (list, tuple)) else [data]
    rows, cols = len(matrices[0]), len(matrices[0][0])
    if not cols or any(len(m) != rows or any(len(row) != cols for row in m) for m in matrices):
      raise ValueError("tensor data must be nonempty rectangular matrices of the same shape")
    values = array.array("i" if dtype == "int32" else "f", [v for m in matrices for row in m for v in row])
    if dtype == "bf16": values = array.array("H", map(round_bf16, array.array("I", values.tobytes())))
    t = cls.frombytes(rt, rows, cols, values, len(matrices), dtype)
    return t._relabel((*lead, len(matrices) // math.prod(lead))) if lead else t

  @classmethod
  def frombytes(cls, rt, rows, cols, data, batch=1, dtype="bf16"):
    """Upload little-endian values in row-major order, with matrices stored consecutively. Convert to tile
    layout in 8-byte chunks, then write each bank: through the BAR below DMA_UPLOAD bytes, and through
    upload_kernel from there. Either way the values are in DRAM on return. Model weights use this path."""
    if dtype in BFP: raise ValueError("upload as bf16, then cast on the chip to a block-float format")
    if dtype not in ITEMSIZE: raise ValueError(f"unknown tensor dtype: {dtype}")
    batch, rows, cols = _shape((batch, rows, cols))
    n, tr, tc, data = ITEMSIZE[dtype], -(-rows // 32), -(-cols // 32), memoryview(data).cast("B")
    if len(data) != n * batch * rows * cols:
      raise ValueError(f"{len(data)} bytes for {batch} x {rows} x {cols} of {dtype}")
    t = cls(rt, rows, cols, batch=batch, dtype=dtype)
    tb, run = tile_bytes(dtype), 2 * n  # bytes per tile, and 8-byte words per run of 16 values (a row of a face)
    tiles = memoryview(bytearray(t.tiles * tb)).cast("Q")  # the tiles in order, zero where they're padding
    for b, r in ((b, r) for b in range(batch) for r in range(rows if rows > 1 else 32)):  # a single row fills its tile
      at = n * (b * rows + (r if rows > 1 else 0)) * cols
      row = bytes(data[at:at + n]) * 32 if cols == 1 else bytes(data[at:at + n * cols]) + bytes(n * (32 * tc - cols))
      src, first = memoryview(row).cast("Q"), (b * tr + r // 32) * tc * tb // 8 + r % 16 * run
      # word q of the runs in the left or right faces
      for half, q in ((half, q) for half in (0, 1) for q in range(run)):
        start = first + (r % 32 // 16 * 2 + half) * tb // 32 + q  # faces as in the tile layout at the top of tensix.py
        tiles[start:start + tc * tb // 8:tb // 8] = src[half * run + q::2 * run]
    tiles = tiles.cast("B")
    # tile i is in bank i % 8 at addr + i // 8 tiles, so each bank's tiles are one run
    banks = [b"".join(tiles[i * tb:(i + 1) * tb] for i in range(bank, t.tiles, len(bh.DRAM)))
             for bank in range(len(bh.DRAM))]
    if len(tiles) < DMA_UPLOAD:
      for bank, run in enumerate(banks): t.rt.chip.write(*bh.DRAM[bank], t.addr, run)
    else: t._upload(banks)
    return t

  def _upload(self, banks):
    """Write each bank's bytes at this tensor's address through upload_kernel. Chunks are staged in the two halves
    of Runtime.back, so the host fills one half while the chip reads the other. The launches bypass a jit recording:
    their staged data is gone by the time it would replay, and the tensor keeps what they wrote. The last one is
    waited for, so the data is in DRAM on return."""
    rt, n, d = self.rt, len(banks[0]), len(bh.DRAM)  # bank 0 has the most tiles, and the others pad up to it with zeros
    half = len(rt.back) // 2 // d // 64 * 64  # each bank's bytes in a half
    workers = max(1, len(rt.cores) // d)  # cut each bank's part into this many pieces, a unit for each worker
    kernel, last = _cached(("upload",), upload_kernel), [0, 0]  # the launch that last read each half
    for c, at in enumerate(range(0, n, half)):
      k, h = min(half, n - at), c % 2
      rt.wait(launch=last[h])
      stage, bus = rt.back[h * d * half:], rt.back_bus + h * d * half
      for b, run in enumerate(banks):
        part = run[at:at + k]
        stage[b * k:b * k + len(part)], stage[b * k + len(part):(b + 1) * k] = part, bytes(k - len(part))
      piece = min(bh.MAX_PACKET, -(-k // (64 * workers)) * 64)
      rt.send(*rt.block([(kernel, (bus & 0xFFFFFFFF, bus >> 32, self.addr + at, k, piece), d * -(-k // piece),
                          "upload")]))
      last[h] = rt.launches
    rt.wait()

  def tobytes(self):
    """Return row-major bytes in the format accepted by frombytes. Convert block floats to bf16.
    Queue a kernel to write tiles into host memory and wait for completion. This avoids a PCIe
    round trip for each host load through the BARs."""
    self.realize()
    n, rows, cols, tc, tb = (2 if self.dtype in BFP else ITEMSIZE[self.dtype], self.rows, self.cols,
                             -(-self.cols // 32), tile_bytes(self.dtype))
    # readback_kernel sends sent bytes of each tile: v rows, with face f starting at faces[f]
    v = 32 if self.dtype in BFP else min(rows, 32)
    sent, faces = (tb if v == 32 else 32 * v * n,
                   [16 * n * f for f in (0, min(v, 16), 2 * min(v, 16), 2 * min(v, 16) + max(v - 16, 0))])
    by = bool(self.runs)
    per, tiles, kernel = (len(self.rt.back) // sent, bytearray(),
                          _cached(("readback", self.dtype, by), lambda: readback_kernel(self.dtype, by)))
    for start in range(0, self.tiles, per):
      k = min(per, self.tiles - start)
      self.rt.launch(kernel, (start, *self._where(True), tc, self.tiles // self.batch, self.rep,
                              self.rt.back_bus & 0xFFFFFFFF, self.rt.back_bus >> 32, v), units=k, name="readback")
      self.rt.wait()
      tiles += self.rt.back[:k * sent]
    if self.dtype in BFP:
      tiles, sent = b"".join(_bf16s(tiles[i * sent:(i + 1) * sent], self.dtype) for i in range(self.tiles)), 2048
    # 8-byte words per tile sent, per run
    words, sw, run, matrix = memoryview(tiles).cast("Q"), sent // 8, 2 * n, self.tiles // self.batch
    # a row across the tile columns
    out, row = bytearray(n * self.batch * rows * cols), memoryview(bytearray(32 * n * tc)).cast("Q")
    # frombytes in reverse: row by row, word q of its runs from the left or right faces of the tiles along it
    for b, r in ((b, r) for b in range(self.batch) for r in range(rows)):
      first = (b * matrix + r // 32 * tc) * sw + (r % 16 * 16 * n) // 8
      for half, q in ((half, q) for half in (0, 1) for q in range(run)):
        start = first + faces[r % 32 // 16 * 2 + half] // 8 + q
        row[half * run + q::2 * run] = words[start:start + tc * sw:sw]
      at = n * (b * rows + r) * cols
      out[at:at + n * cols] = row.cast("B")[:n * cols]
    return bytes(out)

  def tolist(self):  # rows of numbers, or a list of them for a batch, nested by any leading dimensions
    raw = self.tobytes()
    if self.dtype == "bf16" or self.dtype in BFP:
      raw = array.array("I", [b << 16 for b in array.array("H", raw)]).tobytes()
    values = array.array("i" if self.dtype == "int32" else "f", raw).tolist()
    m = [[values[(b * self.rows + r) * self.cols:(b * self.rows + r + 1) * self.cols] for r in range(self.rows)]
         for b in range(self.batch)]
    if len(self.lead) == 1: return m if self.batch > 1 else m[0]
    for n in reversed(self.lead[1:]): m = [m[i:i + n] for i in range(0, len(m), n)]
    return m

  @_matrices
  def gather(self, ids):
    """Read rows selected by ids. Accept a Python list or an int32 tensor row, so indices can be computed on
    the chip. Python indices are bounds-checked; tensor indices aren't."""
    if self.dtype not in ITEMSIZE: raise ValueError("cast block-float storage to bf16 before gathering rows")
    if not isinstance(ids, Tensor):
      if not all(0 <= i < self.rows for i in ids):
        raise IndexError(f"gather picks rows of a {self.rows}-row matrix: {list(ids)}")
      ids = Tensor.of(self.rt, [list(ids)], "int32")
    self._same_runtime(ids)
    if not (self.batch == 1 and ids.dtype == "int32" and ids.shape[:2] == (1, 1)):
      raise ValueError("gather needs batch 1, and ids as an int32 row")
    out, d, ids = Tensor(self.rt, ids.cols, self.cols, dtype=self.dtype), self.dtype, ids.realize()
    addr, _, stride, offset = _plain(self.realize())._where()
    return out.run(("gather", d), lambda: gather_kernel(d), out.addr, addr, stride, offset,
                   -(-self.cols // 32), ids.cols, ids.addr, ids.offset, name="gather")._track("gather", (self,), ids)

  @_matrices
  def put(self, ids, x):
    """Write x into rows selected by ids, in place. Models use this to update KV caches without changing the
    shape of a decode step. For a block-float destination, x is bf16 and is rounded as in cast."""
    self._same_runtime(ids, x)
    if not (ids.dtype == "int32" and ids.shape[:2] == (1, 1) and ids.cols == x.rows):
      raise ValueError("put needs ids as an int32 row, with one position for each row of x")
    if not ((x.batch, x.cols) == (self.batch, self.cols) and x.dtype == ("bf16" if self.dtype in BFP else self.dtype)):
      raise ValueError(f"put needs rows of the cache's batch and width, "
                       f"bf16 for a block float: {x.shape} {x.dtype} into {self.shape} {self.dtype}")
    if self.rep != 1: raise ValueError("put requires a tensor with rep=1")
    x, ids, tc, tr = _plain(x.realize()), ids.realize(), -(-self.cols // 32), -(-self.rows // 32)
    if 2 * x.rows + tr > 2**17: raise ValueError("put has too many rows to track in a worker's L1")
    self.realize()
    _writes(self)
    if x.addr == self.addr: x = _copy(x)
    if ids.addr == self.addr: ids = _copy(ids)
    return self.run(("put", self.dtype), lambda: put_kernel(self.dtype), *self._where(), *x._where(), ids.addr,
                              ids.offset, x.rows, tc, tr, self.rows, units=self.batch * x.rows * tc, name="put")

  def _view(self, rows, cols, batch, offset, bstride=0, runs=None):
    if self.rep != 1:
      raise ValueError("can't view a repeat: a view addresses stored matrices, and a repeat's aren't all stored")
    if self.runs and not runs:
      raise ValueError("a tensor by columns can only be viewed by split, cutting K into a batch")
    v = Tensor(self.rt, rows, cols, "view", batch=batch, dtype=self.dtype)
    v.op, v.addr, v.owner = None, self.realize().addr, _root(self)
    v.offset, v.bstride, v.stride, v.runs = self.offset + offset, bstride, self.stride, runs
    return v
  def _relabel(self, lead):
    """The same matrices under other leading dimensions, which multiply to the batch. A realized tensor gets a view,
    a transpose the transpose of its source's, and another unrealized tensor an op that kernels read through."""
    lead = tuple(lead)
    if lead == self.lead: return self
    if self.op == "lead": return self.srcs[0]._relabel(lead)
    if self.op == "T": return self.srcs[0]._relabel(lead).T
    if self.op is None and self.rep > 1: return (self + 0)._relabel(lead)  # a repeat's matrices aren't all stored
    if self.op is None: t = self._view(*self.shape[-2:], self.batch, 0, self.bstride, self.runs)._track("lead", (self,))
    else: t = Tensor(self.rt, self.rows, self.cols, "lead", (self,), self.batch, self.dtype)
    t.lead = lead
    return t

  def _spread(self, lead):
    """These matrices broadcast over the leading dimensions lead, where this tensor has some of lead's sizes and 1 for
    the rest. When its sizes are one run of lead's, like (A, 1) or (1, B) in (A, B), a kernel reads its matrix
    (b / rep) % n for matrix b of the result, without a copy. Others, like (A, 1, C) in (A, B, C), expand by a move."""
    mine, big = (1,) * (len(lead) - len(self.lead)) + self.lead, [k for k, n in enumerate(lead) if n > 1]
    run = [k for k in big if mine[k] > 1]  # the dimensions this tensor's matrices step through, of those b does
    at = big.index(run[0])
    if run != big[at:at + len(run)]: return self.expand(*lead, self.rows, self.cols)
    t = Tensor(self.rt, self.rows, self.cols, "spread", (self,), math.prod(lead), self.dtype)
    # rep of the result's matrices in a row read each of these. Past n of them it starts over, unless the run is first
    t.lead, t.view = tuple(lead), None
    t.spread = (math.prod(lead[k] for k in big[at + len(run):]), self.batch if at else 0)
    return t

  def __getitem__(self, key):
    """Select a batch entry with x[b], or slice with x[rows, cols] or x[batch, rows, cols].
    A slice shares storage when rows and columns start on tile boundaries, end on a boundary or at the
    matrix edge, and step by 1. The batch may use any positive step. Slicing rows or columns down to
    size 1 needs copies across the tile. Other slices also copy. An unrealized sum of realized
    tensors, such as split-K results, is sliced term by term."""
    if self.rep > 1: return (self + 0)[key]  # slice a copy of a repeat: the copy stores its matrices one after another
    if len(self.lead) > 1: return self._lead_item(key)
    if isinstance(key, int):
      if not -self.batch <= key < self.batch: raise IndexError(f"batch index {key} outside a batch of {self.batch}")
      key %= self.batch
      return self._view(self.rows, self.cols, 1, key * self.bstride)._track("index", (self,), key)
    if not (isinstance(key, tuple) and len(key) in (2, 3) and all(isinstance(k, slice) for k in key)):
      raise ValueError(f"index a tensor as x[b], x[rows, cols] or x[batch, rows, cols] with slices, not x[{key!r}]")
    ranges = [range(*k.indices(n)) for k, n in zip((slice(None),) * (3 - len(key)) + tuple(key), self.shape)]
    shape = tuple(map(len, ranges))
    if not all(shape): raise ValueError(f"an empty slice of {self.shape}")
    if all(g.step > 0 if ax == 0 else g.step == 1 and g.start % 32 == 0 and (g.stop % 32 == 0 or g.stop == n)
           and (len(g) > 1 or n == 1)
           # tile-aligned. a slice of length 1 must be a copy, filling its tile
           for ax, (g, n) in enumerate(zip(ranges, self.shape))):
      if terms := _terms(self): return sum((t[key] for t in terms[1:]), terms[0][key])
      b, r, c = (g.start for g in ranges)
      t = self._view(shape[1], shape[2], shape[0], b * self.bstride + r // 32 * self.stride + c // 32,
                     self.bstride * ranges[0].step)
    else:
      t = self._map(shape, (0, 1, 2), [g.start for g in ranges], [g.step for g in ranges])
    return t._track("slice", (self,), ranges)

  def _lead_item(self, key):
    """x[i], or slices with leading dimensions, where missing leading slices are whole. Leading slices that pick
    evenly spaced matrices, as any slice of the first does, give a slice of the batch they fold into, which slices the
    rows and columns as with three dimensions. Other leading slices copy, in one launch, as does a slice of an
    unrealized move, which joins it."""
    lead, n = self.lead, len(self.shape)
    if isinstance(key, int):
      if not -lead[0] <= key < lead[0]: raise IndexError(f"index {key} outside a leading size of {lead[0]}")
      return self[(slice(key % lead[0], key % lead[0] + 1), *(slice(None),) * (n - 1))]._relabel(lead[1:])
    if not (isinstance(key, tuple) and 2 <= len(key) <= n and all(isinstance(k, slice) for k in key)):
      raise ValueError(f"index {self.shape} as x[i] or with two to {n} slices, not x[{key!r}]")
    key = (slice(None),) * (n - len(key)) + key
    ranges = [range(*k.indices(m)) for k, m in zip(key, self.shape)]
    if not all(ranges): raise ValueError(f"an empty slice of {self.shape}")
    sizes = [math.prod(lead[k + 1:]) for k in range(len(lead))]  # how far an index along each moves through the batch
    picked = [sum(i * m for i, m in zip(c, sizes)) for c in itertools.product(*ranges[:-2])]
    steps = {b - a for a, b in zip(picked, picked[1:])}
    if self.op != "move" and len(steps) <= 1 and min(steps, default=1) > 0:
      batch = slice(picked[0], picked[-1] + 1, min(steps, default=1))
      return self._relabel((self.batch,))[batch, key[-2], key[-1]]._relabel(tuple(map(len, ranges[:-2])))
    t = self._map(tuple(map(len, ranges)), range(n), [g.start for g in ranges], [g.step for g in ranges])
    return t._track("slice", (self,), ranges)

  # Movement. Each op maps the result's elements to this tensor's, and one launch copies them: move_kernel, or
  # transpose_kernel for a bf16 transpose.
  def move(self, shape, offset, strides, lo=None, hi=None):
    """Map elements into a new tensor of shape, with three to five dimensions. Output element (b, r, c) reads
    input element offset + b * strides[0] + r * strides[1] + c * strides[2], in row-major order, and each
    leading dimension past the first adds its index times its stride the same way.
    Output positions outside [lo, hi) on any axis are zero. Every selected source element must lie
    inside this tensor; offsets and strides are signed 32-bit integers. No gradient: reshape, permute,
    expand, pad and slices provide their own gradient rules."""
    try:
      shape, strides = (tuple(map(operator.index, values)) for values in (shape, strides))
      lo = (0,) * len(shape) if lo is None else tuple(map(operator.index, lo))
      hi, offset = shape if hi is None else tuple(map(operator.index, hi)), operator.index(offset)
    except TypeError: raise ValueError("move needs integer dimensions, offset, strides and bounds") from None
    if not 3 <= len(shape) <= DIMS or any(len(values) != len(shape) for values in (strides, lo, hi)):
      raise ValueError(f"move needs three to {DIMS} dimensions, with a stride and bounds for each")
    shape = _shape(shape)
    if not all(0 <= a <= b <= n < 2**31 for a, b, n in zip(lo, hi, shape)):
      raise ValueError("move bounds must lie inside its signed 32-bit output dimensions")
    if not all(-2**31 <= v < 2**31 for v in (offset, *strides)):
      raise ValueError("move offset and strides must fit signed 32 bits")
    if all(a < b for a, b in zip(lo, hi)):
      ends = [(a * s, (b - 1) * s) for a, b, s in zip(lo, hi, strides)]
      first, last = offset + sum(min(v) for v in ends), offset + sum(max(v) for v in ends)
      if not 0 <= first <= last < min(math.prod(self.shape), 2**32):
        raise ValueError("move reads outside its input's logical storage or 32-bit element addresses")
    t = _lead(Tensor(self.rt, *shape[-2:], "move", (self,), math.prod(shape[:-2]), self.dtype), shape[:-2])
    t.map = (offset, *strides, *lo, *hi)
    return t
  def _map(self, shape, axes, starts, steps):
    """Map output index e on axis a to starts[a] + steps[a] * e on source axis axes[a], using zero outside the
    source. Return shape, with as many dimensions as this tensor. Compose with an unrealized move so a chain of
    maps can run in one launch."""
    if self.op == "move": x, offset, strides, lo, hi = self.srcs[0], self.map[0], *_split(self.map[1:], len(self.shape))
    else: x, offset, strides, lo, hi = self, 0, _strides(self.shape), (0,) * len(self.shape), self.shape
    inside = [_inside(o, s, lo[k], hi[k], n) for k, o, s, n in zip(axes, starts, steps, shape)]
    return x.move(shape, offset + sum(strides[k] * o for k, o in zip(axes, starts)),
                  [strides[k] * s for k, s in zip(axes, steps)], *zip(*inside))
  def reshape(self, *shape):  # up to five sizes. missing leading sizes are 1
    shape = _shape(shape)
    if math.prod(shape) != math.prod(self.shape): raise ValueError(f"cannot reshape {self.shape} to {shape}")
    if shape == self.shape: return self
    if shape[-2:] == self.shape[-2:]: return self._relabel(shape[:-2])  # the same matrices, under other leading sizes
    return self.move(shape, 0, _strides(shape))._track("reshape", (self,))
  def permute(self, *order):
    n = len(self.shape)
    if sorted(order) != list(range(n)):
      raise ValueError(f"permute needs an order of the axes 0 to {n - 1}, not {order}")
    if tuple(order) == (*range(n - 2), n - 1, n - 2): return self.T
    if tuple(order) == tuple(range(n)): return self
    return self._map(tuple(self.shape[k] for k in order), order, (0,) * n, (1,) * n)._track("permute", (self,), order)
  def expand(self, *shape):  # repeats dimensions of 1. missing leading sizes are 1
    shape = _shape(shape)
    if _broadcast(self.shape, shape) != shape: raise ValueError(f"cannot expand {self.shape} to {shape}")
    if shape == self.shape: return self
    x, n = self._relabel((1,) * (len(shape) - len(self.shape)) + self.lead), len(shape)  # with shape's dimensions
    return x._map(shape, range(n), (0,) * n, [int(a == b) for a, b in zip(x.shape, shape)])._track("expand", (self,))
  def pad(self, widths):  # zeros, with (before, after) for each of the last len(widths) axes, like numpy
    try: widths = tuple(tuple(map(operator.index, w)) for w in widths)
    except TypeError: raise ValueError("pad needs integer (before, after) pairs") from None
    n = len(self.shape)
    if len(widths) > n or any(len(w) != 2 for w in widths):
      raise ValueError(f"pad needs at most {n} (before, after) pairs, one for each dimension")
    widths = ((0, 0),) * (n - len(widths)) + widths
    shape = tuple(m + w[0] + w[1] for m, w in zip(self.shape, widths))
    return self._map(shape, range(n), [-w[0] for w in widths], (1,) * n)._track("pad", (self,), widths)

  def cat(self, *xs, axis=0):  # like tinygrad: pad each with zeros to its position along axis, then sum
    self._same_runtime(*xs)
    axis, n = operator.index(axis), len(self.shape)
    if not -n <= axis < n: raise ValueError(f"axis {axis} outside {self.shape}")
    ts, axis = (self, *xs), axis % n
    if not all(len(t.shape) == n and t.shape[a] == self.shape[a] for t in xs for a in range(n) if a != axis):
      raise ValueError("cat requires matching dimensions outside its axis")
    m, out, at = sum(t.shape[axis] for t in ts), None, 0
    for t in ts:
      widths = [(0, 0)] * n
      widths[axis] = (at, m - at - t.shape[axis])
      out, at = t.pad(widths) if out is None else out + t.pad(widths), at + t.shape[axis]
    return out

  @_matrices
  def repeat_interleave(self, n):
    """Repeat each matrix n times along the batch without copying: kernels read matrix b from the matrix stored at
    b // n. It's how grouped-query attention shares key and value heads."""
    if self.rep > 1: return (self + 0).repeat_interleave(n)
    v = self._view(self.rows, self.cols, self.batch * n, 0, self.bstride)
    v.rep = n
    return v._track("repeat", (self,), n)

  @_matrices
  def split(self, n, axis=2):
    """Split along axis (2 for columns, 1 for rows) into n equal tile-aligned parts and view them as a batch.
    No data is copied. Split an unrealized sum of realized tensors term by term."""
    n, axis = operator.index(n), operator.index(axis)
    if n <= 0: raise ValueError("split needs a positive number of parts")
    if terms := _terms(self): return sum((t.split(n, axis) for t in terms[1:]), terms[0].split(n, axis))
    if axis not in (1, 2): raise ValueError(f"split cuts columns (axis 2) or rows (axis 1), not axis {axis}")
    if self.runs:  # by columns, split cuts K: each part reads further along the same runs
      if not (axis == 1 and self.batch == 1 and self.rows % (32 * n) == 0):
        raise ValueError("a tensor by columns splits K, with batch 1, into tile-aligned parts")
      return self._view(self.rows // n, self.cols, n, 0,
                        runs=(0, self.runs[1], self.rows // n // 32))._track("split", (self,), axis)
    if axis == 2:
      if not (self.batch == 1 and self.cols % (32 * n) == 0):
        raise ValueError("split needs batch 1 and tile-aligned parts")
      return self._view(self.rows, self.cols // n, n, 0, self.cols // n // 32)._track("split", (self,), axis)
    if not (self.rows % (32 * n) == 0 and (self.batch == 1 or self.bstride == self.rows // 32 * self.stride)):
      raise ValueError("split along rows needs tile-aligned parts of matrices stored one after another")
    return self._view(self.rows // n, self.cols, self.batch * n, 0,
                      self.rows // n // 32 * self.stride)._track("split", (self,), axis)

  @_matrices
  def merge(self):
    """Place batch matrices side by side, reversing split: x.split(n).merge() is x. An unrealized expression
    writes directly into the result columns. Otherwise, copy the matrices there."""
    if self.cols % 32: raise ValueError(f"merge needs matrices a whole number of tiles wide, not {self.cols} columns")
    out = Tensor(self.rt, self.rows, self.batch * self.cols, dtype=self.dtype)
    # + 0 copies a realized tensor, T or move in
    (self if self.op not in (None, "T", "move") else self + 0).realize(out.split(self.batch))
    return out._track("merge", (self,))

  @property
  def T(self):
    """Transpose each matrix. A matmul can read this as B without a copy, transposing tiles during unpack.
    Other uses run transpose_kernel, or move_kernel for fp32 and int32 because the unpacker can only
    transpose into SrcA."""
    if self.op == "T": return self.srcs[0]
    return _lead(Tensor(self.rt, self.cols, self.rows, "T", (self,), self.batch, self.dtype), self.lead)

  def realize(self, into=None):
    """Queue the computation and return this tensor. Allocate DRAM for the result, or write into the tensor or
    view passed as into. A tensor that's already realized has nothing to write; assign copies one."""
    self.rt.check()
    if into is not None:
      self._same_runtime(into)
      if not (into.op is None and into.rep == 1 and not into.runs
              and (into.shape, into.dtype) == (self.shape, self.dtype)):
        raise ValueError("realize needs a realized destination of the same shape and dtype, "
                         "not a repeat or a tensor by columns")
      if self.op is None:
        raise ValueError("realize writes into a destination only from an unrealized tensor. "
                         "assign copies a realized one")
      _writes(into)
    if self.op is None: return self
    if self.op in ("lead", "spread"):  # a view of its source's matrices, with no launch, or a spread copied out
      x = self.srcs[0] if self.op == "lead" else self + 0
      to = None if into is None else into._relabel(x.lead)
      x = x.realize() if to is None else to.assign(x)
      self._alias(x)
      self.op, self.srcs, self.rep, self.runs = None, (), x.rep, x.runs
      return self
    if self.op in ("T", "move"):  # data movement, into a tensor of its own
      if into is not None: raise ValueError("a transpose or a move writes a tensor of its own")
      if self.dtype in BFP:
        raise ValueError("a block float moves only as a matmul's B. Cast it to bf16 to transpose or move it")
      x, op, d = _plain(self.srcs[0].realize()), self.op, self.dtype
      src = x._where()  # check the source before this tensor counts as realized
      self.addr, self.op, self.srcs = self._alloc(), None, ()
      where = (self.addr, *src)
      if op == "T" and d == "bf16":
        return self.run(("transpose", x.rows, x.cols), lambda: transpose_kernel(x.rows, x.cols), *where,
                        name="transpose")
      if op == "T": self.map = (0, x.rows * x.cols, 1, x.cols, 0, 0, 0, *self._dims)
      lead = (len(self.map) - 1) // 3 - 2  # the map's leading axes: one, the batch, for a transpose
      return self.run(("move", d, lead), lambda: move_kernel(d, lead), *where, x.rows, x.cols, self.rows, self.cols,
                      *(v & 0xFFFFFFFF for v in self.map), *self.lead[1:lead], name="move")
    _fit(self)
    inputs = []
    e, dtype = _tree(self, inputs)
    matmul = any(u.op == "matmul" for u in _lazy(self, set()))
    if into is not None and any(t.addr == into.addr and (_layout(t) != _layout(into) or matmul) for t in inputs):
      into.assign(self.realize())  # finish every source read before any destination tile is overwritten
      self.rt.dram_free(self.addr)
      self._alias(into)
      return self
    shapes = tuple(u._dims for u in inputs)
    # when only the result's rows matter, round the inputs' rows up to tiles, so the same kernel serves any row count
    if _rows_free(e): shapes = tuple((b, r if r == 1 else -(-r // 32) * 32, c) for b, r, c in shapes)
    reps = tuple((u.rep, u.cyc) if u.cyc else u.rep for u in inputs)
    key = (e, shapes, dtype, self.dtype, self.rt.grid, reps, self.rt.readers)
    kernel, units, name = _cached(key, lambda: (*fused_kernel(*key), _name(e)))
    if into is not None: self._alias(into)
    else: self.addr = self._alloc()
    self.op, self.srcs = None, ()
    # _where(True) is safe: _expr only lets a tensor by columns in as a matmul's B
    args = [v for t in [self] + inputs for v in t._where(True)] + [self.rows % 32 if self.rows > 1 else 0]
    self.rt.launch(kernel, tuple(args), units or self.tiles, name)
    return self
  def assign(self, x):
    """Write x into this tensor in place, as an optimizer updates a weight. An unrealized x is computed into this
    tensor's DRAM. A matmul with an overlapping input, or an overlapping view with a different layout, computes
    its source before writing. Moves and transposes compute into their own storage first."""
    self._same_runtime(x)
    if not (self.op is None and self.rep == 1 and (x.shape, x.dtype) == (self.shape, self.dtype)):
      raise ValueError("assign requires a realized tensor with rep=1 and an x of the same shape and dtype")
    _writes(self)
    if x.op in ("T", "move"): x = x.realize()
    if x.op is not None: x.realize(self); return self
    if x.addr == self.addr and _layout(x) != _layout(self): x = _copy(x)
    return self.run(("copy", self.dtype), lambda: copy_kernel(self.dtype), *self._where(), *x._where(),
                    -(-self.rows // 32), -(-self.cols // 32), x.rep, name="copy")

  def run(self, key, make, *args, units=None, name=""):
    """Build a kernel with make(), cache it by key, and queue it over units (tiles by default). Realize tensors
    in args and pass their address, batch stride, row stride and offset. Launches execute in order; this
    call doesn't wait for completion."""
    self._same_runtime(*(a for a in args if isinstance(a, Tensor)))
    # the list keeps each copy alive until it's queued
    args = [_plain(a.realize()) if isinstance(a, Tensor) else a for a in args]
    units = self.tiles if units is None else units
    self.rt.launch(_cached(key, make), tuple(v for a in args for v in (a._where() if isinstance(a, Tensor) else (a,))),
                   units, name)
    return self

  @property
  def shape(self): return (*self.lead, self.rows, self.cols)
  @property
  def _dims(self): return (self.batch, self.rows, self.cols)  # the shape that kernels and views work in

  def __bool__(self):
    if math.prod(self.shape) != 1:
      raise ValueError("the truth value of a tensor with more than one element is ambiguous")
    return bool(self.reshape(1, 1).tolist()[0][0])

  # Elementwise ops: tinygrad's primitives, and the rest built from them. exp, log and cos are SFPU programs too, so
  # their constants stay fp32. As numbers they'd be bf16 tensors, and exp(10) as exp2(10 * log2(e)) would be 2% off.
  def _unary(self, op):  # int32 goes through fp32, except relu, which is a max with 0
    if self.dtype == "int32" and op != "relu": return self.float()._unary(op)
    return _lead(Tensor(self.rt, self.rows, self.cols, op, (self,), self.batch, self.dtype), self.lead)
  def number(self, v, dtype):  # a 1 x 1 tensor of v, uploaded once per runtime, so a number costs no PCIe write
    key = (repr(v), dtype)  # repr tells -0.0 from 0.0, and a NaN matches itself
    if key not in self.rt.numbers: self.rt.numbers[key] = Tensor.full(self.rt, 1, 1, v, dtype)
    return self.rt.numbers[key]
  def _binop(self, op, x, reverse=False):
    """numpy broadcasting. A number becomes a 1 x 1 tensor of this dtype (fp32 for a float with int32). Mixed dtypes
    meet in fp32, and so does int32 for ops that have no int32 form, like mul, which is then exact below 2^24."""
    if not isinstance(x, Tensor):
      x = self.number(x, "fp32" if self.dtype == "int32" and isinstance(x, float) else self.dtype)
    a, b = (x, self) if reverse else (self, x)
    if a.dtype != b.dtype or (a.dtype == "int32" and op not in INT32_OPS): a, b = a.float(), b.float()
    if (shape := _broadcast(a.shape, b.shape)) is None: raise ValueError(f"cannot broadcast {a.shape} with {b.shape}")
    batch, lead = math.prod(shape[:-2]), shape[:-2]
    a, b = (u if u.batch in (1, batch) else u._spread(lead) for u in (a, b))  # with some of the leading sizes
    return _lead(Tensor(self.rt, *shape[-2:], op, (a, b), batch, a.dtype), lead)
  def cast(self, dtype):  # bfp4 isn't lazy: bfp4_kernel makes it now from the bf16 value, and says why
    if dtype not in (*ITEMSIZE, *BFP): raise ValueError(f"unknown tensor dtype: {dtype}")
    if dtype == self.dtype and not self.runs: return self
    if self.dtype in BFP or self.runs:
      x, out, tc = self.realize(), Tensor(self.rt, self.rows, self.cols, batch=self.batch), -(-self.cols // 32)
      out.lead, key = self.lead, ("storage", self.dtype, bool(self.runs))
      return out.run(key, lambda: storage_kernel(*key[1:]), out.addr, *x._where(True), tc, self.tiles // self.batch,
                     self.rep,
                     name="storage")._track("cast", (self,)).cast(dtype)
    if dtype != "bfp4":
      return _lead(Tensor(self.rt, self.rows, self.cols, "cast", (self,), self.batch, dtype), self.lead)
    x, out, tc = (_plain(self.cast("bf16").realize()),
                  Tensor(self.rt, self.rows, self.cols, batch=self.batch, dtype="bfp4"), -(-self.cols // 32))
    out.lead = self.lead
    return out.run(("bfp4",), bfp4_kernel, out.addr, *x._where(), tc,
                   -(-self.rows // 32) * tc, name="bfp4")._track("cast", (self,))
  @_matrices
  def columns(self):
    """Store each tile column of B together so matmul can read several tiles along K per request.
    Tile column j goes to bank j % 8, with block-float exponents before values. The result supports
    readback and use as matmul's B, including indexed selection and splits along K. Cast to bf16 for ordinary
    operations."""
    if self.dtype not in ("bf16", *BFP):
      raise ValueError(f"columns lays out a matmul's B, which is bf16 or a block float, not {self.dtype}")
    x, tr, tc = _plain(self.realize()), -(-self.rows // 32), -(-self.cols // 32)
    runs = -(-tc // len(bh.DRAM))  # runs per matrix in each bank
    out = Tensor(self.rt, self.rows, self.cols, "view", batch=self.batch, dtype=self.dtype)
    out.op, out.runs, out.addr = (None, (runs, tr, 0),
                                  self.rt.dram_alloc(self.batch * runs * tr * tile_bytes(self.dtype)))
    return out.run(("columns", self.dtype), lambda: columns_kernel(self.dtype), out.addr, x, tc, tr * tc, runs, tr,
                   name="columns")._track("columns", (self,))
  def float(self): return self.cast("fp32")
  def bfloat16(self): return self.cast("bf16")
  def exp2(self): return self._unary("exp2")
  def log2(self): return self._unary("log2")
  def sin(self): return self._unary("sin")
  def sqrt(self): return self._unary("sqrt")
  def reciprocal(self): return self._unary("recip")
  def exp(self): return self._unary("exp")
  def log(self): return self._unary("log")
  def cos(self): return self._unary("cos")
  def relu(self): return self._unary("relu")
  def rsqrt(self): return self.sqrt().reciprocal()
  def maximum(self, x): return self._binop("max", x)
  def minimum(self, x):  # -max(-a, -b). Negating -2**31 overflows, so int32 complements instead: ~v = -1 - v
    if self.dtype == "int32" and (x.dtype == "int32" if isinstance(x, Tensor) else not isinstance(x, float)):
      return -1 - (-1 - self).maximum(-1 - x)
    return -((-self).maximum(-x))
  def where(self, x, y):
    """x where self isn't 0, else y. A condition of another dtype, or an int32 one that isn't a comparison's 0 or 1,
    becomes 0 or 1 in theirs first, since the SFPU tests its bits as a float's, where -0 is false."""
    dtype = next((t.dtype for t in (x, y) if isinstance(t, Tensor)), self.dtype)
    c = (self if self.dtype == dtype and (dtype != "int32" or self.op in ("cmplt", "cmple", "cmpne")) else
         self.ne(0).cast(dtype))
    return c._binop("keep", x)._binop("join", c._binop("drop", y))
  def __add__(self, x): return self._binop("add", x)
  def __sub__(self, x): return self._binop("sub", x)
  def __mul__(self, x): return self._binop("mul", x)
  def __rsub__(self, x): return self._binop("sub", x, True)
  __radd__, __rmul__ = __add__, __mul__
  def __truediv__(self, x): return self * (1 / x if isinstance(x, (int, float)) else x.reciprocal())
  # int32 subtracts exactly but multiplies through fp32
  def __neg__(self): return 0 - self if self.dtype == "int32" else self * -1
  def __lt__(self, x): return self._binop("cmplt", x)
  def __gt__(self, x): return self._binop("cmplt", x, True)
  def __ge__(self, x): return self._binop("cmple", x, True)
  def __le__(self, x): return self._binop("cmple", x)
  def ne(self, x): return self._binop("cmpne", x)
  def eq(self, x): return 1 - self.ne(x)
  __ne__, __eq__ = ne, eq
  def __hash__(self): return id(self)  # == is elementwise, so hash by identity, like tinygrad
  def sigmoid(self): return (1 + (-self).exp()).reciprocal()
  def silu(self): return self * self.sigmoid()
  # GPT-2's tanh form: 0.5 x (1 + tanh(z)) with z = sqrt(2/pi) (x + 0.044715 x^3), which is x / (1 + e^-2z)
  def gelu(self):
    k = 2 * math.sqrt(2 / math.pi)
    return self * (1 + (self * (self * self * (-0.044715 * k) - k)).exp()).reciprocal()

  # Reductions, and ops built from them
  def sum(self, axis=None): return self._reduce("sum", axis)
  def max(self, axis=None): return self._reduce("max", axis)
  def _reduce(self, f, axis):
    """Reduce like numpy with keepdims: over a leading dimension, down the rows (axis -2), along each row (-1), or
    over everything for None. Each axis is its own kernel, which sweeps lines of tiles along it: the batch, for a
    leading dimension, which the other leading dimensions leave as the whole batch."""
    n = len(self.shape)
    if axis is not None: axis = operator.index(axis)
    if axis is not None and not -n <= axis < n: raise ValueError(f"axis {axis} outside {self.shape}")
    if self.dtype == "int32": return self.float()._reduce(f, axis).cast("int32")  # exact below 2^24
    if axis is None:
      flat = self._relabel((self.batch,))
      return flat._reduce(f, 0)._reduce(f, 1)._reduce(f, 2)._relabel((1,) * len(self.lead))
    axis, lead = axis % n, list(self.lead)
    if axis < n - 2 and self.batch != lead[axis]:
      raise ValueError(f"reducing axis {axis} of {self.shape} needs its other leading sizes 1")
    if axis < n - 2: lead[axis] = 1
    k = 0 if axis < n - 2 else axis - (n - 3)  # the axis of (batch, rows, cols) that the kernel folds
    shape = [1 if ax == k else s for ax, s in enumerate(self._dims)]
    if shape == list(self._dims): return self
    return _lead(Tensor(self.rt, shape[1], shape[2], f"{f}{k}", (self,), shape[0], self.dtype), lead)
  def mean(self, axis=None):
    s = self.sum(axis)
    return s * (math.prod(s.shape) / math.prod(self.shape))
  @_matrices
  def topk(self, k, parts=1, labels=None):
    """Return (values, int32 indices) for the k largest entries in each row, descending. Ties keep column
    order. BRISC compares bf16 or fp32 values exactly, with k <= 64 and no gradient. For rows too wide for
    L1, parts returns k entries per part, labelled by original column. Run topk again with those labels to
    combine the parts."""
    if labels is not None: self._same_runtime(labels)
    k, parts = operator.index(k), operator.index(parts)
    if k <= 0 or parts <= 0: raise ValueError("topk needs positive k and parts")
    cols = self.cols // parts
    if not (self.dtype in ("bf16", "fp32") and k <= min(cols, 64) and parts * cols == self.cols):
      raise ValueError(f"topk needs bf16 or fp32, k up to 64 and a part's width, "
                       f"and equal parts: {self.dtype}, k={k}, {parts} parts of {self.cols}")
    if labels is not None and not (labels.dtype == "int32" and labels.shape == self.shape and parts == 1):
      raise ValueError("topk's labels need int32, x's shape and parts=1")
    x, labels = _plain(self.realize()), None if labels is None else _plain(labels.realize())
    values = Tensor(self.rt, x.rows, parts * k, batch=x.batch, dtype=x.dtype)
    ids = Tensor(self.rt, x.rows, parts * k, batch=x.batch, dtype="int32")
    key = ("topk", k, cols, x.dtype, parts, labels is not None)
    return values.run(key, lambda: topk_kernel(*key[1:]), values.addr, ids.addr, *x._where(), x.rows,
                      *(labels._where() if labels is not None else ()), units=x.batch * parts * -(-x.rows // 32) * 32,
                      name="topk"), ids
  @_matrices
  def to(self, rt, ch, back):
    """Copy to rt's chip over Ethernet channel ch, with channel back at the peer (see Chip.links). Wait for
    queued launches on both chips before clearing the transfer flags. This can't be recorded because replay
    wouldn't clear those flags again."""
    if rt is self.rt or self.rt.log is not None or rt.log is not None:
      raise ValueError("to requires different runtimes with neither recording launches")
    x = self.realize()
    if x.runs: raise ValueError("cast a tensor by columns to bf16 before transferring it to another chip")
    # transfer a view from its own storage, preserving its bits
    if x.owner is not None or x.rep > 1 or x.offset: x = _copy(x)
    out = Tensor(rt, x.rows, x.cols, batch=x.batch, dtype=x.dtype)
    x.rt.wait(), rt.wait()
    flags, n = link_flags(x.dtype)
    for chip, end in ((x.rt.chip, ch), (rt.chip, back)): chip.write(*bh.ETH[end], flags, bytes(n))
    out.run(("link", False, x.dtype), lambda: link_kernel(False, x.dtype), out.addr, out.tiles, *bh.ETH[back], units=1,
            name="link in")
    x.run(("link", True, x.dtype), lambda: link_kernel(True, x.dtype), x.addr, x.tiles, *bh.ETH[ch], units=1,
          name="link out")
    return out
  def softmax(self, axis=-1):  # max first, so exp stays in range
    e = (self - self.max(axis)).exp()
    return e / e.sum(axis)
  def log_softmax(self, axis=-1):
    m = self - self.max(axis)
    return m - m.exp().sum(axis).log()
  def layernorm(self, axis=-1, eps=1e-5):  # to mean 0 and variance 1 along axis
    c = self - self.mean(axis)
    return c * ((c * c).mean(axis) + eps).rsqrt()

  def __matmul__(self, x): return self.matmul(x)
  def matmul(self, x, dtype=None, fidelity=None, index=None):
    """Multiply matrices by batch, repeating a batch of one. Products accumulate in fp32 Dst. With bf16 self, x
    may be bfp8 or bfp4 to reduce bytes read.

    An fp32 operand is split into bf16 hi and lo. Three bf16 matmuls compute hi@hi, hi@lo and lo@hi in fp32, and
    an add sums them, preserving about 16 of fp32's 24 bits. The FPU aligns groups of eight products before adding,
    with error around 2^-11 of the largest terms. See docs/compiler.md, "Matmul precision".

    index is an int32 (rows, k) tensor. For batch entry b, index[b // k, b % k] selects a matrix from x.
    Indices are read on the chip so replay can select different experts. An index >= x.batch gives zero
    without reading a matrix. A matmul by index has no gradient."""
    self._same_runtime(x, *((index,) if index is not None else ()))
    if len(self.shape) > 3 or len(x.shape) > 3:  # the leading dimensions broadcast, and the batches multiply
      if (lead := _broadcast(self.lead, x.lead)) is None:
        raise ValueError(f"matmul needs leading sizes that broadcast: {self.shape} @ {x.shape}")
      a, b = (u if u.batch in (1, math.prod(lead)) else u._spread(lead) for u in (self, x))
      return a._relabel((a.batch,)).matmul(b._relabel((b.batch,)), dtype, fidelity, index)._relabel(lead)
    if dtype == "int32": return self.matmul(x, "fp32", fidelity, index).cast("int32")
    fidelity = FIDELITY if fidelity is None else operator.index(fidelity)
    if not 1 <= fidelity <= 4: raise ValueError("matmul fidelity must be between one and four phases")
    if index is not None:
      if not (index.dtype == "int32" and self.dtype == "bf16" and x.dtype in ("bf16", *BFP) and self.cols == x.rows
              and self.batch in (1, index.rows * index.cols)):
        raise ValueError(f"matmul by index needs bf16 A, "
                         f"a bf16 or block-float B and int32 indices to fit: {self.shape} {self.dtype} @ "
                         f"{x.shape} {x.dtype}, {index.shape} {index.dtype}")
      out = Tensor(self.rt, self.rows, x.cols, "matmul", (self, x, index), index.rows * index.cols, dtype or "bf16")
      out.fidelity = fidelity
      return out
    # an unrealized sum times a small matrix (16 tiles at most, like a 128-wide head's turn) goes term by term, as when
    # RoPE turns the parts of a split K
    if x.tiles <= 16 and (terms := _terms(self)):
      return sum((t.matmul(x, dtype, fidelity) for t in terms[1:]), terms[0].matmul(x, dtype, fidelity))
    if "int32" in (self.dtype, x.dtype):
      raise ValueError(f"matmul needs floats, since the FPU has no int32: {self.dtype} @ {x.dtype}")
    dtype = dtype or ("fp32" if "fp32" in (self.dtype, x.dtype) else "bf16")
    # autograd sees one op, and keeps the bf16 parts for the two fp32 matmuls of backward
    if "fp32" in (self.dtype, x.dtype):
      a, b = _parts(self), _parts(x)
      return _bf16x3(a, b)._track("matmul", (self, x), a, b).cast(dtype)
    if not (self.cols == x.rows and (self.batch == x.batch or 1 in (self.batch, x.batch))):
      raise ValueError(f"matmul needs A's columns to match B's rows, "
                       f"and batches that match or broadcast: {self.shape} @ {x.shape}")
    if self.cols == 1: return (self * x).cast(dtype)  # (R, 1) @ (1, C) is broadcasting
    k, tr = -(-self.cols // 32), -(-self.rows // 32)
    # A row of tiles, like a token's MLP, has too few blocks to keep the workers busy. Cut K into s parts of at least 4
    # tiles as a batch, then sum them. With more rows this doesn't help: the matrices of a batch read their own tiles,
    # while blocks of the same matrix share them.
    s = max([s for s in (2, 4, 8) if self.batch == x.batch == 1 and tr == 1 and self.cols % (32 * s) == 0
             and k // s >= 4
             and s * -(-x.cols // 32) <= len(self.rt.cores)], default=1)
    if s > 1:
      b = x.srcs[0].split(s).T if x.op == "T" else x.split(s, 1)
      parts = self.split(s).matmul(b, dtype, fidelity)
      return sum((parts[i] for i in range(1, s)), parts[0])
    out = Tensor(self.rt, self.rows, x.cols, "matmul", (self, x), max(self.batch, x.batch), dtype)
    out.fidelity = fidelity
    return out

  @_matrices
  def attend(self, k, v, places, window=0, fidelity=None, parts=None):
    """Compute softmax(self @ k^T + mask) @ v using an attention kernel followed by normalization. Keep scores
    in L1. No gradient.

    k and v may use block-float storage or repeat views. places is an int32 column giving the last visible
    key per query row. With window, the first visible key is window - 1 positions earlier. Read only keys
    within these bounds. parts divides keys between workers when decode has too few units. Key length and
    query row count are kernel arguments, so one compiled kernel handles several lengths."""
    self._same_runtime(k, v, places)
    fidelity = FIDELITY if fidelity is None else operator.index(fidelity)
    if not 1 <= fidelity <= 4: raise ValueError("attention fidelity must be between one and four phases")
    if window < 0 or parts is not None and parts <= 0:
      raise ValueError("attention needs a nonnegative window and positive parts")
    ht, kt, rep, ra = -(-self.cols // 32), -(-k.rows // 32), k.rep, -(-self.rows // 32)
    if not (self.dtype == "bf16" and k.dtype == v.dtype and k.dtype in ("bf16", *BFP) and v.rep == rep
            and self.batch == k.batch == v.batch):
      raise ValueError(f"attend needs bf16 queries, keys and values in one format, repeated alike, "
                       f"equal batches: {self.dtype} {k.dtype} {v.dtype}")
    if not (self.cols == k.cols == v.cols and self.cols % 32 == 0 and self.cols >= 64 and k.rows == v.rows):
      raise ValueError(f"attend needs heads of one width, 64 or more in steps of 32, "
                       f"and a value for each key: {self.shape} {k.shape} {v.shape}")
    if not (places.rows == self.rows and places.batch in (1, self.batch) and places.cols == 1
            and places.dtype == "int32"):
      raise ValueError(f"attend needs places as an int32 column with a row for "
                       f"each query: {places.shape} {places.dtype} for {self.shape}")
    wt = k.dtype if k.dtype in BFP else None
    R, units, P, kept = attention_plan(ht, ra, self.batch, wt, kt, len(self.rt.cores), parts)
    q, places = (_plain(t.realize()) for t in (self, places))
    k, v = k.realize(), v.realize()
    out = Tensor(self.rt, self.rows, self.cols + (33 if P > 1 else 1), batch=self.batch * P, dtype="fp32")
    # the exact row count is an argument, so the key rounds it up to tiles
    key = ("attend", ht, ra, R, rep, wt, fidelity, window, P, kept > 0)
    out.run(key, lambda: attention_kernel(*key[1:]), *out._where(), *q._where(), *k._where(), *v._where(),
            *places._where(), kt,
            self.rows, units=units * P, name="attention")
    # cols of part p, from tile at
    part = lambda p, at, cols: out._view(out.rows, cols, self.batch, p * out.bstride + at, out.bstride * P)
    # each sum fills its tile's 32 columns, a valid 1-column view
    if P == 1: return (part(0, 0, self.cols) / part(0, ht, 1)).cast("bf16")
    largest = [part(p, ht + 1, 1) for p in range(P)]
    top = functools.reduce(Tensor.maximum, largest)
    weighted = lambda at, cols: functools.reduce(Tensor.__add__,
                                                 [(m - top).exp() * part(p, at, cols) for p, m in enumerate(largest)])
    return (weighted(0, self.cols) / weighted(ht, 1)).cast("bf16")

  # Convolution and pooling of an image (channels, height, width), built from the ops above so autograd works on them
  @_matrices
  def conv2d(self, w, padding=0):
    """Convolve with w, (out channels, in channels, k * k), as one im2col matmul. Tap (ky, kx) reads each channel,
    flattened to a row, from ky * W + kx on, so column y * W + x of the stacked taps is the patch at (y, x). Columns
    that wrap past the right edge are dropped."""
    k = math.isqrt(w.cols)
    if not (k * k == w.cols and w.rows == self.batch):
      raise ValueError(f"conv2d needs w as (out channels, in channels, k * k), "
                       f"an in channel for each of x's: {self.shape} by {w.shape}")
    x = self.pad(((padding, padding), (padding, padding))) if padding else self
    (c, h, wd), ho, wo = x.shape, x.rows - k + 1, x.cols - k + 1
    # k - 1 zeros at the end, for the wrapped columns of the last tap
    rows = x.reshape(1, c, h * wd).pad(((0, 0), (0, k - 1)))
    taps = [rows[:, :, ky * wd + kx:ky * wd + kx + ho * wd] for ky in range(k) for kx in range(k)]
    out = w.permute(0, 2, 1).reshape(1, w.batch, k * k * c) @ taps[0].cat(*taps[1:], axis=1)
    return out.reshape(w.batch, ho, wd)[:, :, :wo]
  @_matrices
  def max_pool2d(self, k=2):  # max of each k x k block: over each k rows, then each k columns
    c, h, w = self.shape
    if not (h % k == 0 and w % k == 0):
      raise ValueError(f"max_pool2d needs a height and width that {k} divides: {self.shape}")
    return self.reshape(c * h // k, k, w).max(1).reshape(1, c * h * w // (k * k), k).max(2).reshape(c, h // k, w // k)

  # Autograd, like tinygrad's: a rule per op (_backward), made of the same lazy ops, so the backward fuses into kernels
  # too. It recomputes what the forward fused, since that wasn't stored.
  def gradient(self, *targets, gradient=None):
    """Build d(self)/d(target) for each target as lazy tensor operations. Return None for a target that
    self doesn't depend on. self must be scalar unless gradient supplies the derivative of the desired
    output with respect to self. Realize all returned gradients before changing any forward operand;
    backward() does this automatically."""
    if gradient is None and math.prod(self.shape) != 1:
      raise ValueError("self must be a single value, or pass gradient")
    if gradient is not None and gradient.shape != self.shape: raise ValueError("gradient must have the output's shape")
    if gradient is not None: self._same_runtime(gradient)
    one = Tensor.full(self.rt, 1, 1, 1.0, self.dtype)._relabel(self.lead) if gradient is None else None
    grads = {id(self): gradient if gradient is not None else one}
    wanted, found = {id(t) for t in targets}, {}
    for t in reversed(_graph(self)):
      if t._ctx is None: continue
      if (g := grads.pop(id(t), None)) is None: continue
      if id(t) in wanted: found[id(t)] = g
      for s, gs in zip(t._ctx[1], _backward(t, g)):
        if gs is not None and s.requires_grad: grads[id(s)] = grads[id(s)] + gs if id(s) in grads else gs
    grads.update(found)
    return [_full(grads[id(t)], t.shape) if id(t) in grads else None for t in targets]
  def backward(self, gradient=None):
    """Accumulate gradients in .grad for leaves that require them, and return self. Existing gradients are
    added to, as in PyTorch and tinygrad. Queue the forward value and all gradients before returning, so
    later parameter updates cannot change the values used by another parameter's gradient."""
    leaves = [t for t in _graph(self) if t._ctx is None]
    grads = self.gradient(*leaves, gradient=gradient)
    self.realize()
    for t, g in zip(leaves, grads):
      if g is not None: t.grad = (g if t.grad is None else t.grad + g).realize()
    return self

def _shape(shape):
  if not 1 <= len(shape) <= DIMS: raise ValueError(f"a tensor needs one to {DIMS} dimensions")
  shape = tuple(map(operator.index, shape))
  if any(n <= 0 for n in shape): raise ValueError(f"tensor dimensions must be positive: {shape}")
  return (1,) * (3 - len(shape)) + shape

def _unbroadcast(g, shape, full=None):
  """The gradient of an operand of shape that an op broadcast to full: g summed over the axes where shape is 1 and g
  isn't. Where g is 1 as well, as a sum's gradient is, each of full's copies along the axis has g, so they add up to
  g times their number."""
  pad = lambda s: (1,) * (max(len(g.shape), len(full or shape)) - len(s)) + tuple(s)
  copies = math.prod(n for n, a, b in zip(pad(full or shape), pad(g.shape), pad(shape)) if a == b == 1)
  lead = (1,) * (len(g.shape) - len(shape)) + shape[:-2]  # shape's leading sizes, one for each of g's
  if math.prod(lead) == 1 < g.batch: g = g._relabel((g.batch,)).sum(0)  # one matrix for all of g's: one sweep
  for ax in range(len(g.shape) - 2):
    if lead[ax] == 1 and g.shape[ax] > 1: g = g.sum(ax)
  for ax in (-2, -1):
    if shape[ax] == 1 and g.shape[ax] > 1: g = g.sum(ax)
  g = g._relabel(shape[:-2] if g.batch == math.prod(shape[:-2]) else (1,) * (len(shape) - 2))
  return g * copies if copies > 1 else g
def _broadcast(*shapes):  # numpy's broadcast of shapes, or None where they don't broadcast
  n = max(map(len, shapes))
  sizes = list(zip(*((1,) * (n - len(s)) + tuple(s) for s in shapes)))
  return None if any(len(set(z) - {1}) > 1 for z in sizes) else tuple(map(max, sizes))
def _lead(t, lead):  # t, a tensor this call made, with leading dimensions lead
  t.lead = tuple(lead)
  return t
def _full(g, shape): return g if g.shape == shape else g.expand(*shape)  # for ops that can't take g broadcast
def _strides(shape): return tuple(math.prod(shape[k + 1:]) for k in range(len(shape)))  # in row-major order
def _split(values, n): return values[:n], values[n:2 * n], values[2 * n:]  # a move's strides, lo and hi
def _layout(t): return t._dims, t.bstride, t.stride, t.offset, t.rep, t.cyc, t.runs

def _graph(t):  # autograd's operands before their users, without recursion through long histories
  order, seen, stack = [], set(), [(t, False)]
  while stack:
    u, done = stack.pop()
    if done: order.append(u)
    elif id(u) not in seen:
      seen.add(id(u))
      stack.append((u, True))
      stack.extend((s, False) for s in reversed(u._ctx[1] if u._ctx else ()) if s.requires_grad)
  return order

def _backward(t, g):
  """The gradients of t's operands, given t's gradient g. A condition or a comparison has none, and gets None."""
  op, srcs, args = t._ctx
  x, y = (srcs + (None,))[:2]
  if op in ("add", "sub", "mul", "max", "keep", "drop", "join", "cmplt", "cmple", "cmpne"):
    if op in ("cmplt", "cmple", "cmpne"): return (None, None)
    # halves of where: the condition has no gradient
    if op in ("keep", "drop"): return (None, _unbroadcast(x._binop(op, g), y.shape, t.shape))
    if op in ("add", "join"): gx, gy = g, g  # join combines where's disjoint masks
    elif op == "sub": gx, gy = g, -g
    elif op == "mul": gx, gy = g * y, g * x
    else: m = (x > y) + x.eq(y) * 0.5; gx, gy = g * m, g * (1 - m)  # max: ties split it, like tinygrad
    return (_unbroadcast(gx, x.shape, t.shape), _unbroadcast(gy, y.shape, t.shape))
  if op in FOLDS:  # a sum broadcasts back. a max goes to its maxima, split between ties
    if op[:3] == "sum": return (g,)
    m = x.eq(t)
    return (g * m / m.sum(FOLDS[op]),)
  unary = {"exp": lambda: g * t, "exp2": lambda: g * t * math.log(2), "log": lambda: g * x.reciprocal(),
           "log2": lambda: g * (x * math.log(2)).reciprocal(), "sin": lambda: g * x.cos(),
           "cos": lambda: -(g * x.sin()),
           "sqrt": lambda: g * (t * 2).reciprocal(), "recip": lambda: -(g * t * t), "relu": lambda: g * (x > 0),
           "cast": lambda: g.cast("bf16" if x.dtype in BFP else x.dtype), "columns": lambda: g}
  if op in unary: return (unary[op](),)
  if op == "spread": return (_unbroadcast(g, x.shape),)  # each matrix's copies add their gradients up
  if op == "lead":  # the same matrices. a gradient that's one matrix for all of them, as a sum's is, stays one
    g = g if g.batch in (1, t.batch) else _full(g, t.shape)
    return (g._relabel(x.lead if g.batch == t.batch else (1,) * len(x.lead)),)
  g = _full(g, t.shape)  # movement and matmul can't take g broadcast
  if op == "matmul" and len(srcs) == 3: raise NotImplementedError("a matmul by index has no backward")
  if op == "matmul":  # a transposed B gets its gradient as (g.T @ x).T, and the T's own rule cancels that transpose
    turn = y._ctx is not None and y._ctx[0] == "T"
    plain = y
    if y.runs or y.dtype in BFP: plain = y.srcs[0].cast("bf16").T if y.op == "T" else y.cast("bf16")
    yd = "bf16" if y.dtype in BFP else y.dtype
    if args:  # fp32: the forward's parts, transposed, and g's parts, made once for both
      gp, (xp, yp) = _parts(g), args
      gx, gy = _bf16x3(gp, _pT(yp)), _bf16x3(_pT(gp), xp) if turn else _bf16x3(_pT(xp), gp)
    else: gx, gy = g.matmul(plain.T, x.dtype), g.T.matmul(x, yd) if turn else x.T.matmul(g, yd)
    gx, gy = _unbroadcast(gx.cast(x.dtype), x.shape), gy.cast(yd)
    return gx, _unbroadcast(gy, (*y.lead, y.cols, y.rows)).T if turn else _unbroadcast(gy, y.shape)
  if op == "T": return (g.T,)
  if op == "reshape": return (g.reshape(*x.shape),)
  if op == "permute": return (g.permute(*sorted(range(len(args[0])), key=args[0].__getitem__)),)
  if op == "expand": return (_unbroadcast(g, x.shape),)
  if op == "pad": return (g._map(x.shape, range(len(x.shape)), [w[0] for w in args[0]], (1,) * len(x.shape)),)
  if op == "index": return (g.pad(((args[0], x.batch - args[0] - 1), (0, 0), (0, 0))),)
  if op == "split": return (g.merge() if args[0] == 2 else g.reshape(*x.shape),)
  if op == "repeat": return (g.reshape(x.batch, args[0], x.rows * x.cols).sum(1).reshape(*x.shape),)
  if op == "merge": return (g.split(x.batch),)
  if op == "slice":  # back into place with zeros around it. a reversed axis is reversed again
    if not all(abs(r.step) == 1 for r in args[0]): raise NotImplementedError("a slice with a step has no backward")
    g = g[tuple(slice(None, None, r.step) for r in args[0])] if any(r.step < 0 for r in args[0]) else g
    lo = [min(r) for r in args[0]]
    return (g.pad([(a, n - a - len(r)) for a, n, r in zip(lo, x.shape, args[0])]),)
  # sum each row's gradient into its source row: one-hot(ids)^T @ g, whose row i is 1 where ids are i
  if op == "gather":
    return ((Tensor.of(t.rt, [[i] for i in range(x.rows)], "int32") == args[0]).bfloat16().matmul(g, x.dtype),)
  raise NotImplementedError(f"no backward for {op}")

def _fit(t):
  """Realize subexpressions until the remaining tree fits a kernel. Realize long matmuls (K over 256) before
  reductions to avoid recomputing them on each pass. Also realize elementwise values shared with work outside
  the reduction. Above MAX_OPS, prefer the largest shared value; otherwise choose a subtree that fits along the
  heaviest branch."""
  matmuls = lambda: [v for v in _lazy(t, set()) if v.op == "matmul" and id(v) not in kept]
  indexed, block, kept = False, None, set()
  while todo := matmuls():
    u = todo[0]
    dtype = u.srcs[1].dtype
    if (len(u.srcs) == 3 and indexed) or (dtype in BFP and block is not None and dtype != block):
      # one index stream and one block-float unpack format per fused kernel. t no longer reaches u's matmuls
      u.realize()
    else:
      kept.add(id(u))
      indexed |= len(u.srcs) == 3
      if dtype in BFP: block = dtype
  if any(u.op in COMBINE for u in _lazy(t, set())):
    for u in [u for u in _lazy(t, set()) if u.op == "matmul" and u.srcs[0].cols > 256]: u.realize()
    lazy = list(_lazy(t, set()))
    reads = collections.Counter(id(s) for u in lazy for s in u.srcs)
    for x in [u.srcs[0] for u in lazy if u.op in COMBINE]:
      if reads[id(x)] > 1 and all(u.op not in COMBINE and u.op != "matmul" for u in _lazy(x, set())): x.realize()
  while _ops(t) > MAX_OPS:
    lazy = list(_lazy(t, set()))
    readers = collections.Counter(i for u in lazy for i in {id(s) for s in u.srcs})
    shared = [u for u in lazy[1:] if readers[id(u)] > 1 and _ops(u) <= MAX_OPS]
    u = max(shared, key=_ops) if shared else t
    while _ops(u) > MAX_OPS: u = max((s for s in u.srcs if s.op), key=_ops)
    u.realize()
def _terms(t):  # the terms of t, if it's an unrealized sum of realized tensors of its shape, else None
  if t.op != "add" or any(s.shape != t.shape or s.dtype != t.dtype for s in t.srcs): return None
  terms = [[s] if s.op is None else _terms(s) for s in t.srcs]
  return None if None in terms else [u for ts in terms for u in ts]
# an fp32 matmul operand as bf16 (hi, lo), lo None for bf16. a transpose splits its source, then transposes the parts
def _parts(t):
  if t.op == "T": return _pT(_parts(t.srcs[0]))
  if t.dtype == "bf16": return t, None
  # lo is t minus t rounded as the bf16 cast rounds, in one kernel
  return t.bfloat16(), (t - t._unary("ftob")).bfloat16()
def _pT(parts): return tuple(p if p is None else p.T for p in parts)
def _plain(t): return _copy(t) if t.rep > 1 else t  # copy a repeat, for a kernel that finds tiles by their position
# t in a new tensor of its own
def _copy(t): return _lead(Tensor(t.rt, t.rows, t.cols, batch=t.batch, dtype=t.dtype), t.lead).assign(t)
def _root(t): return t if t.owner is None else t.owner  # the tensor that owns t's storage
def _rows_free(e):
  """Whether the kernel needs only the output row count, supplied as an argument. Row reductions, transposed B
  and indexed B also need input row counts."""
  if isinstance(e, int): return True
  if FOLDS.get(e[0]) == 1 or e[0] == "matmul" and isinstance(e[2], tuple) and e[2][0][0] in "TI": return False
  return all(_rows_free(x) for x in e[1:] if isinstance(x, tuple))
def _bf16x3(a, b):  # hi@hi + (hi@lo + lo@hi) in fp32, from the parts of two operands
  (ah, al), (bh, bl) = a, b
  out, lo = (ah.matmul(bh, "fp32", 4),
             [p.matmul(q, "fp32", 4) for p, q in ((ah, bl), (al, bh)) if p is not None and q is not None])
  return out + sum(lo[1:], lo[0]) if lo else out

def _inside(o, s, lo, hi, n):  # the e in [0, n) with lo <= o + s e < hi, as (first, end)
  if s == 0: return (0, n) if lo <= o < hi else (0, 0)
  if s < 0: o, s, lo, hi = -o, -s, 1 - hi, 1 - lo
  first, end = max(-((o - lo) // s), 0), min(-((o - hi) // s), n)
  return (first, end) if first < end else (0, 0)

WHOLE = ("matmul", "T", "move", "cast")  # ops whose operands are realized first, not fused into the same kernel
# what a kernel for t computes with it: not the operands of a WHOLE op, unless it's a cast at the root
def _below(t, root):  # the operands a kernel for t computes. a spread's is read in place, as an input
  return t.srcs if t.op not in (*WHOLE, "spread") or root and t.op == "cast" else ()
def _lazy(t, seen):  # the unrealized ops under t, t first
  stack = [(t, True)]
  while stack:
    u, root = stack.pop()
    if u.op is None or id(u) in seen: continue
    seen.add(id(u))
    yield u
    stack.extend((s, False) for s in reversed(_below(u, root)))

def _ops(t): return sum(u.op not in ("lead", "spread") for u in _lazy(t, set()))  # the most ops a kernel for t holds

def _tree(t, inputs):
  """The expression that fused_kernel computes for t, with realized tensors as indices into inputs, and the dtype it
  computes in. The kernel sweeps along the axis of the first reduction. Reductions along other axes are realized
  first."""
  dtype = t.srcs[0].dtype if t.op in ("cast", "matmul") else t.dtype
  if dtype in BFP: raise ValueError("a block float can only be a matmul's B")
  axis = next((FOLDS[u.op] for u in _lazy(t, set()) if u.op in COMBINE), 2)
  # the sweep is as long as t along the axis. if t is a tile long there, its first reduction sets it
  n = ext(t._dims, axis)
  return _expr(t, t._dims, inputs, [axis, n if n > 1 else None, set(), dtype, t]), dtype

# A block-float tile's values as bf16, the way the unpacker converts them (ISA FormatConversion.md): normalized, with 7
# bits per value. bfp4's 3 bits are shifted up to where bfp8's sit.
def _bf16s(tile, dtype):
  out = array.array("H", bytes(2048))
  for i in range(1024):
    x, e = tile[64 + i] if dtype == "bfp8" else (tile[64 + i // 2] >> 4 * (i % 2) & 15) << 4, tile[i // 16]
    if not x & 127: out[i] = 0xFF80 if x >> 7 else 0  # a magnitude of 0 is 0, or -inf with the sign set
    else:
      m = (x & 127) << 1
      shift = 8 - m.bit_length()
      out[i] = (x >> 7) << 15 | (e - shift) % 256 << 7 | (m << shift) & 0x7E
  return out.tobytes()

def _name(e):  # a kernel's name in profiles: its ops, operands first, each once
  ops = []
  def walk(e):
    if not isinstance(e, tuple): return
    for c in e[1:]: walk(c)
    if e[0] not in ops: ops.append(e[0])
  walk(e)
  return " ".join(ops)

def _expr(t, shape, inputs, kernel):
  """Build the unrealized expression under t for a kernel of t's shape. kernel holds [axis, sweep length in
  tiles, reductions (at most two), dtype, root]. Realize dtype changes, casts, fp32 matmuls and values that
  are constant along a kept axis separately. A reduction also runs separately if its axis or length differs,
  or two reductions are already included."""
  if t.op is None:
    assert t.dtype == kernel[3], f"a {t.dtype} input to a {kernel[3]} kernel"
    if t.runs: raise ValueError("cast a tensor by columns to bf16 for operations other than matmul's B or readback")
    return _input(t, inputs)
  if t.op == "lead": return _expr(t.srcs[0], shape, inputs, kernel)  # the same matrices
  if t is kernel[4] and t.op == "cast": return ("cast", _expr(t.srcs[0], shape, inputs, kernel))
  if t is not kernel[4] and (t.dtype != kernel[3] or t.op == "cast" or (t.op == "matmul" and kernel[3] != "bf16")):
    return _expr(t.realize(), shape, inputs, kernel)
  if t.op == "spread": return _input(_spread_view(t), inputs)  # read in place, like an input
  axis = kernel[0]
  if any(t._dims[ax] != shape[ax] for ax in range(3) if ax != axis): return _expr(t.realize(), shape, inputs, kernel)
  if t.op in COMBINE:
    if FOLDS[t.op] != axis: return _expr(t.realize(), shape, inputs, kernel)
    n = ext(t.srcs[0]._dims, axis)
    if n > 1 and kernel[1] is None: kernel[1] = n
    if (n > 1 and n != kernel[1]) or (id(t) not in kernel[2] and len(kernel[2]) == len(ACCS)):
      return _expr(t.realize(), shape, inputs, kernel)
    kernel[2].add(id(t))
  # kernels of their own, but a matmul can read B transposed (below)
  if t.op in ("T", "move"): return _expr(t.realize(), shape, inputs, kernel)
  if t.op == "matmul":  # a matmul reads its operands as matrices, not tile by tile, so realize them first
    a, b, *picked = map(_bare, t.srcs)  # picked: the indices that pick B's matrices, if any
    if b.op == "T":  # read transposed. a block float, like a bfp8 key cache, is read as is and transposed on the way in
      src = b.srcs[0].realize()
      if src.runs: raise ValueError("a tensor by columns can only be a matmul's B as it is, not transposed")
      b = ("T" + src.dtype, _input(src, inputs)) if src.dtype in BFP else ("T", _expr(src, shape, inputs, kernel))
    elif b.runs: b = ("C" + (b.dtype if b.dtype in BFP else ""), _input(b, inputs))  # laid out by columns()
    else: b = (b.dtype, _input(_operand(b), inputs)) if b.dtype in BFP else _expr(_operand(b), shape, inputs, kernel)
    if picked:
      b = (("I" + b[0], b[1], _input(picked[0].realize(), inputs)) if isinstance(b, tuple) else
           ("I", b, _input(picked[0].realize(), inputs)))
    return ("matmul", _expr(_operand(a), shape, inputs, kernel), b) + ((t.fidelity,) if t.fidelity != 4 else ())
  return (t.op, *(_expr(s, shape, inputs, kernel) for s in t.srcs))

def _bare(t):  # t without relabels, which matmul reads through, since its matrices are the same
  while t.op == "lead": t = t.srcs[0]
  return t
def _operand(t): return _spread_view(t) if t.op == "spread" else t.realize()  # a matmul's operand, read in place
def _spread_view(t):  # a spread as a kernel reads it: its source's matrices, with the result's batch
  if t.view is None:  # made once, so a kernel that reads it twice reads one input
    x = _plain(t.srcs[0].realize())
    t.view = x._view(t.rows, t.cols, t.batch, 0, x.bstride)
    t.view.rep, t.view.cyc = t.spread
  return t.view

def _input(t, inputs):  # the index of a realized tensor among a kernel's inputs, added if new
  for i, u in enumerate(inputs):
    if u is t: return i
  inputs.append(t)
  return len(inputs) - 1

class Recording:
  """Record launches and keep their DRAM allocations for replay. Rebuild commands when a kernel has been
  evicted from L1. If the kernels don't fit together, replay reloads them in batches. A replay can give each Bound
  argument another value, so the same launches read and write other tensors."""
  def __init__(self, rt):
    self.rt, self.launches, self.block, self.pins, self.made, self.active = rt, None, None, set(), set(), False
  def close(self):
    """Release this recording's allocations after queued replays finish. Live tensors keep their storage."""
    if self.active: raise RuntimeError("cannot close an active recording")
    self.rt.unpin(self.pins)
    self.launches, self.block, self.pins, self.made = None, None, set(), set()
  def __del__(self):
    if not self.active: self.close()
  def __enter__(self):
    if self.rt.log is not None: raise RuntimeError("cannot nest recordings on the same runtime")
    self.close()
    self.rt.log, self.rt.recorded, self.rt.recorded_bases = [], self.pins, sorted(self.rt.used)
    self.made = self.rt.recorded_made = set()  # what the recording allocates: its results and intermediates
    self.active = True
    return self
  def __exit__(self, kind, *_):
    self.launches, self.rt.log = self.rt.log, None
    self.rt.recorded, self.rt.recorded_bases, self.rt.recorded_made, self.active = None, [], None, False
    if kind is not None: self.close()
  def send(self, binding=None):
    """Queue the recorded launches. binding maps a Bound argument's key to its value for this replay; an argument
    whose key is missing keeps its recorded value."""
    if self.active or self.launches is None: raise RuntimeError("recording is not ready for replay")
    if self.rt.log is not None: raise RuntimeError("cannot replay while recording on the same runtime")
    binding = binding or {}
    if self.block is None or self.block[2] != self.rt.evictions:
      evictions, take = self.rt.evictions, lambda v: Bound(binding.get(v.key, v), v.key) if isinstance(v, Bound) else v
      cmds, sent = self.rt.block([(kernel, tuple(map(take, args)), units, name)
                                  for kernel, args, units, name in self.launches])
      # If block had to evict, it already sent part of the commands, and the new eviction count makes the next replay
      # rebuild
      self.block = (cmds, sent, evictions,
                    self.rt.bound(cmds, self.launches) if len(sent) == len(self.launches) else None)
      return self.rt.send(cmds, sent)
    cmds, sent, _, bound = self.block
    if bound:  # the block holds the values it was built with
      cmds = bytearray(cmds)
      for at, v in bound: struct.pack_into("<I", cmds, at, binding.get(v.key, v))
    self.rt.send(cmds, sent)

def _writes(t):  # while jit records, note that t's allocation is written in place
  if t.rt.written is not None: t.rt.written.add(t.addr)

def _bindable(t):  # whether a replay can read t where it is, laid out as a new tensor of its shape or by columns
  tr, tc = -(-t.rows // 32), -(-t.cols // 32)
  return t.runs is not None or (t.offset, t.rep, t.stride,
                                t.bstride if t.batch > 1 else 0) == (0, 1, tc, tr * tc if t.batch > 1 else 0)

# t, or if it's a view of a recorded input, the same view of this call's input, with a plain address
def _on_input(t, inputs):
  if not isinstance(t.addr, Bound): return t
  v = inputs[t.addr.key]._view(t.rows, t.cols, t.batch, t.offset, t.bstride, t.runs)
  v.addr, v.rep, v.lead = int(v.addr), t.rep, t.lead
  return v

def jit(f):
  """Run f normally on the first call, record the second call's launches, then replay them.
  Later calls send the same launches with the new inputs' addresses and skip f's Python code.

  Keep input shapes and dtypes fixed, and a tensor by columns in the same layout. Don't replace tensors captured by
  f or read tensors back to the host inside f. Recorded assign operations can update captured tensors and inputs in
  place. Keep written inputs apart from each other and from captured tensors that f only reads. An input laid out
  otherwise than a new tensor, such as a slice or a repeat, is copied first, and copied back if f writes it. Results
  and retained tensors, including .grad, reuse their buffers; the next call overwrites their contents. So an input's
  .grad holds the latest call's gradient instead of adding up. Pass an input that needs a gradient once: a repeated
  one gets the gradient through its last position only."""
  signature, runs, wrote, taken, out, rec, grads = None, (), set(), set(), None, None, ()
  @functools.wraps(f)
  def run(*xs):
    nonlocal signature, runs, wrote, taken, out, rec, grads
    if not xs or not all(isinstance(x, Tensor) for x in xs):
      raise ValueError("jit inputs must be tensors, with at least one input")
    rt = xs[0].rt
    if any(x.rt is not rt for x in xs): raise ValueError("jit inputs must use the same runtime")
    if signature is None:
      result = f(*xs).realize()
      signature = [(x.rt, x.shape, x.dtype) for x in xs]
      return result
    if len(xs) != len(signature): raise ValueError(f"jit expected {len(signature)} inputs, got {len(xs)}")
    for x, (runtime, shape, dtype) in zip(xs, signature):
      if x.rt is not runtime: raise ValueError("jit inputs must use the recorded runtime")
      if (x.shape, x.dtype) != (shape, dtype):
        raise ValueError(f"a jit input is {x.shape} {x.dtype}, where the first call had {shape} {dtype}")
    xs = [x.realize() if x.op is not None else x for x in xs]
    if rec is None:
      # Record on each input where it is, under an address bound to its position. Copy those a replay couldn't read in
      # place.
      ins = [x._view(x.rows, x.cols, x.batch, 0, x.bstride, x.runs) if _bindable(x) else _copy(x) for x in xs]
      for i, (x, t) in enumerate(zip(xs, ins)):
        t.addr, t.requires_grad, t.lead = Bound(t.addr, i), x.requires_grad, x.lead
      rt.written = set()
      try:
        with Recording(rt) as recording: result = f(*ins).realize()
      finally: written, rt.written = rt.written, None
      runs, rec, out, wrote = ([x.runs for x in xs], recording, result,
                               {i for i, t in enumerate(ins) if t.addr in written})
      # storage the recording makes, and captured storage f writes
      taken = rec.made | {a for a in written if not isinstance(a, Bound)}
      for i in wrote:
        if ins[i].owner is None: xs[i].assign(ins[i])  # a copy that f wrote
      grads = [t.grad for t in ins]  # backward gave the gradients to the stand-ins
      for x, g in zip(xs, grads):
        if g is not None: x.grad = _on_input(g, ins)
      return _on_input(out, ins)
    if any(x.runs != r for x, r in zip(xs, runs)):
      raise ValueError("a jit input by columns must keep the layout it was recorded with")
    # Copy an input that the recording's own results overwrite, one that f only reads where this replay writes, and one
    # a replay can't read in place
    writes = taken | {xs[i].addr for i in wrote}
    bound = [x if _bindable(x) and x.addr not in rec.made and (i in wrote or x.addr not in writes) else _copy(x)
             for i, x in enumerate(xs)]
    rec.send({i: t.addr for i, t in enumerate(bound)})
    for i in wrote:
      if bound[i] is not xs[i]: xs[i].assign(bound[i])
    for x, g in zip(xs, grads):
      if g is not None: x.grad = _on_input(g, bound)
    return _on_input(out, bound)
  return run

# Compiled kernels by key. Fused entries hold (image, units, name); other entries hold an image.
kernels = {}
def _cached(key, make):
  if key not in kernels: kernels[key] = make()
  return kernels[key]
