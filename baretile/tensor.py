# Build tensor expressions and compile them into kernels. Fuse operations where possible; compute inputs
# separately when fusion would exceed a kernel's limits or require a different layout or dtype.
import array, collections, functools, math, operator
from baretile import bh
from baretile.tensix import ACCS, BFP, COMBINE, FOLDS, INT32_OPS, ITEMSIZE, MAX_OPS, ext, round_bf16, tile_bytes
from baretile.kernels import attention_parts, attention_store, link_flags
from baretile.kernels import (attention_kernel, bfp4_kernel, columns_kernel, copy_kernel, fused_kernel, gather_kernel, link_kernel, move_kernel,
                             put_kernel, readback_kernel, topk_kernel, transpose_kernel)

FIDELITY = 4  # phases a bf16 matmul takes by default. 4 gives exact products, and tensix.matmul says what fewer drop

class Tensor:
  """Represent a batch of matrices with shape (batch, rows, cols). Supported storage formats are bf16,
  fp32, int32, bfp8 and bfp4. An unrealized tensor holds an operation and its operands.

  Default DRAM storage uses 32 x 32 tiles in row order, with matrices stored consecutively. Tile i is
  in bank i % 8. Partial tiles are padded with zeros. A dimension of size 1 repeats across its tile
  for broadcasting. Kernels preserve this layout, including the padding needed by matmul.

  Batch indexing, tile-aligned slices, split and repeat_interleave can share an owner's storage.
  Tile (b, i, j) of a view is at b // rep * bstride + i * stride + j + offset in that storage.
  Movement that can't use a view is copied by move_kernel."""
  def __init__(self, rt, rows, cols, op=None, srcs=(), batch=1, dtype="bf16"):
    self.addr = None  # construction can fail before there is an allocation to release
    batch, rows, cols = _shape((batch, rows, cols))
    if dtype not in (*ITEMSIZE, *BFP): raise ValueError(f"unknown tensor dtype: {dtype}")
    if any(s.rt is not rt for s in srcs): raise ValueError("tensor operands must use the same runtime")
    self.rt, self.rows, self.cols, self.batch, self.op, self.srcs, self.dtype = rt, rows, cols, batch, op, srcs, dtype
    tr, tc = -(-rows // 32), -(-cols // 32)  # tile rows and columns
    self.tiles, self.bstride, self.stride, self.offset, self.owner, self.rep = batch * tr * tc, tr * tc, tc, 0, None, 1
    self.runs = None  # by columns: (runs per matrix in each bank, tiles per run, K tiles per matrix if split cut K)
    self.addr = None if op else self._alloc()
    self.requires_grad, self.grad, self._ctx = False, None, None
    if op not in (None, "move", "view"): self._track(op, srcs)  # moves and views are tracked by their callers
  def _track(self, op, srcs, *args):  # remember how this was made, if a source needs a gradient
    if any(s.requires_grad for s in srcs): self.requires_grad, self._ctx = True, (op, srcs, args)
    return self
  def requires_grad_(self, on=True):
    self.requires_grad = on
    return self
  def _where(self, runs=False):  # the address and tile strides a kernel reads it by, or for a tensor by columns, its runs
    assert runs or not self.runs, "a tensor by columns can only be a matmul's B, or read back"
    return (self.addr, *self.runs) if self.runs else (self.addr, self.bstride if self.batch > 1 else 0, self.stride, self.offset)
  def _alloc(self): return self.rt.dram_alloc(-(-self.tiles // len(bh.DRAM)) * tile_bytes(self.dtype))
  def _same_runtime(self, *xs):
    if any(x.rt is not self.rt for x in xs): raise ValueError("tensor operands must use the same runtime")

  def __del__(self):  # free the DRAM. the runtime holds it until the launches queued so far are done
    if self.addr is not None and self.owner is None: self.rt.dram_free(self.addr)

  @classmethod
  def full(cls, rt, rows, cols, value, dtype="bf16"): return cls.of(rt, [[value] * cols for _ in range(rows)], dtype)

  @classmethod
  def of(cls, rt, data, dtype="bf16"):
    """Make a tensor from rows of numbers, or a list of those for a batch. bf16 rounds to nearest even."""
    if not len(data) or not len(data[0]): raise ValueError("tensor data must be nonempty")
    matrices = data if isinstance(data[0][0], (list, tuple)) else [data]
    rows, cols = len(matrices[0]), len(matrices[0][0])
    if not cols or any(len(m) != rows or any(len(row) != cols for row in m) for m in matrices):
      raise ValueError("tensor data must be nonempty rectangular matrices of the same shape")
    values = array.array("i" if dtype == "int32" else "f", [v for m in matrices for row in m for v in row])
    if dtype == "bf16": values = array.array("H", map(round_bf16, array.array("I", values.tobytes())))
    return cls.frombytes(rt, rows, cols, values, len(matrices), dtype)

  @classmethod
  def frombytes(cls, rt, rows, cols, data, batch=1, dtype="bf16"):
    """Upload little-endian values in row-major order, with matrices stored consecutively. Convert to tile
    layout in 8-byte chunks and write each bank once. Model weights use this path."""
    assert dtype not in BFP, "upload as bf16, then cast on the chip to a block-float format"
    if dtype not in ITEMSIZE: raise ValueError(f"unknown tensor dtype: {dtype}")
    batch, rows, cols = _shape((batch, rows, cols))
    n, tr, tc, data = ITEMSIZE[dtype], -(-rows // 32), -(-cols // 32), memoryview(data).cast("B")
    if len(data) != n * batch * rows * cols: raise ValueError(f"{len(data)} bytes for {batch} x {rows} x {cols} of {dtype}")
    t = cls(rt, rows, cols, batch=batch, dtype=dtype)
    tb, run = tile_bytes(dtype), 2 * n  # bytes per tile, and 8-byte words per run of 16 values (a row of a face)
    tiles = memoryview(bytearray(t.tiles * tb)).cast("Q")  # the tiles in order, zero where they're padding
    for b, r in ((b, r) for b in range(batch) for r in range(rows if rows > 1 else 32)):  # a single row fills its tile
      at = n * (b * rows + (r if rows > 1 else 0)) * cols
      row = bytes(data[at:at + n]) * 32 if cols == 1 else bytes(data[at:at + n * cols]) + bytes(n * (32 * tc - cols))
      src, first = memoryview(row).cast("Q"), (b * tr + r // 32) * tc * tb // 8 + r % 16 * run
      for half, q in ((half, q) for half in (0, 1) for q in range(run)):  # word q of the runs in the left or right faces
        start = first + (r % 32 // 16 * 2 + half) * tb // 32 + q  # faces as in the tile layout at the top of tensix.py
        tiles[start:start + tc * tb // 8:tb // 8] = src[half * run + q::2 * run]
    tiles = tiles.cast("B")
    for bank in range(len(bh.DRAM)):  # tile i is in bank i % 8 at addr + i // 8 tiles, so each bank's tiles are one write
      t.rt.chip.write(*bh.DRAM[bank], t.addr, b"".join(tiles[i * tb:(i + 1) * tb] for i in range(bank, t.tiles, len(bh.DRAM))))
    return t

  def tobytes(self):
    """Return row-major bytes in the format accepted by frombytes. Convert block floats to bf16.
    Queue a kernel to write tiles into host memory and wait for completion. This avoids a PCIe
    round trip for each host load through the BARs."""
    self.realize()
    n, rows, cols, tc, tb = 2 if self.dtype in BFP else ITEMSIZE[self.dtype], self.rows, self.cols, -(-self.cols // 32), tile_bytes(self.dtype)
    # readback_kernel sends sent bytes of each tile: v rows, with face f starting at faces[f]
    v = 32 if self.dtype in BFP else min(rows, 32)
    sent, faces = tb if v == 32 else 32 * v * n, [16 * n * f for f in (0, min(v, 16), 2 * min(v, 16), 2 * min(v, 16) + max(v - 16, 0))]
    by = bool(self.runs)
    per, tiles, kernel = len(self.rt.back) // sent, bytearray(), _cached(("readback", self.dtype, by), lambda: readback_kernel(self.dtype, by))
    for start in range(0, self.tiles, per):
      k = min(per, self.tiles - start)
      self.rt.launch(kernel, (start, *self._where(True), tc, self.tiles // self.batch, self.rep,
                              self.rt.back_bus & 0xFFFFFFFF, self.rt.back_bus >> 32, v), units=k, name="readback")
      self.rt.wait()
      tiles += self.rt.back[:k * sent]
    if self.dtype in BFP: tiles, sent = b"".join(_bf16s(tiles[i * sent:(i + 1) * sent], self.dtype) for i in range(self.tiles)), 2048
    words, sw, run, matrix = memoryview(tiles).cast("Q"), sent // 8, 2 * n, self.tiles // self.batch  # 8-byte words per tile sent, per run
    out, row = bytearray(n * self.batch * rows * cols), memoryview(bytearray(32 * n * tc)).cast("Q")  # a row across the tile columns
    # frombytes in reverse: row by row, word q of its runs from the left or right faces of the tiles along it
    for b, r in ((b, r) for b in range(self.batch) for r in range(rows)):
      first = (b * matrix + r // 32 * tc) * sw + (r % 16 * 16 * n) // 8
      for half, q in ((half, q) for half in (0, 1) for q in range(run)):
        start = first + faces[r % 32 // 16 * 2 + half] // 8 + q
        row[half * run + q::2 * run] = words[start:start + tc * sw:sw]
      at = n * (b * rows + r) * cols
      out[at:at + n * cols] = row.cast("B")[:n * cols]
    return bytes(out)

  def tolist(self):  # rows of numbers, or a list of them for a batch
    raw = self.tobytes()
    if self.dtype == "bf16" or self.dtype in BFP: raw = array.array("I", [b << 16 for b in array.array("H", raw)]).tobytes()
    values = array.array("i" if self.dtype == "int32" else "f", raw).tolist()
    m = [[values[(b * self.rows + r) * self.cols:(b * self.rows + r + 1) * self.cols] for r in range(self.rows)] for b in range(self.batch)]
    return m if self.batch > 1 else m[0]

  def gather(self, ids):
    """Read rows selected by ids. Accept a Python list or an int32 tensor row, so indices can be computed on
    the chip. Python indices are bounds-checked; tensor indices aren't."""
    if not isinstance(ids, Tensor):
      assert all(0 <= i < self.rows for i in ids)
      ids = Tensor.of(self.rt, [list(ids)], "int32")
    self._same_runtime(ids)
    assert self.batch == 1 and ids.dtype == "int32" and ids.shape[:2] == (1, 1), "gather needs batch 1, and ids as an int32 row"
    out, d, ids = Tensor(self.rt, ids.cols, self.cols, dtype=self.dtype), self.dtype, ids.realize()
    addr, _, stride, offset = _plain(self.realize())._where()
    return out.run(("gather", d), lambda: gather_kernel(d), out.addr, addr, stride, offset,
                   -(-self.cols // 32), ids.cols, ids.addr, ids.offset, name="gather")._track("gather", (self,), ids)

  def put(self, ids, x):
    """Write x into rows selected by ids, in place. Models use this to update KV caches without changing the
    shape of a decode step. For a block-float destination, x is bf16 and is rounded as in cast."""
    self._same_runtime(ids, x)
    assert ids.dtype == "int32" and ids.shape[:2] == (1, 1) and ids.cols == x.rows, "ids must be an int32 row with one position per row of x"
    assert (x.batch, x.cols) == (self.batch, self.cols) and x.dtype == ("bf16" if self.dtype in BFP else self.dtype), \
      f"{x.shape} {x.dtype} into {self.shape} {self.dtype}"
    assert self.rep == 1, "put requires a tensor with rep=1"
    x, ids, tc, tr = _plain(x.realize()), ids.realize(), -(-self.cols // 32), -(-self.rows // 32)
    assert 2 * x.rows + tr <= 2**17, "too many rows to track in a worker's L1"
    return self.realize().run(("put", self.dtype), lambda: put_kernel(self.dtype), *self._where(), *x._where(), ids.addr,
                              ids.offset, x.rows, tc, tr, units=self.batch * x.rows * tc, name="put")

  def _view(self, rows, cols, batch, offset, bstride=0, runs=None):
    assert self.rep == 1, "can't view a repeat: a view addresses stored matrices, and a repeat's aren't all stored"
    assert not self.runs or runs, "a tensor by columns can only be viewed by split, cutting K into a batch"
    v = Tensor(self.rt, rows, cols, "view", batch=batch, dtype=self.dtype)
    v.op, v.addr, v.owner = None, self.realize().addr, self.owner or self
    v.offset, v.bstride, v.stride, v.runs = self.offset + offset, bstride, self.stride, runs
    return v

  def __getitem__(self, key):
    """Select a batch entry with x[b], or slice with x[rows, cols] or x[batch, rows, cols].
    A slice shares storage when rows and columns start on tile boundaries, end on a boundary or at the
    matrix edge, and step by 1. The batch may use any positive step. Slicing rows or columns down to
    size 1 needs copies across the tile. Other slices also copy. An unrealized sum of realized
    tensors, such as split-K results, is sliced term by term."""
    if self.rep > 1: return (self + 0)[key]  # slice a copy of a repeat: the copy stores its matrices one after another
    if isinstance(key, int):
      if not -self.batch <= key < self.batch: raise IndexError(f"batch index {key} outside a batch of {self.batch}")
      key %= self.batch
      return self._view(self.rows, self.cols, 1, key * self.bstride)._track("index", (self,), key)
    assert isinstance(key, tuple) and len(key) in (2, 3) and all(isinstance(k, slice) for k in key), "x[b], x[rows, cols] or x[batch, rows, cols]"
    ranges = [range(*k.indices(n)) for k, n in zip((slice(None),) * (3 - len(key)) + tuple(key), self.shape)]
    shape = tuple(map(len, ranges))
    assert all(shape), f"an empty slice of {self.shape}"
    if all(g.step > 0 if ax == 0 else g.step == 1 and g.start % 32 == 0 and (g.stop % 32 == 0 or g.stop == n) and (len(g) > 1 or n == 1)
           for ax, (g, n) in enumerate(zip(ranges, self.shape))):  # tile-aligned. a slice of length 1 must be a copy, filling its tile
      if terms := _terms(self): return sum((t[key] for t in terms[1:]), terms[0][key])
      b, r, c = (g.start for g in ranges)
      t = self._view(shape[1], shape[2], shape[0], b * self.bstride + r // 32 * self.stride + c // 32, self.bstride * ranges[0].step)
    else:
      t = self._map(shape, (0, 1, 2), [g.start for g in ranges], [g.step for g in ranges])
    return t._track("slice", (self,), ranges)

  # Movement. Each op maps the result's elements to this tensor's, and one move_kernel launch copies them.
  def move(self, shape, offset, strides, lo=(0, 0, 0), hi=None):
    """Map elements into a new tensor of shape (batch, rows, cols). Output element (b, r, c) reads
    input element offset + b * strides[0] + r * strides[1] + c * strides[2], in row-major order.
    Output positions outside [lo, hi) on any axis are zero."""
    t = Tensor(self.rt, shape[1], shape[2], "move", (self,), shape[0], self.dtype)
    t.map = (offset, *strides, *lo, *(hi or shape))
    return t
  def _map(self, shape, axes, starts, steps):
    """Map output index e on axis a to starts[a] + steps[a] * e on source axis axes[a], using zero outside the
    source. Return shape (batch, rows, cols). Compose with an unrealized move so a chain of maps can run in
    one launch."""
    if self.op == "move": x, offset, strides, lo, hi = self.srcs[0], self.map[0], self.map[1:4], self.map[4:7], self.map[7:]
    else: x, offset, strides, lo, hi = self, 0, (self.rows * self.cols, self.cols, 1), (0, 0, 0), self.shape
    inside = [_inside(o, s, lo[k], hi[k], n) for k, o, s, n in zip(axes, starts, steps, shape)]
    return x.move(shape, offset + sum(strides[k] * o for k, o in zip(axes, starts)), [strides[k] * s for k, s in zip(axes, steps)], *zip(*inside))
  def reshape(self, *shape):  # up to three sizes. missing leading sizes are 1
    shape = _shape(shape)
    assert math.prod(shape) == math.prod(self.shape), f"cannot reshape {self.shape} to {shape}"
    return self if shape == self.shape else self.move(shape, 0, (shape[1] * shape[2], shape[2], 1))._track("reshape", (self,))
  def permute(self, *order):
    assert sorted(order) == [0, 1, 2]
    if order == (0, 2, 1): return self.T
    if order == (0, 1, 2): return self
    return self._map(tuple(self.shape[k] for k in order), order, (0, 0, 0), (1, 1, 1))._track("permute", (self,), order)
  def expand(self, *shape):  # repeats dimensions of 1. missing leading sizes are 1
    shape = _shape(shape)
    assert all(n in (m, 1) for n, m in zip(self.shape, shape)), f"cannot expand {self.shape} to {shape}"
    if shape == self.shape: return self
    return self._map(shape, (0, 1, 2), (0, 0, 0), [int(n == m) for n, m in zip(self.shape, shape)])._track("expand", (self,))
  def pad(self, widths):  # zeros, with (before, after) for each of the last len(widths) axes, like numpy
    widths = ((0, 0),) * (3 - len(widths)) + tuple(widths)
    shape = tuple(n + w[0] + w[1] for n, w in zip(self.shape, widths))
    return self._map(shape, (0, 1, 2), [-w[0] for w in widths], (1, 1, 1))._track("pad", (self,), widths)

  def cat(self, *xs, axis=0):  # like tinygrad: pad each with zeros to its position along axis, then sum
    self._same_runtime(*xs)
    axis = operator.index(axis)
    if not -3 <= axis < 3: raise ValueError(f"axis {axis} outside a tensor's three dimensions")
    ts, axis = (self, *xs), axis % 3
    assert all(t.shape[a] == self.shape[a] for t in xs for a in range(3) if a != axis), "cat requires matching dimensions outside its axis"
    n, out, at = sum(t.shape[axis] for t in ts), None, 0
    for t in ts:
      widths = [(0, 0)] * 3
      widths[axis] = (at, n - at - t.shape[axis])
      out, at = t.pad(widths) if out is None else out + t.pad(widths), at + t.shape[axis]
    return out

  def repeat_interleave(self, n):
    """Repeat each matrix n times along the batch without copying: kernels read matrix b from the matrix stored at
    b // n. It's how grouped-query attention shares key and value heads."""
    if self.rep > 1: return (self + 0).repeat_interleave(n)
    v = self._view(self.rows, self.cols, self.batch * n, 0, self.bstride)
    v.rep = n
    return v._track("repeat", (self,), n)

  def split(self, n, axis=2):
    """Split along axis (2 for columns, 1 for rows) into n equal tile-aligned parts and view them as a batch.
    No data is copied. Split an unrealized sum of realized tensors term by term."""
    n, axis = operator.index(n), operator.index(axis)
    if n <= 0: raise ValueError("split needs a positive number of parts")
    if terms := _terms(self): return sum((t.split(n, axis) for t in terms[1:]), terms[0].split(n, axis))
    assert axis in (1, 2)
    if self.runs:  # by columns, split cuts K: each part reads further along the same runs
      assert axis == 1 and self.batch == 1 and self.rows % (32 * n) == 0, "a tensor by columns splits K, with batch 1, into tile-aligned parts"
      return self._view(self.rows // n, self.cols, n, 0, runs=(0, self.runs[1], self.rows // n // 32))._track("split", (self,), axis)
    if axis == 2:
      assert self.batch == 1 and self.cols % (32 * n) == 0, "split needs batch 1 and tile-aligned parts"
      return self._view(self.rows, self.cols // n, n, 0, self.cols // n // 32)._track("split", (self,), axis)
    assert self.rows % (32 * n) == 0 and (self.batch == 1 or self.bstride == self.rows // 32 * self.stride), "tile-aligned parts, contiguous matrices"
    return self._view(self.rows // n, self.cols, self.batch * n, 0, self.rows // n // 32 * self.stride)._track("split", (self,), axis)

  def merge(self):
    """Place batch matrices side by side, reversing split: x.split(n).merge() is x. An unrealized expression
    writes directly into the result columns. Otherwise, copy the matrices there."""
    out = Tensor(self.rt, self.rows, self.batch * self.cols, dtype=self.dtype)
    (self if self.op not in (None, "T", "move") else self + 0).realize(out.split(self.batch))  # + 0 copies a realized tensor or a move in
    return out._track("merge", (self,))

  @property
  def T(self):
    """Transpose each matrix. A matmul can read this as B without a copy, transposing tiles during unpack.
    Other uses run transpose_kernel, or move_kernel for fp32 and int32 because the unpacker can only
    transpose into SrcA."""
    if self.op == "T": return self.srcs[0]
    return Tensor(self.rt, self.cols, self.rows, "T", (self,), self.batch, self.dtype)

  def realize(self, into=None):
    """Queue the computation and return this tensor. Allocate DRAM for the result, or write into the tensor or
    view passed as into."""
    if into is not None:
      self._same_runtime(into)
      assert into.op is None and (into.shape, into.dtype) == (self.shape, self.dtype), "realize needs a destination of the same shape and dtype"
    if self.op is None: return self
    if self.op in ("T", "move"):  # data movement, into a tensor of its own
      assert into is None, "a transpose or a move writes a tensor of its own"
      x, op, self.op, self.srcs, d = _plain(self.srcs[0].realize()), self.op, None, (), self.dtype
      self.addr = self._alloc()
      where = (self.addr, *x._where())
      if op == "T" and d == "bf16":
        return self.run(("transpose", x.rows, x.cols), lambda: transpose_kernel(x.rows, x.cols), *where, name="transpose")
      if op == "T": self.map = (0, x.rows * x.cols, 1, x.cols, 0, 0, 0, *self.shape)
      return self.run(("move", d), lambda: move_kernel(d), *where, x.rows, x.cols, self.rows, self.cols,
                      *(v & 0xFFFFFFFF for v in self.map), name="move")
    _fit(self)
    inputs = []
    e, dtype = _tree(self, inputs)
    shapes = tuple(u.shape for u in inputs)
    # when only the result's rows matter, round the inputs' rows up to tiles, so the same kernel serves any row count
    if _rows_free(e): shapes = tuple((b, r if r == 1 else -(-r // 32) * 32, c) for b, r, c in shapes)
    key = (e, shapes, dtype, self.dtype, self.rt.grid, tuple(u.rep for u in inputs), self.rt.readers)
    kernel, units, name = _cached(key, lambda: (*fused_kernel(*key), _name(e)))
    assert not (into and into.runs), "can't realize into a tensor by columns"
    if into: self.addr, self.owner, self.bstride, self.stride, self.offset = into.addr, into.owner or into, into.bstride, into.stride, into.offset
    else: self.addr = self._alloc()
    self.op, self.srcs = None, ()
    # _where(True) is safe: _expr only lets a tensor by columns in as a matmul's B
    args = [v for t in [self] + inputs for v in t._where(True)] + [self.rows % 32 if self.rows > 1 else 0]
    self.rt.launch(kernel, tuple(args), units or self.tiles, name)
    return self
  def assign(self, x):
    """Write x into this tensor in place, as an optimizer updates a weight. An unrealized x is computed into this
    tensor's DRAM, which is safe because elementwise kernels read each tile before writing it. Matmuls and moves read
    other tiles too, which may already be overwritten, so they're realized first and copied in. Overlapping
    views with different layouts also copy their source before writing."""
    self._same_runtime(x)
    assert self.op is None and self.rep == 1 and (x.shape, x.dtype) == (self.shape, self.dtype), \
      "assign requires a realized tensor with rep=1 and an x of the same shape and dtype"
    if x.op in ("T", "move") or any(t.op == "matmul" for t in _lazy(x, set())): x = x.realize()
    reads = [x] if x.op is None else [s for t in _lazy(x, set()) for s in t.srcs]
    layout = lambda t: (t.shape, t.bstride, t.stride, t.offset, t.rep)
    if any(t.addr == self.addr and layout(t) != layout(self) for t in reads):
      x = x.realize() if x.op is not None else Tensor(self.rt, x.rows, x.cols, batch=x.batch, dtype=x.dtype).assign(x)
    if x.op is not None: x.realize(self); return self
    x = _plain(x)
    return self.run(("copy", self.dtype), lambda: copy_kernel(self.dtype), *self._where(), *x._where(),
                    -(-self.rows // 32), -(-self.cols // 32), name="copy")

  def run(self, key, make, *args, units=None, name=""):
    """Build a kernel with make(), cache it by key, and queue it over units (tiles by default). Realize tensors
    in args and pass their address, batch stride, row stride and offset. Launches execute in order; this
    call doesn't wait for completion."""
    self._same_runtime(*(a for a in args if isinstance(a, Tensor)))
    args = [_plain(a.realize()) if isinstance(a, Tensor) else a for a in args]  # the list keeps each copy alive until it's queued
    units = self.tiles if units is None else units
    self.rt.launch(_cached(key, make), tuple(v for a in args for v in (a._where() if isinstance(a, Tensor) else (a,))), units, name)
    return self

  @property
  def shape(self): return (self.batch, self.rows, self.cols)

  # Elementwise ops: tinygrad's primitives, and the rest built from them. exp, log and cos are SFPU programs too, so
  # their constants stay fp32. As numbers they'd be bf16 tensors, and exp(10) as exp2(10 * log2(e)) would be 2% off.
  def _unary(self, op):  # int32 goes through fp32, except relu, which is a max with 0
    if self.dtype == "int32" and op != "relu": return self.float()._unary(op)
    return Tensor(self.rt, self.rows, self.cols, op, (self,), self.batch, self.dtype)
  def number(self, v, dtype):  # a 1 x 1 tensor of v, uploaded once per runtime, so a number costs no PCIe write
    key = (repr(v), dtype)  # repr tells -0.0 from 0.0, and a NaN matches itself
    if key not in self.rt.numbers: self.rt.numbers[key] = Tensor.full(self.rt, 1, 1, v, dtype)
    return self.rt.numbers[key]
  def _binop(self, op, x, reverse=False):
    """numpy broadcasting. A number becomes a 1 x 1 tensor of this dtype (fp32 for a float with int32). Mixed dtypes
    meet in fp32, and so does int32 for ops that have no int32 form, like mul, which is then exact below 2^24."""
    if not isinstance(x, Tensor): x = self.number(x, "fp32" if self.dtype == "int32" and isinstance(x, float) else self.dtype)
    a, b = (x, self) if reverse else (self, x)
    if a.dtype != b.dtype or (a.dtype == "int32" and op not in INT32_OPS): a, b = a.float(), b.float()
    assert all(n == m or 1 in (n, m) for n, m in zip(a.shape, b.shape)), f"cannot broadcast {a.shape} with {b.shape}"
    batch, rows, cols = map(max, a.shape, b.shape)
    return Tensor(self.rt, rows, cols, op, (a, b), batch, a.dtype)
  def cast(self, dtype):  # bfp4 isn't lazy: bfp4_kernel makes it now from the bf16 value, and says why
    if dtype == self.dtype: return self
    if dtype != "bfp4": return Tensor(self.rt, self.rows, self.cols, "cast", (self,), self.batch, dtype)
    x, out, tc = _plain(self.cast("bf16").realize()), Tensor(self.rt, self.rows, self.cols, batch=self.batch, dtype="bfp4"), -(-self.cols // 32)
    return out.run(("bfp4",), bfp4_kernel, out.addr, *x._where(), tc,
                   -(-self.rows // 32) * tc, name="bfp4")
  def columns(self):
    """Store each tile column of B together so matmul can read several tiles along K per request.
    Tile column j goes to bank j % 8, with block-float exponents before values. The result supports
    readback and use as matmul's B, including indexed selection and splits along K."""
    assert self.dtype in ("bf16", *BFP), "bf16 or a block float: a matmul's B"
    x, tr, tc = _plain(self.realize()), -(-self.rows // 32), -(-self.cols // 32)
    runs = -(-tc // len(bh.DRAM))  # runs per matrix in each bank
    out = Tensor(self.rt, self.rows, self.cols, "view", batch=self.batch, dtype=self.dtype)
    out.op, out.runs, out.addr = None, (runs, tr, 0), self.rt.dram_alloc(self.batch * runs * tr * tile_bytes(self.dtype))
    return out.run(("columns", self.dtype), lambda: columns_kernel(self.dtype), out.addr, x, tc, tr * tc, runs, tr, name="columns")
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
  def minimum(self, x): return -((-self).maximum(-x))
  def where(self, x, y):  # x where self isn't 0, else y. a condition of another dtype becomes 0 or 1 in theirs
    dtype = next((t.dtype for t in (x, y) if isinstance(t, Tensor)), self.dtype)
    c = self if self.dtype == dtype else self.ne(0).cast(dtype)
    return c._binop("keep", x) + c._binop("drop", y)
  def __add__(self, x): return self._binop("add", x)
  def __sub__(self, x): return self._binop("sub", x)
  def __mul__(self, x): return self._binop("mul", x)
  def __rsub__(self, x): return self._binop("sub", x, True)
  __radd__, __rmul__ = __add__, __mul__
  def __truediv__(self, x): return self * (1 / x if isinstance(x, (int, float)) else x.reciprocal())
  def __neg__(self): return 0 - self if self.dtype == "int32" else self * -1  # int32 subtracts exactly but multiplies through fp32
  def __lt__(self, x): return self._binop("cmplt", x)
  def __gt__(self, x): return self._binop("cmplt", x, True)
  def __ge__(self, x): return 1 - (self < x)
  def __le__(self, x): return 1 - (self > x)
  def ne(self, x): return self._binop("cmpne", x)
  def eq(self, x): return 1 - self.ne(x)
  __ne__, __eq__ = ne, eq
  def __hash__(self): return id(self)  # == is elementwise, so hash by identity, like tinygrad
  def sigmoid(self): return (1 + (-self).exp()).reciprocal()
  def silu(self): return self * self.sigmoid()
  def gelu(self):  # GPT-2's tanh form: 0.5 x (1 + tanh(z)) with z = sqrt(2/pi) (x + 0.044715 x^3), which is x / (1 + e^-2z)
    k = 2 * math.sqrt(2 / math.pi)
    return self * (1 + (self * (self * self * (-0.044715 * k) - k)).exp()).reciprocal()

  # Reductions, and ops built from them
  def sum(self, axis=None): return self._reduce("sum", axis)
  def max(self, axis=None): return self._reduce("max", axis)
  def _reduce(self, f, axis):
    """Reduce like numpy with keepdims: axis 0 over the batch, 1 down the rows, 2 along each row, or None for all
    three. Each axis is its own kernel, which sweeps lines of tiles along it."""
    if axis is not None: axis = operator.index(axis)
    if axis is not None and not -3 <= axis < 3: raise ValueError(f"axis {axis} outside a tensor's three dimensions")
    if self.dtype == "int32": return self.float()._reduce(f, axis).cast("int32")  # exact below 2^24
    if axis is None: return self._reduce(f, 0)._reduce(f, 1)._reduce(f, 2)
    shape = [1 if ax == axis % 3 else s for ax, s in enumerate(self.shape)]
    return self if shape == list(self.shape) else Tensor(self.rt, shape[1], shape[2], f"{f}{axis % 3}", (self,), shape[0], self.dtype)
  def mean(self, axis=None):
    s = self.sum(axis)
    return s * (math.prod(s.shape) / math.prod(self.shape))
  def topk(self, k, parts=1, labels=None):
    """Return (values, int32 indices) for the k largest entries in each row, descending. Ties keep column
    order. BRISC compares bf16 or fp32 values exactly, with k <= 64 and no gradient. For rows too wide for
    L1, parts returns k entries per part, labelled by original column. Run topk again with those labels to
    combine the parts."""
    if labels is not None: self._same_runtime(labels)
    k, parts = operator.index(k), operator.index(parts)
    if k <= 0 or parts <= 0: raise ValueError("topk needs positive k and parts")
    cols = self.cols // parts
    assert self.dtype in ("bf16", "fp32") and k <= min(cols, 64) and parts * cols == self.cols, "bf16 or fp32, k up to 64, and equal parts"
    assert labels is None or (labels.dtype == "int32" and labels.shape == self.shape and parts == 1), "labels need int32, x's shape and parts=1"
    x, labels = _plain(self.realize()), None if labels is None else _plain(labels.realize())
    values = Tensor(self.rt, x.rows, parts * k, batch=x.batch, dtype=x.dtype)
    ids = Tensor(self.rt, x.rows, parts * k, batch=x.batch, dtype="int32")
    key = ("topk", k, cols, x.dtype, parts, labels is not None)
    return values.run(key, lambda: topk_kernel(*key[1:]), values.addr, ids.addr, *x._where(), x.rows,
                      *(labels._where() if labels else ()), units=x.batch * parts * -(-x.rows // 32) * 32, name="topk"), ids
  def to(self, rt, ch, back):
    """Copy to rt's chip over Ethernet channel ch, with channel back at the peer (see Chip.links). Wait for
    queued launches on both chips before clearing the transfer flags. This can't be recorded because replay
    wouldn't clear those flags again."""
    assert rt is not self.rt and self.rt.log is None and rt.log is None, "to requires different runtimes with neither recording launches"
    x = self.realize()
    assert not x.runs, "a tensor by columns can only be a matmul's B, or read back"
    if x.owner is not None or x.rep > 1 or x.offset: x = (x + 0).realize()  # copy a view into a tensor of its own first
    out = Tensor(rt, x.rows, x.cols, batch=x.batch, dtype=x.dtype)
    x.rt.wait(), rt.wait()
    flags, n = link_flags(x.dtype)
    for chip, end in ((x.rt.chip, ch), (rt.chip, back)): chip.write(*bh.ETH[end], flags, bytes(n))
    out.run(("link", False, x.dtype), lambda: link_kernel(False, x.dtype), out.addr, out.tiles, *bh.ETH[back], units=1, name="link in")
    x.run(("link", True, x.dtype), lambda: link_kernel(True, x.dtype), x.addr, x.tiles, *bh.ETH[ch], units=1, name="link out")
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

    An fp32 operand is split into bf16 hi and lo. The kernel sums hi@hi, hi@lo and lo@hi, preserving about
    16 of fp32's 24 bits. The FPU aligns groups of eight products before adding, with error around 2^-11 of
    the largest terms. See docs/compiler.md, "Matmul precision".

    index is an int32 (rows, k) tensor. For batch entry b, index[b // k, b % k] selects a matrix from x.
    Indices are read on the chip so replay can select different experts. An index >= x.batch gives zero
    without reading a matrix."""
    self._same_runtime(x, *((index,) if index is not None else ()))
    fidelity = FIDELITY if fidelity is None else operator.index(fidelity)
    if not 1 <= fidelity <= 4: raise ValueError("matmul fidelity must be between one and four phases")
    if index is not None:
      assert index.dtype == "int32" and self.dtype == "bf16" and self.cols == x.rows and self.batch in (1, index.rows * index.cols), \
        f"{self.shape} @ {x.shape} picked by {index.shape} {index.dtype}"
      out = Tensor(self.rt, self.rows, x.cols, "matmul", (self, x, index), index.rows * index.cols, dtype or "bf16")
      out.fidelity = fidelity
      return out
    # an unrealized sum times a small matrix goes term by term, as when RoPE turns the parts of a split K
    if x.tiles <= 16 and (terms := _terms(self)):
      return sum((t.matmul(x, dtype, fidelity) for t in terms[1:]), terms[0].matmul(x, dtype, fidelity))
    assert "int32" not in (self.dtype, x.dtype), "the FPU multiplies floats"
    dtype = dtype or ("fp32" if "fp32" in (self.dtype, x.dtype) else "bf16")
    if "fp32" in (self.dtype, x.dtype):  # autograd sees one op, and keeps the bf16 parts for the two fp32 matmuls of backward
      a, b = _parts(self), _parts(x)
      return _bf16x3(a, b)._track("matmul", (self, x), a, b).cast(dtype)
    assert self.cols == x.rows and (self.batch == x.batch or 1 in (self.batch, x.batch)), f"{self.shape} @ {x.shape}"
    if self.cols == 1: return (self * x).cast(dtype)  # (R, 1) @ (1, C) is broadcasting
    k, tr = -(-self.cols // 32), -(-self.rows // 32)
    # A row of tiles, like a token's MLP, has too few blocks to keep the workers busy. Cut K into s parts as a batch,
    # then sum them. With more rows this doesn't help: the matrices of a batch read their own tiles, while blocks of the
    # same matrix share them.
    s = max([s for s in (2, 4, 8) if self.batch == x.batch == 1 and tr == 1 and self.cols % (32 * s) == 0 and k // s >= 4
             and s * -(-x.cols // 32) <= len(self.rt.cores)], default=1)
    if s > 1:
      parts = self.split(s).matmul(x.split(s, 1), dtype, fidelity)
      return sum((parts[i] for i in range(1, s)), parts[0])
    out = Tensor(self.rt, self.rows, x.cols, "matmul", (self, x), max(self.batch, x.batch), dtype)
    out.fidelity = fidelity
    return out

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
    if window < 0 or parts is not None and parts <= 0: raise ValueError("attention needs a nonnegative window and positive parts")
    ht, kt, rep, ra = -(-self.cols // 32), -(-k.rows // 32), k.rep, -(-self.rows // 32)
    assert self.dtype == "bf16" and k.dtype == v.dtype and k.dtype in ("bf16", *BFP) and v.rep == rep and self.batch == k.batch == v.batch
    assert self.cols == k.cols == v.cols and self.cols % 32 == 0 and self.cols >= 64 and k.rows == v.rows
    assert places.rows == self.rows and places.batch in (1, self.batch) and places.cols == 1 and places.dtype == "int32", \
      f"{self.shape} by {k.shape}, {places.shape}"
    units, wt = self.batch * -(-ra // 8), k.dtype if k.dtype in BFP else None
    P = parts or attention_parts(ht, self.rows, wt, kt, units, len(self.rt.cores))
    q, places = (_plain(t.realize()) for t in (self, places))
    k, v = k.realize(), v.realize()
    out = Tensor(self.rt, self.rows, self.cols + (33 if P > 1 else 1), batch=self.batch * P, dtype="fp32")
    # the exact row count is an argument, so the key rounds it up to tiles
    key = ("attend", ht, 32 * ra, rep, wt, fidelity, window, P, attention_store(ht, self.rows, wt, P)[1] > 0)
    out.run(key, lambda: attention_kernel(*key[1:]), *out._where(), *q._where(), *k._where(), *v._where(), *places._where(), kt,
            self.rows, units=units * P, name="attention")
    part = lambda p, at, cols: out._view(out.rows, cols, self.batch, p * out.bstride + at, out.bstride * P)  # cols of part p, from tile at
    if P == 1: return (part(0, 0, self.cols) / part(0, ht, 1)).cast("bf16")  # each sum fills its tile's 32 columns, a valid 1-column view
    largest = [part(p, ht + 1, 1) for p in range(P)]
    top = functools.reduce(Tensor.maximum, largest)
    weighted = lambda at, cols: functools.reduce(Tensor.__add__, [(m - top).exp() * part(p, at, cols) for p, m in enumerate(largest)])
    return (weighted(0, self.cols) / weighted(ht, 1)).cast("bf16")

  # Convolution and pooling of an image (channels, height, width), built from the ops above so autograd works on them
  def conv2d(self, w, padding=0):
    """Convolve with w, (out channels, in channels, k * k), as one im2col matmul. Tap (ky, kx) reads each channel,
    flattened to a row, from ky * W + kx on, so column y * W + x of the stacked taps is the patch at (y, x). Columns
    that wrap past the right edge are dropped."""
    k = math.isqrt(w.cols)
    assert k * k == w.cols and w.rows == self.batch, f"{self.shape} by {w.shape}: (out, in, k * k)"
    x = self.pad(((padding, padding), (padding, padding))) if padding else self
    (c, h, wd), ho, wo = x.shape, x.rows - k + 1, x.cols - k + 1
    rows = x.reshape(1, c, h * wd).pad(((0, 0), (0, k - 1)))  # k - 1 zeros at the end, for the wrapped columns of the last tap
    taps = [rows[:, :, ky * wd + kx:ky * wd + kx + ho * wd] for ky in range(k) for kx in range(k)]
    out = w.permute(0, 2, 1).reshape(1, w.batch, k * k * c) @ taps[0].cat(*taps[1:], axis=1)
    return out.reshape(w.batch, ho, wd)[:, :, :wo]
  def max_pool2d(self, k=2):  # max of each k x k block: over each k rows, then each k columns
    c, h, w = self.shape
    assert h % k == 0 and w % k == 0, f"{self.shape} in {k} x {k} blocks"
    return self.reshape(c * h // k, k, w).max(1).reshape(1, c * h * w // (k * k), k).max(2).reshape(c, h // k, w // k)

  # Autograd, like tinygrad's: a rule per op (_backward), made of the same lazy ops, so the backward fuses into kernels
  # too. It recomputes what the forward fused, since that wasn't stored.
  def gradient(self, *targets, gradient=None):
    """Build d(self)/d(target) for each target as lazy tensor operations. Return None for a target that
    self doesn't depend on. self must be scalar unless gradient supplies the derivative of the desired
    output with respect to self."""
    assert gradient is not None or self.shape == (1, 1, 1), "self must be a single value, or pass gradient"
    assert gradient is None or gradient.shape == self.shape, "gradient must have the output's shape"
    if gradient is not None: self._same_runtime(gradient)
    order, seen = [], set()
    def visit(t):  # what self was made of, operands first
      if id(t) in seen or t._ctx is None: return
      seen.add(id(t))
      for s in t._ctx[1]:
        if s.requires_grad: visit(s)
      order.append(t)
    visit(self)
    grads = {id(self): gradient if gradient is not None else Tensor.full(self.rt, 1, 1, 1.0, self.dtype)}
    wanted, found = {id(t) for t in targets}, {}
    for t in reversed(order):
      if (g := grads.pop(id(t), None)) is None: continue
      if id(t) in wanted: found[id(t)] = g
      for s, gs in zip(t._ctx[1], _backward(t, g)):
        if gs is not None and s.requires_grad: grads[id(s)] = grads[id(s)] + gs if id(s) in grads else gs
    grads.update(found)
    return [_full(grads[id(t)], t.shape) if id(t) in grads else None for t in targets]
  def backward(self, gradient=None):
    """Accumulate gradients in .grad for leaves that require them, and return self. Existing gradients are
    added to, as in PyTorch and tinygrad."""
    leaves, seen = [], set()
    def visit(t):
      if id(t) in seen: return
      seen.add(id(t))
      if t._ctx is None: leaves.append(t)
      for s in t._ctx[1] if t._ctx else ():
        if s.requires_grad: visit(s)
    visit(self)
    for t, g in zip(leaves, self.gradient(*leaves, gradient=gradient)):
      if g is not None: t.grad = g if t.grad is None else t.grad + g
    return self

def _shape(shape):
  if not 1 <= len(shape) <= 3: raise ValueError("a tensor needs one to three dimensions")
  shape = tuple(map(operator.index, shape))
  if any(n <= 0 for n in shape): raise ValueError(f"tensor dimensions must be positive: {shape}")
  return (1,) * (3 - len(shape)) + shape

def _unbroadcast(g, shape):  # the gradient of a broadcast operand: g summed over the axes where shape is 1 and g isn't
  for ax in range(3):
    if shape[ax] == 1 and g.shape[ax] > 1: g = g.sum(ax)
  return g
def _full(g, shape): return g if g.shape == shape else g.expand(*shape)  # for ops that can't take g broadcast

def _backward(t, g):
  """The gradients of t's operands, given t's gradient g. A condition or a comparison has none, and gets None."""
  op, srcs, args = t._ctx
  x, y = (srcs + (None,))[:2]
  if op in ("add", "sub", "mul", "max", "keep", "drop", "cmplt", "cmpne"):
    if op in ("cmplt", "cmpne"): return (None, None)
    if op in ("keep", "drop"): return (None, _unbroadcast(x._binop(op, g), y.shape))  # halves of where: the condition has no gradient
    if op == "add": gx, gy = g, g
    elif op == "sub": gx, gy = g, -g
    elif op == "mul": gx, gy = g * y, g * x
    else: m = (x > y) + x.eq(y) * 0.5; gx, gy = g * m, g * (1 - m)  # max: ties split it, like tinygrad
    return (_unbroadcast(gx, x.shape), _unbroadcast(gy, y.shape))
  if op in FOLDS:  # a sum broadcasts back. a max goes to its maxima, split between ties
    if op[:3] == "sum": return (g,)
    m = x.eq(t)
    return (g * m / m.sum(FOLDS[op]),)
  unary = {"exp": lambda: g * t, "exp2": lambda: g * t * math.log(2), "log": lambda: g * x.reciprocal(),
           "log2": lambda: g * (x * math.log(2)).reciprocal(), "sin": lambda: g * x.cos(), "cos": lambda: -(g * x.sin()),
           "sqrt": lambda: g * (t * 2).reciprocal(), "recip": lambda: -(g * t * t), "relu": lambda: g * (x > 0),
           "cast": lambda: g.cast(x.dtype)}
  if op in unary: return (unary[op](),)
  g = _full(g, t.shape)  # movement and matmul can't take g broadcast
  if op == "matmul":  # a transposed B gets its gradient as (g.T @ x).T, and the T's own rule cancels that transpose
    turn = y._ctx is not None and y._ctx[0] == "T"
    if args:  # fp32: the forward's parts, transposed, and g's parts, made once for both
      gp, (xp, yp) = _parts(g), args
      gx, gy = _bf16x3(gp, _pT(yp)), _bf16x3(_pT(gp), xp) if turn else _bf16x3(_pT(xp), gp)
    else: gx, gy = g.matmul(y.T, x.dtype), g.T.matmul(x, y.dtype) if turn else x.T.matmul(g, y.dtype)
    gx, gy = _unbroadcast(gx.cast(x.dtype), x.shape), gy.cast(y.dtype)
    return gx, _unbroadcast(gy, (y.batch, y.cols, y.rows)).T if turn else _unbroadcast(gy, y.shape)
  if op == "T": return (g.T,)
  if op == "reshape": return (g.reshape(*x.shape),)
  if op == "permute": return (g.permute(*sorted(range(3), key=args[0].__getitem__)),)
  if op == "expand": return (_unbroadcast(g, x.shape),)
  if op == "pad": return (g[tuple(slice(w[0], w[0] + n) for w, n in zip(args[0], x.shape))],)
  if op == "index": return (g.pad(((args[0], x.batch - args[0] - 1), (0, 0), (0, 0))),)
  if op == "split": return (g.merge() if args[0] == 2 else g.reshape(*x.shape),)
  if op == "repeat": return (g.reshape(x.batch, args[0], x.rows * x.cols).sum(1).reshape(*x.shape),)
  if op == "merge": return (g.split(x.batch),)
  if op == "slice":  # back into place with zeros around it. a reversed axis is reversed again
    assert all(abs(r.step) == 1 for r in args[0]), "a slice with a step has no backward"
    g = g[tuple(slice(None, None, r.step) for r in args[0])] if any(r.step < 0 for r in args[0]) else g
    lo = [min(r) for r in args[0]]
    return (g.pad([(a, n - a - len(r)) for a, n, r in zip(lo, x.shape, args[0])]),)
  if op == "gather":  # sum each row's gradient into its source row: one-hot(ids)^T @ g, whose row i is 1 where ids are i
    return ((Tensor.of(t.rt, [[i] for i in range(x.rows)], "int32") == args[0]).bfloat16().matmul(g, x.dtype),)
  raise NotImplementedError(f"no backward for {op}")

def _fit(t):
  """Realize subexpressions until the remaining tree fits a kernel. Realize long matmuls before reductions to
  avoid recomputing them on each pass. Also realize elementwise values shared with work outside the
  reduction. Above MAX_OPS, prefer the largest shared value; otherwise choose a subtree that fits along the
  heaviest branch."""
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
def _parts(t):  # an fp32 matmul operand as bf16 (hi, lo), lo None for bf16. a transpose splits its source, then transposes the parts
  if t.op == "T": return _pT(_parts(t.srcs[0]))
  if t.dtype == "bf16": return t, None
  return t.bfloat16(), (t - t._unary("ftob")).bfloat16()  # lo is t minus t rounded as the bf16 cast rounds, in one kernel
def _pT(parts): return tuple(p if p is None else p.T for p in parts)
def _plain(t): return (t + 0).realize() if t.rep > 1 else t  # copy a repeat, for a kernel that finds tiles by their position
def _rows_free(e):
  """Whether the kernel needs only the output row count, supplied as an argument. Row reductions, transposed B
  and indexed B also need input row counts."""
  if isinstance(e, int): return True
  if FOLDS.get(e[0]) == 1 or e[0] == "matmul" and isinstance(e[2], tuple) and e[2][0][0] in "TI": return False
  return all(_rows_free(x) for x in e[1:] if isinstance(x, tuple))
def _bf16x3(a, b):  # hi@hi + (hi@lo + lo@hi) in fp32, from the parts of two operands
  (ah, al), (bh, bl) = a, b
  out, lo = ah.matmul(bh, "fp32", 4), [p.matmul(q, "fp32", 4) for p, q in ((ah, bl), (al, bh)) if p is not None and q is not None]
  return out + sum(lo[1:], lo[0]) if lo else out

def _inside(o, s, lo, hi, n):  # the e in [0, n) with lo <= o + s e < hi, as (first, end)
  if s == 0: return (0, n) if lo <= o < hi else (0, 0)
  if s < 0: o, s, lo, hi = -o, -s, 1 - hi, 1 - lo
  first, end = max(-((o - lo) // s), 0), min(-((o - hi) // s), n)
  return (first, end) if first < end else (0, 0)

WHOLE = ("matmul", "T", "move", "cast")  # ops whose operands are realized first, not fused into the same kernel
def _below(t, root):  # what a kernel for t computes with it: not the operands of a WHOLE op, unless it's a cast at the root
  return t.srcs if t.op not in WHOLE or root and t.op == "cast" else ()
def _lazy(t, seen, root=True):  # the unrealized ops under t, t first
  if t.op is None or id(t) in seen: return
  seen.add(id(t))
  yield t
  for s in _below(t, root): yield from _lazy(s, seen, False)

def _ops(t): return sum(1 for _ in _lazy(t, set()))  # how many ops a kernel for t would hold, at most

def _tree(t, inputs):
  """The expression that fused_kernel computes for t, with realized tensors as indices into inputs, and the dtype it
  computes in. The kernel sweeps along the axis of the first reduction. Reductions along other axes are realized
  first."""
  dtype = t.srcs[0].dtype if t.op in ("cast", "matmul") else t.dtype
  assert dtype not in BFP, "a block float can only be a matmul's B"
  axis = next((FOLDS[u.op] for u in _lazy(t, set()) if u.op in COMBINE), 2)
  n = ext(t.shape, axis)  # the sweep is as long as t along the axis. if t is a tile long there, its first reduction sets it
  return _expr(t, t.shape, inputs, [axis, n if n > 1 else None, set(), dtype, t]), dtype

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
    assert not t.runs, "a tensor by columns can only be a matmul's B, or read back"
    return _input(t, inputs)
  if t is kernel[4] and t.op == "cast": return ("cast", _expr(t.srcs[0], shape, inputs, kernel))
  if t is not kernel[4] and (t.dtype != kernel[3] or t.op == "cast" or (t.op == "matmul" and kernel[3] != "bf16")):
    return _expr(t.realize(), shape, inputs, kernel)
  axis = kernel[0]
  if any(t.shape[ax] != shape[ax] for ax in range(3) if ax != axis): return _expr(t.realize(), shape, inputs, kernel)
  if t.op in COMBINE:
    if FOLDS[t.op] != axis: return _expr(t.realize(), shape, inputs, kernel)
    n = ext(t.srcs[0].shape, axis)
    if n > 1 and kernel[1] is None: kernel[1] = n
    if (n > 1 and n != kernel[1]) or (id(t) not in kernel[2] and len(kernel[2]) == len(ACCS)): return _expr(t.realize(), shape, inputs, kernel)
    kernel[2].add(id(t))
  if t.op in ("T", "move"): return _expr(t.realize(), shape, inputs, kernel)  # kernels of their own, but a matmul can read B transposed (below)
  if t.op == "matmul":  # a matmul reads its operands as matrices, not tile by tile, so realize them first
    a, b, *picked = t.srcs  # picked: the indices that pick B's matrices, if any
    if b.op == "T":  # read transposed. a block float, like a bfp8 key cache, is read as is and transposed on the way in
      src = b.srcs[0].realize()
      assert not src.runs, "a tensor by columns can only be a matmul's B as it is, not transposed"
      b = ("T" + src.dtype, _input(src, inputs)) if src.dtype in BFP else ("T", _expr(src, shape, inputs, kernel))
    elif b.runs: b = ("C" + (b.dtype if b.dtype in BFP else ""), _input(b, inputs))  # laid out by columns()
    else: b = (b.dtype, _input(b.realize(), inputs)) if b.dtype in BFP else _expr(b.realize(), shape, inputs, kernel)
    if picked: b = ("I" + b[0], b[1], _input(picked[0].realize(), inputs)) if isinstance(b, tuple) else ("I", b, _input(picked[0].realize(), inputs))
    return ("matmul", _expr(a.realize(), shape, inputs, kernel), b) + ((t.fidelity,) if t.fidelity != 4 else ())
  return (t.op, *(_expr(s, shape, inputs, kernel) for s in t.srcs))

def _input(t, inputs):  # the index of a realized tensor among a kernel's inputs, added if new
  for i, u in enumerate(inputs):
    if u is t: return i
  inputs.append(t)
  return len(inputs) - 1

class Recording:
  """Record launches and keep their DRAM allocations for replay. Rebuild commands when a kernel has been
  evicted from L1. If the kernels don't fit together, replay reloads them in batches."""
  def __init__(self, rt): self.rt, self.launches, self.block = rt, [], None
  def __enter__(self):
    if self.rt.log is not None: raise RuntimeError("cannot nest recordings on the same runtime")
    self.pinned = self.rt.pinned.copy()
    self.rt.log = []
    return self
  def __exit__(self, kind, *_):
    self.launches, self.rt.log = self.rt.log, None
    if kind is not None:  # failed recordings never replay; keep live tensors and release abandoned allocations
      new = self.rt.pinned - self.pinned
      self.rt.pinned -= new
      freed = self.rt.pinned_freed & new
      self.rt.pinned_freed -= new
      for addr in freed: self.rt.dram_free(addr)
      self.launches = []
    del self.pinned
  def send(self):
    if self.block is None or self.block[2] != self.rt.evictions:
      evictions = self.rt.evictions
      self.block = (*self.rt.block(self.launches), evictions)
    self.rt.send(*self.block[:2])

def jit(f):
  """Run f normally on the first call, record the second call's launches, then replay them.
  Later calls copy inputs into the recorded buffers and skip f's Python code.

  Keep input shapes and dtypes fixed. Don't replace tensors captured by f or read tensors back to
  the host inside f. Recorded assign operations can update captured tensors in place. Results and
  retained tensors, including .grad, reuse their buffers; the next call overwrites their contents."""
  signature, inputs, out, rec = None, (), None, None
  @functools.wraps(f)
  def run(*xs):
    nonlocal signature, inputs, out, rec
    if not xs or not all(isinstance(x, Tensor) for x in xs): raise ValueError("jit inputs must be tensors, with at least one input")
    rt = xs[0].rt
    if any(x.rt is not rt for x in xs): raise ValueError("jit inputs must use the same runtime")
    if signature is None:
      result = f(*xs).realize()
      signature = [(x.rt, x.shape, x.dtype) for x in xs]
      return result
    if len(xs) != len(signature): raise ValueError(f"jit expected {len(signature)} inputs, got {len(xs)}")
    for x, (runtime, shape, dtype) in zip(xs, signature):
      if x.rt is not runtime: raise ValueError("jit inputs must use the recorded runtime")
      assert (x.shape, x.dtype) == (shape, dtype), f"{x.shape} {x.dtype} where jit had {shape} {dtype}"
    if rec is None:
      inputs = [x.realize() for x in xs]
      with Recording(rt) as recording: result = f(*inputs).realize()
      rec, out = recording, result
      return out
    for x, to in zip(xs, inputs):
      if x is not to: to.assign(x)
    rec.send()
    return out
  return run

# Compiled kernels by key. Fused entries hold (image, units, name); other entries hold an image.
kernels = {}
def _cached(key, make):
  if key not in kernels: kernels[key] = make()
  return kernels[key]
