# Use baretile as a tinygrad device. This adapter compiles tinygrad graphs before scheduling.
# Movement operations remain views until an operation needs their data. Unsupported operations,
# including random-number generation and other dtypes, run on tinygrad's CPU device.
# TinyJit records and replays launches through tensor.jit.
#   import adapters.tinygrad_baretile, then DEV=BARETILE or Tensor(..., device="BARETILE")
#   TTSIM=/path/to/libttsim.so python3 adapters/tinygrad_baretile.py script.py [args]
# PyTorch uses tinygrad's torch frontend (extra/torch_backend, the "tiny" device).
import collections, dataclasses, math, os, runpy, sys, types
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import tinygrad, tinygrad.tensor, tinygrad.engine.jit
from tinygrad.helpers import JIT
from tinygrad.engine.jit import JitError, _check_no_non_tensor_return, _prepare_jit_inputs
from tinygrad.nn.state import get_parameters
from tinygrad import Tensor, dtypes
from tinygrad.dtype import _to_np_dtype
from tinygrad.device import Allocator, BufferStorage, Compiled, Device
from tinygrad.uop.ops import GroupOp, Ops, PatternMatcher, UOp, UPat, graph_rewrite, identity_element
from tinygrad.schedule.prepare import resolve_function
from baretile import pci, tensor
from baretile.chip import Chip
from baretile.runtime import Runtime

# A View is a small ShapeTracker over a baretile tensor's elements in row-major order. When an op needs the data, the
# View becomes a tensor of three axes: the tensor as stored, a transpose that a matmul reads in place, one move, or
# several moves that swap axes.
NP = {"fp32": np.float32, "bf16": np.uint16, "int32": np.int32}  # bf16 as raw bits

RT = None  # all workers of the chip, unless you set it first. a process can only open the simulator once
def runtime():
  global RT
  if RT is None:
    chip = Chip(pci.open_device())
    RT = Runtime(chip)
  return RT

def fold(shape):  # any rank to (batch, rows, cols). the axes before the last become rows, so short axes aren't padded
  return (1, math.prod(shape[:-1]), shape[-1]) if shape else (1, 1, 1)

class View:
  """A shape over a tensor's elements in row-major order. Element (i0, i1, ...) is element
  offset + sum(i_k * strides_k), or 0 where an index is outside the mask's (lo, hi) on its axis."""
  def __init__(self, shape, strides=None, offset=0, mask=None):
    self.shape, self.offset, self.mask = tuple(shape), offset, mask
    self.strides = tuple(strides) if strides is not None else tuple(math.prod(shape[k + 1:]) for k in range(len(shape)))
  def contiguous(self):  # the elements as stored. an axis of 1 can have any stride
    return self.offset == 0 and self.mask is None and all(n == 1 or s == c for n, s, c in zip(self.shape, self.strides, View(self.shape).strides))
  def permute(self, order):
    return View([self.shape[k] for k in order], [self.strides[k] for k in order], self.offset, self.mask and [self.mask[k] for k in order])
  def expand(self, shape):
    # an expanded axis is all padding if the element it repeats was padding
    mask = self.mask and [(0, m if w[0] < w[1] else 0) if n != m else w for w, n, m in zip(self.mask, self.shape, shape)]
    return View(shape, [0 if n != m else s for s, n, m in zip(self.strides, self.shape, shape)], self.offset, mask)
  def shrink(self, ranges):
    mask = self.mask and [(max(lo - a, 0), min(hi - a, b - a)) for (lo, hi), (a, b) in zip(self.mask, ranges)]
    return View([b - a for a, b in ranges], self.strides, self.offset + sum(a * s for (a, _), s in zip(ranges, self.strides)), mask)
  def pad(self, widths):
    mask = [(lo + a, hi + a) for (lo, hi), (a, _) in zip(self.mask or [(0, n) for n in self.shape], widths)]
    offset = self.offset - sum(a * s for (a, _), s in zip(widths, self.strides))
    return View([n + a + b for n, (a, b) in zip(self.shape, widths)], self.strides, offset, mask)
  def reshape(self, shape):
    """Return a view with the new shape, or None if reshaping requires a copy. Ignore axes of size 1. Merge
    adjacent axes only when their strides are contiguous; split an axis when its size is divisible by the
    new size."""
    if math.prod(shape) != math.prod(self.shape): return None
    if self.contiguous(): return View(shape)
    if self.mask is not None and any(w != (0, n) for w, n in zip(self.mask, self.shape)): return None  # an axis of 1 can be all padding
    old = [(n, s) for n, s in zip(self.shape, self.strides) if n > 1]
    strides, i = [], 0
    for n in shape:
      if n == 1: strides.append(0); continue
      size, s = old[i]
      while size < n and i + 1 < len(old) and old[i][1] == old[i + 1][0] * old[i + 1][1]: i += 1; size *= old[i][0]; s = old[i][1]
      if size == n: strides.append(s); i += 1
      elif size % n == 0: strides.append(s * (size // n)); old[i] = (size // n, s)  # split: the rest goes to the next axes
      else: return None
    return View(shape, strides, self.offset)

class Value:
  """A value on the chip: a View of a baretile tensor, which tinygrad sees as dtype."""
  def __init__(self, t, view, dtype): self.t, self.view, self.dtype = t, view, dtype
  @property
  def shape(self): return self.view.shape
  def whole(self): return self.view.contiguous() and math.prod(self.shape) == math.prod(self.t.shape)  # the view is the tensor as stored
  def covers(self):  # the view is the tensor's elements in some order, maybe broadcast
    return not self.view.mask and math.prod(n for n, s in zip(self.shape, self.view.strides) if s) == math.prod(self.t.shape)

# A matmul reads a bf16 tensor cast to fp32 as the bf16 tensor, and skips the cast. baretile multiplies fp32 by bf16
# exactly, since bf16 has no lo part, and it reads a bf16 transpose in place.
def uncast(v):
  if isinstance(v, Value) and v.t.op == "cast" and v.t.dtype == "fp32" and v.t.srcs[0].dtype == "bf16":
    return Value(v.t.srcs[0], v.view, dtypes.bfloat16)
  return v

def merge(view, groups):
  """Merge each group of axes into (size, stride, lo, hi). Return None if strides aren't contiguous,
  or if padding spans several axes or sits beside another axis larger than 1."""
  out = []
  for g in groups:
    axes, size = [k for k in g if view.shape[k] > 1], math.prod(view.shape[k] for k in g)
    masked = [k for k in g if view.mask and view.mask[k] != (0, view.shape[k])]  # axes of 1 too, since they can be all padding
    if any(view.strides[a] != view.strides[b] * view.shape[b] for a, b in zip(axes, axes[1:])): return None
    if any(view.mask[k][0] >= view.mask[k][1] for k in masked): lo, hi = 0, 0  # an axis that's all padding makes the group all padding
    elif len(masked) > 1 or masked and len(axes) > 1: return None
    else: lo, hi = view.mask[masked[0]] if masked else (0, size)
    out.append((size, view.strides[axes[-1]] if axes else 0, lo, hi))
  return out

def groups3(shape):  # the axis groups that fold makes
  return [[], list(range(len(shape) - 1)), [len(shape) - 1] if shape else []]

def grouping(shape, shape3):  # split shape's axes into three consecutive groups of sizes shape3, or None
  for i in range(len(shape) + 1):
    for j in range(i, len(shape) + 1):
      if (math.prod(shape[:i]), math.prod(shape[i:j]), math.prod(shape[j:])) == tuple(shape3):
        return [list(range(i)), list(range(i, j)), list(range(j, len(shape)))]
  return None

def materialize(v, groups=None):
  """Convert v to a baretile tensor with three axes, grouping its axes as fold does by default. Try the
  existing layout, a transpose, a single move, then several moves to swap axis groups. Fall back to a host
  copy if none works."""
  groups = groups or groups3(v.shape)
  shape3, t = tuple(math.prod(v.shape[k] for k in g) for g in groups), v.t
  if v.whole() and t.shape == shape3: return t
  if v.whole(): return t.reshape(*shape3)
  m = merge(v.view, groups)
  swapped = [shape3[1] * shape3[2], 1, shape3[1]]  # strides with t's last two axes swapped. a batch of 1 can have any
  if m is not None and all(n == 1 or s == w for n, (_, s, _, _), w in zip(shape3, m, swapped)) and v.view.offset == 0 and not v.view.mask \
     and t.shape == (shape3[0], *shape3[:0:-1]): return t.T
  if m is not None:
    return t.move(shape3, v.view.offset, [s for _, s, _, _ in m], [lo for _, _, lo, _ in m], [hi for _, _, _, hi in m])
  if (p := transposes(v)) is not None: return p if p.shape == shape3 else p.reshape(*shape3)
  return upload(download(v).reshape(shape3), v.t.dtype)

def transposes(v):
  """Convert an axis permutation, such as (B, T, H, D) to (B, H, T, D), into moves that swap two groups of
  axes. Each move treats its source as row-major elements. Splitting attention heads out or merging them
  back takes two moves. Return None for other views."""
  axes = [k for k, n in enumerate(v.shape) if n > 1]
  order = sorted(axes, key=lambda k: -v.view.strides[k])  # the axes in the order the tensor stores them
  if v.view.offset or v.view.mask or [v.view.strides[k] for k in order] != list(View([v.shape[k] for k in order]).strides): return None
  t, size = v.t, lambda ks: math.prod(v.shape[q] for q in ks)
  for i, k in enumerate(axes):
    j = order.index(k)
    if j == i: continue
    p, y, z = size(order[:i]), size(order[i:j]), size(order[j:])
    t = t.move((p, z, y), 0, (y * z, 1, z))  # blocks y and z swapped
    order = order[:i] + order[j:] + order[i:j]
  return t

def indices(view):  # the tensor index of each element of the view, masked ones included
  if not view.shape: return np.array(view.offset)
  return view.offset + sum(np.arange(n).reshape([-1 if j == k else 1 for j in range(len(view.shape))]) * s
                           for k, (n, s) in enumerate(zip(view.shape, view.strides)))

def download(v):  # bf16 as raw bits
  a = np.frombuffer(v.t.tobytes(), NP[v.t.dtype])
  view = v.view
  out = np.asarray(a[np.clip(indices(view), 0, a.size - 1)])  # asarray: a 0-d array for an empty shape, not a scalar
  if view.mask is not None:
    for k, (lo, hi) in enumerate(view.mask):
      keep = (np.arange(view.shape[k]) >= lo) & (np.arange(view.shape[k]) < hi)
      out = np.where(keep.reshape([-1 if j == k else 1 for j in range(len(view.shape))]), out, out.dtype.type(0))  # where, since inf * 0 is NaN
  return out

def upload(a, dtype=None, shape3=None):
  """Upload NumPy array a as dtype. The default is fp32, or int32 for integers and booleans. Pass bf16 values
  as raw bits."""
  dtype = dtype or ("int32" if a.dtype.kind in "iub" else "fp32")
  shape3 = shape3 or fold(a.shape)
  return tensor.Tensor.frombytes(runtime(), *shape3[1:], np.ascontiguousarray(a, NP[dtype]).tobytes(), shape3[0], dtype)

def broadcast(v, shape):  # numpy broadcasting, as a view
  if v.shape == tuple(shape): return v
  pad = len(shape) - len(v.shape)
  view = View((1,) * pad + v.shape, (0,) * pad + v.view.strides, v.view.offset, v.view.mask and [(0, 1)] * pad + list(v.view.mask))
  return Value(v.t, view.expand(shape), v.dtype)

def common_groups(shape, values):  # fold an elementwise op the way an operand's tensor already is, else fold's way
  return next((g for v in values if isinstance(v, Value) and v.whole() and (g := grouping(shape, v.t.shape))), groups3(shape))

def operand(v, shape, groups):
  """Prepare v for an elementwise operation with the given shape and axis groups. Keep broadcast dimensions at
  size 1 when each group folds to either 1 or its full size. Otherwise materialize the broadcast."""
  whole = [not v.view.mask or v.view.mask[k] == (0, n) for k, n in enumerate(v.view.shape)]
  small = [1 if s == 0 and n > 1 and whole[k] else n for k, (n, s) in enumerate(zip(v.view.shape, v.view.strides))]
  view = View(small, [0 if n == 1 else s for n, s in zip(small, v.view.strides)], v.view.offset, v.view.mask and
              [(0, 1) if n < m else w for w, n, m in zip(v.view.mask, small, v.view.shape)])  # the mask of a broadcast axis, now 1
  sizes = lambda shp: [math.prod(shp[k] for k in g) for g in groups]
  if small != list(shape) and all(a in (b, 1) for a, b in zip(sizes(small), sizes(shape))): return materialize(Value(v.t, view, v.dtype), groups)
  return materialize(v, groups)

def spread(t, shape, groups):  # expand an elementwise result to shape when all its operands broadcast
  shape3 = tuple(math.prod(shape[k] for k in g) for g in groups)
  return t if t.shape == shape3 else t.expand(*shape3)

def reduce(x, axes, op):
  """Reduce x over axes using sum or max. Return a baretile tensor with the remaining axes in row-major order.
  Reduce directly when axes covers the first or last one or two tensor dimensions; otherwise move the
  reduced axes to the end first."""
  kept = [k for k in range(len(x.shape)) if k not in axes]
  w = x.view.permute(kept + list(axes))
  f = math.prod(x.shape[k] for k in axes)
  b, r, c = x.t.shape
  w_whole, front = w.contiguous() and math.prod(w.shape) == b * r * c, x.whole() and list(axes) == list(range(len(axes)))
  if w_whole and c == f: t, along = x.t, [2]
  elif w_whole and r * c == f: t, along = x.t, [2, 1]
  elif front and b == f: t, along = x.t, [0]
  elif front and b * r == f: t, along = x.t, [0, 1]
  else: t, along = materialize(Value(x.t, w, x.dtype), [[], list(range(len(kept))), list(range(len(kept), len(w.shape)))]), [2]
  for ax in along: t = t.sum(ax) if op == "sum" else t.max(ax)
  return t

DTYPES = {dtypes.float32: "fp32", dtypes.bfloat16: "bf16", dtypes.int32: "int32", dtypes.bool: None}  # a bool is 0 or 1 in any dtype
def npdtype(dt): return np.uint16 if dt is dtypes.bfloat16 else _to_np_dtype(dt)

def cpu(a):  # bf16 as raw bits
  t = Tensor(np.ascontiguousarray(a), device="CPU")
  return t.bitcast(dtypes.bfloat16) if a.dtype == np.uint16 else t
def array(t):  # bf16 as raw bits
  t = t.realize()
  return t.bitcast(dtypes.uint16).numpy() if t.dtype is dtypes.bfloat16 else t.numpy()
def constant(t):  # the number, if a host tensor is a constant broadcast over a shape, else None
  return t.uop.base.arg if t.uop.base.op is Ops.CONST else None

def flat(a, n, npd):  # the bytes of n values, broadcasting a scalar
  a = np.asarray(a)
  return np.ascontiguousarray(np.broadcast_to(a, (n,)) if a.ndim == 0 else a.reshape(-1), npd).tobytes()

# While TinyJit records a call, recording has [holder, value before, stored into] for each buffer it changes, and
# hosted counts the host work it does on the values it reads, which a replay wouldn't redo.
recording, hosted = None, 0
def note_host_work():
  global hosted
  if recording is not None: hosted += 1

def to_chip(v, dtype):
  return Value(upload(array(v), DTYPES[dtype]), View(v.shape), dtype)

fixed = {}  # host operands that read no buffer, like constant math or an arange, by UOp. they can't change
def up(x, dtype, shape3):
  """Convert a host tensor to a chip operand with shape3. Use a scalar if all values are equal. Cache up to 256
  constant host tensors that read no buffer, computing and uploading each once."""
  if (key := (x.uop, dtype, shape3)) in fixed: return fixed[key]
  a = array(x.float() if x.dtype is dtypes.bfloat16 else x)  # values, not bf16 bits
  v = (int(c) if isinstance(c := a.flat[0].item(), bool) else c) if a.size and (a == a.flat[0]).all() else upload(a, dtype, shape3)
  if not any(u.op in (Ops.BUFFER, Ops.ALLOC, Ops.PARAM) for u in x.uop.toposort()):
    if len(fixed) >= 256: del fixed[next(iter(fixed))]
    fixed[key] = v
  return v

def on_baretile(u): return isinstance(u.device, str) and u.device.startswith("BARETILE")

# elementwise ops. baretile's comparisons give 1 or 0 in the operands' dtype, which is how bools are stored
UNARY = {Ops.EXP2: tensor.Tensor.exp2, Ops.LOG2: tensor.Tensor.log2, Ops.SIN: tensor.Tensor.sin, Ops.SQRT: tensor.Tensor.sqrt,
         Ops.RECIPROCAL: tensor.Tensor.reciprocal, Ops.NEG: lambda x: -x}
def is_tensor(x): return isinstance(x, tensor.Tensor)  # a baretile tensor, else a number
BINARY = {Ops.ADD: lambda a, b: a + b, Ops.SUB: lambda a, b: a - b, Ops.MUL: lambda a, b: a * b, Ops.CMPLT: lambda a, b: a < b,
          Ops.MAX: lambda a, b: a.maximum(b) if is_tensor(a) else b.maximum(a), Ops.CMPNE: lambda a, b: a.ne(b) if is_tensor(a) else b.ne(a),
          Ops.CMPEQ: lambda a, b: a.eq(b) if is_tensor(a) else b.eq(a), Ops.FDIV: lambda a, b: a * b.reciprocal() if is_tensor(b) else a / b,
          Ops.AND: lambda a, b: a * b, Ops.OR: lambda a, b: a.maximum(b) if is_tensor(a) else b.maximum(a)}  # AND and OR of bools only
ELEMENTWISE = {*UNARY, *BINARY, Ops.WHERE, Ops.CAST}
EMPTIABLE = GroupOp.Elementwise | GroupOp.Movement | {Ops.REDUCE, Ops.COPY}  # ops whose result can be empty

def spread_to(v, shape):  # broadcast an operand of a binary op: a view on the chip, tinygrad's expand on the host
  if isinstance(v, Tensor): return v.reshape((1,) * (len(shape) - v.ndim) + v.shape).expand(shape)
  return broadcast(v, shape) if isinstance(v, Value) else v

DEFER = object()  # a product that only sums read. the sum makes it a matmul, or forces it

class Evaluator:
  """Evaluate a tinygrad graph. Each UOp becomes a chip Value, a number or a tinygrad CPU tensor."""
  def __init__(self, users): self.values, self.users = {}, users

  def summed(self, u):  # True if u's uses are sums, maybe through a permute, so u is a matmul's product
    return all(r.op is Ops.REDUCE and r.arg[0] is Ops.ADD or r.op is Ops.PERMUTE and self.summed(r) for r in self.users[u])

  def force(self, u):
    if self.values[u] is DEFER: self.values[u] = self.eval(u, defer=False)
    return self.values[u]

  def operand(self, x, shape, groups):
    """x as a number or a baretile tensor, for an elementwise op of tinygrad shape folded in groups."""
    if isinstance(x, (bool, int, float)): return int(x) if isinstance(x, bool) else x
    if isinstance(x, Tensor):  # from the host: a number if it's a constant
      if (c := constant(x)) is not None: return int(c) if isinstance(c, bool) else c
      return up(x, "fp32" if dtypes.is_float(x.dtype) else "int32", tuple(math.prod(shape[k] for k in g) for g in groups))
    return operand(x, shape, groups)

  def chip_op(self, u, srcs):
    """Run u on the chip, or return None if baretile can't."""
    if u.op in ELEMENTWISE and (u.dtype not in DTYPES or any(isinstance(s, Value) and s.dtype not in DTYPES for s in srcs)): return None
    if u.op in (Ops.AND, Ops.OR) and u.dtype is not dtypes.bool: return None  # bitwise, on the host
    shape = tuple(u.shape)
    groups = common_groups(shape, srcs)
    as_u = lambda t: t.cast(DTYPES[u.dtype]) if DTYPES[u.dtype] else t  # baretile returns some int32 ops in fp32, like a product
    whole = lambda t: Value(spread(as_u(t), shape, groups), View(shape), u.dtype)
    if u.op in UNARY: return whole(UNARY[u.op](self.operand(srcs[0], shape, groups)))
    if u.op in BINARY: return whole(BINARY[u.op](*(self.operand(s, shape, groups) for s in srcs)))
    if u.op is Ops.WHERE:
      c, x, y = (self.operand(s, shape, groups) for s in srcs)
      if not is_tensor(c): return None
      return whole(c.where(x, y))
    # a cast of a view that covers its tensor, like a transposed weight, casts the tensor and keeps the view, so a
    # matmul can read the cast's source in place
    if u.op is Ops.CAST:
      if isinstance(v := srcs[0], Value) and v.covers() and DTYPES.get(u.dtype):
        return Value(v.t.cast(DTYPES[u.dtype]), v.view, u.dtype)
      x = self.operand(srcs[0], shape, groups)
      if not is_tensor(x): return None
      return whole(x.ne(0) if u.dtype is dtypes.bool else x.cast(DTYPES[u.dtype]))
    return None

  def reduce(self, u, x):  # REDUCE(op, n) reduces the first n axes of x
    return Value(reduce(x, list(range(u.arg[1])), "sum" if u.arg[0] is Ops.ADD else "max"), View(tuple(u.shape)), u.dtype)

  def matmul(self, u):
    """Recognize REDUCE(ADD) u as a matmul, or return None. Its source must multiply a and b, optionally
    through a PERMUTE. Both operands must span the reduced axes. Of the remaining axes, those spanned by
    both become batch dimensions; those spanned by a become rows, and those spanned by b become columns."""
    p = u.src[0]
    order = p.arg if p.op is Ops.PERMUTE else tuple(range(len(p.shape)))
    m = p.src[0] if p.op is Ops.PERMUTE else p
    if m.op is not Ops.MUL or not all(dtypes.is_float(s.dtype) and s.dtype in DTYPES for s in m.src): return None
    lift = lambda s, v: to_chip(v, s.dtype) if isinstance(v, Tensor) and constant(v) is None else v  # upload a weight still on the host
    a, b = (uncast(spread_to(lift(s, self.read(self.values.get(s))), m.shape)) for s in m.src)
    if not (isinstance(a, Value) and isinstance(b, Value)) or a.view.mask or b.view.mask: return None
    shape, summed = m.shape, list(order[:u.arg[1]])
    full = lambda v, k: shape[k] == 1 or v.view.strides[k] != 0
    if not all(full(a, k) and full(b, k) for k in summed): return None
    free = list(order[u.arg[1]:])
    batch = [k for k in free if full(a, k) and full(b, k)]
    rows, cols = [k for k in free if full(a, k) and not full(b, k)], [k for k in free if full(b, k) and not full(a, k)]
    if len(batch) + len(rows) + len(cols) != len(free) or not (rows or cols): return None
    pick = lambda v, axes: View([shape[k] for k in axes], [v.view.strides[k] for k in axes], v.view.offset)
    nb, nr, nk = len(batch), len(rows), len(summed)
    # rows that don't merge: the leading ones become a batch, which b repeats
    if not batch and nr > 1 and merge(pick(a, rows), [list(range(nr))]) is None: nb, nr = nr - 1, 1
    runs = lambda *sizes: [list(range(sum(sizes[:i]), sum(sizes[:i + 1]))) for i in range(3)]  # consecutive axes in groups of these sizes
    A = materialize(Value(a.t, pick(a, batch + rows + summed), a.dtype), runs(nb, nr, nk))
    B = materialize(Value(b.t, pick(b, batch + summed + cols), b.dtype), runs(len(batch), nk, len(cols)))
    out = A.matmul(B, DTYPES[u.dtype])  # u's dtype, so bf16s cast up to fp32 give an fp32 product
    made = batch + rows + cols  # the result's axes as made, permuted below into free's order for u
    return Value(out, View([shape[k] for k in made]).permute([made.index(k) for k in free]), u.dtype)

  def movement(self, u, x):
    """Apply a movement operation as a view. Materialize x first if the requested reshape can't be represented
    by its view."""
    shape = tuple(u.shape)
    if u.op is Ops.RESHAPE: view = x.view.reshape(shape)
    elif u.op is Ops.PERMUTE: view = x.view.permute(u.arg)
    elif u.op is Ops.EXPAND: view = broadcast(x, shape).view
    elif u.op is Ops.SHRINK: view = x.view.shrink(self.marg(u))
    else: view = x.view.pad(self.marg(u))
    if view is None:  # move x into the three axes that the new shape folds to
      x, view = Value(materialize(x, grouping(x.shape, fold(shape))), View(x.shape), x.dtype), View(shape)
    return Value(x.t, view, x.dtype)

  def marg(self, u):  # (start, end) for SHRINK and (before, after) for PAD, from their (start, size) STACK sources
    ints = lambda src: (src.arg,) if src.op is Ops.CONST else tuple(s.arg if s.op is Ops.CONST else self.values[s] for s in src.src)
    start, size = ints(u.src[1]), ints(u.src[2])
    if u.op is Ops.SHRINK: return [(a, a + n) for a, n in zip(start, size)]
    return [(a, n - old - a) for a, n, old in zip(start, size, u.src[0].shape)]

  def on_host(self, u, srcs):
    """Build u lazily on tinygrad's CPU device. tinygrad fuses these operations and computes them when the chip
    needs the result. Random-number generation uses this path because it sums a padded view that tinygrad
    doesn't materialize."""
    def there(s, v):  # a source for tinygrad: a chip value is downloaded (host work), a host value stays
      if not isinstance(v, (Value, Tensor)): return s
      return (note_host_work() or cpu(download(v)) if isinstance(v, Value) else v).cast(s.dtype).uop
    return Tensor(u.replace(src=tuple(there(s, v) for s, v in zip(u.src, srcs))))

  def eval(self, u, defer=True):
    if u.op in EMPTIABLE and 0 in u.shape: return Tensor.zeros(u.shape, dtype=u.dtype, device="CPU")
    if u.op is Ops.REDUCE and 0 in u.src[0].shape:
      return Tensor.full(u.shape, identity_element(u.arg[0], u.dtype), dtype=u.dtype, device="CPU")
    if defer and (u.op is Ops.MUL or u.op is Ops.PERMUTE and self.values.get(u.src[0]) is DEFER) and self.summed(u): return DEFER
    srcs = [self.values.get(s) for s in u.src]
    if u.op is not Ops.REDUCE: srcs = [self.force(s) if v is DEFER else v for s, v in zip(u.src, srcs)]  # a REDUCE may make a matmul of them
    if u.op is Ops.CONST: return u.arg
    if u.op is Ops.STACK and all(isinstance(v, (bool, int, float)) for v in srcs): return tuple(srcs)  # a shape. else values, stacked on the host
    if u.op in (Ops.BUFFER, Ops.ALLOC): return self.read(u)  # its contents before any STORE in this graph
    if u.op is Ops.COPY and on_baretile(u) and isinstance(v := self.read(srcs[0]), Tensor) and u.dtype in DTYPES and v.ndim:
      return to_chip(v, u.dtype)
    if u.op in (Ops.DETACH, Ops.CONTIGUOUS_BACKWARD, Ops.STAGE, Ops.COPY): return self.read(srcs[0])
    if u.op is Ops.AFTER: return self.now(u.src[0])
    if u.op is Ops.STORE: return self.store(u.src[0], self.read(srcs[1]))
    if u.op is Ops.SINK: return None
    x = [self.read(v) for v in srcs]
    if u.op is Ops.RESHAPE and u.src[0].op is Ops.BUFFER and on_baretile(u.src[0]) and (v := self.lay_out(u, x[0])) is not None: return v
    if u.op in (Ops.RESHAPE, Ops.PERMUTE, Ops.EXPAND, Ops.SHRINK, Ops.PAD):  # symbolic shapes go to the host
      return self.movement(u, x[0]) if isinstance(x[0], Value) and all(isinstance(d, int) for d in u.shape) else self.on_host(u, x)
    if u.op in BINARY or u.op is Ops.WHERE: x = [spread_to(v, u.shape) for v in x]
    if u.op is Ops.REDUCE and x[0] is DEFER:  # a deferred product: a matmul, or the product after all
      if (mm := self.matmul(u)) is not None: return mm
      x[0] = self.force(u.src[0])
    if u.op is Ops.REDUCE and isinstance(x[0], Value) and x[0].dtype in DTYPES and u.arg[0] in (Ops.ADD, Ops.MAX): return self.reduce(u, x[0])
    if any(isinstance(v, Value) for v in x) and (v := self.chip_op(u, x)) is not None: return v
    return self.on_host(u, x)

  def lay_out(self, u, x):
    """Store a BARETILE buffer in the shape requested by its reshape. tinygrad creates flat weight buffers;
    upload their host bytes in the requested shape, or move the existing tensor once. Return None when this
    layout can't be used."""
    h = u.src[0].buffer._buf
    if isinstance(h, Part): return None
    if isinstance(x, Tensor) and constant(x) is None and u.dtype in DTYPES and h.value is None:
      h.set(value=Value(upload(array(x).reshape(u.shape), DTYPES[u.dtype]), View(x.shape), u.dtype), stored=False)
    elif isinstance(x, Value) and x.whole() and x.t.shape != fold(u.shape):
      h.set(value=Value(materialize(Value(x.t, View(u.shape), x.dtype)).realize(), x.view, x.dtype), stored=False)
    else: return None
    return Value(h.value.t, View(u.shape), u.dtype)

  def now(self, x):  # x, a buffer or a view of one, as the buffers hold it after the STOREs so far
    if x.op in (Ops.BUFFER, Ops.ALLOC): return self.read(x)
    if x.op is Ops.AFTER: return self.now(x.src[0])
    v = self.now(x.src[0])
    return self.movement(x, v) if isinstance(v, Value) else self.on_host(x, [v] + [self.values.get(s) for s in x.src[1:]])

  def read(self, v):
    """Return v, resolving buffers to their current contents. A BARETILE buffer contains a chip value or bytes;
    other devices supply bytes."""
    if not (isinstance(v, UOp) and v.op in (Ops.BUFFER, Ops.ALLOC)): return v
    buf = v.buffer
    if buf.nbytes == 0: return Tensor.zeros(v.shape, dtype=v.dtype, device="CPU")
    if on_baretile(v):
      h = buf.ensure_allocated()._buf
      if h.value is not None: return Value(h.value.t, h.value.view, h.value.dtype)
      if (data := h.bytes) is None: return Tensor.zeros(buf.nbytes // v.dtype.itemsize, dtype=v.dtype, device="CPU")  # nothing stored yet
      # up to 32 values, like an optimizer's scalars, are uploaded now
      if isinstance(h, Holder) and v.dtype in DTYPES and len(data) <= 32 * v.dtype.itemsize:
        h.set(value=Value(upload(np.frombuffer(data, npdtype(v.dtype)), DTYPES[v.dtype]), View(v.shape), v.dtype))
        return Value(h.value.t, h.value.view, h.value.dtype)
      return cpu(np.frombuffer(data, npdtype(v.dtype)))  # on the host until a use gives it a shape
    return cpu(np.frombuffer(buf.as_memoryview(), npdtype(v.dtype)).copy().reshape(v.shape))

  def store(self, target, value):
    """Store value into target, reshaping if needed. Writes through a view update part of the buffer on the
    host and replace its bytes. Values read before that write retain the old contents."""
    chain = []
    while target.op not in (Ops.BUFFER, Ops.ALLOC):
      ok = target.op in (Ops.RESHAPE, Ops.AFTER, Ops.DETACH, Ops.SHRINK, Ops.PERMUTE) or target.op is Ops.STAGE and target.arg is None
      assert ok, f"can't store into {target.op}"
      chain, target = chain + [target], target.src[0]
    if target.buffer.nbytes == 0: return None
    buf = target.buffer.ensure_allocated()
    n, npd = buf.nbytes // target.dtype.itemsize, npdtype(target.dtype)
    if any(u.op in (Ops.SHRINK, Ops.PERMUTE) for u in chain):
      view = View((n,))  # the part, as a view of the buffer's elements
      for u in reversed(chain):
        if view is None or u.op in (Ops.AFTER, Ops.DETACH, Ops.STAGE): continue  # a bare STAGE keeps its source's buffer
        view = view.reshape(tuple(u.shape)) if u.op is Ops.RESHAPE else view.permute(u.arg) if u.op is Ops.PERMUTE else view.shrink(self.marg(u))
      assert view is not None and on_baretile(target) and isinstance(buf._buf, Holder), "a partial store needs a BARETILE buffer and a view"
      a = np.frombuffer(contents(buf._buf) or bytes(buf.nbytes), npd).copy()
      a[indices(view)] = np.broadcast_to(host(value, target.dtype), view.shape)
      buf._buf.set(a.tobytes())
      return None
    if isinstance(buf._buf, Part): raise NotImplementedError("storing into part of a BARETILE buffer is not supported")
    data = lambda: host(value, target.dtype)
    if not on_baretile(target): note_host_work(); buf.copyin(memoryview(flat(data(), n, npd)))
    elif isinstance(value, Value):
      # if the view is the tensor as stored, keep the tensor and its three axes. copy a tensor that's computed already,
      # since a replay writes over its tensors in place
      t = value.t if value.whole() else materialize(value)
      buf._buf.set(value=Value((t + 0 if t.op is None else t).realize(), View(value.shape), value.dtype))
    else: buf._buf.set(flat(data(), n, npd))
    return None

# tinygrad's side: the BARETILE device, its buffers, and the graphs it hands over
class Holder:  # contents of a BARETILE buffer: bytes from the host, or a chip Value
  def __init__(self): self.bytes, self.value = None, None
  # new contents, noted while TinyJit records a call. stored is False when the same value is laid out again
  def set(self, bytes=None, value=None, stored=True):
    if recording is not None: recording.setdefault(id(self), [self, self.value, False])[2] |= stored
    if bytes is not None: note_host_work()
    self.bytes, self.value = bytes, value

def host(v, dtype):  # numpy in tinygrad's layout for dtype: bf16 as raw bits, a bool as a byte
  if isinstance(v, Value):
    a = download(v)
    return a != 0 if v.dtype is dtypes.bool else a
  return array(v if isinstance(v, Tensor) else Tensor(v, dtype=dtype, device="CPU"))

def contents(h):  # a holder's bytes in tinygrad's layout, or None
  if h.value is None: return h.bytes
  note_host_work()  # downloading a chip value
  return np.ascontiguousarray(host(h.value, h.value.dtype)).tobytes()

class Part:  # a BARETILE buffer at an offset, as tinygrad views one. it's read-only, and reads go to its base each time
  value = None
  def __init__(self, base, offset, size): self.base, self.offset, self.size = base, offset, size
  @property
  def bytes(self): return None if (c := contents(self.base)) is None else c[self.offset:self.offset + self.size]

class BaretileAllocator(Allocator):
  lru = False  # with the LRU cache, a new buffer could get the old contents of a freed one
  def _alloc(self, size, options): return BufferStorage(Holder())
  def _copyin(self, dest, src):
    if isinstance(dest, Part): raise NotImplementedError("copying into part of a BARETILE buffer is not supported")
    dest.set(bytes(src))
  def _copyout(self, dest, src): dest[:] = c[:len(dest)] if (c := contents(src)) is not None else bytes(len(dest))
  def _offset(self, buf, size, offset): return Part(buf, offset, size)

class BaretileDevice(Compiled):
  def __init__(self, device): super().__init__(device, BaretileAllocator(self), [], None)
  def synchronize(self, timeout=None):
    if RT is not None: RT.wait(60 if timeout is None else timeout)

# inline a tinygrad function as its body, which stores its outputs. precompiled functions too, since tinygrad would
# cache their program, and baretile runs the graph
def inline_call(c):
  return resolve_function(c.replace(arg=dataclasses.replace(c.arg, precompile=False)) if c.arg.precompile else c)
inline = PatternMatcher([(UPat(Ops.CALL, name="c"), inline_call)])
def run(sink):
  sink = graph_rewrite(sink, inline, name="inline calls")
  if allocs := {u: UOp.new_buffer(u.device, u.max_numel(), u.dtype) for u in sink.toposort() if u.op is Ops.ALLOC}:
    sink = sink.substitute(allocs)  # an inlined function's ALLOCs get fresh buffers each call, like tinygrad's schedule
  users = collections.defaultdict(list)
  for u in (order := sink.toposort()):
    for s in u.src: users[s].append(u)
  ev = Evaluator(users)
  for u in order: ev.values[u] = ev.eval(u)

_schedule = tinygrad.tensor.create_linear_with_vars
def schedule(sink):  # graphs that touch BARETILE run here, the others in tinygrad
  if not any(u.op in (Ops.BUFFER, Ops.COPY) and on_baretile(u) for u in sink.toposort()): return _schedule(sink)
  run(sink)
  return UOp(Ops.LINEAR, src=()), {}

mod = types.ModuleType("tinygrad.runtime.ops_baretile")
mod.BaretileDevice = BaretileDevice
sys.modules["tinygrad.runtime.ops_baretile"], tinygrad.runtime.ops_baretile = mod, mod
if "BARETILE" not in Device._devices: Device._devices.append("BARETILE")
tinygrad.tensor.create_linear_with_vars = schedule

def on_chip(t):  # the holder of a BARETILE tensor that spans its buffer, with its contents uploaded in its shape, or None
  if t.uop.base.op is not Ops.BUFFER or not on_baretile(t.uop.base) or t.dtype not in DTYPES: return None
  h = t.uop.base.buffer.ensure_allocated()._buf
  if not isinstance(h, Holder) or (h.value is None and h.bytes is None) or t.uop.base.buffer.nbytes != t.nbytes(): return None
  if h.value is None:
    h.set(value=Value(upload(np.frombuffer(h.bytes, npdtype(t.dtype)).reshape(t.shape), DTYPES[t.dtype]), View((t.numel(),)), t.dtype))
  return h

_tinyjit = tinygrad.engine.jit._TinyJit.__call__
def tinyjit(self, *args, **kwargs):
  """Record the second TinyJit call that runs on BARETILE, or a later call if the second can't be recorded. For
  replay, copy inputs into the recorded buffers and submit the recorded launches as a block. Copy updated
  buffers back to the addresses that the recording reads.

  Run the function normally if it downloads data, generates random numbers using the host counter, binds a
  symbolic variable, or uses inputs outside BARETILE. Also skip replay for partial-buffer inputs and stores
  into inputs or differently laid-out buffers. A recording that can't be replayed retains its DRAM
  allocations."""
  global recording, hosted
  given = [*args, *(v for _, v in sorted(kwargs.items()))]  # inputs as tinygrad finds them: tensors, and tensors in containers
  ins = [t for t in given if t.__class__ is Tensor]
  ins += [t for x in given if isinstance(x, (tuple, list, dict)) for t in (x.values() if isinstance(x, dict) else x)
          if t.__class__ is Tensor and not any(t is u for u in ins)]
  if not JIT or Device.DEFAULT != "BARETILE" and not any(on_baretile(t.uop) for t in ins): return _tinyjit(self, *args, **kwargs)
  _, var_vals, names, info = _prepare_jit_inputs(args, kwargs)  # tinygrad checks the inputs and realizes them
  holders, self.cnt = [on_chip(t) for t in ins], self.cnt + 1
  if recording is not None: raise RuntimeError("nested TinyJit calls are not supported")
  if self.cnt == 1 or getattr(self, "replay", None) is False or None in holders or var_vals:
    ret = self.fxn(*args, **kwargs)
    if params := get_parameters(ret): Tensor.realize(*params)
    return ret
  if self.cnt == 2 or not hasattr(self, "replay"):  # the second call, or a later one if the second couldn't be recorded
    values, recording, hosted, self.inputs = [h.value for h in holders], {}, 0, (names, info)
    try:
      with tensor.Recording(runtime()) as rec:
        ret = self.fxn(*args, **kwargs)
        if params := get_parameters(ret): Tensor.realize(*params)
        _check_no_non_tensor_return(ret)
        # Copy updated buffers back to the addresses read by the recording. Skip inputs: replay copies them in first
        # and reapplies any layout changes.
        back = [(h, old) for h, old, _ in recording.values() if old is not None and h not in holders]
        same = lambda h, old: isinstance(h.value, Value) and (h.value.t.shape, h.value.t.dtype) == (old.t.shape, old.t.dtype)
        into_input = any(stored and h in holders for h, _, stored in recording.values())  # a replay would lose that store
        self.replay = hosted == 0 and not into_input and all(same(h, old) for h, old in back) and (rec, values, ret)
        for h, old in back if self.replay else ():
          if h.value.t is not old.t: old.t.assign(h.value.t)
          h.value = Value(old.t, h.value.view, h.value.dtype)
    finally: recording = None
    if not rec.launches: raise JitError("no launches were recorded")
    return ret
  rec, values, ret = self.replay
  if (names, info) != self.inputs: raise JitError(f"args mismatch in JIT: {names, info} where it recorded {self.inputs}")
  for h, v in zip(holders, values):  # copy each input into the recording's, reshaped if needed
    if h.value.t is not v.t: v.t.assign(h.value.t if h.value.t.shape == v.t.shape else h.value.t.reshape(*v.t.shape))
  rec.send()
  return ret
tinygrad.engine.jit._TinyJit.__call__ = tinyjit

if __name__ == "__main__":  # run a tinygrad script on baretile, unchanged
  from tinygrad.helpers import DEV
  assert len(sys.argv) > 1, "usage: python3 adapters/tinygrad_baretile.py script.py [its arguments]"
  DEV.value = "BARETILE"
  sys.argv = sys.argv[1:]
  sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
  runpy.run_path(sys.argv[0], run_name="__main__")
