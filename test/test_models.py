# Model, server, training, adapter and benchmark tests (layer 4). See conftest.py for simulator setup.
import html, importlib, json, math, os, pathlib, re, struct, subprocess, sys, threading, time, unicodedata, pytest
import numpy as np
from types import SimpleNamespace
from baretile import pci, tensor
from baretile.tensor import Tensor
from baretile.runtime import Runtime
from conftest import bf16, value
import benchmark, train
from baretile.models import cost, gpt2, llama, qwen, serve

def test_train(chip):
  # Train a two-block GPT on a fixed counting sequence so documentation edits don't change the test data.
  # Compare fp32 Adam steps with PyTorch on the same weights, allowing accumulated matmul error in gradients.
  # The kernels don't all fit in L1, so replay must reload them in batches.
  pytest.importorskip("torch")
  losses = train.train(Runtime(chip), 5, data=" ".join(map(str, range(3000))))
  assert all(abs(f - p) < 1e-3 and g < 2e-3 for f, p, g, _ in losses), losses
  assert losses[-1][0] < losses[0][0] - 0.25 and abs(losses[0][0] - losses[0][3]) < 1e-3, losses

def test_adam_original_gradients(chip):
  rt = Runtime(chip)
  a, b = (Tensor.full(rt, 1, 1, v, "fp32").requires_grad_() for v in (1, 2))
  state = {id(p): [Tensor.full(rt, 1, 1, 0, "fp32") for _ in range(2)] for p in (a, b)}
  (a * b).backward()
  train.adam((a, b), state, Tensor.full(rt, 1, 1, 10, "fp32"), Tensor.full(rt, 1, 1, 1 / np.sqrt(0.001), "fp32"))
  assert np.allclose([state[id(p)][0].tolist()[0][0] for p in (a, b)], [0.2, 0.1])

def need_tinygrad(name="tinygrad"):
  """Import name from tinygrad. With TINYGRAD=/path/to/a/checkout, it comes from the checkout and must import, so CI
  can't skip the adapter tests by accident. Without TINYGRAD, an installed tinygrad will do, or the test skips."""
  if "TINYGRAD" not in os.environ: return pytest.importorskip(name)
  if os.environ["TINYGRAD"] not in sys.path: sys.path.insert(0, os.environ["TINYGRAD"])
  return importlib.import_module(name)

def adapter_on(chip):  # the tinygrad adapter, on a runtime of its own over the whole chip
  need_tinygrad()
  from baretile.adapters import tinygrad_device as adapter
  if adapter.RT is None or adapter.RT.closed: adapter.RT = Runtime(chip)
  return adapter

def test_tinygrad_jit_unrecorded(chip):
  # If the second TinyJit call can't be recorded, a later eligible call must still record. Here the second
  # call has a CPU input; subsequent calls put every input on BARETILE and should replay.
  rt = adapter_on(chip).RT
  from tinygrad import Tensor, TinyJit
  calls, x = [], Tensor([1.0, 2.0], device="BARETILE")
  @TinyJit
  def add(x, y):
    calls.append(1)
    return (x + y.to("BARETILE")).realize()
  for device in ("BARETILE", "CPU", "BARETILE", "BARETILE", "BARETILE"):
    assert add(x, Tensor([3.0, 4.0], device=device)).tolist() == [4.0, 6.0], device
  launches = rt.launches
  assert add(x, Tensor([5.0, 6.0], device="BARETILE")).tolist() == [6.0, 8.0] and rt.launches > launches
  assert len(calls) == 3 and add.replay, "eligible calls record again after a CPU-input call"
  from tinygrad import Device
  pending = (tensor.Tensor.full(rt, 1, 1, 2) + 3).realize()
  Device["BARETILE"].synchronize()
  assert rt.read_status(1) == rt.launches % 2**32 and pending.tolist() == [[5]], "synchronize completes queued chip work"

def test_tinygrad_jit_inputs(chip):
  adapter = adapter_on(chip)
  from tinygrad import Tensor, TinyJit
  calls, pins = [], dict(adapter.RT.pinned)
  @TinyJit
  def combine(x, y):
    calls.append(1)
    return (x * 10 + y).realize()
  x, y = (Tensor(a, device="BARETILE").realize() for a in ([1.0, 2.0], [3.0, 4.0]))
  for _ in range(2): assert combine(x, y).tolist() == [13, 24]
  assert combine(y, x).tolist() == [31, 42] and len(calls) == 2
  assert x.tolist() == [1, 2] and y.tolist() == [3, 4], "capture and replay preserve caller inputs"
  assert combine(x, x).tolist() == [11, 22], "repeated input buffers execute with their original aliasing"
  assert combine(x, y).tolist() == [13, 24] and combine(y, x).tolist() == [31, 42]
  combine.reset()
  assert dict(adapter.RT.pinned) == pins, "reset releases the capture immediately"
  assert combine(y, x).tolist() == [31, 42] and len(calls) == 5


@pytest.mark.parametrize("kind", ["direct", "slice", "reshape", "nested"])
def test_tinygrad_jit_returned_inputs(chip, kind):
  adapter = adapter_on(chip)
  from tinygrad import Tensor, TinyJit
  from tinygrad.nn.state import get_parameters
  calls, pins = [], dict(adapter.RT.pinned)
  @TinyJit
  def aliases(x):
    calls.append(1)
    computed = (x + 1).realize()  # a nonempty capture, even when the only result is an input alias
    if kind == "direct": return x
    if kind == "slice": return x[1:]
    if kind == "reshape": return x.reshape(2, 2)
    return {"input": x, "nested": [x[1:], (x.reshape(2, 2),)], "computed": computed}
  for n in (1, 2, 3, 4):
    values = [float(n + i) for i in range(4)]
    x = Tensor(values, device="BARETILE").realize()
    got = [t.tolist() for t in get_parameters(aliases(x))]
    shaped = [values[:2], values[2:]]
    want = {"direct": [values], "slice": [values[1:]], "reshape": [shaped],
            "nested": [values, values[1:], shaped, [v + 1 for v in values]]}[kind]
    assert got == want and x.tolist() == values, (kind, got, want)
    assert dict(adapter.RT.pinned) == pins, "returning an input alias releases its discarded capture"
  assert len(calls) == 4 and aliases.replay is False
  aliases.reset()
  assert dict(adapter.RT.pinned) == pins

@pytest.mark.parametrize("store", [False, True])
def test_tinygrad_jit_failed_capture(chip, store):
  adapter = adapter_on(chip)
  from tinygrad import Tensor, TinyJit
  calls, pins = [], dict(adapter.RT.pinned)
  @TinyJit
  def fails_once(x):
    calls.append(1)
    out = (x + 1).realize()
    if store: x.assign(out).realize()
    if len(calls) == 2: raise RuntimeError("capture failed after launching work")
    return out
  x = Tensor([1.0, 2.0], device="BARETILE").realize()
  assert fails_once(x).tolist() == [2, 3]
  holder, original = adapter.on_chip(x), adapter.on_chip(x).value
  with pytest.raises(RuntimeError, match="capture failed"): fails_once(x)
  assert adapter.recording is None and fails_once.replay is None and dict(adapter.RT.pinned) == pins
  if store: assert x.tolist() == [3, 4], "stores completed before the exception remain visible"
  else: assert holder.value is original and x.tolist() == [1, 2], "failed capture restores the caller's input binding"
  assert fails_once(x).tolist() == ([4, 5] if store else [2, 3])
  fails_once.reset()
  assert dict(adapter.RT.pinned) == pins


def test_tinygrad_jit_changed_state(chip):
  adapter_on(chip)
  from tinygrad import Tensor, TinyJit
  from tinygrad.helpers import Context
  weight, calls = Tensor([2.0], device="BARETILE").realize(), []
  @TinyJit
  def product(x):
    calls.append(1)
    return (x.to("BARETILE") * weight).realize()
  for i in range(1, 4): assert product(Tensor([float(i)], device="BARETILE")).item() == i * 2
  assert len(calls) == 2
  weight.assign(Tensor([10.0], device="BARETILE")).realize()
  assert product(Tensor([3.0], device="BARETILE")).item() == 30 and len(calls) == 2, "a replay reads the weight's new storage"
  assert product(Tensor([4.0], device="BARETILE")).item() == 40 and len(calls) == 2
  # Leaving the BARETILE device context must not dispatch to tinygrad's uninitialized native JIT state.
  assert product(Tensor([5.0], device="CPU")).item() == 50
  assert product(Tensor([6.0], device="BARETILE")).item() == 60
  state = Tensor.zeros(1, device="BARETILE").contiguous().realize()
  @TinyJit
  def accumulate(x):
    state.assign(state + x.to("BARETILE")).realize()
    return state
  with Context(DEV="BARETILE"):
    for n, device, want in [(1, "BARETILE", 1), (1, "BARETILE", 2), (10, "CPU", 12), (1, "BARETILE", 13), (1, "BARETILE", 14)]:
      assert accumulate(Tensor([float(n)], device=device)).item() == want
    @TinyJit
    def tick():
      state.assign(state + 1).realize()
      return state
    for want in (15, 16, 17): assert tick().item() == want

def test_tinygrad_jit_rebinds(chip):
  # A replay reads captured buffers where they are at that call. Two TinyJits that each update what the other reads,
  # and a learning rate assigned outside the TinyJit before each step, replay from the third call, as on tinygrad's CPU.
  rt = adapter_on(chip).RT
  from tinygrad import Tensor, TinyJit
  def run(device):
    a, b, w, lr = (Tensor([1.0], device=device).contiguous().realize() for _ in range(4))
    calls = {"f": 0, "g": 0, "h": 0}
    def jit(name, update):
      def body(x):
        calls[name] += 1
        update(x)
        return (a * 1 + b * 2 + w * 4).realize()
      return TinyJit(body)
    f, g = jit("f", lambda x: a.assign(a + b * x).realize()), jit("g", lambda x: b.assign(b + a * x).realize())
    h = jit("h", lambda x: w.assign(w - lr * x).realize())
    got = [step(Tensor([1.0], device=device)).item() for _ in range(8) for step in (f, g)]
    for i in range(6):
      lr.assign(Tensor([(i + 1) / 4], device=device)).realize()
      got.append(h(Tensor([float(i)], device=device)).item())
    return got, calls, [t.item() for t in (a, b, w)]
  launches, (got, calls, state) = rt.launches, run("BARETILE")
  want = run("CPU")
  assert rt.launches > launches and (got, calls, state) == want and calls == {"f": 2, "g": 2, "h": 2}, ((got, calls, state), want)
  # Passed the result of the call before, a replay would write its result where it reads its input, so it copies that
  # input first. Each worker moves three tiles or more in order, reading each where it wrote the one before.
  rows, calls = 32 * -(-len(rt.cores) // 2), []
  @TinyJit
  def shift(x):  # the columns, a tile to the right
    calls.append(1)
    return x.pad(((0, 0), (32, 0)))[:, :192].contiguous().realize()
  want = (np.arange(rows * 192) % 251).reshape(rows, 192).astype(np.float32)
  x = Tensor(want, device="BARETILE").realize()
  for n in range(4):
    x, want = shift(x), np.pad(want, ((0, 0), (32, 0)))[:, :192]
    assert np.array_equal(x.numpy(), want), n
  assert len(calls) == 2


def test_tinygrad_host_materialize(chip):
  adapter = adapter_on(chip)
  from tinygrad import Tensor, TinyJit
  calls, pins = [], dict(adapter.RT.pinned)
  @TinyJit
  def padded(x):
    calls.append(1)
    return (x.pad(((1, 0), (1, 0), (0, 0))) * 2).realize()
  for n in (1, 2, 3, 4):
    got = padded(Tensor(np.full((2, 3, 4), n, np.float32), device="BARETILE")).numpy()
    assert np.array_equal(got, np.pad(np.full((2, 3, 4), n * 2), ((1, 0), (1, 0), (0, 0))))
  assert len(calls) == 4, "host-dependent materialization cannot replay a frozen upload"
  assert dict(adapter.RT.pinned) == pins, "discarded recordings release their pins"


def test_tinygrad_fallback_dtypes(chip):
  adapter_on(chip)
  from tinygrad import Tensor, dtypes
  x = Tensor([1, 256, 65535], dtype=dtypes.uint16, device="BARETILE")
  assert (x + x).tolist() == [2, 512, 65534] and x.max().item() == 65535
  y = Tensor([1.0, 2.5, -3.0], dtype=dtypes.bfloat16, device="BARETILE")
  assert y.cast(dtypes.float64).tolist() == [1, 2.5, -3]
  b = Tensor([True, False, True], device="BARETILE")
  assert (b + b).cast(dtypes.float32).tolist() == [1, 0, 1]
  assert ((b + b) == b).tolist() == [True, True, True]
  assert ((b + b) & b).cast(dtypes.int32).tolist() == [1, 0, 1]


@pytest.mark.parametrize("size", [4, 64])
def test_tinygrad_jit_reset_host_state(chip, size):
  # State that a host upload replaced, small enough to upload at once or not, makes TinyJit record again.
  adapter_on(chip)
  from tinygrad import Tensor, TinyJit
  state = Tensor.zeros(size, device="BARETILE").contiguous().realize()
  x, calls = Tensor.ones(size, device="BARETILE").contiguous().realize(), []
  @TinyJit
  def step(x):
    calls.append(1)
    state.assign(state + x).realize()
    return state
  assert step(x).tolist() == [1.0] * size
  buf = state.uop.base.buffer
  buf.allocator._copyin(buf._buf, memoryview(bytes(size * 4)))  # reset through a host upload, also for small buffers
  for n in (1, 2, 3, 4): assert step(x).tolist() == [float(n)] * size
  assert len(calls) == 3 and step.replay, "record again after replacing the uploaded host state"

def test_tinygrad_jit_reads_cpu_state(chip):
  # A captured CPU tensor is read on every call, so TinyJit never replays a recording that froze its value.
  adapter_on(chip)
  from tinygrad import Tensor, TinyJit
  captured, calls = Tensor([0.0, 0.0], device="CPU").realize(), []
  @TinyJit
  def add(x):
    calls.append(1)
    return (x + captured.to("BARETILE")).realize()
  for n in range(4):
    captured.assign(Tensor([float(n), float(n + 1)], device="CPU")).realize()
    assert add(Tensor([2.0, 3.0], device="BARETILE")).tolist() == [n + 2, n + 4]
  assert len(calls) == 4 and add.replay is False

def test_tinygrad_jit_capture_setup_failure(chip):
  # A recording that fails before it starts leaves no recording behind, so later TinyJits still record.
  adapter = adapter_on(chip)
  from tinygrad import Tensor, TinyJit
  step = TinyJit(lambda x: (x + 1).realize())
  x = Tensor([2.0, 3.0], device="BARETILE").realize()
  assert step(x).tolist() == [3.0, 4.0]
  adapter.RT = Runtime(chip)  # x still belongs to the displaced runtime
  with pytest.raises(RuntimeError, match="closed|replaced"): step(x)
  assert adapter.recording is None
  fresh = TinyJit(lambda x: (x + 1).realize())
  for n in range(4): assert fresh(Tensor([float(n)], device="BARETILE")).tolist() == [n + 1]

def test_tinygrad_padded_constant_replays(chip):
  # A pad with a value runs on the chip and replays: the padded constant isn't taken for a uniform one.
  adapter_on(chip)
  from tinygrad import Tensor, TinyJit
  calls = []
  @TinyJit
  def padded(x):
    calls.append(1)
    return (x.pad(((1, 1), (0, 0)), value=-1.0) * 2).realize()
  for n in range(4):
    assert padded(Tensor([[float(n), float(n + 1)]], device="BARETILE")).tolist() == [[-2, -2], [n * 2, (n + 1) * 2], [-2, -2]]
  assert len(calls) == 2 and padded.replay

@pytest.mark.parametrize("op,expected", [("mul", [[16777217, 1], [5, -3]]), ("sum", [16777218, 2]), ("max", [16777217, 5])])
def test_tinygrad_integer_precision(chip, op, expected):
  # int32 products, sums and maxima above 2^24 are exact, as in tinygrad, since they run on the host.
  adapter_on(chip)
  from tinygrad import Tensor, dtypes
  values = [[16777217, 1], [5, -3]]
  x = Tensor(values, dtype=dtypes.int32, device="BARETILE").realize()
  one = Tensor.ones(2, 2, dtype=dtypes.int32, device="BARETILE").contiguous().realize()
  got = x * one if op == "mul" else x.sum(1) if op == "sum" else x.max(1)
  assert got.tolist() == expected

def test_tinygrad_closed_synchronize(monkeypatch):
  need_tinygrad()
  from baretile.adapters import tinygrad_device as adapter
  from tinygrad import Device
  class Closed:
    closed = True
    def wait(self, timeout): raise AssertionError("closed runtime must not be synchronized during interpreter shutdown")
  monkeypatch.setattr(adapter, "RT", Closed())
  Device["BARETILE"].synchronize()


def test_tinygrad(chip):
  # tinygrad builds the graph and baretile runs it, checked against tinygrad's CPU device. Each case must launch on the
  # chip, so we know the host didn't compute it.
  rt = adapter_on(chip).RT  # the BARETILE device uses this process's chip
  from tinygrad import Tensor, TinyJit, dtypes, function, nn
  from tinygrad.helpers import Context
  rng = np.random.default_rng(14)
  a, b, w = (rng.standard_normal(shape).astype(np.float32) for shape in ((2, 3, 40), (2, 3, 40), (40, 5)))
  cases = [lambda x, y, w: x * 2 + y, lambda x, y, w: x.sum(1), lambda x, y, w: x.max(-1), lambda x, y, w: x.exp().log(),
           lambda x, y, w: (x > 0).where(x, y), lambda x, y, w: x @ w, lambda x, y, w: x.softmax(-1), lambda x, y, w: x.permute(2, 0, 1) - y.mean(),
           lambda x, y, w: x.pad(((0, 0), (1, 2), (0, 0))) * 1.5, lambda x, y, w: x[:, 1:3, ::2].relu(),
           lambda x, y, w: function(lambda p, q: p * 2 + q)(x, y)]
  for n, f in enumerate(cases):
    want = f(*(Tensor(v, device="CPU") for v in (a, b, w))).numpy()
    launches, got = rt.launches, f(*(Tensor(v, device="BARETILE") for v in (a, b, w))).numpy()
    assert rt.launches > launches and got.shape == want.shape, n
    assert np.abs(got - want).max() <= 1e-3 * (np.abs(want).max() + 1), (n, np.abs(got - want).max())  # matmuls are good to the FPU's 2^-11
  # free an 8-byte buffer. The next one, this int64, must not see what it held
  Tensor.ones(2, device="BARETILE").contiguous().realize().tolist()
  assert Tensor.ones((), dtype=dtypes.int64, device="BARETILE").contiguous().item() == 1
  ints = Tensor([[1, -8, 1], [32, 1, 6]], dtype=dtypes.int32, device="BARETILE")  # products come back int32. bitwise and runs on the host
  assert (ints * ints).tolist() == [[1, 64, 1], [1024, 1, 36]] and (ints & 12).tolist() == [[0, 8, 0], [0, 0, 4]]
  assert ints.argmax(1).tolist() == [0, 0]
  made = Tensor([3, 3], dtype=dtypes.int64, device="BARETILE").cast(dtypes.bfloat16)  # the host makes this bf16 and uploads its values
  assert (made * Tensor([1.0, 2.0], dtype=dtypes.bfloat16, device="BARETILE")).tolist() == [3.0, 6.0]
  assert (Tensor(a, device="BARETILE") > 0).any(1).tolist() == (a > 0).any(1).tolist()  # bools read back a byte each
  assert (Tensor(np.arange(8, dtype=np.float32), device="BARETILE")[:4] * 2).tolist() == [0, 2, 4, 6]  # a view of part of a tensor
  assert (ints.T @ ints).tolist() == (np.array(ints.tolist()).T @ np.array(ints.tolist())).tolist()  # integer products and sums stay exact
  t = Tensor(a, device="BARETILE").realize()
  assert t[1, 1:3].numpy().tolist() == a[1, 1:3].tolist()  # reads a buffer through a tinygrad view of part of it (_offset)
  assert t[:, :0].sum(1).numpy().tolist() == [[0] * 40] * 2 and t[:, :0].numpy().shape == (2, 0, 40)  # nothing to compute
  assert (t[0, 0, :1].pad(((1, 0),))[0:1] * t[0, 0]).numpy().tolist() == [0] * 40  # padding stays padding, broadcast too
  t, want = Tensor.zeros(6, 6, device="BARETILE").contiguous().realize(), np.zeros((6, 6))  # stores into parts of a buffer
  t[2:4, 3:5] = Tensor.ones(2, 2, device="BARETILE"); t[0] = 7.0; t[1:6:2, 0] = Tensor([1.0, 2.0, 3.0], device="BARETILE")
  want[2:4, 3:5], want[0], want[1:6:2, 0] = 1, 7, [1, 2, 3]
  assert t.numpy().tolist() == want.tolist()
  def train(device):
    rng = np.random.default_rng(15)
    l1, l2 = nn.Linear(16, 32), nn.Linear(32, 10)
    for l in (l1, l2):
      l.weight = Tensor(rng.standard_normal(l.weight.shape).astype(np.float32) * 0.2, device=device)
      l.bias = Tensor.zeros(*l.bias.shape, device=device)
    opt, losses = nn.optim.Adam([l1.weight, l1.bias, l2.weight, l2.bias], lr=1e-2), []
    x, y = (Tensor(v, device=device) for v in (rng.standard_normal((8, 16)).astype(np.float32), rng.integers(0, 10, 8).astype(np.int32)))
    with Context(TRAINING=1):
      for _ in range(3):
        opt.zero_grad()
        losses.append(l2(l1(x).relu()).sparse_categorical_crossentropy(y).backward().item())
        opt.step()
    return losses
  want, got = train("CPU"), train("BARETILE")
  assert np.allclose(got, want, atol=1e-3) and got[-1] < got[0] - 0.3, (got, want)
  def jitted(device):  # the same under TinyJit, which replays from the third call on without running step
    rng, calls = np.random.default_rng(15), []
    l1 = nn.Linear(16, 10)
    l1.weight, l1.bias = Tensor(rng.standard_normal((10, 16)).astype(np.float32) * 0.2, device=device), Tensor.zeros(10, device=device).contiguous()
    opt = nn.optim.Adam([l1.weight, l1.bias], lr=1e-2)
    @TinyJit
    def step(x, y):
      calls.append(1)
      with Context(TRAINING=1):
        opt.zero_grad()
        loss = l1(x).sparse_categorical_crossentropy(y).backward()
        opt.step()
        return loss.realize()
    data = [(rng.standard_normal((8, 16)).astype(np.float32), rng.integers(0, 10, 8).astype(np.int32)) for _ in range(5)]
    return [step(Tensor(x, device=device).realize(), Tensor(y, device=device).realize()).item() for x, y in data], len(calls)
  (want, _), (got, calls) = jitted("CPU"), jitted("BARETILE")
  assert np.allclose(got, want, atol=1e-3) and calls == 2, (got, want, calls)

def test_torch_frontend(chip):
  # PyTorch runs on BARETILE through tinygrad's torch frontend (extra/torch_backend, PyTorch's "tiny" device), checked
  # against PyTorch on the CPU. The matmul and conv are good to the FPU's 2^-11.
  rt = adapter_on(chip).RT
  pytest.importorskip("extra.torch_backend", reason="needs a tinygrad checkout in TINYGRAD, since an installed tinygrad has no extra/")
  torch = pytest.importorskip("torch")
  from tinygrad.helpers import Context
  with Context(DEV="BARETILE"):  # the tinygrad device behind "tiny", read at each op
    import extra.torch_backend.backend  # noqa: F401 -- registers "tiny" with PyTorch. its C++ part is built once and cached
    g = torch.Generator().manual_seed(0)
    a, b, w = torch.randn(4, 40, generator=g), torch.randn(4, 40, generator=g), torch.randn(40, 24, generator=g)
    image, kernel, norm = torch.randn(1, 2, 10, 10, generator=g), torch.randn(3, 2, 3, 3, generator=g), torch.nn.LayerNorm(40)
    cases = [lambda x, y, w: (x * 2 + y).relu() - y.exp() / 4, lambda x, y, w: x.sum(1), lambda x, y, w: x @ w,
             lambda x, y, w: x.softmax(-1), lambda x, y, w: norm.to(x.device)(x),
             lambda x, y, w: torch.nn.functional.conv2d(image.to(x.device), kernel.to(x.device))]
    for n, f in enumerate(cases):
      want = f(a, b, w).detach()
      launches, got = rt.launches, f(*(t.to("tiny") for t in (a, b, w))).detach().cpu()
      assert rt.launches > launches and got.shape == want.shape, n
      assert (got - want).abs().max() <= 1e-3 * (want.abs().max() + 1), (n, (got - want).abs().max())
    def train(device):
      torch.manual_seed(1)
      model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 10)).to(device)
      opt, g = torch.optim.Adam(model.parameters(), lr=1e-2), torch.Generator().manual_seed(2)
      x, y = torch.randn(8, 16, generator=g).to(device), torch.randint(0, 10, (8,), generator=g).to(device)
      losses = []
      for _ in range(3):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
      return losses
    want, got = train("cpu"), train("tiny")
    assert np.allclose(got, want, atol=1e-4) and got[-1] < got[0] - 0.2, (got, want)

def test_tinygrad_llm(chip):
  # The Transformer in tinygrad/llm runs two SmolLM-shaped blocks on BARETILE, against our numpy Llama with the same
  # bf16 weights. tinygrad's KV cache is fp16, which the adapter doesn't have, so it stays on the host. The blocks are
  # precompiled tinygrad functions, and the adapter inlines them with fresh buffers on each call.
  need_tinygrad("tinygrad.llm.model")
  from tinygrad import Tensor as TT, dtypes, nn
  from tinygrad.helpers import Context
  from tinygrad.llm.model import Transformer, TransformerConfig
  rt = adapter_on(chip).RT
  c, w = smol(np.random.default_rng(37), 2)
  sd = {"token_embd.weight": w["embed"], "output.weight": w["embed"], "output_norm.weight": w["norm"][0]}
  for l in range(2):  # tinygrad's Linear is (out, in) like Hugging Face's, and keeps q, k, v, gate and up apart
    qkv, gate_up, p = w[f"layers.{l}.qkv_proj"].T, w[f"layers.{l}.gate_up_proj"].T, f"layers.{l}."
    sd |= {f"blk.{l}.{k}": v for k, v in (("attn_q.weight", qkv[:576]), ("attn_k.weight", qkv[576:768]), ("attn_v.weight", qkv[768:]),
           ("attn_output.weight", w[p + "o_proj"].T), ("ffn_gate.weight", gate_up[:1536]), ("ffn_up.weight", gate_up[1536:]),
           ("ffn_down.weight", w[p + "down_proj"].T), ("attn_norm.weight", w[p + "input_layernorm"][0]),
           ("ffn_norm.weight", w[p + "post_attention_layernorm"][0]))}
  with Context(DEV="BARETILE"):
    model = Transformer(TransformerConfig(num_blocks=2, dim=576, hidden_dim=1536, n_heads=9, n_kv_heads=3, norm_eps=c.eps, vocab_size=256,
                                          head_dim=64, rope_theta=c.theta, rope_dim=64, v_head_dim=64, max_context=64))
    nn.state.load_state_dict(model, {k: TT(np.ascontiguousarray(v, np.float32)).cast(dtypes.bfloat16) for k, v in sd.items()}, verbose=False)
    def logits(ids, start):  # Transformer.forward without its sampling
      x = model.token_embd(TT([ids], dtype=dtypes.int32)).float()
      for b in model.blk: x = b(x, start)
      return model.output(model.output_norm(x[:, -1:]))[0, -1].numpy()
    np_kv = llama.kv_cache(c, 64, np.zeros)
    for ids, places in (([7, 200, 31, 64, 5], range(5)), ([42], [5])):
      launches, got, want = rt.launches, logits(ids, places[0]), llama.forward(ids, places, w, c, np_kv)[0]
      assert rt.launches - launches > 50 and np.abs(got - want).max() < 0.02 * np.ptp(want), (np.abs(got - want).max(), np.ptp(want))

@pytest.mark.parametrize("dtype", ["bfp8", "bfp4"])
def test_bfp_numpy(chip, dtype):
  # gpt2.bfp, numpy's model of a block float, must match the chip's cast bit for bit, with exponents spread wide in
  # each run. The numpy models rely on it to round values the way the chip holds them.
  rt, rng = Runtime(chip), np.random.default_rng(5)
  a = bf16(rng.normal(0, 1, (64, 96)) * np.exp(rng.normal(0, 2, (64, 96))))
  assert np.array_equal(value(Tensor.of(rt, a.tolist()).cast(dtype))[0], gpt2.bfp(a, dtype))

@pytest.mark.parametrize("n, dtype, size", [(16, "bf16", 0), (40, "bf16", 0), (40, "bfp8", 0), (16, "bf16", 64)])
def test_gpt2_layer(chip, n, dtype, size):
  # A GPT-2 small block on n tokens, against the same code in numpy float64. 40 tokens are two tile rows, so attention
  # spans tiles. With size, keys and values go into a KV cache of that many positions, the empty ones masked. Biases
  # and norms get 0.1 of spread so they matter. The intermediates are bf16 here, so the test allows 3% error in the
  # block's change to x, which is up to about 2.
  rt, rng, d = Runtime(chip), np.random.default_rng(3), 768
  shapes = {"ln_1.weight": (1, d, 1), "ln_1.bias": (1, d, 0), "attn.c_attn.weight": (d, 3 * d, 0), "attn.c_attn.bias": (1, 3 * d, 0),
            "attn.c_proj.weight": (d, d, 0), "attn.c_proj.bias": (1, d, 0), "ln_2.weight": (1, d, 1), "ln_2.bias": (1, d, 0),
            "mlp.c_fc.weight": (d, 4 * d, 0), "mlp.c_fc.bias": (1, 4 * d, 0), "mlp.c_proj.weight": (4 * d, d, 0), "mlp.c_proj.bias": (1, d, 0)}
  p = {k: bf16(rng.normal(mean, 0.1 if r == 1 else 0.02, (r, c))) for k, (r, c, mean) in shapes.items()}
  x = bf16(rng.normal(0, 1, (n, d)))
  t = {k: Tensor.of(rt, v.tolist()) for k, v in p.items()}
  t |= {k: v.cast(dtype).realize() for k, v in t.items() if v.rows > 1}
  p |= {k: gpt2.values(v) for k, v in t.items() if v.dtype != "bf16"}
  cache = (lambda zeros: zeros((24, size, 64)) if size else None)  # a block's keys and values, or none
  places = Tensor.of(rt, [list(range(n))], "int32") if size else None
  got = gpt2.layer(Tensor.of(rt, x.tolist()), t, Tensor.of(rt, gpt2.mask(range(n), size or n).tolist()), 12,
                   cache(lambda shape: Tensor.of(rt, np.zeros(shape).tolist())), places)
  got = np.array(got.tolist())
  want = gpt2.layer(x, p, gpt2.mask(range(n), size or n), 12, cache(np.zeros), range(n))
  change = np.abs(want - x).max()
  assert change > 1 and np.abs(got - want).max() < 0.03 * change, (np.abs(got - want).max(), change)

@pytest.mark.skipif("GPT2" not in os.environ, reason="set GPT2 to model.safetensors from openai-community/gpt2")
def test_gpt2_model(chip):
  # GPT-2 small with its pretrained weights, on a prompt and 3 decoded tokens, against numpy with the same bf16
  # weights. In fp32, numpy matches Hugging Face to 1e-4. The chip's pick has to be a token that numpy rates as good,
  # not necessarily numpy's. numpy picks " not" (407) with " to" 0.36 behind, which bf16 can't tell apart through 12
  # blocks. The residual reaches 3000, where bf16 steps by 16. It takes about 90 s on ttsim.
  rt = Runtime(chip)
  weights = gpt2.load(os.environ["GPT2"])
  t, w, ids = gpt2.upload(rt, weights), gpt2.arrays(weights), [464, 3616, 286, 1204, 318]  # tiktoken's gpt2 encoding
  kv = gpt2.kv_cache(12, 32, lambda shape: Tensor.of(rt, np.zeros(shape).tolist()))
  run, on = lambda ids, places: gpt2.forward(ids, t, gpt2.mask(places, 32), places, kv), lambda row: Tensor.of(rt, [row], "int32")
  step = tensor.jit(run)
  for i in range(4):
    new = ids if i == 0 else ids[-1:]
    got = gpt2.values((run if i == 0 else step)(on(new), on(list(range(len(ids) - len(new), len(ids))))))[0]
    want = gpt2.logits(ids, w)
    err = np.abs(got - want).max()
    assert (i or want.argmax() == 407) and err < 2 and want[got.argmax()] >= want.max() - 2 * err, (i, err, got.argmax())
    ids.append(int(want.argmax()))

def smol(rng, layers, vocab=256, places=64):  # SmolLM-135M's config, and random weights in its shapes with llama.py's names
  c = llama.model(dim=576, heads=9, kv=3, layers=layers, eps=1e-5, theta=10000.0)
  shapes = {"input_layernorm": (1, 576, 1), "post_attention_layernorm": (1, 576, 1), "q_proj": (576, 576, 0), "k_proj": (576, 192, 0),
            "v_proj": (576, 192, 0), "o_proj": (576, 576, 0), "gate_proj": (576, 1536, 0), "up_proj": (576, 1536, 0), "down_proj": (1536, 576, 0)}
  w = {f"layers.{l}.{k}": bf16(rng.normal(mean, 0.1 if r == 1 else 0.02, (r, n))) for l in range(layers) for k, (r, n, mean) in shapes.items()}
  w |= {"norm": bf16(rng.normal(1, 0.1, (1, 576))), "embed": bf16(rng.normal(0, 1, (vocab, 576)))}
  return c, llama.fuse(w | {"embed.T": w["embed"].T.copy()} | dict(zip(("cos", "sin", "turn"), llama.rotary(c, places))), c)

def near(got, want):  # each row picks numpy's best, or a token within 1% of the span, which bf16 activations can't tell apart
  return all(w.max() - w[g.argmax()] <= 0.01 * np.ptp(w) for g, w in zip(got, want))

def qwen35(rng, vocab=256, places=64):
  """A small random Qwen3.5-MoE and its config, with bf16 Weights named and fused the way llama.load gives them.

  It has 3 Gated DeltaNet blocks, a gated attention block, and the multi-token prediction layer as block 4. Tokens use
  all 8 experts, so a near tie between experts can't decide a test. The embedding is small next to what the
  blocks add, so losing the Gated DeltaNet states moves the logits by 100%."""
  c = llama.model(dim=128, heads=4, kv=2, head_dim=64, layers=4, eps=1e-6, theta=1e7, partial=0.25, gated=True, centered=True, conv=4,
                  kh=2, vh=4, kd=32, vd=32, k=8, norm_topk=True, layer_types=["linear_attention"] * 3 + ["full_attention"], mtp=1)
  bits = lambda *shape, mean=0.0, sd=0.05: llama.Weight(lambda a=gpt2.bf16_bits(rng.normal(mean, sd, shape)): a)
  # embed.T is a separate output layer, untied from embed
  w = {"embed": bits(vocab, 128, sd=0.3), "embed.T": bits(128, vocab), "norm": bits(1, 128, sd=0.3), "mtp.fc": bits(256, 128),
       "mtp.pre_fc_norm_embedding": bits(1, 128, sd=0.3), "mtp.pre_fc_norm_hidden": bits(1, 128, sd=0.3), "mtp.norm": bits(1, 128, sd=0.3)}
  for l in range(c.layers + c.mtp):  # block 4, the multi-token prediction layer, is an attention block
    pre = f"layers.{l}."
    w |= {pre + "input_layernorm": bits(1, 128, sd=0.3), pre + "post_attention_layernorm": bits(1, 128, sd=0.3), pre + "gate": bits(8, 128),
          pre + "experts.gate_up_proj": bits(8, 128, 128), pre + "experts.down_proj": bits(8, 128, 64),
          pre + "shared_expert.gate_proj": bits(128, 64), pre + "shared_expert.up_proj": bits(128, 64),
          pre + "shared_expert.down_proj": bits(64, 128), pre + "shared_expert_gate": bits(1, 128)}
    if l < 3: w |= {pre + "in_proj_qkv": bits(128, 256), pre + "in_proj_z": bits(128, 128), pre + "in_proj_b": bits(128, 4),
                    pre + "in_proj_a": bits(128, 4), pre + "conv1d": bits(256, 1, 4, sd=0.3), pre + "A_log": bits(1, 4, sd=0.5),
                    pre + "dt_bias": bits(1, 4, mean=1.0, sd=0.2), pre + "norm": bits(1, 32, mean=1.0, sd=0.2), pre + "out_proj": bits(128, 128)}
    else: w |= {pre + "q_proj": bits(128, 512), pre + "k_proj": bits(128, 128), pre + "v_proj": bits(128, 128), pre + "o_proj": bits(256, 128),
                pre + "q_norm": bits(1, 64, sd=0.3), pre + "k_norm": bits(1, 64, sd=0.3)}
  w |= {k: llama.Weight(lambda a=a: gpt2.bf16_bits(a)) for k, a in zip(("cos", "sin", "turn"), llama.rotary(c, places))}
  return c, llama.fuse(w, c)

def test_llama_frequencies():
  # Llama 3.1's RoPE scaling slows long wavelengths by 8, keeps short ones, and blends those between. Our frequencies
  # must match Hugging Face's float32 ones within a relative 1e-6. PyTorch's float32 powers differ from NumPy's by up to
  # two float32 steps on some of CI's runners, and between the scaled and unscaled wavelengths, the blend moves by up
  # to four steps for each step its input moves. A mistake in the scaling would be off by far more.
  m, transformers = pytest.importorskip("transformers.modeling_rope_utils"), pytest.importorskip("transformers")
  rope = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
  hf = transformers.LlamaConfig(hidden_size=4096, num_attention_heads=32, rope_theta=500000.0, rope_scaling=rope, max_position_embeddings=131072)
  want = m.ROPE_INIT_FUNCTIONS["llama3"](hf, "cpu")[0].numpy()
  got = llama.frequencies(llama.model(dim=4096, heads=32, theta=500000.0, rope=rope))
  assert got.dtype == np.float32 and np.allclose(got, want, rtol=1e-6, atol=0), np.abs(got / want - 1).max()
  plain = llama.frequencies(llama.model(dim=4096, heads=32, theta=500000.0))
  assert plain[0] == got[0] and plain[-1] == 8 * got[-1], "short wavelengths kept, long ones slowed by the factor"

def safetensors(path, tensors):  # writes {name: (dtype "BF16" or "F32", array of bits or values)} as a safetensors file
  header, data = {}, b""
  for name, (dtype, a) in tensors.items():
    header[name] = {"dtype": dtype, "shape": list(a.shape), "data_offsets": [len(data), len(data) + a.nbytes]}
    data += a.tobytes()
  head = json.dumps(header).encode()
  pathlib.Path(path).write_bytes(struct.pack("<Q", len(head)) + head + data)

def test_llama_load(tmp_path):
  # A checkpoint split like Llama-3.1-8B's, in two files and an index, with its own lm_head and a float32 norm. The
  # RoPE tables cover the 64 positions asked for, not the model's 131072. A family baretile doesn't run is refused by
  # name, though its weights have the same names.
  rng, dim, ffn, vocab = np.random.default_rng(31), 64, 96, 40
  w = lambda r, c: (rng.standard_normal((r, c)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
  parts = {"model.embed_tokens.weight": w(vocab, dim), "lm_head.weight": w(vocab, dim), "model.norm.weight": w(1, dim)[0],
           **{f"model.layers.0.self_attn.{n}_proj.weight": w(r, dim) for n, r in (("q", dim), ("k", 32), ("v", 32), ("o", dim))},
           **{f"model.layers.0.mlp.{n}_proj.weight": w(r, cl) for n, r, cl in (("gate", ffn, dim), ("up", ffn, dim), ("down", dim, ffn))}}
  norms = {f"model.layers.0.{n}.weight": rng.standard_normal(dim).astype(np.float32) for n in ("input_layernorm", "post_attention_layernorm")}
  files = [{k: ("BF16", v) for k, v in list(parts.items())[:5]} | {k: ("F32", v) for k, v in norms.items()},
           {k: ("BF16", v) for k, v in list(parts.items())[5:]}]
  for i, file in enumerate(files): safetensors(tmp_path / f"model-0000{i + 1}-of-00002.safetensors", file)
  weight_map = {k: f"model-0000{i + 1}-of-00002.safetensors" for i, file in enumerate(files) for k in file}
  (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
  rope = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
  (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama", "hidden_size": dim, "num_attention_heads": 2, "num_key_value_heads": 1,
                                                    "num_hidden_layers": 1, "rms_norm_eps": 1e-5, "rope_theta": 500000.0,
                                                    "max_position_embeddings": 131072, "rope_scaling": rope, "vocab_size": vocab,
                                                    "intermediate_size": ffn}))
  c, weights = llama.load(str(tmp_path), 64)
  assert all(isinstance(x, llama.Weight) for x in weights.values())
  assert c.places == 64 and c.rope == rope and c.ffn == ffn and weights["cos"].bits().shape == (64, 4 * 32)
  q, k, v, gate, up = (parts[f"model.layers.0.{n}_proj.weight"] for n in ("self_attn.q", "self_attn.k", "self_attn.v", "mlp.gate", "mlp.up"))
  assert np.array_equal(weights["layers.0.qkv_proj"].bits(), np.concatenate([q.T, k.T, v.T], 1))
  assert np.array_equal(weights["layers.0.gate_up_proj"].bits(), np.concatenate([gate.T, up.T], 1))
  assert np.array_equal(weights["embed.T"].bits(), parts["lm_head.weight"].T), "its own output layer, not the embedding"
  assert np.array_equal(weights["layers.0.input_layernorm"].bits()[0], gpt2.bf16_bits(norms["model.layers.0.input_layernorm.weight"]))
  out = llama.forward([3, 7], [0, 1], llama.arrays(weights), c, llama.kv_cache(c, 32, np.zeros))
  assert out.shape == (1, vocab) and np.isfinite(out).all()
  (tmp_path / "config.json").write_text(json.dumps(json.loads((tmp_path / "config.json").read_text()) | {"model_type": "gemma2"}))
  with pytest.raises(ValueError, match="'gemma2' isn't supported"): llama.load(str(tmp_path))

def test_bpe_merge_order(tmp_path):
  # Vocabulary IDs need not order BPE merges. The whole piece only bypasses merges with ignore_merges; an explicit
  # empty list means no merges. Test both tokenizer.json encodings of a pair and the tie's leftmost occurrence.
  vocab = {"a": 0, "b": 1, "c": 2, "ab": 3, "bc": 4, "abc": 5}
  for pairs, want in (([("b", "c"), ("a", "b"), ("ab", "c")], [0, 4]),
                      ([("a", "b"), ("b", "c"), ("ab", "c")], [5]), ([], [0, 1, 2])):
    for strings in (False, True):
      for ignore in (False, True):
        model = dict(type="BPE", vocab=vocab, merges=[" ".join(p) if strings else p for p in pairs], ignore_merges=ignore)
        (tmp_path / "tokenizer.json").write_text(json.dumps({"model": model}))
        tok = llama.Tokenizer(str(tmp_path))
        assert tok.encode("abc") == ([5] if ignore else want)
        assert tok.encode("abcabc") == want * 2
        assert tok.decode(tok.encode("abcabc")) == "abcabc"
  # A piece of 21,000 letters merges in O(n log n): rescanning every pair after each merge took 20 s. The cache of
  # pieces stays bounded, since a server sees new words forever.
  (tmp_path / "tokenizer.json").write_text(json.dumps({"model": dict(type="BPE", vocab=vocab, merges=[["a", "b"], ["b", "c"], ["ab", "c"]])}))
  tok, start = llama.Tokenizer(str(tmp_path)), time.monotonic()
  assert tok.encode("abc" * 7000) == [5] * 7000 and time.monotonic() - start < 5
  tok.encode(" ".join("".join("abc"[i // 3**k % 3] for k in range(9)) for i in range(12000)))  # 12,000 words
  assert len(tok.pieces) == 10000
  for settings in (dict(type="Unigram"), dict(dropout=0.1), dict(continuing_subword_prefix="##"),
                   dict(merges=["a c"]), dict(merges=[None]), dict(merges=None)):
    (tmp_path / "tokenizer.json").write_text(json.dumps({"model": dict(type="BPE", vocab=vocab) | settings}))
    with pytest.raises(ValueError, match="Tokenizer"): llama.Tokenizer(str(tmp_path))

def test_bpe_stages_and_missing_bytes(tmp_path):
  # SmolLM splits digits before ByteLevel. Whitespace before a digit therefore reaches BPE as one piece. BPE with
  # unk_token=None drops absent alphabet bytes before merging, so a missing byte can join two known letters.
  vocab = {"a": 0, "b": 1, "ab": 2, "Ġ": 3, "ĠĠ": 4, "Ċ": 5, "ĉ": 6, "Ċĉ": 7, "2": 8, "1": 9, "12": 10}
  model = dict(type="BPE", vocab=vocab, merges=[["a", "b"], ["Ġ", "Ġ"], ["Ċ", "ĉ"], ["1", "2"]], unk_token=None)
  byte = dict(type="ByteLevel", add_prefix_space=False, use_regex=True)
  def tokenizer(pre, extra=None):
    (tmp_path / "tokenizer.json").write_text(json.dumps(dict(model=model | (extra or {}), pre_tokenizer=pre)))
    return llama.Tokenizer(str(tmp_path))
  for each in (False, True):
    tok = tokenizer(dict(type="Sequence", pretokenizers=[dict(type="Digits", individual_digits=each), byte]))
    assert tok.encode("  2") == [4, 8] and tok.encode("\n\t2") == [7, 8]
    assert tok.encode("12") == ([9, 8] if each else [10])
  tok = tokenizer(byte | dict(use_regex=False))
  assert tok.encode("a?b") == tok.encode("a\x1db") == [2] and tok.encode("?") == []
  with pytest.raises(ValueError, match="unknown-token"): tokenizer(byte, dict(unk_token="<unk>"))
  with pytest.raises(ValueError, match="pre-tokenizer"): tokenizer(byte | dict(add_prefix_space=True))

def test_generation_defaults():
  # A nonempty generation config without EOS inherits the selected text model's token, even in a multimodal config.
  model = dict(eos_token_id=1, temperature=0.8, text_config=dict(eos_token_id=2, top_k=30))
  assert serve.generation_defaults(model, {"temperature": 0.5}) == ([2], dict(temperature=0.5, top_k=30))
  assert serve.generation_defaults(model, {"eos_token_id": [3, 4]})[0] == [3, 4]
  assert serve.generation_defaults(model, {"eos_token_id": None})[0] == []
  assert serve.generation_defaults({"eos_token_id": 1, "text_config": {"top_p": 0.9}}, {"temperature": 0.5}) == \
    ([1], dict(top_p=0.9, temperature=0.5))
  assert serve.generation_defaults({}, {}) == ([], {})

def test_tokenizer(tmp_path):
  # llama.Tokenizer must encode like tiktoken over the same tokenizer.json, and decode back, with the pre-tokenizers of
  # Llama 3, SmolLM, Qwen (NFC first) and gpt-oss. It spells out \p{L}, \p{N} and \s for Python's re, whose \s also
  # matches four control characters. "\x1cthe" catches that. A piece that's in the vocabulary is taken as is, like the
  # lone emoji, whose bytes no merge joins.
  tiktoken = pytest.importorskip("tiktoken")
  bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))  # GPT-2's bytes as printable characters
  char = dict(zip(bs + [b for b in range(256) if b not in bs], map(chr, bs + [256 + i for i in range(256 - len(bs))])))
  tokens = [bytes([b]) for b in range(256)] + [b"th", b"he", b"the", b" the", b"in", b"ing", b"12", b"123", b"\r\n", b"\x1cthe",
                                               "é".encode(), "😀".encode()]
  vocab = {"".join(char[b] for b in t): i for i, t in enumerate(tokens)}  # ids in merge order
  text = ("The thing's THE THING'LL 1234567 x²³ ½ ٣٤٥ in\u00a0the\u2003end \t\tthing\r\n\r\nthe \x1c\x1dctrl \x1cthe café "
          "😀😀 x😀y e\u0301 McDonald's ǅungla ʰa/\n")
  llama3 = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
  o200k = (r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?"
           r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
           r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+")
  for split, nfc in ((llama3, False), (None, False), (llama3, True), (o200k, False)):
    pre = {"type": "Split", "pattern": {"Regex": split or llama.SPLIT}}
    (tmp_path / "tokenizer.json").write_text(json.dumps({"model": {"vocab": vocab}, "added_tokens": [{"content": "<|end|>", "id": len(vocab)}],
                                                         "pre_tokenizer": pre, "normalizer": {"type": "NFC"} if nfc else None}))
    ours = llama.Tokenizer(str(tmp_path))
    theirs = tiktoken.Encoding("t", pat_str=split or llama.SPLIT, mergeable_ranks=dict(zip(tokens, range(len(tokens)))),
                               special_tokens={"<|end|>": len(vocab)})
    plain = unicodedata.normalize("NFC", text) if nfc else text
    assert ours.encode(text) == theirs.encode(plain), (split, nfc)
    assert ours.decode(ours.encode(text) + [len(vocab)]) == plain + "<|end|>"
    assert ours.encode(text + "<|end|>", special=True) == theirs.encode(plain, allowed_special="all") + [len(vocab)]
  data = json.loads((tmp_path / "tokenizer.json").read_text())
  (tmp_path / "tokenizer.json").write_text(json.dumps(data | {"added_tokens": []}))
  ours = llama.Tokenizer(str(tmp_path))
  assert ours.encode("the", special=True) == ours.encode("the") == [258], "no special tokens must leave BPE merges intact"

@pytest.mark.parametrize("n", [16, 40])
def test_llama_layer(chip, n):
  # A Llama block in SmolLM-135M's shapes on n tokens, against the same code in numpy float64. The intermediates are
  # bf16 here, so the test allows 3% error in the block's change to x.
  rt, rng = Runtime(chip), np.random.default_rng(12)
  c, w = smol(rng, 1)
  p = {k.removeprefix("layers.0."): v for k, v in w.items() if k.startswith("layers.0.")}
  x, (cos, sin, turn) = bf16(rng.normal(0, 1, (n, 576))), llama.rotary(c, n)
  mask = gpt2.mask(range(n), n)
  on = lambda a: Tensor.of(rt, a.tolist())
  cache, np_cache = on(np.zeros((6, n, 64))), np.zeros((6, n, 64))
  places = Tensor.of(rt, [list(range(n))], "int32")
  got = llama.layer(on(x), {k: on(v) for k, v in p.items()}, c, (on(cos), on(sin), on(turn)), (on(mask), places.reshape(1, n, 1), 0), cache, places)
  want = llama.layer(x, p, c, (np.float64(cos), np.float64(sin), turn), np.float64(mask), np_cache, range(n))
  change = np.abs(want - x).max()
  assert change > 0.5 and np.abs(np.array(got.tolist()) - want).max() < 0.03 * change, (np.abs(np.array(got.tolist()) - want).max(), change)

def test_kv_cache(chip):
  # A two-block Llama decodes 5 tokens after a 5-token prompt, against numpy. Positions are data, not shapes, so the
  # steps launch the same kernels and replay from the third.
  rt, rng = Runtime(chip), np.random.default_rng(13)
  c, w = smol(rng, 2)
  t = {k: Tensor.of(rt, v.tolist()) for k, v in w.items()}
  kv, np_kv = llama.kv_cache(c, 32, lambda shape: Tensor.of(rt, np.zeros(shape).tolist())), llama.kv_cache(c, 32, np.zeros)
  on = lambda row: Tensor.of(rt, [row], "int32")
  step, ids = tensor.jit(lambda ids, places: llama.forward(ids, places, t, c, kv)), [7, 200, 31, 64, 5]
  for i in range(6):
    new = ids if i == 0 else ids[-1:]
    places = list(range(len(ids) - len(new), len(ids)))
    got = llama.logits(llama.forward(on(new), on(places), t, c, kv) if i == 0 else step(on(new), on(places)))
    want = llama.forward(new, places, w, c, np_kv)[0]
    assert np.abs(got - want).max() < 0.03 * np.abs(want).max(), (i, np.abs(got - want).max(), np.abs(want).max())
    ids.append(int(want.argmax()))
  assert np.allclose(want, llama.forward(ids[:-1], range(len(ids) - 1), w, c, llama.kv_cache(c, 32, np.zeros))[0], atol=1e-3)

def test_moe(chip):
  # A mixture of experts (qwen.moe) against numpy, with the experts picked and their weights read on the chip. Up to
  # 40 tokens use qwen.touched, and 160 use qwen.group, which must replay bit for bit. A free slot (live 0) gets output
  # 0. A two-block model must decode the same from a replay as op by op, so a replayed step reads the experts its
  # token chose.
  rt, rng = Runtime(chip), np.random.default_rng(31)
  experts = lambda dim, n, m: {"router": bf16(rng.normal(0, 0.5, (dim, n))), "experts.gate_up": bf16(rng.normal(0, 0.1, (n, dim, 2 * m))),
                               "experts.down": bf16(rng.normal(0, 0.1, (n, m, dim)))}
  c, w = llama.model(dim=64, k=4), experts(64, 16, 64)
  t = {k: Tensor.of(rt, v.tolist()) for k, v in w.items()}
  for n in (1, 3, 8, 40, 160):  # row 1, if there is one, is a free slot
    h, live = bf16(rng.normal(0, 1, (n, 64))), np.float32(np.arange(n) != 1).reshape(n, 1)
    got, want = value(qwen.moe(Tensor.of(rt, h.tolist()), t, c, live=Tensor.of(rt, live.tolist())))[0], qwen.moe(h, w, c, live=live)
    assert np.abs(got - want).max() < 0.02 * np.abs(want).max() and not got[1:2].any(), (n, np.abs(got - want).max())
  ids, weights = Tensor.of(rt, [[0, 3], [5, 3], [7, 1]], "int32"), Tensor.of(rt, [[0.5, 0.5], [0, 0], [0.25, 0.75]])
  assert value(qwen.touched(ids, 8, weights)[0])[0, :, 0].tolist() == [0, 1, 3, 7, 8, 8], "expert 5, weighted 0, not listed"
  grouped, prompts = tensor.jit(lambda h: qwen.moe(h, t, c)), np.random.default_rng(41)  # each replayed call does its own grouping
  for _ in range(3):
    h = Tensor.of(rt, bf16(prompts.normal(0, 1, (160, 64))).tolist())
    assert np.array_equal(value(grouped(h)), value(qwen.moe(h, t, c))), "replayed expert grouping must match a fresh call"
  c, w = smol(rng, 2)
  c.k = 2
  for l in range(2):
    del w[f"layers.{l}.gate_up_proj"], w[f"layers.{l}.down_proj"]
    w |= {f"layers.{l}.{k}": v for k, v in experts(576, 8, 64).items()}
  t, on = {k: Tensor.of(rt, v.tolist()) for k, v in w.items()}, lambda row: Tensor.of(rt, [row], "int32")
  def decode(replay):  # logits for a prompt and the 4 tokens numpy picks after it, and numpy's logits
    kv, np_kv = llama.kv_cache(c, 32, lambda shape: Tensor.of(rt, np.zeros(shape).tolist())), llama.kv_cache(c, 32, np.zeros)
    step, ids, out = tensor.jit(lambda ids, places: llama.forward(ids, places, t, c, kv)), [1, 2, 3, 4, 5], []
    for i in range(5):
      new = ids if i == 0 else ids[-1:]
      places = list(range(len(ids) - len(new), len(ids)))
      out.append((llama.logits(llama.forward(on(new), on(places), t, c, kv) if i == 0 or not replay else step(on(new), on(places))),
                  want := llama.forward(new, places, w, c, np_kv)[0]))
      ids.append(int(want.argmax()))
    return out
  # In this prompt, each token's k-th and next router logits are 6% of their span apart, so bf16 rounding leaves the
  # chip picking numpy's experts. The experts' outputs are large and of either sign, so the sums lose more bits than in
  # a dense MLP, and the tolerance is 6%.
  replayed, alone = decode(True), decode(False)
  assert all(np.array_equal(a[0], b[0]) for a, b in zip(replayed, alone)), "replayed decode must match a fresh call"
  assert all(np.abs(got - want).max() < 0.06 * np.abs(want).max() for got, want in alone), [np.abs(g - w).max() / np.abs(w).max() for g, w in alone]
  held = []  # topk and group free the DRAM they take, so the amount held stays the same across calls
  for _ in range(3):
    qwen.group(Tensor.of(rt, rng.normal(0, 1, (40, 16)).tolist()).topk(4)[1], 16)
    rt.wait(), rt.reclaim()
    held.append(sum(rt.used.values()))
  assert held[0] == held[1] == held[2], held

def test_qwen3_moe(chip, tmp_path):
  # Qwen3-MoE from Hugging Face's files, against transformers in float32 with random bf16 weights. The experts are
  # weighted by their probabilities among all experts, since norm_topk_prob is off by default. numpy must be within
  # 0.5% (0.12% measured), and the chip within 2% (1%) with the same picks. At TT's performance precision the chip must
  # be within 1% of the span of numpy with the same block floats (0.8%).
  torch, transformers = pytest.importorskip("torch"), pytest.importorskip("transformers")
  torch.manual_seed(0)
  config = transformers.Qwen3MoeConfig(hidden_size=64, intermediate_size=128, moe_intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                                       num_key_value_heads=2, head_dim=32, num_experts=8, num_experts_per_tok=2, decoder_sparse_step=2,
                                       vocab_size=256, max_position_embeddings=64, rope_theta=1e6, tie_word_embeddings=False)
  model = transformers.Qwen3MoeForCausalLM(config).to(torch.bfloat16).float()
  model.save_pretrained(tmp_path)
  ids = [3, 1, 4, 1, 5, 9, 2, 6]
  with torch.no_grad(): want = model(torch.tensor([ids])).logits[0].numpy()
  (tmp_path / "config.json").write_text(json.dumps(json.loads((tmp_path / "config.json").read_text()) | {"num_hidden_layers": 1}))
  assert not any(k.startswith("layers.1.") for k in llama.load(str(tmp_path), 32)[1]), "a config cut to one block loads only that block"
  (tmp_path / "config.json").write_text(json.dumps(json.loads((tmp_path / "config.json").read_text()) | {"num_hidden_layers": 2}))
  c, weights = llama.load(str(tmp_path), 32)
  names = {k.split(".", 2)[2] for k in weights if k.startswith("layers.1.")}
  assert (c.k, llama.head(c)) == (2, 32) and {"router", "experts.gate_up", "experts.down", "qk_norm", "qk_on"} <= names
  rel = lambda got: max(np.abs(got[r] - want[r]).max() / np.abs(want[r]).max() for r in range(len(ids)))
  assert rel(llama.forward(ids, list(range(8)), llama.arrays(weights), c, llama.kv_cache(c, 32, np.zeros), every=True)) < 0.005
  rt = Runtime(chip)
  on = lambda row: Tensor.of(rt, [row], "int32")
  got = gpt2.values(llama.forward(on(ids), on(list(range(8))), llama.upload(rt, weights), c, llama.kv_cache(c, 32, llama.zeros(rt)), every=True))
  assert rel(got) < 0.02 and np.array_equal(got.argmax(-1), want.argmax(-1)), rel(got)
  c.precision = llama.precision(c, "tt")  # the experts' gate and up in bfp4 at one phase, their down in bfp8 at two
  t = llama.upload(rt, weights, c.precision)
  assert (t["layers.1.experts.gate_up"].dtype, t["layers.1.experts.down"].dtype, t["layers.1.router"].dtype) == ("bfp4", "bfp8", "bf16")
  got = gpt2.values(llama.forward(on(ids), on(list(range(8))), t, c, llama.kv_cache(c, 32, llama.zeros(rt, "bfp8")), every=True))
  want = llama.forward(ids, list(range(8)), llama.arrays(weights, c.precision), c, llama.kv_cache(c, 32, np.zeros), every=True)
  assert np.array_equal(got.argmax(-1), want.argmax(-1)) and np.abs(got - want).max() < 0.01 * np.ptp(want), np.abs(got - want).max() / np.ptp(want)

FAMILIES = {"llama": ("Llama", {}), "qwen3": ("Qwen3", dict(head_dim=64)), "qwen2": ("Qwen2", {}), "mistral": ("Mistral", dict(sliding_window=None)),
            "qwen2_moe": ("Qwen2Moe", dict(moe_intermediate_size=64, shared_expert_intermediate_size=96, num_experts=8, num_experts_per_tok=2)),
            "mixtral": ("Mixtral", dict(num_local_experts=8, num_experts_per_tok=2)),
            "gpt_oss": ("GptOss", dict(head_dim=32, num_local_experts=8, num_experts_per_tok=2, sliding_window=4,
                                       layer_types=["sliding_attention", "full_attention"],
                                       rope_parameters={"rope_type": "yarn", "factor": 8.0, "beta_fast": 32.0, "beta_slow": 1.0, "truncate": False,
                                                        "original_max_position_embeddings": 8, "rope_theta": 1e4}))}

@pytest.mark.parametrize("family", FAMILIES)
def test_hf_families(chip, tmp_path, family):
  # Each family as Hugging Face saves it, against transformers in float32 with random bf16 weights. For gpt-oss,
  # leaving out any one of the sinks, sliding window, YaRN, biases or clamped GLU takes numpy past 0.5%. The chip may
  # pick a token within 0.1% of the span of the best, since Qwen2-MoE has such a tie.
  torch, transformers = pytest.importorskip("torch"), pytest.importorskip("transformers")
  name, extra = FAMILIES[family]
  torch.manual_seed(0)
  config = getattr(transformers, name + "Config")(hidden_size=128, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                                                  num_key_value_heads=2, vocab_size=256, max_position_embeddings=64, rope_theta=1e4,
                                                  tie_word_embeddings=False, **extra)
  model = getattr(transformers, name + "ForCausalLM")(config)
  for name, p in model.named_parameters():  # biases start at 0 and norms at 1, so randomize them to make them matter
    if p.ndim == 1 or "bias" in name: p.data.normal_(1, 0.2) if torch.all(p == 1) else p.data.normal_(0, 0.5)
  model = model.to(torch.bfloat16).float()
  model.save_pretrained(tmp_path)
  ids = [3, 1, 4, 1, 5, 9, 2, 6]
  with torch.no_grad(): want = model(torch.tensor([ids])).logits[0].numpy()
  c, weights = llama.load(str(tmp_path), 32)
  rel = lambda got: max(np.abs(got[r] - want[r]).max() / np.abs(want[r]).max() for r in range(len(ids)))
  numpy = rel(llama.forward(ids, list(range(8)), llama.arrays(weights), c, llama.kv_cache(c, 32, np.zeros), every=True))
  assert numpy < 0.005, numpy
  rt = Runtime(chip)
  on = lambda row: Tensor.of(rt, [row], "int32")
  got = gpt2.values(llama.forward(on(ids), on(list(range(8))), llama.upload(rt, weights), c, llama.kv_cache(c, 32, llama.zeros(rt)), every=True))
  ties = [want[r].max() - want[r][got[r].argmax()] <= 0.001 * np.ptp(want[r]) for r in range(len(ids))]  # its best, or one bf16 can't tell from it
  assert rel(got) < 0.02 and all(ties), (rel(got), ties)

def test_autonomous_generation(chip, tmp_path, monkeypatch):
  # test/generation.py feeds the chip its own picks and holds each step to the first workload's gates against
  # transformers: on the logits in bf16, and on the five best at tt's precision. A small random Llama passes both. With a
  # gate of 0 it fails, and the report keeps the failing step's logits.
  torch, transformers = pytest.importorskip("torch"), pytest.importorskip("transformers")
  import generation
  torch.manual_seed(0)
  config = transformers.LlamaConfig(hidden_size=128, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                    vocab_size=256, max_position_embeddings=128, rope_theta=1e4, tie_word_embeddings=False)
  transformers.LlamaForCausalLM(config).to(torch.bfloat16).float().save_pretrained(tmp_path)
  bytes_tokenizer(tmp_path)
  rt = Runtime(chip)
  report = generation.check(rt, str(tmp_path), steps=4)
  assert report["passed"] and [len(p["tokens"]) for p in report["prompts"]] == [4, 4, 4], report
  assert generation.check(rt, str(tmp_path), steps=3, mode="tt", prompts=generation.PROMPTS[:1])["passed"]
  monkeypatch.setattr(generation, "ERROR", 0)
  report = generation.check(rt, str(tmp_path), steps=2, prompts=generation.PROMPTS[:1])
  assert not report["passed"] and "chip_top" in report["prompts"][0]["steps"][0], report

def test_qwen3_5_moe(chip, tmp_path, monkeypatch):
  # Qwen3.5-MoE's language model (also Qwen3.6's) from Hugging Face's files, against transformers in float32 with
  # random bf16 weights: a 100-token prompt (two DeltaNet chunks of 64), then replayed decode steps. numpy must be
  # within 0.5% (0.13% and 0.06% measured), and the chip within 3% of numpy (1.2%, 0.8%). At 2 positions of 100 the
  # chip picks a token 0.3% of the span away from numpy's, which bf16 can't tell apart. Tokens use all 8 experts,
  # because with a top 2, some of the 400 choices are near ties that bf16 breaks either way. They're 1e-5 to 1e-3
  # apart with any seed.
  torch, transformers = pytest.importorskip("torch"), pytest.importorskip("transformers")
  if not hasattr(transformers, "Qwen3_5MoeTextConfig"): pytest.skip("needs a transformers with Qwen3.5-MoE")
  # hide fla and causal_conv1d, whose GPU kernels transformers would otherwise use, so it runs its torch code
  monkeypatch.setitem(sys.modules, "fla", None), monkeypatch.setitem(sys.modules, "causal_conv1d", None)
  torch.manual_seed(0)
  config = transformers.Qwen3_5MoeTextConfig(hidden_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                                             linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
                                             num_experts=8, num_experts_per_tok=8, moe_intermediate_size=64, shared_expert_intermediate_size=64,
                                             vocab_size=256, max_position_embeddings=256, layer_types=["linear_attention"] * 3 + ["full_attention"])
  model = transformers.Qwen3_5MoeForCausalLM(config)
  for p in model.parameters():
    if p.ndim == 1: p.data.normal_(0, 0.3) if torch.all(p == 0) else p.data.normal_(1, 0.2)
  model = model.to(torch.bfloat16).float()
  model.save_pretrained(tmp_path)
  rng = np.random.default_rng(0)
  ids, steps = rng.integers(0, 256, 100).tolist(), rng.integers(0, 256, 3).tolist()
  with torch.no_grad():
    out = model(torch.tensor([ids]), use_cache=True)
    want, cache, want_steps = out.logits[0].numpy(), out.past_key_values, []
    for s in steps:
      out = model(torch.tensor([[s]]), past_key_values=cache, use_cache=True)
      want_steps.append(out.logits[0, -1].numpy())
      cache = out.past_key_values
  c, weights = llama.load(str(tmp_path), 128)
  names = {k.split(".", 2)[2] for k in weights if k.startswith("layers.0.")}
  assert c.gated and c.centered and c.partial == 0.25 and {"conv", "decay", "dt_bias"} <= names
  rel = lambda got, want: max(np.abs(g - w).max() / np.abs(w).max() for g, w in zip(got, want))
  w, kv = llama.arrays(weights), llama.kv_cache(c, 128, np.zeros)
  numpy = llama.forward(ids, list(range(100)), w, c, kv, every=True)
  numpy_steps = [llama.forward([s], [100 + i], w, c, kv)[0] for i, s in enumerate(steps)]
  assert rel(numpy, want) < 0.005 and rel(numpy_steps, want_steps) < 0.005, (rel(numpy, want), rel(numpy_steps, want_steps))
  rt = Runtime(chip)
  on = lambda row: Tensor.of(rt, [row], "int32")
  t, ckv = llama.upload(rt, weights), llama.kv_cache(c, 128, llama.zeros(rt))
  got = gpt2.values(llama.forward(on(ids), on(list(range(100))), t, c, ckv, every=True))
  step = tensor.jit(lambda ids, places: llama.forward(ids, places, t, c, ckv))
  got_steps = [llama.logits(step(on([s]), on([100 + i]))) for i, s in enumerate(steps)]
  assert rel(got, numpy) < 0.03 and near(got, numpy), rel(got, numpy)
  assert rel(got_steps, numpy_steps) < 0.03 and near(got_steps, numpy_steps), rel(got_steps, numpy_steps)

def test_qwen3_5_sequences(chip):
  # The small Qwen3.5 runs several sequences together, like a server's slots. Each must match the same sequence run
  # alone in numpy within 3% (0.8% measured). A decode step is one replayed call for all the slots. Free slot 1
  # (live 0) must keep its states.
  rng = np.random.default_rng(29)
  c, weights = qwen35(rng)
  w, size, prompts = llama.arrays(weights), 32, {0: [7, 200, 31], 2: [5, 64, 9, 9, 9]}
  p = {k.removeprefix("layers.0."): w[k] for k in w if k.startswith("layers.0.")}
  h, alone = rng.normal(0, 1, (3, 128)), [llama.kv_cache(c, size, np.zeros) for _ in range(3)]
  for s in alone: s[0][1][:] = rng.normal(0, 0.1, s[0][1].shape)  # give each sequence a state of its own
  many = llama.kv_cache(c, 3 * size, np.zeros, 3)
  for u in range(3): many[0][1][4 * u:4 * u + 4] = alone[u][0][1]
  together = qwen.deltanet(h, p, c, many[0], lambda op: None, np.array([[1.0], [0.0], [1.0]]))
  for u in (0, 2): assert np.allclose(together[u], qwen.deltanet(h[u:u + 1], p, c, alone[u][0], lambda op: None)[0], atol=1e-5)
  assert np.array_equal(many[0][1][4:8], alone[1][0][1]), "a free slot's state must not change"
  rt = Runtime(chip)
  t, kv, np_kv = llama.upload(rt, weights), llama.kv_cache(c, 3 * size, llama.zeros(rt), 3), {u: llama.kv_cache(c, size, np.zeros) for u in prompts}
  on, ids = lambda row, d="int32": Tensor.of(rt, [row], d), {u: list(s) for u, s in prompts.items()}
  rel = lambda got, want: np.abs(got - want).max() / np.abs(want).max()
  for u, s in prompts.items():
    got = gpt2.values(llama.forward(on(s), on(list(range(len(s)))), t, c, llama.slot(c, kv, u, 3), every=True))[len(s) - 1]
    want = llama.forward(s, list(range(len(s))), w, c, np_kv[u])[0]
    assert rel(got, want) < 0.03 and near([got], [want]), (u, rel(got, want))
    ids[u].append(int(want.argmax()))
  step = tensor.jit(lambda ids, places: llama.forward(ids, places, t, c, kv, 3))
  for i in range(4):
    places = [len(ids[u]) - 1 if u in ids else size - 1 for u in range(3)]
    got = gpt2.values(step(on([ids[u][-1] if u in ids else 0 for u in range(3)]), on(places)))
    for u in prompts:
      want = llama.forward(ids[u][-1:], [places[u]], w, c, np_kv[u])[0]
      assert rel(got[u], want) < 0.03 and near([got[u]], [want]), (i, u, rel(got[u], want))
      ids[u].append(int(want.argmax()))
  free = llama.slot(c, kv, 1, 3)
  assert not any(np.any(np.frombuffer(s.tobytes(), np.float32)) for l in range(3) for s in free[l]), "the free slot's states stay 0"

def test_model_geometry(chip):
  # Reject unsupported DeltaNet layouts and expert workspaces before output allocation or submission. None of the
  # malformed tensors below is read by a kernel, so an uninitialized allocation is sufficient for each input.
  rt = Runtime(chip)
  t = lambda shape: Tensor(rt, shape[1], shape[2], batch=shape[0], dtype="fp32")
  base = {k: t(shape) for k, shape in dict(S=(8, 32, 32), x=(1, 4, 392), conv=(3, 2, 256), taps=(4, 1, 256),
                                         dt=(1, 1, 4), decay=(1, 1, 4), first=(8, 32, 32), accept=(1, 1, 2), conv2=(3, 2, 256)).items()}
  bad = [dict(S=t((8, 33, 32))), dict(S=t((8, 32, 33))), dict(S=t((9, 32, 32))), dict(x=t((1, 3, 392))),
         dict(conv=t((3, 2, 257))), dict(conv=t((3, 2, 128))), dict(conv=t((3, 2, 320))), dict(first=None),
         dict(accept=None), dict(conv2=None), dict(first=t((4, 32, 32))), dict(accept=t((1, 2, 1))), dict(conv2=t((3, 1, 256))),
         dict(x=t((1, 2, 392))), dict(dt=t((1, 1, 8))), dict(live=Tensor(rt, 2, 1))]
  for changed in bad:
    before = dict(rt.used), rt.launches
    with pytest.raises(ValueError, match="delta"): qwen.delta(**(base | changed))
    assert (rt.used, rt.launches) == before
  ids, weights = Tensor(rt, 8192, 1, dtype="int32"), Tensor(rt, 8192, 1)
  before = dict(rt.used), rt.launches
  with pytest.raises(ValueError, match="fit L1"): qwen.touched(ids, 1, weights)
  with pytest.raises(ValueError, match="fit L1"): qwen.touched_kernel(8192, 1, 1)
  assert (rt.used, rt.launches) == before
  fitting = next(rows for rows in range(8192, 0, -32) if (qwen.DATA + rows // 32 * 6144 + 8 + 63) // 64 * 64 + 4096 <= qwen.OUTS)
  assert qwen.touched_l1(fitting, 1, 1) + 4096 <= qwen.OUTS
  with pytest.raises(ValueError, match="fit L1"): qwen.touched_l1(fitting + 32, 1, 1)

def test_all_linear_model(chip):
  # A truncated hybrid can contain only recurrent blocks. Direct calls use RoPE capacity; the server supplies its
  # smaller slot capacity so its idle position leaves recurrent state untouched.
  rt, rng = Runtime(chip), np.random.default_rng(31)
  c, weights = qwen35(rng, places=64)
  c.layers, c.mtp = 3, 0
  weights = {k: v for k, v in weights.items() if not k.startswith(("mtp.", "layers.3.", "layers.4."))}
  w, t = llama.arrays(weights), llama.upload(rt, weights)
  on = lambda row, dtype="int32": Tensor.of(rt, [row], dtype)
  ids = [7, 200, 31]
  got = gpt2.values(llama.forward(on(ids), on([0, 1, 2]), t, c, llama.kv_cache(c, 32, llama.zeros(rt)), every=True))
  want = llama.forward(ids, [0, 1, 2], w, c, llama.kv_cache(c, 32, np.zeros), every=True)
  assert np.abs(got - want).max() < 0.03 * np.abs(want).max()
  before = dict(rt.used), rt.launches
  for k in ("cos", "sin"):
    with pytest.raises(ValueError, match="RoPE"):
      serve.Engine(rt, t | {k: t[k][:, :32, :]}, c, None, slots=2, places=64)
    assert (rt.used, rt.launches) == before
  engine = serve.Engine(rt, t, c, None, slots=2, places=32, keep=0, park=0)
  state = llama.kv_cache(c, 32, np.zeros)
  for place, token in enumerate(ids[:2]):
    logits, _ = engine.decode(on([token, 0]), on([place, 31]), engine.column([1, 1]), 32)
    got, want = gpt2.values(logits)[0], llama.forward([token], [place], w, c, state)[0]
    assert np.abs(got - want).max() < 0.03 * np.abs(want).max()
  idle = llama.slot(c, engine.kv, 1, 2)
  assert all(not np.any(np.frombuffer(s.tobytes(), np.float32)) for block in idle for s in block)

def test_delta_step(chip):
  # qwen.delta runs Gated DeltaNet decode in one kernel, with a work unit per key head. It must match numpy_delta to 1e-5 with the
  # conv inputs exact. The shapes go up to Qwen3.6's 32 heads of 128 by 128. The second sequence is inactive (live 0). Its
  # conv and states stay as they were, and where reads are counted, it reads only its live and accept. With m=2 it's a
  # verify step, which starts from the states after both tokens of the last step or after the first, as accept says.
  rt, rng = Runtime(chip), np.random.default_rng(1)
  up = lambda a: Tensor.frombytes(rt, a.shape[-2], a.shape[-1], np.float32(a).tobytes(), a.shape[0], "fp32")
  for m in (1, 2):
    for n, kh, vh, kd, vd in ((1, 2, 4, 32, 32), (3, 2, 4, 64, 96), (4, 16, 32, 128, 128)):
      width = 2 * kh * kd + vh * vd
      x = np.float32(rng.normal(0, 1, (1, n * m, width + vh * vd + 2 * vh)))
      conv, conv2 = (np.float32(rng.normal(0, 1, (3, n, width))) for _ in range(2))
      taps, dt = np.float32(rng.normal(0, 0.5, (4, 1, width))), np.float32(rng.normal(0, 1, (1, 1, vh)))
      decay = np.float32(-rng.uniform(0.5, 2, (1, 1, vh)))
      S, first = (np.float32(rng.normal(0, 0.1, (n * vh, kd, vd))) for _ in range(2))
      live, accept = np.float32(np.arange(n * m) // m != 1).reshape(1, n * m, 1), np.float32(np.arange(n) % 2).reshape(1, 1, n)
      ts = {k: up(a) for k, a in dict(S=S, x=x, conv=conv, taps=taps, dt=dt, decay=decay, first=first, accept=accept, conv2=conv2).items()}
      call = lambda live: qwen.delta(*(ts[k] for k in ("S", "x", "conv", "taps", "dt", "decay")), Tensor.of(rt, live[0].tolist()),
                                     *((ts["first"], ts["accept"], ts["conv2"]) if m > 1 else ()))
      got = value(call(live))
      want_S, want_first, want_conv, want_conv2 = np.float64(S), np.float64(first), np.float64(conv), np.float64(conv2)
      want = qwen.numpy_delta(want_S, x, want_conv, taps, dt, decay, live, *((want_first, accept, want_conv2) if m > 1 else ()))
      for a, b in ((got, want), (value(ts["S"]), want_S)) + (((value(ts["first"]), want_first),) if m > 1 else ()):
        assert np.abs(a - b).max() < 1e-5 * np.abs(b).max(), (m, n, kd, vd, np.abs(a - b).max() / np.abs(b).max())
      for a, b in ((value(ts["conv"]), want_conv),) + (((value(ts["conv2"]), want_conv2),) if m > 1 else ()):
        assert np.array_equal(a, np.float32(b)), (m, n, "the conv's inputs")
      if n == 1: continue
      assert np.array_equal(value(ts["S"])[vh:2 * vh], S[vh:2 * vh]) and np.array_equal(value(ts["conv"])[:, 1], conv[:, 1]), \
        "inactive sequence state must remain unchanged"
      if chip.pci.counts:  # repeat with both sequences active, then with the second inactive
        reads = []
        for alive in (np.ones_like(live), live):
          since = rt.launches
          call(alive)
          reads += [r[3] for r in rt.profile(since) if r[0] == "delta"]
        assert n * (reads[1] - kh * 64 * m) == (n - 1) * reads[0], (m, n, reads)

def test_delta_pieces(chip):
  # qwen.chunk runs a Gated DeltaNet prompt in pieces of 64 tokens, and must match a piece at a time to 1e-6. Then the
  # small Qwen3.5 runs a 150-token prompt, three pieces with the last one padded, within 5% of numpy.
  rng = np.random.default_rng(47)
  c, weights = qwen35(rng, places=256)
  w = llama.arrays(weights)
  q, k = rng.normal(0, 0.3, (4, 150, 32)), rng.normal(0, 0.3, (4, 150, 32))
  k /= np.linalg.norm(k, axis=-1, keepdims=True)
  v, beta, g, S = rng.normal(0, 1, (4, 150, 48)), rng.uniform(0, 1, (4, 150, 1)), -rng.uniform(0, 0.2, (4, 150, 1)), rng.normal(0, 0.1, (4, 32, 48))
  o, after = qwen.chunk(q, k, v, beta, g, S)
  for s0 in range(0, 150, 64):  # the same pieces as chunk's, in separate calls
    piece, S = qwen.chunk(*(t[:, s0:s0 + 64, :] for t in (q, k, v, beta, g)), S)
    assert np.abs(piece - o[:, s0:s0 + 64, :]).max() < 1e-6
  assert np.abs(after - S).max() < 1e-6
  rt = Runtime(chip)
  t, on = llama.upload(rt, weights), lambda row: Tensor.of(rt, [row], "int32")
  ids = [int(i) for i in rng.integers(0, 256, 150)]
  want = llama.forward(ids, list(range(150)), w, c, llama.kv_cache(c, 256, np.zeros), every=True)
  got = gpt2.values(llama.forward(on(ids), on(list(range(150))), t, c, llama.kv_cache(c, 256, llama.zeros(rt)), every=True))
  assert max(np.abs(a - b).max() / np.abs(b).max() for a, b in zip(got, want)) < 0.05 and near(got, want)

def test_speculative(chip):
  # Two slots verify their token and draft in one replayed call, with a third slot free. A right draft is accepted. A
  # wrong one is rejected (accept 0), and the next step goes on from the Gated DeltaNet states as they were after the
  # first token. The multi-token prediction layer (qwen.draft) must match numpy too, with the output layer and with a
  # bfp4 copy of it.
  rng = np.random.default_rng(37)
  c, weights = qwen35(rng)
  w, size = llama.arrays(weights), 32
  rt = Runtime(chip)
  t, kv = llama.upload(rt, weights), llama.kv_cache(c, 3 * size, llama.zeros(rt), 3, speculative=True)
  on, rel = lambda row, d="int32": Tensor.of(rt, [row], d), lambda got, want: np.abs(got - want).max() / np.abs(want).max()
  ref = lambda s: llama.forward(s, list(range(len(s))), w, c, llama.kv_cache(c, size, np.zeros), every=True)  # numpy, from an empty cache
  seqs = {0: [7, 200, 31], 1: [5, 64, 9, 9, 9]}  # each starts as a prompt and gains the tokens it knows
  for u, s in seqs.items():
    got, h = llama.forward(on(s), on(list(range(len(s)))), t, c, llama.slot(c, kv, u, 3), hidden=True)
    want = ref(s)
    assert rel(gpt2.values(got)[-1], want[-1]) < 0.03
    s.append(int(want[-1].argmax()))
    if u == 0:  # the multi-token prediction layer over the prompt, from each position's hidden state and the token after it
      _, nh = llama.forward(s[:-1], list(range(len(s) - 1)), w, c, llama.kv_cache(c, size, np.zeros), hidden=True)
      mine = gpt2.values(qwen.draft(on(s[1:]), on(list(range(len(s) - 1))), h, t, c, llama.slot(c, kv, u, 3)))
      theirs = qwen.draft(s[1:], list(range(len(s) - 1)), nh, w, c, llama.kv_cache(c, size, np.zeros))
      assert max(rel(mine[r], theirs[r]) for r in range(len(s) - 1)) < 0.03
      c.precision = {("draft head", None): ("bfp4", 1)}
      head = llama.upload(rt, {k: weights[k] for k in ("embed.T", "mtp.fc")}, c.precision)["embed.T draft"]
      assert head.dtype == "bfp4"
      mine = gpt2.values(qwen.draft(on(s[1:]), on(list(range(len(s) - 1))), h, t | {"embed.T draft": head}, c, llama.slot(c, kv, u, 3)))
      theirs = qwen.draft(s[1:], list(range(len(s) - 1)), nh, llama.arrays(weights, c.precision), c, llama.kv_cache(c, size, np.zeros))
      assert max(rel(mine[r], theirs[r]) for r in range(len(s) - 1)) < 0.03
      c.precision = None
  step = tensor.jit(lambda ids, places, accept: llama.forward(ids, places, t, c, kv, 3, accept=accept))
  accept = [1.0, 1.0, 1.0]
  for i in range(6):
    right = {u: int(ref(s)[-1].argmax()) for u, s in seqs.items()}
    drafts = {u: right[u] if (i + u) % 2 == 0 else (right[u] + 1) % 256 for u in seqs}
    ids = [x for u in range(3) for x in ([seqs[u][-1], drafts[u]] if u in seqs else [0, 0])]
    places = [p for u in range(3) for p in ([len(seqs[u]) - 1, len(seqs[u])] if u in seqs else [size - 1, size - 1])]
    got = gpt2.values(step(on(ids), on(places), on(accept, "fp32")))
    for u, s in seqs.items():
      want = ref(s + [drafts[u]])[-2:]
      assert rel(got[2 * u], want[0]) < 0.03 and rel(got[2 * u + 1], want[1]) < 0.03, (i, u, rel(got[2 * u], want[0]), rel(got[2 * u + 1], want[1]))
      accept[u] = float(drafts[u] == right[u])
      s += [drafts[u], int(want[1].argmax())] if accept[u] else [right[u]]

def test_mxfp4(tmp_path):
  # gpt-oss keeps its experts in MXFP4. qwen.mxfp4 must unpack them to bf16 bit for bit like transformers does, and
  # llama.load must split their interleaved gate and up columns apart.
  torch, packed = pytest.importorskip("torch"), pytest.importorskip("transformers.integrations.mxfp4")
  st = pytest.importorskip("safetensors.numpy")
  rng = np.random.default_rng(43)
  blocks, scales = rng.integers(0, 256, (2, 64, 3, 16), dtype=np.uint8), rng.integers(118, 136, (2, 64, 3), dtype=np.uint8)
  want = packed.convert_moe_packed_tensors(torch.from_numpy(blocks), torch.from_numpy(scales)).view(torch.int16).numpy().view(np.uint16)
  assert np.array_equal(qwen.mxfp4(blocks, scales), want)
  config = {"hidden_size": 96, "num_attention_heads": 3, "num_key_value_heads": 1, "num_hidden_layers": 1, "rms_norm_eps": 1e-5,
            "rope_theta": 1e4, "max_position_embeddings": 32, "vocab_size": 32, "intermediate_size": 32, "head_dim": 32,
            "num_experts_per_tok": 1, "model_type": "gpt_oss"}
  (tmp_path / "config.json").write_text(json.dumps(config))
  f32 = lambda *shape: rng.normal(0, 1, shape).astype(np.float32)
  shapes = (("q", (96, 96)), ("k", (32, 96)), ("v", (32, 96)), ("o", (96, 96)))
  attention = {f"model.layers.0.self_attn.{n}_proj.weight": f32(*shape) for n, shape in shapes}
  st.save_file(attention | {"model.embed_tokens.weight": f32(32, 96), "model.norm.weight": f32(96), "lm_head.weight": f32(32, 96),
                "model.layers.0.mlp.experts.gate_up_proj_blocks": blocks, "model.layers.0.mlp.experts.gate_up_proj_scales": scales,
                "model.layers.0.mlp.experts.gate_up_proj_bias": f32(2, 64), "model.layers.0.mlp.experts.down_proj": f32(2, 32, 96),
                "model.layers.0.mlp.experts.down_proj_bias": f32(2, 96), "model.layers.0.mlp.router.weight": f32(2, 96),
                "model.layers.0.mlp.router.bias": f32(2)},
               str(tmp_path / "model.safetensors"))
  c, weights = llama.load(str(tmp_path), 32)
  assert np.array_equal(weights["layers.0.experts.gate_up"].bits(), np.concatenate([want[..., ::2], want[..., 1::2]], -1)) and c.limit == 7.0

TWO_CHIPS = r"""
import os
from baretile import bh, pci
from baretile.chip import Chip
from baretile.runtime import Runtime
from baretile.tensor import Tensor
chips = [Chip(pci.open_device(n)) for n in range(2)]
rts = [Runtime(chip) for chip in chips]
for n, rt in enumerate(rts): assert (Tensor.of(rt, [[1.0, 2.0]]) * (n + 2)).tolist() == [[n + 2.0, 2 * n + 4.0]], n
(a, links_a), (b, links_b) = (chip.links() for chip in chips)
assert a != b and len(links_a) == len(links_b) == 2, (a, links_a, b, links_b)
assert all(peer == b and links_b[back] == (a, ch) for ch, (peer, back) in links_a.items()), (links_a, links_b)
for src, dst, links in ((chips[0], chips[1], links_a), (chips[1], chips[0], links_b)):
  for ch, (_, back) in links.items():
    data, flag = os.urandom(8192), os.urandom(16)
    src.write(*bh.ETH[ch], 0x20000, data + flag)
    dst.write(*bh.ETH[back], 0x40000, bytes(len(data) + 16))
    src.send(ch, 0x20000, 0x40000, len(data))
    src.send(ch, 0x20000 + len(data), 0x40000 + len(data), 16)
    src.wait(lambda: dst.read(*bh.ETH[back], 0x40000 + len(data), 16) == flag, "deliver the flag", 2.0)
    assert dst.read(*bh.ETH[back], 0x40000, len(data)) == data, ch
import random
x = [[[random.uniform(-4, 4) for _ in range(100)] for _ in range(70)] for _ in range(2)]
t = Tensor.of(rts[0], x)  # 24 tiles: 6 chunks through the 2 slots, so the credits count
for ch, (_, back) in links_a.items():
  y = t.to(rts[1], ch, back)
  assert y.rt is rts[1] and y.tolist() == t.tolist() and (y * 2).to(rts[0], back, ch).tolist() == (t * 2).tolist(), ch
import struct
raw = struct.pack("<4f", 1.0, 2.0, -0.0, 3.0)
view = Tensor.frombytes(rts[0], 1, 2, raw, batch=2, dtype="fp32")[1]
assert view.to(rts[1], ch, back).tobytes() == raw[8:]
for dtype in ("bfp8", "bfp4"):
  source = Tensor.of(rts[0], [[[1.5] * 32] * 32] * 2).cast(dtype).realize()
  for view in (source[1], source.repeat_interleave(2)):
    assert view.to(rts[1], ch, back).tobytes() == view.tobytes()
import numpy as np  # a model split across the two chips: its first block on chip 0, its second and its head on chip 1
from types import SimpleNamespace
from baretile.models import gpt2, llama
c, rng = llama.model(dim=128, heads=4, kv=2, layers=2, eps=1e-5, theta=1e4), np.random.default_rng(3)
draw = lambda mean, std, shape: gpt2.bf16_floats(gpt2.bf16_bits(rng.normal(mean, std, shape)))
w = {f"layers.{l}.{k}": draw(*how) for l in range(2) for k, how in (("input_layernorm", (1, 0.1, (1, 128))), ("qkv_proj", (0, 0.05, (128, 256))),
     ("o_proj", (0, 0.05, (128, 128))), ("post_attention_layernorm", (1, 0.1, (1, 128))), ("gate_up_proj", (0, 0.05, (128, 512))),
     ("down_proj", (0, 0.05, (256, 128))))}
w |= {"norm": draw(1, 0.1, (1, 128)), "embed": draw(0, 1, (256, 128))}
w |= {"embed.T": w["embed"].T.copy()} | dict(zip(("cos", "sin", "turn"), llama.rotary(c, 32)))
ids = [5, 17, 3, 200, 9]
want = llama.forward(ids, list(range(5)), w, c, llama.kv_cache(c, 32, np.zeros))[0]
up = [{k: Tensor.of(rt, v.tolist()) for k, v in w.items()} for rt in rts]
places = [Tensor.of(rt, [list(range(5))], "int32") for rt in rts]
caches = [llama.kv_cache(c, 32, llama.zeros(rt)) for rt in rts]
f = llama.fns(up[0]["norm"])
x = f.rows(up[0]["embed"], Tensor.of(rts[0], [ids], "int32"))
ch, (_, back) = next(iter(links_a.items()))
for l, n in ((0, 0), (1, 1)):
  if x.rt is not rts[n]: x = x.to(rts[n], ch, back)  # the hidden state crosses the link
  rotary = (f.rows(up[n]["cos"], places[n]), f.rows(up[n]["sin"], places[n]), up[n]["turn"])
  p = gpt2.block(up[n], f"layers.{l}.")
  x = llama.layer(x, p, c, rotary, gpt2.mask(places[n], 32), caches[n][l], places[n], l, True)
got = llama.logits(f.mm(f.rows(llama.rms(x, up[1]["norm"], c), [4]), up[1]["embed.T"], None))
assert np.abs(got - want).max() < 0.02 * np.ptp(want) and got.argmax() == want.argmax(), (np.abs(got - want).max(), np.ptp(want))
print("ok")
"""

@pytest.mark.skipif(pci.simulator() is None, reason="a ramp opens the device itself, and a card can't be open in two processes")
@pytest.mark.parametrize("ramp", sorted(p.name for p in (pathlib.Path(__file__).parent.parent / "ramps").glob("*.py")))
def test_ramp(ramp):
  # Run separately because each ramp opens its own device and ttsim allows one instance per process.
  import subprocess
  root = pathlib.Path(__file__).parent.parent
  run = subprocess.run([sys.executable, str(root / "ramps" / ramp)], capture_output=True, text=True, timeout=600, cwd=root)
  assert run.returncode == 0 and run.stdout.strip().splitlines()[-1].startswith("the ramp ran"), run.stdout + run.stderr

@pytest.mark.skipif(pci.simulator() is None, reason="the examples open the device themselves, and a card can't be open in two processes")
def test_readme_examples():
  # The README's Python blocks print what their "# " lines say, and baretile.hello prints what the README says it does,
  # apart from the count of workers, which depends on the chip.
  root = pathlib.Path(__file__).parent.parent
  readme = (root / "README.md").read_text()
  run = lambda *args: subprocess.run([sys.executable, *args], capture_output=True, text=True, timeout=600, cwd=root)
  for block in re.findall(r"```python\n(.*?)```", readme, re.S):
    out, said = run("-c", block), [line[2:] for line in block.splitlines() if line.startswith("# ")]
    assert out.returncode == 0 and out.stdout.splitlines() == said, out.stdout + out.stderr
  said, out = re.search(r"`baretile.hello` prints:\n\n```text\n(.*)\n```", readme)[1], run("-m", "baretile.hello")
  workers = lambda text: re.sub(r"^\d+ Tensix workers", "N Tensix workers", text.strip())
  assert out.returncode == 0 and workers(out.stdout) == workers(said), out.stdout + out.stderr

def test_website_matches_readme():
  # Each line in the homepage's code blocks is a line in one of the README's, so the two can't drift apart. Comments
  # at the end of a homepage line are its own.
  root = pathlib.Path(__file__).parent.parent
  readme = {line for block in re.findall(r"```\w*\n(.*?)```", (root / "README.md").read_text(), re.S) for line in block.splitlines()}
  page = (root / "website/public/index.html").read_text()
  lines = [html.unescape(re.sub(r"<[^>]+>", "", line)) for block in re.findall(r"<pre[^>]*><code>(.*?)</code></pre>", page, re.S)
           for line in block.splitlines()]
  missing = [line for line in (re.sub(r"\s+# .*$", "", line) for line in lines) if line.strip() and line not in readme]
  assert lines and not missing, missing

def test_doc_links():
  # Each link in the Markdown files to a file of the repository, by a relative path or on GitHub's main branch, finds
  # the file, and each #anchor in one finds a heading there, as GitHub makes them. Links in code blocks don't count.
  root = pathlib.Path(__file__).parent.parent
  headings = lambda p: {re.sub(r"[^\w\- ]", "", h.strip().lower()).replace(" ", "-") for h in re.findall(r"^#+ (.+)$", p.read_text(), re.M)}
  broken = []
  for page in [*root.glob("*.md"), *root.glob("docs/*.md"), *root.glob("website/*.md")]:
    for target in re.findall(r"\]\(([^)\s]+)\)", re.sub(r"```.*?```", "", page.read_text(), flags=re.S)):
      if m := re.fullmatch(r"https://github\.com/baretile/baretile/(?:blob|tree)/main/([^#]*)(?:#(.*))?", target): path, anchor = root / m[1], m[2]
      elif re.match(r"\w+:", target): continue  # another site
      else:
        file, _, anchor = target.partition("#")
        path = page.parent / file if file else page
      if not path.exists() or anchor and path.suffix == ".md" and anchor not in headings(path): broken.append(f"{page.name}: {target}")
  assert not broken, broken

@pytest.mark.skipif("TTSIM2" not in os.environ, reason="set TTSIM2 to ttsim's two-chip libttsim_bh_x2.so")
def test_two_chips():
  # ttsim's two-chip build is a P300: two Blackholes joined by two Ethernet links. It runs in its own process, since a
  # process can load only one libttsim. Chip.links must pair the links, Chip.send and Tensor.to must move data across
  # each link in both directions, and a model split across the two chips must give numpy's logits.
  import subprocess
  run = subprocess.run([sys.executable, "-c", TWO_CHIPS], env=os.environ | {"TTSIM": os.environ["TTSIM2"]}, capture_output=True, text=True,
                       timeout=600, cwd=pathlib.Path(__file__).parent.parent)
  assert run.returncode == 0 and run.stdout.strip().endswith("ok"), run.stdout + run.stderr

@pytest.mark.skipif("LLAMA8B" not in os.environ, reason="set LLAMA8B=1 to test full block dimensions; this takes several minutes on ttsim")
def test_llama_8b_shapes(chip):
  # One block in Llama-3.1-8B's shapes at TT's performance precision, so B2's matmuls, casts and attention run at
  # full size. Only the vocabulary is cut, to 2048. On ttsim the casts in llama.upload take minutes, and upload waits
  # for them. The MLP's 4096 x 28672 in bfp4 is 114,688 tiles.
  rt, rng = Runtime(chip), np.random.default_rng(8)
  rope = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
  c = llama.model(dim=4096, heads=32, kv=8, layers=1, eps=1e-5, theta=500000.0, rope=rope, vocab=2048, ffn=14336, places=96)
  bits = lambda shape, scale, mean=0.0: gpt2.bf16_bits(rng.normal(mean, scale, shape))
  w = {"layers.0.input_layernorm": bits((1, 4096), 0.1, 1), "layers.0.post_attention_layernorm": bits((1, 4096), 0.1, 1),
       "layers.0.qkv_proj": bits((4096, 6144), 0.02), "layers.0.o_proj": bits((4096, 4096), 0.02), "layers.0.gate_up_proj": bits((4096, 28672), 0.02),
       "layers.0.down_proj": bits((14336, 4096), 0.02), "norm": bits((1, 4096), 0.1, 1), "embed": bits((2048, 4096), 1.0)}
  w |= {"embed.T": np.ascontiguousarray(w["embed"].T)} | {k: gpt2.bf16_bits(a) for k, a in zip(("cos", "sin", "turn"), llama.rotary(c, c.places))}
  weights, c.precision = {k: llama.Weight(lambda v=v: v) for k, v in w.items()}, llama.precision(c, "tt")
  t = llama.upload(rt, weights, c.precision)
  arrays, on = llama.arrays(weights, c.precision), lambda row: Tensor.of(rt, [row], "int32")
  kv, np_kv = llama.kv_cache(c, 96, llama.zeros(rt, "bfp8")), llama.kv_cache(c, 96, np.zeros)
  prompt = [int(i) for i in rng.integers(0, 2048, 64)]
  for new, places in ((prompt, list(range(64))), ([5], [64]), ([9], [65])):
    out = llama.forward(on(new), on(places), t, c, kv).realize()
    rt.wait(600)
    got, want = llama.logits(out), llama.forward(new, places, arrays, c, np_kv)[0]
    assert got.argmax() == want.argmax() and np.abs(got - want).max() < 0.02 * np.ptp(want), (np.abs(got - want).max(), np.ptp(want))

@pytest.mark.skipif(pci.simulator() is None, reason="bring-up opens the device itself, and a card can't be open in two processes")
def test_bringup(tmp_path):
  # test/bringup.py runs its stages in order and records each one. Without clang, preflight fails, and the run stops
  # there with the rest not run. The first workload's gate reads the accuracy that llama.main prints.
  import bringup, subprocess
  run = lambda record, *args, **env: subprocess.run([sys.executable, bringup.__file__, str(tmp_path / record), *args], capture_output=True,
                                                    text=True, timeout=1200, env=os.environ | env)
  results = lambda record: {s["stage"]: s["result"] for s in json.loads((tmp_path / record / "record.json").read_text())["stages"]}
  out = run("up", "--only", "preflight,chip,transfers")
  assert out.returncode == 0 and [k for k, v in results("up").items() if v == "passed"] == ["preflight", "chip", "transfers"], out.stdout + out.stderr
  out = run("down", PATH="/nonexistent")
  assert out.returncode == 1 and set(results("down").values()) == {"failed", "not run"} and results("down")["preflight"] == "failed"
  assert "Install clang" in (tmp_path / "down" / "preflight.log").read_text()
  assert bringup.accurate("top-1 90.0%, top-5 98.0% (12s)") and not bringup.accurate("top-1 89.8%, top-5 99.0%") and not bringup.accurate("")

def test_benchmark_bad_stack():
  # An unknown stack is a ValueError, also under python -O.
  import subprocess
  run = subprocess.run([sys.executable, "-O", benchmark.__file__, "bad-stack", "unused-model", "unused-reference"],
                       capture_output=True, text=True, timeout=60)
  assert run.returncode != 0 and "ValueError: the stack must be" in run.stderr, run.stdout + run.stderr

def test_benchmark_table():
  # The speed column is baretile's over TT-NN's, inverted for times, so above 1 means baretile is faster. Work at the
  # card's one-phase peak rate has a roofline of 1, so a two-phase matmul gets there at half the flops.
  ours = {"versions": {"lines of code": {"baretile": 1}}, "clock": "the host's", "B1 matmul 4096^3 bf16, 2 phases": {"ms": 3.0},
          "B2 decode, batch 1": {"tokens/s": 60.0}, "B3 prefill, 32768 tokens": {"does not run": "MemoryError"}}
  theirs = {"versions": {"lines of code": {"tt-metal": 2}}, "clock": "the host's", "B1 matmul 4096^3 bf16, 2 phases": {"ms": 1.5},
            "B2 decode, batch 1": {"tokens/s": 50.0}, "B3 prefill, 32768 tokens": {"seconds to first token": 9.0}}
  rows = benchmark.table(ours, theirs).splitlines()
  assert rows[2].endswith("| 0.50 (not above 1) |") and rows[3].endswith("| 1.20 (above 1) |") and rows[4].endswith("|  |"), rows
  assert all(r.count("|") == 5 for r in rows) and "MemoryError" in rows[4], rows
  flops = cost.CARD["tflops"] / 2  # a second of a two-phase matmul at the peak
  assert benchmark.roofline(flops * 2, 1.0) == 1
  c = llama.model(layers=2, dim=256, heads=4, kv=2, ffn=512, vocab=1000)
  s, attention = cost.shape(c, "tt"), 2 * 4 * 4 * llama.head(c) * 10 * 11 // 2  # two blocks, scores and values, 4 heads, causal over 10 tokens
  assert benchmark.prefill_work(c, 10) == 2 * 10 * (s.work - s.head_work) + 2 * s.head_work + s.prefill_phases * attention

def test_benchmark_load(chip, tmp_path):
  # B4's load against serve.py for a small model: two agents, two streamed turns each. The second turn extends the
  # first, so most of its prompt should already be cached.
  rt, rng = Runtime(chip), np.random.default_rng(37)
  c, w = smol(rng, 2, places=128)
  engine = serve.Engine(rt, {k: Tensor.of(rt, v.tolist()) for k, v in w.items()}, c, bytes_tokenizer(tmp_path), slots=2, places=128, chunk=32)
  threading.Thread(target=engine.run, daemon=True).start()
  httpd = serve.server(engine, "smol", port=0)
  threading.Thread(target=httpd.serve_forever, daemon=True).start()
  got = benchmark.load(f"http://localhost:{httpd.server_address[1]}", 2, SimpleNamespace(system=6, task=2, tool=2, out=4, n=2),
                       " ".join(f"w{i}" for i in range(40)))
  httpd.shutdown()
  assert got["turns"] == 4 and got["tokens/s"] > 0 and got["seconds to first text"] > 0 and got["prompt tokens cached"] > 0.3, got

def test_benchmark_streams():
  # External servers may omit usage/cache details or emit all text in one event. None of these imply token timing.
  from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
  cases = [({"choices": [{"text": "abc"}]}, {"choices": []}),
           ({"choices": [{"text": "abc"}]}, {"choices": [], "usage": {"completion_tokens": 3, "prompt_tokens": 2}}),
           ({"choices": [], "usage": {"completion_tokens": 0, "prompt_tokens": 0, "prompt_tokens_details": {"cached_tokens": 0}}},)]
  bodies = []
  class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_POST(self):
      bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
      self.send_response(200)
      self.end_headers()
      for chunk in cases.pop(0): self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
      self.wfile.write(b"data: [DONE]\n\n")
  server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
  worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
  worker.start()
  try:
    call = lambda: benchmark.load(f"http://127.0.0.1:{server.server_port}", 1, SimpleNamespace(system=1, task=1, tool=1, out=3, n=1), "a b c")
    missing, combined, empty = call(), call(), call()
    assert missing["turns"] == 1 and missing["tokens/s"] is None and missing["tokens/s a user"] is None
    assert missing["seconds to first text"] > 0 and missing["prompt tokens cached"] is None
    assert combined["tokens/s"] > 0 and combined["tokens/s a user"] > 0 and combined["prompt tokens cached"] is None
    assert empty["tokens/s"] == empty["tokens/s a user"] == 0 and empty["seconds to first text"] is None
    assert empty["prompt tokens cached"] is None and not any(r["failures"] for r in (missing, combined, empty))
    assert all(b["stream_options"] == {"include_usage": True} for b in bodies)
  finally:
    server.shutdown()
    server.server_close()
    worker.join(1)

@pytest.mark.parametrize("headers", [False, True])
def test_benchmark_deadline(headers):
  # Keep sending bytes so no individual socket read times out. The workload must still finish and keep turn one.
  from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
  requests = []
  class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_POST(self):
      self.rfile.read(int(self.headers["Content-Length"]))
      requests.append(1)
      if len(requests) == 1:
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'data: {"choices":[{"text":"abc"}],"usage":{"completion_tokens":3,"prompt_tokens":1}}\n\ndata: [DONE]\n\n')
        return
      try:
        if headers: self.wfile.write(b"HTTP/1.1 200 OK\r\nX-Trickle: ")
        else:
          self.send_response(200)
          self.end_headers()
          self.wfile.write(b": ")
        for _ in range(500):  # 5 seconds, well past the deadline
          self.wfile.write(b"x")
          self.wfile.flush()
          time.sleep(0.01)
      except OSError: pass
  server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
  worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
  worker.start()
  start = time.monotonic()
  try:
    got = benchmark.load(f"http://127.0.0.1:{server.server_port}", 1,
                         SimpleNamespace(system=1, task=1, tool=1, out=3, n=2), "a b c d", timeout=1)
    assert time.monotonic() - start < 3 and got["turns"] == 1 and got["tokens/s"] > 0, got
    assert len(got["failures"]) == 1 and "deadline" in got["failures"][0], got
  finally:
    server.shutdown()
    server.server_close()
    worker.join(1)

def test_benchmark_subprocess_deadline(tmp_path):
  # The child ignores termination. Its partial output survives the deadline and the forced process-group exit. The
  # deadline leaves a slow machine time to start Python and print; logged then waits as long again before SIGKILL.
  log = tmp_path / "benchmark.log"
  start = time.monotonic()
  code, expired = benchmark.logged([sys.executable, "-u", "-c",
    "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('partial result',flush=True); time.sleep(30)"], log, timeout=1)
  assert expired and code < 0 and time.monotonic() - start < 4
  assert "partial result" in log.read_text() and "deadline exceeded" in log.read_text()

def test_chunked_prefill(chip):
  # llama.prefill runs a long prompt a chunk at a time, so attention's scores are a chunk by the cache instead of the
  # prompt squared. B3's 32,768 tokens need this. Here 70 tokens go in chunks of 32, 32 and 6.
  rt, rng = Runtime(chip), np.random.default_rng(29)
  c, w = smol(rng, 2, places=96)
  t, prompt = {k: Tensor.of(rt, v.tolist()) for k, v in w.items()}, [int(i) for i in rng.integers(0, 256, 70)]
  on = lambda row: Tensor.of(rt, [row], "int32")
  kv, np_kv = llama.kv_cache(c, 96, lambda shape: Tensor.of(rt, np.zeros(shape).tolist())), llama.kv_cache(c, 96, np.zeros)
  want = llama.prefill(prompt, w, c, np_kv, 32)[0]
  for ids, chunk in (([], 32), (prompt, 0), (prompt, -1)):
    with pytest.raises(ValueError, match="prefill"): llama.prefill(ids, w, c, np_kv, chunk)
  assert np.allclose(want, llama.forward(prompt, range(70), w, c, llama.kv_cache(c, 96, np.zeros))[0], atol=1e-3)
  got = llama.logits(llama.prefill(prompt, t, c, kv, 32, on))
  assert np.abs(got - want).max() < 0.03 * np.abs(want).max(), np.abs(got - want).max()
  nxt = int(want.argmax())
  got, want = llama.logits(llama.forward(on([nxt]), on([70]), t, c, kv)), llama.forward([nxt], [70], w, c, np_kv)[0]
  assert np.abs(got - want).max() < 0.03 * np.abs(want).max(), np.abs(got - want).max()

def test_llama_precision(chip):
  # TT's performance precision puts the MLP's gate and up in bfp4 and the rest in bfp8, except in Llama-3.1-8B's last
  # block. numpy with the same block-float values but exact products must pick the same tokens, within 1% of the span.
  # upload waits for each cast, so the bf16 copy of the next weight reuses the same DRAM.
  big = llama.model(layers=32, dim=4096)
  assert llama.precision(big, "tt")["gate_up", 31] == ("bfp8", 2) and llama.precision(big, "tt")["gate_up", 30] == ("bfp4", 1)
  assert llama.precision(big, "tt")["head", None] == ("bfp8", 2) and llama.precision(big, "bf16")["qkv", 0] == ("bf16", 4)
  tt = llama.precision(big, "tt")
  assert [tt[k] for k in (("attention", "decode"), ("attention", "prefill"), ("kv", None))] == [(None, 2), (None, 4), ("bfp8", None)]
  # "experts" fits Qwen3.6 on one card. It puts the MLP matrices in bfp4 at one phase, the rest in bfp8 at two, and a
  # draft's output layer in bfp4
  experts = llama.precision(big, "experts")
  keys = (("gate_up", 31), ("down", 0), ("qkv", 5), ("head", None), ("draft head", None), ("kv", None))
  assert [experts[k] for k in keys] == [("bfp4", 1), ("bfp4", 1), ("bfp8", 2), ("bfp8", 2), ("bfp4", 1), ("bfp8", None)]
  assert llama.weight_format("layers.3.shared_expert.down_proj", experts) == llama.weight_format("layers.3.experts.down", experts) == "bfp4"
  assert llama.weight_format("embed.T draft", experts) == "bfp4" and ("draft head", None) not in llama.precision(big, "tt")
  rt, rng = Runtime(chip), np.random.default_rng(5)
  c, w = smol(rng, 2)
  c.precision = llama.precision(c, "tt")
  weights = {k: llama.Weight(lambda v=v: (np.float32(v).view(np.uint32) >> 16).astype(np.uint16)) for k, v in w.items()}
  t = llama.upload(rt, weights, c.precision)
  rt.wait()  # the DRAM that upload freed comes back
  reached, held = max(a + n for a, n in rt.used.items()), sum(rt.used.values())  # bytes in each DRAM bank
  assert reached - held <= -(-max(x.tiles for x in t.values()) // 8) * 2048, "the DRAM of a cast weight's bf16 copy wasn't reused"
  formats = {k: x.dtype for k, x in t.items() if "proj" in k or k == "embed.T"}
  assert formats == {k: "bfp4" if "gate_up" in k else "bfp8" for k in t if "proj" in k or k == "embed.T"}
  sizes = SimpleNamespace(**vars(c), vocab=256, places=64, ffn=1536)  # benchmark.weights, from the sizes alone, must match upload
  assert {k: (x.rows, x.cols, x.dtype) for k, x in t.items()} == {k: (x.rows, x.cols, x.dtype) for k, x in benchmark.weights(sizes).items()}
  arrays = llama.arrays(weights, c.precision)  # numpy rounds to the formats the way the chip casts (test_bfp)
  kv, np_kv = llama.kv_cache(c, 32, llama.zeros(rt, "bfp8")), llama.kv_cache(c, 32, np.zeros)  # numpy's cache rounds like a bfp8 one
  assert kv[0].dtype == "bfp8"
  on = lambda row: Tensor.of(rt, [row], "int32")
  for ids, places in (([7, 200, 31], [0, 1, 2]), ([5], [3])):  # the prompt, then a token decoded
    got, want = llama.logits(llama.forward(on(ids), on(places), t, c, kv)), llama.forward(ids, places, arrays, c, np_kv)[0]
    assert got.argmax() == want.argmax() and np.abs(got - want).max() < 0.01 * np.ptp(want), (np.abs(got - want).max(), np.ptp(want))

def test_token_accuracy(chip):
  # llama.accuracy is Tenstorrent's token accuracy, which B2 uses. Each step is fed the reference's next token, so the
  # model's picks don't steer the run. A reference that's wrong at one position costs top-1 that step.
  rt, rng = Runtime(chip), np.random.default_rng(6)
  c, w = smol(rng, 2)
  t = {k: Tensor.of(rt, v.tolist()) for k, v in w.items()}
  tokens = [int(i) for i in rng.integers(0, 256, 44)]
  prompt, after, top5, kv = tokens[:36], tokens[36:], [], llama.kv_cache(c, 64, np.zeros)
  for i in range(8):
    ids, places = (prompt, range(36)) if i == 0 else ([after[i - 1]], [35 + i])
    top5.append([int(k) for k in np.argsort(-llama.forward(ids, places, w, c, kv)[0])[:5]])
  assert llama.accuracy(rt, t, c, prompt, after, top5, 8)[:2] == (1, 1)
  top5[3] = top5[3][1:] + top5[3][:1]  # move its most likely to last, so it's still in the top 5
  top1, five, picks = llama.accuracy(rt, t, c, prompt, after, top5, 8)
  assert (top1, five) == (7 / 8, 1) and picks[3] == top5[3][-1], (top1, five, picks)
  assert llama.accuracy(None, w, c, prompt, after, top5, 8) == (top1, five, picks), "numpy (rt None) should give the same picks and scores"
  for steps in (0, 9):  # a step past the reference's predictions has nothing to score against
    with pytest.raises(ValueError, match="1 to 8 steps"): llama.accuracy(None, w, c, prompt, after, top5, steps)

def test_benchmark_quick_accuracy():
  # --quick decodes after a short prompt to save time, but token accuracy has to be scored from the prompt the
  # reference saw, since its top five came from that context. Scored against its own top five, a model gets 8 of 8.
  rng = np.random.default_rng(6)
  c, w = smol(rng, 2, places=96)
  w |= {k: v * 100 for k, v in w.items() if k.endswith("o_proj")}  # so attention, and with it the context, moves the logits
  tokens = [int(i) for i in rng.integers(0, 256, 76)]
  prompt, after = tokens[:68], tokens[68:]
  logits = llama.forward(prompt + after[:7], list(range(75)), w, c, llama.kv_cache(c, 96, np.zeros), every=True)[67:]
  top5 = [[int(k) for k in np.argsort(-row)[:5]] for row in logits]
  short, scored = benchmark.prompts(prompt, benchmark.SIZES[True])
  assert len(short) == 32 and llama.accuracy(None, w, c, scored, after, top5, 8)[:2] == (1, 1)
  assert llama.accuracy(None, w, c, short, after, top5, 8)[:2] != (1, 1), "the short prompt should score worse"

def test_benchmark_bytes():
  # The bytes a decode token reads, which B2's effective DRAM bandwidth comes from. Llama-3.1-8B reads 6.20 GB a token,
  # so 41.3 tokens/s is half the card's 512 GB/s.
  c = llama.model(dim=4096, heads=32, kv=8, layers=32, vocab=128256, ffn=14336, places=736)
  c.precision = llama.precision(c, "tt")
  t = benchmark.weights(c)
  names = ("layers.0.gate_up_proj", "layers.31.gate_up_proj", "layers.0.down_proj", "embed.T", "embed")
  assert [t[k].dtype for k in names] == ["bfp4", "bfp8", "bfp8", "bfp8", "bf16"]
  read = benchmark.bytes_per_token(t, c, 612.5)
  assert round(read / 1e9, 2) == 6.20 and round(41.3 * read / 512e9, 3) == 0.5, read

def test_batched_decode(chip):
  # Four sequences with prompts of different lengths decode together, each checked against numpy decoding it alone.
  # Their KV caches sit side by side in a shared one, and tokens read only their own rows.
  rt, rng = Runtime(chip), np.random.default_rng(19)
  c, w = smol(rng, 2)
  t, size, prompts = {k: Tensor.of(rt, v.tolist()) for k, v in w.items()}, 32, [[7, 200, 31], [5, 64], [9, 9, 9, 9, 9], [100]]
  kv = llama.kv_cache(c, size * len(prompts), lambda shape: Tensor.of(rt, np.zeros(shape).tolist()))
  np_kv = [llama.kv_cache(c, size, np.zeros) for _ in prompts]
  on = lambda row: Tensor.of(rt, [row], "int32")
  step, ids = tensor.jit(lambda ids, places: llama.forward(ids, places, t, c, kv, len(prompts))), [list(p) for p in prompts]
  for u, p in enumerate(prompts):  # prefill each prompt into its slot's rows. Its last token's logits give the next token
    got = gpt2.values(llama.forward(on(p), on(list(range(len(p)))), t, c, llama.slot(c, kv, u, len(prompts))))[0]
    want = llama.forward(p, range(len(p)), w, c, np_kv[u])[0]
    assert np.abs(got - want).max() < 0.03 * np.abs(want).max(), (u, np.abs(got - want).max())
    ids[u].append(int(want.argmax()))
  for i in range(4):
    places = [len(s) - 1 for s in ids]
    got = gpt2.values(step(on([s[-1] for s in ids]), on(places)))
    for u, s in enumerate(ids):
      want = llama.forward(s[-1:], [places[u]], w, c, np_kv[u])[0]
      assert np.abs(got[u] - want).max() < 0.03 * np.abs(want).max(), (i, u, np.abs(got[u] - want).max())
      s.append(int(want.argmax()))

@pytest.mark.skipif(pci.simulator() is None, reason="the server opens the device itself, and a card can't be open in two processes")
def test_serve_command(tmp_path):
  # python3 -m baretile.models.serve serves a model folder, which is what fetch makes of a Hugging Face ID, as in the
  # README. A small random Llama answers a completion over HTTP.
  import socket, subprocess, urllib.request
  rng, dim, ffn, vocab = np.random.default_rng(5), 64, 128, 256
  w = lambda *shape: (rng.normal(0, 0.1, shape).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
  weights = {"model.embed_tokens.weight": w(vocab, dim), "lm_head.weight": w(vocab, dim), "model.norm.weight": w(dim),
             **{f"model.layers.0.{n}.weight": w(dim) for n in ("input_layernorm", "post_attention_layernorm")},
             **{f"model.layers.0.self_attn.{n}_proj.weight": w(r, dim) for n, r in (("q", dim), ("k", 32), ("v", 32), ("o", dim))},
             **{f"model.layers.0.mlp.{n}_proj.weight": w(*rc) for n, rc in (("gate", (ffn, dim)), ("up", (ffn, dim)), ("down", (dim, ffn)))}}
  safetensors(tmp_path / "model.safetensors", {k: ("BF16", v) for k, v in weights.items()})
  (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama", "hidden_size": dim, "num_attention_heads": 2, "num_key_value_heads": 1,
                                                    "num_hidden_layers": 1, "rms_norm_eps": 1e-5, "rope_theta": 1e4, "max_position_embeddings": 64,
                                                    "vocab_size": vocab, "intermediate_size": ffn}))
  bytes_tokenizer(tmp_path)
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
  root = pathlib.Path(__file__).parent.parent
  server = subprocess.Popen([sys.executable, "-m", "baretile.models.serve", str(tmp_path), str(port), "1", "64"], cwd=root, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ | {"PYTHONPATH": str(root)})
  try:
    for line in server.stdout:  # it says when it's ready
      if f"on port {port}" in line: break
    else: pytest.fail("the server stopped before it was ready")
    ask = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps({"prompt": "Hi", "max_tokens": 3, "temperature": 0}).encode())
    reply = json.load(urllib.request.urlopen(ask, timeout=600))
    assert reply["choices"][0]["finish_reason"] == "length" and reply["usage"]["completion_tokens"] == 3, reply
  finally:
    server.kill()
    server.wait()

def test_serve_timeout(tmp_path):
  # A non-streaming request that receives no text within wait seconds must get HTTP 504 and release its slot.
  # The stand-in engine accepts requests without answering.
  import queue, urllib.error, urllib.request
  class Hung:
    tok, places, waiting, room = bytes_tokenizer(tmp_path), 64, queue.Queue(), threading.BoundedSemaphore(1)
    lock, failed, busy = threading.Lock(), False, [None]
    offer = serve.Engine.offer
  hung = Hung()
  http = serve.server(hung, "hung", port=0, wait=0.2)
  threading.Thread(target=http.serve_forever, daemon=True).start()
  with pytest.raises(urllib.error.HTTPError) as e:
    urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{http.server_address[1]}/v1/completions",
                                                  json.dumps({"prompt": "Hi"}).encode(), {"Content-Type": "application/json"}), timeout=60)
  http.shutdown()
  assert e.value.code == 504 and hung.waiting.get_nowait().gone, "the engine should drop a request that timed out"

def test_serve_hangup(tmp_path):
  # A client that hangs up before its answer is done lets its slot go within a second, streamed or not. The stand-in
  # engine sends a piece every 10 ms until the request is gone.
  import socket
  class Chatty:
    tok, places, busy, made = bytes_tokenizer(tmp_path), 64, [None], []
    def offer(self, r):
      self.made.append(r)
      def run():
        while not r.gone: r.out.put("."), time.sleep(0.01)
      threading.Thread(target=run, daemon=True).start()
      return True
  chatty = Chatty()
  http = serve.server(chatty, "chatty", port=0)
  threading.Thread(target=http.serve_forever, daemon=True).start()
  try:
    for stream in (False, True):
      body = json.dumps({"prompt": "Hi", "max_tokens": 1000, "stream": stream}).encode()
      with socket.create_connection(("127.0.0.1", http.server_address[1]), timeout=5) as s:
        s.sendall(b"POST /v1/completions HTTP/1.0\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body))
        until(lambda: len(chatty.made) == 1 + stream, "the request reached the engine")
        time.sleep(0.1)
      until(lambda: chatty.made[-1].gone, "the request whose client hung up should end", 2)
  finally: http.shutdown(); http.server_close()

def bytes_tokenizer(folder):  # writes a byte-level vocabulary of the 256 bytes to tokenizer.json in folder
  bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
  chars = map(chr, bs + [256 + i for i in range(256 - len(bs))])
  vocab = dict(zip(chars, bs + [b for b in range(256) if b not in bs]))
  (folder / "tokenizer.json").write_text(json.dumps({"model": {"vocab": vocab}, "added_tokens": []}))
  return llama.Tokenizer(str(folder))

def test_heads(chip):
  # serve.heads finds each row's 64 largest logits on the chip, in numpy's stable argsort order, and the log of the sum
  # of exp((x - max(x)) t) over the row. Qwen3.6's vocabulary splits into 97 parts of 2,560. Llama 3's doesn't split into parts of
  # 96 tiles or fewer, so it takes 24 parts of 5,344, as big as topk's L1 holds. serve.Head.of must agree on the host.
  rt, rng = Runtime(chip), np.random.default_rng(3)
  x = np.float32(rng.integers(-3, 4, (40, 300))) * np.float32(rng.choice([1, 0.5], (40, 300)))  # many ties, -0 among them
  x[x == 0] = np.where(rng.random((x == 0).sum()) < 0.5, -0.0, 0.0)
  assert all(np.array_equal(serve.Head.of(row, 1.0, k).ids, np.argsort(-row, kind="stable")[:k]) for row, k in zip(x, range(1, 81, 2)))
  for vocab in (512, 128256, 248320):
    x, ts = rng.normal(0, 3, (3, vocab)).astype(np.float32), [1.0, 1 / 0.7, 2.0]
    x[1, 5] = x[1, vocab - 7] = x[1].max() + 1
    logits = Tensor.frombytes(rt, 3, vocab, x.tobytes(), dtype="fp32")
    values, ids, lse = serve.heads(logits, Tensor.of(rt, [[t] for t in ts], "fp32"))
    want, scaled = np.argsort(-x, -1, kind="stable")[:, :64], np.float64(x) * np.array(ts)[:, None]
    assert np.array_equal(gpt2.values(ids), want) and np.array_equal(gpt2.values(values), np.take_along_axis(x, want, -1)), vocab
    exact = np.log(np.exp(scaled - scaled.max(1, keepdims=True)).sum(1))
    assert np.allclose([serve.Head.of(row, t).lse for row, t in zip(x, ts)], exact, rtol=1e-13, atol=1e-13)
    # Relative lse can approach zero. An absolute error of 1e-6 means about one ppm in the normalization mass.
    assert np.allclose(gpt2.values(lse)[:, 0], exact, rtol=1e-6, atol=1e-6), (gpt2.values(lse)[:, 0], exact)
    assert np.array_equal(gpt2.values(serve.heads(logits, k=1)[1])[:, 0], want[:, 0])

def test_sampling():
  # Request.pick samples from the head of a row (serve.Head), checked against a full sort of the vocabulary. top_p on a
  # flat distribution reaches past the head into the rest of the row. At temperature 0 the penalties match vLLM's,
  # and repetition divides a positive logit.
  rng = np.random.default_rng(5)
  for logits in (rng.normal(0, 2, 5000), rng.normal(0, 1e-3, 5000)):  # the second has a nucleus of more than 1,024 tokens
    p, order, head = np.exp(logits - logits.max()), np.argsort(-logits), serve.Head.of(logits)
    nucleus = set(order[:np.searchsorted(np.cumsum(p[order]) / p.sum(), 0.9) + 1].tolist())
    for keep, kw in ((nucleus, dict(top_p=0.9)), (set(order[:3].tolist()), dict(top_k=3)), (set(np.flatnonzero(p >= 0.2).tolist()), dict(min_p=0.2))):
      r = serve.Request(None, [], seed=1, **kw)
      got = {r.pick(head) for _ in range(3000 if len(keep) < 1000 else 20000)}
      assert got <= keep and (len(got) > 0.9 * len(keep) if len(keep) > 1000 else got == keep), (kw, len(got), len(keep))
  logits = rng.normal(0, 1.5, 400)  # at temperature 0.8, a fifth of the draws fall past the head's 64
  p, r = np.exp((logits - logits.max()) / 0.8), serve.Request(None, [], temperature=0.8, seed=2)
  p, head, tail = p / p.sum(), serve.Head.of(logits, 1 / 0.8), np.argsort(-logits)[64:]
  share = np.bincount([r.pick(head) for _ in range(20000)], minlength=400) / 20000
  assert 0.15 < p[tail].sum() and abs(share[tail].sum() - p[tail].sum()) < 0.015 and np.all(np.abs(share - p) < 5 * np.sqrt(p / 20000) + 1e-3)
  logits = np.array([3.0, 2.5, 1.0, -1.0])
  for made, prompt, kw, want in (([0], [], dict(presence=1.0), 1), ([0, 0], [], dict(frequency=0.4), 1), ([0], [], dict(frequency=0.4), 0),
                                 ([], [0], dict(repetition=1.3), 1), ([], [0], dict(repetition=1.1), 0), ([], [3], dict(repetition=10.0), 0)):
    r = serve.Request(None, prompt, temperature=0, **kw)
    r.made = made
    assert r.pick(serve.Head.of(logits)) == want, (made, prompt, kw)

def test_request_inputs(tmp_path):
  for key, values in {"max_tokens": (0, -1, 1.5, True), "temperature": (-1, math.inf, math.nan, 10**400, "1"), "top_p": (0, 1.1),
                      "min_p": (-0.1, 1.1), "top_k": (-1, 1.5), "repetition": (0, -1), "presence": (math.nan,),
                      "frequency": (math.inf,), "logprobs": (-1, 21, 1.5), "stop": ([""], [3])}.items():
    for v in values:
      with pytest.raises(ValueError): serve.Request(None, [1], **{key: v})
  r = serve.Request(bytes_tokenizer(tmp_path), [1], temperature=0, logprobs=20)
  r.take(np.array([0., 1., 2.]), True)
  assert r.made == [2] and [t for t, _ in r.probs[0][2]] == [2, 1, 0], "logprobs can exceed a small vocabulary"

def test_sampling_boundaries(chip):
  # Small finite temperatures must not overflow either the inverse temperature or the device normalization.
  for kw, name in (({"temperature": 1e-320}, "temperature"), ({"temperature": 1e300}, "temperature"),
                   ({"repetition": 1e-320}, "repetition_penalty"), ({"frequency": 1e300}, "frequency_penalty")):
    with pytest.raises(ValueError, match=name): serve.Request(None, [], **kw)
  rt = Runtime(chip)
  logits = np.full((1, 512), 1e20, np.float32)
  temperature = 1e-30
  values, ids, lse = (gpt2.values(x)[0] for x in serve.heads(Tensor.frombytes(rt, 1, 512, logits.tobytes(), dtype="fp32"),
                                                         Tensor.of(rt, [[1 / temperature]], "fp32")))
  assert np.allclose(lse, math.log(512), rtol=1e-6), lse
  for head in (serve.Head(values, ids, lse[0], lambda: logits[0]), serve.Head.of(logits[0], 1 / temperature)):
    r = serve.Request(None, [], temperature=temperature, min_p=0.1, seed=1)
    assert len({r.pick(head) for _ in range(100)}) > 64, "equal logits retain all candidates at a small temperature"
  r = serve.Request(None, [], temperature=temperature, min_p=0.1)
  assert r.draw(np.array([-np.finfo(np.float32).max, np.finfo(np.float32).max])) == 1

def host_engine(slots=1):  # the slots of an Engine, with what offer and take use. Its requests save no session
  import queue
  engine = object.__new__(serve.Engine)
  engine.lock, engine.failed, engine.room = threading.Lock(), False, threading.BoundedSemaphore(slots)
  engine.waiting, engine.busy, engine.held, engine.places = queue.Queue(), [None] * slots, [[1] for _ in range(slots)], 4096
  engine.save = lambda u: None
  def seat(r, u=0):  # offers r and puts it in slot u, as admit would
    assert engine.offer(r)
    engine.busy[u] = engine.waiting.get_nowait()
    return r
  engine.seat = seat
  return engine

def test_request_sampling_failure(tmp_path):
  # A bad sampling row releases only its request. Other slots and later offers still work.
  tok, engine = bytes_tokenizer(tmp_path), host_engine(2)
  first, second = [engine.seat(serve.Request(tok, [u + 1], max_tokens=1, temperature=0), u) for u in range(2)]
  assert engine.take(0, np.array([np.nan, 1.]))
  assert first.finish == "error" and first.error and first.room is None and first.out.get_nowait() is None
  assert engine.take(1, np.array([0., 1.])) and second.made == [1] and second.finish == "length"
  later = engine.seat(serve.Request(tok, [3], max_tokens=1, temperature=0))
  assert not engine.failed and engine.take(0, np.array([1., 0.])) and later.made == [0]

def test_request_host_failures(tmp_path):
  # Request.take runs on the host, apart from a Head reading its row from the chip. A failure of the request's own work,
  # here its output constraint, ends that request alone, whatever its type. A failed read, as when the chip hangs or its
  # runtime closes, still stops the engine, though RecursionError is a RuntimeError too.
  tok, engine = bytes_tokenizer(tmp_path), host_engine()
  def recurse(text, pending=b""): return recurse(text, pending)
  def broken(text, pending=b""): return text.get("x")
  for shape, error in ((recurse, "RecursionError"), (broken, "AttributeError")):
    r = engine.seat(serve.Request(tok, [1], temperature=0, shape=shape))
    assert engine.take(0, np.array([0., 1.])) and engine.busy == [None] and not engine.failed
    assert r.finish == "error" and r.error.startswith(error) and r.room is None and r.out.get_nowait() is None
  for failure in (TimeoutError, RuntimeError):
    def hung(): raise failure("the chip failed")
    r = engine.seat(serve.Request(tok, [1], temperature=0, presence=1.0))  # a penalty reads the whole row
    with pytest.raises(failure): engine.take(0, serve.Head([1.0], [1], 0.0, hung))
    r.end(); engine.busy[0] = None

def test_json_nesting_limit(tmp_path):
  # Arrays and objects nest at most serve.NESTING deep in constrained JSON. A model that keeps opening brackets ends its
  # request with an error instead of recursing past Python's limit in the engine thread. Deeper schemas and constants
  # are rejected as the request is parsed.
  n, tok = serve.NESTING, bytes_tokenizer(tmp_path)
  fits = serve.formatted({"type": "json_object"})
  assert fits('{"a": ' + "[" * (n - 1)) == 1 and fits('{"a": ' + "[" * n) == 0 and fits('{"a": ' + "[" * (n - 1) + "]" * (n - 1) + "}") == 2
  schema = lambda sc: serve.formatted({"type": "json_schema", "json_schema": {"schema": sc}})
  deep, nested = {}, 1
  for _ in range(n - 1): deep = {"type": "array", "items": deep}
  for _ in range(n): nested = [nested]
  assert schema(deep)("[" * n) == 1 and schema(deep)("[" * (n + 1)) == 0 and schema({"const": nested})(json.dumps(nested)) == 2
  for sc in ({"items": deep}, {"const": [nested]}, {"type": "array", "items": {"enum": [nested]}}):
    with pytest.raises(ValueError, match="nest"): schema(sc)
  engine = host_engine()
  r = engine.seat(serve.Request(tok, [1], max_tokens=4000, temperature=0, eos=[0], shape=fits))
  for c in '{"a": ' + "[" * 3000:
    logits = np.full(256, -np.inf)
    logits[ord(c)] = 1
    if engine.take(0, logits): break
  assert r.finish == "error" and "output constraint" in r.error and not engine.failed and engine.busy == [None]
  assert tok.decode(r.made) == '{"a": ' + "[" * (n - 1)

def received(s):  # all that a socket gets until the server closes it
  out = b""
  while chunk := s.recv(4096): out += chunk
  return out

def until(done, what, wait=5):  # polls done, and fails the test with what after wait seconds
  end = time.monotonic() + wait
  while not done():
    if time.monotonic() > end: pytest.fail(what)
    time.sleep(0.001)

def refused(reply, code):  # the message of a connection-time refusal with status code, whose body is a JSON error
  head, _, body = reply.partition(b"\r\n\r\n")
  error = json.loads(body)["error"]
  assert head.startswith(b"HTTP/1.0 %d" % code) and b"Content-Type: application/json" in head and error["code"] == code, reply
  return error["message"]

def test_http_input_limits(tmp_path):
  # A request must arrive within input_wait, with one Content-Length and at most body_limit bytes. Otherwise it gets
  # 408, 400 or 413 before it reaches the engine. The deadline covers headers too, and a client that trickles bytes.
  import socket
  engine = SimpleNamespace(tok=bytes_tokenizer(tmp_path), places=64)
  http = serve.server(engine, "limits", port=0, body_limit=64, input_wait=0.3, connections=1)
  thread = threading.Thread(target=http.serve_forever, daemon=True)
  thread.start()
  connect = lambda: socket.create_connection(("127.0.0.1", http.server_address[1]), timeout=2)
  prefix = b"POST /v1/completions HTTP/1.0\r\n"
  try:
    for headers in (b"Content-Length: -1", b"Content-Length: bad", b"Content-Length: 1\r\nContent-Length: 1",
                    b"Content-Length: 1\r\nTransfer-Encoding: chunked", b"Content-Length: 65"):
      with connect() as s:
        s.sendall(prefix + headers + b"\r\n\r\n")
        assert received(s).startswith(b"HTTP/1.0 " + (b"413" if headers.endswith(b"65") else b"400")), headers
    # A client that trickles bytes still gets 408 when the deadline passes.
    with connect() as slow:
      slow.sendall(prefix + b"Content-Length: 64\r\n\r\n{")
      start = time.monotonic()
      for _ in range(6):
        time.sleep(0.06)
        try: slow.sendall(b" ")
        except OSError: break
      assert received(slow).startswith(b"HTTP/1.0 408") and time.monotonic() - start < 1.5
    for request in (b"GET /v1/models HTTP/1.0\r\nX-Slow: ", prefix + b"Content-Type: application/json\r\nX-Slow: "):
      with connect() as s:
        s.sendall(request)
        assert received(s).startswith(b"HTTP/1.0 408"), request
    # One unfinished request takes the only handler. Another connection gets a 503 at once that names the limit.
    with connect() as held:
      held.sendall(prefix + b"Content-Length: 64\r\n\r\n{")
      until(lambda: http.clients["127.0.0.1"] == 1, "the request did not take a handler", 1)
      with connect() as extra: assert "connection limit (1)" in refused(received(extra), 503)
      assert received(held).startswith(b"HTTP/1.0 408")
    with connect() as s:
      s.sendall(b"GET /v1/models HTTP/1.0\r\n\r\n")
      assert received(s).startswith(b"HTTP/1.0 200"), "expired requests release their handlers"
  finally:
    http.shutdown(); http.server_close(); thread.join(timeout=2)
    assert not thread.is_alive()

def test_http_connection_limits(tmp_path):
  # The connection limits follow the engine's slots, so one client can fill them all, but one address can't hold more
  # than a quarter of the handlers. Its idle sockets get 429 past that, and other clients are still served.
  import socket
  tok = bytes_tokenizer(tmp_path)
  wide = serve.server(SimpleNamespace(tok=tok, places=64, busy=[None] * 40), "wide", port=0)
  assert (wide.connections, wide.per_client) == (192, 48), "4 x (40 slots + 8) connections, a quarter of them per address"
  wide.server_close()
  with pytest.raises(ValueError, match="per_client"): serve.server(SimpleNamespace(tok=tok, places=64, busy=[None]), "bad", port=0, per_client=0)
  http = serve.server(SimpleNamespace(tok=tok, places=64, busy=[None] * 2), "limits", port=0, input_wait=5)
  thread = threading.Thread(target=http.serve_forever, daemon=True)
  thread.start()
  def connect(client):
    try: return socket.create_connection(("127.0.0.1", http.server_address[1]), timeout=5, source_address=(client, 0))
    except OSError as e: pytest.skip(f"no loopback connection from {client}: {e}")
  idle = []
  try:
    for k in range(http.per_client):  # each holds a handler that waits for headers
      idle.append(connect("127.0.0.2"))
      until(lambda: http.clients["127.0.0.2"] == k + 1, "the server did not take the connection")
    with connect("127.0.0.2") as extra: assert f"address's connection limit ({http.per_client})" in refused(received(extra), 429)
    with connect("127.0.0.3") as other:
      other.sendall(b"GET /v1/models HTTP/1.0\r\n\r\n")
      assert received(other).startswith(b"HTTP/1.0 200")
    for s in idle: s.close()
    until(lambda: not http.clients, "closed connections must release their handlers")
  finally:
    for s in idle: s.close()
    http.shutdown(); http.server_close(); thread.join(timeout=2)
    assert not thread.is_alive()

def test_download_redirect_credentials(tmp_path, monkeypatch):
  import http.server, urllib.request
  received = []
  class Sink(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
      received.append((self.path, self.headers.get("Authorization")))
      data = b"{}"
      self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def log_message(self, *args): pass
  other = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Sink)
  class Hub(Sink):
    def do_GET(self):
      if self.path == "/same": return super().do_GET()
      self.send_response(302)
      self.send_header("Location", "/same" if self.path.endswith("config.json") else f"http://127.0.0.1:{other.server_port}/other")
      self.end_headers()
  hub = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Hub)
  threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (hub, other)]
  for t in threads: t.start()
  monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{hub.server_port}")
  monkeypatch.setenv("HF_TOKEN", "fake-test-token")
  try:
    # Exercise fetch's opener, then stop before its synthetic index is parsed.
    with pytest.raises(KeyError, match="weight_map"): serve.fetch("org/model", str(tmp_path))
    assert received and all(token == ("Bearer fake-test-token" if path == "/same" else None) for path, token in received)
    assert {path for path, _ in received} == {"/same", "/other"}
    redirect = serve.ScopedRedirect()
    req = urllib.request.Request("https://example.test/a", headers={"Authorization": "Bearer fake"})
    for url, want in (("https://example.test:443/b", "Bearer fake"), ("http://example.test/b", None),
                      ("https://example.test:444/b", None), ("https://other.test/b", None)):
      assert redirect.redirect_request(req, None, 302, "", {}, url).get_header("Authorization") == want
  finally:
    for s in (hub, other): s.shutdown(); s.server_close()
    for t in threads: t.join(timeout=2); assert not t.is_alive()

def test_tokenizer_without_specials(tmp_path):
  bytes_tokenizer(tmp_path)
  data = json.loads((tmp_path / "tokenizer.json").read_text())
  data["model"]["vocab"] |= {"th": 256, "the": 257}
  (tmp_path / "tokenizer.json").write_text(json.dumps(data))
  tok = llama.Tokenizer(str(tmp_path))
  assert tok.encode("the", special=True) == tok.encode("the") == [257]

def test_serve_bad_requests(tmp_path):
  import urllib.error, urllib.request
  class Stand:
    tok, places, failed, busy = bytes_tokenizer(tmp_path), 64, False, [None]
    def offer(self, r):
      if self.failed: raise RuntimeError("engine failed")
      self.last, r.finish = r, "stop"
      r.end()
      return True
  stand = Stand()
  http = serve.server(stand, "stand", "<|im_start|>", port=0, wait=0.1)
  threading.Thread(target=http.serve_forever, daemon=True).start()
  def ask(body, chat=False):
    return urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{http.server_address[1]}/v1/" +
                                  ("chat/completions" if chat else "completions"), json.dumps(body).encode()), timeout=5)
  try:
    for body in ([], None, {"prompt": ""}, {"prompt": []}, *({"prompt": "Hi", k: v} for k, v in
                 (("temperature", -1), ("temperature", math.nan), ("max_tokens", 0), ("top_k", 1.5), ("stop", [4])))):
      with pytest.raises(urllib.error.HTTPError) as e: ask(body)
      assert e.value.code == 400 and json.load(e.value)["error"]["message"]
    hi = lambda content: {"messages": [{"role": "user", "content": content}]}
    for body in ({"messages": [None]}, hi(3), hi([{"type": "image_url", "image_url": {"url": "x"}}]), hi("Hi") | {"max_completion_tokens": 0}):
      with pytest.raises(urllib.error.HTTPError) as e: ask(body, chat=True)
      assert e.value.code == 400, body
    ask(hi("Hi") | {"max_tokens": 9, "max_completion_tokens": 3}, chat=True)
    ids = stand.last.ids
    assert stand.last.max_tokens == 3, "max_completion_tokens, OpenAI's chat name, comes first"
    ask(hi([{"type": "text", "text": "H"}, {"type": "text", "text": "i"}]), chat=True)
    assert stand.last.ids == ids, "text parts join into the message's text"
    assert json.load(ask({"prompt": "Hi"}))["choices"][0]["finish_reason"] == "stop"
    stand.failed = True
    with pytest.raises(urllib.error.HTTPError) as e: ask({"prompt": "Hi"})
    assert e.value.code == 503
  finally: http.shutdown(); http.server_close()

def test_tool_definitions(tmp_path):
  # Tool definitions that calls and chosen can't read get HTTP 400 as the request is parsed, before any reaches the
  # engine thread. So does a body nested too deeply for the JSON parser. Boolean property schemas and null parameters work.
  import urllib.error, urllib.request
  class Stand:  # answers every request with a call to f
    tok, places, busy, offered = bytes_tokenizer(tmp_path), 4096, [None], 0
    def offer(self, r):
      self.offered += 1
      r.out.put("<tool_call>\n<function=f>\n<parameter=x>\n[1]\n</parameter>\n</function>\n</tool_call>")
      r.finish = "stop"
      r.end()
      return True
  stand = Stand()
  http = serve.server(stand, "qwen", "<|im_start|> <tool_call>\n<function=", port=0)
  threading.Thread(target=http.serve_forever, daemon=True).start()
  def ask(body):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    return urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{http.server_address[1]}/v1/chat/completions", data), timeout=10)
  tool = lambda **kw: [{"type": "function", "function": {"name": "f"} | kw}]
  messages = [{"role": "user", "content": "Hi"}]
  try:
    for tools in ({}, "f", [None], [{"function": "f"}], [{"function": {"name": ""}}], tool(parameters="oops"), tool(parameters=[]),
                  tool(parameters={"properties": "x"}), tool(parameters={"properties": {"x": "string"}})):
      for choice in ("auto", "required"):
        with pytest.raises(urllib.error.HTTPError) as e: ask({"messages": messages, "tools": tools, "tool_choice": choice})
        assert e.value.code == 400 and json.load(e.value)["error"]["message"], (tools, choice)
    with pytest.raises(urllib.error.HTTPError) as e: ask(b'{"messages": ' + b"[" * 100000 + b"]" * 100000 + b"}")
    assert e.value.code == 400 and stand.offered == 0
    for tools, x in ((tool(parameters={"type": "object", "properties": {"x": True}}), "[1]"), (tool(parameters=None), "[1]"),
                     (tool(parameters={"properties": {"x": {"type": "array"}}}), [1])):
      made = json.load(ask({"messages": messages, "tools": tools, "tool_choice": "required"}))["choices"][0]["message"]["tool_calls"]
      assert [json.loads(c["function"]["arguments"]) for c in made] == [{"x": x}], tools
  finally: http.shutdown(); http.server_close()

def test_engine_failure():
  # A failure while admitting a request must release it and queued requests. Future offers fail immediately.
  import queue
  engine = object.__new__(serve.Engine)
  engine.lock, engine.failed, engine.room, engine.places = threading.Lock(), False, threading.BoundedSemaphore(2), 64
  engine.waiting, engine.busy = queue.Queue(), [None, None]
  for ids in ([], list(range(63))):
    with pytest.raises(ValueError, match="1 to 62 tokens"): engine.offer(serve.Request(None, ids))
  def fail(r): raise RuntimeError("admission failed")
  engine.admit = fail
  requests = [serve.Request(None, [1]), serve.Request(None, [2])]
  for r in requests: assert engine.offer(r)
  with pytest.raises(RuntimeError, match="admission failed"): engine.run()
  for r in requests: assert r.finish == "error" and r.out.get_nowait() is None and r.room is None
  assert engine.waiting.empty()
  with pytest.raises(RuntimeError, match="engine stopped"): engine.offer(serve.Request(None, [3]))

def test_padded_vocabulary(tmp_path):
  # A model's logits can cover more ids than its tokenizer has, as Qwen3's and gpt-oss's do. Those ids never fit a
  # constrained answer, and drawing one unconstrained adds no text.
  tok = bytes_tokenizer(tmp_path)
  r = serve.Request(tok, [], 20, 0, eos=[0], shape=serve.formatted({"type": "json_object"}))
  for c in '{"a": 1}\0':  # each step the padding outranks the next character, and at the end the EOS
    logits = np.zeros(300, np.float32)
    logits[256:], logits[ord(c)] = 5, 4
    if r.take(logits, True): break
  assert tok.decode(r.made) == '{"a": 1}' and r.finish == "stop"
  r = serve.Request(tok, [], 1, 0, logprobs=1)
  logits = np.zeros(300, np.float32)
  logits[299] = 1
  r.take(logits, True)
  assert r.made == [299] and r.text == "" and r.finish == "length"

def test_shapes(tmp_path):
  # Request.pick restricts output to the tool call or JSON schema requested. Logits that favor a greeting must still
  # produce a tool call when required; random logits must produce valid JSON under the schema.
  tok = bytes_tokenizer(tmp_path)
  tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
  call = "<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>"
  def spell(text, n, choice):  # greedy: each step favors the text's next character (or the end, 0), then "H"
    r = serve.Request(tok, [], n, 0, eos=[0], shape=serve.chosen(choice, tools))
    for i in range(n):
      logits = np.zeros(256, np.float32)
      logits[ord("H")], logits[ord(text[i]) if i < len(text) else 0] = 1, 2
      if r.take(logits, True): break
    return tok.decode(r.made), r.finish
  assert spell("Hi there", 40, None) == ("Hi there", "stop")
  said, finish = spell("Hi there", 40, "required")
  assert said.startswith("<tool_call>\n<function=read>") and finish == "length", said
  assert spell(call, 100, "required") == (call, "stop"), "a complete call may end"
  assert spell(call.replace("read", "rest"), 40, {"type": "function", "function": {"name": "read"}})[0].startswith("<tool_call>\n<function=read>")
  said, finish = spell(call, len(call) + 5, "none")
  assert "<tool_call>" not in said and said.startswith("<tool_callH") and finish == "stop", said
  with pytest.raises(ValueError): serve.chosen({"type": "function", "function": {"name": "write"}}, tools)
  schema = {"type": "object", "properties": {"ok": {"type": "boolean"}, "unit": {"enum": ["C", "F"]}, "n": {"type": "integer"},
                                             "tags": {"type": "array", "items": {"$ref": "#/$defs/tag"}, "maxItems": 2},
                                             "note": {"anyOf": [{"const": "hi"}, {"type": "null"}]}},
            "required": ["ok", "unit", "n", "tags", "note"], "additionalProperties": False, "$defs": {"tag": {"enum": ["a", "b"]}}}
  fits = serve.formatted({"type": "json_schema", "json_schema": {"schema": schema}})
  whole = '{"ok": true, "unit": "C", "n": -12, "tags": ["a"], "note": null}'
  assert [fits(whole[:k]) for k in (0, 1, 9, 20, 32, len(whole) - 1, len(whole))] == [1, 1, 1, 1, 1, 1, 2] and fits(whole + "\n") == 2
  for text in ('{"ok": 1', '{"unit": "K"', '{"n": 1.5', '{"n": 1.', '{"n": -e', '{"x"', '{"tags": ["a", "b", "a"]', '{"note": "ho"', whole + "x",
               "```json", "{\n  "):
    assert fits(text) == 0, text
  assert serve.formatted({"type": "json_object"})('{"a": [1, {"b": "c\\n"}]}') == 2 and serve.formatted({"type": "json_object"})("[1]") == 0
  for seed in range(3):  # with random logits most bytes don't fit, so most tokens are drawn again
    rng, r = np.random.default_rng(seed), serve.Request(tok, [], 300, 1.0, eos=[0], seed=seed, shape=fits)
    while not r.take(rng.normal(0, 1, 256).astype(np.float32), True): pass
    got = json.loads(tok.decode(r.made))
    assert r.finish == "stop" and set(got) == set(schema["properties"]) and isinstance(got["ok"], bool) and isinstance(got["n"], int)
    assert got["unit"] in ("C", "F") and set(got["tags"]) <= {"a", "b"} and len(got["tags"]) <= 2 and got["note"] in ("hi", None), got

def test_schema_edges():
  def fits(schema): return serve.formatted({"type": "json_schema", "json_schema": {"schema": schema}})
  for schema in ({"$ref": "#/$defs/loop", "$defs": {"loop": {"$ref": "#/$defs/loop"}}},
                 {"type": "object", "properties": {"x": {"$ref": "#/$defs/missing"}}},
                 {"type": 3}, {"type": []}, {"anyOf": []}, {"items": 3}, {"maxItems": -1}):
    with pytest.raises(ValueError): fits(schema)
  assert fits(False)("null") == 0 and fits(True)("null") == 2
  assert fits({"type": "object", "properties": {"x": False}})('{"x": 1}') == 0
  assert fits({"type": "string", "anyOf": [True, False]})("null") == 0
  assert fits({"anyOf": [{"const": 1}, {"const": 10}]})("10") == 2
  one = fits({"oneOf": [{"type": "number"}, {"type": "integer"}]})
  assert one("1") == 1 and one("1.5") == 2 and one("1 ") == 0
  assert fits({"oneOf": [{"type": "boolean"}, {"enum": [True]}]})("true") == 0

def test_schema_intersections():
  def fits(schema): return serve.formatted({"type": "json_schema", "json_schema": {"schema": schema}})
  for schema, text, want in (({"type": "string", "enum": [1]}, "1", 0),
                             ({"enum": [1, 2], "const": 2}, "1", 0), ({"enum": [1, 2], "const": 2}, "2", 2),
                             ({"enum": [True]}, "1", 0), ({"const": 1}, "true", 0),
                             ({"type": "string", "anyOf": [{"type": "integer"}, {"const": "ok"}]}, '"ok"', 2),
                             ({"anyOf": [{"type": "number"}], "oneOf": [{"const": 1}, {"const": 2}]}, "3", 0)):
    assert fits(schema)(text) == want, (schema, text)
  schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"],
            "anyOf": [{"properties": {"y": {"type": "integer"}}, "required": ["y"]}]}
  assert fits(schema)('{"x": 1, "y": 2}') == 0 and fits(schema)('{"x": "ok", "y": 2}') == 2
  assert fits(schema)('{"x": "ok"}') == 0 and fits(schema)('{"y": 2}') == 0
  schema = {"$defs": {"base": {"type": "object", "properties": {"x": {"type": "string"}}}}, "$ref": "#/$defs/base",
            "properties": {"x": {"enum": ["a", "b"]}}, "required": ["x"]}
  assert fits(schema)('{"x": "a"}') == 2
  for text in ('{"x": 1}', '{"x": "c"}', '{}'): assert fits(schema)(text) == 0
  same = fits({"enum": [{"é": ["😀", True, 1, None], "b": 2}]})
  assert same('{"b": 2, "\\u00e9": ["\\ud83d\\ude00", true, 1, null]}') == 2
  assert same('{"b": 2, "é": ["😀", 1, 1, null]}') == 0

def test_unicode_output_constraints(tmp_path):
  tok = bytes_tokenizer(tmp_path)
  for wanted, spellings in (("é", ('"é"', '"\\u00e9"', '"\\u00E9"')),
                           ("😀", ('"😀"', '"\\ud83d\\ude00"')),
                           ("/\n\\\"", ('"\\/\\n\\\\\\\""',))):
    shape = serve.formatted({"type": "json_schema", "json_schema": {"schema": {"enum": [wanted]}}})
    for text in spellings:
      assert all(shape(text[:i]) == 1 for i in range(len(text))) and shape(text) == 2, text
      data = list(text.encode("utf-8"))
      r = serve.Request(tok, [], len(data) + 5, temperature=0, eos=[0], shape=shape)
      for token in data + [0]:
        logits = np.full(256, -np.inf)
        logits[token] = 1
        if r.take(logits, True): break
      assert r.made == data and r.finish == "stop" and json.loads(tok.decode(r.made)) == wanted
  shape = serve.formatted({"type": "json_schema", "json_schema": {"schema": {"enum": ["é"]}}})
  r = serve.Request(tok, [], temperature=0, eos=[0], shape=shape)
  assert not r.fits(0xc3), "a non-ASCII character cannot start a JSON value"
  r.made = [ord('"')]
  assert r.fits(0xc3) and not r.fits(0xe0) and not r.fits(0x80)
  r.made.append(0xc3)
  assert not r.fits(0) and r.fits(0xa9), "EOS cannot discard a pending UTF-8 character"
  r.made = [ord('"')]
  logits = np.full(256, -np.inf)
  logits[ord('Q')] = 1
  with pytest.raises(ValueError, match="output constraint"): r.take(logits, True)
  assert r.finish != "stop", "a constraint dead end is an error, not a successful empty answer"

def test_reasoning_stream_prefixes():
  texts = ("<think>Reasoning.\n</think>\n\nAnswer.", "Reasoning.\n</think>\n\nAnswer.",
           "<|channel|>analysis<|message|>Reasoning.\n<|end|><|channel|>final<|message|>Answer.", "Reasoning.<")
  for text in texts:
    seen = ("", "")
    for i in range(len(text) + 1):
      parts = serve.answer(text[:i], final=False)
      assert all(new.startswith(old) for old, new in zip(seen, parts)), (text[:i], seen, parts)
      seen = parts
    full = serve.answer(text)
    assert all(new.startswith(old) for old, new in zip(seen, full))
    if text.endswith("Answer."): assert full == ("Reasoning.", "Answer.")
    else: assert full == ("Reasoning.<", ""), "flush an unfinished marker as ordinary text when generation ends"

def test_serve(chip, tmp_path):
  # The serve.py engine, against numpy's greedy tokens. Three requests on two slots decode together. When two run
  # again, they find their slots still holding their prompts, and each prefills one token. A free slot idles on a row
  # no sequence reaches, since any other row would spoil what the slot holds. A long prompt is prefilled a chunk
  # between steps, so the others keep getting tokens. Then the HTTP side. When no slot is free, a request gets an
  # immediate 429 so the client can retry or use another provider.
  import urllib.request
  rt, rng = Runtime(chip), np.random.default_rng(23)
  c, w = smol(rng, 2)
  w = {k: bf16(v * 10) if k.endswith("_proj") else v for k, v in w.items()}  # sharp attention, so a wrong cached row changes the tokens
  t = {k: Tensor.of(rt, v.tolist()) for k, v in w.items()}
  tok = bytes_tokenizer(tmp_path)
  engine = serve.Engine(rt, t, c, tok, slots=2, places=64, chunk=16)
  threading.Thread(target=engine.run, daemon=True).start()
  def done(r):
    while r.out.get(timeout=600) is not None: pass
    return r.made
  prompts = [[7, 200, 31, 64, 5], [5, 64], list(range(40, 60))]  # the last is longer than a chunk
  together = [serve.Request(tok, p, 6, 0) for p in prompts]
  for r in together: engine.waiting.put(r)
  together = [done(r) for r in together]
  def greedy(prompt, n):  # numpy's tokens for the same prompt, as baretile.models.llama decodes
    kv, ids = llama.kv_cache(c, 64, np.zeros), list(prompt)
    logits = llama.forward(ids, list(range(len(ids))), w, c, kv)[0]
    for i in range(n - 1): ids.append(int(logits.argmax())); logits = llama.forward(ids[-1:], [len(ids) - 1], w, c, kv)[0]
    return ids[len(prompt):] + [int(logits.argmax())]
  assert together == [greedy(p, 6) for p in prompts], "requests must get numpy's greedy tokens while they decode together"
  before = engine.prefilled
  again = [serve.Request(tok, p, 6, 0) for p in prompts[:0:-1]]
  served(engine, *again)
  assert [r.made for r in again] == together[:0:-1]
  assert engine.prefilled - before == 2, "only the last token of each prompt should be prefilled, the rest are in their slots"
  class Watched(serve.Request):  # records engine.prefilled at each of its tokens, from the engine's thread
    def take(self, logits, room): self.seen = getattr(self, "seen", []) + [engine.prefilled]; return super().take(logits, room)
  r, deadline = Watched(tok, [3, 1, 4], 12, 0), time.time() + 600
  engine.waiting.put(r)
  while not getattr(r, "seen", None): assert time.time() < deadline; time.sleep(0.01)
  engine.waiting.put(long := serve.Request(tok, list(range(100, 150)), 2, 0))  # 50 tokens: four chunks
  done(r), done(long)
  steps = np.diff(r.seen)
  assert 16 in steps and steps.max() <= 16, ("at most one chunk of the long prompt between two of its tokens", r.seen)
  r = serve.Request(tok, [], 9, 0, stop=["bc"])
  said = lambda: [r.out.get_nowait() for _ in range(r.out.qsize())]
  assert not r.take(np.eye(256)[97], True) and said() == ["a"]
  assert not r.take(np.eye(256)[98], True) and said() == [], "b might begin the stop string"
  assert r.take(np.eye(256)[99], True) and said() == [] and (r.said, r.finish) == ("a", "stop"), "the engine sends its end (None)"
  r = serve.Request(tok, [], 9, 0, eos=[0])
  assert not r.take(np.eye(256)[97], True) and r.take(np.eye(256)[0], True) and (r.made, r.said, r.finish) == ([97], "a", "stop")
  r = serve.Request(tok, [], 9, 0)
  r.gone = True  # its client has gone
  assert r.take(np.eye(256)[97], True) and (r.made, r.finish) == ([], "gone") and r.out.empty()
  http = serve.server(engine, "smol", port=0, defaults={"temperature": 0})
  threading.Thread(target=http.serve_forever, daemon=True).start()
  url = f"http://127.0.0.1:{http.server_address[1]}/v1"
  post = lambda body: urllib.request.urlopen(urllib.request.Request(url + "/completions", json.dumps(body).encode(),
                                                                    {"Content-Type": "application/json"}), timeout=600)
  r = serve.Request(tok, tok.encode("Hi there"), 6, 0)
  engine.waiting.put(r)
  text = tok.decode(done(r))
  # held is what the slot already holds. leaving out temperature gets the model's default
  reply, held = json.load(post({"prompt": "Hi there", "max_tokens": 6})), len(tok.encode("Hi there")) - 1
  assert reply["choices"][0]["text"] == text and reply["choices"][0]["finish_reason"] == "length" and reply["usage"]["completion_tokens"] == 6
  assert reply["usage"]["prompt_tokens_details"]["cached_tokens"] == held, "all but the prompt's last token, which runs again"
  events, usage = streamed(post({"prompt": "Hi there", "max_tokens": 6, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}))
  assert "".join(e["text"] for e in events) == text and events[-1]["finish_reason"] == "length" and usage["completion_tokens"] == 6
  schema = {"type": "object", "properties": {"answer": {"enum": ["yes", "no"]}}, "required": ["answer"], "additionalProperties": False}
  shaped = json.load(post({"prompt": "Hi there", "max_tokens": 80, "temperature": 0,
                           "response_format": {"type": "json_schema", "json_schema": {"schema": schema}}}))["choices"][0]
  assert json.loads(shaped["text"])["answer"] in ("yes", "no") and shaped["finish_reason"] == "stop", shaped  # the schema allows no more text
  # greedy: each token is the most likely
  chances = json.load(post({"prompt": "Hi there", "max_tokens": 6, "temperature": 0, "logprobs": 2}))["choices"][0]["logprobs"]
  assert len(chances["tokens"]) == 6 and all(p == max(top.values()) <= 0 for p, top in zip(chances["token_logprobs"], chances["top_logprobs"]))
  assert json.load(urllib.request.urlopen(url + "/models", timeout=60))["data"][0]["id"] == "smol"
  # two streams take both slots, and each holds its slot from the headers on
  holding = [post({"prompt": "Hi there", "max_tokens": 20, "temperature": 0, "stream": True}) for _ in range(2)]
  with pytest.raises(urllib.error.HTTPError) as e: post({"prompt": "Hi", "max_tokens": 2})
  assert e.value.code == 429 and json.load(e.value)["error"]["code"] == 429, "with no slot free, expected an immediate 429, not a wait"
  for h in holding: assert streamed(h)[1] is None, "usage only when the request asks for it"
  assert json.load(post({"prompt": "Hi", "max_tokens": 2}))["usage"]["completion_tokens"] == 2, "the slots should be free once the streams end"
  http.shutdown()
  hi = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]
  assert serve.chat("... <|im_start|> ...", hi) == "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
  assert serve.chat("... <|start_header_id|> ...", hi[1:]) == ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
                                                               "<|start_header_id|>assistant<|end_header_id|>\n\n")
  think = serve.server(engine, "smol", "<|im_start|> <think>", port=0)  # a thinking model's template (Qwen3's)
  threading.Thread(target=think.serve_forever, daemon=True).start()
  ask = lambda body: urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{think.server_address[1]}/v1/chat/completions",
                                                                   json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600)
  r = serve.Request(tok, tok.encode(serve.chat("<|im_start|> <think>", hi[1:]), special=True), 6, 0)
  engine.waiting.put(r)
  text, body = tok.decode(done(r)), {"messages": hi[1:], "max_tokens": 6, "temperature": 0}
  message = json.load(ask(body))["choices"][0]["message"]
  assert message == {"role": "assistant", "reasoning_content": text.strip(), "content": ""}  # no </think> came
  events = streamed(ask(body | {"stream": True, "logprobs": True, "top_logprobs": 1}))[0]
  assert "".join(e["delta"].get("reasoning_content", "") for e in events) == text.strip() and not any("content" in e["delta"] for e in events)
  made = [c for e in events if e.get("logprobs") for c in e["logprobs"]["content"]]  # each piece's tokens, with their log probabilities
  joined = b"".join(bytes(c["bytes"]) for c in made).decode("utf-8", "replace")
  assert joined == text and all(c["top_logprobs"][0]["logprob"] == c["logprob"] for c in made)
  # with thinking off, the prompt carries an empty reasoning block, which makes it too long for these slots
  off = len(tok.encode(serve.chat("<|im_start|> <think>", hi[1:], False), special=True))
  with pytest.raises(urllib.error.HTTPError) as e: ask(body | {"chat_template_kwargs": {"enable_thinking": False}})
  assert json.load(e.value)["error"]["message"].startswith(f"{off} tokens")
  think.shutdown()
  assert serve.chat("<|im_start|> <think>", hi[1:], False) == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
  assert serve.answer("<think>\nLet me see.\n</think>\n\nHi!") == ("Let me see.", "Hi!") and serve.answer("<think>\nHm") == ("Hm", "")
  harmony = serve.chat("<|channel|>", hi, effort="low")  # gpt-oss's harmony has its system message, ours as a developer message, and channels
  assert harmony.startswith("<|start|>system<|message|>You are ChatGPT") and "Reasoning: low" in harmony and harmony.endswith(
    "<|start|>developer<|message|># Instructions\n\nBe brief.\n\n<|end|><|start|>user<|message|>Hi<|end|><|start|>assistant")
  assert serve.answer("<|channel|>analysis<|message|>Greet.<|end|><|start|>assistant<|channel|>final<|message|>Hello!") == ("Greet.", "Hello!")
  assert serve.answer("<|channel|>analysis<|message|>Gree") == ("Gree", "") and serve.answer("<|channel|>final<|message|>Hi") == ("", "Hi")
  assert serve.answer("<|channel|>") == serve.answer("<|channel|>analysis") == ("", ""), "a partial channel name is neither reasoning nor answer"

class Kept(serve.Request):  # a request that keeps the logits row behind each of its tokens (Head.row)
  def take(self, head, room):
    self.logits = getattr(self, "logits", []) + [head.row() if isinstance(head, serve.Head) else head]
    return super().take(head, room)

def agrees(r, w, c, tol=0.03):
  # a finished request's logits must be within tol of numpy's, and its picks near numpy's
  ids = r.ids + r.made
  want = llama.forward(ids, list(range(len(ids))), w, c, llama.kv_cache(c, 64, np.zeros), every=True)[len(r.ids) - 1:][:len(r.logits)]
  worst = max(np.abs(g - v).max() / np.abs(v).max() for g, v in zip(r.logits, want))
  assert worst < tol and near(r.logits, want), worst

def streamed(response):  # the choices of a server-sent stream, and the usage in its last chunk if the request asked for it
  events = [json.loads(line[6:]) for line in response.read().decode().split("\n\n") if line.startswith("data: {")]
  usage = events.pop()["usage"] if events[-1]["choices"] == [] else None
  return [e["choices"][0] for e in events], usage

def served(engine, *requests):  # sends requests to an engine, each after the one before ends. The requests in a list go in together
  for batch in requests:
    for r in batch if isinstance(batch, list) else [batch]: engine.waiting.put(r)
    for r in batch if isinstance(batch, list) else [batch]:
      while r.out.get(timeout=600) is not None: pass

@pytest.mark.parametrize("kv", ["bfp8", "bfp4"])
def test_serve_sessions(chip, tmp_path, kv):
  # The serve.py engine with Gated DeltaNet blocks. An agent's next turn resumes from the session kept when its last
  # request ended, or from the one kept after its prompt, and prefills only the new tokens. With bfp4, park 0 leaves
  # no room on the chip, so sessions go to the host and come back in bf16, which holds their values.
  # In bfp4, with 3 bits a value, a key that the chip rounds the other way from numpy moves by an eighth of the largest
  # in its run. numpy's logits move 2.9% when the qkv weights are off by bf16 rounding, against 1.0% in bfp8, so
  # the bound is 6%. In bfp8, over 6 decode steps from each of 6 prompts, the chip's largest error is 3.0% with the
  # median at 0.9%, and 3.1% and 1.0% when the residual stream was bf16, so the bound is 4%.
  rt, rng = Runtime(chip), np.random.default_rng(31)
  c, weights = qwen35(rng)
  w, tok, c.precision = llama.arrays(weights), bytes_tokenizer(tmp_path), llama.precision(c, "bf16", kv=kv)
  c.mtp = 0  # decode a token a step. test_serve_speculative drafts with the MTP layer
  engine = serve.Engine(rt, llama.upload(rt, {k: v for k, v in weights.items() if not k.startswith(("mtp.", "layers.4."))}), c, tok,
                        slots=2, places=64, chunk=16, park=0 if kv == "bfp4" else None)
  threading.Thread(target=engine.run, daemon=True).start()
  together = [Kept(tok, p, 6, 0) for p in ([7, 200, 31, 64, 5], [5, 64], list(range(40, 60)))]  # the last is longer than a chunk
  served(engine, together)
  tol = 0.06 if kv == "bfp4" else 0.04
  for r in together: agrees(r, w, c, tol)
  first = together[0]
  for turn, saved in ((first.ids + first.made + [9, 8], first.ids + first.made[:-1]), (first.ids + [1, 2, 3], first.ids)):
    before, r = engine.prefilled, Kept(tok, turn, 4, 0)
    served(engine, r)
    agrees(r, w, c, tol)
    assert engine.prefilled - before == len(turn) - len(saved) == len(turn) - r.cached, (engine.prefilled - before, r.cached)
  assert {type(v[-1]) for v in engine.sessions.values()} == {bytes if kv == "bfp4" else Tensor}, "bfp4 sessions go to the host, bfp8 to the chip"

def test_serve_speculative(chip, tmp_path):
  # The serve.py engine drafts with the multi-token prediction layer of the small Qwen3.5. An oracle's drafts are nearly
  # all accepted, and the random layer's are nearly all rejected. Either way the tokens must agree with numpy, and a
  # next turn resumes from the session with the states from before any rejected draft.
  rt, rng = Runtime(chip), np.random.default_rng(31)
  c, weights = qwen35(rng)
  w, tok, t = llama.arrays(weights), bytes_tokenizer(tmp_path), llama.upload(rt, weights)
  class Oracle(serve.Engine):
    def guess(self, u, best):
      s = self.busy[u].ids + self.busy[u].made
      return int(llama.forward(s, list(range(len(s))), w, c, llama.kv_cache(c, 64, np.zeros))[0].argmax())
  prompts = ([7, 200, 31, 64, 5], [5, 64], list(range(40, 60)))
  for kind in (Oracle, serve.Engine):
    engine = kind(rt, t, c, tok, slots=2, places=64, chunk=16)
    threading.Thread(target=engine.run, daemon=True).start()
    together = [Kept(tok, p, 8, 0) for p in prompts]
    served(engine, together)
    for r in together: agrees(r, w, c)
    assert engine.spec and (engine.agreed >= 0.8 * engine.drafted if kind is Oracle
                            else engine.agreed <= 0.2 * engine.drafted), (engine.agreed, engine.drafted)
  first = together[0]
  before, r = engine.prefilled, Kept(tok, first.ids + first.made + [9, 8], 4, 0)
  served(engine, r)
  agrees(r, w, c)
  assert engine.prefilled - before == 3, engine.prefilled - before

def test_session_snapshot_without_device_space(chip):
  import collections
  from baretile import bh
  rt, engine = Runtime(chip), object.__new__(serve.Engine)
  conv = Tensor.of(rt, np.arange(3 * 2 * 32, dtype=np.float32).reshape(3, 2, 32).tolist(), "fp32")
  state, kv = Tensor.full(rt, 32, 32, 3, "fp32"), Tensor.full(rt, 32, 32, 7)
  engine.rt, engine.c = rt, llama.model(layer_types=["linear_attention", "full_attention"])
  engine.slot, engine.held, engine.taken, engine.coherent = [[(qwen.Row(conv, 1), state), kv]], [[1, 2]], [1.0], [True]
  engine.sessions, engine.park, engine.keep = collections.OrderedDict(), 0, 1 << 20
  want = [tuple(x.tobytes() for x in engine.slot[0][0]), kv.tobytes()]
  rt.wait()
  # Leave only 64 bytes per bank. A Row.get() or a device snapshot cannot fit here.
  reserved = [rt.dram_alloc(room - 64) for _, room in list(rt.holes) if room > 64]
  try:
    for park in (0, 1 << 20):  # direct host storage, then a failed device copy followed by host storage
      engine.park = park
      engine.save(0)
      assert engine.sessions[(1, 2)] == want
      engine.sessions.clear()
    engine.keep = 1  # too small even on the host: dropping this optional snapshot is successful
    engine.save(0)
    assert not engine.sessions
  finally:
    for addr in reserved: rt.dram_free(addr)
    rt.wait()
  engine.keep, engine.park = 1 << 20, 1 << 20
  engine.save(0)
  assert isinstance(engine.sessions[(1, 2)][1], Tensor)
  first = engine.sessions[(1, 2)]
  engine.park = sum(rt.used[x.addr] * len(bh.DRAM) for part in first for x in (part if isinstance(part, tuple) else (part,)))
  del first
  engine.held[0] = [3, 4]
  engine.save(0)
  assert engine.sessions[(1, 2)] == want and isinstance(engine.sessions[(3, 4)][1], Tensor), "evict before copying"

def test_session_restore_without_device_space(chip):
  # A session kept on the host that no longer fits in DRAM is dropped when a prompt would restore it. Uploads that fit
  # are freed, the slot keeps its state, and the prompt prefills from the start. Copies to the chip that fit before
  # a save fails are freed too. With room again, the session restores.
  import collections
  rt, engine = Runtime(chip), object.__new__(serve.Engine)
  conv = lambda v: Tensor.of(rt, np.full((3, 2, 32), v, np.float32).tolist(), "fp32")
  slot = lambda v: [(qwen.Row(conv(v), 1), Tensor.full(rt, 32, 32, v, "fp32")), Tensor.full(rt, 64, 32, v)]
  contents = lambda s: [value(s[0][0].t), value(s[0][1]), value(s[1])]
  engine.rt, engine.c, engine.linear = rt, llama.model(layer_types=["linear_attention", "full_attention"]), True
  engine.slot, engine.held, engine.taken, engine.coherent, engine.busy, engine.used = [slot(7)], [[1, 2]], [1.0], [True], [None], [0]
  engine.sessions, engine.park, engine.keep = collections.OrderedDict(), 0, 1 << 20
  engine.save(0)  # to host memory, since park is 0
  kept = engine.sessions[(1, 2)]
  engine.slot[0], engine.held[0] = slot(5), []
  before = contents(engine.slot[0])
  rt.wait()
  tail = max(rt.holes)
  assert tail[1] > 8192 + 64
  for spare in (0, 8192):  # every hole keeps 64 bytes, and the free tail of DRAM spare more: room for two 4 KiB allocations
    reserved = [rt.dram_alloc(room - 64 - (spare if (addr, room) == tail else 0)) for addr, room in sorted(rt.holes) if room > 64]
    try:
      rt.wait()
      free = sorted(rt.holes)
      if spare:
        engine.park, engine.held[0] = 1 << 20, [4]
        engine.save(0)  # copies the convolution row and the state, then fails on the KV rows
        rt.wait()
        assert isinstance(engine.sessions.pop((4,))[1], bytes) and sorted(rt.holes) == free, "free partial copies"
        engine.park, engine.held[0] = 0, []
      engine.busy[0], engine.sessions[(1, 2)] = None, kept
      r = serve.Request(None, [1, 2, 3])
      engine.admit(r)  # with room, uploads the index row and the convolution row, then fails on the state
      rt.wait()
      assert sorted(rt.holes) == free, "free partial uploads"
    finally:
      for addr in reserved: rt.dram_free(addr)
      rt.wait()
    assert engine.busy[0] is r and r.cached == 0 and engine.held[0] == [] and not engine.sessions
    assert all(np.array_equal(a, b) for a, b in zip(contents(engine.slot[0]), before)), "the slot keeps its state"
  engine.busy[0], engine.sessions[(1, 2)] = None, kept
  r = serve.Request(None, [1, 2, 3])
  engine.admit(r)
  history, state, kv = contents(engine.slot[0])
  assert r.cached == 2 and engine.held[0] == [1, 2] and (history[:, 1] == 7).all() and (history[:, 0] == 5).all()
  assert (state == 7).all() and (kv[:, :32] == 7).all() and (kv[:, 32:] == 5).all()

def test_prompt_snapshot_includes_prediction(chip, tmp_path):
  rt, rng = Runtime(chip), np.random.default_rng(31)
  c, weights = qwen35(rng)
  tok, t = bytes_tokenizer(tmp_path), llama.upload(rt, weights)
  for limit in (1, 3):
    engine = serve.Engine(rt, t, c, tok, slots=1, places=64, chunk=16)
    r = serve.Request(tok, [7, 200, 31, 64, 5], max_tokens=limit, temperature=0)
    engine.admit(r)
    engine.fill(0)
    key = tuple(r.ids)
    assert key in engine.sessions
    saved, live = engine.sessions[key][-1], engine.slot[0][-1][:, :32, :]
    assert saved.tobytes() == live.tobytes() and np.count_nonzero(value(saved)), "save after the final draft update"
    wanted = saved.tobytes()
    engine.restore(0, key)
    assert engine.slot[0][-1][:, :32, :].tobytes() == wanted
    if engine.busy[0] is not None: r.gone = True; engine.take(0, None)
  # If the first prediction is EOS, there is no next embedding for the draft update. Do not reuse that prefix.
  r = serve.Request(tok, [7, 200, 31, 64, 5], temperature=0, eos=range(256))
  engine = serve.Engine(rt, t, c, tok, slots=1, places=64, chunk=16)
  engine.admit(r); engine.fill(0)
  assert r.finish == "stop" and not engine.sessions and not engine.coherent[0]
  next_request = serve.Request(tok, r.ids + [1], temperature=0)
  engine.admit(next_request)
  assert next_request.cached == 0 and engine.coherent[0], "incomplete prediction history must be rebuilt"

def test_chat_tools(tmp_path):
  # serve.qwen_chat must give the strings that Qwen3.6's chat_template.jinja makes through transformers'
  # apply_chat_template. Tool calls in an answer come out in OpenAI's format, typed by the tool's schema, and go back
  # into the template as they came. Over HTTP, a stand-in engine answers with a tool call, in one response and
  # streamed.
  import queue, urllib.request
  qwen = "<|im_start|> <think> <tool_call>\n<function="  # the markers that identify a template as Qwen3.5's
  tools = [{"type": "function", "function": {"name": "read", "description": "A file's text", "parameters": {"type": "object",
            "properties": {"path": {"type": "string"}, "lines": {"type": "integer"}}, "required": ["path"]}}}]
  call = {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": {"path": "a.txt", "lines": 2}}}
  messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "What is in a.txt?"},
              {"role": "assistant", "content": "", "reasoning_content": "I should read it.", "tool_calls": [call]},
              {"role": "tool", "tool_call_id": "call_1", "content": "hello"}]
  want = ("<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n" + json.dumps(tools[0]) + "\n</tools>" +
          serve.CALLING + "\n\nBe brief.<|im_end|>\n<|im_start|>user\nWhat is in a.txt?<|im_end|>\n<|im_start|>assistant\n<think>\n"
          "I should read it.\n</think>\n\n<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n<parameter=lines>\n2\n</parameter>\n"
          "</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\nhello\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n")
  assert serve.chat(qwen, messages, tools=tools) == want
  assert serve.chat(qwen, messages[:2]) == ("<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nWhat is in a.txt?<|im_end|>\n"
                                            "<|im_start|>assistant\n<think>\n")
  assert serve.chat(qwen, messages[:2], False).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "with thinking off, reasoning is empty"
  said = ("I should read it.\n</think>\n\nLet me look.\n\n<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n"
          "<parameter=lines>\n2\n</parameter>\n</function>\n</tool_call>")
  text, made = serve.calls(serve.answer(said)[1], tools)
  calls = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in made]
  assert text == "Let me look." and calls == [("read", {"path": "a.txt", "lines": 2})]
  again = messages[:2] + [{"role": "assistant", "content": text, "reasoning_content": "I should read it.", "tool_calls": made}]
  prompt = serve.chat(qwen, again + messages[3:], tools=tools)
  assert "Let me look.\n\n<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n<parameter=lines>\n2\n</parameter>" in prompt
  class Stand:  # an engine that answers requests with `said`, 7 characters at a time
    tok, places, waiting, busy = bytes_tokenizer(tmp_path), 4096, queue.Queue(), [None]
    def offer(self, r): self.waiting.put(r); return True  # there's room for any request
    def run(self):
      while r := self.waiting.get():
        for i in range(0, len(said), 7): r.out.put(said[i:i + 7])
        r.finish = "stop"
        r.out.put(None)
  stand = Stand()
  threading.Thread(target=stand.run, daemon=True).start()
  http = serve.server(stand, "qwen", qwen, port=0)
  threading.Thread(target=http.serve_forever, daemon=True).start()
  ask = lambda body: urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{http.server_address[1]}/v1/chat/completions",
                                                                   json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=60)
  reply = json.load(ask({"messages": messages[:2], "tools": tools}))["choices"][0]
  message = reply["message"]
  assert reply["finish_reason"] == "tool_calls" and message["content"] == "Let me look." and message["reasoning_content"] == "I should read it."
  assert json.loads(reply["message"]["tool_calls"][0]["function"]["arguments"]) == {"path": "a.txt", "lines": 2}
  events = streamed(ask({"messages": messages[:2], "tools": tools, "stream": True}))[0]
  assert "".join(e["delta"].get("content", "") for e in events).strip() == "Let me look." and events[-1]["finish_reason"] == "tool_calls"
  assert "".join(e["delta"].get("reasoning_content", "") for e in events) == "I should read it."
  assert [json.loads(c["function"]["arguments"]) for e in events for c in e["delta"].get("tool_calls", [])] == [{"path": "a.txt", "lines": 2}]
  with pytest.raises(urllib.error.HTTPError) as e:
    ask({"messages": messages[:2], "tools": tools, "tool_choice": {"type": "function", "function": {"name": "write"}}})
  assert e.value.code == 400 and "no such tool" in json.load(e.value)["error"]["message"]
  with pytest.raises(urllib.error.HTTPError) as e: ask({"messages": messages[:2], "response_format": {"type": "xml"}})
  assert e.value.code == 400
  for said in ("I should read it.\n</think>\n\nReading.\n\n<tool_call>\n<function=re", "I see.\n</think>\n\nA <"):
    # once the answer ends, a tool call cut short and text that might have begun one are shown as they are
    shown = serve.answer(said)[1]
    reply = json.load(ask({"messages": messages[:2], "tools": tools}))["choices"][0]
    assert reply["message"]["content"] == shown and reply["finish_reason"] == "stop" and "tool_calls" not in reply["message"]
    events = streamed(ask({"messages": messages[:2], "tools": tools, "stream": True}))[0]
    assert "".join(e["delta"].get("content", "") for e in events) == shown and events[-1]["finish_reason"] == "stop"
  http.shutdown()

def test_attention_cost_units():
  # cost.attention counts reads in kernels.attention_plan's units. A unit reads each key and value tile once for all its
  # tile rows of queries. While there are fewer units than workers, a unit takes one tile row, so more workers share the
  # work. With a unit for every worker, a unit takes up to 8 tile rows, and they share its reads.
  c, n = llama.model(dim=64, heads=1, kv=1, layers=1), cost.WORKERS
  units = lambda entries, queries: cost.attention(c, entries, queries, 32, "bf16") / (2 * 32 * 64 * 2)  # a unit reads a key and a value tile
  assert [units(1, queries) for queries in (1, 64, 65, 256)] == [1, 2, 3, 8], "one tile row a unit"
  assert [units(n, queries) for queries in (32, 256, 257, 1024)] == [n, n, 2 * n, 4 * n], "up to 8 tile rows a unit"

def test_cost_fidelity():
  # Make DRAM negligible so the estimate exposes the phase cost, including TT's mixed MLP and attention phases.
  c = llama.model(dim=64, heads=1, kv=1, layers=2, ffn=128, vocab=256)
  use, card = cost.USE | {"context": 64}, cost.CARD | {"bandwidth": 1e30, "tflops": 1e12}
  for mode in ("bf16", "bfp8", "bfp4"):
    s = cost.shape(c, mode)
    step, prompt = cost.step(c, s, use, card, 2), cost.prompt(c, s, use, card, 64)
    assert s.work == 4 * s.values and step.work == 4 * step.flops and prompt.work == 4 * prompt.flops
    assert step.compute == step.work / card["tflops"] and prompt.seconds == prompt.work / card["tflops"]
    overridden = cost.prompt(c, s, use, card | {"phases": 2}, 64)
    assert prompt.seconds == 2 * overridden.seconds
  s = cost.shape(c, "tt")
  gate_up = c.layers * 2 * c.dim * c.ffn
  assert s.work == 2 * s.values - gate_up
  step, prompt = cost.step(c, s, use, card, 2), cost.prompt(c, s, use, card, 64)
  assert step.work == 2 * step.flops - 2 * 2 * gate_up
  attention = c.layers * 4 * 64 * c.heads * llama.head(c) * 64
  assert prompt.work == 2 * prompt.flops - 2 * 64 * gate_up + 2 * attention
  # A one-token remainder uses decode attention, as does a workload made entirely of one-token chunks.
  tail = cost.prompt(c, s, use | {"context": 65}, card, 64)
  assert tail.work - prompt.work == 2 * s.work + c.layers * 4 * c.heads * llama.head(c) * 65 * s.decode_phases
  singles = cost.prompt(c, s, use | {"context": 3}, card, 1)
  assert singles.work == 2 * 3 * s.work + c.layers * 4 * c.heads * llama.head(c) * (1 + 2 + 3) * s.decode_phases
  # Llama-3.1-8B's last gate/up uses two phases rather than TT's usual one.
  large = llama.model(dim=4096, heads=32, kv=8, layers=32, ffn=14336, vocab=128256)
  s = cost.shape(large, "tt")
  assert s.work == 2 * s.values - 31 * 2 * 4096 * 14336
  # B3's mixed-phase roofline uses causal attention and phase-weighted work against the one-phase peak.
  s, attention = cost.shape(c, "tt"), c.layers * 4 * c.heads * llama.head(c) * 64 * 65 // 2
  flops = 2 * 64 * (s.values - c.vocab * c.dim) + 2 * c.vocab * c.dim + attention
  assert benchmark.prefill_work(c, 64) == 2 * flops - 2 * 64 * gate_up + 2 * attention

def test_cost_draft_head():
  # Experts use one phase, projections two, routers and MTP inputs four, and the draft head one. The draft
  # vocabulary projection runs once per prompt chunk, including a short final chunk, while its layers run per token.
  c = llama.model(dim=64, heads=1, kv=1, layers=1, ffn=128, vocab=256, experts=4, k=2, moe=64, shared=32, mtp=1)
  s, use, card = cost.shape(c, "experts"), cost.USE | {"context": 10}, cost.CARD | {"bandwidth": 1e30, "tflops": 1e12}
  mixer, mlp, router, head = 4 * 64 * 64, 3 * 64 * (32 + 2 * 64), 64 * 4, 256 * 64
  assert s.work == 2 * mixer + mlp + 4 * router + 2 * head
  assert s.draft_work == 2 * mixer + mlp + 4 * router + 4 * 2 * 64 * 64 and s.draft_head_work == head
  base = llama.model(**(vars(c) | {"mtp": 0}))
  for chunk in (1, 4, 16):
    plain, draft = cost.prompt(base, cost.shape(base, "experts"), use, card, chunk), cost.prompt(c, s, use, card, chunk)
    chunks = math.ceil(use["context"] / chunk)
    attention = sum(4 * min(chunk, 10 - at) * 64 * (at + min(chunk, 10 - at)) for at in range(0, 10, chunk))
    assert draft.flops - plain.flops == 2 * 10 * s.draft_values + 2 * chunks * head + attention
    assert draft.work - plain.work == 2 * 10 * s.draft_work + 2 * chunks * s.draft_head_work + 4 * attention
  step = cost.step(c, s, use, card, 2, speculative=True)
  # Verification and drafting both attend to the cache for two positions per user.
  assert step.flops == 2 * 4 * (s.values + s.draft_values + head) + 2 * 4 * 4 * 64 * 10

def test_cost(tmp_path):
  # baretile.models.cost works from a config alone. Qwen3.6-35B-A3B has 35.5B parameters (0.9B in the multi-token prediction
  # layer) and reads 2.95B a token, about 21.9 GB in the one-card formats. Its KV cache is 20 KB a token (10 attention
  # blocks with 2 KV heads of 256), and its Gated DeltaNet states are 66 MB a session (30 blocks of 32 heads of 128 x
  # 128 in fp32, plus the last 3 conv inputs). Llama-3.1-8B reads benchmark.py's 6.20 GB a token, within 2%. At 256
  # tokens, a prompt chunk reads almost all the experts of each block, so Qwen3.6's 4K-token prompt is over 6 times
  # slower in chunks of 256 than of 2,048. At 32K tokens, attention's reads slow it down again.
  qwen = {"text_config": {"hidden_size": 2048, "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256, "num_hidden_layers": 40,
          "rms_norm_eps": 1e-6, "max_position_embeddings": 262144, "vocab_size": 248320, "num_experts": 256, "num_experts_per_tok": 8,
          "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512, "linear_num_key_heads": 16, "linear_num_value_heads": 32,
          "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "mtp_num_hidden_layers": 1,
          "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 10, "model_type": "qwen3_5_moe_text",
          "rope_parameters": {"rope_theta": 1e7, "partial_rotary_factor": 0.25, "rope_type": "default"}}}
  llama8b = {"model_type": "llama", "hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8, "num_hidden_layers": 32,
             "rms_norm_eps": 1e-5, "max_position_embeddings": 131072, "vocab_size": 128256, "intermediate_size": 14336, "rope_theta": 5e5}
  for name, j in (("qwen", qwen), ("llama", llama8b)): (tmp_path / name).mkdir(); (tmp_path / name / "config.json").write_text(json.dumps(j))
  c = llama.config(str(tmp_path / "qwen"))
  s, use = cost.shape(c, "experts"), dict(cost.USE)
  assert 35.4e9 < s.held < 35.6e9 and 2.9e9 < s.values < 3.0e9 and 21.8e9 < s.bytes < 22.2e9, (s.held, s.values, s.bytes)  # includes the draft head
  one = cost.session(c, use | {"context": 1})
  assert (one.kv, one.state) == (10 * 2 * 2 * 256 * 2, 30 * 4 * (32 * 128 * 128 + 3 * (2 * 16 * 128 + 32 * 128)))
  l8 = llama.config(str(tmp_path / "llama"))
  read = cost.shape(l8, "bfp4").every + cost.session(l8, use | {"context": 612.5, "kv": "bfp8"}).kv
  assert abs(read / 6.20e9 - 1) < 0.02, read
  card = dict(cost.CARD)
  t = cost.step(c, s, use, card, 4)
  assert math.isclose(cost.hourly(card) / (4 * t.rate * 3600) * 1e6, cost.hourly(card) / (4 / t.seconds) / 3600 * 1e6)
  (tmp_path / "probe.json").write_text(json.dumps({"DRAM reads of bfp8 chunks, 4 in flight: grid workers, own banks": 420.0,
                                                   "DRAM reads of bfp4 chunks, 4 in flight: grid workers, own banks": 300.0,
                                                   "matmul step, 1 phase, unpacks included": 16.0, "launch, 100 queued: chip each": 2700.0,
                                                   "upload over PCIe": 20.0, "read back over PCIe": 12.0}))
  measured = cost.probed(card, str(tmp_path / "probe.json"))
  probe = str(tmp_path / "probe.json")
  llama_cost = lambda **keys: cost.report(str(tmp_path / "llama"), probe=probe, **keys)
  assert llama_cost() != llama_cost(launch=1e-3), "explicit settings override the probe"
  assert measured["bandwidth"] == 300e9 and abs(measured["tflops"] / 658e12 - 1) < 0.01 and measured["launch"] == 2e-6 and measured["pcie"] == 12e9
  slow, fast, long = (cost.prompt(c, s, use | {"kv": "bfp8", "context": n}, card, chunk) for n, chunk in ((4096, 256), (4096, 2048), (32768, 2048)))
  assert slow.rate * 6 < fast.rate and long.rate * 1.5 < fast.rate, (slow.rate, fast.rate, long.rate)
  # a chunk of 2,048 reads its activations again for each group of columns. 256 doesn't, since the accumulators fit
  assert cost.regroups(c, 256) == 0 < cost.regroups(c, 2048), cost.regroups(c, 2048)
  assert "a prompt of 8192 tokens" in cost.report(str(tmp_path / "qwen"), "experts", context="8192", kv="bfp8")
  for keys in ({"contex": 32}, {"context": 0}, {"context": 1.5}, {"bandwidth": 0}, {"phases": 0}, {"reach": 2},
               {"reserve": card["dram"]}, {"kv": "fp64"}):
    with pytest.raises(ValueError): cost.report(str(tmp_path / "llama"), **keys)
  assert "each operation's precision phases" in cost.report(str(tmp_path / "llama"))
  assert "4 phases for every operation (override)" in cost.report(str(tmp_path / "llama"), phases=4)
  with pytest.raises(ValueError, match="precision"): llama.precision(c, "bpf8")
  windowed = llama.model(dim=128, heads=4, kv=2, layers=2, sliding=32, layer_types=["sliding_attention", "full_attention"])
  assert cost.session(windowed, use | {"context": 1024}).kv == 2 * 1024 * 2 * 2 * 32 * 2, "reserve full KV rows, even with a sliding window"
  held = cost.shape(l8).bytes
  result = cost.report(str(tmp_path / "llama"), dram=held + card["reserve"] + 1)
  assert "0 sessions fit" in result and "no room for a session" in result and "tok/s" not in result

def test_fetch(tmp_path, monkeypatch):
  # serve.fetch from a local server standing in for Hugging Face (HF_ENDPOINT). It gets the shards the index names and
  # the configs that exist, and a finished download isn't fetched again.
  import functools, http.server, threading
  hub = tmp_path / "hub" / "org" / "model" / "resolve" / "main"
  hub.mkdir(parents=True)
  weight_map = {"a": "model-1.safetensors", "b": "model-2.safetensors", "c": "model-1.safetensors"}
  files = {"config.json": "{}", "tokenizer.json": "{}", "generation_config.json": "{}", "chat_template.jinja": "<|im_start|>",
           "model-1.safetensors": "1", "model-2.safetensors": "2", "model.safetensors.index.json": json.dumps({"weight_map": weight_map})}
  for name, text in files.items(): (hub / name).write_text(text)
  quiet = type("Quiet", (http.server.SimpleHTTPRequestHandler,), {"log_message": lambda *a: None})
  hf = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(quiet, directory=str(tmp_path / "hub")))
  threading.Thread(target=hf.serve_forever, daemon=True).start()
  monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{hf.server_address[1]}")
  folder = serve.fetch("org/model", str(tmp_path / "cache"))
  assert sorted(os.listdir(folder)) == sorted([*files, ".done"]) and all((pathlib.Path(folder) / n).read_text() == t for n, t in files.items())
  for model in ("../escape", "/missing-model", "org/../escape"):
    with pytest.raises(ValueError, match="model"): serve.fetch(model, str(tmp_path / "cache"))
  (pathlib.Path(folder) / ".done").unlink()
  index = pathlib.Path(folder) / "model.safetensors.index.json"
  for shard in ("../escape.safetensors", "/tmp/escape.safetensors", "config.json"):
    index.write_text(json.dumps({"weight_map": {"x": shard}}))
    with pytest.raises(ValueError, match="path|shard"): serve.fetch("org/model", str(tmp_path / "cache"))
    assert not (pathlib.Path(folder) / ".done").exists()
  index.write_text(files[index.name])
  assert serve.fetch("org/model", str(tmp_path / "cache")) == folder
  hf.shutdown()
  assert serve.fetch("org/model", str(tmp_path / "cache")) == folder and serve.fetch(folder) == folder, "fetches once, and uses a folder as is"

@pytest.mark.skipif("LLAMA" not in os.environ, reason="set LLAMA to a local HuggingFaceTB/SmolLM-135M checkpoint folder")
def test_llama_model(chip):
  # SmolLM-135M with its pretrained weights, on a prompt and the token decoded after it, against numpy. In fp32, numpy
  # matches Hugging Face to 2e-5. It takes about a minute on ttsim.
  rt = Runtime(chip)
  c, weights = llama.load(os.environ["LLAMA"])
  w, t = llama.arrays(weights), llama.upload(rt, weights)
  kv, np_kv = llama.kv_cache(c, 32, lambda shape: Tensor.of(rt, np.zeros(shape).tolist())), llama.kv_cache(c, 32, np.zeros)
  new, places = [504, 2455, 282, 1029, 314], list(range(5))
  for _ in range(2):
    got = llama.logits(llama.forward(Tensor.of(rt, [new], "int32"), Tensor.of(rt, [places], "int32"), t, c, kv))
    want = llama.forward(new, places, w, c, np_kv)[0]
    assert np.abs(got - want).max() < 1 and got.argmax() == want.argmax(), (np.abs(got - want).max(), got.argmax(), want.argmax())
    new, places = [int(want.argmax())], [places[-1] + 1]
  assert new == [1962], "a [question]"
