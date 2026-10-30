# Run a script and write its completed launches to viz.html. The page shows a timeline, kernel timings, graphs of
# fused expressions and each kernel's C programs. It contains its own data and needs a browser with JavaScript enabled.
#   TTSIM=/path/to/libttsim.so python3 -m baretile.viz script.py [args], or -m module [args]
# The page includes the runtimes still open when the script ends. Closing a runtime, or replacing it with an
# overlapping one, discards its launch records.
import json, pathlib, runpy, struct, sys
from baretile import pci, tensor
from baretile.runtime import CORES, PROGRAMS, TIMES, Runtime

def save(path, *runtimes, since=0):
  """Write completed launches after since for each runtime, limited to its last TIMES launch records.
  This doesn't wait for queued work. Call rt.wait() first to include it, or save the completed work after
  a timeout or interrupt. A closed or replaced runtime raises RuntimeError, since its records went with it."""
  made = {v if isinstance(v, bytes) else v[0]: (k, v) for k, v in tensor.kernels.items()}  # what each image was compiled from
  data = json.dumps([_runtime(rt, since, made) for rt in runtimes]).replace("</", "<\\/")  # nothing in it can close the script tag
  pathlib.Path(path).write_text(pathlib.Path(__file__).with_name("viz.html").read_text().replace("{data}", data))

def _runtime(rt, since, made):
  done = rt.finished()  # the last launch that finished
  kernels, index, launches, at, last = [], {}, [], 0, None
  for n in range(max(since, rt.launches - TIMES) + 1, done + 1):
    name, workers, image = rt.sent[n % TIMES]
    go, end, flits = struct.unpack_from("<3I", rt.times, n % TIMES * 16)
    at += 0 if last is None else (go - last) % 2**32  # the clock has 32 bits, so a gap of 2^32 cycles or more shows shorter
    last = go
    if image not in index:
      index[image] = len(kernels)
      programs = [[core, src] for core, src in zip(CORES, PROGRAMS.get(image, ())) if src]  # the C each core ran
      kernels.append(dict(name=name, launches=0, cycles=0, bytes=0 if rt.counts else None, programs=programs) |
                     _kernel(*made.get(image, (None, None))))
    k = kernels[index[image]]
    k["launches"], k["cycles"] = k["launches"] + 1, k["cycles"] + (end - go) % 2**32
    if rt.counts: k["bytes"] += 64 * flits
    launches.append((index[image], n, at, (end - go) % 2**32, 64 * flits if rt.counts else None, workers))
  return dict(title=f"{len(rt.cores)} workers, dispatcher at {rt.dispatcher}", sim=isinstance(rt.chip.pci, pci.Sim),
              launches=launches, kernels=kernels, total=rt.launches)

def _kernel(key, value):
  """Return a fused kernel's expression graph, or another tensor kernel's cache key. Kernels absent from
  tensor.kernels have no compiler details to display."""
  if key is None: return {}
  if isinstance(value, bytes): return dict(text=repr(key))
  e, shapes, dtype, out, _, reps, _ = key
  return dict(text=repr(e), dtype=dtype, out=out, units=value[1], graph=_graph(e, shapes, reps))

def _graph(e, shapes, reps):
  """Return [label, detail, operands] nodes, with operands before the operations that use them. Each distinct
  subexpression has one node, even when a reduction computes it on several passes."""
  nodes, at = [], {}
  def visit(e, b=False):  # b: e is how a matmul reads its B
    if e in at: return at[e]
    if isinstance(e, int): node = [f"in {e}", "x".join(map(str, shapes[e])) + (f", rep {reps[e]}" if reps[e] > 1 else ""), []]
    elif b: node = ["read B", _how(e[0]), [visit(x) for x in e[1:]]]
    elif e[0] == "matmul": node = ["matmul", f"fidelity {e[3]}" if len(e) > 3 else "", [visit(e[1]), visit(e[2], True)]]
    else: node = [e[0], "", [visit(x) for x in e[1:]]]
    nodes.append(node)
    at[e] = len(nodes) - 1
    return at[e]
  visit(e)
  return nodes

def _how(mode):  # describe B's read mode: I selects matrices by index, C stores tile columns together, T transposes
  rest = mode.removeprefix("I")
  return ", ".join(w for w in ("picked by ids" * (rest != mode), {"C": "by columns", "T": "transposed"}.get(rest[:1], ""), rest.lstrip("CT")) if w)

if __name__ == "__main__":
  # Keep runtimes alive until save, including when the script raises an exception or receives Ctrl-C.
  if len(sys.argv) < 2 or sys.argv[1] == "-m" and len(sys.argv) < 3: sys.exit("usage: python3 -m baretile.viz script.py [args], or -m module [args]")
  runtimes, init = [], Runtime.__init__
  def keep(rt, *args, **kwargs):
    init(rt, *args, **kwargs)
    runtimes.append(rt)
  Runtime.__init__ = keep
  module = sys.argv[1] == "-m"  # run a module, as python3 -m does, such as baretile.models.llama
  sys.argv = sys.argv[1 + module:]
  if not module: sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parent))  # what python3 puts first for a script
  try: runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True) if module else runpy.run_path(sys.argv[0], run_name="__main__")
  finally:
    # A closed or replaced runtime's launch records went with its host memory, so the page holds the open ones
    if live := [rt for rt in runtimes if not rt.closed]:
      save("viz.html", *live)
      print(f"wrote viz.html for {len(live)} runtime{'s' * (len(live) > 1)}. Open it in a browser")
    elif runtimes: print("no viz.html: the script closed its runtimes, and their launch records went with them", file=sys.stderr)
