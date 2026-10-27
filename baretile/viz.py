# A page that shows what a program launched: a timeline of the launches on the dispatcher's clock, the kernels by the
# cycles they took, and the expression each fused kernel computes, drawn as a graph. The page is one HTML file with
# the data in it, so you can open it in any browser or send it to someone.
#   TTSIM=/path/to/libttsim.so python3 -m baretile.viz script.py [args]
# runs the script the way python3 would, then writes viz.html for every runtime the script made.
import json, pathlib, runpy, struct, sys
from baretile import pci, tensor
from baretile.runtime import TIMES, Runtime

def save(path, *runtimes, since=0):
  """Write the page for each runtime's finished launches after since. A runtime keeps the times of its last TIMES
  launches, so the page leaves out earlier ones. save doesn't wait, which lets it run after a hang or an interrupt.
  Call rt.wait() first, or tolist(), to include everything queued."""
  made = {v if isinstance(v, bytes) else v[0]: (k, v) for k, v in tensor.kernels.items()}  # what each image was compiled from
  data = json.dumps([_runtime(rt, since, made) for rt in runtimes]).replace("</", "<\\/")  # nothing in it can close the script tag
  pathlib.Path(path).write_text(pathlib.Path(__file__).with_name("viz.html").read_text().replace("{data}", data))

def _runtime(rt, since, made):
  done = rt.launches - (rt.launches - rt.read_status(1)) % 2**32  # the last launch that finished
  kernels, index, launches, at, last = [], {}, [], 0, None
  for n in range(max(since, rt.launches - TIMES) + 1, done + 1):
    name, workers, image = rt.sent[n % TIMES]
    go, end, flits = struct.unpack_from("<3I", rt.times, n % TIMES * 16)
    at += 0 if last is None else (go - last) % 2**32  # the clock has 32 bits, so a gap of 2^32 cycles or more shows shorter
    last = go
    if image not in index:
      index[image] = len(kernels)
      kernels.append(dict(name=name, launches=0, cycles=0, bytes=0 if rt.counts else None) | _kernel(*made.get(image, (None, None))))
    k = kernels[index[image]]
    k["launches"], k["cycles"] = k["launches"] + 1, k["cycles"] + (end - go) % 2**32
    if rt.counts: k["bytes"] += 64 * flits
    launches.append((index[image], n, at, (end - go) % 2**32, 64 * flits if rt.counts else None, workers))
  return dict(title=f"{len(rt.cores)} workers, dispatcher at {rt.dispatcher}", sim=isinstance(rt.chip.pci, pci.Sim),
              launches=launches, kernels=kernels, total=rt.launches)

def _kernel(key, value):
  """How a kernel was made, from its key in tensor.kernels: the expression of a fused kernel, as a graph, or the key of
  any other kernel. A kernel launched with Runtime.launch alone isn't in the cache, so nothing is known about it."""
  if key is None: return {}
  if isinstance(value, bytes): return dict(text=repr(key))
  e, shapes, dtype, out, _, reps, _ = key
  return dict(text=repr(e), dtype=dtype, out=out, units=value[1], graph=_graph(e, shapes, reps))

def _graph(e, shapes, reps):
  """A fused expression as nodes of [label, detail, operands], operands first. Each distinct subexpression is one node,
  since fused_kernel computes it once."""
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

def _how(mode):  # a matmul's mode for B in words. fused_kernel's docstring has the modes: I picks by ids, C by columns, T transposes
  rest = mode.removeprefix("I")
  return ", ".join(w for w in ("picked by ids" * (rest != mode), {"C": "by columns", "T": "transposed"}.get(rest[:1], ""), rest.lstrip("CT")) if w)

if __name__ == "__main__":
  # Keep every runtime the script makes, and write the page even if the script stops early, as after a hang or Ctrl-C.
  if len(sys.argv) < 2: sys.exit("usage: python3 -m baretile.viz script.py [args]")
  runtimes, init = [], Runtime.__init__
  def keep(rt, *args, **kwargs):
    init(rt, *args, **kwargs)
    runtimes.append(rt)
  Runtime.__init__ = keep
  sys.argv = sys.argv[1:]
  sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parent))  # what python3 puts first for a script
  try: runpy.run_path(sys.argv[0], run_name="__main__")
  finally:
    if runtimes:
      save("viz.html", *runtimes)
      print(f"wrote viz.html with the launches of {len(runtimes)} runtime{'s' * (len(runtimes) > 1)}. Open it in a browser")
