# Check that a model generates on its own. The chip picks each token and is fed its own picks, and each step's logits
# are compared with an independent transformers reference at the same prefix. These are the first workload's
# autonomous generation gates (docs/workload.md). In bf16, at every step the chip's logits are within 3% of the
# reference's largest, and its pick within 1% of the reference's span below the reference's best; the reference runs
# in fp32 with its weights rounded to bf16. In another mode, such as tt, every pick is among the five best of a
# reference in bf16. NumPy's logits at the same prefixes come beside them, to tell an error in loading or the model
# from one on the chip.
#   python3 test/generation.py path/to/SmolLM-135M [report.json] [steps] [mode]
# It needs PyTorch and transformers. EOS doesn't stop a prompt, so each runs all its steps. The report holds each
# prompt's tokens and text, and each step's errors, with the logits of a step that failed.
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baretile.models import llama
from baretile.tensor import Tensor, jit

PROMPTS = ("The meaning of life is", "The capital of France is", "1, 2, 3,")  # as UTF-8, without BOS or a chat template
ERROR, PICK = 0.03, 0.01  # the gate: the largest error over the reference's largest logit, and the pick's loss over its span

def reference(folder, bf16):  # transformers' model as a function from ids to logits: in bf16, or in fp32 with bf16 weights
  import torch, transformers
  model = transformers.AutoModelForCausalLM.from_pretrained(folder, dtype=torch.bfloat16 if bf16 else torch.float32, attn_implementation="eager")
  for p in model.eval().parameters(): p.data = p.data.to(torch.bfloat16).to(p.dtype)
  def run(ids):
    with torch.no_grad(): return model(torch.tensor([ids])).logits[0, -1].double().numpy()
  return run

def check(rt, folder, steps=32, mode="bf16", prompts=PROMPTS):
  """Generate steps tokens after each prompt on rt's chip, from the chip's own picks, and compare each step with
  transformers and NumPy. Return the report, whose passed says whether every step met the gate."""
  close = mode == "bf16"  # the gate on the logits, else the one on the five best
  tok, ref = llama.Tokenizer(folder), reference(folder, bf16=not close)
  encoded = [tok.encode(p) for p in prompts]
  size = -(-(max(map(len, encoded)) + steps) // 32) * 32  # cache rows for the longest prompt and its steps
  c, weights = llama.load(folder, size)
  c.precision = llama.precision(c, mode)
  t, w = llama.upload(rt, weights, c.precision), llama.arrays(weights, c.precision)
  on = lambda row: Tensor.of(rt, [row], "int32")
  report = dict(folder=os.path.abspath(folder), mode=mode, steps=steps, gate=f"error {ERROR}, pick loss {PICK}" if close else "top 5", prompts=[])
  meets = lambda r: r["error"] <= ERROR and r["pick_loss"] <= PICK if close else r["in_top5"]
  for prompt, ids in zip(prompts, encoded):
    kv, np_kv = llama.kv_cache(c, size, llama.zeros(rt, llama.cache_format(c))), llama.kv_cache(c, size, np.zeros)
    step = jit(lambda x, places, kv=kv: llama.forward(x, places, t, c, kv))  # a token a step, replayed from the third on
    new, made, rows = ids, [], []
    for i in range(steps):
      places = list(range(len(ids) + len(made) - len(new), len(ids) + len(made)))
      got = llama.logits(llama.forward(on(new), on(places), t, c, kv) if i == 0 else step(on(new), on(places))).astype(np.float64)
      numpy, want = llama.forward(new, places, w, c, np_kv)[0].astype(np.float64), ref(ids + made)
      pick, top, span = int(got.argmax()), max(np.abs(want).max(), 1e-6), max(np.ptp(want), 1e-6)
      row = dict(token=pick, error=float(np.abs(got - want).max() / top), pick_loss=float((want.max() - want[pick]) / span),
                 in_top5=bool(pick in np.argsort(-want)[:5]), numpy_error=float(np.abs(numpy - want).max() / top))
      if not meets(row):  # what the chip and the reference made of the step
        top5 = lambda x: [(int(k), float(x[k])) for k in np.argsort(-x)[:5]]
        row |= dict(chip_top=top5(got), reference_top=top5(want))
      rows.append(row)
      made.append(pick)
      new = [pick]
    report["prompts"].append(dict(prompt=prompt, tokens=made, text=tok.decode(made), steps=rows,
                                  passed=all(map(meets, rows))))
  report["passed"] = all(p["passed"] for p in report["prompts"])
  return report

if __name__ == "__main__":
  if len(sys.argv) < 2: sys.exit("usage: python3 test/generation.py path/to/model [report.json] [steps] [mode]")
  from baretile import pci
  from baretile.chip import Chip
  from baretile.runtime import Runtime
  folder, out = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "generation.json"
  steps, mode = int(sys.argv[3]) if len(sys.argv) > 3 else 32, sys.argv[4] if len(sys.argv) > 4 else "bf16"
  report = check(Runtime(Chip(pci.open_device())), folder, steps, mode)
  with open(out, "w") as f: json.dump(report, f, indent=2)
  for p in report["prompts"]:
    worst = max(p["steps"], key=lambda r: r["error"])
    print(f"{'passed' if p['passed'] else 'FAILED'} {p['prompt']!r} -> {p['text']!r}: largest error {100 * worst['error']:.2f}% "
          f"(NumPy {100 * max(r['numpy_error'] for r in p['steps']):.2f}%), largest pick loss {100 * max(r['pick_loss'] for r in p['steps']):.2f}%")
  print(f"{'passed' if report['passed'] else 'failed'}: {len(report['prompts'])} prompts of {steps} steps, report in {out}")
  sys.exit(not report["passed"])
