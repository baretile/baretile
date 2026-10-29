# Train a small character-level GPT-2 on README and the docs using fp32 and Adam.
# PyTorch checks the loss and gradients at each step using the same weights. A separate PyTorch
# run starts from the same weights to compare loss curves; floating-point rounding lets them diverge.
#   TTSIM=/path/to/libttsim.so python3 test/train.py [steps]
import math, os, pathlib, sys, time
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baretile import tensor
from baretile.tensor import Tensor
from baretile.models import gpt2

def text():  # README.md, then the pages in docs/ by name
  root = pathlib.Path(__file__).parent.parent
  return "".join(p.read_text() for p in [root / "README.md", *sorted((root / "docs").glob("*.md"))])

def init(c, seed=0):
  """GPT-2's initial weights by name, as float32 numpy. Weights are N(0, 0.02), the position embedding N(0, 0.01), and
  the projections back into the residual are scaled down by depth. Layer norms start at 1 and biases at 0."""
  rng, d, w = np.random.default_rng(seed), c.dim, {}
  normal = lambda *shape, std=0.02: rng.normal(0, std, shape).astype(np.float32)
  w["wte.weight"], w["wpe.weight"] = normal(c.vocab, d), normal(c.ctx, d, std=0.01)
  for l in range(c.layers):
    for name, (rows, cols) in {"attn.c_attn": (d, 3 * d), "attn.c_proj": (d, d), "mlp.c_fc": (d, 4 * d), "mlp.c_proj": (4 * d, d)}.items():
      w[f"h.{l}.{name}.weight"] = normal(rows, cols, std=0.02 / math.sqrt(2 * c.layers) if name.endswith("c_proj") else 0.02)
      w[f"h.{l}.{name}.bias"] = np.zeros((1, cols), np.float32)
    for ln in ("ln_1", "ln_2"): w[f"h.{l}.{ln}.weight"], w[f"h.{l}.{ln}.bias"] = np.ones((1, d), np.float32), np.zeros((1, d), np.float32)
  w["ln_f.weight"], w["ln_f.bias"] = np.ones((1, d), np.float32), np.zeros((1, d), np.float32)
  return w

def forward(ids, w, mask, c):  # like gpt2.forward, but returns logits for all positions and transposes wte for the output layer
  f = gpt2.fns(w["wte.weight"])
  x = f.rows(w["wte.weight"], ids) + w["wpe.weight"]  # x has c.ctx rows, as many as wpe, so the position embeddings add row by row
  for l in range(c.layers): x = gpt2.layer(x, gpt2.block(w, f"h.{l}."), mask, c.heads)
  return gpt2.layernorm(x, w["ln_f.weight"], w["ln_f.bias"]) @ f.t(w["wte.weight"])

def cross_entropy(logits, onehot):  # the mean of -log p(target) over positions, where onehot picks out the targets
  return (logits.log_softmax() * onehot).sum() * (-1 / logits.rows)

B1, B2, EPS = 0.9, 0.999, 1e-8  # Adam's defaults in PyTorch
LR = 3e-3
def adam(params, state, lr, bias2):
  """Update parameters and optimizer state using PyTorch's Adam equations.
  lr is the bias-corrected learning rate, LR / (1 - B1^step). bias2 is 1 / sqrt(1 - B2^step).
  Both are 1 x 1 tensors so their values can change between recorded steps."""
  for p in params:
    g, (m, v) = p.grad.realize(), state[id(p)]
    m.assign(m * B1 + g * (1 - B1))
    v.assign(v * B2 + g * g * (1 - B2))
    p.assign(p - m * lr / (v.sqrt() * bias2 + EPS))

def train(rt, steps, log=None, data=None):
  """Train for steps steps on baretile and compare each step with PyTorch on the same weights.
  Also train an independent PyTorch copy from the same initial weights. Small rounding differences
  can make the two loss curves diverge: Adam's first update depends on the gradient's sign, which
  matmul error can change for gradients near zero.

  Return a tuple per step: baretile loss, PyTorch loss on the same weights, maximum relative gradient
  error, and the independent PyTorch loss. Normalize each weight's gradient error by its largest
  reference gradient. data defaults to the text of README.md and docs/."""
  import torch
  data = data or text()
  chars = sorted(set(data))
  c = SimpleNamespace(ctx=64, dim=64, heads=2, layers=2)
  c.vocab, ids, rng = len(chars), np.array([chars.index(ch) for ch in data]), np.random.default_rng(1)
  w = init(c)
  mask = gpt2.mask(range(c.ctx), c.ctx)
  tw = {k: Tensor.of(rt, a.tolist(), "fp32").requires_grad_() for k, a in w.items()}
  state = {id(t): [Tensor.full(rt, t.rows, t.cols, 0.0, "fp32") for _ in range(2)] for t in tw.values()}
  pw = {k: torch.tensor(a, requires_grad=True) for k, a in w.items()}
  opt = torch.optim.Adam(pw.values(), lr=LR)
  tmask, pmask, losses = Tensor.of(rt, mask.tolist(), "fp32"), torch.tensor(mask), []
  @tensor.jit  # records on the second call and replays after that, so the host only copies the inputs in
  def train_step(x, onehot, step_lr, bias2):  # returns the loss, after backward() fills .grad and Adam updates the weights
    for t in tw.values(): t.grad = None
    loss = cross_entropy(forward(x, tw, tmask, c), onehot)
    loss.backward()
    adam(tw.values(), state, step_lr, bias2)
    return loss
  for step in range(1, steps + 1):
    start = time.time()
    at = int(rng.integers(0, len(ids) - c.ctx - 1))
    x, y = ids[at:at + c.ctx].tolist(), ids[at + 1:at + c.ctx + 1].tolist()
    same = {k: torch.tensor(gpt2.values(t), requires_grad=True) for k, t in tw.items()}  # PyTorch on baretile's current weights
    loss = train_step(Tensor.of(rt, [x], "int32"), Tensor.of(rt, np.eye(c.vocab, dtype=np.float32)[y].tolist(), "fp32"),
                      Tensor.full(rt, 1, 1, LR / (1 - B1**step), "fp32"), Tensor.full(rt, 1, 1, 1 / math.sqrt(1 - B2**step), "fp32"))
    want = torch.nn.functional.cross_entropy(forward(x, same, pmask, c), torch.tensor(y))
    want.backward()
    grads = max(float(np.abs(gpt2.values(t.grad) - same[k].grad.numpy()).max() / np.abs(same[k].grad.numpy()).max()) for k, t in tw.items())
    opt.zero_grad()
    own = torch.nn.functional.cross_entropy(forward(x, pw, pmask, c), torch.tensor(y))
    own.backward()
    opt.step()
    losses.append((loss.tolist()[0][0], want.item(), grads, own.item()))
    if log: log(f"step {step}: loss {losses[-1][0]:.4f}, PyTorch's {want.item():.4f} on the same weights (gradients within {grads:.1e}), "
                f"{own.item():.4f} in its own run ({time.time() - start:.1f}s)")
  return losses


def main():
  from baretile import pci
  from baretile.chip import Chip
  from baretile.runtime import Runtime
  chip = Chip(pci.open_device())
  losses = train(Runtime(chip), int(sys.argv[1]) if len(sys.argv) > 1 else 300, log=print)
  mean = lambda i, part: sum(step[i] for step in part) / len(part)
  print(f"at worst, the loss was within {max(abs(f - p) for f, p, _, _ in losses):.0e} of PyTorch's on the same weights and the gradients within "
        f"{max(g for _, _, g, _ in losses):.0e}")
  n = min(50, len(losses) // 2) or 1  # compare the mean loss of the first n steps with the last n
  print(f"mean loss of the first and last {n} steps: {mean(0, losses[:n]):.2f} -> {mean(0, losses[-n:]):.2f}, PyTorch's own run "
        f"{mean(3, losses[:n]):.2f} -> {mean(3, losses[-n:]):.2f}")

if __name__ == "__main__": main()
