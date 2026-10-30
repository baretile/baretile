# Train GPT-2 from scratch on GPT-2 tokens, in fp32 with AdamW. A step adds up the gradients of c.accum sequences of
# c.ctx tokens, clips their norm and updates the weights, and each of those runs as a replayed recording (tensor.jit),
# so the host only sends each sequence's tokens. The tokens come from llm.c's .bin shards, such as FineWeb's
# (docs/models.md#training).
#   python3 -m baretile.models.train train.bin [val.bin] [key=value ...]
# GPT2 has the settings and their defaults: GPT-2 124M's shape, and nanoGPT's schedule and AdamW.
import json, math, sys, time
from types import SimpleNamespace
import numpy as np
from baretile.tensor import Tensor, jit
from baretile.models import gpt2

# 12 blocks 768 wide with 12 heads, on 1,024 tokens, and the vocabulary padded from 50,257 to a multiple of 64. A step
# is 512 sequences, 524,288 tokens, and 19,073 steps read FineWeb's 10B tokens once. The learning rate warms up
# linearly over warmup steps, then follows a cosine down to a tenth at the last step. AdamW decays the matrices and
# embeddings, but not the biases or the layer norms. log names a file for each step's numbers, as JSON lines.
GPT2 = dict(layers=12, dim=768, heads=12, ctx=1024, vocab=50304, accum=512, steps=19073, lr=6e-4, warmup=700, decay=0.1, b1=0.9, b2=0.95,
            eps=1e-8, clip=1.0, val_every=250, val_batches=20, seed=0, log="")

def shard(path):
  """The GPT-2 tokens in an llm.c .bin shard, memory-mapped. A shard has a 1 KiB header of 256 int32s, starting with
  magic 20240520, version 1 and the token count, and then the tokens as uint16s."""
  header = np.fromfile(path, np.int32, 256)
  if len(header) < 3 or header[0] != 20240520 or header[1] != 1: raise ValueError(f"{path} isn't an llm.c token shard (magic 20240520, version 1)")
  return np.memmap(path, np.uint16, "r", offset=1024, shape=(int(header[2]),))

def sequences(tokens, n):  # runs of n + 1 tokens, each starting where the last one's targets end, from the start again after the last
  if len(tokens) < n + 1: raise ValueError(f"{len(tokens)} tokens can't make a sequence of {n} and its next token")
  while True:
    for at in range(0, len(tokens) - n, n): yield tokens[at:at + n + 1].astype(np.int32)

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

def decays(name): return name.endswith(".weight") and ".ln_" not in name and not name.startswith("ln_")  # AdamW decays the matrices

def forward(ids, w, mask, c):  # like gpt2.forward, but returns logits for all positions and transposes wte for the output layer
  f = gpt2.fns(w["wte.weight"])
  x = f.rows(w["wte.weight"], ids) + w["wpe.weight"]  # x has c.ctx rows, as many as wpe, so the position embeddings add row by row
  for l in range(c.layers): x = gpt2.layer(x, gpt2.block(w, f"h.{l}."), mask, c.heads)
  return gpt2.layernorm(x, w["ln_f.weight"], w["ln_f.bias"]) @ f.t(w["wte.weight"])

def cross_entropy(logits, targets):
  """The mean of -log p(target) over positions. targets is an int32 column of token ids, one a row of logits, and the
  chip compares it with every column's id to pick out the targets, so no one-hot matrix crosses PCIe."""
  hot = gpt2.columns(logits.rt, logits.cols).eq(targets).cast("fp32")
  return (logits.log_softmax() * hot).sum() * (-1 / logits.rows)

def norm(grads):  # the total norm of grads, a 1 x 1 tensor
  return sum(((g * g).sum() for g in grads[1:]), (grads[0] * grads[0]).sum()).sqrt()

def clip(total, limit):  # the factor that brings a gradient norm of total down to limit, as torch.nn.utils.clip_grad_norm_ has it
  return ((total + 1e-6).reciprocal() * limit).minimum(1.0)

def adamw(params, grads, state, c, lr, step_lr, bias2, scale=None):
  """Update params and their state [m, v] as PyTorch's AdamW does, from grads times scale, if given. Each maps names to
  tensors, and decays() picks the parameters to decay. lr is this step's rate, step_lr is lr / (1 - b1^step) and
  bias2 is 1 / sqrt(1 - b2^step), each a 1 x 1 tensor so they can change between replays."""
  for name, p in params.items():
    g, (m, v) = grads[name].realize() if scale is None else grads[name] * scale, state[name]
    m.assign(m * c.b1 + g * (1 - c.b1))
    v.assign(v * c.b2 + g * g * (1 - c.b2))
    kept = p * (1 - lr * c.decay) if decays(name) and c.decay else p
    p.assign(kept - m * step_lr / (v.sqrt() * bias2 + c.eps))

def rate(step, c):  # nanoGPT's schedule: a linear warmup over c.warmup steps, then a cosine down to c.lr / 10 at c.steps
  if step <= c.warmup: return c.lr * step / c.warmup
  return c.lr / 10 + 0.45 * c.lr * (1 + math.cos(math.pi * min(1, (step - c.warmup) / max(1, c.steps - c.warmup))))

class Trainer:
  """GPT-2's weights, their gradients and AdamW's state on rt's chip, from init(c, c.seed), and the recorded steps that
  train them. A replay reads and writes the buffers that its recording did, so each sequence adds its gradients onto
  self.grads, and its share of the loss onto self.loss, in place. The update clears both in place."""
  def __init__(self, rt, c):
    self.rt, self.c, self.step = rt, c, 0
    on = lambda a: Tensor.frombytes(rt, *a.shape, a.astype(np.float32).tobytes(), 1, "fp32")
    zeros = lambda t: Tensor.zeros(rt, t.rows, t.cols, dtype="fp32")
    self.w = {k: on(a).requires_grad_() for k, a in init(c, c.seed).items()}
    self.grads, self.state = {k: zeros(t) for k, t in self.w.items()}, {k: [zeros(t), zeros(t)] for k, t in self.w.items()}
    self.loss = Tensor.zeros(rt, 1, 1, dtype="fp32")
    mask = on(gpt2.mask(range(c.ctx), c.ctx))
    @jit
    def accumulate(ids, targets):  # adds this sequence's share of the gradients onto self.grads and of the loss onto self.loss
      for t in self.w.values(): t.grad = None
      share = cross_entropy(forward(ids, self.w, mask, c), targets) * (1 / c.accum)
      share.backward()
      for k, t in self.w.items(): self.grads[k].assign(self.grads[k] + t.grad)
      self.loss.assign(self.loss + share)
      return share  # jit needs a result. The next replay overwrites it, so read self.loss instead
    @jit
    def update(lr, step_lr, bias2):  # clips the gradients, updates the weights and clears the gradients. Returns their norm
      total = norm(list(self.grads.values()))
      adamw(self.w, self.grads, self.state, c, lr, step_lr, bias2, clip(total, c.clip) if c.clip else None)
      for g in [*self.grads.values(), self.loss]: g.assign(zeros(g))
      return total
    self._accumulate, self._update = accumulate, update
    self._evaluate = jit(lambda ids, targets: cross_entropy(forward(ids, self.w, mask, c), targets))

  def tokens(self, run):  # a run of ctx + 1 tokens as the ids to read, a row, and the targets to predict, a column
    rt, ids = self.rt, np.asarray(run, np.int32)
    return Tensor.frombytes(rt, 1, len(ids) - 1, ids[:-1].tobytes(), 1, "int32"), Tensor.frombytes(rt, len(ids) - 1, 1, ids[1:].tobytes(), 1, "int32")

  def train(self, runs):
    """One step on c.accum runs of tokens from runs. Return the mean loss, the gradients' norm before clipping and the
    learning rate."""
    loss = self.gradients(runs)
    return loss, *self.update()

  def gradients(self, runs):  # add up the gradients of c.accum runs of tokens from runs in self.grads, and return their mean loss
    for _ in range(self.c.accum): self._accumulate(*self.tokens(next(runs)))
    return self.loss.tolist()[0][0]

  def update(self):  # clip self.grads, take AdamW's next step with them, and clear them. Return their norm and the learning rate
    self.step, c = self.step + 1, self.c
    lr, number = rate(self.step, c), lambda v: Tensor.full(self.rt, 1, 1, v, "fp32")
    total = self._update(number(lr), number(lr / (1 - c.b1 ** self.step)), number(1 / math.sqrt(1 - c.b2 ** self.step)))
    return total.tolist()[0][0], lr

  def validate(self, tokens, n):  # the mean loss over the first n runs of tokens, the same ones each time, without training on them
    runs = sequences(tokens, self.c.ctx)
    return sum(self._evaluate(*self.tokens(next(runs))).tolist()[0][0] for _ in range(n)) / n

def main():
  from baretile import pci
  from baretile.chip import Chip
  from baretile.runtime import Runtime
  paths = [a for a in sys.argv[1:] if "=" not in a]
  if not paths: sys.exit("usage: python3 -m baretile.models.train train.bin [val.bin] [key=value ...]\n  settings: "
                         + ", ".join(f"{k}={v}" for k, v in GPT2.items()))
  c = SimpleNamespace(**GPT2)
  for a in sys.argv[1:]:
    if "=" in a:
      k, v = a.split("=", 1)
      if k not in GPT2: sys.exit(f"unknown setting {k}. Settings: {', '.join(GPT2)}")
      setattr(c, k, type(GPT2[k])(v))
  if c.dim % c.heads: sys.exit(f"dim={c.dim} must divide into heads={c.heads} heads")
  train, val = sequences(shard(paths[0]), c.ctx), shard(paths[1]) if len(paths) > 1 else None
  trainer, log = Trainer(Runtime(Chip(pci.open_device())), c), open(c.log, "a") if c.log else None
  print(f"GPT-2, {c.layers} blocks {c.dim} wide, on {c.ctx}-token sequences: {c.accum * c.ctx:,} tokens a step for {c.steps:,} steps")
  for step in range(1, c.steps + 1):
    start = time.time()
    loss, total, lr = trainer.train(train)
    took, record = time.time() - start, dict(step=step, loss=loss, norm=total, lr=lr)
    line = f"step {step}: loss {loss:.4f}, lr {lr:.2e}, gradient norm {total:.3f}"
    if val is not None and (step % c.val_every == 0 or step == c.steps):
      record["val"] = trainer.validate(val, c.val_batches)
      line += f", validation loss {record['val']:.4f}"
    print(f"{line} ({took:.1f}s, {c.accum * c.ctx / took:,.0f} tokens/s)")
    if log: log.write(json.dumps(record | dict(seconds=took)) + "\n"), log.flush()

if __name__ == "__main__": main()
