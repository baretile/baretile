# Llama-family text models, run with baretile tensors or NumPy arrays. The command compares their
# greedy choices from pretrained weights. Decode uses a fixed KV cache and replays from its third
# call onward. Several sequences can decode together in separate cache slots.
# Qwen-specific experts, Gated DeltaNet and multi-token prediction are in qwen.py.
#   TTSIM=/path/to/libttsim.so python3 models/llama.py path/to/SmolLM-135M "prompt" [tokens] [bf16|bfp8|bfp4|tt|experts]
#   ... python3 models/llama.py path/to/model path/to/<model>.refpt [steps] [mode] [numpy]
# The folder needs config.json, tokenizer.json and safetensors weights, with an index for sharded weights.
# precision() defines the weight formats and fidelity for each mode. NumPy uses the converted weight
# values too. Append numpy to a reference accuracy command to run it without a chip.
import functools, json, math, os, re, sys, time, unicodedata
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baretile.tensor import Tensor, jit
from models import gpt2, qwen

def fns(x):
  """Adapt NumPy and baretile operations to the same model code. mm accepts fidelity phases for the chip; NumPy
  uses full products. sink_softmax includes gpt-oss sink logits in the denominator without adding values."""
  if isinstance(x, np.ndarray):
    topk = lambda a, k: (np.take_along_axis(a, i := np.argsort(-a, -1, kind="stable")[..., :k], -1), i)
    def sink_softmax(a, sink):
      e = np.exp(a - (m := np.maximum(a.max(-1, keepdims=True), sink)))
      return e / (e.sum(-1, keepdims=True) + np.exp(sink - m))
    return SimpleNamespace(**vars(gpt2.fns(x)), mean=lambda a: a.mean(-1, keepdims=True), rsqrt=lambda a: 1 / np.sqrt(a), topk=topk,
                           silu=lambda a: a / (1 + np.exp(-a)), sigmoid=lambda a: 1 / (1 + np.exp(-a)), repeat=lambda a, n: np.repeat(a, n, 0),
                           mm=lambda a, b, phases, dtype=None: a @ b, experts=qwen.numpy_experts, minimum=np.minimum, maximum=np.maximum,
                           sink_softmax=sink_softmax, exp=np.exp, log=np.log, bf16=lambda a: a, now=lambda a: a, swap=lambda a: a.transpose(1, 0, 2),
                           cat=lambda *xs: np.concatenate(xs, -2), heads=lambda a: a.T[:, :, None], assign=lambda a, b: a.__setitem__(..., b),
                           tri=lambda n, lower=False: (np.tril if lower else np.triu)(np.ones((n, n), np.float32)),
                           parts=lambda a, n: a.reshape(a.shape[0] * n, a.shape[1] // n, a.shape[2]), expand=np.broadcast_to,
                           pad=lambda a, rows: np.pad(a, ((0, 0), (0, rows), (0, 0))), empty=lambda shape: np.empty(shape, np.float32),
                           delta=qwen.numpy_delta, where=np.where)
  sink_softmax = lambda a, sink: (e := (a - (m := a.max(2).maximum(sink))).exp()) / (e.sum(2) + (sink - m).exp())
  return SimpleNamespace(**vars(gpt2.fns(x)), mean=lambda a: a.mean(2), rsqrt=Tensor.rsqrt, silu=Tensor.silu, sigmoid=Tensor.sigmoid,
                         repeat=Tensor.repeat_interleave, mm=lambda a, b, phases, dtype=None: a.matmul(b, dtype, fidelity=phases), topk=Tensor.topk,
                         experts=qwen.chip_experts, where=lambda cond, a, b: cond.where(a, b), minimum=Tensor.minimum, maximum=Tensor.maximum,
                         sink_softmax=sink_softmax, exp=Tensor.exp, log=Tensor.log, bf16=Tensor.bfloat16, now=Tensor.realize,
                         cat=lambda a, *xs: a.cat(*xs, axis=1), heads=lambda a: a.permute(2, 1, 0), swap=lambda a: a.permute(1, 0, 2),
                         assign=lambda a, b: a.assign(b), tri=lambda n, lower=False: triangle(x.rt, n, lower), parts=lambda a, n: a.split(n, 1),
                         pad=lambda a, rows: a.pad(((0, 0), (0, rows), (0, 0))), expand=lambda a, shape: a.expand(*shape), delta=qwen.delta,
                         empty=lambda shape: Tensor(x.rt, shape[1], shape[2], batch=shape[0], dtype="fp32"))

@functools.cache
def triangle(rt, n, lower=False):  # n x n fp32 ones on and above the diagonal (below with lower), made once on the chip
  return Tensor.of(rt, (np.tril if lower else np.triu)(np.ones((n, n))).tolist(), "fp32")

def precision(c, mode="bf16", kv=None):
  """The format and fidelity phases of each matmul's weights, as {(op, block): (format, phases)}.

  Keys:
    (op, block)              op is qkv, o, gate_up or down, for each block
    ("head", None)           the output layer
    ("attention", stage)     stage is "decode" or "prefill". B is the cache, so the format is None
    ("kv", None)             the KV cache's format: kv if given, else bfp8 for tt and experts, else bf16
    ("draft head", None)     experts only: a bfp4 copy of the output layer at one phase, for the multi-token
                             prediction layer to draft with

  Modes:
    bf16, bfp8   the matrices in that format at four phases, which gives exact products
    bfp4         the MLP's gate and up in bfp4 and the rest in bfp8, at four phases
    tt           tt_transformers' performance mode at tt-metal 9f98dee9. The MLP's gate and up are bfp4 at one phase,
                 and the other matrices and the output layer bfp8 at two. Attention takes two phases in decode and
                 four in prefill. For Llama-3.1-8B (32 blocks, 4096 wide) the last block's gate and up are bfp8 at two,
                 as in its model_params/Llama-3.1-8B-Instruct/performance_decoder_config.json.
    experts      Qwen3.6-35B-A3B on one card. The MLP's matrices are bfp4 at one phase, which multiplies all the bits
                 of a bfp4 value. They're the gate, up and down of the experts and of the shared expert. The rest and
                 the output layer are bfp8 at two phases.
  Outside tt's decode, attention takes four phases."""
  if mode not in ("bf16", "bfp8", "bfp4", "tt", "experts"): raise ValueError(f"unknown precision mode: {mode}")
  if kv is not None and kv not in ("bf16", "bfp8", "bfp4"): raise ValueError(f"unknown KV cache format: {kv}")
  four = {"bfp4": ("gate_up",), "tt": ("gate_up",), "experts": ("gate_up", "down")}.get(mode, ())  # the ops in bfp4
  fast = mode in ("tt", "experts")  # one phase for a bfp4 B and two for bfp8, else four
  def entry(op):
    if mode in ("bf16", "bfp8"): return mode, 4
    f = "bfp4" if op in four else "bfp8"
    return f, (1 if f == "bfp4" else 2) if fast else 4
  p = {(op, l): entry(op) for l in range(c.layers + c.mtp) for op in ("qkv", "o", "gate_up", "down")}
  if mode == "tt" and (c.layers, c.dim) == (32, 4096): p["gate_up", 31] = ("bfp8", 2)
  p["head", None] = entry("head")
  if mode == "experts": p["draft head", None] = ("bfp4", 1)
  attention = {("attention", "decode"): (None, 2 if mode == "tt" else 4), ("attention", "prefill"): (None, 4)}
  return p | attention | {("kv", None): (kv or ("bfp8" if fast else "bf16"), None)}

def precision_of(c, op, block=None): return (c.precision or {}).get((op, block), (None, None))  # (format, phases), None where unset

def cache_format(c): return precision_of(c, "kv")[0] or "bf16"  # the KV cache's format

def rms(x, w, c):  # RMSNorm scaled by w. Qwen3.5 scales by 1 + w as n w + n, which stays exact where 1 + w would round in bf16
  f = fns(x)
  n = x * f.rsqrt(f.mean(x * x) + c.eps)
  return n * w + n if c.centered else n * w

def head(c): return c.head_dim or c.dim // c.heads  # the width of a head. Qwen3's config gives it, a Llama's is dim / heads

def attends(c, w):  # whether Tensor.attend can run attention. It needs heads 64 wide or more in a multiple of 32, and no sinks
  return head(c) % 32 == 0 and head(c) >= 64 and not any(k.endswith("sinks") for k in w)

# Hugging Face's RoPE: x cos + (-x2, x1) sin, where x1 and x2 are the two halves of a head. x @ turn gives (-x2, x1).
def rope(x, cos, sin, turn):
  return x * cos + (x @ turn) * sin

def layer(x, p, c, rotary, mask, cache, rows, l=0, prefill=False, live=None, accept=None, slots=None):
  """Block l: attention, then a SwiGLU MLP, each after an RMSNorm and added back. A block whose weights have a conv
  runs Gated DeltaNet in place of attention (qwen.deltanet). With a router, it runs a mixture of experts (qwen.moe).

    p        the block's weights, transposed for x @ w, with q, k and v side by side and gate and up side by side.
             Other families add Qwen2's qkv_bias, Qwen3's qk_norm and qk_on, and gpt-oss's sinks and o_proj.bias
    c        the config. Its precision, if set, gives each matmul's fidelity phases
    rotary   (cos, sin, turn), with the table rows for this call's positions
    cache    the key heads, then the value heads, with a row per position. This call writes at rows. A Gated
             DeltaNet block has its states here instead
    mask     -1e9 where a token must not look (gpt2.mask). On the chip, around's comes with its positions and window
    prefill  a prompt's tokens, not a decode step
    live     from around. A free slot's token picks no expert and leaves its Gated DeltaNet states as they were
  accept goes to qwen.deltanet, and slots comes from forward."""
  f, phases = fns(x), lambda op, block=l: precision_of(c, op, block)[1]  # None means tensor.FIDELITY
  h = rms(x, p["input_layernorm"], c)
  # Each half adds onto the residual stream in its last kernel (the output's matmul, or the experts' sum). f.now
  # realizes the stream there, so later norms don't add up the blocks' outputs again.
  mixed = (qwen.deltanet(h, p, c, cache, phases, live, accept) if "conv" in p
           else attention(h, p, c, rotary, mask, cache, rows, phases, prefill, slots))
  x = f.now(x + mixed)
  h = rms(x, p["post_attention_layernorm"], c)
  return f.now(x + (qwen.moe(h, p, c, phases, live) if "router" in p else mlp(h, p, phases)))

def attention(h, p, c, rotary, mask, cache, rows, phases, prefill, slots=None):
  """Compute attention over this block's cache. RoPE has already scaled q by 1/sqrt(head width).

  With slots, each sequence uses its own cache rows. Each batch entry pairs a key head with a sequence and
  groups its query heads as rows. Key heads are views, so they need no copies. Ordinary attention reads to
  the mask's width. Tensor.attend reads only the rows reached by the token positions."""
  f, gated, hd, rep = fns(h), c.gated, head(c), c.heads // c.kv
  mask, places, window = mask if isinstance(mask, tuple) else (mask, None, 0)  # on the chip, around gives all three
  nh = c.heads + 2 * c.kv + (c.heads if gated else 0)  # Qwen3.5's output gate adds a head for each query head, after v's
  qkv = f.mm(h, p["qkv_proj"], phases("qkv"))
  qkv = f.split(qkv + p["qkv_bias"] if "qkv_bias" in p else qkv, nh)  # Qwen2's bias
  # Qwen3 RMS-norms each head of q and k before RoPE. v's heads and the gates' go through the same kernel, and qk_on
  # leaves them as they are.
  if "qk_norm" in p:
    r = f.rsqrt(f.mean(qkv * qkv) + c.eps)
    qkv = qkv * f.where(p["qk_on"], r * p["qk_norm"] + r if c.centered else r * p["qk_norm"], 1)
  qkv = f.merge(rope(qkv, f.split(rotary[0], nh), f.split(rotary[1], nh), rotary[2]))  # a token's heads side by side again
  heads = f.split(qkv, nh)
  kv = heads[c.heads:c.heads + 2 * c.kv, :, :]  # this call's keys and values together. RoPE left v's heads as they were
  rounded = isinstance(kv, np.ndarray) and cache_format(c) != "bf16"  # numpy rounds them like a block-float cache
  f.put(cache, rows, gpt2.bfp(kv, cache_format(c)) if rounded else kv)
  attention, n = phases("attention", "prefill" if prefill else "decode"), h.shape[-2]
  q, k, v = heads[:c.heads, :, :], cache[:c.kv, :, :], cache[c.kv:, :, :]
  sinks = p.get("sinks")  # gpt-oss: each head has a sink, a logit that counts in the softmax but has no value
  if slots:  # a batch entry for each key head and sequence, with its query heads' rows side by side
    k, v, t = f.parts(k, slots), f.parts(v, slots), n // slots
    q = f.split(qkv[:, :c.heads * hd], c.kv).reshape(c.kv * slots, t * rep, hd)
    if sinks is not None: sinks = f.expand(sinks.reshape(c.kv, 1, rep), (c.kv, n, rep)).reshape(c.kv * slots, t * rep, 1)
  if (m := mask.shape[-1]) < k.shape[-2]: k, v = k[:, :m, :], v[:, :m, :]  # views of the first m rows
  if not slots: k, v = f.repeat(k, rep), f.repeat(v, rep)  # a key head for each query head, in place
  if places is not None and attends(c, p):  # one kernel, which keeps the scores on the chip
    o = q.attend(k, v, places, window, attention)
  else:
    s = f.mm(q, f.t(k), attention)
    o = f.mm(f.softmax(s + mask) if sinks is None else f.sink_softmax(s + mask, sinks), v, attention)
  if slots: o = o.reshape(c.kv, n, rep * hd)  # each key head's query heads side by side again
  if gated: o = o * f.sigmoid(f.split(qkv[:, (c.heads + 2 * c.kv) * hd:], o.shape[0]))
  o = f.mm(f.merge(o), p["o_proj"], phases("o"))
  return o + p["o_proj.bias"] if "o_proj.bias" in p else o

def mlp(h, p, phases, pre=""):  # SwiGLU: silu(h @ gate) * (h @ up), then down. gate_up_proj holds gate and up side by side
  f, m = fns(h), p[pre + "down_proj"].shape[-2]
  gate_up = f.mm(h, p[pre + "gate_up_proj"], phases("gate_up"))
  return f.mm(glu(gate_up[:, :m], gate_up[:, m:]), p[pre + "down_proj"], phases("down"))

def glu(gate, up, limit=None):  # silu(gate) * up. A limit makes it gpt-oss's, which clamps up both ways and gate from above
  f = fns(gate)
  if limit is None: return f.silu(gate) * up
  gate = f.minimum(gate, limit)
  return gate * f.sigmoid(gate * 1.702) * (f.minimum(f.maximum(up, -limit), limit) + 1)

def turning(c): return int(head(c) * (c.partial or 1))  # how many dimensions of a head RoPE turns: all, or Qwen3.5's first quarter

def frequencies(c):
  """RoPE's frequency for each pair of turning dimensions, in fp32, computed the way Hugging Face does it.
  Llama 3's scaling (rope_type llama3) slows long wavelengths down by its factor, keeps short ones, and blends those in
  between. YaRN, which gpt-oss uses, keeps short wavelengths, slows long ones by its factor, and ramps between them."""
  hd = turning(c)
  inv = 1 / c.theta ** (np.arange(0, hd, 2, dtype=np.float32) / hd)
  if (r := c.rope) and r.get("rope_type") == "llama3":
    old, wavelen = r["original_max_position_embeddings"], 2 * math.pi / inv
    scaled = np.where(wavelen > old / r["low_freq_factor"], inv / r["factor"], inv)
    smooth = (old / wavelen - r["low_freq_factor"]) / (r["high_freq_factor"] - r["low_freq_factor"])
    between = (wavelen >= old / r["high_freq_factor"]) & (wavelen <= old / r["low_freq_factor"])
    inv = np.where(between, (1 - smooth) * scaled / r["factor"] + smooth * scaled, scaled).astype(np.float32)
  if r and r.get("rope_type") == "yarn":
    # the dimension that turns n times over the original context length
    turns = lambda n: hd * math.log(r["original_max_position_embeddings"] / (n * 2 * math.pi)) / (2 * math.log(c.theta))
    low, high = turns(r.get("beta_fast") or 32), turns(r.get("beta_slow") or 1)
    if r.get("truncate", True): low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, hd - 1)
    ramp = np.clip((np.arange(hd // 2, dtype=np.float32) - low) / (high - low if high != low else 0.001), 0, 1)
    inv = (inv / r["factor"] * ramp + inv * (1 - ramp)).astype(np.float32)
  return inv

def rotary(c, n):
  """The cos and sin tables for positions 0 to n - 1, and turn. Like Hugging Face's, the angles are fp32 and the tables
  bf16.

  The tables repeat a head's row across the heads of q, k and v, so one kernel turns them all. v's heads, Qwen3.5's
  gate heads and the dimensions that don't turn get cos 1 and sin 0. q's rows also carry attention's 1/sqrt(head width),
  so q comes out of RoPE already scaled. The scaling is exact when 1/sqrt(head width) is a power of 2. x @ turn is
  (-x2, x1) in the turning dimensions and 0 past them. It only moves values of x and flips signs, so it's exact too."""
  hd, rd, inv = head(c), turning(c), frequencies(c)
  angles = np.arange(n, dtype=np.float32)[:, None] * inv[None]
  bf16 = lambda a: gpt2.bf16_floats(gpt2.bf16_bits(a))
  turn, half = np.zeros((hd, hd), np.float32), np.arange(rd // 2)
  turn[half + rd // 2, half], turn[half, half + rd // 2] = -1, 1
  still = c.kv + (c.heads if c.gated else 0)
  heads = lambda t, v: np.concatenate([t * np.float32(hd ** -0.5)] * c.heads + [t] * c.kv + [np.full_like(t, v)] * still, 1)
  r = c.rope or {}  # YaRN scales the tables up by its attention factor
  scale = r.get("attention_factor") or (0.1 * math.log(r["factor"]) + 1 if r.get("rope_type") == "yarn" and r["factor"] > 1 else 1)
  table = lambda g, v: np.concatenate([g(angles) * np.float32(scale)] * 2 + [np.full((n, hd - rd), v, np.float32)], 1)
  return [bf16(heads(table(g, v), v)) for g, v in ((np.cos, 1), (np.sin, 0))] + [turn]

class Weight:
  """Read a checkpoint matrix on demand so the host doesn't need the whole model in memory. bits() returns bf16
  bits arranged for x @ w; values() returns those values as float32."""
  def __init__(self, bits): self.bits = bits
  def values(self): return gpt2.bf16_floats(self.bits())

def rope_weights(c, n):  # rotary's cos, sin and turn for n positions, as weights to upload
  return {name: Weight(lambda a=a: gpt2.bf16_bits(a)) for name, a in zip(("cos", "sin", "turn"), rotary(c, n))}

def fuse(w, c):
  """Rearrange each block's weights for fewer, wider matmuls.

  - q, k and v become one matrix, and so do gate and up. Qwen3.5's output gates go after v.
  - A mixture of experts gets its router transposed for x @ w, and its experts' gate_up and down as two batches.
  - Qwen3's q and k norms become a batch with a row per head, 1s for v's, and qk_on says which heads to norm.
  - Gated DeltaNet's projections become one matrix, and its conv's taps a batch."""
  cat = lambda xs: np.ascontiguousarray(np.concatenate(xs, 1))
  join = lambda *xs: Weight(lambda: cat([x.bits() for x in xs])) if isinstance(xs[0], Weight) else cat(xs)
  stack = lambda xs: Weight(lambda: np.stack([x.bits() for x in xs])) if isinstance(xs[0], Weight) else np.stack(xs)
  const = lambda shape, v: Weight(lambda: np.full(shape, gpt2.bf16_bits(np.float32(v)), np.uint16))
  swap = lambda a: np.ascontiguousarray(np.swapaxes(a, -1, -2))
  turned = lambda x: Weight(lambda: swap(x.bits())) if isinstance(x, Weight) else swap(x)  # a batch's (out x in) matrices, for x @ w
  gated, hd = c.gated, head(c)
  for pre in {k[:k.index(".", 7) + 1] for k in w if k.startswith("layers.")}:
    if pre + "conv1d" in w:  # Gated DeltaNet: q, k and v, then z, b and a, in one matmul
      w[pre + "qkv_proj"] = join(*(w.pop(pre + "in_proj_" + n) for n in ("qkv", "z", "b", "a")))
      w[pre + "o_proj"], conv, A = w.pop(pre + "out_proj"), w.pop(pre + "conv1d"), w.pop(pre + "A_log")
      w[pre + "conv"] = Weight(lambda conv=conv: np.ascontiguousarray(conv.bits().reshape(-1, c.conv).T.reshape(c.conv, 1, -1)))
      w[pre + "decay"] = Weight(lambda A=A: gpt2.bf16_bits(-np.exp(A.values())))
    else:
      q = w.pop(pre + "q_proj")
      if gated:  # q_proj has each head's query followed by its gate
        part = lambda i, q=q: Weight(lambda: np.ascontiguousarray(q.bits().reshape(-1, c.heads, 2, hd)[:, :, i].reshape(-1, c.heads * hd)))
        q, gate = part(0), part(1)
      w[pre + "qkv_proj"] = join(q, w.pop(pre + "k_proj"), w.pop(pre + "v_proj"), *([gate] if gated else []))
    if pre + "q_proj.bias" in w: w[pre + "qkv_bias"] = join(*(w.pop(pre + n) for n in ("q_proj.bias", "k_proj.bias", "v_proj.bias")))
    if pre + "shared_expert_gate" in w:  # Qwen2-MoE's shared expert, and its gate
      w[pre + "shared_expert.gate_up_proj"] = join(w.pop(pre + "shared_expert.gate_proj"), w.pop(pre + "shared_expert.up_proj"))
      w[pre + "shared_expert_gate"] = turned(w[pre + "shared_expert_gate"])
    if pre + "q_norm" in w:
      q, k = w.pop(pre + "q_norm"), w.pop(pre + "k_norm")
      still = c.kv + (c.heads if gated else 0)  # v's heads and the gate heads, which aren't normed
      w[pre + "qk_norm"] = stack([q] * c.heads + [k] * c.kv + [const((1, hd), 1)] * still)
      w[pre + "qk_on"] = stack([const((1, 1), 1)] * (c.heads + c.kv) + [const((1, 1), 0)] * still)
    if pre + "sinks" in w: w[pre + "sinks"] = Weight(lambda s=w[pre + "sinks"]: s.bits().reshape(-1, 1, 1))  # gpt-oss: a sink for each head
    router = w.pop(pre + "router") if pre + "router" in w else w.pop(pre + "gate", None)  # gpt-oss calls it router, Qwen and Mixtral gate
    if router is None:  # a SwiGLU
      w[pre + "gate_up_proj"] = join(w.pop(pre + "gate_proj"), w.pop(pre + "up_proj"))
      continue
    w[pre + "router"] = turned(router)
    if pre + "experts.gate_up_proj_bias" in w:  # gpt-oss's experts: a batch already laid out for x @ w, gate and up interleaved
      apart = lambda x: Weight(lambda: np.ascontiguousarray(np.concatenate([x.bits()[..., ::2], x.bits()[..., 1::2]], -1)))
      w[pre + "experts.gate_up"] = apart(w.pop(pre + "experts.gate_up_proj"))
      w[pre + "experts.gate_up_bias"] = apart(w.pop(pre + "experts.gate_up_proj_bias"))
      w[pre + "experts.down"], w[pre + "experts.down_bias"] = w.pop(pre + "experts.down_proj"), w.pop(pre + "experts.down_proj_bias")
    elif pre + "experts.gate_up_proj" in w:  # already batches, which is how transformers 5 stores them (Qwen3.5)
      w[pre + "experts.gate_up"], w[pre + "experts.down"] = turned(w.pop(pre + "experts.gate_up_proj")), turned(w.pop(pre + "experts.down_proj"))
    else:  # separate matrices for each expert (Qwen3, Mixtral)
      n = sum(k.startswith(pre + "experts.") and k.endswith(".down_proj") for k in w)
      expert = lambda name: [w.pop(f"{pre}experts.{e}.{name}") for e in range(n)]
      gates, ups = expert("gate_proj"), expert("up_proj")
      w[pre + "experts.gate_up"], w[pre + "experts.down"] = stack([join(g, u) for g, u in zip(gates, ups)]), stack(expert("down_proj"))
  return w

def around(w, c, places, size, slots=None, length=None):
  """Return (rows, rotary, mask, window, live) for tokens at places. rows selects cache positions; rotary
  selects RoPE rows. mask covers size positions, window is the sliding-window mask or None, and live marks
  active tokens (None means all active).

  With slots, each sequence owns size / slots cache rows and supplies the same number of tokens. The last
  row is reserved for a free slot: live is zero there and attention sees only the first cache row. length
  limits ordinary masks to a prefix of each slot. On the chip, masks also carry positions and window bounds
  for Tensor.attend, which reads only visible rows."""
  numpy = not isinstance(places, Tensor)
  f, places = fns(w["norm"]), np.asarray(places) if numpy else places
  rotary, n, live = (f.rows(w["cos"], places), f.rows(w["sin"], places), w["turn"]), places.shape[-1], None
  at = places.reshape(-1, 1) if numpy else places.reshape(1, n, 1)  # each token's position, as a column
  if not slots: rows, width, column = places, length or size, at
  else:
    part, rep = size // slots, c.heads // c.kv
    rows = places + firsts(None if numpy else places.rt, n, slots, part)
    live = f.now(f.bf16(at < part - 1))  # bf16, like the expert weights it multiplies
    seen = f.where(at < part - 1, at, 0)  # a free slot's token sees only the first row
    column = f.expand(seen.reshape(1, n, 1), (1, n, rep)).reshape(1, n * rep, 1)
    column = f.expand(column, (c.kv, n * rep, 1)).reshape(c.kv * slots, n // slots * rep, 1)  # rows for each key head and sequence
    width = length or part
  masks = lambda win: gpt2.masked(column, width, win) if numpy else (gpt2.masked(column, width, win), column, win or 0)
  return rows, rotary, masks(None), c.sliding and masks(c.sliding), live

# The first cache row of each token's sequence, with n / slots tokens of each sequence in turn. On the chip it's an int32
# row.
@functools.cache
def firsts(rt, n, slots, part):
  row = [i // (n // slots) * part for i in range(n)]
  return np.asarray(row) if rt is None else Tensor.of(rt, [row], "int32")

def forward(ids, places, w, c, caches, slots=None, every=False, accept=None, hidden=False, length=None, capacity=None):
  """Return logits after processing ids at places, updating each block's cache. Each token sees positions
  up to its own. NumPy takes lists for ids and places; baretile takes int32 rows. Weight lookups happen
  one block at a time so NumPy doesn't need the whole model in memory.

  slots: decode several sequences together, with an equal number of consecutive tokens per sequence.
  Each sequence has its own cache region. An inactive slot uses its region's reserved final row
  (see around). With slots, return logits for every token.
  every: return fp32 logits for every token even without slots; otherwise return only the last token's.
  accept: select the DeltaNet state after an accepted or rejected draft (see qwen.deltanet).
  hidden: also return normalized hidden states for qwen.draft.
  length: limit ordinary attention to this many cache rows per sequence. Use a multiple of 32 beyond
  the current positions. Tensor.attend uses positions to limit its reads regardless of length.
  capacity: positions per sequence, needed for all-linear stacks whose slot capacity is smaller than
  the RoPE tables. Otherwise capacity comes from an attention cache, or from the tables if none exists."""
  f = fns(w["norm"])
  size = next((k.shape[-2] for k in caches if not isinstance(k, list)), None)
  table = min(w[k].shape[-2] for k in ("cos", "sin"))
  if capacity is not None and (type(capacity) is not int or not 0 < capacity <= table or size is not None and size != capacity * (slots or 1)):
    raise ValueError("forward capacity must fit RoPE tables and match attention cache rows per sequence")
  size = size if size is not None else (capacity or table) * (slots or 1)
  x = f.rows(w["embed"], ids)
  rows, rotary, mask, window, live = around(w, c, places, size, slots, length)
  prefill = slots is None and x.shape[-2] > 1
  for l in range(c.layers):
    seen = window if window is not None and windowed(c, l) else mask
    x = layer(x, gpt2.block(w, f"layers.{l}."), c, rotary, seen, caches[l], rows, l, prefill, live, accept, slots)
  h = rms(x, w["norm"], c)
  logits = f.mm(h if slots or every else f.rows(h, [x.shape[-2] - 1]), w["embed.T"], precision_of(c, "head")[1], "fp32")  # as the matmul sums them
  return (logits, h) if hidden else logits

def prefill(ids, w, c, caches, tokens, on=lambda row: row):
  """Run a prompt from position 0 in chunks of tokens, returning logits after the final token. Each chunk
  updates caches before the next runs. Attention scores then have one row per chunk token instead of one per
  prompt token. on converts IDs and positions to the representation used by forward, an int32 tensor on
  baretile."""
  for i in range(0, len(ids), tokens): out = forward(on(ids[i:i + tokens]), on(list(range(i, min(i + tokens, len(ids))))), w, c, caches)
  return out

def kv_cache(c, size, zeros, slots=1, speculative=False):
  """Allocate caches for the blocks, including the prediction block, using zeros(shape). Attention caches store
  key heads followed by value heads, with size rows. DeltaNet caches hold [conv history, S] for slots
  sequences and use zeros(shape, "float32"). Speculative decoding keeps two copies of those states."""
  states = ((c.conv - 1, slots, 2 * c.kh * c.kd + c.vh * c.vd), (slots * c.vh, c.kd, c.vd)) if c.kh else ()
  def cache(l):
    if not linear(c, l): return zeros((2 * c.kv, size, head(c)))
    return [zeros(shape, "float32") for _ in range(1 + speculative) for shape in states]
  return [cache(l) for l in range(c.layers + c.mtp)]

def linear(c, l): return l < len(c.layer_types or ()) and c.layer_types[l] == "linear_attention"  # is block l a Gated DeltaNet block?

def windowed(c, l): return not c.layer_types or c.layer_types[l] == "sliding_attention"  # whether block l keeps to a sliding window, if any

def slot(c, caches, u, slots):
  """View sequence u in caches allocated for slots sequences. Select its KV rows and DeltaNet state.
  Convolution history uses a Row wrapper because it has a row per sequence."""
  part = lambda kv: kv[:, u * kv.shape[-2] // slots:(u + 1) * kv.shape[-2] // slots, :]
  conv = lambda s: s[:, u:u + 1, :] if isinstance(s, np.ndarray) else qwen.Row(s, u)
  return [[conv(s) if i % 2 == 0 else s[u * c.vh:(u + 1) * c.vh, :, :] for i, s in enumerate(kv)] if linear(c, l) else part(kv)
          for l, kv in enumerate(caches)]

def read_json(path):
  with open(path, encoding="utf-8") as f: return json.load(f)

# The config fields past a Llama's dim, heads, kv, layers, eps, theta, vocab, ffn and places. The defaults give a Llama
# with none of the extras.
DEFAULTS = dict(head_dim=None, rope=None, partial=None, sliding=None, layer_types=None, gated=False, centered=False, k=0, experts=0, moe=0,
                shared=0, dense=(), norm_topk=True, limit=None, conv=None, kh=None, vh=None, kd=None, vd=None, mtp=0, tied=False, precision=None)
def model(**c): return SimpleNamespace(**DEFAULTS | c)  # a config with the given fields, and DEFAULTS for the rest

def config(folder, places=4096):
  """Read config.json, selecting the text config for a multimodal model. Cap places at the model's maximum when
  sizing RoPE tables."""
  j = read_json(os.path.join(folder, "config.json"))
  j = j.get("text_config", j)
  rope = j.get("rope_parameters") or (j.get("rope_scaling") or {}) | {"rope_theta": j["rope_theta"]}  # transformers 5, else the older two
  theta, kind = rope.pop("rope_theta"), rope.pop("type", None)  # older configs say type instead of rope_type
  partial = rope.pop("partial_rotary_factor", j.get("partial_rotary_factor"))
  for k in ("mrope_section", "mrope_interleaved"): rope.pop(k, None)  # Qwen3.5's. A text token's three positions are all the same
  qwen35 = j.get("model_type", "").startswith(("qwen3_5", "qwen3_next"))  # its RMSNorms scale by 1 + w (centered), and its attention is gated
  rope.setdefault("rope_type", kind or "default")
  if rope["rope_type"] not in ("default", "llama3", "yarn"): raise ValueError(f"RoPE scaling {rope} isn't supported")
  experts, layers = j.get("num_experts") or j.get("num_local_experts") or 0, j["num_hidden_layers"]
  return model(dim=j["hidden_size"], heads=j["num_attention_heads"], kv=j["num_key_value_heads"], layers=layers, eps=j["rms_norm_eps"], theta=theta,
               places=min(j["max_position_embeddings"], places), vocab=j["vocab_size"], ffn=j.get("intermediate_size"), head_dim=j.get("head_dim"),
               rope=rope if rope["rope_type"] != "default" else None, partial=partial, layer_types=j.get("layer_types"),
               sliding=j.get("sliding_window") if j.get("use_sliding_window", True) else None,
               experts=experts, k=j.get("num_experts_per_tok") or 0,
               moe=j.get("moe_intermediate_size") or (experts and j.get("intermediate_size")) or 0,
               shared=j.get("shared_expert_intermediate_size") or 0, norm_topk=j.get("norm_topk_prob", True),
               dense=[l for l in range(layers) if l in j.get("mlp_only_layers", []) or (l + 1) % j.get("decoder_sparse_step", 1)],
               limit=j.get("swiglu_limit", 7.0 if j.get("model_type") == "gpt_oss" else None),  # gpt-oss clamps its GLU
               gated=j.get("attn_output_gate", qwen35), centered=qwen35, conv=j.get("linear_conv_kernel_dim"),  # Qwen3.5's
               kh=j.get("linear_num_key_heads"), vh=j.get("linear_num_value_heads"),
               kd=j.get("linear_key_head_dim"), vd=j.get("linear_value_head_dim"),
               mtp=j.get("mtp_num_hidden_layers", 0), tied=j.get("tie_word_embeddings", False))

def load(folder, places=4096):
  """Return (config, {name: Weight}) from model.safetensors or the shards listed in model.safetensors.index.json.

  Names drop "model.". Linear weights are transposed for x @ w, and vectors become one-row matrices. The output layer
  is the embedding unless the model has its own lm_head. fuse joins the matrices and stacks a mixture's experts. RoPE's
  tables have a row for each of the first `places` positions, capped at the model's maximum. A model that also sees
  images, like Qwen3.5 and Qwen3.6, is read as its language model without the vision tower. A multi-token prediction
  layer becomes a block after the last."""
  c = config(folder, places)
  index = os.path.join(folder, "model.safetensors.index.json")
  files = sorted(set(read_json(index)["weight_map"].values())) if os.path.exists(index) else ["model.safetensors"]
  out, tensors = {}, {}
  for file in files:
    header, data = gpt2.safetensors(os.path.join(folder, file))
    tensors |= {name: (data, t) for name, t in header.items() if name != "__metadata__"}
  for full, (data, t) in tensors.items():
    if full.endswith("_scales"): continue  # an MXFP4 tensor's scales, read with its blocks
    name = full.removeprefix("model.").removeprefix("language_model.").removesuffix(".weight")
    for part in ("self_attn.", "linear_attn.", "mlp.", "block_sparse_moe."): name = name.replace(part, "")
    name = re.sub(r"(experts\.\d+)\.w([123])$", lambda m: m[1] + (".gate_proj", ".down_proj", ".up_proj")[int(m[2]) - 1], name)  # Mixtral's
    if name.startswith("visual."): continue
    if name.startswith("layers.") and int(name.split(".")[1]) >= c.layers: continue  # a config cut down to its first blocks
    name = re.sub(r"^mtp\.layers\.(\d+)\.", lambda m: f"layers.{c.layers + int(m[1])}.", name)  # multi-token prediction as blocks after the last
    if name.endswith("_blocks"):
      scales = tensors[full.removesuffix("_blocks") + "_scales"]
      out[name.removesuffix("_blocks")] = Weight(lambda b=(data, t), s=scales: qwen.mxfp4(gpt2.u8(*b), gpt2.u8(*s)))
    else:
      turn = bool(re.search(r"_proj$|in_proj_|^lm_head$|^mtp\.fc$", name)) and len(t["shape"]) == 2  # transpose linear layers for x @ w
      out[name] = Weight(functools.partial(gpt2.tensor_bits, data, t, turn))
  c.mtp = len({k.split(".")[1] for k in out if k.startswith("layers.") and int(k.split(".")[1]) >= c.layers})  # the multi-token prediction blocks
  out["embed"] = out.pop("embed_tokens")
  out["embed.T"] = out.pop("lm_head", None) or Weight(lambda e=out["embed"]: np.ascontiguousarray(e.bits().T))  # tied, or its own
  out |= rope_weights(c, c.places)
  return c, fuse(out, c)

def weight_format(name, prec):
  """The format of a weight on the chip. Matrices that matmuls read as B take their format from prec (see precision).
  Those are the blocks' matrices and the output layer, and experts take the format of their block's MLP. Gated
  DeltaNet's conv, decay, dt_bias and norm are fp32, since it computes with them in fp32. The rest are bf16, and so are
  the matrices when there's no prec."""
  op = name.split(".")[-1].removesuffix("_proj")
  if name.startswith("layers.") and op in ("conv", "decay", "dt_bias", "norm"): return "fp32"
  key = {"embed.T": ("head", None), "embed.T draft": ("draft head", None)}.get(name) or ((op, int(name.split(".")[1])) if matrix(name) else None)
  return prec[key][0] if prec and key in prec else "bf16"

def matrix(name): return name.endswith(("_proj", "experts.gate_up", "experts.down"))  # a block's matrix, read only as B

def drafting(weights, prec):  # the weights, plus a copy of the output layer for the draft when prec has a draft head
  return weights | {"embed.T draft": weights["embed.T"]} if prec and ("draft head", None) in prec and "mtp.fc" in weights else weights

def upload(rt, weights, prec=None):
  """Upload weights one at a time as baretile tensors. Convert each weight to its requested format on
  the chip and wait before uploading the next, allowing its temporary bf16 storage to be reused.
  Peak storage is the model in its chosen formats plus one bf16 matrix. Store matmul weights by tile
  columns with Tensor.columns so workers can read several tiles per request."""
  t = {}
  for name, w in drafting(weights, prec).items():
    bits = w.bits()
    x = Tensor.frombytes(rt, *bits.shape[-2:], bits.tobytes(), bits.shape[0] if bits.ndim == 3 else 1)
    if (f := weight_format(name, prec)) != "bf16":
      x = x.cast(f).realize()  # rebinding x frees the bf16 copy
      rt.wait(3600)  # casting a big matrix is one launch, and it takes minutes on ttsim
    if matrix(name) or name.startswith("embed.T"):
      x = x.columns()
      rt.wait(3600)
    t[name] = x
  return t

class Made(dict):  # construct a value on each lookup without caching it
  def __getitem__(self, k): return super().__getitem__(k)()

def arrays(weights, prec=None):
  """Return lazily constructed NumPy float32 weights with the values held on the chip. weight_format
  selects each storage format. gpt2.bfp rounds block floats as the chip does, 1024 rows at a time to bound NumPy's
  temporaries.
  Each lookup constructs a new array without caching it, so NumPy holds only the current model block."""
  def made(name, w):
    v, f = w.values(), weight_format(name, prec)
    if f in ("bf16", "fp32"): return v
    rows = v.reshape(-1, v.shape[-1])  # a batch of experts too
    return np.concatenate([gpt2.bfp(rows[i:i + 1024], f).astype(np.float32) for i in range(0, len(rows), 1024)]).reshape(v.shape)
  return Made({name: functools.partial(made, name, w) for name, w in drafting(weights, prec).items()})

def logits(f_out): return gpt2.values(f_out)[0]  # a baretile row of logits, as a numpy vector

def zeros(rt, dtype="bf16"):  # kv_cache's zeros on the chip in dtype, like bfp8 for a bfp8 KV cache. "float32" gives fp32
  return lambda shape, d=None: Tensor.zeros(rt, shape[1], shape[2], shape[0], {"float32": "fp32"}.get(d, dtype))

def reference(path):
  """Read a Tenstorrent .refpt file from tt_transformers/tests/reference_outputs. The file, produced by
  tests/generate_reference_hf.py, contains 1,024 tokens from A Tale of Two Cities and the reference
  model's five most likely next tokens at each position.
  Return (prompt, following tokens, top five): the prompt is the first half, and top five starts at
  the prompt's final position. Entry i therefore scores the prediction made at step i."""
  import torch
  data = torch.load(path)
  tokens = data["reference_tokens"][0].tolist()
  half = len(tokens) // 2
  return tokens[:half], tokens[half:], data["top5_tokens"][half - 1:].tolist()

def accuracy(rt, t, c, prompt, after, top5, steps=500):
  """Measure top-1 and top-5 agreement with Tenstorrent's reference tokens. Prefill the prompt, then feed
  the reference token at each step, regardless of the model's prediction. On the chip, jit records
  and replays these steps. Return (top-1 fraction, top-5 fraction, predicted tokens).

  With rt=None, use NumPy weights from arrays and full-precision products. This separates errors
  from weight conversion from errors in chip arithmetic. Reference tokens are known in advance, so
  NumPy processes the prompt and following tokens together, reading each block's weights once."""
  n, size = len(prompt), -(-(len(prompt) + steps) // 32) * 32
  if rt is None:
    ids = prompt + after[:steps - 1]
    picks = [int(row.argmax()) for row in forward(ids, list(range(len(ids))), t, c, kv_cache(c, size, np.zeros), every=True)[n - 1:]]
  else:
    on, kv = lambda row: Tensor.of(rt, [row], "int32"), kv_cache(c, size, zeros(rt, cache_format(c)))
    replay = jit(lambda ids, places: forward(ids, places, t, c, kv))
    picks = [int(logits(forward(on(prompt), on(list(range(n))), t, c, kv)).argmax())]
    for i in range(steps - 1): picks.append(int(logits(replay(on([after[i]]), on([n + i]))).argmax()))
  return sum(p == f[0] for p, f in zip(picks, top5)) / steps, sum(p in f for p, f in zip(picks, top5)) / steps, picks

WHITE = "\t-\r \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000"  # Unicode's White_Space: what tiktoken's \s matches
SPLIT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""  # SmolLM's: GPT-2's, but with digits split apart
GPT2_SPLIT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""  # what ByteLevel's use_regex splits by

@functools.cache
def categories(names):  # Unicode categories like L or Lu as ranges for a character class. Python's re has no \p{...}
  runs = {n: [] for n in names}
  for cp in range(0x110000):
    cat = unicodedata.category(chr(cp))
    for r in (runs.get(cat[0]), runs.get(cat)):
      if r is None: continue
      if r and r[-1][1] == cp - 1: r[-1][1] = cp
      else: r.append([cp, cp])
  return {k: "".join(re.escape(chr(a)) + (f"-{re.escape(chr(b))}" if b > a else "") for a, b in r) for k, r in runs.items()}

# Converts tokenizer.json's split regex for Python's re. It spells out \p{L}, \p{Lu} and the rest, and \s too, since
# Python's \s also matches four control characters. Inside a character class they go in without brackets.
def python_regex(pat):
  ranges = categories(tuple(sorted(set(re.findall(r"\\p\{(\w+)\}", pat))))) | {"s": WHITE}
  out, inside, i = [], False, 0
  while i < len(pat):
    if pat.startswith(("\\p{", "\\s", "\\S"), i):
      key, i = (pat[i + 3:pat.index("}", i)], pat.index("}", i) + 1) if pat[i + 1] == "p" else (pat[i + 1], i + 2)
      if inside and key == "S": raise ValueError("\\S inside a character class isn't supported")
      out.append(f"[^{WHITE}]" if key == "S" else ranges[key] if inside else f"[{ranges[key]}]")
    else:
      n = 2 if pat[i] == "\\" else 1
      inside = pat[i] == "[" if not inside else pat[i] != "]"
      out.append(pat[i:i + n])
      i += n
  return re.compile("".join(out))

class Tokenizer:
  """Encode and decode byte-level BPE using tokenizer.json, without a tokenizer library.
  Run pre-tokenizer stages in order and apply the configured merges. ignore_merges keeps
  whole vocabulary pieces; files without a merges list use token IDs as ranks, like tiktoken. Special-token
  spellings are ordinary text unless encode is called with special=True. With no unknown token, bytes
  absent from the alphabet are skipped before merging, as in Hugging Face BPE."""
  def __init__(self, folder):
    t = read_json(os.path.join(folder, "tokenizer.json"))
    model = t["model"]
    if model.get("type", "BPE") != "BPE": raise ValueError("Tokenizer needs a byte-level BPE model")
    if model.get("dropout") or model.get("continuing_subword_prefix") or model.get("end_of_word_suffix") or model.get("byte_fallback"):
      raise ValueError("Tokenizer needs deterministic byte-level BPE without subword affixes or byte fallback")
    if model.get("unk_token") is not None: raise ValueError("Tokenizer does not support unknown-token replacement")
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))  # GPT-2's bytes as printable characters
    byte = dict(zip(map(chr, bs + [256 + i for i in range(256 - len(bs))]), bs + [b for b in range(256) if b not in bs]))
    special = {a["content"]: a["id"] for a in t.get("added_tokens", [])}
    encoded = lambda tok: bytes(byte[ch] for ch in tok)
    self.vocab = {encoded(tok): i for tok, i in model["vocab"].items() if tok not in special}
    self.merges = None
    if "merges" in model:
      if not isinstance(model["merges"], list): raise ValueError("Tokenizer needs BPE merges as a list")
      self.merges = {}
      for rank, pair in enumerate(model["merges"]):
        pair = pair.split(" ") if isinstance(pair, str) else pair
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or not all(isinstance(part, str) for part in pair):
          raise ValueError(f"Tokenizer needs BPE merges of two strings: {pair}")
        if any(part not in model["vocab"] for part in (*pair, "".join(pair))):
          raise ValueError(f"Tokenizer needs BPE merges of two vocabulary pieces: {pair}")
        self.merges[tuple(map(encoded, pair))] = rank
    self.ignore_merges = model.get("ignore_merges", self.merges is None)
    self.bytes = {i: b for b, i in self.vocab.items()} | {i: s.encode() for s, i in special.items()}
    self.special, self.specials = special, re.compile("(" + "|".join(map(re.escape, sorted(special, key=len, reverse=True))) + ")")
    pre = t.get("pre_tokenizer") or {}
    patterns = [] if pre else [SPLIT]  # legacy rank-only files without a pre-tokenizer
    for p in pre.get("pretokenizers", [pre]) if pre else ():
      kind = p.get("type")
      if kind == "Split" and "Regex" in p.get("pattern", {}) and not p.get("invert") and p.get("behavior", "Isolated") == "Isolated":
        patterns.append(p["pattern"]["Regex"])
      elif kind == "Digits": patterns.append(r"\p{N}" + ("" if p.get("individual_digits", False) else "+"))
      elif kind == "ByteLevel" and not p.get("add_prefix_space", True):
        if p.get("use_regex", True): patterns.append(GPT2_SPLIT)
      else: raise ValueError(f"Tokenizer does not support pre-tokenizer {p}")
    self.splits, self.pieces = [python_regex("(" + p + ")") for p in patterns], {}
    if any(p.groups != 1 for p in self.splits): raise ValueError("Tokenizer split patterns must not contain capturing groups")
    self.nfc = (t.get("normalizer") or {}).get("type") == "NFC"  # Qwen composes the text first, so an e and its accent become one letter

  def bpe(self, piece):  # a piece's bytes as ids
    if piece not in self.pieces:
      parts = [piece] if self.ignore_merges and piece in self.vocab else [b for i in range(len(piece)) if (b := piece[i:i + 1]) in self.vocab]
      while len(parts) > 1:
        ranks = (self.vocab.get(a + b, math.inf) if self.merges is None else self.merges.get((a, b), math.inf)
                 for a, b in zip(parts, parts[1:]))
        rank, i = min((rank, i) for i, rank in enumerate(ranks))
        if rank == math.inf: break
        parts[i:i + 2] = [parts[i] + parts[i + 1]]
      self.pieces[piece] = [self.vocab[p] for p in parts]
    return self.pieces[piece]
  def encode(self, text, special=False):  # with special, a special token's text becomes that token, as in a chat template
    if special and self.special:
      return [i for part in self.specials.split(text) if part for i in ([self.special[part]] if part in self.special else self.encode(part))]
    pieces = [unicodedata.normalize("NFC", text) if self.nfc else text]
    for split in self.splits: pieces = [part for piece in pieces for part in split.split(piece) if part]
    return [i for piece in pieces for i in self.bpe(piece.encode())]
  def decode(self, ids): return b"".join(self.bytes[i] for i in ids).decode("utf-8", "replace")

def main():
  from baretile import pci
  from baretile.chip import Chip
  from baretile.runtime import Runtime
  folder, prompt, steps = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "The meaning of life is", int(sys.argv[3]) if len(sys.argv) > 3 else 8
  mode = sys.argv[4] if len(sys.argv) > 4 else "bf16"
  if prompt.endswith(".refpt"):  # token accuracy against Tenstorrent's reference file over `steps` steps. Tenstorrent runs 500
    first, after, top5 = reference(prompt)
    c, weights = load(folder, -(-(len(first) + steps) // 32) * 32)
    c.precision, on_host, start = precision(c, mode), "numpy" in sys.argv[5:], time.time()  # numpy has the same weight values and exact products
    if on_host: top1, top5_share, _ = accuracy(None, arrays(weights, c.precision), c, first, after, top5, steps)
    else:
      chip = Chip(pci.open_device())
      rt = Runtime(chip)
      t = upload(rt, weights, c.precision)
      print(f"weights on the chip in {time.time() - start:.0f}s, {mode}")
      top1, top5_share, _ = accuracy(rt, t, c, first, after, top5, steps)
    where = " in numpy" if on_host else ""
    return print(f"token accuracy over {steps} steps after a {len(first)}-token prompt{where}: "
                 f"top-1 {100 * top1:.1f}%, top-5 {100 * top5_share:.1f}% ({time.time() - start:.0f}s)")
  enc, chip = Tokenizer(folder), Chip(pci.open_device())
  rt = Runtime(chip)
  start = time.time()
  ids = enc.encode(prompt)
  size = -(-(len(ids) + steps) // 32) * 32  # rows for the positions the generation reaches, rounded up to a multiple of 32
  c, weights = load(folder, size)
  c.precision = precision(c, mode)
  t, w = upload(rt, weights, c.precision), arrays(weights, c.precision)
  count = sum(x.batch * x.rows * x.cols for k, x in t.items() if k not in ("cos", "sin", "turn"))  # counts a tied output layer twice
  print(f"{count / 1e6:.0f}M weights on the chip in {time.time() - start:.0f}s")
  kv, np_kv = kv_cache(c, size, zeros(rt, cache_format(c))), kv_cache(c, size, np.zeros)
  on = lambda row: Tensor.of(rt, [row], "int32")  # a row of ids or positions, on the chip
  step = jit(lambda ids, places: forward(ids, places, t, c, kv))  # a token per step, replayed from the third step on
  chip = lambda ids, places, i: logits(forward(on(ids), on(places), t, c, kv) if i == 0 else step(on(ids), on(places)))
  gpt2.generate(enc, ids, steps, chip, lambda ids, places: forward(ids, places, w, c, np_kv)[0])

if __name__ == "__main__": main()
