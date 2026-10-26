# GPT-2 small on baretile with OpenAI's pretrained weights. It generates greedily and checks each step against the same
# model in numpy, which uses the same bf16 weights with float32 math. The model is written once and runs on baretile
# Tensors and numpy arrays. The KV cache has a fixed size and a token's position is data, so decode steps launch the
# same kernels, and from the third step on they're replays (tensor.jit).
#   TTSIM=/path/to/libttsim.so python3 models/gpt2.py path/to/model.safetensors "prompt" [tokens] [bfp8]
# The weights are model.safetensors from Hugging Face's openai-community/gpt2, and the tokenizer is tiktoken's "gpt2".
# With bfp8, the weight matrices are bfp8 on the chip, and numpy uses the values the chip holds.
import functools, json, math, os, struct, sys, time
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baretile.tensor import Tensor, jit

HEADS = 12

def bf16_bits(a, packer=False):  # to bf16 bits, rounded to nearest even like uploads, or half up like the packer
  b = np.asarray(a, np.float32).view(np.uint32)
  return ((b + (0x8000 if packer else 0x7FFF + (b >> 16 & 1))) >> 16).astype(np.uint16)
def bf16_floats(bits): return (np.asarray(bits).astype(np.uint32) << 16).view(np.float32)  # bf16 bits to float32 values

def bfp(a, dtype="bfp8"):
  """Rounds a to a block float the way the chip does when it casts or writes a bfp8 KV cache, and returns the values.
  It rounds to bf16 first. Then each run of 16 values along a row shares its largest exponent and keeps 7 bits of
  magnitude below it, rounded half up. bfp4 keeps 3 bits, rounded to nearest."""
  bits = bf16_bits(a).astype(np.uint32)
  s, e, m = bits >> 15, bits >> 7 & 255, (bits & 127) | np.where(bits >> 7 & 255, 128, 0)
  runs = lambda x: x.reshape(*x.shape[:-1], -1, 16)
  top = runs(e).max(-1, keepdims=True)
  shift = np.minimum(top - runs(e), 8)  # a shift of 8 or more leaves 0
  q = np.minimum((runs(m) + (1 << shift)) >> (shift + 1), 127)
  if dtype == "bfp4": q = np.where(shift < 4, np.minimum((runs(m) + (16 << shift)) >> (shift + 5), 7), 0) << 4  # rounded to the nearest 3 bits
  return (np.where(runs(s), -1.0, 1.0) * q * 2.0 ** (top.astype(np.int64) - 133)).reshape(np.shape(a))

def safetensors(path):  # the header, with each tensor's dtype, shape and byte range, and the data, memory-mapped
  with open(path, "rb") as f:
    size = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(size))
  return header, np.memmap(path, np.uint8, "r", offset=8 + size)

def u8(data, t): assert t["dtype"] == "U8"; return data[t["data_offsets"][0]:t["data_offsets"][1]].reshape(t["shape"])  # raw bytes, for MXFP4

def tensor_bits(data, t, transpose=False):  # a tensor as bf16 bits. fp32 is rounded to nearest even, and a vector becomes a one-row matrix
  raw = data[t["data_offsets"][0]:t["data_offsets"][1]]
  if t["dtype"] == "F32": bits = bf16_bits(raw.view(np.float32))
  else: assert t["dtype"] == "BF16", t["dtype"]; bits = raw.view(np.uint16)
  bits = bits.reshape(t["shape"] if len(t["shape"]) >= 2 else (1, -1))  # gpt-oss's experts come as a batch of matrices
  return np.ascontiguousarray(bits.T) if transpose else bits

def fns(x):  # what numpy and baretile spell differently
  if isinstance(x, np.ndarray):
    split = lambda a, n: a.reshape(a.shape[0], n, -1).transpose(1, 0, 2)
    softmax = lambda a: (e := np.exp(a - a.max(-1, keepdims=True))) / e.sum(-1, keepdims=True)
    layernorm = lambda a: (a - a.mean(-1, keepdims=True)) / np.sqrt(a.var(-1, keepdims=True) + 1e-5)
    return SimpleNamespace(rows=lambda a, ids: a[list(ids)], split=split, softmax=softmax, layernorm=layernorm,
                           merge=lambda a: a.transpose(1, 0, 2).reshape(a.shape[1], -1), t=lambda a: a.swapaxes(-1, -2),
                           gelu=lambda a: 0.5 * a * (1 + np.tanh(math.sqrt(2 / math.pi) * (a + 0.044715 * a**3))),
                           put=lambda a, places, b: a.__setitem__((slice(None), list(places)), b))
  if type(x).__module__ == "torch":  # the reference in test/train.py
    import torch
    F = torch.nn.functional
    return SimpleNamespace(rows=lambda a, ids: a[list(ids)], softmax=lambda a: a.softmax(-1),
                           split=lambda a, n: a.reshape(a.shape[0], n, -1).transpose(0, 1), merge=lambda a: a.transpose(0, 1).reshape(a.shape[1], -1),
                           layernorm=lambda a: F.layer_norm(a, a.shape[-1:]), t=lambda a: a.transpose(-1, -2),
                           gelu=lambda a: F.gelu(a, approximate="tanh"))
  return SimpleNamespace(rows=Tensor.gather, split=Tensor.split, softmax=Tensor.softmax,
                         layernorm=Tensor.layernorm, merge=Tensor.merge, t=lambda a: a.T, gelu=Tensor.gelu, put=Tensor.put)

def layernorm(x, g, b): return fns(x).layernorm(x) * g + b

def block(w, pre): return {k[len(pre):]: w[k] for k in w if k.startswith(pre)}  # the weights whose names start with pre, pre cut off

def layer(x, p, mask, heads=HEADS, cache=None, places=None):
  """A transformer block: attention, then the MLP, each after a layer norm and added back.

  p has the block's weights, named as in the file without "h.<layer>.". mask is 0 where a token may look and -1e9 where
  it may not. cache has a row per position, the key heads first and then the value heads. One put writes this call's
  keys and values at places, and attention reads all the cache's rows."""
  f, d = fns(x), x.shape[-1]
  hd = d // heads
  qkv = layernorm(x, p["ln_1.weight"], p["ln_1.bias"]) @ p["attn.c_attn.weight"] + p["attn.c_attn.bias"]
  q, k, v = (f.split(qkv[:, i * d:(i + 1) * d], heads) for i in range(3))  # the heads as a batch
  if cache is not None:
    f.put(cache, places, f.split(qkv[:, d:], 2 * heads))
    k, v = cache[:heads, :, :], cache[heads:, :, :]
  s = q @ f.t(k) * (1 / math.sqrt(hd)) + mask
  y = f.merge(f.softmax(s) @ v) @ p["attn.c_proj.weight"]  # the heads side by side, then c_proj
  x = x + (y + p["attn.c_proj.bias"])  # add x last. It's the large one, so it's rounded only once
  h = layernorm(x, p["ln_2.weight"], p["ln_2.bias"]) @ p["mlp.c_fc.weight"] + p["mlp.c_fc.bias"]
  return x + (f.gelu(h) @ p["mlp.c_proj.weight"] + p["mlp.c_proj.bias"])

def forward(ids, w, mask, places=None, caches=None):
  """Returns the logits for the token after ids. The output layer is the token embedding again, transposed (w["wte.T"]).

  places are the positions of ids in the sequence, from 0 if not given. caches has each block's keys and values (see
  kv_cache), for decoding a token at a time. ids and places are lists for numpy and int32 rows for baretile, so the
  positions are data and the steps launch the same kernels (jit)."""
  f, layers = fns(w["wte.weight"]), 1 + max(int(k.split(".")[1]) for k in w if k.startswith("h."))
  x = f.rows(w["wte.weight"], ids) + f.rows(w["wpe.weight"], range(len(ids)) if places is None else places)
  for l in range(layers):
    x = layer(x, block(w, f"h.{l}."), mask, HEADS, caches and caches[l], places)
  return f.rows(layernorm(x, w["ln_f.weight"], w["ln_f.bias"]), [x.shape[-2] - 1]) @ w["wte.T"]

def mask(places, size, window=None):
  """The attention mask for tokens at places over size positions. It's -1e9 where a token would look past its own
  position, or window or more positions back when there's a window, and 0 elsewhere. On the chip it's built from an
  int32 row of places, so a decode step's mask is data."""
  return masked(np.asarray(places)[:, None] if not isinstance(places, Tensor) else places.reshape(1, places.cols, 1), size, window)

def masked(column, size, window=None):  # mask, from places in columns laid out like the rows of attention's scores
  if not isinstance(column, Tensor):
    k = np.arange(size)
    return np.where((k > column) | ((k <= column - window) if window else False), np.float32(-1e9), np.float32(0))
  cols = columns(column.rt, size)
  out = cols > column
  if window: out = out + (cols < column - (window - 1))
  return (out * -1e9).cast("bf16")

@functools.cache
def columns(rt, size): return Tensor.frombytes(rt, 1, size, np.arange(size, dtype=np.int32).tobytes(), 1, "int32")  # key positions, made once

def kv_cache(layers, size, zeros): return [zeros((2 * HEADS, size, 64)) for _ in range(layers)]  # a cache per block, made by zeros(shape)

def load(path):
  """Reads a safetensors file into {name: (bf16 bits, the same values as float32)}, with vectors as one-row matrices.
  The file is 8 bytes of header size, a JSON header with each tensor's dtype, shape and byte range, then the data."""
  header, data = safetensors(path)
  bits = {name: tensor_bits(data, t) for name, t in header.items()  # skip attention's causal masks
          if name != "__metadata__" and not name.endswith((".attn.bias", ".attn.masked_bias"))}
  return {name: (b, bf16_floats(b)) for name, b in bits.items()}

# Uploads the weights as baretile Tensors and transposes the output layer on the chip. The matrices that matmuls read as
# B, the blocks' and the output layer, are cast to dtype on the chip and laid out by columns.
def upload(rt, weights, dtype="bf16"):
  t = {name: Tensor.frombytes(rt, *bits.shape, bits.tobytes()) for name, (bits, _) in weights.items()}
  t["wte.T"] = t["wte.weight"].T.realize()  # transpose once, so no token reads it transposed
  return t | {k: x.cast(dtype).realize().columns() for k, x in t.items() if k == "wte.T" or k.startswith("h.") and x.rows > 1}

def arrays(weights, t=None):  # numpy arrays. With upload's t, weights not in bf16 are read back from the chip
  w = {name: values for name, (_, values) in weights.items()} | {"wte.T": weights["wte.weight"][1].T}
  return w | {k: values(x) for k, x in (t or {}).items() if x.dtype != "bf16"}

def values(x):  # a baretile matrix in numpy. bf16 and block floats, which read back as bf16, become float32
  raw, kind = x.tobytes(), {"fp32": np.float32, "int32": np.int32}.get(x.dtype)
  return (np.frombuffer(raw, kind) if kind else bf16_floats(np.frombuffer(raw, np.uint16))).reshape(x.rows, x.cols)

def logits(ids, w):  # the logits for the token after ids, in numpy, as a vector
  return forward(ids, w, mask(range(len(ids)), len(ids)))[0]

def generate(enc, ids, steps, chip, numpy):
  """Generates on baretile and numpy side by side, the prompt in the first step and then a token per step.

  chip(ids, places, i) returns baretile's logits at step i, and numpy(ids, places) returns numpy's. The two continue
  with numpy's pick, so a near tie can't send them down different paths. Prints the text, baretile's pick where it
  differs, and how often the two agree."""
  new, agree = ids, 0
  print(enc.decode(ids), end="", flush=True)
  for i in range(steps):
    start, places = time.time(), list(range(len(ids) - len(new), len(ids)))
    got, want = chip(new, places, i), numpy(new, places)
    agree += int(got.argmax()) == int(want.argmax())
    new = [int(want.argmax())]
    ids += new
    print(enc.decode(new) + ("" if got.argmax() == want.argmax() else f"[baretile: {enc.decode([int(got.argmax())])!r}]"), end="", flush=True)
    print(f" ({time.time() - start:.0f}s, logits within {np.abs(got - want).max():.2f})", end="", file=sys.stderr, flush=True)
  print(f"\nbaretile picked the same token as numpy {agree} times out of {steps}")

def main():
  import tiktoken
  from baretile import pci
  from baretile.chip import Chip
  from baretile.runtime import Runtime
  path, prompt, steps = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "The meaning of life is", int(sys.argv[3]) if len(sys.argv) > 3 else 8
  dtype = sys.argv[4] if len(sys.argv) > 4 else "bf16"
  enc, chip = tiktoken.get_encoding("gpt2"), Chip(pci.open_device())
  rt = Runtime(chip)
  start = time.time()
  weights = load(path)
  t = upload(rt, weights, dtype)
  w = arrays(weights, t)
  print(f"{sum(bits.size for bits, _ in weights.values()) / 1e6:.0f}M weights on the chip in {time.time() - start:.0f}s")
  ids, layers = enc.encode(prompt), 1 + max(int(k.split(".")[1]) for k in w if k.startswith("h."))
  size = -(-(len(ids) + steps) // 32) * 32  # rows for the positions the generation reaches, rounded up to a multiple of 32
  kv, np_kv = kv_cache(layers, size, lambda shape: Tensor.of(rt, np.zeros(shape).tolist())), kv_cache(layers, size, np.zeros)
  on = lambda row: Tensor.of(rt, [row], "int32")  # a row of ids or positions, on the chip
  run = lambda ids, places, w, kv: forward(ids, w, mask(places, size), places, kv)
  step, since = jit(lambda ids, places: run(ids, places, t, kv)), [0]  # a token per step, replayed from the third step on
  def chip(ids, places, i):
    since[0] = rt.launches
    return values(run(on(ids), on(places), t, kv) if i == 0 else step(on(ids), on(places)))[0]
  generate(enc, ids, steps, chip, lambda ids, places: run(ids, places, w, np_kv)[0])
  print("the last token's kernels, by cycles on the chip's clock (Runtime.profile):")
  for name, n, cycles, _ in rt.profile(since[0])[:8]: print(f"{cycles:9d} {n:4d}  {name[:90]}")

if __name__ == "__main__": main()
