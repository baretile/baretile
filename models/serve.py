# OpenAI-compatible HTTP server for the models in llama.py. Requests decode together in separate
# cache slots. Fixed shapes let steps replay, including when slots are idle. Reuse prompt prefixes
# from free slots or saved sessions; keep sessions in device DRAM, then host memory as space fills.
#   TTSIM=/path/to/libttsim.so python3 models/serve.py model [port] [slots] [places] [bf16|bfp8|bfp4|tt|experts] [bf16|bfp8|bfp4|""] [connections]
# model is a local folder or Hugging Face ID. Then come the precision mode, the KV cache format ("" for the mode's) and
# the most connections served at once, of which one client address may hold a quarter (serve.server has the default).
#   curl localhost:8000/v1/completions -d '{"prompt": "The meaning of life is", "max_tokens": 8, "temperature": 0}'
import codecs, collections, functools, json, math, os, queue, re, select, shutil, socket, sys, tempfile, threading, time, uuid
import urllib.error, urllib.parse, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baretile import bh
from baretile.kernels import topk_room
from baretile.tensix import tile_bytes
from baretile.tensor import Tensor, jit
from models import gpt2, llama, qwen

class Head:
  """The largest logits in a row, with token IDs in descending logit order. lse is log(sum(exp((x-max(x))*t)))
  over the whole row, where t is inverse temperature. Subtract the maximum before scaling to avoid overflow.
  row() reads the full row when sampling needs it. Head.of builds the same representation from NumPy logits."""
  def __init__(self, values, ids, lse, row): self.values, self.ids, self.lse, self.row = np.asarray(values, np.float64), ids, float(lse), row
  @staticmethod
  def of(logits, t=1.0, k=64):  # finds the k-th largest with a partition, then sorts the logits at least that large. Ties keep the lower id first
    logits = finite(np.asarray(logits, np.float64))
    bar = np.partition(logits, len(logits) - k)[len(logits) - k] if k < len(logits) else logits.min()
    ids = np.flatnonzero(logits >= bar)
    ids = ids[np.argsort(-logits[ids], kind="stable")][:k]
    m = float(logits[ids[0]])
    return Head(logits[ids], ids, math.log(np.exp((logits - m) * t).sum()), lambda: logits)

def finite(logits):  # logits that sampling can use: a nonempty row whose maximum is finite, so no logit is NaN or +inf
  if logits.ndim != 1 or not logits.size or not np.isfinite(logits.max()):
    raise ValueError("sampling needs a nonempty logit row with a finite maximum and no NaN")
  return logits

def unsure(text, marks):  # how many characters at the end of text may begin one of marks
  return max((k for m in marks for k in range(1, len(m)) if text.endswith(m[:k])), default=0)

def scale(r): return 1 / r.temperature if r and r.temperature > 0 else 1.0  # the 1/temperature of a request. Any value works for greedy

def heads(logits, t=None, k=64):
  """The head of each row of logits, on the chip: (values, ids[, lse]), realized, which is what Head holds.

  logits is an fp32 matrix, rows by the vocabulary. topk runs on parts of each row that fit the L1 of a worker, then
  on the best of the parts, by their labels. Given t, a column with the 1/temperature of each row, it also makes lse
  in fp32, with a kernel for the parts and another to combine them.
  A part is at most 96 tiles where the vocabulary splits that way, so more workers share the work. Otherwise it's as
  many as topk fits in L1: Llama 3's 128,256 goes as 24 parts of 5,344. There are at most 125 parts, since the second
  topk holds 64 of the best of each part in the L1 of one worker. Returns None for a vocabulary that doesn't split
  into such parts, and then the host reads the logits back and finds the heads in numpy."""
  tiles = logits.cols // 32
  cut = lambda most: next((p for p in range(1, 126) if tiles % p == 0 and tiles // p <= most), None)
  parts = cut(96) or cut(topk_room("fp32", -(-k // 32) * 32)) if logits.cols % 32 == 0 else None
  if parts is None: return None
  best, ids = logits.topk(-(-k // 32) * 32 if parts > 1 else k, parts)
  values, ids = best.topk(k, labels=ids) if parts > 1 else (best, ids)
  if t is None: return values.realize(), ids.realize()
  x, m = logits.split(parts), values.max(2).realize()  # one maximum for the row, including every part
  mass = ((x - m) * t).exp().sum(2)
  lse = (mass.sum(0) if parts > 1 else mass).log()
  return values.realize(), ids.realize(), lse.realize()

class Request:
  """Track a prompt, sampling settings and generated text. Stop at eos, a stop string or max_tokens. out
  receives text once it can't begin a stop string, then None at completion. finish records the reason; gone
  cancels a request after the client leaves.

  cached counts reused prompt tokens. probs stores log probabilities from raw logits, and marks maps text
  pieces to positions in probs. shape checks whether text can satisfy an output constraint. room holds the
  semaphore acquired by Engine.offer until the request ends."""
  def __init__(self, tokenizer, ids, max_tokens=16, temperature=1.0, top_p=1.0, stop=(), eos=(), seed=None, top_k=0, min_p=0.0,
               presence=0.0, frequency=0.0, repetition=1.0, logprobs=None, shape=None):
    for name, v, lo, hi in (("max_tokens", max_tokens, 1, math.inf), ("top_k", top_k, 0, math.inf), ("logprobs", logprobs, 0, 20)):
      if v is None and name == "logprobs": continue
      if type(v) is not int or not lo <= v <= hi: raise ValueError(f"{name} must be an integer between {lo} and {hi}")
    for name, v in (("temperature", temperature), ("top_p", top_p), ("min_p", min_p), ("presence_penalty", presence),
                    ("frequency_penalty", frequency), ("repetition_penalty", repetition)):
      try: valid = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
      except OverflowError: valid = False
      if not valid: raise ValueError(f"{name} must be a finite number")
    if temperature < 0: raise ValueError("temperature must be nonnegative")
    tiny, most = float(np.finfo(np.float32).tiny), float(np.finfo(np.float32).max)
    if temperature and not tiny <= 1 / temperature <= most:  # the chip takes 1/temperature as fp32
      raise ValueError("temperature must be zero or have an inverse that is a normal fp32 value")
    # The logits are fp32. Within these bounds the penalized logits stay finite in the host's float64
    if abs(presence) > most or abs(frequency) > most: raise ValueError("presence_penalty and frequency_penalty must fit fp32")
    if not tiny <= repetition <= most: raise ValueError("repetition_penalty must be positive and a normal fp32 value")
    if not 0 < top_p <= 1 or not 0 <= min_p <= 1: raise ValueError("top_p must be in (0, 1] and min_p in [0, 1]")
    if stop is not None and not isinstance(stop, (str, list, tuple)): raise ValueError("stop must be a string or a list of strings")
    stop = [stop] if isinstance(stop, str) else list(stop or ())
    if any(not isinstance(s, str) or not s for s in stop): raise ValueError("stop must contain nonempty strings")
    self.tok, self.ids, self.max_tokens, self.temperature, self.top_p = tokenizer, list(ids), max_tokens, temperature, top_p
    self.stop, self.eos, self.rng = stop, set(eos), np.random.default_rng(seed)
    self.top_k, self.min_p, self.presence, self.frequency, self.repetition, self.logprobs = top_k, min_p, presence, frequency, repetition, logprobs
    self.shape = shape
    self.made, self.said, self.out, self.finish, self.gone, self.cached, self.probs, self.marks = [], "", queue.Queue(), None, False, 0, [], []
    self.text, self.utf8 = "", codecs.getincrementaldecoder("utf-8")("replace")  # the text so far, except a character that's still coming
    self.room, self.error = None, None

  def whole(self):
    """Whether sampling needs the full logit row. Penalties or shape constraints can select tokens outside the
    head. Log probabilities also need the row at temperature 1."""
    return bool(self.presence or self.frequency or self.repetition != 1 or self.shape or self.logprobs is not None)

  def pick(self, head):
    """Pick a token from Head when possible, otherwise read the full row. Apply presence and frequency
    penalties to generated tokens, and repetition penalties to prompt and generated tokens, then call draw.

    shape(text, pending=b"") returns 0 for an invalid prefix, 1 for an incomplete valid prefix, or 2 for complete text.
    pending holds the bytes of an incomplete UTF-8 character.
    Reject a token that makes the prefix invalid and draw again without it. Accept eos only for complete
    text. This avoids testing every vocabulary token when the first few draws already fit."""
    if not self.whole() and (token := self.sample(head)) is not None: return token
    logits = np.asarray(head.row(), np.float64)
    if self.presence or self.frequency or self.repetition != 1:
      made = np.bincount(self.made, minlength=len(logits))[:len(logits)]
      logits = logits - self.presence * (made > 0) - self.frequency * made
      seen = (made > 0) | (np.bincount(self.ids, minlength=len(logits))[:len(logits)] > 0)
      logits = np.where(seen, np.where(logits > 0, logits / self.repetition, logits * self.repetition), logits)
    if self.shape is None: return self.draw(logits)
    logits = logits.copy()  # rejected tokens become -inf below, and the row may be the Head's own
    for _ in range(32):
      if not np.isfinite(logits).any(): break
      if self.fits(token := self.draw(logits)): return token
      logits[token] = -np.inf
    # If no token fits, only a complete value may end successfully without an EOS token.
    token = next((int(t) for t in np.argsort(-logits) if logits[t] > -np.inf and self.fits(int(t))), None)
    if token is None and not self.fits(None): raise ValueError("no token can complete the requested output constraint")
    return token

  def fits(self, token):
    # A token may stop partway through UTF-8. Keep incomplete bytes pending, and reject malformed sequences.
    end = token is None or token in self.eos
    if not end and token not in self.tok.bytes: return False  # a padded vocabulary has ids the tokenizer can't spell
    decoder = codecs.getincrementaldecoder("utf-8")()
    try: text = decoder.decode(b"".join(self.tok.bytes[t] for t in self.made + ([] if end else [token])), final=end)
    except UnicodeDecodeError: return False
    state = self.shape(text, decoder.getstate()[0])
    return state == 2 if end else state > 0

  def sample(self, head):
    """The token that draw would pick, found from the head without reading the row back. Returns None where the head
    doesn't decide it: a top_k larger than the head, or a top_p or min_p cut that reaches past it. Both are known before
    drawing. With no top_k, top_p or min_p, a draw that lands past the mass of the head reads the row back and draws
    from the tokens past the head, so all tokens keep their probabilities."""
    finite(head.values)
    if self.temperature == 0: return int(head.ids[0])
    if not math.isfinite(head.lse) or head.lse < 0: return self.draw(head.row())
    t, k = 1 / self.temperature, len(head.ids)
    p, total = np.exp((head.values - head.values[0]) * t), math.exp(head.lse)  # relative to the largest
    keep = None  # None means no cut
    if self.top_k or self.top_p < 1:
      if self.top_k > k: return None
      run = np.cumsum(p[:self.top_k or k])
      if self.top_p < 1 and not self.top_k and run[-1] < self.top_p * total: return None  # a flatter nucleus reaches past the head
      keep = np.arange(np.searchsorted(run, self.top_p * (run[-1] if self.top_k else total)) + 1 if self.top_p < 1 else len(run))
    if self.min_p:
      if keep is None and p[-1] >= self.min_p: return None  # tokens past the head may still pass the min_p cut
      keep = np.flatnonzero(p >= self.min_p) if keep is None else keep[p[keep] >= self.min_p]
    if keep is None:  # no cut: the draw lands in the mass of the head, or in the rest
      run, u = np.cumsum(p), self.rng.random() * total
      if u < run[-1]: return int(head.ids[min(int(np.searchsorted(run, u, side="right")), k - 1)])
      rest = np.exp((np.float64(head.row()) - head.values[0]) * t)
      rest[head.ids] = 0
      cum = np.cumsum(rest)
      return min(int(np.searchsorted(cum, (u - run[-1]) / max(total - run[-1], 1e-300) * cum[-1], side="right")), len(rest) - 1)
    w = np.cumsum(p[keep])
    return int(head.ids[keep[min(int(np.searchsorted(w, self.rng.random() * w[-1], side="right")), len(keep) - 1)]])

  def draw(self, logits):
    """Choose the highest logit at temperature 0; otherwise sample at the requested temperature.
    Apply top_k first (0 keeps all tokens), then keep the smallest prefix whose probability reaches
    top_p, then remove tokens below min_p times the highest probability. Try the 1,024 largest logits
    first to avoid sorting the whole vocabulary when they contain enough probability."""
    logits = finite(np.asarray(logits, np.float64))
    if self.temperature == 0: return int(logits.argmax())
    p, keep = np.exp((logits - logits.max()) / self.temperature), None  # None means no cut
    if self.top_k or self.top_p < 1:
      keep = np.argpartition(-p, (n := min(self.top_k or 1024, len(p))) - 1)[:n]
      keep = keep[np.argsort(-p[keep])]
      if not self.top_k and p[keep].sum() < self.top_p * p.sum(): keep = np.argsort(-p)  # a flatter distribution needs more than 1,024
      if self.top_p < 1: keep = keep[:np.searchsorted(np.cumsum(p[keep]), self.top_p * (p[keep] if self.top_k else p).sum()) + 1]
    if self.min_p: keep = np.flatnonzero(p >= self.min_p * p.max()) if keep is None else keep[p[keep] >= self.min_p * p.max()]
    w = p if keep is None else p[keep]
    i = min(int(np.searchsorted(np.cumsum(w), self.rng.random() * w.sum(), side="right")), len(w) - 1)
    return i if keep is None else int(keep[i])

  def take(self, head, room):
    """Generate the next token from a Head or a NumPy row of logits. room indicates whether the cache
    slot has space to process this token. Return True when the request has ended."""
    if self.gone: self.finish = "gone"; return True
    head = head if isinstance(head, Head) else Head.of(head, scale(self))
    token = self.pick(head)
    end = token is None or token in self.eos
    if not end:
      self.made.append(token)
      if self.logprobs is not None:  # from the raw logits, before any penalty or temperature. vLLM does the same
        logits = np.asarray(head.row(), np.float64)
        lp = logits - logits.max()
        lp = lp - np.log(np.exp(lp).sum())
        n = min(self.logprobs, len(lp))
        top = np.argpartition(-lp, n - 1)[:n] if n else []
        self.probs.append((token, float(lp[token]), [(int(t), float(lp[t])) for t in sorted(top, key=lambda t: -lp[t])]))
    before = len(self.text)  # where to look for a stop string that reaches into what this token adds
    self.text += self.utf8.decode(b"" if end else self.tok.bytes.get(token, b""))
    start = max(0, before - max(map(len, self.stop), default=1) + 1)
    cut = min((i for i in (self.text.find(s, start) for s in self.stop) if i >= 0), default=-1)
    if end or cut >= 0 or len(self.made) == self.max_tokens or not room:
      text = self.text + self.utf8.decode(b"", True)
      text = text if cut < 0 else text[:cut]
      if self.shape and (end or cut >= 0) and self.shape(text) != 2:
        raise ValueError("generation stopped before the requested output constraint was complete")
      self.say(text)
      self.finish = "stop" if end or cut >= 0 else "length"
      return True
    self.say(self.text[:len(self.text) - unsure(self.text, self.stop)])  # hold back what may begin a stop string
    return False

  def say(self, text):
    if len(text) > len(self.said): self.marks.append(len(self.probs)); self.out.put(text[len(self.said):]); self.said = text

  def end(self):
    if self.room: self.room.release(); self.room = None  # frees its slot for the next offer
    self.out.put(None)

class Engine:
  """Decode sequences together, with places cache rows and separate DeltaNet state per slot. Steps include idle
  slots to keep shapes fixed for replay. Idle slots use a reserved final row and leave recurrent state
  unchanged.

  Tensor.attend reads up to each sequence's position. Other attention paths read to the next power of two,
  at least 32, and keep a replay per length. Prefill one prompt chunk between decode steps so a long prompt
  doesn't block other requests completely.

  Save a session after prefill and at completion. Sessions contain tokens, KV rows and recurrent state and
  can be restored into any slot. Keep them in device DRAM up to park bytes (free memory minus 2 GiB by
  default). Move the least recently used sessions to the host up to keep bytes, then discard them. KV rows
  support shorter prefixes, but DeltaNet state must match the exact tokens that produced it.

  With a prediction layer, verify each slot's token and draft in one call. Accept the draft when it matches
  the model's sampled choice, then make new drafts in another call. Rejected drafts restore the state after
  the first token on the next step."""
  def __init__(self, rt, t, c, tokenizer, slots=4, places=512, chunk=None, keep=4 << 30, park=None):
    if slots < 1: raise ValueError("slots must be positive")
    if places < 32 or places % 32: raise ValueError("places must be a positive multiple of 32, so that slots are whole tiles")
    if places > min(t[k].shape[-2] for k in ("cos", "sin")): raise ValueError("places exceeds the loaded RoPE tables")
    if chunk is not None and chunk < 1: raise ValueError("chunk must be positive")
    if keep < 0 or park is not None and park < 0: raise ValueError("session cache budgets must be nonnegative")
    self.rt, self.t, self.c, self.tok, self.places, self.keep = rt, t, c, tokenizer, places, keep
    self.chunk = chunk or (2048 if c.experts else 256)  # with a mixture of experts, a chunk reads nearly all the experts of the blocks (cost.prompt)
    self.spec = c.mtp > 0 and "mtp.fc" in t  # drafts with the multi-token prediction layer
    self.kv = llama.kv_cache(c, slots * places, llama.zeros(rt, llama.cache_format(c)), slots, self.spec)
    self.slot = [llama.slot(c, self.kv, u, slots) for u in range(slots)]  # the caches as slot u sees them
    self.park = max(0, sum(room for _, room in rt.holes) * len(bh.DRAM) - (2 << 30)) if park is None else park
    self.linear = any(llama.linear(c, l) for l in range(c.layers))  # DeltaNet state can resume only from the exact prefix that produced it
    self.attends = llama.attends(c, t)  # attention in one kernel reads the rows up to each position, so a replay serves any length
    self.held, self.busy = [[] for _ in range(slots)], [None] * slots
    self.coherent = [True] * slots  # prediction history matches held; an early EOS can leave its final update absent
    self.used = [0] * slots  # when each slot last took a request. Of slots that hold as much of a prompt, admit picks the least recently used
    # for each call and length it reads: its replay, what its last call made (the logits and their heads), and the
    # hidden states from check
    self.replays, self.made, self.hidden = {}, {}, {}
    self.drafts, self.taken = [0] * slots, [1.0] * slots  # each slot's draft, and whether the last was taken, so states go on from both tokens
    self.drafted = self.agreed = 0  # drafts checked against the pick of the model, and those taken. This shows what speculation saves
    self.waiting, self.prefilled = queue.Queue(), 0  # the tokens prefilled so far. This shows what the prefix cache saves
    self.room = threading.BoundedSemaphore(slots)  # a slot for each offered request, from its offer to its end
    self.lock, self.failed = threading.Lock(), False  # serialize offers against failure cleanup
    self.sessions = collections.OrderedDict()  # tokens -> each block's KV cache rows or Gated DeltaNet states, as tensors on the chip or bytes

  def on(self, row, dtype="int32"): return Tensor.of(self.rt, [row], dtype)

  def offer(self, r):  # queues r if a slot is free and holds the slot until r ends. Otherwise it returns False without waiting
    with self.lock:
      if self.failed: raise RuntimeError("the engine stopped on an error")
      if not 0 < len(r.ids) < self.places - 1: raise ValueError(f"a prompt needs 1 to {self.places - 2} tokens, not {len(r.ids)}")
      if not self.room.acquire(blocking=False): return False
      r.room = self.room
      self.waiting.put(r)
      return True

  def length(self, last):  # the rows of each slot that a call reads when the last position of its sequences is last
    return self.places if self.attends else min(self.places, 1 << max(5, last.bit_length()))

  def replay(self, f, n):  # the replay (jit) of f for calls that read n rows of each slot. made keeps what each call makes
    if (f, n) not in self.replays:
      def run(*xs):
        self.made[f, n] = f(*xs, length=n)
        return self.made[f, n][0]
      self.replays[f, n] = jit(run)
    return self.replays[f, n]

  def rows(self, f, n):
    """The rows from the last call of f at cache length n: Heads where the chip found them, else NumPy rows of
    logits. Read the full logits at most once if sampling needs them."""
    logits, made = self.made[f, n]
    whole = functools.cache(lambda: gpt2.values(logits))
    if made is None: return list(whole())
    values, ids, lse = (gpt2.values(x) for x in made)
    return [Head(values[i], ids[i], lse[i, 0], lambda i=i: whole()[i]) for i in range(len(values))]

  def column(self, ts): return Tensor.of(self.rt, [[x] for x in ts], "fp32")  # the 1/temperature of each row, as a column

  def reach(self, held, ids):  # length of the reusable prefix of ids in held; always process the prompt's last token again
    if self.linear: return len(held) if len(held) < len(ids) and list(ids[:len(held)]) == list(held) else 0
    return min(next((i for i, (a, b) in enumerate(zip(held, ids)) if a != b), min(len(held), len(ids))), len(ids) - 1)

  def admit(self, r):  # choose a free slot or saved session with the longest reusable prompt prefix
    if r.gone: return r.end()
    free = [u for u, b in enumerate(self.busy) if b is None]
    for u in free:
      if not self.coherent[u]: self.restore(u)
    u = max(free, key=lambda u: (self.reach(self.held[u], r.ids), -self.used[u]))
    best = max(self.sessions, key=lambda k: self.reach(k, r.ids), default=())
    if self.reach(best, r.ids) > self.reach(self.held[u], r.ids):  # a session that doesn't fit back in DRAM is dropped, and u prefills
      if self.restore(v := min(free, key=lambda u: self.used[u]), best): u = v
    self.used[u], self.busy[u] = max(self.used) + 1, r
    n = self.reach(self.held[u], r.ids)
    if self.linear and n < len(self.held[u]): self.restore(u)  # zero its states and start over
    elif not self.taken[u]:  # after a rejected draft, resume from the state before that draft
      for l, s in enumerate(self.slot[u]):
        if llama.linear(self.c, l): s[0].assign(s[2]), s[1].assign(s[3])
      self.taken[u] = 1.0
    self.held[u], r.cached = r.ids[:n], n

  def save(self, u):
    """Keep a coherent prefix in the session cache. Evict before allocating device copies; write directly to
    the host when the device budget is too small. If a snapshot cannot fit, omit it without failing the request.
    After a rejected draft, save the recurrent state from before that draft."""
    key, rows = tuple(self.held[u]), -(-len(self.held[u]) // 32) * 32
    if not key or not self.coherent[u] or key in self.sessions or not self.park + self.keep: return
    parts = [tuple(s[:2] if self.taken[u] else s[2:]) if llama.linear(self.c, l) else s[:, :rows, :]
             for l, s in enumerate(self.slot[u])]
    each = lambda v: [x for part in v for x in (part if isinstance(part, tuple) else (part,))]
    mapped = lambda f, v: [tuple(f(x) for x in part) if isinstance(part, tuple) else f(part) for part in v]
    dtype = lambda x: x.t.dtype if isinstance(x, qwen.Row) else x.dtype
    def device(x):  # the bytes of a copy in DRAM: whole tiles, spread over every bank
      tiles = x.batch * -(-x.rows // 32) * -(-x.cols // 32)
      return -(-tiles // len(bh.DRAM)) * len(bh.DRAM) * tile_bytes(dtype(x))
    host = lambda x: x.batch * x.rows * x.cols * (4 if dtype(x) in ("fp32", "int32") else 2)  # tobytes gives KV rows as bf16
    def keys(kind): return [k for k, v in self.sessions.items() if isinstance(each(v)[0], kind)]
    def tohost(x):
      if not isinstance(x, qwen.Row): return x.tobytes()
      # Download each stored matrix and select the row on the host. Row.get() would need another device allocation.
      width = 4 * x.cols  # convolution history is fp32
      return b"".join(x.t[b].tobytes()[x.u * width:(x.u + 1) * width] for b in range(x.batch))
    def spill(v):  # v as bytes on the host, after dropping the oldest host sessions to make room. None if it can't fit
      need = sum(map(host, each(v)))
      if need > self.keep: return None
      held = keys(bytes)
      while sum(sum(map(len, each(self.sessions[k]))) for k in held) + need > self.keep:
        del self.sessions[held.pop(0)]
      return mapped(tohost, v)
    need, held = sum(map(device, each(parts))), keys(Tensor)
    try:
      if need <= self.park:
        while sum(sum(map(device, each(self.sessions[k]))) for k in held) + need > self.park:
          k = held.pop(0)
          if kept := spill(self.sessions[k]): self.sessions[k] = kept  # which frees the device copy
          else: del self.sessions[k]
        def copy(x):
          if isinstance(x, qwen.Row): return x.get().realize()
          return Tensor(self.rt, x.rows, x.cols, batch=x.batch, dtype=x.dtype).assign(x)
        try: saved = mapped(copy, parts)
        except MemoryError: saved = spill(parts)  # fragmentation or other workloads can leave less than the cache budget
      else: saved = spill(parts)
      if saved is not None: self.sessions[key] = saved
    except MemoryError: pass  # caching is optional; no partially copied session is published

  def restore(self, u, key=()):
    """Put a kept session into slot u, which then holds its tokens, and return True. With no key, zero the states.
    Upload every part kept on the host before changing the slot. If DRAM can't hold them, free the uploads, drop
    the session and return False with the slot unchanged: a snapshot is only a cache."""
    if key: self.sessions.move_to_end(key)
    rows, kept = -(-len(key) // 32) * 32, self.sessions[key] if key else [None] * len(self.slot[u])
    up = lambda x, n, data, dtype: data if isinstance(data, Tensor) else Tensor.frombytes(self.rt, n, x.cols, data, x.batch, dtype)
    try:
      at = self.on(list(range(rows))) if any(isinstance(p, bytes) for p in kept) else None
      staged = [None if p is None else tuple(up(x, x.rows, d, "fp32") for x, d in zip(s[:2], p)) if llama.linear(self.c, l) else
                up(s, rows, p, "bf16") for l, (s, p) in enumerate(zip(self.slot[u], kept))]
    except MemoryError: staged = None  # this drops the uploads made so far
    if staged is None:
      del self.sessions[key]
      return False
    for l, (s, p, d) in enumerate(zip(self.slot[u], kept, staged)):
      if llama.linear(self.c, l):
        for x, v in zip(s[:2], d or (None, None)): x.assign(Tensor.zeros(self.rt, x.rows, x.cols, x.batch, "fp32") if v is None else v)
      elif isinstance(p, Tensor): s[:, :rows, :].assign(p)
      elif p is not None: s.put(at, d)  # rows from the host are bf16, which put converts to a block-float cache
    self.held[u], self.taken[u], self.coherent[u] = list(key), 1.0, True
    return True

  def fill(self, u):
    """Prefill the next prompt chunk in slot u. The final chunk's last logits select the first new token.
    With speculation, also run the prediction layer over the chunk; its last position gives the first draft."""
    r, at = self.busy[u], len(self.held[u])
    if r.gone: return self.take(u, None)
    ids = r.ids[at:at + self.chunk]
    places, n = self.on(list(range(at, at + len(ids)))), self.length(at + len(ids) - 1)
    out = llama.forward(self.on(ids), places, self.t, self.c, self.slot[u], hidden=self.spec, length=n, capacity=self.places)
    out, h = out if self.spec else (out, None)
    self.held[u] += ids
    self.prefilled += len(ids)
    final = len(self.held[u]) == len(r.ids)
    self.coherent[u] = not self.spec  # until the prediction layer has seen this chunk too
    done = final and self.take(u, gpt2.values(out)[0], 1 + self.spec, end=False)
    after = (r.ids + r.made)[at + 1:at + len(ids) + 1]
    if self.spec and r.finish != "error" and len(after) == len(ids):
      drafts = gpt2.values(qwen.draft(self.on(after), places, h, self.t, self.c, self.slot[u], length=n, every=False))
      self.coherent[u] = True
      if r.made: self.drafts[u], self.taken[u] = self.guess(u, int(drafts[-1].argmax())), 1.0
    if final:
      if done: self.release(u)
      else: self.save(u)

  def advance(self):  # decode one token for every occupied slot
    live = [r if r and r.made else None for r in self.busy]
    if not any(live): return
    if self.spec: return self.speculate(live)
    ids = [r.made[-1] if r else 0 for r in live]
    places = [len(h) if r else self.places - 1 for h, r in zip(self.held, live)]
    n, ts = self.length(max(p for p, r in zip(places, live) if r)), [scale(r) for r in live]
    self.replay(self.decode, n)(self.on(ids), self.on(places), self.column(ts))
    for u, (r, head) in enumerate(zip(live, self.rows(self.decode, n))):
      if r: self.held[u].append(r.made[-1]); self.take(u, head)

  # The request in slot u takes its next token. A finished request leaves its slot, and its session is kept before
  # the request ends, so whoever waits on it finds the chip idle. Request.take works on the host, except where a Head
  # reads its row back from the chip. Any other failure there, from sampling to the client's output constraint, ends
  # only this request. A failed read stops the engine. The type of an exception can't tell them apart: a schema's
  # RecursionError is a RuntimeError too, so the read records its own failure.
  def take(self, u, head, room=1, end=True):
    r, unread = self.busy[u], False
    if isinstance(head, Head):
      row = head.row
      def read():
        nonlocal unread
        try: return row()
        except BaseException: unread = True; raise
      head.row = read
    try: done = r.take(head, len(self.held[u]) + room < self.places)
    except Exception as e:
      if unread: raise
      r.finish, r.error, done = "error", f"{type(e).__name__}: {e}", True
    if done and end: self.release(u)
    return done

  def release(self, u):  # ends the request in slot u, keeping its session unless it failed
    r = self.busy[u]
    if r.finish != "error": self.save(u)
    self.busy[u] = None
    r.end()

  def decode(self, ids, places, t, length):  # replayed: the next tokens of the slots, giving their logits and their heads at t
    logits = llama.forward(ids, places, self.t, self.c, self.kv, len(self.busy), length=length, capacity=self.places).realize()
    return logits, heads(logits, t)

  def check(self, ids, places, accept, t, length):
    """Replayed when speculating. Takes in the token and draft of each slot, from the states that accept picks.
    Returns their logits and heads, and keeps their hidden states for guesses."""
    logits, self.hidden[length] = llama.forward(ids, places, self.t, self.c, self.kv, len(self.busy), accept=accept, hidden=True,
                                               length=length, capacity=self.places)
    return logits.realize(), heads(logits, t)

  def guesses(self, ids, places, taken, length):
    """Run the prediction layer over two positions per slot for replay. After rejection, repeat the first
    position instead of using the second. Return logits and the best token per row."""
    n, d = len(self.busy), self.c.dim
    h = self.hidden[length].reshape(n, 2, d)
    first = h[:, 0:1, :]
    # the second row of each slot: its second hidden state where taken is 1, and its first again where taken is 0
    h = first.cat(first + taken.reshape(n, 1, 1) * (h[:, 1:2, :] - first), axis=1).reshape(2 * n, d).cast("bf16")
    logits = qwen.draft(ids, places, h, self.t, self.c, self.kv, n, length).realize()
    return logits, heads(logits, k=1)

  def guess(self, u, best): return best  # slot u's draft: the best token of the multi-token prediction layer. test_serve_speculative overrides it

  def speculate(self, live):
    """Verify each slot's token and draft, accept drafts that match the model's sampled choices, then
    compute drafts for the next step."""
    n, at = len(live), [len(h) for h in self.held]
    ids = [x for u, r in enumerate(live) for x in ([r.made[-1], self.drafts[u]] if r else [0, 0])]
    places = [p for u, r in enumerate(live) for p in ([at[u], at[u] + 1] if r else [self.places - 1] * 2)]
    m, ts = self.length(max(at[u] + 1 for u, r in enumerate(live) if r)), [x for r in live for x in (scale(r),) * 2]  # the draft positions too
    self.replay(self.check, m)(self.on(ids), self.on(places), self.on(self.taken, "fp32"), self.column(ts))
    rows = self.rows(self.check, m)
    pairs = [[0, 0, self.places - 1, self.places - 1] for _ in range(n)]  # the next tokens of each slot and their positions, for the drafts
    for u, r in enumerate(live):
      if not r: continue
      self.held[u].append(r.made[-1])
      self.taken[u] = 0.0
      if self.take(u, rows[2 * u], 2): continue
      self.drafted += 1
      if r.made[-1] != self.drafts[u]: pairs[u] = [r.made[-1], r.made[-1], at[u], at[u]]; continue
      self.held[u].append(r.made[-1])
      self.taken[u], self.agreed = 1.0, self.agreed + 1
      if not self.take(u, rows[2 * u + 1], 2): pairs[u] = [self.drafts[u], r.made[-1], at[u], at[u] + 1]
    if not any(self.busy[u] is r for u, r in enumerate(live) if r): return
    tokens, at = self.on([x for p in pairs for x in p[:2]]), self.on([q for p in pairs for q in p[2:]])
    self.replay(self.guesses, m)(tokens, at, self.on(self.taken, "fp32"))
    logits, made = self.made[self.guesses, m]
    best = gpt2.values(made[1])[:, 0] if made else gpt2.values(logits).argmax(1)  # the best token of each row is its draft
    for u, r in enumerate(live):
      if r and self.busy[u] is r: self.drafts[u] = self.guess(u, int(best[2 * u + 1]))

  def run(self):  # the thread of the engine: admit while a slot is free, prefill a chunk, take a step. On an error, the requests end
    admitting = None
    try:
      while True:
        if not any(self.busy): self.admit(admitting := self.waiting.get()); admitting = None
        while None in self.busy and not self.waiting.empty(): self.admit(admitting := self.waiting.get()); admitting = None
        if filling := [u for u, r in enumerate(self.busy) if r and not r.made]: self.fill(filling[0])
        self.advance()
    except BaseException:
      with self.lock:
        self.failed = True
        pending = list(self.busy) + [admitting]
        while not self.waiting.empty(): pending.append(self.waiting.get_nowait())
        for r in set(pending) - {None}: r.finish = "error"; r.end()
      raise

FAMILIES = (("qwen", "<tool_call>\n<function="), ("harmony", "<|channel|>"), ("llama", "<|start_header_id|>"), ("chatml", "<|im_start|>"))
def family(template): return next((name for name, mark in FAMILIES if mark in template), None)  # Qwen3.5's has ChatML's mark too, so qwen goes first

def words(m):  # a message's text: a string, or OpenAI's list of text parts, joined as Qwen's template joins them
  c = m.get("content")
  if isinstance(c, list) and all(isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str) for p in c):
    c = "".join(p["text"] for p in c)
  if c is not None and not isinstance(c, str): raise ValueError("message content must be a string or a list of text parts")
  return c or ""

def chat(template, messages, thinking=True, effort="medium", tools=None):
  """Format messages using the model's chat template: Qwen3.5/3.6, gpt-oss (Harmony), Llama 3 or ChatML."""
  kind = family(template)
  if kind == "qwen": return qwen_chat(messages, thinking, tools)
  if tools: raise ValueError("tools only work with the chat template of Qwen3.5")
  if kind == "harmony":  # gpt-oss has a fixed system message. A first system message in the request goes as a developer message
    system = (f"You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: {time.strftime('%Y-%m-%d')}\n\n"
              f"Reasoning: {effort}\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.")
    head, turns = f"<|start|>system<|message|>{system}<|end|>", messages
    if messages and messages[0]["role"] in ("system", "developer"):
      head, turns = head + f"<|start|>developer<|message|># Instructions\n\n{words(messages[0])}\n\n<|end|>", messages[1:]
    said = lambda m: "<|start|>user<|message|>" if m["role"] == "user" else "<|start|>assistant<|channel|>final<|message|>"
    return head + "".join(f"{said(m)}{words(m)}<|end|>" for m in turns) + "<|start|>assistant"
  if kind == "llama":  # Llama 3
    turn = lambda m: f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{words(m)}<|eot_id|>"
    return "<|begin_of_text|>" + "".join(map(turn, messages)) + "<|start_header_id|>assistant<|end_header_id|>\n\n"
  if kind == "chatml":  # the SmolLM instruct models and Qwen. When Qwen3 isn't thinking, its reasoning is given empty
    prompt = "".join(f"<|im_start|>{m['role']}\n{words(m)}<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n"
    return prompt if thinking or "<think>" not in template else prompt + "<think>\n\n</think>\n\n"
  raise ValueError("unsupported chat template; expected Qwen3.5, gpt-oss, Llama 3 or ChatML")

CALLING = ("\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
           "<function=example_function_name>\n"
           "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
           "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the "
           "specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required "
           "parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function "
           "call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do "
           "not tell the user about function calls\n</IMPORTANT>")  # how to call a tool, word for word from the Qwen3.5 template

def qwen_chat(messages, thinking=True, tools=None):
  """Format text messages as in Qwen3.5/3.6 chat_template.jinja. Describe tools in the system turn, encode
  calls as XML, and put results in user turns. Keep reasoning in assistant turns after the last question so
  a tool response can extend the cached conversation."""
  text = lambda m: words(m).strip()
  answered = lambda m: text(m).startswith("<tool_response>") and text(m).endswith("</tool_response>")  # a tool result, sent as a user turn
  question = [i for i, m in enumerate(messages) if m["role"] == "user" and not answered(m)]
  if not question: raise ValueError("messages must include a user message other than a tool response")
  system = text(messages[0]) if messages[0]["role"] == "system" else None
  out = ""
  if tools: out = "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>" + "".join(
    "\n" + json.dumps(t, ensure_ascii=False) for t in tools) + "\n</tools>" + CALLING + (f"\n\n{system}" if system else "") + "<|im_end|>\n"
  elif system is not None: out = f"<|im_start|>system\n{system}<|im_end|>\n"
  for i, m in enumerate(messages):
    if m["role"] == "user": out += f"<|im_start|>user\n{text(m)}<|im_end|>\n"
    elif m["role"] == "assistant":
      content, reasoning = text(m), m.get("reasoning_content")
      if not isinstance(reasoning, str):
        reasoning = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n") if "</think>" in content else ""
        content = content.split("</think>")[-1].lstrip("\n")
      out += "<|im_start|>assistant\n" + (f"<think>\n{reasoning.strip()}\n</think>\n\n" if i > question[-1] else "") + content
      for j, call in enumerate(m.get("tool_calls") or []):
        f = call.get("function", call)
        args = json.loads(f.get("arguments") or "{}") if isinstance(f.get("arguments"), str) else f.get("arguments") or {}
        out += ("\n\n" if content.strip() else "") if j == 0 else "\n"
        given = lambda v: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        out += f"<tool_call>\n<function={f['name']}>\n" + "".join(f"<parameter={k}>\n{given(v)}\n</parameter>\n" for k, v in args.items())
        out += "</function>\n</tool_call>"
      out += "<|im_end|>\n"
    elif m["role"] == "tool":
      if i == 0 or messages[i - 1]["role"] != "tool": out += "<|im_start|>user"
      out += f"\n<tool_response>\n{text(m)}\n</tool_response>"
      if i == len(messages) - 1 or messages[i + 1]["role"] != "tool": out += "<|im_end|>\n"
    elif m["role"] != "system" or i: raise ValueError(f"a {m['role']} message isn't allowed there")
  return out + "<|im_start|>assistant\n" + ("<think>\n" if thinking else "<think>\n\n</think>\n\n")

def function(tool): return tool.get("function", tool)  # a tool's function. Some clients send the function as the tool

def described(tools):
  """Check a request's tool definitions and return them. Each is an object whose function, or the tool itself, has a
  nonempty string name. Parameters, if not null, are an object schema whose properties, if not null, map names to
  schemas: objects or booleans. chosen's validator and calls read these shapes, the first in the engine thread, so a
  bad one fails here."""
  if tools is None: return None
  if not isinstance(tools, list): raise ValueError("tools must be a list of tool objects")
  for t in tools:
    f = function(t) if isinstance(t, dict) else None
    if not isinstance(f, dict) or not isinstance(f.get("name"), str) or not f["name"]:
      raise ValueError("each tool needs a function with a nonempty string name")
    params = f.get("parameters")
    props = params.get("properties") if isinstance(params, dict) else None
    if params is not None and not isinstance(params, dict): raise ValueError(f"tool {f['name']}: parameters must be an object schema")
    if props is not None and (not isinstance(props, dict) or any(not isinstance(p, (dict, bool)) for p in props.values())):
      raise ValueError(f"tool {f['name']}: properties must map names to schemas")
  return tools

def calls(text, tools=()):
  """Parse Qwen XML tool calls into (preceding text, OpenAI tool_calls). Encode arguments as JSON. Keep
  schema-declared strings as text; parse other values as JSON when possible. tools have the shapes that described checks."""
  head, mark, rest = text.partition("<tool_call>")
  props = lambda f: (f.get("parameters") or {}).get("properties") or {}
  schema = {function(t)["name"]: props(function(t)) for t in tools or ()}
  def value(name, key, v):
    kind = schema.get(name, {}).get(key, {})
    if not isinstance(kind, dict) or kind.get("type", "string") == "string": return v  # a boolean schema declares no type
    try: return json.loads(v)
    except (ValueError, RecursionError): return v  # RecursionError: nested too deeply for the parser
  found = [(name, {k: value(name, k, v) for k, v in re.findall(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", body, re.S)})
           for name, body in re.findall(r"<function=([^>\n]+)>(.*?)</function>", mark + rest, re.S)]
  call = lambda name, args: {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}
  return head.strip(), [{"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function", "function": call(name, args)} for name, args in found]

def chosen(choice, tools):
  """Build a prefix validator for tool_choice, or None for auto. none disallows calls; required demands calls
  at the start; a named function allows only that function. For reasoning models, validate the answer after
  the reasoning. tools have the shapes described checks."""
  if choice in (None, "auto"): return None
  if choice == "none": return lambda text, pending=b"": 0 if "<tool_call>" in text else 1 if pending else 2
  names = [function(t)["name"] for t in tools or ()]
  if choice != "required": names = [n for n in names if n == choice["function"]["name"]]
  if not names: raise ValueError(f"tool_choice: there's no such tool in tools: {choice}")
  can = lambda part, whole: whole.startswith(part) or part.startswith(whole)  # part could still become whole, or starts with it
  def fits(text, pending=b""):  # bytes keep an unfinished Unicode tool name meaningful
    raw = text.encode("utf-8") + pending
    if not can(raw, b"<tool_call>\n<function="): return 0
    if not all(any(can(part, (n + ">").encode("utf-8")) for n in names) for part in raw.split(b"<function=")[1:]): return 0
    return 2 if not pending and text.rstrip().endswith("</tool_call>") and calls(text, tools)[1] else 1
  return fits

NESTING = 32  # how deeply formatted lets schemas and generated arrays and objects nest. Its checks recurse at each level
class Short(Exception): pass  # the text ends inside a value that could still fit
class Bad(Exception): pass  # the text can't begin a value that fits

def formatted(form):
  """Build a prefix validator for response_format, or None for unrestricted text. json_object requires an
  object; json_schema requires a value matching the supported schema fields.

  Supported fields are type (one or a list), properties, required, additionalProperties, items, minItems,
  maxItems, enum, const, anyOf, oneOf and nonrecursive local $ref into $defs or definitions. String patterns, formats and
  numeric bounds aren't enforced. Sibling fields intersect, including siblings of $ref as in draft 2020-12.
  Strings may use literal Unicode or equivalent JSON escapes, including in enum and const.

  Allow at most one whitespace character between tokens and 32 characters per number to prevent unbounded
  generation. Allow commas only where another key or item can follow, so a valid prefix can still be
  completed. Arrays and objects nest at most NESTING deep, so a deeper bracket is an invalid prefix. A schema may
  nest NESTING levels of subschemas, counting references, and an enum or const value only what its level leaves."""
  kind = (form or {}).get("type", "text")
  if kind == "text": return None
  if kind not in ("json_object", "json_schema"): raise ValueError(f"response_format: unknown type {kind}")
  root = {"type": "object"} if kind == "json_object" else form["json_schema"]["schema"]
  def target(ref):
    if not isinstance(ref, str) or not re.fullmatch(r"#/(\$defs|definitions)/[^/]+", ref): raise ValueError(f"unsupported schema reference: {ref}")
    group, name = ref[2:].split("/")
    try: return root[group][name.replace("~1", "/").replace("~0", "~")]
    except (KeyError, TypeError): raise ValueError(f"missing schema reference: {ref}") from None
  def nested(v, n=0):  # how many arrays and objects nest in a JSON value, counted no further than NESTING + 1
    if n > NESTING or not isinstance(v, (list, dict)): return n
    return max((nested(x, n + 1) for x in (v.values() if isinstance(v, dict) else v)), default=n + 1)
  def check(sc, refs=(), depth=0):  # reject malformed, cyclic or deep schemas before they reach the engine thread
    if isinstance(sc, bool): return
    if depth >= NESTING: raise ValueError(f"schemas may nest at most {NESTING} levels")
    if not isinstance(sc, dict): raise ValueError("a schema must be an object or boolean")
    if "$ref" in sc:
      ref = sc["$ref"]
      if ref in refs: raise ValueError("recursive schema references are not supported")
      check(target(ref), (*refs, ref), depth + 1)
    kinds = sc.get("type", [])
    kinds = [kinds] if isinstance(kinds, str) else kinds
    if not isinstance(kinds, list) or "type" in sc and not kinds or \
        any(k not in ("object", "array", "string", "number", "integer", "boolean", "null") for k in kinds):
      raise ValueError("invalid schema type")
    if not isinstance(sc.get("properties", {}), dict): raise ValueError("schema properties must be an object")
    required = sc.get("required", [])
    if not isinstance(required, list) or any(not isinstance(k, str) for k in required): raise ValueError("required must be a list of property names")
    for k in ("minItems", "maxItems"):
      if k in sc and (type(sc[k]) is not int or sc[k] < 0): raise ValueError(f"{k} must be a nonnegative integer")
    if sc.get("minItems", 0) > sc.get("maxItems", math.inf): raise ValueError("minItems exceeds maxItems")
    for k in ("enum", "anyOf", "oneOf"):
      if k in sc and (not isinstance(sc[k], list) or not sc[k]): raise ValueError(f"{k} must be a nonempty list")
    for v in [*sc.get("enum", []), *([sc["const"]] if "const" in sc else [])]:
      if depth + nested(v) > NESTING: raise ValueError(f"enum and const values may nest at most {NESTING} levels with their schema")
      try: json.dumps(v, allow_nan=False)
      except (TypeError, ValueError): raise ValueError("enum and const must contain finite JSON values") from None
    for child in [*sc.get("properties", {}).values(), sc.get("items", True), sc.get("additionalProperties", True),
                  *sc.get("anyOf", []), *sc.get("oneOf", [])]: check(child, refs, depth + 1)
  check(root)
  def fits(text, pending=b""):
    i = 0
    def ws():
      nonlocal i
      start = i
      while i < len(text) and text[i] in " \t\n\r": i += 1
      if i - start > 1: raise Bad
    def peek():
      ws()
      if i == len(text): raise Bad if pending else Short
      return text[i]
    def word(w):  # a literal (true, false or null), as far as the text goes
      nonlocal i
      if not w.startswith(text[i:i + len(w)]): raise Bad
      if i + len(w) > len(text): raise Bad if pending else Short
      i += len(w)
    def string(allowed=None):
      # Compare UTF-16 units so a partial surrogate escape can still become an allowed non-BMP character.
      nonlocal i
      start, i, units = i, i + 1, b""
      allowed = None if allowed is None else [a.encode("utf-16-le", "surrogatepass") for a in allowed]
      while i < len(text) and text[i] != '"':
        c = text[i]
        if ord(c) < 32: raise Bad
        if c == "\\":
          if allowed is not None and not any(a.startswith(units) and len(a) > len(units) for a in allowed): raise Bad
          i += 1
          if i == len(text): raise Bad if pending else Short
          c = text[i]
          if c == "u":
            digits = text[i + 1:i + 5]
            if not re.fullmatch("[0-9a-fA-F]*", digits): raise Bad
            if len(digits) < 4:
              if pending: raise Bad
              if allowed is not None and not any(a.startswith(units) and len(a) > len(units) and
                  f"{int.from_bytes(a[len(units):len(units) + 2], 'little'):04x}".startswith(digits.lower()) for a in allowed): raise Bad
              raise Short
            c, i = chr(int(digits, 16)), i + 4
          elif c in '"\\/bfnrt': c = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}[c]
          else: raise Bad
        units += c.encode("utf-16-le", "surrogatepass")
        i += 1
        if allowed is not None and not any(a.startswith(units) for a in allowed): raise Bad
      if i == len(text):
        if pending and allowed is not None:
          remaining = [a[len(units):].decode("utf-16-le", "surrogatepass") for a in allowed if a.startswith(units)]
          if not any(r and not 0xd800 <= ord(r[0]) <= 0xdfff and r[0].encode("utf-8").startswith(pending) for r in remaining): raise Bad
        raise Short
      if allowed is not None and units not in allowed: raise Bad
      i += 1
      return json.loads(text[start:i])
    def combine(parsers, mode="all"):
      # Every parser starts at the same value. Evaluate all siblings even if an earlier one is incomplete.
      nonlocal i
      start, ends, shorts, bad = i, [], 0, False
      for parse in parsers:
        i = start
        try: parse(); ends.append(i)
        except Short: shorts += 1
        except Bad: bad = True
      i = max(ends, default=start)
      if mode == "all":
        if bad: raise Bad
        if shorts: raise Short
      elif not ends: raise Short if shorts else Bad
      elif mode == "one" and ends.count(i) != 1:
        if not pending and i == len(text) and text[start:start + 1] in "-0123456789": raise Short
        raise Bad
    def literal(v, depth):
      nonlocal i
      c, start = peek(), i
      if isinstance(v, str):
        if c != '"': raise Bad
        string([v])
      elif isinstance(v, dict):
        value({"type": "object", "properties": {k: {"const": x} for k, x in v.items()},
               "required": list(v), "additionalProperties": False}, depth)
      elif isinstance(v, list):
        if c != "[" or depth >= NESTING: raise Bad
        i += 1
        for n, x in enumerate(v):
          if n:
            if peek() != ",": raise Bad
            i += 1
          literal(x, depth + 1)
        if peek() != "]": raise Bad
        i += 1
      else:
        value({"type": "null" if v is None else "boolean" if isinstance(v, bool) else "number"}, depth)
        if json.loads(text[start:i]) != v:
          if not pending and i == len(text) and json.dumps(v).startswith(text[start:i]): raise Short
          raise Bad
    def value(sc, depth=0):  # depth counts the arrays and objects around this value
      nonlocal i
      if sc is False: raise Bad
      if sc is True: sc = {}
      key = next((k for k in ("$ref", "anyOf", "oneOf", "enum", "const") if k in sc), None)
      if key:  # the value must fit key's alternatives and the schema's other fields. With no other fields, skip parsing them
        rest = {k: v for k, v in sc.items() if k != key}
        if key == "$ref": alternatives = lambda: value(target(sc[key]), depth)
        elif key in ("enum", "const"):
          choices = sc["enum"] if key == "enum" else [sc["const"]]
          alternatives = lambda: combine([lambda v=v: literal(v, depth) for v in choices], "any")
        else: alternatives = lambda: combine([lambda alt=alt: value(alt, depth) for alt in sc[key]], "one" if key == "oneOf" else "any")
        combine([*([lambda: value(rest, depth)] if rest else []), alternatives])
        return
      c, kinds = peek(), sc.get("type") or ["object", "array", "string", "number", "boolean", "null"]
      kinds = [kinds] if isinstance(kinds, str) else kinds
      if c == '"' and "string" in kinds: string()
      elif c in "-0123456789" and {"number", "integer"} & set(kinds):
        start = i
        while i < len(text) and text[i] in "-+0123456789.eE": i += 1
        number = text[start:i]
        if len(number) > 32: raise Bad
        # If it isn't a complete number, it may be the start of one that more digits complete
        if not re.fullmatch(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][-+]?[0-9]+)?", number):
          begun = re.fullmatch(r"-?((0|[1-9][0-9]*)(\.[0-9]*|(\.[0-9]+)?[eE][-+]?[0-9]*)?)?", number)
          if not pending and i == len(text) and begun and len(number) < 32 and ("number" in kinds or not re.search("[.eE]", number)): raise Short
          raise Bad
        if "number" not in kinds and re.search("[.eE]", number): raise Bad
      elif c in "tfn":
        if ("null" if c == "n" else "boolean") not in kinds: raise Bad
        word({"t": "true", "f": "false", "n": "null"}[c])
      elif c == "{" and "object" in kinds and depth < NESTING:
        i, seen = i + 1, set()
        props, extra = sc.get("properties", {}), sc.get("additionalProperties", True)
        while True:
          if peek() == "}" and set(sc.get("required", ())) <= seen: i += 1; return
          if seen and (text[i] != "," or extra is False and set(props) <= seen): raise Bad  # a comma only where a key may follow
          if seen: i += 1
          if peek() != '"': raise Bad
          key = string(None if extra is not False else [k for k in props if k not in seen])
          if key in seen or key not in props and extra is False: raise Bad
          if peek() != ":": raise Bad
          i += 1
          value(props.get(key, extra), depth + 1)
          seen.add(key)
      elif c == "[" and "array" in kinds and depth < NESTING:
        i, n = i + 1, 0
        while True:
          if peek() == "]" and n >= sc.get("minItems", 0): i += 1; return
          if n and text[i] != ",": raise Bad
          if n: i += 1
          if n >= sc.get("maxItems", math.inf): raise Bad
          value(sc.get("items", {}), depth + 1)
          n += 1
      else: raise Bad
    try:
      value(root)
      ws()
      return 2 if i == len(text) and not pending else 0
    except Short: return 1
    except Bad: return 0
  return fits

def answer(text, final=True):
  """Split generated text into reasoning and answer. The answer is empty until its start marker arrives.
  Qwen3 separates them at </think>. gpt-oss uses analysis and final channels separated by <|end|>. While more text
  can come (final False), hold back the end of the reasoning where it may begin a marker, so a stream never shows
  part of one. Once generation ends, a partial marker is ordinary text."""
  held = lambda s, marks: s if final else s[:len(s) - unsure(s, marks)]
  if not final and any(marker.startswith(text) for marker in ("<think>", "<|channel|>")): return "", ""
  if text.startswith("<|channel|>"):
    if "<|message|>" not in text: return "", ""  # the name of the channel is still coming
    head, _, content = text.partition("<|channel|>final<|message|>")
    reasoning = head.removeprefix("<|channel|>analysis<|message|>").partition("<|end|>")[0]
    return held(reasoning, ("<|end|>", "<|channel|>final<|message|>")).strip(), content
  reasoning, _, content = text.partition("</think>")
  return held(reasoning.removeprefix("<think>"), ("</think>",)).strip(), content.lstrip()

def server(engine, name, template="", eos=(), port=8000, wait=600, defaults=None, body_limit=8 << 20, input_wait=10, connections=None,
           per_client=None):
  """Create an HTTP server for an engine whose thread is already running. Serve /v1/models, /v1/completions and
  /v1/chat/completions, with complete responses or server-sent events.

  Return reasoning separately in reasoning_content. Qwen's enable_thinking=False disables reasoning; gpt-oss
  uses reasoning_effort (low, medium or high). Qwen tool calls use OpenAI's tool_calls format and follow
  text when streamed.

  Usage includes prompt_tokens_details.cached_tokens. A stream ends with it in a separate chunk, with empty choices,
  when stream_options.include_usage asks for it. End a request if no text arrives within wait seconds, or when its
  client hangs up. Sampling defaults come from defaults, populated from generation_config.json for temperature,
  top_k, top_p, min_p and repetition_penalty, then from the handler's built-in defaults.

  A connection has input_wait seconds to send its headers and a body of at most body_limit bytes, else it gets 408 or
  413. Each connection has a handler thread. At most connections of them run, 4 × (slots + 8) by default, and at
  most per_client from one address, a quarter of connections by default. So one client fills every slot, but its idle
  sockets can't take every handler. Past the first limit a connection gets 503, past the second 429, each with a
  JSON error that names the limit."""
  for key, v in (("wait", wait), ("input_wait", input_wait)):
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0:
      raise ValueError(f"{key} must be a finite positive number")
  connections = 4 * (len(engine.busy) + 8) if connections is None else connections
  per_client = max(1, connections // 4) if per_client is None and type(connections) is int else per_client
  for key, v in (("body_limit", body_limit), ("connections", connections), ("per_client", per_client)):
    if type(v) is not int or v < 1: raise ValueError(f"{key} must be a positive integer")
  defaults, thinks, harmony = defaults or {}, "<think>" in template, family(template) == "harmony"
  late = {"error": {"message": f"the request's headers and body took more than {input_wait}s"}}
  class Handler(BaseHTTPRequestHandler):
    def handle(self):
      # The deadline is the only bound on reading the request. It shuts the socket for reading, so the read ends and the
      # request gets its 408. A socket timeout would restart at every read, and one that fired during the headers would
      # make http.server drop the connection without a reply.
      self.expired = threading.Event()
      def expire():
        self.expired.set()
        try: self.connection.shutdown(socket.SHUT_RD)
        except OSError: pass
      self.deadline = threading.Timer(input_wait, expire)
      self.deadline.daemon = True
      self.deadline.start()
      try: super().handle()
      finally: self.deadline.cancel()
    def input_done(self):
      self.deadline.cancel()
      self.connection.settimeout(wait)
    def body(self):
      try:
        if self.expired.is_set(): raise TimeoutError  # the deadline cut the headers short
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") or len(lengths) != 1 or not re.fullmatch(r"[0-9]+", lengths[0]):
          raise ValueError("provide one nonnegative Content-Length and no Transfer-Encoding")
        size = int(lengths[0])
        if size > body_limit: raise OverflowError(f"request body exceeds {body_limit} bytes")
        data = self.rfile.read(size)
        if self.expired.is_set(): raise TimeoutError
        if len(data) != size: raise ValueError("request body ended before Content-Length")
        return data
      finally: self.input_done()
    def reply(self, code, body):
      data = json.dumps(body).encode()
      self.send_response(code)
      self.send_header("Content-Type", "application/json"), self.send_header("Content-Length", str(len(data)))
      self.end_headers()
      self.wfile.write(data)
    def do_GET(self):
      self.input_done()
      if self.expired.is_set(): return self.reply(408, late)
      if self.path != "/v1/models": return self.reply(404, {"error": {"message": f"not found: {self.path}"}})
      self.reply(200, {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "baretile"}]})
    def do_POST(self):
      is_chat = self.path == "/v1/chat/completions"
      if not is_chat and self.path != "/v1/completions": return self.reply(404, {"error": {"message": f"not found: {self.path}"}})
      try:
        q = json.loads(self.body())
        if not isinstance(q, dict): raise ValueError("the request must be a JSON object")
        thinking = (q.get("chat_template_kwargs") or {}).get("enable_thinking", True) is not False
        tools = described(q.get("tools")) if is_chat else None
        prompt = chat(template, q["messages"], thinking, q.get("reasoning_effort") or "medium", tools) if is_chat else q["prompt"]
        if not isinstance(prompt, str): raise ValueError("prompt must be a string")
        ids = engine.tok.encode(prompt, special=is_chat)
        if not ids: raise ValueError("a prompt must contain at least one token")
        top = (q.get("top_logprobs") or 0 if q.get("logprobs") else None) if is_chat else q.get("logprobs")  # a completion's is a count
        apart = is_chat and (thinking and thinks or harmony)
        choice, form = q.get("tool_choice"), formatted(q.get("response_format"))
        fits = chosen(choice, tools)
        if form and choice in (None, "auto", "none"):  # the format, or a tool call where tools allow one (auto)
          call = chosen("required", tools) if tools and choice != "none" else None
          fits = form if call is None else (lambda text, pending=b"": max(form(text, pending), call(text, pending)))
        def shaped(text, pending=b""):
          started = "</think>" in text or "<|channel|>final<|message|>" in text
          return fits(answer(text)[1], pending if started else b"")
        shape = shaped if fits and apart else fits
        given = lambda k, default: q[k] if q.get(k) is not None else defaults[k] if defaults.get(k) is not None else default
        limit = "max_completion_tokens" if is_chat and q.get("max_completion_tokens") is not None else "max_tokens"  # OpenAI's chat name first
        counted = (q.get("stream_options") or {}).get("include_usage")  # a stream ends with usage only when asked
        r = Request(engine.tok, ids, max_tokens=given(limit, engine.places if is_chat else 16), temperature=given("temperature", 1.0),
                    top_p=given("top_p", 1.0), stop=q.get("stop"), eos=eos, seed=q.get("seed"), top_k=given("top_k", 0),
                    min_p=given("min_p", 0.0), presence=given("presence_penalty", 0.0), frequency=given("frequency_penalty", 0.0),
                    repetition=given("repetition_penalty", 1.0), logprobs=top, shape=shape)
      except TimeoutError: return self.reply(408, late)  # the deadline cut the headers or body short
      except OverflowError as e: return self.reply(413, {"error": {"message": str(e)}})
      except (ValueError, KeyError, TypeError, AttributeError, RecursionError) as e:  # RecursionError: a body nested too deeply to parse
        return self.reply(400, {"error": {"message": f"{type(e).__name__}: {e}"}})
      if len(ids) >= engine.places - 1: return self.reply(400, {"error": {"message": f"{len(ids)} tokens, but a slot holds {engine.places - 2}"}})
      # Reject immediately when all slots are busy, so the client can retry or use another provider.
      try: offered = engine.offer(r)
      except RuntimeError as e: return self.reply(503, {"error": {"message": str(e), "code": 503}})
      if not offered: return self.reply(429, {"error": {"message": "every slot is busy", "code": 429}})
      head = {"id": f"baretile-{id(r)}", "object": "chat.completion" if is_chat else "text_completion", "created": int(time.time()), "model": name}
      seen = {"text": "", "reasoning_content": "", "content": ""}
      called = lambda text: calls(answer(text)[1] if apart else text, tools) if tools else (None, [])  # the text and tool calls of the answer
      def parts(text, final=True):  # splits a thinking model's text into reasoning and answer, and cuts the answer at its tool calls
        out = dict(zip(("reasoning_content", "content"), answer(text, final))) if apart else {"content": text}
        if tools and (not final or called(text)[1]):  # while more can come, hold back what may begin a call. at the end, cut at calls that parsed
          shown = out["content"].partition("<tool_call>")[0]
          out["content"] = shown if final else shown[:len(shown) - unsure(shown, ("<tool_call>",))]
        return out
      def delta(text, final):  # the next piece of a chat: what's new in each part
        seen["text"] += text
        new = parts(seen["text"], final)
        out = {k: v[len(seen[k]):] for k, v in new.items() if len(v) > len(seen[k])}
        seen.update(new)
        return out
      def piece(text, done):
        return ({"index": 0, "delta": delta(text, done is not None)} if is_chat else {"index": 0, "text": text}) | {"finish_reason": done}
      word, raw = lambda t: engine.tok.bytes.get(t, b"").decode("utf-8", "replace"), lambda t: list(engine.tok.bytes.get(t, b""))
      def chances(made):  # log probabilities in OpenAI's format: a chat's under content, a completion's as lists
        one = lambda t, p: {"token": word(t), "logprob": p, "bytes": raw(t)}
        if is_chat: return {"content": [one(t, p) | {"top_logprobs": [one(a, b) for a, b in top]} for t, p, top in made]}
        return {"tokens": [word(t) for t, _, _ in made], "token_logprobs": [p for _, p, _ in made],
                "top_logprobs": [{word(a): b for a, b in top} for _, _, top in made]}
      def logprobs(k=None):  # the log probabilities of all its tokens, or of those in its k-th piece
        if r.logprobs is None: return {"logprobs": None}
        return {"logprobs": chances(r.probs if k is None else r.probs[r.marks[k - 1] if k else 0:r.marks[k]])}
      usage = lambda: {"prompt_tokens": len(ids), "completion_tokens": len(r.made), "total_tokens": len(ids) + len(r.made),
                       "prompt_tokens_details": {"cached_tokens": r.cached}}
      hangup = select.poll()
      hangup.register(self.connection, select.POLLIN)  # after its request a client sends nothing, so input means it hung up
      def next_text():  # the next piece of r's text. ConnectionError once the client hangs up, queue.Empty after wait seconds without any
        end = time.monotonic() + wait
        while not hangup.poll(0):
          try: return r.out.get(timeout=min(1, wait))
          except queue.Empty:
            if time.monotonic() > end: raise
        raise ConnectionError("the client hung up")
      if q.get("stream"):
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        send = lambda body: (self.wfile.write(f"data: {json.dumps(body)}\n\n".encode()), self.wfile.flush())
        head["object"] += ".chunk" if is_chat else ""
        try:
          if is_chat: send(head | {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
          for k, text in enumerate(iter(next_text, None)):  # each piece with the log probabilities of its tokens
            send(head | {"choices": [piece(text, None) | logprobs(k)]})
          if made := called(seen["text"])[1]:
            calling = {"tool_calls": [{"index": i} | c for i, c in enumerate(made)]}
            send(head | {"choices": [{"index": 0, "delta": calling, "finish_reason": None}]})
          send(head | {"choices": [piece("", "tool_calls" if made else r.finish)]})
          if counted: send(head | {"choices": [], "usage": usage()})
          return self.wfile.write(b"data: [DONE]\n\n")
        except (OSError, queue.Empty): r.gone = True  # the client left, or no text came, so let its slot go
        return
      try: text = "".join(iter(next_text, None))
      except ConnectionError: r.gone = True; return  # nobody is waiting for it now, so let its slot go
      except queue.Empty:
        r.gone = True  # nobody is waiting for it now, so let its slot go
        return self.reply(504, {"error": {"message": f"no text came in {wait}s: the chip may be hung"}})
      if r.finish == "error": return self.reply(500, {"error": {"message": r.error or "the engine stopped on an error"}})
      message, (content, made) = parts(text), called(text)
      if made: message |= {"content": content or None, "tool_calls": made}
      choice = ({"index": 0, "message": {"role": "assistant"} | message} if is_chat else {"index": 0, "text": text}) | logprobs()
      self.reply(200, head | {"choices": [choice | {"finish_reason": "tool_calls" if made else r.finish}], "usage": usage()})
  class Server(ThreadingHTTPServer):
    def process_request(self, request, address):
      with self.lock:  # clients counts the handlers of each address
        if sum(self.clients.values()) >= connections: refused = 503, f"the server's connection limit ({connections}) is reached"
        elif self.clients[address[0]] >= per_client: refused = 429, f"this address's connection limit ({per_client}) is reached"
        else: self.clients[address[0]] += 1; refused = None
      if refused:  # from the thread that accepts connections, so the reply mustn't wait long
        code, message = refused
        data = json.dumps({"error": {"message": message, "code": code}}).encode()
        try:
          request.settimeout(min(input_wait, 1))
          request.sendall(f"HTTP/1.0 {code} {Handler.responses[code][0]}\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\n"
                          f"Connection: close\r\n\r\n".encode() + data)
        except OSError: pass
        finally: self.shutdown_request(request)
        return
      try: super().process_request(request, address)
      except BaseException: self.leave(address[0]); raise
    def process_request_thread(self, request, address):  # ThreadingMixIn's, freeing the handler before the client sees the end
      try: self.finish_request(request, address)
      except Exception: self.handle_error(request, address)
      finally:
        self.leave(address[0])
        self.shutdown_request(request)
    def leave(self, client):
      with self.lock:
        self.clients[client] -= 1
        if not self.clients[client]: del self.clients[client]
  http = Server(("", port), Handler)
  http.lock, http.clients = threading.Lock(), collections.Counter()
  http.connections, http.per_client = connections, per_client
  return http

class ScopedRedirect(urllib.request.HTTPRedirectHandler):
  """Keep download credentials only across redirects within the same origin."""
  def redirect_request(self, req, fp, code, msg, headers, newurl):
    redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
    def origin(url):
      p = urllib.parse.urlsplit(url)
      return p.scheme, p.hostname, p.port or {"http": 80, "https": 443}.get(p.scheme)
    if redirected is not None and origin(req.full_url) != origin(newurl): redirected.remove_header("Authorization")
    return redirected

def fetch(model, cache=os.path.expanduser("~/.cache/baretile/hub")):
  """Return a local model folder, downloading a Hugging Face ID if needed. Fetch config, tokenizer, optional
  chat and generation settings, and safetensors weights or indexed shards. HF_ENDPOINT changes the server;
  HF_TOKEN authorizes gated downloads. Mark completed downloads with .done and reuse them."""
  if os.path.isdir(model): return model
  if not re.fullmatch(r"[\w.-]+(?:/[\w.-]+)?", model) or any(p in (".", "..") for p in model.split("/")):
    raise ValueError("model must be an existing folder or a Hugging Face model ID")
  opener = urllib.request.build_opener(ScopedRedirect())
  folder = os.path.join(cache, model)
  if os.path.exists(os.path.join(folder, ".done")): return folder
  os.makedirs(folder, exist_ok=True)
  def get(name, needed=True):
    if not isinstance(name, str) or not re.fullmatch(r"[\w./-]+", name) or any(p in ("", ".", "..") for p in name.split("/")):
      raise ValueError(f"invalid model file path: {name!r}")
    path = os.path.join(folder, name)
    if os.path.exists(path): return path
    token = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if "HF_TOKEN" in os.environ else {}
    url = f"{os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}/{model}/resolve/main/{name}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(path), suffix=".part", delete=False) as f:
      try:
        try:
          with opener.open(urllib.request.Request(url, headers=token), timeout=60) as r: shutil.copyfileobj(r, f)
        except urllib.error.HTTPError as e:
          if needed or e.code != 404: raise
          return None
        f.close()
        os.replace(f.name, path)
      finally:
        if os.path.exists(f.name): os.unlink(f.name)
    return path
  for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"):
    get(name, name in ("config.json", "tokenizer.json"))
  if index := get("model.safetensors.index.json", False):
    for shard in sorted(set(llama.read_json(index)["weight_map"].values())):
      if not isinstance(shard, str) or not shard.endswith(".safetensors"): raise ValueError(f"invalid safetensors shard: {shard!r}")
      get(shard)
  else: get("model.safetensors")
  with open(os.path.join(folder, ".done"), "w"): pass
  return folder

def generation_defaults(config, generation):
  """Resolve EOS and sampling defaults per field: generation_config.json, then config.json's text_config (a
  multimodal model's text model), then config.json itself. An explicit null EOS disables automatic end tokens."""
  fields = config | (config.get("text_config") or {}) | generation
  eos = fields.get("eos_token_id")
  defaults = {k: v for k, v in fields.items() if k in ("temperature", "top_k", "top_p", "min_p", "repetition_penalty")}
  return eos if isinstance(eos, list) else [] if eos is None else [eos], defaults

def main():
  from baretile import pci
  from baretile.chip import Chip
  from baretile.runtime import Runtime
  folder, args = fetch(sys.argv[1]), sys.argv[2:] + [None] * 6
  port, slots, places, mode, kv = int(args[0] or 8000), int(args[1] or 4), int(args[2] or 512), args[3] or "bf16", args[4] or None
  connections = int(args[5]) if args[5] else None
  chip = Chip(pci.open_device())
  rt, start = Runtime(chip), time.time()
  c, weights = llama.load(folder, places)
  c.precision = llama.precision(c, mode, kv)
  engine = Engine(rt, llama.upload(rt, weights, c.precision), c, llama.Tokenizer(folder), slots, places)
  config = lambda f: llama.read_json(os.path.join(folder, f)) if os.path.exists(os.path.join(folder, f)) else {}
  generation, name = config("generation_config.json"), os.path.basename(folder.rstrip("/"))
  eos, defaults = generation_defaults(config("config.json"), generation)
  threading.Thread(target=engine.run, daemon=True).start()
  jinja = os.path.join(folder, "chat_template.jinja")  # where transformers 5 keeps it
  template = config("tokenizer_config.json").get("chat_template") or (open(jinja, encoding="utf-8").read() if os.path.exists(jinja) else "")
  http = server(engine, name, template, eos, port, defaults=defaults, connections=connections)
  print(f"{name} is on the chip after {time.time() - start:.0f}s ({mode}, a {llama.cache_format(c)} KV cache), {slots} slots of {places} positions, "
        f"prefilling {engine.chunk} tokens at a time, on port {port} with {http.connections} connections, {http.per_client} per address", flush=True)
  http.serve_forever()

if __name__ == "__main__": main()
