# An OpenAI-compatible server for a Llama-family model on baretile. Requests join and leave a batch that decodes
# together. Sequences have slots in a shared KV cache (and in the Gated DeltaNet states), and a step is one replayed
# call for all the slots, or two when speculating. A prompt goes to the free slot that already holds the longest prefix
# of it, or takes a kept session that holds a longer one. Sessions are kept on the chip, and on the host once the chip
# is full. A shared prefix, like a system prompt or an agent's conversation so far, isn't prefilled again.
#   TTSIM=/path/to/libttsim.so python3 models/serve.py model [port] [slots] [places] [bf16|bfp8|bfp4|tt|experts] [bf16|bfp8|bfp4]
# model is a folder or a Hugging Face model id. Then come the formats of the weights (llama.precision) and the KV cache.
#   curl localhost:8000/v1/completions -d '{"prompt": "The meaning of life is", "max_tokens": 8, "temperature": 0}'
import codecs, collections, functools, json, math, os, queue, re, shutil, sys, threading, time, urllib.error, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baretile import bh
from baretile.kernels import topk_room
from baretile.tensix import tile_bytes
from baretile.tensor import Tensor, jit
import gpt2, llama, qwen

class Head:
  """A row of logits as the chip gives it (heads). values and ids are its k largest logits, descending, and their
  tokens. lse is the log of the sum of exp(x t) over the row, where t is the 1/temperature of the request. row() reads
  the row back from the chip, for a draw that needs it. Head.of makes a Head from a row of logits in numpy."""
  def __init__(self, values, ids, lse, row): self.values, self.ids, self.lse, self.row = np.asarray(values, np.float64), ids, float(lse), row
  @staticmethod
  def of(logits, t=1.0, k=64):  # finds the k-th largest with a partition, then sorts the logits at least that large. Ties keep the lower id first
    bar = np.partition(logits, len(logits) - k)[len(logits) - k] if k < len(logits) else logits.min()
    ids = np.flatnonzero(logits >= bar)
    ids = ids[np.argsort(-logits[ids], kind="stable")][:k]
    m = float(logits[ids[0]])
    return Head(logits[ids], ids, m * t + math.log(np.exp((np.float64(logits) - m) * t).sum()), lambda: logits)

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
  x, m = logits.split(parts), best.split(parts).max(2).realize()  # the largest of each part, taken from its best, so the row is read once
  lse = ((x - m) * t).exp().sum(2).log() + m * t  # for each part
  if parts > 1: top = lse.max(0); lse = (lse - top).exp().sum(0).log() + top
  return values.realize(), ids.realize(), lse.realize()

class Request:
  """A prompt and how to go on from it (pick), until a token in eos, a stop string, or max_tokens.

  Each piece of new text goes into out as soon as no stop string could still begin in it. None follows once the engine
  is done with the request (end), and finish says why. If the client has gone (gone), the request ends at its next
  token, or before it takes a slot. cached counts the prompt tokens that its slot or a kept session already held, which
  weren't prefilled. With logprobs set, probs gets the log probability of each token made, and those of the logprobs
  most likely tokens, from the raw logits of the model. marks has the length of probs at each piece of text. shape is
  what the text must keep to (pick), or None. room is the slot that it holds from its offer to its end (Engine.offer),
  or None if it wasn't offered."""
  def __init__(self, tokenizer, ids, max_tokens=16, temperature=1.0, top_p=1.0, stop=(), eos=(), seed=None, top_k=0, min_p=0.0,
               presence=0.0, frequency=0.0, repetition=1.0, logprobs=None, shape=None):
    self.tok, self.ids, self.max_tokens, self.temperature, self.top_p = tokenizer, list(ids), max_tokens, temperature, top_p
    self.stop, self.eos, self.rng = [stop] if isinstance(stop, str) else list(stop or ()), set(eos), np.random.default_rng(seed)
    self.top_k, self.min_p, self.presence, self.frequency, self.repetition, self.logprobs = top_k, min_p, presence, frequency, repetition, logprobs
    self.shape = shape
    self.made, self.said, self.out, self.finish, self.gone, self.cached, self.probs, self.marks = [], "", queue.Queue(), None, False, 0, [], []
    self.text, self.utf8 = "", codecs.getincrementaldecoder("utf-8")("replace")  # the text so far, except a character that's still coming
    self.room = None

  def whole(self):
    """Whether it needs the row of logits past the head. Penalties can move tokens from past the head into it, a shape
    may turn down all of the head, and log probabilities are at temperature 1."""
    return bool(self.presence or self.frequency or self.repetition != 1 or self.shape or self.logprobs is not None)

  def pick(self, head):
    """The next token. It comes from the head of the logits (Head) when the head decides the draw (sample), else from
    the row read back. The row first takes the penalties: presence and frequency by the tokens made so far, and
    repetition by those and the prompt. vLLM applies them the same way. Then draw picks a token.
    shape is a function of the text made: 0 if the text can't begin anything that fits, 1 if it can, 2 if it fits and
    is complete. A drawn token is taken if its text can still fit, or for an end of sequence, if the text is complete.
    Otherwise it's drawn again without that token. Masking 248K tokens a step costs more: once the model is in the
    shape, this is a check or two."""
    if not self.whole() and (token := self.sample(head)) is not None: return token
    logits = head.row()
    if self.presence or self.frequency or self.repetition != 1:
      made = np.bincount(self.made, minlength=len(logits))[:len(logits)]
      logits = logits - self.presence * (made > 0) - self.frequency * made
      seen = (made > 0) | (np.bincount(self.ids, minlength=len(logits))[:len(logits)] > 0)
      logits = np.where(seen, np.where(logits > 0, logits / self.repetition, logits * self.repetition), logits)
    if self.shape is None: return self.draw(logits)
    logits = np.array(logits, np.float32)
    for _ in range(32):
      if self.fits(token := self.draw(logits)): return token
      logits[token] = -np.inf
    # The likeliest token that fits, else None, which ends the request. None comes for a complete text when the model
    # has no end of sequence token, or for a shape that leads nowhere.
    return next((int(t) for t in np.argsort(-logits) if logits[t] > -np.inf and self.fits(int(t))), None)

  def fits(self, token):  # whether the text with token still fits. For an end of sequence, whether the text is complete
    return self.shape(self.tok.decode(self.made)) == 2 if token in self.eos else self.shape(self.tok.decode(self.made + [token])) > 0

  def sample(self, head):
    """The token that draw would pick, found from the head without reading the row back. Returns None where the head
    doesn't decide it: a top_k larger than the head, or a top_p or min_p cut that reaches past it. Both are known before
    drawing. With no top_k, top_p or min_p, a draw that lands past the mass of the head reads the row back and draws
    from the tokens past the head, so all tokens keep their probabilities."""
    if self.temperature == 0: return int(head.ids[0])
    t, k = 1 / self.temperature, len(head.ids)
    p, total = np.exp((head.values - head.values[0]) * t), math.exp(head.lse - head.values[0] * t)  # relative to the largest
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
    """Draws from a row of logits: greedy at temperature 0, else sampled at that temperature. It keeps the top_k most
    likely (0 for all), then the fewest of those whose probabilities sum to top_p, then those at least min_p of the most
    likely. It looks for the fewest among the 1,024 most likely first, since sorting a vocabulary of 248K takes 5 ms on
    the host."""
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
    """Takes the next token from the Head of the logits, or from their row in numpy. room says whether the slot
    has a position left to run this token. Returns whether the request is done."""
    if self.gone: self.finish = "gone"; return True
    head = head if isinstance(head, Head) else Head.of(head, scale(self))
    token = self.pick(head)
    end = token is None or token in self.eos
    if not end:
      self.made.append(token)
      if self.logprobs is not None:  # from the raw logits, before any penalty or temperature. vLLM does the same
        logits = head.row()
        lp = logits - logits.max()
        lp = lp - np.log(np.exp(lp).sum())
        top = np.argpartition(-lp, self.logprobs)[:self.logprobs] if self.logprobs else []
        self.probs.append((token, float(lp[token]), [(int(t), float(lp[t])) for t in sorted(top, key=lambda t: -lp[t])]))
    before = len(self.text)  # where to look for a stop string that reaches into what this token adds
    self.text += self.utf8.decode(b"" if end else self.tok.bytes[token])
    start = max(0, before - max(map(len, self.stop), default=1) + 1)
    cut = min((i for i in (self.text.find(s, start) for s in self.stop) if i >= 0), default=-1)
    if end or cut >= 0 or len(self.made) == self.max_tokens or not room:
      text = self.text + self.utf8.decode(b"", True)
      self.say(text if cut < 0 else text[:cut])
      self.finish = "stop" if end or cut >= 0 else "length"
      return True
    hold = max((k for s in self.stop for k in range(1, len(s)) if self.text.endswith(s[:k])), default=0)  # what may begin a stop string
    self.say(self.text[:len(self.text) - hold])
    return False

  def say(self, text):
    if len(text) > len(self.said): self.marks.append(len(self.probs)); self.out.put(text[len(self.said):]); self.said = text

  def end(self):
    if self.room: self.room.release(); self.room = None  # frees its slot for the next offer
    self.out.put(None)

class Engine:
  """Sequences decoding together on the chip. Each has a slot of `places` rows in the shared KV cache, and Gated
  DeltaNet states of its own.

  A step runs all the slots, so its shapes don't change and it can be replayed (jit). A slot that's free or still
  prefilling runs on its last row, which no sequence reaches. It sees only the first row of its slot and leaves its
  states alone. With attention in one kernel, slots read their rows up to the positions of their sequences, and one
  replay serves any length. Otherwise attention reads the smallest power of 2 rows (at least 32) that holds the
  longest sequence, with a replay recorded for each. Either way a step costs what the sequences hold, not the size of
  their slots. A prompt is prefilled `chunk` tokens at a time, after what its slot already holds, one chunk between
  steps. A long prompt slows the other sequences down but doesn't stop them.

  A session is the tokens, cache rows and states of a slot. It's kept when its prompt is in, and again when its
  request ends. The next turn of an agent starts with its last prompt and answer, so it goes on from where it left
  off, whichever slot it gets. Sessions are copied on the chip, into DRAM that the caches don't use, up to `park`
  bytes. By default that's what's free when the engine starts, minus 2 GiB. Past that, the least recently used go to
  the host, up to `keep` bytes, and past that they're dropped. A copy on the chip, either way, is a few launches and
  no PCIe. The rows of a KV cache serve any prefix of what they hold, but a Gated DeltaNet state can't be rewound to a
  prefix of the tokens it took in.

  A model with a multi-token prediction layer (Qwen3.5) decodes speculatively. A step verifies the next tokens of the
  slots, each with a draft after it, in one call (accept in llama.forward). Where the model picks the draft as the
  second token, it takes that token too. Picks are sampled as usual, so the tokens are the model's own at any temperature.
  Another call makes the next drafts (qwen.draft). When a draft is turned down, the next step puts back its Gated
  DeltaNet states."""
  def __init__(self, rt, t, c, tokenizer, slots=4, places=512, chunk=None, keep=4 << 30, park=None):
    assert places % 32 == 0, "places must be a multiple of 32, so that slots are whole tiles"
    self.rt, self.t, self.c, self.tok, self.places, self.keep = rt, t, c, tokenizer, places, keep
    self.chunk = chunk or (2048 if c.experts else 256)  # with a mixture of experts, a chunk reads nearly all the experts of the blocks (cost.prompt)
    self.spec = c.mtp > 0 and "mtp.fc" in t  # drafts with the multi-token prediction layer
    self.kv = llama.kv_cache(c, slots * places, llama.zeros(rt, llama.cache_format(c)), slots, self.spec)
    self.slot = [llama.slot(c, self.kv, u, slots) for u in range(slots)]  # the caches as slot u sees them
    self.park = max(0, sum(room for _, room in rt.holes) * len(bh.DRAM) - (2 << 30)) if park is None else park
    self.linear = any(llama.linear(c, l) for l in range(c.layers))  # Gated DeltaNet states then decide where a slot can go on from
    self.attends = llama.attends(c, t)  # attention in one kernel reads the rows up to each position, so a replay serves any length
    self.held, self.busy = [[] for _ in range(slots)], [None] * slots
    self.used = [0] * slots  # when each slot last took a request. Of slots that hold as much of a prompt, admit picks the least recently used
    # for each call and length it reads: its replay, what its last call made (the logits and their heads), and the
    # hidden states from check
    self.replays, self.made, self.hidden = {}, {}, {}
    self.drafts, self.taken = [0] * slots, [1.0] * slots  # each slot's draft, and whether the last was taken, so states go on from both tokens
    self.drafted = self.agreed = 0  # drafts checked against the pick of the model, and those taken. This shows what speculation saves
    self.waiting, self.prefilled = queue.Queue(), 0  # the tokens prefilled so far. This shows what the prefix cache saves
    self.room = threading.BoundedSemaphore(slots)  # a slot for each offered request, from its offer to its end
    self.sessions = collections.OrderedDict()  # tokens -> each block's KV cache rows or Gated DeltaNet states, as tensors on the chip or bytes

  def on(self, row, dtype="int32"): return Tensor.of(self.rt, [row], dtype)

  def offer(self, r):  # queues r if a slot is free and holds the slot until r ends. Otherwise it returns False without waiting
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

  def rows(self, f, n, ts):
    """The rows from the last call of f that read n rows, as Heads at the 1/temperature of each row (ts). It reads
    back the heads from the chip, and the logits too, once, if a draw needs them."""
    logits, made = self.made[f, n]
    whole = functools.cache(lambda: gpt2.values(logits))
    if made is None: return [Head.of(row, x) for row, x in zip(whole(), ts)]
    values, ids, lse = (gpt2.values(x) for x in made)
    return [Head(values[i], ids[i], lse[i, 0], lambda i=i: whole()[i]) for i in range(len(ts))]

  def column(self, ts): return Tensor.of(self.rt, [[x] for x in ts], "fp32")  # the 1/temperature of each row, as a column

  def reach(self, held, ids):  # how much of prompt ids can go on from the state after held. Its last token always runs again
    if self.linear: return len(held) if len(held) < len(ids) and list(ids[:len(held)]) == list(held) else 0
    return min(next((i for i, (a, b) in enumerate(zip(held, ids)) if a != b), min(len(held), len(ids))), len(ids) - 1)

  def admit(self, r):  # into the free slot that holds most of its prompt, or one that takes the kept session that holds more
    if r.gone: return r.end()
    free = [u for u, b in enumerate(self.busy) if b is None]
    u = max(free, key=lambda u: (self.reach(self.held[u], r.ids), -self.used[u]))
    best = max(self.sessions, key=lambda k: self.reach(k, r.ids), default=())
    if self.reach(best, r.ids) > self.reach(self.held[u], r.ids): self.restore(u := min(free, key=lambda u: self.used[u]), best)
    self.used[u], self.busy[u] = max(self.used) + 1, r
    n = self.reach(self.held[u], r.ids)
    if self.linear and n < len(self.held[u]): self.restore(u)  # zero its states and start over
    elif not self.taken[u]:  # after a turned-down draft, a prompt goes on from the states after the token before it
      for l, s in enumerate(self.slot[u]):
        if llama.linear(self.c, l): s[0].assign(s[2]), s[1].assign(s[3])
      self.taken[u] = 1.0
    self.held[u], r.cached = r.ids[:n], n

  def save(self, u):
    """Keeps the session of slot u by copying it on the chip: its tokens, its rows of the KV caches (rounded up to a
    tile) and its states. After a turned-down draft, those are the states after the token before it. Past park bytes
    on the chip, the least recently used go to the host. Past keep bytes there, they're dropped."""
    key, rows = tuple(self.held[u]), -(-len(self.held[u]) // 32) * 32
    if not key or key in self.sessions or not self.park + self.keep: return
    def copy(x):  # the row of a conv state is copied by a move
      x = x.get() if isinstance(x, qwen.Row) else x
      return Tensor(self.rt, x.rows, x.cols, batch=x.batch, dtype=x.dtype).assign(x)
    self.sessions[key] = [tuple(copy(x) for x in (s[:2] if self.taken[u] else s[2:])) if llama.linear(self.c, l) else copy(s[:, :rows, :])
                          for l, s in enumerate(self.slot[u])]
    each = lambda v: [x for part in v for x in (part if isinstance(part, tuple) else (part,))]  # the tensors or bytes of a session
    size = lambda v: sum(len(x) if isinstance(x, bytes) else x.batch * -(-x.rows // 32) * -(-x.cols // 32) * tile_bytes(x.dtype) for x in each(v))
    for kind, room in ((Tensor, self.park), (bytes, self.keep)):
      held = [k for k, v in self.sessions.items() if isinstance(each(v)[0], kind)]
      while sum(size(self.sessions[k]) for k in held) > room:
        k = held.pop(0)
        if kind is bytes or not self.keep: del self.sessions[k]
        else: self.sessions[k] = [tuple(x.tobytes() for x in part) if isinstance(part, tuple) else part.tobytes() for part in self.sessions[k]]

  def restore(self, u, key=()):  # puts a kept session into slot u, which then holds its tokens. With no key, the states are zeroed
    if key: self.sessions.move_to_end(key)
    rows = -(-len(key) // 32) * 32
    for l, s in enumerate(self.slot[u]):
      data = self.sessions[key][l] if key else None
      if llama.linear(self.c, l):
        for i, x in enumerate(s[:2]):
          x.assign(data[i] if data and isinstance(data[i], Tensor) else
                   Tensor.frombytes(self.rt, x.rows, x.cols, data[i] if data else bytes(4 * x.batch * x.rows * x.cols), x.batch, "fp32"))
      elif isinstance(data, Tensor): s[:, :rows, :].assign(data)
      elif data: s.put(self.on(list(range(rows))), Tensor.frombytes(self.rt, rows, s.cols, data, s.batch))
    self.held[u], self.taken[u] = list(key), 1.0

  def fill(self, u):
    """Prefills the next chunk of the prompt in slot u. The last logits of the last chunk give the first new token.
    When speculating, the multi-token prediction layer also runs over the positions of the chunk, and the last
    position gives the first draft."""
    r, at = self.busy[u], len(self.held[u])
    if r.gone: return self.take(u, None)
    ids = r.ids[at:at + self.chunk]
    places, n = self.on(list(range(at, at + len(ids)))), self.length(at + len(ids) - 1)
    out = llama.forward(self.on(ids), places, self.t, self.c, self.slot[u], hidden=self.spec, length=n)
    out, h = out if self.spec else (out, None)
    self.held[u] += ids
    self.prefilled += len(ids)
    if len(self.held[u]) == len(r.ids): self.save(u); self.take(u, Head.of(gpt2.values(out)[0], scale(r)), 1 + self.spec)
    if self.spec and self.busy[u] is r:  # the next token at each position: from the prompt, and past its end the one just taken
      after = self.on((r.ids + r.made)[at + 1:at + len(ids) + 1])
      drafts = gpt2.values(qwen.draft(after, places, h, self.t, self.c, self.slot[u], length=n, every=False))
      if r.made: self.drafts[u], self.taken[u] = self.guess(u, int(drafts[-1].argmax())), 1.0

  def advance(self):  # a step of all the slots. The last token of each decoding slot goes in, and its next one comes out
    live = [r if r and r.made else None for r in self.busy]
    if not any(live): return
    if self.spec: return self.speculate(live)
    ids = [r.made[-1] if r else 0 for r in live]
    places = [len(h) if r else self.places - 1 for h, r in zip(self.held, live)]
    n, ts = self.length(max(p for p, r in zip(places, live) if r)), [scale(r) for r in live]
    self.replay(self.decode, n)(self.on(ids), self.on(places), self.column(ts))
    for u, (r, head) in enumerate(zip(live, self.rows(self.decode, n, ts))):
      if r: self.held[u].append(r.made[-1]); self.take(u, head)

  # The request in slot u takes its next token. A finished request leaves its slot, and its session is kept before
  # the request ends, so whoever waits on it finds the chip idle.
  def take(self, u, head, room=1):
    r = self.busy[u]
    if r.take(head, len(self.held[u]) + room < self.places): self.save(u); self.busy[u] = None; r.end()
    return self.busy[u] is None

  def decode(self, ids, places, t, length):  # replayed: the next tokens of the slots, giving their logits and their heads at t
    logits = llama.forward(ids, places, self.t, self.c, self.kv, len(self.busy), length=length).realize()
    return logits, heads(logits, t)

  def check(self, ids, places, accept, t, length):
    """Replayed when speculating. Takes in the token and draft of each slot, from the states that accept picks.
    Returns their logits and heads, and keeps their hidden states for guesses."""
    logits, self.hidden[length] = llama.forward(ids, places, self.t, self.c, self.kv, len(self.busy), accept=accept, hidden=True, length=length)
    return logits.realize(), heads(logits, t)

  def guesses(self, ids, places, taken, length):
    """Replayed: the multi-token prediction layer over the last two positions of each slot. Where the draft was turned
    down, the second is the first again. Returns the logits, and the best token of each row."""
    n, d = len(self.busy), self.c.dim
    h = self.hidden[length].reshape(n, 2, d)
    first = h[:, 0:1, :]
    # the second row of each slot: its second hidden state where taken is 1, and its first again where taken is 0
    h = first.cat(first + taken.reshape(n, 1, 1) * (h[:, 1:2, :] - first), axis=1).reshape(2 * n, d).cast("bf16")
    logits = qwen.draft(ids, places, h, self.t, self.c, self.kv, n, length).realize()
    return logits, heads(logits, k=1)

  def guess(self, u, best): return best  # slot u's draft: the best token of the multi-token prediction layer. test_serve overrides it

  def speculate(self, live):
    """A step of all the slots. It verifies the token and the draft of each slot, takes the draft where the pick of
    the model agrees, then makes the next drafts."""
    n, at = len(live), [len(h) for h in self.held]
    ids = [x for u, r in enumerate(live) for x in ([r.made[-1], self.drafts[u]] if r else [0, 0])]
    places = [p for u, r in enumerate(live) for p in ([at[u], at[u] + 1] if r else [self.places - 1] * 2)]
    m, ts = self.length(max(at[u] + 1 for u, r in enumerate(live) if r)), [x for r in live for x in (scale(r),) * 2]  # the draft positions too
    self.replay(self.check, m)(self.on(ids), self.on(places), self.on(self.taken, "fp32"), self.column(ts))
    rows = self.rows(self.check, m, ts)
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
    try:
      while True:
        if not any(self.busy): self.admit(self.waiting.get())
        while None in self.busy and not self.waiting.empty(): self.admit(self.waiting.get())
        if filling := [u for u, r in enumerate(self.busy) if r and not r.made]: self.fill(filling[0])
        self.advance()
    except BaseException:
      for r in self.busy + list(self.waiting.queue):
        if r: r.finish = "error"; r.end()
      raise

FAMILIES = (("qwen", "<tool_call>\n<function="), ("harmony", "<|channel|>"), ("llama", "<|start_header_id|>"), ("chatml", "<|im_start|>"))
def family(template): return next((name for name, mark in FAMILIES if mark in template), None)  # the family, by a mark that only it has

def chat(template, messages, thinking=True, effort="medium", tools=None):
  """Turns messages into the prompt format that the model was trained on. The family of template picks the format:
  Qwen3.5 (qwen), gpt-oss (harmony), Llama 3, or ChatML."""
  kind = family(template)
  if kind == "qwen": return qwen_chat(messages, thinking, tools)
  if tools: raise ValueError("tools only work with the chat template of Qwen3.5")
  if kind == "harmony":  # gpt-oss has a fixed system message. A first system message in the request goes as a developer message
    system = (f"You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: {time.strftime('%Y-%m-%d')}\n\n"
              f"Reasoning: {effort}\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.")
    head, turns = f"<|start|>system<|message|>{system}<|end|>", messages
    if messages and messages[0]["role"] in ("system", "developer"):
      head, turns = head + f"<|start|>developer<|message|># Instructions\n\n{messages[0]['content']}\n\n<|end|>", messages[1:]
    said = lambda m: "<|start|>user<|message|>" if m["role"] == "user" else "<|start|>assistant<|channel|>final<|message|>"
    return head + "".join(f"{said(m)}{m['content']}<|end|>" for m in turns) + "<|start|>assistant"
  if kind == "llama":  # Llama 3
    turn = lambda m: f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
    return "<|begin_of_text|>" + "".join(map(turn, messages)) + "<|start_header_id|>assistant<|end_header_id|>\n\n"
  if kind == "chatml":  # the SmolLM instruct models and Qwen. When Qwen3 isn't thinking, its reasoning is given empty
    prompt = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n"
    return prompt if thinking or "<think>" not in template else prompt + "<think>\n\n</think>\n\n"
  raise ValueError("the chat template isn't one we know: Qwen3.5, gpt-oss, Llama 3 or ChatML")

CALLING = ("\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
           "<function=example_function_name>\n"
           "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
           "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the "
           "specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required "
           "parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function "
           "call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do "
           "not tell the user about function calls\n</IMPORTANT>")  # how to call a tool, word for word from the Qwen3.5 template

def qwen_chat(messages, thinking=True, tools=None):
  """The chat template of Qwen3.5 and Qwen3.6 (chat_template.jinja, for text). The tools are described in the system
  turn, tool calls are XML, and tool results go in a user turn of their own. Assistant turns after the last question
  keep their reasoning. Those turns are the tool calls of an agent since the question, so its next prompt goes on from
  its last prompt and answer."""
  text = lambda m: (m.get("content") or "").strip()
  answered = lambda m: text(m).startswith("<tool_response>") and text(m).endswith("</tool_response>")  # a tool result, sent as a user turn
  question = [i for i, m in enumerate(messages) if m["role"] == "user" and not answered(m)]
  if not question: raise ValueError("there's no user question among the messages")
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

def calls(text, tools=()):
  """Turns the tool calls of an answer from Qwen3.5's XML into OpenAI's form. Returns (the text before them, the
  calls), with the arguments of each call as JSON. The schema of the tool decides each value: a string stays as it is,
  and anything else is read as JSON where it parses."""
  head, mark, rest = text.partition("<tool_call>")
  schema = {t.get("function", t).get("name"): t.get("function", t).get("parameters", {}).get("properties", {}) for t in tools or ()}
  def value(name, key, v):
    if schema.get(name, {}).get(key, {}).get("type", "string") == "string": return v
    try: return json.loads(v)
    except ValueError: return v
  found = [(name, {k: value(name, k, v) for k, v in re.findall(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", body, re.S)})
           for name, body in re.findall(r"<function=([^>\n]+)>(.*?)</function>", mark + rest, re.S)]
  call = lambda name, args: {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}
  return head.strip(), [{"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function", "function": call(name, args)} for name, args in found]

def chosen(choice, tools):
  """The shape that tool_choice gives an answer, as Request.pick takes it, or None for anything (auto). none allows
  no tool call. required means one or more tool calls and nothing before them. A function by name allows calls to
  that function and no others. The answer of a thinking model is its text after the reasoning (answer)."""
  if choice in (None, "auto"): return None
  if choice == "none": return lambda text: 0 if "<tool_call>" in text else 2
  names = [t.get("function", t).get("name") for t in tools or ()]
  if choice != "required": names = [n for n in names if n == choice["function"]["name"]]
  if not names: raise ValueError(f"tool_choice: there's no such tool in tools: {choice}")
  can = lambda part, whole: whole.startswith(part) or part.startswith(whole)  # part could still become whole, or starts with it
  def fits(text):  # the call must come first, with no whitespace before it, since that could go on forever
    if not can(text, "<tool_call>\n<function="): return 0
    if not all(any(can(part, n + ">") for n in names) for part in text.split("<function=")[1:]): return 0
    return 2 if text.rstrip().endswith("</tool_call>") and calls(text, tools)[1] else 1
  return fits

class Short(Exception): pass  # the text ends inside a value that could still fit
class Bad(Exception): pass  # the text can't begin a value that fits

def formatted(form):
  """The shape that response_format gives an answer, like chosen's, or None for any text. json_object means a JSON
  object, and json_schema a JSON value that fits its schema.

  We keep to type (one or a list), properties, required, additionalProperties, items, minItems and maxItems, enum and
  const (as json.dumps writes them), anyOf and oneOf, and $ref into $defs or definitions. The rest, like the pattern
  or format of a string or the bounds of a number, isn't enforced. There's at most one space, tab or newline between
  tokens, and a number has at most 32 characters, since either could go on forever. A model held to a shape can
  produce endless whitespace, and Qwen models have done it in vLLM. A comma is allowed only where a key or an item
  may follow, so no prefix of a shape is a dead end."""
  kind = (form or {}).get("type", "text")
  if kind == "text": return None
  if kind not in ("json_object", "json_schema"): raise ValueError(f"response_format: unknown type {kind}")
  root = {"type": "object"} if kind == "json_object" else form["json_schema"]["schema"]
  def fits(text):
    i = 0
    def ws():
      nonlocal i
      start = i
      while i < len(text) and text[i] in " \t\n\r": i += 1
      if i - start > 1: raise Bad
    def peek():
      ws()
      if i == len(text): raise Short
      return text[i]
    def word(w):  # a literal (true, false or null), as far as the text goes
      nonlocal i
      if not w.startswith(text[i:i + len(w)]): raise Bad
      if i + len(w) > len(text): raise Short
      i += len(w)
    def string(allowed=None):  # returns its value. Short if the text ends inside it, Bad once it can't be any of allowed, as JSON
      nonlocal i
      start, i = i, i + 1
      while i < len(text) and text[i] != '"':
        if ord(text[i]) < 32: raise Bad
        if text[i] == "\\":
          escape = text[i + 1:i + 2]
          if escape and escape not in '"\\/bfnrtu' or escape == "u" and not re.fullmatch("[0-9a-fA-F]*", text[i + 2:i + 6]): raise Bad
          i += 6 if escape == "u" else 2
        else: i += 1
      if allowed is not None and not any(json.dumps(a, ensure_ascii=False).startswith(text[start:i + 1]) for a in allowed): raise Bad
      if i >= len(text): raise Short
      i += 1
      return json.loads(text[start:i])
    def value(sc):
      nonlocal i
      sc = sc or {}
      while "$ref" in sc: sc = root["$defs" if "$defs" in sc["$ref"] else "definitions"][sc["$ref"].rsplit("/", 1)[1]]
      if "anyOf" in sc or "oneOf" in sc:  # the first alternative that parses, else Short if any could still parse
        start, short = i, False
        for alt in sc.get("anyOf") or sc["oneOf"]:
          try: return value({k: v for k, v in sc.items() if k not in ("anyOf", "oneOf")} | alt)
          except Short: short = True
          except Bad: pass
          i = start
        raise Short if short else Bad
      c, kinds = peek(), sc.get("type") or ["object", "array", "string", "number", "boolean", "null"]
      kinds = [kinds] if isinstance(kinds, str) else kinds
      if "enum" in sc or "const" in sc:  # one of them, as json.dumps writes it
        texts = [json.dumps(v, ensure_ascii=False) for v in sc.get("enum", [sc.get("const")])]
        if done := [t for t in texts if text.startswith(t, i)]: i += len(max(done, key=len)); return
        raise Short if any(t.startswith(text[i:]) for t in texts) else Bad
      if c == '"' and "string" in kinds: string()
      elif c in "-0123456789" and {"number", "integer"} & set(kinds):
        start = i
        while i < len(text) and text[i] in "-+0123456789.eE": i += 1
        number = text[start:i]
        if len(number) > 32: raise Bad
        # If it isn't a complete number, it may be the start of one that more digits complete
        if not re.fullmatch(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][-+]?[0-9]+)?", number):
          begun = re.fullmatch(r"-?((0|[1-9][0-9]*)(\.[0-9]*|(\.[0-9]+)?[eE][-+]?[0-9]*)?)?", number)
          if i == len(text) and begun and len(number) < 32 and ("number" in kinds or not re.search("[.eE]", number)): raise Short
          raise Bad
        if "number" not in kinds and re.search("[.eE]", number): raise Bad
      elif c in "tfn":
        if ("null" if c == "n" else "boolean") not in kinds: raise Bad
        word({"t": "true", "f": "false", "n": "null"}[c])
      elif c == "{" and "object" in kinds:
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
          value(props.get(key, extra if isinstance(extra, dict) else {}))
          seen.add(key)
      elif c == "[" and "array" in kinds:
        i, n = i + 1, 0
        while True:
          if peek() == "]" and n >= sc.get("minItems", 0): i += 1; return
          if n and text[i] != ",": raise Bad
          if n: i += 1
          if n >= sc.get("maxItems", math.inf): raise Bad
          value(sc.get("items", {}))
          n += 1
      else: raise Bad
    try:
      value(root)
      ws()
      return 2 if i == len(text) else 0
    except Short: return 1
    except Bad: return 0
  return fits

def answer(text):
  """Splits the text of a thinking model into its reasoning and its answer, which is empty until it starts. Qwen3
  splits at </think>. gpt-oss reasons in its analysis channel until <|end|>, and answers in its final channel."""
  if text.startswith("<|channel|>"):
    if "<|message|>" not in text: return "", ""  # the name of the channel is still coming
    head, _, content = text.partition("<|channel|>final<|message|>")
    return head.removeprefix("<|channel|>analysis<|message|>").partition("<|end|>")[0].strip(), content
  reasoning, _, content = text.partition("</think>")
  return reasoning.removeprefix("<think>").strip(), content.lstrip()

def server(engine, name, template="", eos=(), port=8000, wait=600, defaults=None):
  """An HTTP server for the engine, whose thread must be running. It serves /v1/models, /v1/completions and
  /v1/chat/completions, in one response or streamed as server-sent events.

  A model that thinks (its template has <think>, or gpt-oss channels) gives its reasoning apart from its answer, in
  reasoning_content. Qwen3 doesn't think when chat_template_kwargs has enable_thinking false. gpt-oss reasons as hard
  as reasoning_effort says: low, medium or high. Given tools (Qwen3.5's template), the tool calls in the answer come
  as OpenAI's tool_calls, after the text in a stream.
  usage counts the cached prompt tokens (prompt_tokens_details), which is how OpenRouter bills them. A stream ends
  with usage in a chunk of its own, which OpenRouter does too. If a piece of text isn't ready in `wait` seconds, the
  request ends: a hung chip is reported, not waited on. A sampling parameter that a request leaves out comes from
  defaults: temperature, top_k, top_p, min_p and repetition_penalty from the generation_config.json of the model.
  vLLM does the same. Failing that, it's OpenAI's default."""
  defaults, thinks, harmony = defaults or {}, "<think>" in template, family(template) == "harmony"
  class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body):
      data = json.dumps(body).encode()
      self.send_response(code)
      self.send_header("Content-Type", "application/json"), self.send_header("Content-Length", str(len(data)))
      self.end_headers()
      self.wfile.write(data)
    def do_GET(self):
      if self.path != "/v1/models": return self.reply(404, {"error": {"message": f"not found: {self.path}"}})
      self.reply(200, {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "baretile"}]})
    def do_POST(self):
      is_chat = self.path == "/v1/chat/completions"
      if not is_chat and self.path != "/v1/completions": return self.reply(404, {"error": {"message": f"not found: {self.path}"}})
      try:
        q = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        thinking, tools = (q.get("chat_template_kwargs") or {}).get("enable_thinking", True) is not False, is_chat and q.get("tools")
        prompt = chat(template, q["messages"], thinking, q.get("reasoning_effort") or "medium", tools) if is_chat else q["prompt"]
        ids = engine.tok.encode(prompt, special=is_chat)
        top = (q.get("top_logprobs") or 0 if q.get("logprobs") else None) if is_chat else q.get("logprobs")  # a completion's is a count
        if top is not None and not 0 <= top <= 20: raise ValueError("logprobs and top_logprobs must be between 0 and 20")
        apart = is_chat and (thinking and thinks or harmony)
        choice, form = q.get("tool_choice"), formatted(q.get("response_format"))
        fits = chosen(choice, tools)
        if form and choice in (None, "auto", "none"):  # the format, or a tool call where tools allow one (auto)
          call = chosen("required", tools) if tools and choice != "none" else None
          fits = form if call is None else (lambda text: max(form(text), call(text)))
        shape = fits and apart and (lambda text: fits(answer(text)[1])) or fits  # a thinking model's shape applies to its answer, after the reasoning
        given = lambda k, default: q[k] if q.get(k) is not None else defaults.get(k, default)
        r = Request(engine.tok, ids, max_tokens=q.get("max_tokens") or (engine.places if is_chat else 16), temperature=given("temperature", 1.0),
                    top_p=given("top_p", 1.0), stop=q.get("stop"), eos=eos, seed=q.get("seed"), top_k=max(given("top_k", 0), 0),
                    min_p=given("min_p", 0.0), presence=q.get("presence_penalty") or 0.0, frequency=q.get("frequency_penalty") or 0.0,
                    repetition=given("repetition_penalty", 1.0), logprobs=top, shape=shape)
      except (ValueError, KeyError, TypeError) as e: return self.reply(400, {"error": {"message": f"{type(e).__name__}: {e}"}})
      if len(ids) >= engine.places - 1: return self.reply(400, {"error": {"message": f"{len(ids)} tokens, but a slot holds {engine.places - 2}"}})
      # OpenRouter wants a 429 when no slot is free, so it can try the next provider for the request. It doesn't count
      # the 429 against our uptime.
      if not engine.offer(r): return self.reply(429, {"error": {"message": "every slot is busy", "code": 429}})
      head = {"id": f"baretile-{id(r)}", "object": "chat.completion" if is_chat else "text_completion", "created": int(time.time()), "model": name}
      seen = {"text": "", "reasoning_content": "", "content": ""}
      def parts(text):  # splits a thinking model's text into reasoning and answer, and cuts the answer at its tool calls
        out = dict(zip(("reasoning_content", "content"), answer(text))) if apart else {"content": text}
        if tools:  # also hold back what may be the start of a tool call
          shown = out["content"].partition("<tool_call>")[0]
          out["content"] = shown[:len(shown) - max((k for k in range(1, 11) if shown.endswith("<tool_call>"[:k])), default=0)]
        return out
      def delta(text):  # the next piece of a chat: what's new in each part
        seen["text"] += text
        new = parts(seen["text"])
        out = {k: v[len(seen[k]):] for k, v in new.items() if len(v) > len(seen[k])}
        seen.update(new)
        return out
      called = lambda text: calls(answer(text)[1] if apart else text, tools) if tools else (None, [])  # the text and tool calls of the answer
      piece = lambda text, done: ({"index": 0, "delta": delta(text)} if is_chat else {"index": 0, "text": text}) | {"finish_reason": done}
      word, raw = lambda t: engine.tok.decode([t]), lambda t: list(engine.tok.bytes[t])
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
      if q.get("stream"):
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        send = lambda body: (self.wfile.write(f"data: {json.dumps(body)}\n\n".encode()), self.wfile.flush())
        head["object"] += ".chunk" if is_chat else ""
        try:
          if is_chat: send(head | {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
          for k, text in enumerate(iter(lambda: r.out.get(timeout=wait), None)):  # each piece with the log probabilities of its tokens
            send(head | {"choices": [piece(text, None) | logprobs(k)]})
          if made := called(seen["text"])[1]:
            calling = {"tool_calls": [{"index": i} | c for i, c in enumerate(made)]}
            send(head | {"choices": [{"index": 0, "delta": calling, "finish_reason": None}]})
          send(head | {"choices": [piece("", "tool_calls" if made else r.finish)]})
          send(head | {"choices": [], "usage": usage()})
          return self.wfile.write(b"data: [DONE]\n\n")
        except (OSError, queue.Empty): r.gone = True  # the client left, or no text came, so let its slot go
        return
      try: text = "".join(iter(lambda: r.out.get(timeout=wait), None))
      except queue.Empty:
        r.gone = True  # nobody is waiting for it now, so let its slot go
        return self.reply(504, {"error": {"message": f"no text came in {wait}s: the chip may be hung"}})
      if r.finish == "error": return self.reply(500, {"error": {"message": "the engine stopped on an error"}})
      message, (content, made) = parts(text), called(text)
      if made: message |= {"content": content or None, "tool_calls": made}
      choice = ({"index": 0, "message": {"role": "assistant"} | message} if is_chat else {"index": 0, "text": text}) | logprobs()
      self.reply(200, head | {"choices": [choice | {"finish_reason": "tool_calls" if made else r.finish}], "usage": usage()})
  return ThreadingHTTPServer(("", port), Handler)

def fetch(model, cache=os.path.expanduser("~/.cache/baretile/hub")):
  """The folder of a model. If model is a folder, it's returned as is. Otherwise it's a Hugging Face model id, and
  we download its files into cache: the config and tokenizer, the generation and chat configs it has (and
  chat_template.jinja), and the weights, in one file or in the shards that its index names. HF_ENDPOINT sets the
  server, and HF_TOKEN is for a gated model. A finished download is marked with .done and isn't fetched again."""
  if os.path.isdir(model): return model
  folder = os.path.join(cache, model)
  if os.path.exists(os.path.join(folder, ".done")): return folder
  os.makedirs(folder, exist_ok=True)
  def get(name, needed=True):
    path = os.path.join(folder, name)
    if os.path.exists(path): return path
    token = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if "HF_TOKEN" in os.environ else {}
    url = f"{os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}/{model}/resolve/main/{name}"
    try:
      with urllib.request.urlopen(urllib.request.Request(url, headers=token), timeout=60) as r, open(path + ".part", "wb") as f:
        shutil.copyfileobj(r, f)
    except urllib.error.HTTPError as e:
      if needed or e.code != 404: raise
      return None
    os.replace(path + ".part", path)
    return path
  for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"):
    get(name, name in ("config.json", "tokenizer.json"))
  if index := get("model.safetensors.index.json", False):
    for shard in sorted(set(llama.read_json(index)["weight_map"].values())): get(shard)
  else: get("model.safetensors")
  open(os.path.join(folder, ".done"), "w").close()
  return folder

def main():
  from baretile import pci
  from baretile.chip import Chip
  from baretile.runtime import Runtime
  folder, args = fetch(sys.argv[1]), sys.argv[2:] + [None] * 5
  port, slots, places, mode, kv = int(args[0] or 8000), int(args[1] or 4), int(args[2] or 512), args[3] or "bf16", args[4]
  chip = Chip(pci.open_device())
  rt, start = Runtime(chip), time.time()
  c, weights = llama.load(folder, places)
  c.precision = llama.precision(c, mode, kv)
  engine = Engine(rt, llama.upload(rt, weights, c.precision), c, llama.Tokenizer(folder), slots, places)
  config = lambda f: llama.read_json(os.path.join(folder, f)) if os.path.exists(os.path.join(folder, f)) else {}
  generation, name = config("generation_config.json"), os.path.basename(folder.rstrip("/"))
  eos = (generation or config("config.json")).get("eos_token_id")
  eos = eos if isinstance(eos, list) else [] if eos is None else [eos]
  defaults = {k: v for k, v in generation.items() if k in ("temperature", "top_k", "top_p", "min_p", "repetition_penalty")}
  threading.Thread(target=engine.run, daemon=True).start()
  print(f"{name} is on the chip after {time.time() - start:.0f}s ({mode}, a {llama.cache_format(c)} KV cache), {slots} slots of {places} positions, "
        f"prefilling {engine.chunk} tokens at a time, on port {port}", flush=True)
  jinja = os.path.join(folder, "chat_template.jinja")  # where transformers 5 keeps it
  template = config("tokenizer_config.json").get("chat_template") or (open(jinja, encoding="utf-8").read() if os.path.exists(jinja) else "")
  server(engine, name, template, eos, port, defaults=defaults).serve_forever()

if __name__ == "__main__": main()
