# What a model costs to serve on Blackhole cards, worked out from its config.json without the weights (llama.config). It
# counts what a card holds: the weights in the mode's formats (llama.precision) and the sessions' KV cache rows and Gated
# DeltaNet states. It counts what a decode step reads, computes and launches: the weights, the experts its tokens pick,
# and the sessions' rows and states. From those come how many sessions fit, tokens/s per user and per card, and dollars
# per million tokens. The card's rates come from its spec until you measure them with the probe
# (python3 -m baretile.probe 8 probe.json, then probe=probe.json here). The launches per block are baretile's, counted
# on ttsim. It models what baretile would do, and only a card can say how far off it is.
#   python3 models/cost.py path/to/model [bf16|bfp8|bfp4|tt|experts] [key=value ...]   # keys from CARD and USE
import json, math, os, sys
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baretile import bh
from baretile.kernels import attention_parts, attention_store, grid_parts
from baretile.tensix import CHUNKS, tile_bytes
import llama

BYTES = {d: tile_bytes(d) / 1024 for d in ("bf16", "bfp8", "bfp4")}  # bytes per value, with its share of the tile's exponents
# Launches in a decode step, counted on ttsim at Qwen3.6's shapes with eight slots:
#   attention  the norms, projections, RoPE, the cache put, attention in parts and their sum, the output and its gate
#   experts    the extra launches of a mixture of experts
#   deltanet   the projections, the delta kernel, and the output's norm and gate
#   rest       the embedding gather, the position columns, the final norm, the output layer and sampling's heads
# The verify entries are the same for a speculative step, and draft is the call to the multi-token prediction layer.
LAUNCHES = {"attention": 20, "experts": 15, "deltanet": 5, "rest": 12, "attention verify": 20, "deltanet verify": 5, "rest verify": 21, "draft": 51}
# Bytes a decode step reads beyond the weights, experts, states and attention's rows, at Qwen3.6's shapes. The first is
# a block's activations in 32-row tiles. The second is the rest's, the logits that sampling reads, for each tile row of
# tokens. outputs, in step, counts the outputs of the picked experts separately: the gate and up, their GLU and the
# down, for the tile rows of their tokens.
ACTIVATIONS = 4e6, 45e6
# RISC-V work that a card waits on beyond DRAM, as (tokens in the step, cycles). It's the ttsim cycles that the launches
# of a block, its experts included, or of the rest take beyond their reads at 265 bytes a cycle (358 GB/s at 1.35 GHz).
# Counted at Qwen3.6's shapes with ttsim-chaos's read counter and no injected latency. past interpolates linearly
# between the counts.
CONTROL = {"attention": ((1, 39300), (8, 25400), (32, 41600)), "deltanet": ((1, 36200), (8, 12900), (32, 15100)),
           "rest": ((1, 154200), (8, 731000), (32, 1641800)), "attention verify": ((2, 47300), (16, 38400), (64, 65600)),
           "deltanet verify": ((2, 63600), (16, 27200), (64, 22900)), "rest verify": ((2, 412000), (16, 1479000), (64, 5148200)),
           "draft": ((2, 50400), (16, 41600), (64, 65700))}
# A p150 has 120 Tensix cores, and one of them is the dispatcher. GRID is the worker grid as Runtime.grid has it.
WORKERS, GRID = 119, (tuple(range(12)), tuple(range(9)))
# A p150. The cost is the p150a's launch price plus a share of a host, over three years, and the power of the card and
# its share of the host at $0.15 a kWh.
CARD = dict(dram=len(bh.DRAM) * 4 * 2**30, reserve=2**30,  # 8 GDDR6 banks of 4 GiB, with a GiB kept for kernels and activations
            bandwidth=bh.DRAM_GBPS * 1e9, reach=0.7,  # 512 GB/s, and the share of it that a decode step gets (a guess)
            tflops=664e12, phases=2,  # bfp8 matmuls at one fidelity phase (16 cycles a step), and the phases they take
            launch=1e-6, pcie=25e9,  # a launch's seconds on the card (a guess), and PCIe 4.0 x16 in bytes/s
            price=1299, host=600, years=3, watts=300, host_watts=75, kwh=0.15)
# context is the number of positions in a session's slot, and they're all filled. A step reads the rows of a slot up to
# the power of 2 that holds the longest session. kv is the KV cache's format, and report uses the mode's unless you
# set it. accept is the share of drafts that get accepted.
USE = dict(context=32768, kv="bf16", accept=0.0)

def probed(card, path):  # the card with the probe's numbers in place of the spec's (python3 -m baretile.probe 8 path)
  p = json.load(open(path))
  chunks = [p[f"DRAM reads of {f} chunks, {CHUNKS} in flight: grid workers, own banks"] for f in ("bfp8", "bfp4")]  # how decode reads weights
  return card | dict(bandwidth=min(chunks) * 1e9, reach=1.0,
                     tflops=2 * 32**3 / p["matmul step, 1 phase, unpacks included"] * WORKERS * bh.AICLK,
                     launch=p["launch, 100 queued: chip each"] / bh.AICLK, pcie=min(p["upload over PCIe"], p["read back over PCIe"]) * 1e9)

def shape(c, mode="bf16"):
  """The sizes of a model's matrices in mode's formats (llama.precision), as a decode step sees them:
    every         bytes that steps always read, which is everything but the experts and the embedding
    expert        bytes of an expert
    moe           how many blocks have experts
    draft         bytes of the multi-token prediction layer, without its experts
    draft_values  the values that the draft's call multiplies by
    embed         bytes of the embedding, which is held, but a step reads only its tokens' rows
    head          bytes of the output layer that a draft reads
    bytes         everything held
    values        the values that a token's matmuls take
    held          the parameters"""
  prec, d, hd, e, m, sh = llama.precision(c, mode), c.dim, llama.head(c), c.experts, c.moe, c.shared
  fmt = lambda op, l=None: BYTES[prec.get((op, l), ("bf16",))[0]]  # the routers, the conv's taps and mtp.fc stay bf16, as llama holds them
  mixture = lambda l: e and l not in c.dense  # the multi-token prediction block, c.layers, has experts if the model does
  mixer = d * c.heads * hd * (2 if c.gated else 1) + 2 * d * c.kv * hd + c.heads * hd * d  # an attention block's values: q and its gate, k, v, o
  delta = d * (2 * c.kh * c.kd + 2 * c.vh * c.vd + 2 * c.vh) + c.vh * c.vd * d if c.kh else 0  # a Gated DeltaNet block's: projections and o
  mixers = lambda l: delta if llama.linear(c, l) else mixer
  mlp = lambda l, width: d * width * (2 * fmt("gate_up", l) + fmt("down", l))  # a SwiGLU's bytes
  ffn = lambda l: mlp(l, sh) + d * e * BYTES["bf16"] if mixture(l) else mlp(l, c.ffn)  # a block's MLP without experts: a shared expert and a router
  ffn_values = lambda l: 3 * d * (sh + c.k * m) + d * e if mixture(l) else 3 * d * c.ffn  # a token's experts included
  moe, t = [l for l in range(c.layers) if mixture(l)], c.layers
  every = sum(mixers(l) * fmt("qkv", l) + ffn(l) for l in range(c.layers)) + c.vocab * d * fmt("head")
  expert = mlp(moe[0], m) if moe else 0
  draft = c.mtp * (mixer * fmt("qkv", t) + ffn(t) + 2 * d * d * BYTES["bf16"])
  own = c.mtp * c.vocab * d * fmt("draft head") if ("draft head", None) in prec else 0  # the draft's own output layer
  values = sum(mixers(l) + ffn_values(l) for l in range(c.layers)) + c.vocab * d
  draft_values = c.mtp * (mixer + ffn_values(t) + 2 * d * d)
  held = len(moe) + (c.mtp if e else 0)  # the blocks with experts, counting the multi-token prediction layer
  return SimpleNamespace(every=every, expert=expert, moe=len(moe), draft=draft, draft_values=draft_values, embed=c.vocab * d * BYTES["bf16"],
                         head=own or c.vocab * d * fmt("head"), bytes=every + held * e * expert + draft + own + c.vocab * d * BYTES["bf16"],
                         values=values, held=values + draft_values + held * (e - c.k) * 3 * d * m + c.vocab * d * (0 if c.tied else 1))

def session(c, use, speculative=False):
  """The bytes a session holds. They're KV cache rows for each attention block, up to a sliding window's, and for the
  multi-token prediction layer, plus the Gated DeltaNet states. Speculating keeps two copies of the states, from before
  and after a draft."""
  row = 2 * c.kv * llama.head(c) * BYTES[use["kv"]]
  window = lambda l: c.sliding if c.sliding and llama.windowed(c, l) else use["context"]
  kv = sum(row * min(use["context"], window(l)) for l in range(c.layers) if not llama.linear(c, l))
  draft = c.mtp * row * use["context"]
  linear = sum(llama.linear(c, l) for l in range(c.layers))
  state = (1 + speculative) * linear * 4 * (c.vh * c.kd * c.vd + (c.conv - 1) * (2 * c.kh * c.kd + c.vh * c.vd)) if linear else 0
  return SimpleNamespace(kv=kv, draft=draft, state=state, bytes=kv + draft + state)

# the expected number of a block's experts that at least one of the tokens picks
def touched(c, tokens): return c.experts * (1 - (1 - c.k / c.experts) ** tokens) if c.experts else 0

def regroups(c, m):
  """The bytes of activations that a prompt chunk of m tokens reads again. When the accumulators for a grid matmul's
  result tiles don't all fit in L1, workers take their columns a group at a time and read A once per group.
  kernels.grid_parts plans this for each matmul separately. This counts the blocks' projections and MLPs. A mixture's
  experts aren't counted, since they run by expert and not on the grid."""
  hd, d, rt = llama.head(c), c.dim, -(-m // 32)
  def again(k, n):  # the bytes of A that a matmul with K k and N n reads more than once
    plan, w = grid_parts(rt, -(-n // 32), -(-k // 32), GRID, 16, 2) if rt > 1 else None, -(-n // 32)
    return (-(-(-(-w // min(len(GRID[0]), w))) // plan[1]) - 1) * m * k * BYTES["bf16"] if plan else 0
  def matrices(l):  # (K, N) for each matrix of block l
    mixer = ((d, 2 * c.kh * c.kd + 2 * c.vh * c.vd + 2 * c.vh), (c.vh * c.vd, d)) if llama.linear(c, l) else \
            ((d, c.heads * hd * (2 if c.gated else 1) + 2 * c.kv * hd), (c.heads * hd, d))
    width = c.shared if c.experts and l not in c.dense else c.ffn  # a mixture's shared expert, if it has one
    return mixer + (((d, 2 * width), (width, d)) if width else ())
  return sum(again(k, n) for l in range(c.layers) for k, n in matrices(l))

def attention(c, entries, queries, rows, kv):
  """The bytes an attention block reads in Tensor.attend. A unit is up to eight tile rows of 32 queries, for each of
  `entries` heads, or key heads with a token's query heads side by side. Units read the `rows` keys and values that
  their positions reach, the values once and the keys once. Keys past what a part keeps in L1 between its sweeps
  (attention_store) are read twice. Parts share keys the way attend shares them."""
  ht, kt, wt = llama.head(c) // 32, -(-rows // 32), kv if kv in ("bfp8", "bfp4") else None
  units = entries * -(-queries // 256)
  parts = attention_parts(ht, queries, wt, kt, units, WORKERS)
  return units * (2 * kt + max(0, kt - parts * attention_store(ht, queries, wt, parts)[1])) * 32 * llama.head(c) * BYTES[kv]

def past(kind, n): return float(np.interp(n, *zip(*CONTROL[kind])))  # CONTROL's cycles for kind at n tokens, flat past its counts

def step(c, s, use, card, users, speculative=False, cards=1):
  """A decode step that serves `users` sessions. A session gets a token, or when speculative a token and a draft, and
  then a second call makes the next drafts. Returns the bytes read (and the states written), flops, launches, control,
  seconds and tokens/s per user. The step waits on DRAM or the matmuls, whichever is slower, then on its launches,
  then on the RISC-V work its kernels do beyond their reads, if the card's matches ttsim's count."""
  n, one, d, v = users * (1 + speculative), session(c, use, speculative), c.dim, " verify" * speculative
  blocks = sum(not llama.linear(c, l) for l in range(c.layers))
  outputs = lambda tokens: touched(c, tokens) * -(-tokens // 32) * 32 * (3 * c.moe + d) * BYTES["bf16"]  # for a block
  # the states are read once and written once. A verify step reads one of its two sets and writes both
  reads = s.every + s.moe * touched(c, n) * s.expert + users * one.state * (1.5 if speculative else 2)
  reads += blocks * attention(c, c.kv * users, c.heads // c.kv * (1 + speculative), use["context"], use["kv"])
  reads += c.layers * ACTIVATIONS[0] + s.moe * outputs(n) + ACTIVATIONS[1] * -(-n // 32)
  flops = 2 * n * s.values + n * blocks * 4 * c.heads * llama.head(c) * use["context"]
  linear = c.layers - blocks
  launches = blocks * LAUNCHES["attention" + v] + linear * LAUNCHES["deltanet" + v] + s.moe * LAUNCHES["experts"] + LAUNCHES["rest" + v]
  control = blocks * past("attention" + v, n) + linear * past("deltanet" + v, n) + past("rest" + v, n)
  if speculative:  # the drafting call runs the layer over two positions per user, and the output layer again
    reads += s.draft + c.mtp * (touched(c, 2 * users) * s.expert + ACTIVATIONS[0] + outputs(2 * users)) + s.head
    reads += c.mtp * attention(c, c.kv * users, 2 * c.heads // c.kv, use["context"], use["kv"])
    flops += 2 * 2 * users * (s.draft_values + c.vocab * d)
    launches += LAUNCHES["draft"]
    control += c.mtp * past("draft", n)
  control /= bh.AICLK
  dram, compute = reads / (card["bandwidth"] * card["reach"] * cards), flops / (card["tflops"] / card["phases"] * cards)
  seconds = max(dram, compute) + launches * card["launch"] + control
  rate = (1 + use["accept"] if speculative else 1) / seconds
  return SimpleNamespace(reads=reads, seconds=seconds, launches=launches, control=control, flops=flops, rate=rate)

def prompt(c, s, use, card, chunk, cards=1):
  """Prefills a prompt that fills a session (use's context) in chunks of `chunk` tokens, and returns its seconds and
  tokens/s. serve.py's engine prefills the same way.

  A chunk reads the matrices other than the experts and the embedding, plus the experts its tokens pick. It reads the
  session's rows up to its last position, a key head's rows per eight tile rows of a query head (`attention`), and
  its activations again where a matmul goes a group of columns at a time (`regroups`). It computes its tokens' matmuls,
  with the output layer for the last token only, and their scores over those rows. A multi-token prediction layer runs
  over the prompt's positions too. A chunk takes as long as DRAM or the matmuls, whichever is slower. Launches and
  control are left out."""
  hd, d, n = llama.head(c), c.dim, use["context"]
  blocks = sum(not llama.linear(c, l) for l in range(c.layers)) + c.mtp
  seconds = 0.0
  for at in range(0, n, chunk):
    m = min(chunk, n - at)
    rows = at + m
    reads = s.every + s.moe * touched(c, m) * s.expert + blocks * attention(c, c.heads, m, rows, use["kv"]) + regroups(c, m)
    flops = 2 * m * (s.values - c.vocab * d) + 2 * c.vocab * d + blocks * 4 * m * c.heads * hd * rows
    if c.mtp:
      reads += s.draft + c.mtp * touched(c, m) * s.expert + s.head
      flops += 2 * m * (s.draft_values + c.vocab * d)
    seconds += max(reads / (card["bandwidth"] * card["reach"] * cards), flops / (card["tflops"] / card["phases"] * cards))
  return SimpleNamespace(seconds=seconds, rate=n / seconds)

def hourly(card): return (card["price"] + card["host"]) / (card["years"] * 8766) + (card["watts"] + card["host_watts"]) / 1000 * card["kwh"]

def report(folder, mode="bf16", **keys):
  """The cost report for the model in folder, in mode's formats. It reads only config.json, and keys override CARD and
  USE."""
  card, use = dict(CARD), dict(USE)
  for k, v in keys.items():
    if k != "probe": (card if k in card else use)[k] = v if k == "kv" else float(v)
  if "probe" in keys: card = probed(card, keys["probe"])
  c, name = llama.config(folder), os.path.basename(os.path.normpath(folder))
  use["context"], use["kv"] = int(use["context"]), keys.get("kv") or llama.precision(c, mode)["kv", None][0]
  s = shape(c, mode)
  cards = math.ceil(s.bytes / (card["dram"] - card["reserve"]))
  room, hour = cards * (card["dram"] - card["reserve"]) - s.bytes, hourly(card) * cards
  out = [f"{name}: {s.held / 1e9:.1f}B parameters, {s.values / 1e9:.2f}B active per token. In {mode} mode it takes "
         f"{s.bytes / 1e9:.1f} GB on {cards} card{'s' * (cards > 1)}, {room / 1e9:.1f} GB free, ${hour:.3f} an hour"]
  for speculative in (False, True) if c.mtp else (False,):
    one = session(c, use, speculative)
    most = max(int(room // one.bytes), 1)
    states = f" and {one.state / 1e6:.0f} MB of Gated DeltaNet states" if one.state else ""
    out.append(f"  a session of {use['context']} tokens holds {one.kv / 1e9:.2f} GB of KV cache ({one.kv / use['context'] / 1e3:.1f} KB per token, "
               f"{use['kv']}){states}. {most} sessions fit on the card{'s' * (cards > 1)}")
    taken = f", {use['accept']:.0%} of drafts accepted" if speculative else ""
    out.append(f"  decode, {'a token and a draft' if speculative else 'one token'} per user per step{taken}:")
    out.append("     users     GB read   step, ms   launches    RISC-V, ms    tok/s per user    total    $/M tokens out"
               + ("  break-even acceptance" if speculative else ""))
    for users in [u for u in (1, 2, 4, 8, 16, 32, 64, 128) if u <= most]:
      t = step(c, s, use, card, users, speculative, cards)
      pays = f"{t.seconds / step(c, s, use, card, users, False, cards).seconds - 1:23.0%}" if speculative else ""
      out.append(f"  {users:8d} {t.reads / 1e9:11.2f} {t.seconds * 1e3:10.1f} {t.launches:10d} {t.control * 1e3:13.1f} {t.rate:17.1f} "
                 f"{users * t.rate:8.0f} {hour / (users * t.rate * 3600) * 1e6:17.3f}{pays}")
  out.append(f"  a prompt of {use['context']} tokens, prefilled in chunks:")
  out.append("     chunk   seconds             tok/s    $/M tokens in")
  for chunk in (256, 1024, 2048, 4096):
    p = prompt(c, s, use, card, chunk, cards)
    out.append(f"  {chunk:8d} {p.seconds:9.2f} {p.rate:17,.0f} {hour / (p.rate * 3600) * 1e6:16.4f}")
  kept = session(c, use).bytes  # a copy kept in DRAM is read and written
  out.append(f"  resuming a session: {2 * kept / (card['bandwidth'] * card['reach']) * 1e3:.1f} ms from a copy on the chip, "
             f"{kept / card['pcie'] * 1e3:.0f} ms from the host over PCIe at {card['pcie'] / 1e9:.0f} GB/s")
  return "\n".join(out)

if __name__ == "__main__":
  args = [a for a in sys.argv[2:] if "=" not in a]
  print(report(sys.argv[1], args[0] if args else "bf16", **dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)))
