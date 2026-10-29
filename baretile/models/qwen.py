# Extensions to the Llama model in llama.py: mixtures of experts, Gated DeltaNet and multi-token
# prediction. The model operations run with baretile tensors or NumPy arrays. Kernels used only by
# these models live here: delta updates recurrent state, group sorts a long prompt's tokens by expert,
# and touched lists the experts that a decode step's or a short prompt's tokens selected. gpt-oss
# weights are converted from MXFP4.
import numpy as np
from baretile import bh
from baretile.runtime import DATA, ZEROS, compile_kernel
from baretile.tensix import (ACROSS, BANKS, DST_FREE, DST_HALF, FILLED, FOLDS, MATH_END, MINUS_ONE, ONE, OUTS, REPORT, ROWS, SEM_L1, SEM_PACKED,
                            SFPU_START, SYNC_WORDS, UNPACKABLE, ZERO, Loop, aim, fill, ins, kernel_config, mad, mask_columns, mask_rows, math_program,
                            mul, pack, pack_program, sfp_const, sfp_load, sfp_store, sfpu, slot_base, tile_of, tt, unpack_program)
from baretile.tensor import Tensor
from baretile.models import gpt2, llama

def deltanet(h, p, c, state, phases, live=None, accept=None):
  """Apply Qwen3.5/3.6's Gated DeltaNet to this call's tokens and update recurrent state in place.
  Tokens belong to one sequence, or form equal-length groups for several sequences. live marks active
  tokens; inactive tokens leave state unchanged. After the projections, computation uses fp32.
  Decode and verification run the convolution, gates and recurrence in one kernel. Prompts use
  64-token chunks.

  state contains [conv, S]:
  - conv holds the previous c.conv - 1 convolution inputs, with one matrix per history position and
    one row per sequence.
  - S holds a c.kd x c.vd matrix per value head, with c.vh batch entries per sequence. Each token
    updates it as S = e^g S + beta k (v - e^g S^T k)^T, then reads the result with its query.

  Speculative verification processes a token and its draft. state[2:] keeps copies after the first
  token. On the next call, accept=1 resumes from the state after both tokens; accept=0 resumes from
  the copies before the rejected draft.

  p contains qkv_proj (q, k, v, z, b and a concatenated), conv taps, decay (-exp(A_log)), dt_bias,
  norm and o_proj."""
  f, (past, S) = llama.fns(h), state[:2]
  past = past.get() if isinstance(past, Row) else past  # the conv state of one sequence out of several
  n, seqs, qk, vw, r = h.shape[-2], past.shape[1], c.kh * c.kd, c.vh * c.vd, c.vh // c.kh
  m, width, taps = n // seqs, 2 * qk + vw, p["conv"]  # tokens per sequence, channels of the conv, and its taps (a batch entry each)
  x = f.mm(h, p["qkv_proj"], phases("qkv"), "fp32")
  alive = lambda: live.reshape(1, seqs, m)[:, :, :1]  # the first token of each sequence, with sequences as rows
  keep = lambda old, new: new if live is None else old + alive() * (new - old)
  big = lambda old, new: new if live is None else old + f.repeat(alive().reshape(seqs, 1, 1), c.vh) * (new - old)
  if (m == 1 and accept is None or m == 2 and accept is not None) and not isinstance(state[0], Row):
    # A decode step's token, or a verify step's token and then its draft, in one kernel with a unit for each key head
    # of each sequence (f.delta).
    # The kernel reads the rows of x as they are and runs the conv, the gates and the recurrence. It writes the conv
    # inputs and the states in place. In a verify step, accept says whether to go on from the state after the last
    # draft or after the token before it. The state after the first token goes into the copies in state[2:].
    o = f.delta(S, x, past, taps, p["dt_bias"], p["decay"], live, *((state[3], accept, state[2]) if m == 2 else ()))
  else:  # a prompt, maybe in one slot of several. chunk batches its pieces of 64, then runs the state through them in turn
    # The conv inputs of each sequence are its last ones, then this call's. Realize them before the last ones are
    # written over.
    rows = f.now(f.cat(f.swap(past), x[:, :width].reshape(seqs, m, width)))
    y = sum(rows[:, j:j + m, :] * taps[j:j + 1, :, :] for j in range(c.conv))
    f.assign(state[0], keep(state[0], f.swap(rows[:, m:, :])))
    def each(t, d):  # a batch entry for each head of each sequence
      heads = f.split(t, t.shape[-1] // d) if d > 1 else f.heads(t)
      return heads if seqs == 1 else f.swap(heads.reshape(heads.shape[0], seqs, m * d)).reshape(seqs * heads.shape[0], m, d)
    y = f.silu(y).reshape(n, width)
    unit = lambda t: t * f.rsqrt(f.mean(t * t) * c.kd + 1e-6)  # q and k of each head to length 1, along a row
    q, k = unit(each(y[:, :qk], c.kd)) * c.kd ** -0.5, unit(each(y[:, qk:2 * qk], c.kd))
    b, a = x[:, width + vw:width + vw + c.vh], x[:, width + vw + c.vh:] + p["dt_bias"]
    g = each(p["decay"] * (f.maximum(a, 0) + f.log(1 + f.exp(-f.maximum(a, -a)))), 1)  # softplus: the log decay of each head
    o, S2 = chunk(f.repeat(q, r), f.repeat(k, r), each(y[:, 2 * qk:], c.vd), f.sigmoid(each(b, 1)), g, S)
    o = f.now(o)  # o reads the state, so realize it before the state is written over
    f.assign(state[1], big(state[1], S2))
    o = o if seqs == 1 else f.swap(o.reshape(seqs, c.vh, m * c.vd)).reshape(c.vh, n, c.vd)
  o = o * f.rsqrt(f.mean(o * o) + c.eps) * p["norm"] * f.silu(f.split(x[:, width:width + vw], c.vh))  # each head RMS-normed and gated by z
  return f.mm(f.merge(f.bf16(o)), p["o_proj"], phases("o"))

class Row:
  """View row u across a batch of matrices, one row per sequence. llama.slot uses this for DeltaNet convolution
  history. Tensor views can't select this row directly, so reads use a move and writes use put."""
  def __init__(self, t, u): self.t, self.u, self.at, self.batch, self.rows, self.cols = t, u, Tensor.of(t.rt, [[u]], "int32"), t.batch, 1, t.cols
  def get(self): return self.t[:, self.u:self.u + 1, :]
  def assign(self, x): self.t.put(self.at, x.get() if isinstance(x, Row) else x); return self
  def tobytes(self): return self.get().tobytes()

def numpy_delta(S, x, conv, taps, dt, decay, live=None, first=None, accept=None, conv2=None):
  """Run delta with NumPy for llama.fns, using the same arguments, outputs and in-place state updates."""
  n, width = conv.shape[1], conv.shape[-1]
  (vh, kd, vd), m = (S.shape[0] // n, *S.shape[1:]), x.shape[-2] // n
  qk, vw, x = (width - vh * vd) // 2, vh * vd, np.asarray(x).reshape(n, m, -1)
  each = lambda a: np.ones(n, bool) if a is None else np.asarray(a).reshape(n, -1)[:, 0] != 0  # by sequence, from the first token
  alive, taken = each(live), each(accept) if m > 1 else np.ones(n, bool)
  chain = np.where(taken[None, :, None], conv, conv2) if m > 1 else conv  # the last inputs, oldest first
  state = np.where(np.repeat(taken, vh)[:, None, None], S, first) if m > 1 else S
  unit, out = lambda a: a / np.sqrt((a * a).sum(-1, keepdims=True) + 1e-6), []
  for t in range(m):
    xt = x[:, t]
    inputs = np.concatenate([chain, xt[None, :, :width]])
    y, chain = (inputs * taps).sum(0), inputs[1:]
    y = y / (1 + np.exp(-y))
    q, k = unit(y[:, :qk].reshape(n, -1, kd)) * kd ** -0.5, unit(y[:, qk:2 * qk].reshape(n, -1, kd))
    b, a = xt[:, 2 * qk + 2 * vw:2 * qk + 2 * vw + vh], xt[:, 2 * qk + 2 * vw + vh:] + np.reshape(dt, vh)
    beta, g = 1 / (1 + np.exp(-b)), np.reshape(decay, vh) * (np.maximum(a, 0) + np.log1p(np.exp(-np.abs(a))))
    column = lambda a: np.repeat(a, vh // a.shape[1], 1).reshape(n * vh, kd, 1)  # the key head of each value head, as a column
    row = lambda a: a.reshape(n * vh, 1, -1)  # each value head as a row
    o, state = step(column(q), column(k), row(y[:, 2 * qk:]), row(beta), row(g), state)
    (into, states), heads = (conv2, first) if t < m - 1 else (conv, S), np.repeat(alive, vh)
    into[:, alive], states[heads] = chain[:, alive], state[heads]
    out.append(o.reshape(n, vh, vd) * alive[:, None, None])
  return np.stack(out, 2).transpose(1, 0, 2, 3).reshape(vh, n * m, vd)

def step(q, k, v, beta, g, S):
  """Apply the gated delta rule to one token in NumPy, matching transformers' torch_recurrent_gated_delta_rule.
  q and k are columns. Return (output, updated state)."""
  e = np.exp(g)
  v = beta * (v - e * (k * S).sum(-2, keepdims=True))  # correct the decayed state's prediction toward v
  S = e * S + k * v
  return (q * S).sum(-2, keepdims=True), S

def chunk(q, k, v, beta, g, S, size=64):
  """Apply the gated delta rule to a prompt in 64-token chunks, following transformers'
  torch_chunk_gated_delta_rule. Take initial state S, queries q, keys k, values v, and columns beta
  and g. Return each token's output and the final state.

  Batch the work independent of S across chunks. Pad the last chunk with tokens whose key, value
  and decay are zero, so they don't change state. Then process chunks in order, using four matmuls
  per chunk to update state and write outputs.

  Within a chunk, corrections solve (I + L) u = beta v. L is strictly lower triangular and describes
  how each token's correction affects later tokens. Since L^64 = 0, setting P = -L gives
  (I + L)^-1 = (I + P)(I + P^2)(I + P^4)... . Each factor takes one or two matmuls, avoiding a loop
  over individual tokens."""
  f, (h, m, dv) = llama.fns(q), v.shape
  n, s = -(-m // size), min(m, size)  # the number of pieces, and the tokens in each
  if n > 1:  # copy each, since it may be a repeat or a view, and pad to a multiple of 64. piece i of head j is batch entry j n + i
    q, k, v, beta, g = (f.parts(f.pad(t, n * s - m) if n * s > m else t + 0, n) for t in (q, k, v, beta, g))
  upper = f.tri(s)
  cum = f.t(f.t(g) @ upper)  # the log decay from the start of the piece through each token
  decay = f.exp(f.minimum(cum - f.t(cum), 0)) * f.tri(s, True)  # the decay from token j to token i where j <= i, else 0
  kbeta = k * beta
  P, u, w, span = -((kbeta @ f.t(k)) * decay * (1 - upper)), v * beta, kbeta * f.exp(cum), 1
  while span < s:
    u, w, span = u + P @ u, w + P @ w, 2 * span
    if span < s: P = P @ P
  last = cum[:, s - 1:s, :]
  now, piece = (f.now, lambda t, i: t[i::n, :, :]) if n > 1 else (lambda t: t, lambda t, i: t)  # a prompt of one piece stays lazy
  qs, a, kl, u, w = now(q * f.exp(cum)), now((q @ f.t(k)) * decay), now(f.t(k * f.exp(last - cum))), now(u), now(w)  # what the loop reads
  out = f.empty((h, n * s, dv)) if n > 1 else None
  for i in range(n):
    vnew = piece(u, i) - piece(w, i) @ S  # subtract the predictions from the state before this chunk
    o = piece(qs, i) @ S + piece(a, i) @ vnew
    if n > 1: f.assign(piece(f.parts(out, n), i), o)
    S = now(S * f.exp(piece(last, i)) + piece(kl, i) @ vnew)
  return o if n == 1 else out if n * s == m else out[:, :m, :], S

def delta_kernel(kt, vt, vh, rep, m=1):
  """The kernel for delta: the Gated DeltaNet decode step of each sequence, or a verify step's token and its draft,
  from the projections of the tokens to their outputs. It's fp32 throughout and reads each state once.

  Arguments: the first unit and the number of units, then four words for each tensor (DRAM address, batch tile
  stride, row tile stride, tile offset), in this order:
  - S, the states
  - the output: a matrix for each head, with the m tokens of each sequence as rows in turn
  - live: bf16, a row for each token
  - with m = 2: first (the states after the first token), and accept (a column for each sequence)
  - x: a row for each token with q, k, v, z, b and a side by side, the m tokens of a sequence in turn
  - conv: the last 3 inputs of the conv, oldest first, with a row for each sequence. With m = 2, conv2 follows: the
    same after the first token.
  - taps: 4 matrices of one row
  - dt and decay: a column for each value head

  A unit is a key head of a sequence and its rep value heads, so no two units read or write the same channels of the
  conv inputs. BRISC reads the states of the unit, and the runs of its channels from the conv inputs, the rows of x
  and the taps. q and k go into columns 0-3 of a tile with a row for each channel. v goes into rows 0-3 with a column
  for each channel. b, a, dt and decay go at position 0. Once BRISC has read the inputs, it writes back the ones that
  each token leaves: they move up a position, and the token's input goes last. With m = 2 those after the first token
  go into conv2. BRISC skips this for a sequence that isn't live.

  The math thread masks everything outside those positions to 0 in Dst, so no tile in L1 needs zeroing. Then it folds.
  The 4 inputs of a channel times its 4 taps are summed across the row (or down the column for v), then go through
  SiLU into a broadcast tile. q and k go to length 1 by the sum of their squares, folded down, and q is scaled by
  kd^-1/2. beta = sigmoid(b) and e^g = exp(decay softplus(a + dt)).
  The rule then splits by column. For each value head, each column tile j of its S, and each token in turn, one SFPU
  pass sums u = S^T k and w = S^T q down the tiles of the column. Then d = beta (v - e^g u), the output of the token
  is e^g w + d (k . q), and the new tiles of the column are e^g S + k d, which is the S of the next token.
  Write the final states to S. With two tokens, save the states after the first token in first. For an inactive
  sequence, BRISC reads only live and accept, leaves the states unchanged, and writes zero outputs."""
  kh, kd, vd = vh // rep, 32 * kt, 32 * vt
  qk, vw = kh * kd, vh * vd
  S_ = lambda e, i, j: f"(F0 + ({e}) * {kt * vt} + ({i}) * {vt} + ({j}))"  # tile (i, j) of the S of value head e
  T0 = rep * kt * vt  # then the taps for q and k (in columns) and for v (in rows), then dt and decay
  TQ, TK = (lambda i, o=o: f"(F0 + {T0 + o} + ({i}))" for o in (0, kt))
  TV = lambda e, j: f"(F0 + {T0 + 2 * kt} + ({e}) * {vt} + ({j}))"
  DT, DC = (lambda e, o=o: f"(F0 + {T0 + 2 * kt + rep * vt + o} + ({e}))" for o in (0, rep))
  per = 2 * kt + rep * vt + 2 * rep  # tiles of the taps, dt and decay. Each token has as many: its q, k and v inputs, b and a
  I0 = T0 + per  # then the tiles of each token
  IQ, IK = (lambda t, i, o=o: f"(F0 + {I0 + o} + ({t}) * {per} + ({i}))" for o in (0, kt))
  IV = lambda t, e, j: f"(F0 + {I0 + 2 * kt} + ({t}) * {per} + ({e}) * {vt} + ({j}))"
  BB, AA = (lambda t, e, o=o: f"(F0 + {I0 + 2 * kt + rep * vt + o} + ({t}) * {per} + ({e}))" for o in (0, rep))
  F = I0 + m * per  # slots in a frame
  mine = per + 1  # the broadcast tiles of a token, packed: k, q and v, then e^g, beta and k . q
  KC, QC, VV, E, BT, KQ = (lambda t, o=o: 1 + 2 * F + t * mine + o
                           for o in (0, kt, 2 * kt, 2 * kt + rep * vt, 2 * kt + rep * vt + rep, mine - 1))
  COL = 1 + 2 * F + m * mine  # the new tiles of the column after the first of two tokens: the S of the second
  OUT = COL + kt * (m - 1)  # an output slot for each Dst half
  sources = 7 + m  # the rows of the conv that a unit reads: the 3 last inputs, one for each token, and the 4 taps
  STAGE = slot_base("fp32") + (OUT + 2) * 4096  # BRISC stages the q and k runs of each source here, then a 64-byte block per scalar
  scalars = 2 + 2 * rep + 2 * m * rep  # live and accept, dt and decay, and b and a for each token
  if STAGE + sources * 256 * kt + 64 * scalars > OUTS or OUT + 2 > UNPACKABLE: raise ValueError("delta_kernel: a unit's tiles don't fit in L1")
  # Dst rows, 64 to a tile in fp32. R2 and R3 take fills, R4 and R5 the two sums (or a norm, or decay), R6 takes e^g,
  # and R7 takes d or x set aside.
  R2, R3, R4, R5, R6, R7 = 128, 192, 256, 320, 384, 448
  programs = {  # each runs at Dst row 0 (aim(0)) and stores its result to row r
    "kq first": lambda r: [sfp_load(0, R2), sfp_load(1, R3), mul(2, 0, 1), sfp_store(2, R4)],
    "kq": lambda r: [sfp_load(0, R2), sfp_load(1, R3), sfp_load(2, R4), mad(2, 0, 1, 2), sfp_store(2, R4)],
    "sums first": lambda r: [sfp_load(0, R2), sfp_load(1, R3), sfp_load(2, R6), mul(3, 1, 0), mul(4, 2, 0), sfp_store(3, R4, False),
                             sfp_store(4, R5)],
    "sums": lambda r: [sfp_load(0, R2), sfp_load(1, R3), sfp_load(2, R6), sfp_load(3, R4), sfp_load(4, R5), mad(3, 1, 0, 3),
                       mad(4, 2, 0, 4), sfp_store(3, R4, False), sfp_store(4, R5)],
    "d": lambda r: [sfp_load(0, R2), sfp_load(1, R3), sfp_load(2, R6), sfp_load(3, R4), mad(4, 2, 3, 1, neg=1), mul(5, 0, 4),
                    sfp_store(5, R7)],  # beta (v - e u)
    "out": lambda r: [sfp_load(0, R6), sfp_load(1, R5), sfp_load(2, R7), sfp_load(3, R2), mul(4, 0, 1), mad(5, 2, 3, 4),
                      sfp_store(5, r)],  # e w + d (k . q)
    "state": lambda r: [sfp_load(0, R2), sfp_load(1, R3), sfp_load(2, R6), sfp_load(3, R7), mul(4, 2, 0), mad(5, 1, 3, 4),
                        sfp_store(5, r)],  # e S + k d
    "copy": lambda r: [sfp_load(0, R4), sfp_store(0, r)],
    "mul": lambda r: [sfp_load(0, R2), sfp_load(1, R3), mul(2, 0, 1), sfp_store(2, r)],  # inputs times their taps
    "add": lambda r: [sfp_load(0, R2), sfp_load(1, R3), mad(2, 0, ONE, 1), sfp_store(2, r)],  # a + dt
    "aside": lambda r: [sfp_load(0, r), sfp_store(0, R7)],  # keeps x for the x sigmoid(x) of SiLU and the max(x, 0) of softplus
    "neg": lambda r: [sfp_load(0, r), mul(1, 0, MINUS_ONE), sfp_store(1, r)],
    "-abs": lambda r: [sfp_load(0, r), ins("SFPABS", vd=1, vc=0, mod1=1), mul(2, 1, MINUS_ONE), sfp_store(2, r)],
    "1+": lambda r: [sfp_load(0, r), mad(1, 0, ONE, ONE), sfp_store(1, r)],
    "times x": lambda r: [sfp_load(0, r), sfp_load(1, R7), mul(2, 0, 1), sfp_store(2, r)],
    "+ max(x, 0)": lambda r: [sfp_load(0, R7), ins("SFPMOV", vd=1, vc=ZERO), ins("SFPSWAP", vd=1, vc=0, mod1=1), sfp_load(2, r),
                              mad(3, 2, ONE, 0), sfp_store(3, r)],
    "times R4": lambda r: [sfp_load(0, r), sfp_load(1, R4), mul(2, 0, 1), sfp_store(2, r)],  # times decay, or q or k times its scale
    "squares first": lambda r: [sfp_load(0, R2), mul(1, 0, 0), sfp_store(1, R4)],
    "squares": lambda r: [sfp_load(0, R2), sfp_load(1, R4), mad(1, 0, 0, 1), sfp_store(1, R4)],
    "+ eps": lambda r: [sfp_load(0, R4), *sfp_const(5, 1e-6), mad(1, 0, ONE, 5), sfp_store(1, R4)],
    "q's scale": lambda r: [sfp_load(0, R4), *sfp_const(5, kd ** -0.5), mul(1, 0, 5), sfp_store(1, R4)]}
  # The programs, folds, masks and SFPU programs that the math thread runs are functions (helpers, below), so their
  # words are in the program once and the loops that call them stay short. A program takes the row for its result. The
  # rest work wherever aim_at last aimed.
  name = lambda n: "p_" + "".join(ch if ch.isalnum() else "_" for ch in n)
  call = lambda *names: " ".join(f"{name(n)}();" for n in names)
  apply = lambda n: f"{name(n)}(0);"  # for a program that stores to its own rows
  to_h = lambda n: f"{name(n)}(h);"  # for a program whose result goes to Dst half h
  aim_at = lambda row: tt(aim(row))
  free = tt([ins("STALLWAIT", stall_res=bh.STALL_UNPACK, wait_res=bh.WAIT_SFPU)])  # fills wait until the SFPU is done with their rows
  fills = lambda *pairs: free + " " + tt([w for slot, row in pairs for w in fill(slot, row)] + FILLED)
  half = tt([DST_FREE])  # waits until a Dst half is free for a result
  done = f"{tt(MATH_END)} h ^= {DST_HALF};"
  lib = lambda *names: f"{aim_at('h')} {call(*names)} {aim_at(0)}"  # SFPU library programs, run on the tile at h
  silu = f"{to_h('aside')} {to_h('neg')} {lib('fexp')} {to_h('1+')} {lib('frecip')} {to_h('times x')}"
  masks = {"columns 4 on": mask_columns("sum", 4), "rows 4 on": mask_rows("sum", 4), "all but place 0": mask_rows("sum", 1) + mask_columns("sum", 1)}
  only = lambda mask: f"{aim_at('h')} {call(mask)}"  # zeroes the tile at h outside the positions that were filled
  conv = lambda x, taps, mask, kind: f"{aim_at(0)} {fills((x, R2), (taps, R3))} {to_h('mul')} {only(mask)} {call(kind)} {aim_at(0)} {silu}"
  # Generate (math, pack) words for Dst half h. Pack intermediates into slot and final results into the output slot
  # for this half, with `home` as the destination tile. A result used both ways stays in slot until written to DRAM;
  # keep identifies the output slot used to track its completion.
  def result(words, slot=None, home=None, keep=None):
    math = f"{half} {words} {done}" + (" posted++;" if slot is not None else "")
    if home is None: return math, f"{tt(pack('h', slot, SEM_L1))} h ^= {DST_HALF};"
    to, keep = (slot, keep) if slot is not None else (f"{OUT} + h / {DST_HALF}", f"h / {DST_HALF}")
    return math, f"output({keep}, {to}, {home}); {tt(pack('h', to, SEM_PACKED, *([SEM_L1] if slot is not None else [])))} h ^= {DST_HALF};"
  loop = lambda var, n, pair: tuple(f"for (unsigned {var} = 0; {var} < {n}; {var}++) {{ {w} }}" for w in pair)  # a (math, pack) pair, n times
  # Scales the kt tiles of q or k at slot c to length 1, and by scale if given. The sum of their squares is folded down,
  # 1 / sqrt of it goes in R4, then each tile is multiplied by it.
  def unit_length(c, scale):
    return [(f"drain(); {aim_at(0)} {fills((c, R2))} {apply('squares first')} for (unsigned i = 1; i < {kt}; i++) {{ {fills((f'{c} + i', R2))} "
             f"{apply('squares')} }} {aim_at(R4)} {call('sum1')} {aim_at(0)} {apply('+ eps')} {aim_at(R4)} {call('sqrt', 'frecip')} {aim_at(0)}"
             + (f" {apply(scale)}" if scale else ""), ""),
            loop("i", kt, result(f"{fills((f'{c} + i', 'h'))} {to_h('times R4')}", f"{c} + i"))]
  steps = []
  for t in range(m):  # the broadcast tiles of each token: the conv, the gates and k . q
    steps += [loop("i", kt, result(conv(IQ(t, "i"), TQ("i"), "columns 4 on", "sum2"), f"{QC(t)} + i")),
              loop("i", kt, result(conv(IK(t, "i"), TK("i"), "columns 4 on", "sum2"), f"{KC(t)} + i")),
              *unit_length(QC(t), "q's scale"), *unit_length(KC(t), None),
              loop("e", rep, loop("j", vt, result(conv(IV(t, "e", "j"), TV("e", "j"), "rows 4 on", "sum1"), f"{VV(t)} + e * {vt} + j"))),
              loop("e", rep, result(f"{aim_at('h')} {fills((BB(t, 'e'), 'h'))} {only('all but place 0')} {call('sum2', 'sum1')} {aim_at(0)} "
                                    f"{to_h('neg')} {lib('fexp')} {to_h('1+')} {lib('frecip')}", f"{BT(t)} + e")),  # beta = sigmoid(b)
              loop("e", rep, result(f"{aim_at(0)} {fills((DC('e'), R4))} {aim_at(R4)} {call('all but place 0', 'sum2', 'sum1')} {aim_at(0)} "
                                    f"{fills((AA(t, 'e'), R2), (DT('e'), R3))} {to_h('add')} {only('all but place 0')} {call('sum2', 'sum1')} "
                                    f"{aim_at(0)} {to_h('aside')} {to_h('-abs')} {lib('fexp')} {to_h('1+')} {lib('log')} {to_h('+ max(x, 0)')} "
                                    f"{to_h('times R4')} {lib('fexp')}", f"{E(t)} + e")),  # e^g = exp(decay softplus(a + dt))
              (f"drain(); {aim_at(0)} {fills((KC(t), R2), (QC(t), R3))} {apply('kq first')} for (unsigned i = 1; i < {kt}; i++) "
               f"{{ {fills((f'{KC(t)} + i', R2), (f'{QC(t)} + i', R3))} {apply('kq')} }} {aim_at(R4)} {call('sum1')} {aim_at(0)}", ""),
              result(f"{free} {to_h('copy')}", KQ(t)), ("drain();", "")]
  column = []  # column tile j of the S of each value head, through each token in turn
  for t in range(m):
    col = (lambda i: S_("e", i, "j")) if t == 0 else (lambda i: f"{COL} + ({i})")
    column += [("drain();" if t else "", ""),
               (f"{fills((col(0), R2), (KC(t), R3), (QC(t), R6))} {apply('sums first')} "
                f"for (unsigned i = 1; i < {kt}; i++) {{ {fills((col('i'), R2), (f'{KC(t)} + i', R3), (f'{QC(t)} + i', R6))} {apply('sums')} }} "
                f"{aim_at(R4)} {call('sum1')} {aim_at(R5)} {call('sum1')} {aim_at(0)} "
                f"{fills((f'{BT(t)} + e', R2), (f'{VV(t)} + e * {vt} + j', R3), (f'{E(t)} + e', R6))} {apply('d')}", ""),
               result(f"{fills((KQ(t), R2))} {to_h('out')}", home=f"{1 << 30:#x} | (unit * {rep} + e) << 8 | {t} << 6 | j"),
               loop("i", kt, result(f"{fills((col('i'), R2), (f'{KC(t)} + i', R3))} {to_h('state')}", f"{COL} + i" if t < m - 1 else None,
                                    f"{(m - 1 - t) << 31:#x} | (unit * {rep} + e) << 8 | (i * {vt} + j)", "2 + i"))]
  steps.append(loop("e", rep, loop("j", vt, (" ".join(c for c, _ in column), " ".join(c for _, c in column)))))
  math, packs = [w for w, _ in steps], [w for _, w in steps]
  units = lambda body: f"""  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned unit = a[0] + u, F0 = 1 + (u & 1) * {F};
    {body}
  }}"""
  # For each unit, BRISC waits for its frame to be free, then puts its states, the runs of its channels and its scalars in
  # place, then writes back the updated conv inputs. It writes results to DRAM as they're packed.
  # Below are the indices in args of first (which is S when m is 1), x, the conv inputs, the taps, dt and decay.
  first, X = (14, 22) if m > 1 else (2, 14)
  C, C2, TAPS = X + 4, X + 8, X + 4 + 4 * m
  DTA, DCA = TAPS + 4, TAPS + 8
  sc = lambda k: f"STAGE + {sources * 256 * kt} + 64 * ({k})"  # the 64-byte block of scalar k
  scalar = lambda i, row, col, k, size=4: (f"{{ unsigned t_ = {tile_of(i, 0, f'({row}) / 32', f'({col}) / 32', 'A')}, "  # size is 2 for bf16
                                           f"o_ = at(({row}) % 32, ({col}) % 32) * {size}; noc_get(BANK(t_), AT(A[{i}], t_, {1024 * size}) "
                                           f"+ o_ / 64 * 64, {sc(k)}, 64); word[{k}] = o_ % 64 / {size}; }}")
  value = lambda k, size=4: f"((volatile {'unsigned short' if size == 2 else 'unsigned'} *)({sc(k)}))[word[{k}]]"
  staged = lambda src, g: f"(STAGE + ({src}) * {256 * kt} + ({g}) * {128 * kt})"  # the q (g 0) or k (g 1) runs staged from source src
  brisc = SYNC_WORDS + BANKS + f"""
#define SLOT(i) ({slot_base("fp32"):#x} + (i) * 4096)
#define STAGE {STAGE:#x}
static const unsigned *A;
static unsigned fetched, events, frame_done[2], sent, alive[4], word[{scalars}];
static void ready(void) {{ noc_wait(); SYNC[FETCHED] = ++fetched; }}
static void send(void) {{  // write packed output rows (zero if inactive) and active state tiles to DRAM
  for (; sent < peek(PACKED); SYNC[SENT] = ++sent) {{
    unsigned slot = peek(RING + sent % 4 * 2), t = peek(RING + sent % 4 * 2 + 1), unit = t >> 8 & 0x3FFFFF, s = unit / {vh}, h = unit % {vh};
    unsigned live = alive[unit / {rep} % 4];
    if (t >> 30 & 1) for (unsigned f = 0; f < 2; f++) {{  // an output tile, for the token of the head: the runs of its row
      unsigned r = s * {m} + (t >> 6 & 3), o = {tile_of(6, 'h', 'r / 32', 't & 63', 'A')}, off = at(r % 32, 16 * f) * 4;
      noc_put((live ? SLOT(slot) : {ZEROS:#x}) + off, BANK(o), AT(A[6], o, 4096) + off, 64);
    }} else if (live) {{  // a tile of the states, or of those after the first token
      unsigned i = t >> 31 ? {first} : 2, o = {tile_of('i', 'unit', f'(t & 255) / {vt}', f'(t & 255) % {vt}', 'A')};
      noc_put(SLOT(slot), BANK(o), AT(A[i], o, 4096), 4096);
    }}
    noc_wait();
  }}
}}
static void reached(unsigned n) {{ while (peek(UNPACKED) < n) send(); }}
// Reads the row of source src: the 3 last conv inputs (from the states at A[c]), the rows of the tokens in x, or the 4
// taps. It takes the runs of the channels in n tiles from col on, from both faces that the row crosses, into L1 at to.
// Each tile goes step bytes on from the last, and the second face goes half bytes on from the first.
static void get_runs(unsigned src, unsigned s, unsigned c, unsigned col, unsigned n, unsigned to, unsigned step, unsigned half) {{
  unsigned i = src < 3 ? c : src < {3 + m} ? {X} : {TAPS}, b = src < 3 ? src : src < {3 + m} ? 0 : src - {3 + m};
  unsigned row = src < 3 ? s : src < {3 + m} ? s * {m} + src - 3 : 0, t = {tile_of('i', 'b', 'row / 32', 'col / 32', 'A')};
  unsigned off = at(row % 32, 0) * 4;
  for (unsigned j = 0; j < n; j++, t++, to += step) {{
    noc_get(BANK(t), AT(A[i], t, 4096) + off, to, 64);
    noc_get(BANK(t), AT(A[i], t, 4096) + off + 1024, to + half, 64);
  }}
}}
// The reverse: writes the runs of row s from L1 at from, laid out the same, into conv input d at A[i]
static void put_runs(unsigned i, unsigned d, unsigned s, unsigned col, unsigned n, unsigned from, unsigned step, unsigned half) {{
  unsigned t = {tile_of('i', 'd', 's / 32', 'col / 32', 'A')}, off = at(s % 32, 0) * 4;
  for (unsigned j = 0; j < n; j++, t++, from += step) {{
    noc_put(from, BANK(t), AT(A[i], t, 4096) + off, 64);
    noc_put(from + half, BANK(t), AT(A[i], t, 4096) + off + 1024, 64);
  }}
}}
void kmain(volatile unsigned *args) {{
  A = (const unsigned *)args;
  for (unsigned u = 0; u < A[1]; u++) {{
    unsigned unit = A[0] + u, F0 = 1 + (u & 1) * {F}, s = unit / {kh}, hk = unit % {kh};
    unsigned cq = hk * {kd}, ck = {qk} + hk * {kd}, cv = {2 * qk} + hk * {rep * vd};  // its q and k channels, and the v channels of its value heads
    reached(frame_done[u & 1]);
    {scalar(10, f's * {m}', 0, 0, 2)} {scalar(18, 0, 's', 1) if m > 1 else ""}
    noc_wait();
    alive[unit % 4] = {value(0, 2)} != 0;
    if (!alive[unit % 4]) {{ ready(); frame_done[u & 1] = ++events; continue; }}  // a sequence that isn't live reads nothing
    // accept selects the state after the previous token or its draft
    unsigned taken = {value(1) if m > 1 else 1}, from = taken ? 2 : {first}, c = taken ? {C} : {C2};
    for (unsigned e = 0; e < {rep}; e++) for (unsigned i = 0; i < {kt}; i++) for (unsigned j = 0; j < {vt}; j++) {{
      unsigned t = {tile_of('from', f's * {vh} + hk * {rep} + e', 'i', 'j', 'A')};
      noc_get(BANK(t), AT(A[from], t, 4096), SLOT({S_('e', 'i', 'j')}), 4096);
    }}
    for (unsigned src = 0; src < {sources}; src++) for (unsigned g = 0; g < 2; g++)  // stage the runs of q and k
      get_runs(src, s, c, g ? ck : cq, {kt}, {staged('src', 'g')}, 128, 64);
    for (unsigned r = 0; r < 4; r++) {{  // v into rows 0-3 of its tiles, for the inputs of each token and the taps
      for (unsigned t = 0; t < {m}; t++) get_runs(t + r, s, c, cv, {rep * vt}, SLOT({IV('t', 0, 0)}) + r * 64, 4096, 1024);
      get_runs({3 + m} + r, s, c, cv, {rep * vt}, SLOT({TV(0, 0)}) + r * 64, 4096, 1024);
    }}
    for (unsigned e = 0; e < {rep}; e++) {{
      unsigned h = hk * {rep} + e;
      {scalar(DTA, 0, 'h', '2 + e')} {scalar(DCA, 0, 'h', f'{2 + rep} + e')}
      for (unsigned t = 0; t < {m}; t++) {{
        {scalar(X, f's * {m} + t', f'{2 * qk + 2 * vw} + h', f'{2 + 2 * rep} + t * {2 * rep} + e')}
        {scalar(X, f's * {m} + t', f'{2 * qk + 2 * vw + vh} + h', f'{2 + 3 * rep} + t * {2 * rep} + e')}
      }}
    }}
    noc_wait();
    for (unsigned g = 0; g < 2; g++) for (unsigned i = 0; i < {kt}; i++) {{  // q and k, a row per channel: input or tap j in column j
      unsigned *taps = (unsigned *)SLOT(g ? {TK('i')} : {TQ('i')}), *ins[{m}];
      // tile i of source 0, with channel r at word r (each source sits {64 * kt} words past the one before)
      const unsigned *from = (const unsigned *){staged(0, 'g')} + i * 32;
      for (unsigned t = 0; t < {m}; t++) ins[t] = (unsigned *)SLOT(g ? {IK('t', 'i')} : {IQ('t', 'i')});
      for (unsigned r = 0; r < 32; r++) for (unsigned j = 0; j < 4; j++) {{
        taps[at(r, j)] = from[({3 + m} + j) * {64 * kt} + r];
        for (unsigned t = 0; t < {m}; t++) ins[t][at(r, j)] = from[(t + j) * {64 * kt} + r];
      }}
    }}
    for (unsigned e = 0; e < {rep}; e++) {{
      ((volatile unsigned *)SLOT({DT('e')}))[0] = {value('2 + e')}, ((volatile unsigned *)SLOT({DC('e')}))[0] = {value(f'{2 + rep} + e')};
      for (unsigned t = 0; t < {m}; t++) {{
        ((volatile unsigned *)SLOT({BB('t', 'e')}))[0] = {value(f'{2 + 2 * rep} + t * {2 * rep} + e')};
        ((volatile unsigned *)SLOT({AA('t', 'e')}))[0] = {value(f'{2 + 3 * rep} + t * {2 * rep} + e')};
      }}
    }}
    // after reading, update convolution history for token t with sources t + 1 to t + 3
    for (unsigned t = 0; t < {m}; t++) for (unsigned d = 0; d < 3; d++) {{
      unsigned into = t == {m - 1} ? {C} : {C2}, src = t + 1 + d, tv = src > 3 ? src - 3 : 0;
      for (unsigned g = 0; g < 2; g++) put_runs(into, d, s, g ? ck : cq, {kt}, {staged('src', 'g')}, 128, 64);
      put_runs(into, d, s, cv, {rep * vt}, SLOT({IV('tv', 0, 0)}) + (src - tv) * 64, 4096, 1024);  // from the tiles of token tv
    }}
    l1_settle(SLOT(F0));
    ready(); frame_done[u & 1] = ++events;
  }}
  while (sent < A[1] * {rep * vt * m * (kt + 1)}) send();
}}"""
  def body(p):  # a program's words, its result to row 0 or the other Dst half. One that ignores r is emitted once
    other, zero = (tt(SFPU_START + [Loop(32, p(r))]) for r in (DST_HALF, 0))
    return zero if other == zero else f"if (h) {{ {other} }} else {{ {zero} }}"
  helper = lambda n, words, arg="void": f"__attribute__((noinline)) static void {name(n)}({arg}) {{ {words} }}\n"
  helpers = "".join([helper(k, tt(ACROSS[FOLDS[k]](k))) for k in ("sum1", "sum2")] +  # sum1 sums down the columns, sum2 along the rows
                    [helper(n, tt(sfpu([n], {}))) for n in ("fexp", "frecip", "log", "sqrt")] +
                    [helper(n, tt(words)) for n, words in masks.items()] + [helper(n, body(p), "unsigned h") for n, p in programs.items()])
  cfg = kernel_config("fp32", "fp32")
  return compile_kernel(brisc, unpack_program(units("fetched();"), cfg),
                        math_program({}, units("fetched(); " + " ".join(math) + " " + tt(REPORT)), True, helpers),
                        pack_program(units(" ".join(packs)), 2 + kt))

def delta(S, x, conv, taps, dt, decay, live=None, first=None, accept=None, conv2=None):
  """The Gated DeltaNet decode step in one kernel (delta_kernel), for a token of each sequence, or a verify step's
  token and its draft. It starts from the projections of the tokens, updates the states in place, and reads each state
  once. Each token does S = e^g S + beta k (v - e^g S^T k)^T, and the output of each head is S^T q after that.

  - S: the states, kd x vd, for each sequence and value head
  - x: the projections of each token in a row (q, k, v, z, b and a side by side), the tokens of a sequence in turn
  - conv: the last 3 inputs of the conv, 3 matrices with the oldest first and a row for each sequence. Each token
    moves them up a position and adds its input last.
  - taps: 4 matrices of one row. dt and decay: a row with a column for each value head.
  - live: a bf16 column with an entry for each token, all 1 if not given. Where the first token of a sequence has 0,
    its conv and states aren't read or written, and its outputs are 0.
  - first and conv2, for a token and its draft: the states and conv inputs after the first token. accept, a row
    with 1 or 0 for each sequence, says to go on from S and conv (1) or from these (0). first and conv2 then take the
    state after this step's token, and S and conv the state after its draft.

  q, k and v are the SiLU of the conv. q and k go to length 1, and q is scaled by kd^-1/2. beta = sigmoid(b) and
  g = decay softplus(a + dt). Everything but live is fp32. Returns the outputs in fp32, as (value heads, the tokens of
  each sequence in turn, vd). Head dimensions must be multiples of 32. There's no gradient."""
  n, kd, vd, width = conv.rows, S.rows, S.cols, conv.cols
  if kd % 32 or vd % 32: raise ValueError("delta needs key and value head dimensions divisible by 32")
  if x.rows % n or S.batch % n: raise ValueError("delta needs the same number of sequences in inputs and states")
  m, vh = x.rows // n, S.batch // n
  kh, extra = divmod(width - vh * vd, 2 * kd)
  if extra or kh < 1 or vh % kh: raise ValueError("delta needs complete q/k/v projections and value heads in groups of key heads")
  aux = (first, accept, conv2)
  if m not in (1, 2) or any((t is None) == (m == 2) for t in aux):
    raise ValueError("delta needs one token, or two tokens with first, accept and conv2 together")
  ins = (S, x, conv, taps, dt, decay, *(aux if m == 2 else ()))
  if not all(t.dtype == "fp32" for t in ins): raise ValueError("delta needs fp32 states and inputs")
  if any(t.rt is not S.rt for t in (*ins, *((live,) if live is not None else ()))):
    raise ValueError("delta needs states and inputs on the same runtime")
  if not (x.shape == (1, n * m, width + vh * vd + 2 * vh) and conv.batch == 3 and taps.shape == (4, 1, width)):
    raise ValueError(f"delta needs a row of q, k, v, z, b and a for each token, 3 conv inputs and 4 taps: {x.shape} {conv.shape} {taps.shape}")
  if dt.shape != (1, 1, vh) or decay.shape != dt.shape: raise ValueError(f"delta needs dt and decay as a row of {vh}")
  if live is not None and (live.shape != (1, n * m, 1) or live.dtype != "bf16"):
    raise ValueError(f"delta needs live as a bf16 column of {n * m}")
  if m == 2 and not (first.shape == S.shape and conv2.shape == conv.shape and accept.shape == (1, 1, n)):
    raise ValueError("delta needs first like S, conv2 like conv, and accept as a row of sequences")
  live = live if live is not None else S.number(1.0, "bf16").expand(1, n * m, 1).realize()
  o = Tensor.frombytes(S.rt, n * m, vd, bytes(4 * vh * n * m * vd), vh, "fp32")  # zeroed, because no unit writes its padding rows
  key = ("delta", -(-kd // 32), -(-vd // 32), vh, vh // kh, m)
  return o.run(key, lambda: delta_kernel(*key[1:]), S, o, live, *((first, accept) if m > 1 else ()), x, conv, *((conv2,) if m > 1 else ()),
               taps, dt, decay, units=n * kh, name="delta")

def moe(h, p, c, phases=lambda op: None, live=None):
  """A mixture of experts in place of the MLP, as in Qwen3 and Mixtral. The router picks the c.k best experts for each
  token (topk, on the chip). Their SwiGLUs are weighted by the softmax of their logits and summed, which is the same as
  the renormalized top-k probabilities of Qwen3. A shared expert, if the block has one, adds its SwiGLU, scaled by the
  sigmoid of shared_expert_gate.

  p has router (dim x experts), experts.gate_up and experts.down (the matrices of the experts as a batch), and the
  biases of gpt-oss: router.bias, experts.gate_up_bias and experts.down_bias, a row for each expert. phases(op) gives
  the fidelity phases of gate_up and down. c.limit is for the GLU of gpt-oss (glu). live is a column with 1 or 0 for
  each token. A token with 0 weights its experts by 0, so a decode step reads none of them for it (touched)."""
  f, norm = llama.fns(h), c.norm_topk
  logits = f.mm(h, p["router"], None)
  logits = logits + p["router.bias"] if "router.bias" in p else logits  # gpt-oss has a router bias
  values, ids = f.topk(logits if norm else f.softmax(logits), c.k)  # without norm_topk_prob, probabilities among all the experts
  w = f.softmax(values) if norm else values
  y = f.experts(h, p, ids, w if live is None else w * live, (phases("gate_up"), phases("down")), c.limit)
  if "shared_expert_gate" not in p: return y
  # Qwen2-MoE also has a shared expert, which all tokens go through
  return y + f.sigmoid(f.mm(h, p["shared_expert_gate"], None)) * llama.mlp(h, p, phases, "shared_expert.")

def numpy_experts(h, p, ids, w, phases, limit=None):  # the GLUs of the experts that each token chose, weighted by w and summed
  gate_up, down, m = p["experts.gate_up"], p["experts.down"], p["experts.down"].shape[-2]
  bias = lambda name, e: p[name][e] if name in p else 0
  def expert(x, e):
    g = x @ gate_up[e] + bias("experts.gate_up_bias", e)
    return llama.glu(g[:m], g[m:], limit) @ down[e] + bias("experts.down_bias", e)
  return np.stack([sum(w[t, j] * expert(h[t], e) for j, e in enumerate(ids[t])) for t in range(len(h))])

def chip_experts(h, p, ids, w, phases, limit=None):
  """Run the selected experts on the chip. Indexed matmuls read expert weights where they are stored.

  For up to 128 tokens, touched creates a batch entry for each selected expert. Tokens become rows of each
  entry, with zero routing weight for tokens that didn't select that expert. Multiply the SwiGLU outputs by
  these weights and sum across entries. Workers read the token rows once and share each expert's weight
  reads.

  For longer prompts, group sorts tokens by expert and pads each group to 32 rows. Each tile becomes a batch
  entry, so weights are read once per 32 tokens. Gather the k outputs per token, then weight and sum them.
  Above 128 tokens, choose the layout with fewer tile rows. Both layouts support replay; unused entries
  produce zeros. Gather expert biases using the same indices."""
  gate_up, down = p["experts.gate_up"], p["experts.down"]
  (n, k), m, e = ids.shape[1:], down.rows, gate_up.batch
  dense = n <= 128 or -(-n // 32) * min(n * k, e) <= -(-n * k // 32) + min(n * k, e)  # tile rows per matrix, each way
  if dense: (picks, weights), x = touched(ids, e, w), h
  else:
    order, picks, places = group(ids, e)
    x = h.gather(order).split(picks.rows, 1)
  entries = picks.rows
  row = picks.reshape(1, entries).minimum(e - 1) if {"experts.gate_up_bias", "experts.down_bias"} & set(p) else None  # the picks as a row
  # the bias of the expert of each entry. An entry with no expert gets the bias of the last expert, weighted 0
  bias = lambda name: p[name].gather(row).reshape(entries, 1, p[name].cols)
  plus = lambda y, name, by=1: y + bias(name) * by if name in p else y
  x = plus(x.matmul(gate_up, fidelity=phases[0], index=picks), "experts.gate_up_bias")
  g = llama.glu(x[:, :, :m], x[:, :, m:], limit)
  if dense: return plus((g * weights).matmul(down, fidelity=phases[1], index=picks), "experts.down_bias", weights).sum(0)
  y = plus(g.matmul(down, fidelity=phases[1], index=picks), "experts.down_bias").reshape(order.cols, down.cols)
  run = places.cols // k  # the length of the run of places for each choice
  terms = [y.gather(places[:, :, j * run:j * run + n]) * w[:, :, j:j + 1] for j in range(k)]
  return sum(terms[1:], terms[0])

# The tiles of slots that group needs for any choices: a tile for each 32 choices, plus one for each value that can
# come, since its run may end partway through a tile. Also the length of a run of places, the rows rounded up to a tile.
def group_slots(rows, k, n): return -(-rows * k // 32) + min(n, rows * k), -(-rows // 32) * 32

def group_kernel(rows, k, n):
  """Sort token choices by expert on one BRISC worker. Arguments are [first unit, units, order address, tiles
  address, places address, choices address, choices row tile stride, choices tile offset].

  Read rows x k choices with values below n. Count each expert, assign consecutive slots, and pad each
  expert's group to 32. Build order (token per slot), tiles (expert per tile) and places (slot per choice,
  grouped by choice index) as flat arrays in L1. Send them as tiles, repeating rows or columns to match
  tensor storage."""
  t, rp = group_slots(rows, k, n)
  fills = DATA + -(-rows // 32) * 4096  # after the choices: the count of each value, then its next slot
  out = (fills + 4 * (n + 33 * t + k * rp) + 63) // 64 * 64
  if out + 4096 > OUTS: raise ValueError("group_kernel: the choices and their arrays don't fit in L1")
  return compile_kernel(f"""
#define IDS {DATA:#x}
#define FILL {fills:#x}
#define ORDER (FILL + {4 * n})
#define TILES (ORDER + {4 * 32 * t})
#define PLACES (TILES + {4 * t})
#define OUT {out:#x}
{ROWS}
static unsigned choice(unsigned r, unsigned j) {{ return ((volatile unsigned *)IDS)[r / 32 * 1024 + at(r % 32, j)]; }}
static void send(unsigned base, unsigned i) {{ noc_write(OUT, BANK(i), AT(base, i, 4096), 4096); noc_wait(); }}
void kmain(volatile unsigned *a) {{
  volatile unsigned *fill = (volatile unsigned *)FILL, *order = (volatile unsigned *)ORDER;
  volatile unsigned *tiles = (volatile unsigned *)TILES, *places = (volatile unsigned *)PLACES, *out = (volatile unsigned *)OUT;
  if (!a[1]) return;
  for (unsigned i = 0; i < {-(-rows // 32)}; i++) {{ unsigned u = a[7] + i * a[6]; noc_get(BANK(u), AT(a[5], u, 4096), IDS + i * 4096, 4096); }}
  for (unsigned e = 0; e < {n}; e++) fill[e] = 0;
  for (unsigned s = 0; s < {32 * t}; s++) order[s] = 0;
  for (unsigned i = 0; i < {t}; i++) tiles[i] = {n};  // n marks unused entries, which indexed matmul skips
  for (unsigned s = 0; s < {k * rp}; s++) places[s] = 0;
  noc_wait();
  for (unsigned r = 0; r < {rows}; r++) for (unsigned j = 0; j < {k}; j++) fill[choice(r, j)]++;
  for (unsigned e = 0, s = 0; e < {n}; e++) {{  // the run of each value: its count rounded up to a multiple of 32, after the runs before it
    unsigned count = fill[e];
    fill[e] = s;
    for (unsigned end = s + (count + 31) / 32 * 32; s < end; s += 32) tiles[s / 32] = e;
  }}
  for (unsigned r = 0; r < {rows}; r++)
    for (unsigned j = 0; j < {k}; j++) {{ unsigned s = fill[choice(r, j)]++; order[s] = r; places[j * {rp} + r] = s; }}
  for (unsigned i = 0; i < {t}; i++) {{  // order and places are rows: fill row 0 of each tile and copy it down. tiles is a column
    for (unsigned c = 0; c < 32; c++) out[at(0, c)] = order[i * 32 + c];
    row_down(OUT, 4), send(a[2], i);
  }}
  for (unsigned i = 0; i < {-(-t // 32)}; i++) {{
    for (unsigned r = 0; r < 32; r++) for (unsigned c = 0; c < 32; c++) out[at(r, c)] = i * 32 + r < {t} ? tiles[i * 32 + r] : {n};
    send(a[3], i);
  }}
  for (unsigned i = 0; i < {k * rp // 32}; i++) {{
    for (unsigned c = 0; c < 32; c++) out[at(0, c)] = places[i * 32 + c];
    row_down(OUT, 4), send(a[4], i);
  }}
}}""")

def group(ids, n):
  """Group rows x k int32 choices by expert, with IDs below n. Return (order, tiles, places), all int32.
  group_kernel runs this on BRISC.

  order is a row of 32*t slots, where t is group_slots. Each slot holds the source token row, or zero for
  padding. Expert groups start on tile boundaries. tiles is a column of t expert IDs, with n for unused
  tiles. places has k runs, one slot index per token per choice, with each run padded to a tile. Indexed
  matmul uses tiles to select the expert weights."""
  if not (ids.dtype == "int32" and ids.batch == 1 and ids.cols <= 32):
    raise ValueError("group needs ids as a matrix of int32 choices, at most 32 to a row")
  x, rows, k = ids.realize(), ids.rows, ids.cols  # a matrix, so there's no repeat to copy
  t, rp = group_slots(rows, k, n)
  order, tiles, places = Tensor(ids.rt, 1, 32 * t, dtype="int32"), Tensor(ids.rt, t, 1, dtype="int32"), Tensor(ids.rt, 1, k * rp, dtype="int32")
  order.run(("group", rows, k, n), lambda: group_kernel(rows, k, n), order.addr, tiles.addr, places.addr, x.addr, x.stride, x.offset,
            units=1, name="group")
  return order, tiles, places

def touched_l1(rows, k, n):
  """The L1 address of touched's output tile, after its ids, weights and expert list. Reject inputs that would put
  the end of the tile past OUTS."""
  if type(n) is not int or n < 1: raise ValueError("touched needs a positive integer expert count")
  out = (DATA + -(-rows // 32) * 6144 + 4 * -(-n // 32) + 4 * min(rows * k, n) + 63) // 64 * 64
  if out + 4096 > OUTS: raise ValueError(f"touched inputs and expert list do not fit L1: {rows} rows, {k} choices, {n} experts")
  return out

def touched_kernel(rows, k, n):
  """The kernel for touched, which lists the experts that some row picks. It runs on BRISC alone, with a unit for each
  entry.

  Arguments: first unit, units, the addresses of experts and weights, then the address, tile stride for rows and tile
  offset of the choices, then the same for the bf16 weights of the choices.
  Workers read the choices (rows x k, values below n) and their weights once. They mark the experts picked with a
  nonzero weight in a bitmask, and list them in increasing order, then n (none) after them. Entry s writes its tile
  of each tile row of its weights: row r holds the weight that row r gave expert s of the list (0 if it didn't pick
  it), across the row. The first entry of each 32 also writes its tile of the experts column, each expert across its
  row, as the tile of a column holds it."""
  out = touched_l1(rows, k, n)
  tr, slots = -(-rows // 32), min(rows * k, n)
  return compile_kernel(f"""
#define IDS {DATA:#x}
#define WEIGHTS {DATA + tr * 4096:#x}
#define LIST {DATA + tr * 6144:#x}
#define OUT {out:#x}
{ROWS}
// non-volatile reads of L1 that the NoC wrote. They're safe after the fence in noc_wait
static unsigned choice(unsigned r, unsigned j) {{ return ((const unsigned *)IDS)[r / 32 * 1024 + at(r % 32, j)]; }}
static unsigned weight(unsigned r, unsigned j) {{ return ((const unsigned short *)WEIGHTS)[r / 32 * 1024 + at(r % 32, j)]; }}
// the position of the lowest set bit of m. rv32im has no instruction to count zeros, so this uses a de Bruijn sequence
static unsigned lowest(unsigned m) {{
  static const unsigned char place[32] = {{0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8, 31, 27, 13, 23, 21, 19, 16, 7, 26,
                                           12, 18, 6, 11, 5, 10, 9}};
  return place[(m & -m) * 0x077CB531u >> 27];
}}
void kmain(volatile unsigned *a) {{
  unsigned *list = (unsigned *)LIST, *mark = list + {slots}, *out = (unsigned *)OUT;
  if (!a[1]) return;
  for (unsigned i = 0; i < {tr}; i++) {{
    unsigned u = a[6] + i * a[5], v = a[9] + i * a[8];
    noc_get(BANK(u), AT(a[4], u, 4096), IDS + i * 4096, 4096), noc_get(BANK(v), AT(a[7], v, 2048), WEIGHTS + i * 2048, 2048);
  }}
  copy({ZEROS:#x}, OUT, 4096);  // rows of a weights tile past the choices stay 0
  for (unsigned w = 0; w < {-(-n // 32)}; w++) mark[w] = 0;
  noc_wait();
  for (unsigned r = 0; r < {rows}; r++)
    for (unsigned j = 0; j < {k}; j++) if (weight(r, j)) {{ unsigned e = choice(r, j); mark[e / 32] |= 1u << e % 32; }}
  unsigned count = 0;
  for (unsigned w = 0; w < {-(-n // 32)}; w++) for (unsigned m = mark[w]; m; m &= m - 1) list[count++] = w * 32 + lowest(m);
  for (; count < {slots}; count++) list[count] = {n};
  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned s = a[0] + u, e = list[s];
    for (unsigned i = 0; i < {tr}; i++) {{  // tile row i of its weights: the weight of each row for e, across the row
      for (unsigned r = 0; r < 32 && i * 32 + r < {rows}; r++) {{
        unsigned w = 0;
        for (unsigned j = 0; j < {k}; j++) if (choice(i * 32 + r, j) == e) w = weight(i * 32 + r, j);
        for (unsigned *row = out + at(r, 0) / 2, c = 0; c < 8; c++) row[c] = row[128 + c] = w | w << 16;  // its runs in both faces
      }}
      unsigned t = s * {tr} + i;
      l1_settle(OUT);
      noc_write(OUT, BANK(t), AT(a[3], t, 2048), 2048), noc_wait();
    }}
    if (s % 32) continue;
    // the tile of the experts column for entries s to s + 31, each expert across its row
    for (unsigned r = 0; r < 32; r++) for (unsigned *row = out + at(r, 0), e = s + r < {slots} ? list[s + r] : {n}, c = 0; c < 16; c++)
      row[c] = row[256 + c] = e;
    l1_settle(OUT);
    noc_write(OUT, BANK(s / 32), AT(a[2], s / 32, 4096), 4096), noc_wait();
    copy({ZEROS:#x}, OUT, 4096), noc_wait();
  }}
}}""")

def touched(ids, n, weights):
  """List distinct experts selected with nonzero weight. ids is an int32 (rows, k) matrix with values
  below n; weights holds the corresponding bf16 routing weights. touched_kernel runs this on BRISC.
  Return:
  - experts: a column of min(rows * k, n) entries, sorted by expert ID and padded with n.
  - weights: bf16 (entries, rows, 1), with zero where a token didn't select that expert.

  Indexed matmul then runs each selected expert over the token rows. Padding entries select no
  matrix and return zeros. Weight the results and sum across experts."""
  if not (ids.dtype == "int32" and weights.dtype == "bf16" and ids.shape == weights.shape and ids.batch == 1 and ids.cols <= 32):
    raise ValueError(f"touched needs int32 ids and bf16 weights of one shape, at most 32 choices to a row: {ids.shape} {weights.shape}")
  touched_l1(ids.rows, ids.cols, n)
  if ids.rt is not weights.rt: raise ValueError("touched needs ids and weights on the same runtime")
  x, w, rows, k = ids.realize(), weights.realize(), ids.rows, ids.cols  # matrices, so there are no repeats to copy
  slots = min(rows * k, n)
  experts, out = Tensor(ids.rt, slots, 1, dtype="int32"), Tensor(ids.rt, rows, 1, batch=slots)
  experts.run(("touched", rows, k, n), lambda: touched_kernel(rows, k, n), experts.addr, out.addr, x.addr, x.stride, x.offset,
              w.addr, w.stride, w.offset, units=slots, name="touched")
  return experts, out

def mxfp4(blocks, scales):
  """Convert gpt-oss MXFP4 expert weights to bf16 bits exactly; each value has at most two significant bits.

  The values are 4 bits (sign, 2 bits of exponent, 1 of mantissa), two to a byte with the low nibble first. Blocks of 32
  share a power of 2, stored as a byte with 127 for 1. They come as a batch of (out x in) matrices, which are
  transposed for x @ w."""
  values = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)
  v = np.stack([values[blocks & 15], values[blocks >> 4]], -1).reshape(*scales.shape, 32)
  v = np.ldexp(v, scales.astype(np.int32)[..., None] - 127).reshape(*scales.shape[:-1], -1)
  return gpt2.bf16_bits(np.ascontiguousarray(np.swapaxes(v, -1, -2)))

def draft(ids, places, h, w, c, caches, slots=None, length=None, every=True):
  """Predict one token beyond ids using the multi-token prediction block at c.layers. h is the hidden state at
  places; ids contains the following token. Return logits for the token after ids.

  RMS-normalize the embedding and h, concatenate them for fc, then run the block, norm and output layer.
  Update caches[c.layers] at places. slots and length follow llama.forward. With every=False, return only
  the last position, as needed after a prompt chunk."""
  f, d, l = llama.fns(h), c.dim, c.layers
  e, h = llama.rms(f.rows(w["embed"], ids), w["mtp.pre_fc_norm_embedding"], c), llama.rms(h, w["mtp.pre_fc_norm_hidden"], c)
  fc = w["mtp.fc"]
  x = f.mm(e, fc[:d, :], None) + f.mm(h, fc[d:, :], None)
  rows, rotary, mask, _, live = llama.around(w, c, places, caches[l].shape[-2], slots, length)
  x = llama.layer(x, gpt2.block(w, f"layers.{l}."), c, rotary, mask, caches[l], rows, l, slots is None and x.shape[-2] > 1, live, slots=slots)
  own = "embed.T draft" in w  # a copy of the output layer for the draft, when the precision has a "draft head"
  last = x if every else f.rows(x, [x.shape[-2] - 1])
  return f.mm(llama.rms(last, w["mtp.norm"], c), w["embed.T draft" if own else "embed.T"],
              llama.precision_of(c, "draft head" if own else "head")[1], "fp32")
