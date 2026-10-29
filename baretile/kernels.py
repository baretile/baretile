# Compile tensor expressions using the instructions and programs in tensix.py.
# fused_kernel handles elementwise operations, matmuls, SFPU programs and reductions. It keeps
# intermediates in L1 and writes results to DRAM. The remaining kernels implement attention,
# data movement, topk and format conversion.
import collections
from baretile import bh
from baretile.runtime import DATA, ZEROS, compile_kernel
from baretile.tensix import (ACCS, B8, B8_SLOTS, BANKS, BFP, BINARY, CASTS, CC_OFF, CC_ON, CHUNKS, COMBINE, DEPTH, DST_HALF, FILLED, FLAGS_B,
                            FOLDS, FP32_OPS, FRAMES, INT32_OPS, ITEMSIZE, MATH_END, MAX_OPS, ONE, OUTS, PAIR, REPORT, ROWS, SEM_L1, SEM_PACKED,
                            SFPU, SFPU_START, UNPACKABLE, WALK, ZERO_SLOT, Loop, accumulating, across_rows, aim, allocate, bfp_b, brisc_program,
                            chunk_tiles, elementwise, ext, field, fill, fold, haloize, ins, kernel_config, mad, mask_columns, mask_rows_at,
                            math_begin, math_program, matmul, pack, pack_program, reconfigure, reduce_finish, reduce_step, reload, repack, setadc,
                            sfpu, sfp_bits, sfp_load, sfp_store, slot_base, step_unpack, table, tile_bytes, tile_of, tt, unpack, unpack_program,
                            walk)

def grid_parts(rt, w, kt, grid, base, more=0):
  """Choose L1 buffers for a grid matmul (mc) producing rt x w tiles with kt tiles along K. Slots below base
  are already in use.

  Return None if A's rows and B's three buffers fit below slot 256. Otherwise return (kp, q, acc): split K
  into kp-tile parts, process q columns per worker at a time, and accumulate the partial results in fp32
  from L1 address acc. Read A again for each group of q columns. Prefer parts of at least 4 tiles, reducing
  q if necessary, and reserve more slots for single inputs. Return False if the accumulators won't fit.

  Choose the layout independently of B's format, so a block-float B and a bf16 B containing the same values
  produce bitwise-identical results."""
  bp, bq = -(-rt // min(len(grid[1]), rt)), -(-w // min(len(grid[0]), w))
  if base + (bp + 3) * kt <= UNPACKABLE: return None
  for least in (min(4, kt - 1), 1):
    for q in range(bq, 0, -1):
      q = -(-bq // -(-bq // q))  # make the groups as even as possible
      acc = OUTS - bp * q * tile_bytes("fp32")
      for kp in range(min(kt - 1, B8_SLOTS), least - 1, -1):
        top = base + more + 3 * (bp + 1) * kp
        if top <= UNPACKABLE and slot_base("bf16") + top * tile_bytes("bf16") + 3 * kp * tile_bytes("bfp8") <= acc:
          return -(-kt // -(-kt // kp)), q, acc  # make the parts as even as possible
  return False

def fused_kernel(expr, shapes, dtype="bf16", out=None, grid=None, reps=None, readers=None):
  """Compile an expression into one kernel. Return (image, work units).

  Expressions use nested tuples. An integer refers to an input by its index in shapes:
  - (op, x, y) for add, sub, mul and binary SFPU operations.
  - (op, x) for unary operations, cast and reductions in FOLDS (sum0, max2, ...).
  - ("matmul", a, b[, phases]) for matmul, with four fidelity phases by default. a is an input.
    b is an input or (mode, input[, ids]), where mode controls how B is read:
    - T transposes B during unpack.
    - bfp8 and bfp4 convert block floats to bf16 in SrcA. Tbfp8 and Tbfp4 also transpose B.
    - C, Cbfp8 and Cbfp4 read the layout produced by Tensor.columns; strides describe column runs.
    - Prefix any mode with I to select B's matrices using int32 ids. For example, ("Ibfp4", x, ids).
      I alone uses ordinary bf16 storage. With k columns in ids, result batch entry b selects
      x[ids[b // k, b % k]]. An index past x's batch returns zeros without reading a matrix.
  For example, ("add", ("matmul", 0, ("bfp8", 1)), 2) multiplies inputs 0 and 1, with input 1 in bfp8,
  then adds input 2. ("sum2", ("mul", 0, 0)) sums the squares in each row of input 0.

  shapes lists (batch, rows, columns) per input; dimensions of size 1 broadcast. dtype sets the input
  and computation format. out sets the final storage format, allowing a cast or an fp32 matmul result.
  grid and readers come from Runtime: grid enables multicast sharing, and readers assigns DRAM banks
  and ports to workers. reps gives each input's batch repeat count, default 1; matrix b reads b // rep.

  Kernel arguments:
  - a[0:2]: first work unit and unit count.
  - a[2:6]: output DRAM address, batch tile stride, row tile stride and tile offset.
  - a[6 + 4*i:10 + 4*i]: the same four values for input i.
  - Final word: valid rows in the last output tile row, with 0 meaning 32. Zero the remaining rows.
    Passing this count at runtime lets one compiled kernel handle several row counts.

  A work unit is normally one output tile. With one matmul and no reduction, it is a block of up to
  4 x 4 tiles, or a worker's block on the grid. With a reduction, it is a line of tiles along that axis:
  a row, a column, or the same tile position across the batch.

  bf16 arithmetic uses SrcA, SrcB and the FPU. fp32 and int32 elementwise kernels bypass the 19-bit
  Src registers: the math thread loads operands into Dst, and the SFPU computes every operation.
  These kernels contain no matmul; Tensor handles fp32 matmul through bf16 parts.

  Two L1 frames alternate between tiles. Each holds input slots and intermediate results. Packing
  an intermediate in dtype rounds it as a separate kernel would. An SFPU operation can instead run
  in place on an unshared result in Dst, before that rounding. Other unary operations load their
  operand by adding the zero tile.

  A matmul sums pairs of tiles along K. B streams through DEPTH slots with up to DEPTH reads in
  flight. Keep A's row in L1 when it fits; some decode layouts use space above the ordinary slots.
  Otherwise A streams too. With several tile rows, grid workers share inputs by multicast.
  If K must be split, grid_parts chooses the buffers and output groups; A is read again for each group.

  A reduction accumulates along a line in Dst, reduces within the tile with across_rows or
  across_cols, then packs its result. Dependent operations use that result on a later pass.
  Passes recompute intermediates, so softmax computes exp twice. Inputs reused across passes stay
  in L1 when they fit. Values only one tile long along the sweep, including reduction results,
  get separate slots and are computed once per line when first needed.

  All four programs follow the same loops over work units, passes and tiles. BRISC moves data;
  the three TRISCs handle unpack, math and pack."""
  uses, seen_exprs = {}, set()
  def count(e):  # how many ops read each subexpression
    if isinstance(e, int) or e in seen_exprs: return
    seen_exprs.add(e)
    for c in (() if e[0] == "matmul" else e[1:]):
      uses[c] = uses.get(c, 0) + 1
      count(c)
  count(expr)
  out, dst = out or dtype, dtype != "bf16"  # dst: the math thread fills Dst from L1 and the SFPU does all the math
  # ops is a list of dicts with kind, x, y, posts and shape. A matmul's also has ids, bt, wt, bc, kt and phases.
  # refs maps each subexpression to its input number or ("op", j).
  ops, refs = [], {}
  size = lambda r: shapes[r] if isinstance(r, int) else ops[r[1]]["shape"]  # (batch, rows, columns)
  def visit(e):
    if isinstance(e, int): return e
    if e in refs: return refs[e]
    if e[0] == "matmul":  # b is an input, or (how, x[, ids])
      how, y, *ids = e[2] if isinstance(e[2], tuple) else ("", e[2])
      how, ids = how.removeprefix("I"), ids[0] if ids else None
      bc, how = how.startswith("C"), how.removeprefix("C")  # bc: B is laid out by columns (Tensor.columns)
      a, bt, wt = shapes[e[1]], how.startswith("T"), how.removeprefix("T") or None
      batch = shapes[ids][1] * shapes[ids][2] if ids is not None else max(a[0], shapes[y][0])
      op = dict(kind="matmul", x=e[1], y=y, ids=ids, bt=bt, wt=wt, bc=bc, kt=ext(a, 2), shape=(batch, a[1], shapes[y][1 if bt else 2]),
                phases=e[3] if len(e) > 3 else 4)
    elif e[0] in COMBINE:
      x = visit(e[1])
      op = dict(kind=e[0], x=x, shape=tuple(1 if ax == FOLDS[e[0]] else s for ax, s in enumerate(size(x))))
    elif e[0] == "cast" or e[0] in SFPU and e[0] not in BINARY:
      # A unary op, or the root's cast. That packs as out, after an SFPU program between floats and ints if needed.
      x = visit(e[1])
      posts = (([CASTS[dtype, out]] if (dtype, out) in CASTS else []) if e[0] == "cast" else
               [FP32_OPS.get(e[0], e[0]) if dtype == "fp32" else e[0]])
      # Run in place on the result of the op below. A bf16 cast rounds that result first, as its own pack would.
      if isinstance(x, tuple) and uses[e[1]] == 1:
        ops[x[1]]["posts"] += ["ftob"] * (e[0] == "cast" and dtype == "bf16") + posts
        refs[e] = x
        return x
      op = dict(kind="add", x=x, y="zero", shape=size(x), posts=posts)  # x + 0 into Dst, then the SFPU
    else:
      x, y = visit(e[1]), visit(e[2])
      op = dict(kind=e[0], x=x, y=y, shape=tuple(map(max, size(x), size(y))))
    ops.append({"posts": [], **op})
    refs[e] = ("op", len(ops) - 1)
    return refs[e]
  root = visit(expr)[1]
  # If the root packs as another dtype, the packer switches for the root's pack and back. A lone op needs no switch.
  cfg, root_cfg = kernel_config(dtype, out if len(ops) == 1 else dtype), kernel_config(dtype, out)
  around = (tt(repack(cfg, root_cfg)), tt(repack(root_cfg, cfg))) if out != dtype and len(ops) > 1 else ("", "")
  repacked = lambda j, words: f"{around[0]} {tt(words)} {around[1]}" if j == root else tt(words)  # op j's pack. The root's packs as out
  operands = lambda op: [r for r in (op.get("x"), op.get("y")) if r not in (None, "zero")] if op["kind"] != "matmul" else []
  reduced = lambda r: isinstance(r, tuple) and ops[r[1]]["kind"] in COMBINE
  axes = {FOLDS[op["kind"]] for op in ops if op["kind"] in COMBINE}
  assert len(axes) <= 1, "a kernel reduces along one axis"
  # passes sweep along axis. Without a reduction it's 2, and units are blocks of tiles.
  reducing, axis = bool(axes), next(iter(axes), 2)
  # n is the extent of the result along each axis. Along the reduced axis, that's how far the ops sweep. A matmul's A
  # can be wider, since its columns are K, which streams through the matmul and isn't swept.
  inputs = len(shapes)
  n = [max([ext(op["shape"], ax) for op in ops] + [ext(size(r), ax) for op in ops for r in operands(op)]) for ax in range(3)]
  nb, rt, w = n
  L = n[axis]  # the length of a sweep
  assert all(ext(size(r), axis) in (1, L) for op in ops for r in operands(op)), \
    "all operands along the reduction axis must broadcast or have the same tile count"
  swept = lambda r: r != "zero" and ext(size(r), axis) == L and not reduced(r)  # r varies along the sweep. If not, it's one tile a line
  assert len(ops) <= MAX_OPS, f"SEM_L1 counts at most {MAX_OPS} intermediates"
  # Passes. A swept op runs in the first pass after the reductions it depends on have finished. A narrow op is ready
  # then too.
  # ready is the first pass that can read a value. sweep is the pass in which a reduction sweeps its operand.
  ready, sweep = {}, {}
  for j, op in enumerate(ops):
    deps = [ready[r[1]] if isinstance(r, tuple) else 0 for r in operands(op)]
    if op["kind"] in COMBINE: sweep[j] = max(deps, default=0); ready[j] = sweep[j] + 1
    else: ready[j] = max(deps, default=0)
  last = sweep[root] if root in sweep else ready[root]
  ops_of = {"fp32": FP32_OPS, "int32": INT32_OPS}.get(dtype, {})
  sfpu_name = lambda op: ops_of.get(op["kind"], op["kind"] if dtype != "int32" else None)  # a binary op's SFPU program
  sfpu_op = lambda op: op["kind"] in BINARY or (dst and op.get("y", "zero") != "zero")
  assert dtype != "int32" or all(sfpu_name(op) for op in ops if sfpu_op(op)), "an int32 kernel can only add, subtract, compare and pick"
  slots = allocate({"mmT" if op["bt"] else "mm" for op in ops if op["kind"] == "matmul"} | {"mul" for op in ops if op["kind"] == "mul" and not dst} |
                   {u for op in ops for u in op["posts"]} | {op["kind"] for op in ops if op["kind"] in COMBINE} |
                   {sfpu_name(op) for op in ops if sfpu_op(op)})
  frame, matmuls = inputs + len(ops), [j for j, op in enumerate(ops) if op["kind"] == "matmul"]
  assert not (dst and matmuls), "only bf16 kernels have matmuls"
  base = FRAMES + 2 * frame + len(ops)  # the first slot past the frames and the narrow ops
  rows_at, top = [], base  # where each matmul keeps its row of A in L1, or None if it doesn't fit and A streams too
  if not (top <= UNPACKABLE and slot_base(dtype) + top * tile_bytes(dtype) <= OUTS):
    raise ValueError("too many tiles in a kernel")
  for j in matmuls:
    fits = top + ops[j]["kt"] <= UNPACKABLE
    rows_at.append(top if fits else None)
    top += ops[j]["kt"] if fits else 0
  me = ("op", root)
  rows, cols = size(me)[1:]  # the result's
  # With one matmul and no passes, a unit is a block of bp x bq result tiles, done column by column. The block's bp rows
  # of A and its current column of B stay in L1, and each is read from DRAM once a block. With a grid, workers take a
  # block each and share the tiles by multicast. mc is the grid's columns and rows in use.
  # If K is too long for that, it goes in parts (grid_parts). Workers sum their result tiles over the parts in L1
  # (tensix.accumulating). The last part starts each tile from its sum (tensix.reload) and runs the rest of the tree.
  mc, parts = None, None
  if (grid and len(matmuls) == 1 and ops[matmuls[0]]["ids"] is None and not reducing and nb == 1 and rt > 1 and
      ext(ops[matmuls[0]]["shape"], 2) == w):
    kt = ops[matmuls[0]]["kt"]
    bp, bq = -(-rt // min(len(grid[1]), rt)), -(-w // min(len(grid[0]), w))
    n_grid = (-(-w // bq), -(-rt // bp))
    parts = grid_parts(rt, w, kt, grid, base, inputs)
    if parts is not False and n_grid[0] * n_grid[1] > 1 and max(n_grid) <= FLAGS_B: mc = n_grid
    else: parts = None
  if parts:
    (kp, group, acc_at), n_parts = parts, -(-ops[matmuls[0]]["kt"] // parts[0])
  # bcast: with one row of tiles, as in a token's matmul, grid workers take blocks of the result's columns. The first
  # grid row reads A's tiles in shares and multicasts them to the grid, so A's row is read from DRAM once. No two
  # workers share B, so it streams as usual. In a batch, like a K cut into parts, each matrix gets its own grid rows,
  # and its blocks are padded out to fill them.
  # long: a row of A too long for the slots below 256, like the down projection of a decode step with a K of 448 tiles.
  # It stays in L1 above the slots the kernel uses, if it fits there. Unpacker 1 reaches it in regions of 256 slots,
  # and its base moves region to region between a tile's steps.
  # local: with one matrix whose width is a multiple of 8 tiles, and B neither transposed nor picked, tile column j of B is
  # in bank (j + B's tile offset) % 8 if its row stride is a multiple of 8. B by columns works at any width. Then grid
  # workers take the columns in the bank that Runtime.readers gives them, in turn with the bank's other readers. A
  # worker reads that bank through the port it was given, and other banks through port 0. local holds each worker's
  # port, bank, position among the bank's readers, and the number of readers the bank has.
  one = grid and len(matmuls) == 1 and not reducing and rt == 1 and ext(ops[matmuls[0]]["shape"], 2) == w
  # With one matmul and nothing reducing, a B laid out by columns streams beside A's row in chunks (tensix.chunk_tiles).
  # Each chunk is one read, or two for a block float, with CHUNKS in flight, through their own ring. ring is its size
  # in tiles, and chunked (below) is ring when B streams this way.
  ring = CHUNKS * chunk_tiles(ops[matmuls[0]]["wt"]) if len(matmuls) == 1 and ops[matmuls[0]]["bc"] and not reducing else 0
  stream = ring * tile_bytes(ops[matmuls[0]]["wt"] or "bf16") if ring else DEPTH * tile_bytes("bfp8")
  long = (one and rows_at[0] is None and ops[matmuls[0]]["ids"] is None and  # leave room for the singles and B's stream
          slot_base(dtype) + (top + inputs + ops[matmuls[0]]["kt"]) * tile_bytes(dtype) + stream <= OUTS)
  bcast, local = None, None
  if one and not mc and (rows_at[0] is not None or long) and nb <= len(grid[1]):
    per = len(grid[1]) // nb * len(grid[0])  # workers for each matrix
    bp, bq = 1, -(-w // min(per, w))
    blocks = -(-w // bq)
    gx_n = len(grid[0]) if nb > 1 else min(len(grid[0]), blocks)
    if nb * blocks > 1: bcast = (gx_n, -(-blocks // gx_n))
    mm, banks = ops[matmuls[0]], collections.Counter(b for b, _ in readers or ())
    if nb == 1 and (w % len(bh.DRAM) == 0 or mm["bc"]) and not mm["bt"] and mm["ids"] is None and len(banks) == len(bh.DRAM):
      local = [bh.noc_xy(*port) | b << 12 | [c for c, _ in readers[:u]].count(b) << 16 | banks[b] << 24 for u, (b, port) in enumerate(readers)]
      most = -(-w // len(bh.DRAM))  # columns in each bank
      bcast, bq = (len(grid[0]), len(grid[1])), -(-most // min(banks.values()))
  ours = table("reader", local) + "\n" if local else ""  # all four programs get the readers table, since they walk the same tiles
  long = long and bcast is not None  # without bcast, A streams like any row that doesn't fit
  # shared: a tile row of A that the matrices of a bigger batch all read, like a decode step's tokens going through the
  # experts that any of them picked. The grid's first row reads it once, in shares, and multicasts it to the grid.
  # Workers outside the grid read it from DRAM. All grid workers take part, so the batch needs at least as many units
  # as the grid has workers.
  shared = (grid and not mc and not bcast and len(matmuls) == 1 and not reducing and rt == 1 and nb > 1 and rows_at[0] is not None
            and size(ops[matmuls[0]]["x"])[0] == 1 and nb * w >= len(grid[0]) * len(grid[1]))
  if not mc and not bcast:
    bp = min(4, rt) if len(matmuls) == 1 and not reducing else 1
    while bp > 1 and (base + (bp + 1) * ops[matmuls[0]]["kt"] > UNPACKABLE or ops[matmuls[0]]["wt"] and ops[matmuls[0]]["kt"] > B8_SLOTS):
      bp -= 1
    bq = min(4, w) if bp > 1 else 1
  if bp > 1 or mc: rows_at = [base]  # A's bp rows, then B's column, or B's three buffers with mc
  kept_b = rows_at[0] + (3 * bp * kp if parts else bp * ops[matmuls[0]]["kt"]) if bp > 1 or mc else None  # with parts, past A's buffers
  # A block-float B goes to its own slots at B8, and unpacker 0 switches to them for its steps. With mc there's a region
  # for each of its three buffers. Otherwise there's one region, for a block's column or for the stream of B's tiles.
  chunked = ring if ring and not mc and kept_b is None and (rows_at[0] is not None or long) else 0
  b8 = (kp if parts else ops[matmuls[0]]["kt"] if kept_b is not None else chunked or DEPTH) if any(ops[j]["wt"] for j in matmuls) else 0
  wts = {ops[j]["wt"] for j in matmuls if ops[j]["wt"]}  # the block-float formats of B
  if len(wts) > 1: raise ValueError("a kernel takes one block-float format")
  wt = next(iter(wts), "bfp8")
  # the number of blocks down and across a matrix
  br, bc = (mc[1], len(grid[0])) if mc else (1, bcast[0] * bcast[1]) if bcast else (-(-rt // bp), -(-w // bq))
  decls, tail = "", ""  # BRISC's tables and statics, and the code it runs last
  if mc:
    (gx_n, gy_n), (xs, ys) = mc, grid
    decls = (table("row_rect", [bh.noc_rect((xs[0], y), (xs[gx_n - 1], y)) for y in ys[:gy_n]]) +
             table("col_rect", [bh.noc_rect((x, ys[0]), (x, ys[gy_n - 1])) for x in xs[:gx_n]])) + "\n"
    tail = f"unflag({gx_n}, {gy_n});"
  part_code = [("", "")] * 4  # with parts, the code each program runs before and after a part
  if parts:
    decls += "static unsigned part_end[3];  // events at the end of each of the last three parts, for A's buffers\n"
    accs = [accumulating(acc_at, False), accumulating(acc_at, True)]  # the packer writing the accumulators, then adding onto them
    ends = [ins("STALLWAIT", stall_res=bh.STALL_SYNC, wait_res=bh.WAIT_PACK), ins("SEMPOST", sem_sel=SEM_L1)]
    part_code = [("", "part_end[n % 3] = events;"),  # BRISC notes when its unpacker is past each part
                 # TRISC0 drains before the last part, so the earlier parts are all in the accumulators. When the
                 # worker has no columns in a group, TRISC0 relays its pending reports. BRISC waits on them before it
                 # sends the group's A rows, and no tile's wait would relay them now.
                 (f"if (!qv) {{ tensix_idle(); relay(); }} if (p == {n_parts - 1}) {{ posted++; drain(); }}", ""),
                 ("", ""),
                 (f"if (p == 0) {{ {tt(repack(cfg, accs[0]))} }} " + (f"if (p == 1) {{ {tt(repack(accs[0], accs[1]))} }} " if n_parts > 2 else "") +
                  f"if (p == {n_parts - 1}) {{ {tt(repack(accs[n_parts > 2], cfg))} }}", f"if (p == {n_parts - 2}) {{ {tt(ends)} }}")]
  def plan(p):  # pass p's reductions, the swept ops it computes (operands first) and the narrow ops it computes once
    reductions, swept_ops = [j for j in sweep if sweep[j] == p], []
    def need(r):
      if isinstance(r, tuple) and swept(r) and r[1] not in swept_ops:
        for o in operands(ops[r[1]]): need(o)
        swept_ops.append(r[1])
    for r in [ops[j]["x"] for j in reductions] + ([me] if p == last and swept(me) else []): need(r)
    return reductions, swept_ops, [j for j, op in enumerate(ops) if not swept(("op", j)) and op["kind"] not in COMBINE and ready[j] == p]
  plans = [plan(p) for p in range(last + 1)]
  reads = [{r for j in sum(pl, []) for r in operands(ops[j]) if isinstance(r, int)} for pl in plans]  # the inputs each pass reads
  # In passes, an input that is one tile along the sweep gets its own slot and is fetched once a line. An input read in
  # several passes keeps its line of tiles in L1, fetched in the first pass, if they fit below slot 256. Neither is
  # read from DRAM twice.
  kept, passes_of, narrow = {}, lambda r: [p for p in range(last + 1) if r in reads[p]], lambda r: ext(shapes[r], axis) == 1
  room = min(UNPACKABLE, (OUTS - slot_base(dtype)) // tile_bytes(dtype))
  for r in range(inputs) if reducing else ():
    if narrow(r) and passes_of(r) and top < room: kept[r], top = top, top + 1
    elif not narrow(r) and len(passes_of(r)) > 1 and top + L <= room: kept[r], top = top, top + L
  # the last pass that reads a kept input: of the narrow inputs if keep is True, of the others if not
  last_read = lambda keep: max([passes_of(r)[-1] for r in kept if narrow(r) == keep], default=None)
  # In passes, a matmul computed in several passes, like the scores under a softmax, keeps the line's tiles of B in L1
  # from the first pass, as a kept input does. At most one matmul gets this. B can't be a block float or picked by
  # indices, its row of A must be kept, and the tiles must fit.
  line_b, computed = None, [None]
  for j in matmuls:
    mine = [p for p, pl in enumerate(plans) if j in pl[1]]
    if (reducing and line_b is None and len(mine) > 1 and not ops[j]["wt"] and ops[j]["ids"] is None and rows_at[matmuls.index(j)] is not None and
        top + L * ops[j]["kt"] <= UNPACKABLE): line_b, computed, lines_at, top = j, mine, top, top + L * ops[j]["kt"]
  if line_b is not None: decls += f"static unsigned line_done[{L}];\n"
  single = {}  # without passes, a scalar input also gets its own slot, fetched once a worker
  if kept_b is not None and parts: top = max(top, kept_b + (0 if b8 else 3) * kp)  # a block-float B goes past the slots
  elif kept_b is not None: top = max(top, kept_b + (3 if mc else 1) * ops[matmuls[0]]["kt"])  # past A's rows and B's buffers
  for r in range(inputs) if not reducing else ():
    if all(ext(shapes[r], ax) == 1 for ax in range(3)) and top < room: single[r], top = top, top + 1
  if long: rows_at, top = [top], top + ops[matmuls[0]]["kt"]  # past the singles, which stay below slot 256
  b8_at = slot_base(dtype) + top * tile_bytes(dtype) if long or parts and b8 else None  # a block-float B goes past it, or at B8
  if chunked and not ops[matmuls[0]]["wt"]: ring_slot, top = top, top + chunked  # a bf16 B's ring goes past it too
  if slot_base(dtype) + top * tile_bytes(dtype) > OUTS: raise ValueError("too many tiles in a kernel")
  # An input or swept op lives in the tile's frame (fr in the C is 0 or 1), unless it's kept. A narrow op has its own
  # slot.
  def slot(r):
    if r == "zero": return ZERO_SLOT
    if r in single: return single[r]
    if isinstance(r, int) and r in kept: return kept[r] if narrow(r) else f"{kept[r]} + k"
    if isinstance(r, int) or swept(r): return f"{FRAMES + (r if isinstance(r, int) else inputs + r[1])} + fr * {frame}"
    return FRAMES + 2 * frame + r[1]
  arg = lambda i: 6 + 4 * i  # input i's [address, batch stride, row stride, offset] in a, after the result's
  def which(r):  # the matrix of r that goes with matrix b of the result: b, b / rep for a repeat, or 0 for a batch of 1
    if size(r)[0] == 1: return "0"
    return f"b / {reps[r]}" if reps and isinstance(r, int) and reps[r] > 1 else "b"
  def matrix(r, base):  # the batch term of r's tile number
    return "" if size(r)[0] == 1 else f"{which(r)} * a[{base + 1}] + "
  # The tile of r that goes with b, i_ and j. r is an input or the result, its [address, strides, offset] at a[base].
  def where(r, base):
    s = size(r)
    i, j = "i_" if ext(s, 1) > 1 else 0, "j" if ext(s, 2) > 1 else 0
    return f"{matrix(r, base)}{i} * a[{base + 2}] + {j} + a[{base + 3}]"
  tile = lambda i: where(i, arg(i))
  B, U, M, P = range(4)  # the programs. BRISC moves tiles, and TRISC 0, 1 and 2 feed the unpack, math and pack threads
  # The math thread's last instructions for op j. In a tile at the edge, the root's padding in Dst is zeroed first.
  def math_end(j):
    if j != root: return tt(MATH_END)
    at_end = lambda ax: f"{('i_', 'j')[ax - 1]} == {ext(size(me), ax) - 1}"
    edge = [f"if ({at_end(2)}) {{ {tt(mask_columns('sum', cols % 32))} }}"] * (cols > 1 and cols % 32 > 0)
    edge += [f"if ({at_end(1)} && a[{arg(inputs)}]) {{ {mask_rows_at(f'a[{arg(inputs)}]')} }}"] * (rows > 1)  # from an argument
    return " ".join(edge + [tt(MATH_END)])
  def pslot(r):  # where the packer puts r: its slot, or a slot at OUTS for a result in another dtype
    return ("fr" if swept(me) else 0) if out != dtype and r == me else slot(r)
  def output(j):  # for the pack thread: where op j's result goes, if it's the root
    return f"output({'fr' if swept(me) else 0}, {pslot(me)}, {where(me, 2)}); " if j == root else ""
  # BRISC's fetch of input r in pass p. It goes into the frame, or into its kept slot in the first pass that reads it,
  # once the line before is done with that slot. A single input comes with the first tile, which tiles already counts.
  def fetch(p, r):
    if r in single: return f"if (tiles == 1) get(a[{arg(r)}], {tile(r)}, {single[r]});"
    if r not in kept: return f"get(a[{arg(r)}], {tile(r)}, {slot(r)});"
    if p != passes_of(r)[0]: return ""
    return (f"if (k == 0) {{ reached(narrow_done); get(a[{arg(r)}], {tile(r)}, {kept[r]}); }}" if narrow(r) else
            f"reached(kept_done[k]); get(a[{arg(r)}], {tile(r)}, {kept[r]} + k);")
  if kept: decls += "static unsigned narrow_done;  // events once the line before has last read a kept slot\n"
  if any(not narrow(r) for r in kept): decls += f"static unsigned kept_done[{L}];\n"
  picked = [ops[j]["ids"] for j in matmuls if ops[j]["ids"] is not None]  # the int32 tensor that picks B's matrices
  indices = ""
  if picked:
    (r,), k, ib = picked, shapes[picked[0]][2], tile_bytes("int32")
    it = -(-shapes[r][1] // 32)  # tiles of indices, kept in L1 below OUTS
    if not (shapes[r][0] == 1 and k <= 32 and OUTS - it * ib >= B8 + 3 * B8_SLOTS * tile_bytes("bfp8")):
      raise ValueError("indices must be one tile column below OUTS")
    indices = (f"#define IDS {OUTS - it * ib:#x}\nstatic unsigned ids(unsigned b) {{ unsigned t = b / {k}, j = b % {k};  // index (t, j)\n"
               f"  return ((volatile unsigned *)IDS)[t / 32 * 1024 + (t % 32 / 16 * 2 + j / 16) * 256 + t % 16 * 16 + j % 16]; }}\n")
  passes = []
  for p, (reductions, swept_ops, once) in enumerate(plans):
    c = [[], [], [], []]  # each program's C for a tile
    final = f"p == {n_parts - 1}" if parts else None  # with parts, the test that picks out the last part
    fetches = [fetch(p, r) for r in sorted(reads[p]) if r not in single]
    # Singles arrive with the first tile, even when the epilogue waits for the last K part.
    c[B] += (["reached(frame_done[fr]);"] + [fetch(p, r) for r in sorted(reads[p]) if r in single] +
             ([f"if ({final}) {{ {' '.join(fetches)} }}"] if parts and fetches else fetches) + ["ready();"])
    c[U].append("fetched();")
    # A picked matmul's math reads the indices, which are in L1 once the first frame is.
    if dst or picked: c[M].append("fetched();")
    for j in once + swept_ops:
      op, me_j = ops[j], ("op", j)
      cond, posted = "if (k == 0) " if j in once else "", "" if j == root else " posted++;"
      if parts and op["kind"] != "matmul": cond += f"if ({final}) "  # the rest of the tree works on the finished sums
      pk = repacked(j, pack("h", pslot(me_j), SEM_PACKED if j == root else SEM_L1))
      if parts and op["kind"] == "matmul":  # pack the sums so far onto the tile's accumulator, except in the last part
        c[P].append(f"{cond}{{ if ({final}) {{ {output(j)}{pk} }} else {{ {tt(pack('h', 'z'))} }} h ^= {DST_HALF}; }}")
      else: c[P].append(f"{cond}{{ {output(j)}{pk} h ^= {DST_HALF}; }}")
      if op["kind"] == "matmul":
        # bw is B's width in tiles, q says which matmul this is, and bfp whether B is a block float
        kt, bw, q, bfp = op["kt"], ext(op["shape"], 2), matmuls.index(j), 1 if op["wt"] else 0
        bt, bs = op["bt"], 0 if bfp else 1  # bs: B's first stream slot. A block float's stream slots are at B8
        x, y = arg(op["x"]), arg(op["y"])
        bj = 'j' if bw > 1 else 0  # B's tile (m, j), or for B transposed the input's tile (j, m)
        in_batch = f"ids(b) * a[{y + 1}] + " if op["ids"] is not None else matrix(op["y"], y)  # B's matrix: picked, or the batch's
        b_which = "ids(b)" if op["ids"] is not None else which(op["y"])  # which matrix. A column kept in L1 is tagged with it
        live = f"if (ids(b) < {shapes[op['y']][0]}) " if op["ids"] is not None else ""  # an entry that picks no matrix reads nothing: zeros
        km = "(k0 + m)" if parts else "m"  # a step's tile along K. With parts, it counts from the part's first tile
        b_tile = f"{in_batch}{bj} * a[{y + 2}] + {km} + a[{y + 3}]" if bt else f"{in_batch}{km} * a[{y + 2}] + {bj} + a[{y + 3}]"
        # B by columns. Tile column j's run is in bank j % 8. In each bank the runs start at a[y], with a[y + 1] runs a
        # matrix and a[y + 2] tiles a run. Tile m along K is at k_of(m) in its run. With K cut into a batch, each matrix
        # starts a[y + 3] tiles further along the same runs.
        if op["bc"]:
          run = f"a[{y}] + ({b_which} * a[{y + 1}] + {bj} / {len(bh.DRAM)}) * a[{y + 2}] * {tile_bytes(wt if bfp else dtype)}"
          k_of = lambda m: f"{b_which} * a[{y + 3}] + " + (f"(k0 + {m})" if parts else m)
          run_b = lambda m, slot, n: (f"getbc({bj}, {run}, a[{y + 2}], {k_of(m)}, {slot}, {n})" if bfp else
                                      f"getc({bj}, {run}, {k_of(m)}, {slot}, {n})")  # n tiles from m on
          fetch_b = lambda slot: run_b("m", slot, 1)
          step_b = lambda slot, a_slots: f"stepc({bfp}, {bj}, {run}, a[{y + 2}], {k_of('m')}, {slot}, {a_slots})"
        else:
          fetch_b = lambda slot: f"{'getb' if bfp else 'get'}(a[{y}], {b_tile}, {slot})"
          step_b = lambda slot, a_slots: f"step({bfp}, a[{y}], {b_tile}, {slot}, {a_slots})"
        # unpacker 0 transposes faces for a transposed B, and reads a block float
        on = tt(haloize(1) * bt + (bfp_b(wt, b8, base=b8_at) if bfp else []))
        off = tt(haloize(0) * bt + (bfp_b(wt, 0) if bfp else []))
        # K is how many steps a tile takes now, seq the column's position among all of B's columns (each part's), kb
        # the size of B's buffers, a_at the buffer for this tile's rows of A, and flag the value of their flags.
        K, seq, kb, a_at, flag = ("kn", "s", kp, f"{rows_at[q]} + n % 3 * {bp * kp}", "n + 1") if parts else (kt, "jj", kt, rows_at[q], 1)
        if mc and bfp:
          region_b = lambda r: tt(bfp_b(wt, b8, base=b8_at + r * b8 * tile_bytes(wt)) if parts else bfp_b(wt, b8, r))
          on = tt(haloize(1) * bt) + " " + " else ".join(f"if ({seq} % 3 == {r}) {{ {region_b(r)} }}" for r in range(3))
        a_tile = f"{matrix(op['x'], x)}i_ * a[{x + 2}] + {km} + a[{x + 3}]"
        buf = mc and f"{0 if bfp else kept_b} + {seq} % 3 * {kb}"  # with mc, B's buffer for column jj. A block float's is in B8
        # With mc, worker (gy, gx) of the grid computes a block. Workers in a grid row read their shares of the block's
        # rows of A and multicast them along the row. Workers in a grid column do the same with B's columns, down the
        # column, and column jj goes into B's buffer jj % 3. A share is flagged (FLAGS) once all the workers that need
        # it have it.
        if mc:
          gx_n, gy_n = mc
          gx, gy, a_n = f"unit % {bc}", f"unit / {bc}", f"(i0 + {bp} < {rt} ? {bp} : {rt} - i0) * {K}"
          part = lambda n, k, parts: (f"{k} * {n} / {parts}", f"({k} + 1) * {n} / {parts}")  # worker k's share of n tiles
          (a0, a1), (b0, b1) = part(a_n, gx, gx_n), part(K, gy, gy_n)
          get_a = (f"for (unsigned e = {a0}; e < {a1}; e++) "
                   f"{{ unsigned i_ = i0 + e / {K}, m = e % {K}; get(a[{x}], {a_tile}, {a_at} + e); }} "
                   f"share(SLOT({a_at} + {a0}), ({a1} - {a0}) * {tile_bytes(dtype)}, row_rect[{gy}], {gx_n - 1}, {gx}, {flag});")
          # With parts, A's rows go at the start of a part, from all the workers of the grid row, whether or not they
          # have columns in this group. Once this worker is done with part n - 2, all the workers of its grid row have
          # posted part n - 1. They're done with n - 3, which used A's buffer for part n. B's columns follow the same
          # rule.
          if parts:
            part_code[B] = (f"if ({gx} < {gx_n}) {{ if (n >= 2) reached(part_end[(n + 1) % 3]); {get_a} await(0, {gx_n}, {flag}); }}",
                            part_code[B][1])
          get_b = (f"{run_b(f'({b0})', f'{buf} + ({b0})', f'({b1}) - ({b0})')}; " if op["bc"] else  # by columns, the share's runs
                   f"for (unsigned m = {b0}; m < {b1}; m++) {fetch_b(f'{buf} + m')}; ")
          share_b = lambda l1, n: f"share({l1}, ({b1} - {b0}) * {n}, col_rect[{gx}], {gy_n - 1}, {FLAGS_B} + {gy}, {seq} + 1);"
          if bfp:  # into region jj % 3 of B8: the share's exponents, then its values
            region = f"{b8_at if parts else B8} + {seq} % 3 * {kb * tile_bytes(wt)}"
            get_b += (f"if ({gy_n - 1} && {b1} > {b0}) mcast({region} + 64 * ({b0}), ({b1} - {b0}) * 64, col_rect[{gx}], {gy_n - 1}); " +
                      share_b(f"{region} + {64 * kb} + {BFP[wt]} * ({b0})", BFP[wt]))
          else: get_b += share_b(f"SLOT({buf} + {b0})", tile_bytes(dtype))
          c[B].append("{ if (ii == 0) { "  # a column's first tile brings B's column, and in the first column A's rows
                      + ("" if parts else f"if (jj == 0) {{ {get_a} }} ") +
                      # column jj - 2 is done here, so this column's flag lets the others put jj + 1 in its buffer
                      f"if ({seq} >= 2) reached(col_end[({seq} + 1) % 3]); "
                      f"{get_b} await({FLAGS_B}, {gy_n}, {seq} + 1); " + ("" if parts else f"if (jj == 0) await(0, {gx_n}, {flag}); ") + "} "
                      f"for (unsigned m = 0; m < {K}; m++) kept_step(); if (ii + 1 == {bp} || i_ + 1 == {rt}) col_end[{seq} % 3] = events; }}")
        elif kept_b is not None:  # A's bp rows and B's column stay in L1
          b_slot, a_slots = f"nb ? {0 if bfp else kept_b} + m : KEPT", f"a[{x}], {a_tile}, na ? {rows_at[q]} + i_ % {bp} * {kt} + m : KEPT"
          # place 15 holds B's column, past A's rows at places 0 to bp - 1, and bp is at most 4 here
          c[B].append(f"{cond}{live}{{ unsigned na = fresh(i_ % {bp}, {which(op['x'])} * {rt} + i_), nb = fresh(15, {b_which} * {w} + j); "
                      f"for (unsigned m = 0; m < {kt}; m++) {step_b(b_slot, a_slots)}; tagged(0); }}")
        if kept_b is not None:  # unpacker 0 walks B's column and unpacker 1 the row of A
          column = 0 if bfp else buf if mc else kept_b  # a block-float B's column starts its region
          reload_u, posted_u = ("", posted) if not parts else (f"if ({final}) {{ {tt(reload(acc_at, 'z'))} }} ",  # the sum so far first
                                                                posted and f" if ({final}) {{{posted} }}")
          c[U].append(f"{cond}{{ {reload_u}{tt([setadc(bh.UNPACK1, f'{a_at} + i_ % {bp} * {K}'), setadc(bh.UNPACK0, column)])} "
                      f"{on} {live}for (unsigned m = 0; m < {K}; m++) {{ landed(); {tt(WALK)} }} {off}{posted_u} }}")
        elif rows_at[q] is None:  # A's tile streams in beside B's, each step
          c[B].append(f"{cond}{{ {live}for (unsigned m = 0; m < {kt}; m++) "
                      f"{step_b(f'{bs} + steps % {DEPTH}', f'a[{x}], {a_tile}, {1 + DEPTH} + steps % {DEPTH}')}; tagged(0); }}")
          c[U].append(f"{cond}{{ {on} {live}for (unsigned m = 0; m < {kt}; m++) {{ unsigned z = landed(); "
                      f"{tt(unpack(f'{bs} + z', f'{1 + DEPTH} + z') + REPORT)} }} {off}{posted} }}")
        # B's line stays in L1 across passes. The first pass that computes it fetches it once the line before is done.
        elif j == line_b:
          line = f"{lines_at} + k * {kt}"  # A's row is kept as below
          if p == computed[0]:
            c[B].append(f"{{ reached(line_done[k]); unsigned new = fresh({q}, {which(op['x'])} * {rt} + i_); for (unsigned m = 0; m < {kt}; m++) "
                        f"{step_b(f'{line} + m', f'a[{x}], {a_tile}, new ? {rows_at[q]} + m : KEPT')}; tagged(0); }}")
          else: c[B].append(f"for (unsigned m = 0; m < {kt}; m++) kept_step();")
          c[U].append(f"{{ {tt([setadc(bh.UNPACK1, rows_at[q]), setadc(bh.UNPACK0, line)])} {on} "
                      f"for (unsigned m = 0; m < {kt}; m++) {{ landed(); {tt(WALK)} }} {off}{posted} }}")
        else:
          new = "0" if bcast or shared else f"fresh({q}, {which(op['x'])} * {rt} + i_)"  # with bcast or shared, A's row is in already
          # B's column comes n tiles a chunk, into part p of the ring, with A's tiles beside it if the row is new. A
          # block float's ring is at b8_at or B8, a bf16's at ring_slot.
          if chunked:
            n, first = chunk_tiles(op["wt"]), 0 if bfp else ring_slot
            c[B].append(f"{cond}{live}{{ unsigned new = {new}; for (unsigned m0 = 0; m0 < {kt}; m0 += {n}) {{ "
                        f"unsigned nk = {kt} - m0 < {n} ? {kt} - m0 : {n}, p = chunk_part(); {run_b('m0', f'{first} + p * {n}', 'nk')}; "
                        f"if (new) for (unsigned m = m0; m < m0 + nk; m++) get(a[{x}], {a_tile}, {rows_at[q]} + m); chunk_issued(p, nk); }} "
                        "tagged(0); }")
            steps_u = lambda lo, hi: f"for (unsigned m = {lo}; m < {hi}; m++) {{ unsigned z = landed_chunk(m, {n}); {tt(step_unpack('z'))} }}"
            if not bfp:  # move unpacker 0 to the ring for B's steps
              at = lambda slot: tt(reconfigure(SEC0_base=(slot_base(dtype) + slot * tile_bytes(dtype)) // 16 - 1))
              on, off = at(ring_slot), at(0)
          else:
            c[B].append(f"{cond}{live}{{ unsigned new = {new}; for (unsigned m = 0; m < {kt}; m++) "
                        f"{step_b(f'{bs} + steps % {DEPTH}', f'a[{x}], {a_tile}, new ? {rows_at[q]} + m : KEPT')}; tagged(0); }}")
            steps_u = lambda lo, hi: f"for (unsigned m = {lo}; m < {hi}; m++) {{ unsigned z = landed(); {tt(step_unpack(f'{bs} + z'))} }}"
          walk_a = f"{live}{steps_u(0, kt)}"
          # A's row goes region by region. Region r starts at slot 256 r, at Z 0. Unpacker 1's base moves to each
          # region, then back to the slots.
          if long:
            cut = [0] + [256 * r - rows_at[q] for r in range(1, -(-(rows_at[q] + kt) // 256))] + [kt]
            rebase = lambda r: tt(reconfigure(SEC1_base=(slot_base(dtype) + 256 * r * tile_bytes(dtype)) // 16 - 1) +
                                  [setadc(bh.UNPACK1, 0)] * (r > 0))
            walk_a = " ".join((rebase(r) if r else "") + " " + steps_u(lo, hi) for r, (lo, hi) in enumerate(zip(cut, cut[1:]))) + " " + rebase(0)
          c[U].append(f"{cond}{{ {tt([setadc(bh.UNPACK1, rows_at[q])])} {on} {walk_a} {off}{posted} }}")
        # Parts start from a zeroed Dst half, the last from the sum so far. The rest of the tree runs on the full sum.
        if parts:
          reloaded, zeroed = tt(math_begin(False) + elementwise("add", slots)), tt(math_begin(True))
          c[M].append(f"{cond}{{ if ({final}) {{ {reloaded} }} else {{ {zeroed} }} {tt(walk(True) * bt)} "
                      f"for (unsigned m = 0; m < kn; m++) {{ {tt(matmul(slots, op['phases'], bt))} }} {tt(walk(False) * bt)} "
                      f"if ({final}) {{ {tt(sfpu(op['posts'], slots))} {math_end(j)} }} else {{ {tt(MATH_END)} }} h ^= {DST_HALF}; }}")
        else: c[M].append(f"{cond}{{ {tt(math_begin(True) + walk(True) * bt)} "  # walk a transposed B, then set the modes back
                          f"{live}for (unsigned m = 0; m < {kt}; m++) {{ {tt(matmul(slots, op['phases'], bt))} }} "
                          f"{tt(walk(False) * bt + sfpu(op['posts'], slots))} {math_end(j)} h ^= {DST_HALF}; }}")
        continue
      drain = "drain(); " if any(isinstance(r, tuple) for r in operands(op)) else ""
      x, y = slot(op["x"]), slot(op["y"])
      prog = sfpu([sfpu_name(op)], slots) if sfpu_op(op) else []
      if dst:  # the math thread fills x into the op's Dst half and y 128 rows on, then runs the SFPU
        fills = fill(x, "h") + (fill(y, "h + 128") if op["y"] != "zero" else []) + FILLED
        math = tt(math_begin(False) + fills + prog + sfpu(op['posts'], slots))
        c[M].append(f"{cond}{{ {drain}{math} {math_end(j)} h ^= {DST_HALF};{posted} }}")
        continue
      if op["kind"] in BINARY:  # x and y each go into Dst for the SFPU: x in the op's half, y 128 rows on
        unpacks, fpu = unpack(x, ZERO_SLOT) + unpack(y, ZERO_SLOT), elementwise("add", slots) + elementwise("add", slots, dst=128)
      else: unpacks, fpu = unpack(x, y), elementwise(op["kind"], slots)
      c[U].append(f"{cond}{{ {drain}{tt(unpacks)}{posted} }}")
      math = tt(math_begin(op['kind'] == 'mul') + fpu + prog + sfpu(op['posts'], slots))
      c[M].append(f"{cond}{{ {math} {math_end(j)} h ^= {DST_HALF}; }}")
    for acc, j in enumerate(reductions):
      if acc >= len(ACCS): raise ValueError("at most two reductions in a pass")
      op, me_j, x, kind = ops[j], ("op", j), ops[j]["x"], ops[j]["kind"]
      drain = "drain(); " if isinstance(x, tuple) else ""
      edge = size(x)[axis] % 32 if axis and size(x)[axis] > 1 else 0  # the operand's rows or columns in its last tile, if cut short
      end = f"k == {L - 1}" if swept(x) else "k == 0"  # the line's last tile. A narrow operand is one tile a line
      src = slot(x) if dst else None
      def step(first):  # the last tile's padding must not count
        plain, cut = tt(reduce_step(kind, slots, acc, first, 0, src)), tt(reduce_step(kind, slots, acc, first, edge, src))
        return f"if ({end}) {{ {cut} }} else {{ {plain} }}" if edge else plain
      finish, posted = f'{tt(reduce_finish(kind, slots, acc, op["posts"]))} {math_end(j)}', "" if j == root else " posted++;"
      pk = repacked(j, pack(ACCS[acc][0], pslot(me_j), SEM_PACKED if j == root else SEM_L1, ACCS[acc][1]))
      unpacks = "" if dst else tt(unpack(slot(x), ZERO_SLOT))  # with dst, math fills Dst and nothing is unpacked
      mine = M if dst else U  # the thread that drains and counts
      if swept(x):  # the operand is swept along the line
        c[mine].append(f"{drain}{unpacks} if ({end}) {{{posted} }}")
        c[M].append(f"if (k == 0) {{ {step(True)} }} else {{ {step(False)} }} if ({end}) {{ {finish} }}")
      else:  # reduced in the first tile of the pass
        c[mine].append(f"if ({end}) {{ {drain}{unpacks}{posted} }}")
        c[M].append(f"if ({end}) {{ {step(True)} {finish} }}")
      c[P].append(f"if ({end}) {{ {output(j)}{pk} }}")
    c[M if dst else U].append(tt(REPORT))
    c[B].append("frame_done[fr] = ++events;" + " narrow_done = events;" * (p == last_read(True)) +
                " kept_done[k] = events;" * (p == last_read(False)) + " line_done[k] = events;" * (p == computed[-1]))
    passes.append((bool(swept_ops or reductions), c))
  # A unit's first column. With local, it's the first column in the worker's bank, plus 8 for each reader of that bank
  # before this one. Tile column j of B is in bank (j + B's tile offset) % 8, or in bank j % 8 for B by columns.
  col0 = f"unit % {bc} * {bq}"
  if local:
    first_b = "0" if ops[matmuls[0]]["bc"] else f"a[{arg(ops[matmuls[0]]['y']) + 3}]"
    col0 = f"((reader[unit] >> 12 & 7) - {first_b}) % {len(bh.DRAM)} + {len(bh.DRAM)} * (reader[unit] >> 16 & 255)"
  def program(c):  # the loops that all four programs share, around each program's part of a tile
    tile_c = lambda b: f"{{ unsigned fr = tiles++ & 1; {' '.join(b[c])} }}"  # the two frames take turns
    if not reducing:  # a unit is a block, done column by column, skipping tiles past the edge of the matrix
      cols = f"jj = g * {group}; jj < (g + 1) * {group} && jj < {bq}" if parts else f"jj = 0; jj < {bq}"  # with parts, a group's
      sz = (f", s = {n_parts} * (v < g * {group} ? v : g * {group}) + p * qv + jj - g * {group}, z = (jj - g * {group}) * {bp} + ii"
            if parts else "")
      tiles_c = f"""for (unsigned {cols}; jj++) for (unsigned ii = 0; ii < {bp}; ii++) {{
      unsigned i_ = i0 + ii, j = {f"j0 + {len(bh.DRAM)} * (reader[unit] >> 24) * jj" if local else "j0 + jj"}, k = 0{sz};
      if (i_ >= {rt} || j >= {w}) continue;
      {tile_c(passes[0][1])}
    }}"""
      # With parts, the block's columns go a group at a time, and each group goes through the parts of K in turn. v is
      # how many of the block's columns are inside the matrix, and qv how many are in this group. k0 is the part's first
      # tile along K, kn its number of tiles, and n its position among the parts of all groups. s is a column's position
      # among all the columns the block takes, which picks B's buffers and flags. z is a tile's accumulator. start and
      # end are the program's code before and after a part.
      if parts:
        kt = ops[matmuls[0]]["kt"]
        start, end = part_code[c]
        tiles_c = (f"unsigned v = j0 + {bq} <= {w} ? {bq} : {w} > j0 ? {w} - j0 : 0;\n"
                   f"    for (unsigned g = 0; g < {-(-bq // group)}; g++) for (unsigned p = 0; p < {n_parts}; p++) {{\n"
                   f"    unsigned k0 = p * {kp}, kn = k0 + {kp} < {kt} ? {kp} : {kt} - k0, n = g * {n_parts} + p, "
                   f"qv = v > g * {group} ? (v - g * {group} < {group} ? v - g * {group} : {group}) : 0; {start}\n    {tiles_c}\n    {end} }}")
      # With local, a unit's columns come from its worker's reader (col0), so the step to the next unit below would be
      # wrong. But a worker takes one unit, since the units are the grid's workers, and the bcast prologue relies on
      # it too.
      return f"""  unsigned unit = a[0], b = unit / {br * bc}, i0 = unit % {br * bc} / {bc} * {bp}, j0 = {col0};
  for (unsigned u = 0; u < a[1]; u++, unit++) {{
    {tiles_c}
    if ((j0 += {bq}) == {bc * bq}) {{ j0 = 0; if ((i0 += {bp}) == {br * bp}) {{ i0 = 0; b++; }} }}  // the next unit, without dividing
  }}"""
    index = ("b", "i_", "j")  # a tile's position: matrix, row, column. A unit fixes the two the sweep doesn't walk
    (outer, _), (inner, m) = [(index[ax], n[ax]) for ax in range(3) if ax != axis]
    sweeps = "\n".join(f"""    for (unsigned {index[axis]} = 0, k = 0; k < {L if full else 1}; {index[axis]}++, k++) {{
      {tile_c(b)}
    }}""" for full, b in passes)
    return f"""  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned unit = a[0] + u, {outer} = unit / {m}, {inner} = unit % {m};
{sweeps}
  }}"""
  total = f"a[1] * {L if swept(me) else 1}" if reducing else f"tiles / {n_parts}" if parts else "tiles"  # result tiles to write to DRAM
  first = ""
  # With bcast, a worker waits for A's row before its first tile. The first grid row of its matrix reads and sends it.
  if bcast:
    (gx_n, gy_n), (xs, ys), kt, x = bcast, grid, ops[matmuls[0]]["kt"], arg(ops[matmuls[0]]["x"])
    rects = [bh.noc_rect((xs[0], ys[g]), (xs[gx_n - 1], ys[g + gy_n - 1])) for g in range(0, nb * gy_n, gy_n)]  # each matrix's grid rows
    decls, tail = table("rect", rects) + "\n" + ours, f"unflag({gx_n}, 0);"
    in_batch = matrix(ops[matmuls[0]]["x"], x)
    near = (f"  for (unsigned i = 0; i < {len(bh.DRAM)}; i++) near[i] = bank[i];\n"
            "  near[reader[a[0]] >> 12 & 7] = reader[a[0]] & 0xFFF;  // its own bank through its reader's port\n") if local else ""
    first = (near + f"  {{ unsigned b = a[0] / {bc}, gx = a[0] % {gx_n}, e = gx * {kt} / {gx_n}, end = (gx + 1) * {kt} / {gx_n};\n"
             f"    if (a[0] % {bc} < {gx_n}) {{ for (unsigned m = e; m < end; m++) get(a[{x}], {in_batch}m + a[{x + 3}], {rows_at[0]} + m);\n"
             f"      share(SLOT({rows_at[0]} + e), (end - e) * {tile_bytes(dtype)}, rect[b], {bc - 1}, gx, 1); }}\n    await(0, {gx_n}, 1);\n  }}\n")
  # With shared, grid workers wait for A's row before their first tile. The grid's first row reads and sends it.
  if shared:
    (xs, ys), kt, x = grid, ops[matmuls[0]]["kt"], arg(ops[matmuls[0]]["x"])
    decls, tail = table("grid_x", xs) + table("grid_y", ys) + "\n", f"unflag({len(xs)}, 0);"
    get_a = lambda e, end: f"for (unsigned m = {e}; m < {end}; m++) get(a[{x}], m + a[{x + 3}], {rows_at[0]} + m);"
    first = (f"  {{ unsigned me = noc_me(), gx = {len(xs)}, gy = {len(ys)};\n"
             f"    for (unsigned i = 0; i < {len(xs)}; i++) if (grid_x[i] == (me & 63)) gx = i;\n"
             f"    for (unsigned i = 0; i < {len(ys)}; i++) if (grid_y[i] == me >> 6) gy = i;\n"
             f"    if (gy == {len(ys)}) {{ {get_a(0, kt)} noc_wait(); }}\n"
             f"    else {{ unsigned e = gx * {kt} / {len(xs)}, end = (gx + 1) * {kt} / {len(xs)};\n"
             f"      if (gy == 0) {{ {get_a('e', 'end')} share(SLOT({rows_at[0]} + e), (end - e) * {tile_bytes(dtype)}, "
             f"{bh.noc_rect((xs[0], ys[0]), (xs[-1], ys[-1]))}, {len(xs) * len(ys) - 1}, gx, 1); }}\n"
             f"      await(0, {len(xs)}, 1); }}\n  }}\n")
  # With picked, the tiles of indices come in first, 32 tokens each. They go past the slots and a block float's buffers,
  # in space the kernel doesn't otherwise use.
  if picked:
    first = (f"  for (unsigned i = 0; i < {it}; i++) {{ unsigned t = a[{arg(r) + 3}] + i * a[{arg(r) + 2}];  // the indices' tile row i\n"
             f"    noc_get(BANK(t), AT(a[{arg(r)}], t, {ib}), IDS + i * {ib}, {ib}); }}\n  noc_wait();\n" + first)
  brisc = brisc_program(total, first + program(B), dtype, out, decls + indices, tail, b8, wt, b8_at=b8_at, near=bool(local))
  image = compile_kernel(brisc, unpack_program(program(U), cfg, indices + ours), math_program(slots, program(M), dst, indices + ours),
                         pack_program(program(P), decls=ours))
  return image, (nb * rt * w // L if reducing else nb * br * bc)  # and the number of units in the result

def attention_l1(ht, R, wt, parts):
  """Lay out attention_kernel's L1 for a unit of R tile rows of queries.

  - R is at most 8. A row's largest scores take a quarter of a Dst accumulator tile, and Dst holds two.
  - fs is the slots in a frame: the masks, then the keys and values if they're bf16.
  - T is the output tiles for each row: p v, the sums, and with parts the largest scores.
  - Slot 1 holds a tile of ones and slot 2 a tile of -inf. The two frames come next, then each row's p and
    its queries. used is the first free slot after them.
  - From the top of L1 down: the output tiles at oacc, which are each row's p v and then its sums, in fp32. Then the
    unit's places, and at ranges the key ranges of two units. A key range is the first and last key tile, the first
    and end of the unmasked ones, and whether the part has none."""
  if ht < 2 or not 1 <= R <= 8 or parts < 1: raise ValueError("attention needs heads of 64 columns or more, 1 to 8 tile rows a unit, positive parts")
  fs, T = R + (0 if wt else 2 * ht), ht + 1 + (parts > 1)
  p_slot = 3 + 2 * fs
  oacc = bh.L1_SIZE - R * T * tile_bytes("fp32")
  places = oacc - R * tile_bytes("int32")
  used, ranges = p_slot + R + R * ht, places - 64
  if used > UNPACKABLE or ranges < B8 + 4 * ht * tile_bytes(wt or "bf16") or (wt and 2 * ht > B8_SLOTS):
    raise ValueError(f"attention head width {32 * ht} with {32 * R} query rows a unit does not fit in L1")
  return fs, T, (p_slot, p_slot + R), oacc, places, used, ranges

def attention_store(ht, R, wt, parts):
  """Reserve L1 space to retain keys between the two attention sweeps. Return the storage location and the
  number of key rows that fit.

  For bf16, use the first free slot. For block floats, return region bases. Each region holds complete key
  rows in at most 64 slots, the unpacker's limit. Allocate below B8 after the slots in use, and above the
  regions reserved for frames."""
  *_, used, top = attention_l1(ht, R, wt, parts)
  if not wt: return used, (UNPACKABLE - used) // ht
  per = 64 // ht  # key rows in a region
  size = per * ht * tile_bytes(wt)
  windows = ((slot_base("bf16") + used * tile_bytes("bf16") + 63) // 64 * 64, B8), (B8 + 2 * 2 * ht * tile_bytes(wt), top)
  bases = [b for lo, hi in windows for b in range(lo, hi - size + 1, size)]
  return bases, len(bases) * per

def attention_parts(ht, R, wt, kt, units, workers):
  """Choose how many workers share a unit's kt key tiles, for a decode step's few units over a long cache.

  Use up to 8 parts with at least 8 key tiles per part, limited by the available workers. Combining parts costs
  additional launches. If the keys still exceed attention_store's capacity, increase the number of parts, up to 8."""
  room = lambda P: attention_store(ht, R, wt, P)[1]
  try: split = room(2)
  except ValueError: return 1  # a part's extra tile of maxima doesn't fit
  P = max(1, min(8, kt // 8, workers // units))
  return max(P, min(8, -(-kt // split))) if split and -(-kt // P) > room(P) else P

def attention_plan(ht, ra, entries, wt, kt, workers, parts=None):
  """Plan Tensor.attend for entries matrices of ra tile rows of queries over kt key tiles, as (R, units, parts, kept):
  the tile rows a unit takes, the units, the parts that share each unit's keys and the key rows a part keeps in L1.

  A unit's rows share every key and value tile it reads, so a unit takes up to 8 tile rows, but fewer while there would
  be fewer units than workers, and no more than fit in L1. Keys are split into parts only for queries of one or two
  tile rows, a decode step's."""
  def fits(R):
    try: attention_l1(ht, R, wt, parts or 1)
    except ValueError: return False
    return True
  R = next((R for R in range(min(8, ra), 0, -1) if (R == 1 or entries * -(-ra // R) >= workers) and fits(R)), 1)
  units = entries * -(-ra // R)
  P = parts or (attention_parts(ht, R, wt, kt, units, workers) if ra <= 2 else 1)
  return R, units, P, attention_store(ht, R, wt, P)[1]

def attention_kernel(ht, ra, R, rep, wt=None, phases=4, window=0, parts=1, keep=False):
  """Compute weighted sums and normalizers for softmax(q k^T + mask) v, keeping scores on the chip.

  q is (E, rows, 32 ht), in ra tile rows. k and v are (E / rep, 32 kt, 32 ht), in bf16 or the block float wt as a KV cache, and each of
  their matrices serves rep heads of q. places is (E or 1, rows, 1) int32 and holds the last key that each row sees.
  With a window, the first key it sees is window - 1 before that. The mask hides all other keys with -inf.

  The kernel's arguments:
  - a[0] and a[1]: the first unit and the number of units.
  - a[2] to a[21]: the DRAM address, tile strides for batch and row, and tile offset of the result, q, k, v and places.
  - a[22] and a[23]: kt, the number of key tiles, and the rows of q.

  A unit is R tile rows of a head's queries, up to 8, and the rows share the key and value tiles they read. BRISC reads
  the places of its rows first. From them it works out which key tiles the rows reach, and builds mask tiles for the
  key tiles that some row sees part of. The kernel then sweeps those key tiles twice, a frame at a time. A sweep makes
  each tile of scores in Dst, starting from the mask or 0 and adding the queries times the key tiles, which are read
  transposed.
  - The first sweep makes each tile of scores in a scratch tile of Dst, folds it across its columns and folds that into
    the row's largest so far. A row's largest stays in fp32 in a quarter of a Dst accumulator tile, so two hold 8 rows.
  - The second sweep computes scores first, spreads the row's largest over the scratch tile and subtracts it with the
    SFPU, so exp gives p. It packs p into L1, then multiplies it
    by the values and by a tile of ones, which gives each row's sum of p. The packer adds each product onto its output
    tile in L1, in fp32.

  The result is (E, rows, 32 ht + 1) in fp32: each head's p v, followed by each row's sum of p. Divide
  p v by the sum to obtain softmax(...) v. Read value tiles once per unit. Key tiles are read on each
  sweep unless keep retains them in L1. Scores stay in L1.

  parts splits each unit's key tiles between that many workers, for a decode step's few units over a long cache. The
  result is then (E parts, rows, 32 ht + 33): each part's p v and sums, then a tile of its rows' largest scores, for
  the caller to combine. A part with no key tiles takes the unit's first key tile and hides all its keys with a mask
  tile of -inf. For a row with no visible keys, subtracting its -inf maximum instead produces -inf, so p is zero.
  The caller receives maxima in fp32, without rounding them through bf16.

  With keep, retain as many initial key rows as attention_store can hold between sweeps, so those rows
  are read from DRAM once per unit."""
  fs, T, (p_slot, q_row), oacc, places, _, ranges = attention_l1(ht, R, wt, parts)
  per = -(-ra // R)  # units per head
  ones, none, fr0 = 1, 2, 3  # tiles of ones and of -inf, made by the SFPU once a launch, then the frames
  # The slot of frame fr's mask tile for row r, and where the unpackers find its key tile m and value tile j. A block
  # float's are in the frame's region at B8. into() is where BRISC puts a tile: getb's region and slot.
  mask = lambda r: f"{fr0} + fr * {fs} + {r}"
  key = lambda m: f"{m}" if wt else f"{fr0} + fr * {fs} + {R} + {m}"
  val = lambda j: f"{ht} + {j}" if wt else f"{fr0} + fr * {fs} + {R + ht} + {j}"
  into = lambda z: f"fr * {2 * ht} + {z}" if wt else z
  # keep: a unit's first key rows stay in L1, the rows n = kt - k0 below room. A block float's row n is in region
  # n / rows_a, from slot n % rows_a * ht on. A bf16 row n starts at slot store + n * ht.
  if keep:
    store, room = attention_store(ht, R, wt, parts)
    rows_a = 64 // ht
  kept = f"(kt - k0 < {room})" if keep else "0"  # the key row is in the store
  cfg = kernel_config()
  first, onto = accumulating(oacc, False), accumulating(oacc, True)  # pack the output tiles in fp32, then add onto them
  slots = allocate({"mmT", "mm", "exp", "one"} | ({"none"} if parts > 1 else set()))
  # Row r's largest scores are a quarter of accumulator tile r / 4: the even columns of its left faces, moved on 16
  # rows to the right faces and 2 to the odd columns as r % 4 picks. A tile of scores is at S, past the halves.
  acc, S = ACCS[0][0], 2 * DST_HALF
  quarter = (0, 4, 8, 12, 32, 36, 40, 44)  # the SFPLOAD addresses of a tile's even columns in its left faces, 4 rows each
  largest = lambda b, aimed: f"{acc} + 64 * (r / 4) + r % 4 / 2 * 16 + r % 2 * 2 + {b} - {aimed}"  # relative to the aim

  each = lambda body: f"for (unsigned r = 0; r < nr; r++) {{ {body} }}"
  rng = f"(volatile unsigned *){ranges:#x} + (u & 1) * 8"
  read = f"asm volatile(\"fence\" ::: \"memory\"); volatile unsigned *g = {rng}; unsigned k0 = g[0], k1 = g[1], f0 = g[2], f1 = g[3], empty = g[4];"
  units = lambda start, pass0, between, pass1, end: f"""  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned w_ = (a[0] + u) / {parts}, part = (a[0] + u) % {parts}, e = w_ / {per}, i0 = w_ % {per} * {R}, nr = {ra} - i0 < {R} ? {ra} - i0 : {R};
    {start}
    for (unsigned kt = k0; kt <= k1; kt++) {{ unsigned fr = tiles++ & 1; {pass0} }}
    {between}
    for (unsigned kt = k0; kt <= k1; kt++) {{ unsigned fr = tiles++ & 1; {pass1} }}
    {end}
  }}"""
  partial = "(!empty && (kt < f0 || kt >= f1))"  # a key tile that some row sees part of. BRISC builds its mask tiles
  ob = f"e * {parts} + part" if parts > 1 else "e"  # the result's batch entry
  # For each unit, BRISC reads the places and works out the key tiles from them: k0 to k1, and the unmasked ones from f0
  # up to f1. With parts, it takes this part's share. It tells the TRISCs as if a frame were fetched.
  start = ("unsigned lo = ~0u, hi = 0; " +
           each(f"noc_get(BANK({tile_of(18, 'e', 'i0 + r', 0)}), AT(a[18], {tile_of(18, 'e', 'i0 + r', 0)}, 4096), {places:#x} + r * 4096, 4096);") +
           " noc_wait(); " +
           each(f"volatile unsigned *t = (volatile unsigned *)({places:#x} + r * 4096); "
                f"for (unsigned q = 0; q < 32 && (i0 + r) * 32 + q < a[23]; q++) "
                f"{{ unsigned p = t[at(q, 0)]; lo = p < lo ? p : lo; hi = p > hi ? p : hi; }}") +
           f" volatile unsigned *g = {rng}; unsigned k0 = {f'lo + 1 > {window} ? (lo + 1 - {window}) / 32 : 0' if window else '0'}, "
           f"k1 = hi / 32 < a[22] - 1 ? hi / 32 : a[22] - 1, "
           f"f0 = {f'hi + 1 > {window} ? (hi + 1 - {window} + 31) / 32 : 0' if window else '0'}, f1 = (lo + 1) / 32; " +
           (f"unsigned n_ = k1 - k0 + 1, a0 = k0 + n_ * part / {parts}, a1 = k0 + n_ * (part + 1) / {parts}, empty = a0 == a1; "
            "if (empty) k1 = k0; else k0 = a0, k1 = a1 - 1; " if parts > 1 else "unsigned empty = 0; ") +
           "g[0] = k0, g[1] = k1, g[2] = f0, g[3] = f1, g[4] = empty; asm volatile(\"fence\" ::: \"memory\"); ready(); ++events;")
  getk = "getb" if wt else "get"
  framed = f"for (unsigned m = 0; m < {ht}; m++) {getk}(a[10], {tile_of(10, f'e / {rep}', 'kt', 'm')}, {into(key('m'))});"  # into the frame
  if not keep: stored = ""
  elif wt: stored = (f"for (unsigned m = 0, n = kt - k0, z = n % {rows_a} * {ht}; m < {ht}; m++, z++) "
                     f"{{ unsigned t = {tile_of(10, f'e / {rep}', 'kt', 'm')}, at = AT(a[10], t, {tile_bytes(wt)}); "
                     f"noc_get(BANK(t), at, store[n / {rows_a}] + 64 * z, 64); "
                     f"noc_get(BANK(t), at + 64, store[n / {rows_a}] + {64 * rows_a * ht} + {BFP[wt]} * z, {BFP[wt]}); }}")
  else: stored = f"for (unsigned m = 0; m < {ht}; m++) get(a[10], {tile_of(10, f'e / {rep}', 'kt', 'm')}, {store} + (kt - k0) * {ht} + m);"
  keys = lambda values: f"if (!{kept}) {{ {framed} }}" + (f" else {{ {stored} }}" if not values else "")  # the first sweep fills the store
  fetch = lambda values: (f"reached(frame_done[fr]); if {partial} {{ " + each(f"masked({mask('r')}, r, kt);") + " } " + keys(values) +
                          (f" for (unsigned j = 0; j < {ht}; j++) "
                           f"{getk}(a[14], {tile_of(14, f'e / {rep}', 'kt', 'j')}, {into(val('j'))});" if values else "") +
                          f" l1_settle(SLOT({mask(0)})); ready(); frame_done[fr] = ++events;")
  queries = ("if (kt == k0) { reached(events); " +
             each(f"for (unsigned m = 0; m < {ht}; m++) get(a[6], {tile_of(6, 'e', 'i0 + r', 'm')}, {q_row} + r * {ht} + m);") + " }")
  brisc = units(start, f"{queries} {fetch(False)}", "", fetch(True), f"expect += nr * {T};")
  region = lambda words: f"if (fr) {{ {tt(words(1))} }} else {{ {tt(words(0))} }}"  # unpacker 0 at frame fr's block floats
  on_k, off_k = region(lambda f: haloize(1) + (bfp_b(wt, 2 * ht, f) if wt else [])), tt(haloize(0) + (bfp_b(wt, 0) if wt else []))
  first_key = key(0)
  if keep and wt:  # a stored row's region goes by a switch, since its base is in the instructions
    on_k = (f"if {kept} {{ {tt(haloize(1))} switch ((kt - k0) / {rows_a}) {{ " +
            " ".join(f"case {g}: {{ {tt(bfp_b(wt, rows_a * ht, base=b))} }} break;" for g, b in enumerate(store)) + f" }} }} else {{ {on_k} }}")
    first_key = f"{kept} ? (kt - k0) % {rows_a} * {ht} : {key(0)}"
  elif keep: first_key = f"{kept} ? {store} + (kt - k0) * {ht} : {key(0)}"
  on_v, off_v = (region(lambda f: bfp_b(wt, 2 * ht, f)), tt(bfp_b(wt, 0))) if wt else ("", "")
  # row r's queries and a tile row of keys, in pairs: a key tile into SrcA, a query tile into SrcB
  walk_k = f"{on_k} {tt([setadc(bh.UNPACK1, f'{q_row} + r * {ht}'), setadc(bh.UNPACK0, first_key)])} {tt([Loop(ht, PAIR)])} {off_k}"
  the_mask = f"empty ? {none} : {partial} ? {mask('r')} : {ZERO_SLOT}"
  unpacker = units(f"fetched(); {read} {tt(REPORT)}",
                   "fetched(); " + each(f"{tt(unpack(the_mask, ZERO_SLOT))} {walk_k}") + f" {tt(REPORT)}", "",
                   "fetched(); drain(); " + each(f"{tt(unpack(the_mask, ZERO_SLOT))} {walk_k} posted++;") + " drain(); "
                   f"{on_v} " + each(f"for (unsigned j = 0; j < {ht}; j++) {{ {tt(unpack(val('j'), f'{p_slot} + r'))} }}") + f" {off_v} " +
                   each(tt(unpack(ones, f"{p_slot} + r"))) + f" {tt(REPORT)}", "")
  told = "fetched(); " + read + " needed += 2 * (k1 - k0 + 1);"  # skip the unit's frames, since math and pack needn't wait for them
  scores = lambda op, row: (f"{tt(aim(row) + elementwise(op, slots) + walk(True))} "
                            f"for (unsigned m = 0; m < {ht}; m++) {{ {tt(matmul(slots, phases, True))} }} {tt(walk(False))}")
  begin = tt(math_begin(False))
  product = f"{tt(math_begin(True))} {tt(matmul(slots, phases))} {tt(MATH_END)} h ^= {DST_HALF};"  # p times a value tile, or times the ones
  # After across_rows, every lane of the scores at S holds its row's largest. The first key tile's is kept, later
  # ones are folded into it.
  kept_max = lambda first: tt(SFPU_START + [w for b in quarter for w in [
    sfp_load(0, b), *([] if first else [field(sfp_load(1), largest(b, S)), fold("max2", 0, 1)]), field(sfp_store(0, step=False), largest(b, S))]])
  # Row r's largest over the whole tile at `to`, relative to the aim at h
  spread = lambda to: tt(SFPU_START + [w for b in quarter for w in [
    field(sfp_load(0), largest(b, "h")), *(field(sfp_store(0, step=False), f"{to} + {b + d}") for d in (0, 2, 16, 18))]])
  # Subtract only after the full score is accumulated, using the fp32 maximum spread at S. An all-masked
  # row has maximum -inf: send it to exp as -inf rather than the NaN from -inf - -inf.
  subtract = tt(SFPU_START + [Loop(32, [sfp_load(), field(sfp_load(1), f"{S} - h"), mad(0, 1, ONE, 0, neg=1),
                                     *sfp_bits(2, 0xFF800000), ins("SFPXOR", vd=2, vc=1), CC_ON, ins("SFPSETCC", vc=2, mod1=6),
                                     *sfp_bits(0, 0xFF800000), CC_OFF, sfp_store(0)])])
  math = units(told,
               each(f"{scores('add', S)} {tt(across_rows('max2'))} if (kt == k0) {{ {kept_max(True)} }} else {{ {kept_max(False)} }}"),
               each(f"{begin} {spread('0')} {tt(MATH_END)} h ^= {DST_HALF};") if parts > 1 else "",  # each row's largest, for the packer
               each(f"{begin} {scores('add', 'h')} {spread(f'{S} - h')} {subtract} {tt(sfpu(['exp'], slots))} {tt(MATH_END)} h ^= {DST_HALF};") +
               " " + each(f"for (unsigned j = 0; j < {ht}; j++) {{ {product} }}") + " " + each(product), "")
  out = lambda j: f"r * {T} + {j}"  # row r's output tile j. Its sums are at j = ht and its largest at ht + 1
  largest_out = (each("while (peek(SENT) < outputs) relay(); " + tt(repack(cfg, first) + pack("h", out(ht + 1)) + repack(first, cfg)) +
                      f" h ^= {DST_HALF};") if parts > 1 else "")
  packed = lambda slot: tt(pack("h", slot, SEM_L1)) + f" h ^= {DST_HALF};"
  packer = units(f"needed++; while (peek(FETCHED) < needed) relay(); {read} needed += 2 * (k1 - k0 + 1);", "", largest_out,
                 each(packed(f"{p_slot} + r")) +
                 f" if (kt == k0) {{ while (peek(SENT) < outputs) relay(); {tt(repack(cfg, first))} }} else {{ {tt(repack(cfg, onto))} }} " +
                 each(f"for (unsigned j = 0; j < {ht}; j++) {{ {tt(pack('h', out('j')))} h ^= {DST_HALF}; }}") + " " +
                 each(f"{tt(pack('h', out(ht)))} h ^= {DST_HALF};") +
                 f" if (kt == k0) {{ {tt(repack(first, cfg))} }} else {{ {tt(repack(onto, cfg))} }}",
                 each(f"for (unsigned j = 0; j < {T}; j++) {{ while (peek(SENT) + 4 <= outputs) relay(); "  # write all output tiles
                      f"SYNC[RING + outputs % 4 * 2] = {out('j')}; SYNC[RING + outputs % 4 * 2 + 1] = {tile_of(2, ob, 'i0 + r', 'j')}; outputs++; "
                      f"{tt([ins('SEMPOST', sem_sel=SEM_PACKED)])} }}"))
  # The SFPU makes two tiles once a launch: ones for the sums, and -inf for the mask of an empty part.
  made = [("one", ones)] + [("none", none)] * (parts > 1)
  math = "".join(f"{tt(math_begin(False) + sfpu([p], slots) + MATH_END)} h ^= {DST_HALF};" for p, _ in made) + math
  packer = "".join(f"{tt(pack('h', z, SEM_L1))} h ^= {DST_HALF};" for _, z in made) + packer
  hide = 0xFF80  # bf16 -inf
  decls = "static unsigned expect;" + (table("store", store) if keep and wt else "") + f"""
// The mask of tile row r for key tile kt, from its places: -inf where a row doesn't see the key, else 0.
// Two keys go in a word.
static void masked(unsigned slot, unsigned r, unsigned kt) {{
  volatile unsigned *places = (volatile unsigned *)({places:#x} + r * 4096), *m = (volatile unsigned *)SLOT(slot);
  for (unsigned q = 0; q < 32; q++) {{
    int p = places[at(q, 0)], first = {f"p - {window - 1}" if window else "0"};
    for (unsigned c = 0; c < 32; c += 2) {{
      int j = kt * 32 + c;
      m[at(q, c) / 2] = (j > p || j < first ? {hide:#x} : 0) | (j + 1 > p || j + 1 < first ? {hide << 16:#x}u : 0);
    }}
  }}
}}"""
  return compile_kernel(brisc_program("expect", brisc, "bf16", "fp32", decls, b8=2 * ht if wt else 0, wt=wt or "bfp8", outs=oacc),
                        unpack_program(f"posted += {len(made)}; drain();" + unpacker, cfg), math_program(slots, math),
                        pack_program("static unsigned needed;" + packer))

def gather_kernel(dtype):
  """Copy selected tensor rows into a new tensor using BRISC.

  args: [first unit, units, out, input address, input tile stride, input tile offset,
  result tile columns, result rows, index address, index tile offset]. Indices form an int32 row.

  Each unit writes one result tile. A worker keeps its current index tile, then reads two runs of 16 values per selected
  row, one from each face. DRAM reads must preserve the low six address bits, so each run is fetched
  as the containing 64-byte block. Read all blocks for a tile together, then copy the values into place.
  A single-row result repeats that row throughout its tile using NoC copies within L1."""
  tb, item = tile_bytes(dtype), ITEMSIZE[dtype]
  return compile_kernel(f"""
#define SLOT {DATA:#x}
#define IDS {DATA + tb:#x}
#define BLOCKS {DATA + tb + 4096:#x}
{ROWS}
static unsigned id(unsigned r) {{ return ((volatile unsigned *)IDS)[at(0, r)]; }}  // index r of this tile's 32, in row 0
static unsigned run_of(unsigned n, unsigned f) {{ return at(n % 32, 16 * f) * {item}; }}  // row n's run in face f, in its tile, in bytes
void kmain(volatile unsigned *a) {{
  for (unsigned u = 0, held = ~0u; u < a[1]; u++) {{
    unsigned t = a[0] + u, i = t / a[6], j = t % a[6], rows = a[7] - i * 32 < 32 ? a[7] - i * 32 : 32, it = a[9] + i;  // it: the indices' tile
    if (it != held) noc_get(BANK(it), AT(a[8], it, {tile_bytes("int32")}), IDS, {tile_bytes("int32")}), held = it;
    if (rows < 32) copy({ZEROS:#x}, SLOT, {tb});  // full tiles overwrite every row; partial tiles keep zero padding
    noc_wait();
    for (unsigned r = 0; r < rows; r++) for (unsigned f = 0; f < 2; f++) {{  // row r's run in face f, through block 2 r + f of BLOCKS
      unsigned n = id(r), from = n / 32 * a[4] + j + a[5];
      noc_get(BANK(from), AT(a[3], from, {tb}) + run_of(n, f) / 64 * 64, BLOCKS + (2 * r + f) * 64, 64);
    }}
    noc_wait();
    for (unsigned r = 0; r < rows; r++) for (unsigned f = 0; f < 2; f++) {{
      volatile unsigned *to = (volatile unsigned *)(SLOT + run_of(r, f));
      volatile unsigned *by = (volatile unsigned *)(BLOCKS + (2 * r + f) * 64 + run_of(id(r), f) % 64);
      for (unsigned k = 0; k < {4 * item}; k++) to[k] = by[k];  // a run's 16 values, in words
    }}
    if (a[7] == 1) row_down(SLOT, {ITEMSIZE[dtype]});
    noc_write(SLOT, BANK(t), AT(a[2], t, {tb}), {tb});
    noc_wait();
  }}
}}""")

def copy_kernel(dtype):
  """Copy tiles unchanged between tensors or views of the same shape, including block-float storage.
  assign uses this for results that are already computed. Each BRISC unit copies one tile, overlapping
  its read with the previous tile's write.

  args: [first unit, units, output address, output batch stride, output row stride, output tile offset,
  input address, input batch stride, input row stride, input tile offset, tile rows, tile columns, input repeat]."""
  tb = tile_bytes(dtype)
  return compile_kernel(f"""
{BANKS}
void kmain(volatile unsigned *a) {{
  unsigned tr = a[10], tc = a[11];
  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned t = a[0] + u, b = t / (tr * tc), i = t / tc % tr, j = t % tc, slot = {DATA:#x} + u % 2 * {tb};
    unsigned from = b / a[12] * a[7] + i * a[8] + j + a[9], to = b * a[3] + i * a[4] + j + a[5];
    noc_get(BANK(from), AT(a[6], from, {tb}), slot, {tb});
    noc_wait();  // this waits for the write of the tile before too, from the other slot
    noc_write(slot, BANK(to), AT(a[2], to, {tb}), {tb});
  }}
  noc_wait();
}}""")

# C shared by put_kernel and bfp4_kernel
NIBBLE = """
// The sign and the 3 bits of bf16 b below exponent top, rounded to the nearest.
static unsigned nibble(unsigned b, unsigned top) {
  unsigned e = b >> 7 & 255, shift = top - e, m = (b & 127) | (e ? 128 : 0), q = shift < 4 ? (m + (16u << shift)) >> (shift + 5) : 0;
  if (q > 7) q = 7;
  return q ? (b >> 15 & 1) << 3 | q : 0;  // no sign on a zero: the unpacker reads -0 as -inf
}"""

def put_kernel(dtype):
  """Put rows of x into rows of a tensor, in place. It's the inverse of gather. BRISC does it alone.

  args: [first unit, units, the tensor's and x's address, tile strides for batch and row, and tile offset, then the
  address and tile offset of the int32 row that holds the places of the n rows, then n, and the tensor's tile columns
  and tile rows, then its logical row count].

  A unit is a row of x in a matrix and tile column of the tensor, so the work grows with the rows put, not with the
  size of the tensor. Workers read all the places, 32 to a tile, and chain the rows that land in each tile row, first
  to last. The unit of the first row to land in a tile row does the work. It reads the tensor's tile once, then the
  tile of x for each row. It copies each row's two runs, the left and right face, in order, so a later row to the
  same place wins. Then it writes the tile back. No two workers write the same tile. 32 rows that fill a tile in
  order, like a prompt's starting at a tile boundary, use their tile of x as it is. Places past the tensor's last tile
  row are skipped, and a place in the padding of its last tile row is written there.

  Into a block float, like a KV cache in bfp8 or bfp4, x is bf16 and each run of 16 rounds as a cast would. The run
  shares its largest exponent. bfp8 keeps 7 bits of magnitude rounded half up, as the packer does. bfp4 keeps 3 bits
  rounded to the nearest, as bfp4_kernel does. A 0 loses its sign, since the unpacker reads -0 as -inf."""
  bfp = dtype in BFP
  xd = "bf16" if bfp else dtype  # x's dtype
  tb, xb = tile_bytes(dtype), tile_bytes(xd)  # bytes in a tile of the tensor, of x
  copy = ("for (unsigned f = 0; f < 2; f++) pack(FROM + run_of(i, f), TILE, r, f);" if bfp else
          f"for (unsigned f = 0; f < 2; f++) for (unsigned k = 0; k < {4 * ITEMSIZE[xd]}; k++) "
          f"((volatile unsigned *)(TILE + run_of(r, f)))[k] = ((volatile unsigned *)(FROM + run_of(i, f)))[k];")
  copy = f"for (unsigned r = a[15] == 1 ? 0 : place[i], end = a[15] == 1 ? 32 : r + 1; r < end; r++) {{ {copy} }}"
  # value k of the run into the block-float tile: two to a byte for bfp4, 7 bits of magnitude rounded half up for bfp8
  rounded = ("if (k % 2) continue; ((volatile unsigned char *)tile)[64 + n * 8 + k / 2] = nibble(v[k], top) | nibble(v[k + 1], top) << 4;"
             if dtype == "bfp4" else
             "unsigned b = v[k], e = b >> 7 & 255, m = (b & 127) | (e ? 128 : 0), shift = top - e < 8 ? top - e : 8, "
             "q = (m + (1u << shift)) >> (shift + 1); if (q > 127) q = 127; "
             "((volatile unsigned char *)tile)[64 + n * 16 + k] = q ? (b >> 15) << 7 | q : 0;")
  as_is = (f"noc_get(BANK(from), AT(a[6], from, {xb}), FROM, {xb}); noc_wait(); "
           f"for (unsigned r = 0; r < 32; r++) for (unsigned f = 0; f < 2; f++) pack(FROM + run_of(r, f), TILE, r, f);" if bfp else
           f"noc_get(BANK(from), AT(a[6], from, {tb}), TILE, {tb}); noc_wait();")
  return compile_kernel(f"""
#define TILE {DATA:#x}
#define FROM {DATA + 4096:#x}
#define IDS {DATA + 8192:#x}
#define PLACES {DATA + 12288:#x}
{BANKS}
static unsigned id(unsigned r) {{ return ((volatile unsigned *)IDS)[at(0, r)]; }}  // place r of the 32, in row 0
static unsigned run_of(unsigned r, unsigned f) {{ return at(r % 32, 16 * f) * {ITEMSIZE[xd]}; }}  // row r's run in face f, in x, in bytes
{NIBBLE}
// Pack the 16 bf16 values at src into row r's run in face f of the block-float tile: its exponent, then its values.
static void pack(unsigned src, unsigned tile, unsigned r, unsigned f) {{
  volatile unsigned short *v = (volatile unsigned short *)src;
  unsigned top = 0, n = at(r % 32, 16 * f) / 16;  // which of the tile's 64 runs
  for (unsigned k = 0; k < 16; k++) {{ unsigned e = v[k] >> 7 & 255; if (e > top) top = e; }}
  ((volatile unsigned char *)tile)[n] = top;
  for (unsigned k = 0; k < 16; k++) {{
    {rounded}
  }}
}}
void kmain(volatile unsigned *a) {{
  unsigned n = a[12], ct = a[13], tr = a[14];
  // each row's place, the next row to land in its tile row (~0: none), and each tile row's first row
  volatile unsigned *place = (volatile unsigned *)PLACES, *next = place + n, *first = next + n;
  for (unsigned i = 0; i < n; i += 32) {{
    unsigned it = a[11] + i / 32;
    noc_get(BANK(it), AT(a[10], it, {tile_bytes("int32")}), IDS, {tile_bytes("int32")});
    noc_wait();
    for (unsigned k = 0; k < 32 && i + k < n; k++) place[i + k] = id(k);
  }}
  for (unsigned i = 0; i < tr; i++) first[i] = ~0u;
  for (unsigned i = n; i-- > 0;) if (place[i] / 32 < tr) {{ next[i] = first[place[i] / 32]; first[place[i] / 32] = i; }}
  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned unit = a[0] + u, b = unit / (n * ct), i0 = unit / ct % n, j = unit % ct, ti = place[i0] / 32;
    if (ti >= tr || first[ti] != i0) continue;  // the tile belongs to an earlier row's unit, or there's none
    unsigned to = b * a[3] + ti * a[4] + j + a[5], held = ~0u, whole = i0 % 32 == 0 && i0 + 32 <= n;
    for (unsigned k = 0; whole && k < 32; k++) whole = place[i0 + k] == 32 * ti + k;
    if (whole) {{ unsigned from = b * a[7] + i0 / 32 * a[8] + j + a[9]; {as_is} }}
    else noc_get(BANK(to), AT(a[2], to, {tb}), TILE, {tb});  // in flight with the first row's tile of x
    for (unsigned i = i0; i != ~0u; i = next[i]) {{
      if (whole && i < i0 + 32) continue;
      unsigned from = b * a[7] + i / 32 * a[8] + j + a[9];
      if (from != held) noc_get(BANK(from), AT(a[6], from, {xb}), FROM, {xb}), held = from;
      noc_wait();
      {copy}
    }}
    noc_write(TILE, BANK(to), AT(a[2], to, {tb}), {tb});
    noc_wait();
  }}
}}""")

# how many tiles of a part fit in topk's L1
def topk_room(dtype, k): return (OUTS - DATA - tile_bytes(dtype) - 4096 - 64 * k) // tile_bytes(dtype)

def topk_kernel(k, cols, dtype="bf16", parts=1, labels=False):
  """Find the k largest values in each row, and their columns. Of equal values, the first column wins.

  args: [first unit, units, the values' address, the indices' address, then x's address, tile strides for batch and
  row, tile offset and rows, then the labels' address, tile strides and offset].

  BRISC does it alone. A unit is a row of a tile row in a part of x, so a decode step's few rows spread over as many
  workers. With up to 8 rows in x, units read their row's runs. With more, a worker reads the part's tile row once for
  all its units in it. BRISC scans the row run by run and keeps the k best so far in a heap, with the worst at the
  root. Of equal values, the later column counts as worse. A better value replaces the root. Values compare by a key,
  which is the bf16 or fp32 bits as an unsigned integer, with negatives flipped and -0 as 0. Once the worst kept value
  has no sign bit, a compare of the raw bits as signed integers is enough. At the end the heap is sorted, best first.

  parts cuts each row into that many parts of cols columns. A split row needs k and cols to be multiples of 32. The parts'
  k best go side by side in the results, which are (batch, rows, parts k). Indices count columns across the row, not
  within the part. With labels, an index is read from an int32 tensor of x's shape at the picked column, as the 64
  bytes around it. Labels are for a topk over the best indices of the parts.

  Units write their row of each result tile, in both faces, with zeros past the k columns and past x's rows. Values are
  bf16 or fp32 and indices int32. The tile's 32 units write all of it between them, and no two write the same bytes. A
  dimension of 1 fills its tile with copies, as in any tensor. A k of 1 fills the row with its column. When x has one
  row, the tile row's first unit scans it and writes complete tiles, with the row copied down."""
  ct, kt, item, bits = -(-cols // 32), -(-k // 32), ITEMSIZE[dtype], 8 * ITEMSIZE[dtype]  # ct, kt: x's tiles and the results' in a part
  tb = tile_bytes(dtype)
  pick = "0" if k * parts == 1 else "j"  # the pick that result column j holds. A k of 1 in one part is copied across
  label = f"((volatile unsigned *)(LABELS + {pick} * 64))[at(s, col[{pick}] % 32) % 16]" if labels else f"p * {cols} + col[{pick}]"
  if not (parts == 1 or (k % 32 == 0 and cols % 32 == 0)): raise ValueError("with parts, k and cols must be multiples of 32")
  if ct > topk_room(dtype, k): raise ValueError("a part's tile row doesn't fit in L1")
  return compile_kernel(f"""
#define X {DATA:#x}
#define VALUES {DATA + ct * tb:#x}
#define IDS {DATA + ct * tb + tb:#x}
#define LABELS {DATA + ct * tb + tb + 4096:#x}
{ROWS}
typedef {"unsigned short" if item == 2 else "unsigned"} datum;
typedef {"short" if item == 2 else "int"} signed_datum;
// The k best so far, as a heap with the worst at the root (of equal keys, the later column).
// A better value replaces the root.
#define WORSE(a, ca, b, cb) ((a) < (b) || ((a) == (b) && (ca) > (cb)))
static void swap(unsigned *top, unsigned *col, unsigned i, unsigned j) {{
  unsigned t = top[i], c = col[i];
  top[i] = top[j], col[i] = col[j], top[j] = t, col[j] = c;
}}
static void down(unsigned *top, unsigned *col, unsigned i, unsigned n) {{  // sift i down, moving worse children up
  unsigned key = top[i], c = col[i];
  for (unsigned l; (l = 2 * i + 1) < n; ) {{
    unsigned m = l + 1 < n && WORSE(top[l + 1], col[l + 1], top[l], col[l]) ? l + 1 : l;
    if (!WORSE(top[m], col[m], key, c)) break;
    top[i] = top[m], col[i] = col[m], i = m;
  }}
  top[i] = key, col[i] = c;
}}
static unsigned key(unsigned v) {{  // a float's bits as an unsigned integer that sorts in the floats' order
  const unsigned sign = 1u << {bits - 1}, all = {(1 << bits) - 1:#x};
  return v == sign ? sign : v & sign ? all & ~v : v | sign;
}}
void kmain(volatile unsigned *a) {{
  datum *x = (datum *)X;
  volatile datum *values = (volatile datum *)VALUES;  // a result tile of each. This unit writes its row in it
  volatile unsigned *ids = (volatile unsigned *)IDS;
  unsigned rows = a[8], tr = (rows + 31) / 32, held = ~0u, top[{k}], col[{k}], kept = 0;
  for (unsigned u = 0; u < a[1]; u++) {{
    // row r of tile row i of x, in part p of matrix b. first is the part's first tile
    unsigned t = (a[0] + u) / 32, r = (a[0] + u) % 32, i = t % tr, p = t / tr % {parts}, b = t / tr / {parts};
    unsigned first = b * a[5] + i * a[6] + p * {ct} + a[7];
    unsigned n = rows == 1 ? 1 : rows - i * 32 < 32 ? rows - i * 32 : 32, s = rows == 1 ? 0 : r;  // x's rows here, and the row r holds
    unsigned face = at(r, 0);  // the row's run in the left face
    if (rows == 1 && r) continue;  // with one row of x, the tile row's first unit writes all its copies
    if (s < n) {{
      kept = 0;
      if (t != held || rows <= 8) {{  // up to 8 rows read their runs. More read the part's tile row, once a worker
        for (unsigned j = 0; j < {ct}; j++)
          if (rows <= 8) for (unsigned f = 0; f < 2; f++) {{
            unsigned o = at(s, 16 * f) * {item};
            noc_get(BANK(first + j), AT(a[4], first + j, {tb}) + o, X + j * {tb} + o, {16 * item});
          }}
          else noc_get(BANK(first + j), AT(a[4], first + j, {tb}), X + j * {tb}, {tb});
        noc_wait();
        held = t;
      }}
      const signed_datum *row = (const signed_datum *)x + at(s, 0);  // its run of 16 in face 0 of tile 0
      // last is the key of the worst value kept. While that value has no sign bit, raw holds its bits, and a later
      // value beats it if it's bigger as a signed integer. This costs one compare where the key costs four.
      unsigned last = 0;
      int raw = -1;
      for (unsigned c0 = 0; c0 < {cols}; c0 += 16) {{  // the row's run of 16 in each face of each tile, in order
        const signed_datum *run = row + c0 / 32 * 1024 + c0 % 32 / 16 * 256;
        if (raw >= 0 && c0 + 16 <= {cols}) {{  // skip a full run that's no better than the worst kept, with one branch
          int better = 0;
          for (unsigned c = 0; c < 16; c++) better |= run[c] > raw;
          if (!better) continue;
        }}
        for (unsigned c = c0, end = c0 + 16 < {cols} ? c0 + 16 : {cols}; c < end; c++) {{
          signed_datum d = run[c - c0];
          if (kept == {k} && (raw >= 0 ? d <= raw : key((datum)d) <= last)) continue;  // no better than the worst kept
          unsigned v = key((datum)d), i;
          if (kept < {k}) {{  // it moves up while it's worse than its parent
            for (top[i = kept++] = v, col[i] = c; i && WORSE(top[i], col[i], top[(i - 1) / 2], col[(i - 1) / 2]); i = (i - 1) / 2)
              swap(top, col, i, (i - 1) / 2);
          }} else top[0] = v, col[0] = c, down(top, col, 0, {k});  // it replaces the worst
          if (kept == {k}) last = top[0], raw = last >= {1 << (bits - 1):#x}u ? (int)(last & {(1 << (bits - 1)) - 1:#x}) : -1;
        }}
      }}
      for (unsigned n = kept; n > 1; n--) swap(top, col, 0, n - 1), down(top, col, 0, n - 1);  // the worst goes to the end in turn: best first
      {f"""for (unsigned j = 0; j < {k}; j++) {{  // each pick's label: the 64 bytes around it
        unsigned c = col[j], lt = b * a[10] + i * a[11] + c / 32 + a[12], o = at(s, c % 32) * 4;
        noc_get(BANK(lt), AT(a[9], lt, 4096) + o / 64 * 64, LABELS + j * 64, 64);
      }}
      noc_wait();""" if labels else ""}
    }}
    for (unsigned q = 0; q < {kt}; q++) {{  // the unit's row of each result tile: its picks there, then 0 past them
      for (unsigned c = 0; c < 32; c++) {{
        unsigned j = q * 32 + c, v = 0, id = 0;
        if (s < n && {pick} < {k}) v = x[col[{pick}] / 32 * 1024 + at(s, col[{pick}] % 32)], id = {label};
        values[at(r, c)] = v, ids[at(r, c)] = id;
      }}
      unsigned o = (b * tr + i) * {parts * kt} + p * {kt} + q;  // the result tile
      if (rows == 1) {{  // write the full tile, its row doubled down by copies inside L1
        row_down(VALUES, {item}), row_down(IDS, 4);
        noc_write(VALUES, BANK(o), AT(a[2], o, {tb}), {tb}), noc_write(IDS, BANK(o), AT(a[3], o, 4096), 4096);
      }}
      else for (unsigned h = 0; h < 2; h++) {{  // the row's runs in both faces, from the same places in L1
        unsigned d = face + h * 256;
        noc_write(VALUES + d * {item}, BANK(o), AT(a[2], o, {tb}) + d * {item}, {16 * item});
        noc_write(IDS + d * 4, BANK(o), AT(a[3], o, 4096) + d * 4, 64);
      }}
      noc_wait();
    }}
  }}
}}""")

# A copy across an Ethernet link keeps two slots of LINK_TILES tiles each at LINK_L1, in the Ethernet tile at each end.
LINK_L1, LINK_TILES = 0x20000, 4
def link_flags(dtype): return LINK_L1 + 2 * LINK_TILES * tile_bytes(dtype), 48  # the two flags and the credit, zeroed before a copy

def link_kernel(send, dtype):
  """Either end of a copy across an Ethernet link (Tensor.to), the sender's if send. BRISC runs it on one worker.

  args: [first unit, units, the tensor's address, its tiles, then x and y of this chip's Ethernet tile].

  The sender reads the tensor LINK_TILES tiles at a time and writes the chunks into two slots in turn, in the L1 of
  its Ethernet tile. The tile's TX queue writes the chunk to the same place in the peer, then a flag with the chunk's
  number + 1. The link delivers in order, so the chunk is there when its flag is. The receiver waits for each flag,
  writes the chunk into its tensor and sends back a credit: the number of chunks done. The sender waits for the credit
  before it fills a slot again, and for the last credit before it ends, so the next copy's flags and credit can start
  at 0. A word in another tile, like a queue register, a flag or the credit, is read and written over the NoC from an
  L1 address with the same low 4 bits, as the NoC requires."""
  tb, q = tile_bytes(dtype), lambda name: f"{bh.ETH_TXQ0 + bh.ETH_TXQ[name]:#x}"
  slot = LINK_TILES * tb
  body = f"""
    while (c >= 2 && peek(CREDIT) < c - 1) {{}}  // slot s is free once chunk c - 2 is done
    for (unsigned i = 0; i < n; i++) noc_get(BANK(t + i), AT(a[2], t + i, {tb}), BUF + i * {tb}, {tb});
    noc_wait();
    noc_write(BUF, ex, ey, SLOT(s), n * {tb}), noc_wait();
    link_write(SLOT(s), n * {tb});
    poke(FLAG(s), c + 1), link_write(FLAG(s), 16);""" if send else f"""
    while (peek(FLAG(s)) != c + 1) {{}}  // the chunk is here once its flag is
    noc_get(ex, ey, SLOT(s), BUF, n * {tb}), noc_wait();
    for (unsigned i = 0; i < n; i++) noc_write(BUF + i * {tb}, BANK(t + i), AT(a[2], t + i, {tb}), {tb});
    noc_wait();
    poke(CREDIT, c + 1), link_write(CREDIT, 16);"""
  # the sender ends once the last credit is back, so nothing of this copy is left on the link
  end = f"\n  while (peek(CREDIT) < (a[3] + {LINK_TILES - 1}) / {LINK_TILES}) {{}}" if send else ""
  return compile_kernel(f"""
#define BUF {DATA:#x}
#define WORD {DATA + slot:#x}
#define SLOT(s) ({LINK_L1:#x} + (s) * {slot})
#define FLAG(s) ({LINK_L1 + 2 * slot:#x} + 16 * (s))
#define CREDIT {LINK_L1 + 2 * slot + 32:#x}
{BANKS}
static unsigned ex, ey;  // this end's Ethernet tile
static unsigned peek(unsigned addr) {{
  unsigned at = WORD + (addr & 15);
  noc_get(ex, ey, addr, at, 4); noc_wait();
  return *(volatile unsigned *)at;
}}
static void poke(unsigned addr, unsigned v) {{
  unsigned at = WORD + 16 + (addr & 15);
  *(volatile unsigned *)at = v; noc_write(at, ex, ey, addr, 4); noc_wait();
}}
static void link_write(unsigned at, unsigned n) {{  // n bytes at `at` in the tile's L1 to the peer, with TX queue 0
  // read the busy bit after another of the queue's registers. Polled too soon after a send, it lies (BH-55)
  do peek({q("CMD")}); while (peek({q("STATUS")}) & {bh.ETH_BUSY:#x});
  poke({q("START")}, at), poke({q("SIZE")}, n), poke({q("DEST")}, at), poke({q("CMD")}, {bh.ETH_TT_L1_WRITE});
}}
void kmain(volatile unsigned *a) {{
  if (!a[1]) return;
  ex = a[4], ey = a[5];
  for (unsigned c = 0, t = 0; t < a[3]; c++, t += {LINK_TILES}) {{
    unsigned n = a[3] - t < {LINK_TILES} ? a[3] - t : {LINK_TILES}, s = c % 2;{body}
  }}{end}
}}""")

def readback_kernel(dtype, runs=False):
  """Copy a tensor's tiles into host memory (Runtime.back) for tobytes. Reading DRAM over the BARs would wait on PCIe
  for each load.

  args: [first unit, units, the first tile, the tensor's address, tile strides for batch and row, tile offset, tile
  columns, tiles a matrix, repeat, host address low and high, rows a tile].

  BRISC does it alone, a tile a unit, in order from the first. Tiles alternate between two L1 buffers, so a tile's read
  overlaps the previous tile's write. A tensor of fewer than 32 rows sends the rows of each face that hold values, back
  to back, and skips the padding. For a token's logits that's row 0 of faces 0 and 1. Block-float tiles go in full.

  With runs, the tensor is laid out by columns (Tensor.columns). Then the strides and offset describe its runs, and
  a block-float tile takes two reads."""
  n, item = tile_bytes(dtype), 0 if dtype in BFP else ITEMSIZE[dtype]
  place, get = "from = b / a[9] * a[4] + i * a[5] + j + a[6]", f"noc_get(BANK(from), AT(a[3], from, {n}), l1, {n});"
  if runs:  # tile m of tile column j's run, in bank j % 8, as columns_kernel lays it out
    place = f"run = a[3] + (b * a[4] + j / {len(bh.DRAM)}) * a[5] * {n}, m = b * a[6] + i"
    get = (f"noc_get(BANK(j), run + 64 * m, l1, 64); noc_get(BANK(j), run + 64 * a[5] + {BFP[dtype]} * m, l1 + 64, {BFP[dtype]});"
           if dtype in BFP else f"noc_get(BANK(j), run + m * {n}, l1, {n});")
  return compile_kernel(f"""
{BANKS}
void kmain(volatile unsigned *a) {{
  unsigned long long host = a[10] | (unsigned long long)a[11] << 32;
  unsigned v = a[12], sent = {f"v * 32 * {item}" if item else n};  // the rows sent, and their bytes
  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned k = a[0] + u, t = a[2] + k, b = t / a[8], i = t % a[8] / a[7], j = t % a[7], {place};
    unsigned l1 = {DATA:#x} + u % 2 * {n};
    unsigned long long to = host + (unsigned long long)k * sent;
    {get}
    noc_wait();  // this also waits for the writes out of this buffer two tiles ago
    if (sent == {n}) noc_write(l1, {bh.PCIE[0]}, {bh.PCIE[1]}, to, {n});
    else for (unsigned f = 0; f < 4; f++) {{  // faces 0 and 1 hold rows 0-15, 2 and 3 rows 16-31
      unsigned r = f < 2 ? (v < 16 ? v : 16) : (v > 16 ? v - 16 : 0);
      if (r) noc_write(l1 + f * {256 * item}, {bh.PCIE[0]}, {bh.PCIE[1]}, to, r * {16 * item});
      to += r * {16 * item};
    }}
  }}
  noc_wait();
}}""")

def upload_kernel():
  """Copy bytes staged in host memory (Runtime.back) into the 8 DRAM banks, for frombytes. Writing DRAM through the
  BARs takes a PCIe write for each word, with the TLB's strict ordering.

  args: [first unit, units, host address low and high, DRAM address, bytes for each bank, bytes a unit].

  The host stages the banks back to back, the same number of bytes for each, and they go to the same address in every
  bank. A unit is a piece of one bank. Pieces alternate between two L1 buffers, so a piece's read from the host
  overlaps the previous piece's write to DRAM."""
  return compile_kernel(f"""
{BANKS}
void kmain(volatile unsigned *a) {{
  unsigned long long host = a[2] | (unsigned long long)a[3] << 32;
  unsigned each = a[5], piece = a[6], pieces = (each + piece - 1) / piece;
  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned k = a[0] + u, b = k / pieces, at = k % pieces * piece, n = each - at < piece ? each - at : piece;
    unsigned l1 = {DATA:#x} + u % 2 * {bh.MAX_PACKET};
    noc_read({bh.PCIE[0]}, {bh.PCIE[1]}, host + (unsigned long long)b * each + at, l1, n);
    noc_wait();  // this also waits for the write out of this buffer two pieces ago
    noc_put(l1, BANK(b), a[4] + at, n);
  }}
  noc_wait();
}}""")

def zeros_kernel():
  """Write zeros over a tensor's storage in all 8 DRAM banks from the tile of zeros in each worker's L1 (ZEROS), so
  nothing crosses PCIe. All-zero bytes are zeros in every storage format, block floats included.

  args: [first unit, units, address, bytes in each bank]. A unit is 4 KiB of one bank."""
  return compile_kernel(f"""
{BANKS}
void kmain(volatile unsigned *a) {{
  unsigned pieces = (a[3] + 4095) / 4096;
  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned k = a[0] + u, b = k / pieces, at = k % pieces * 4096;
    noc_put({ZEROS:#x}, BANK(b), a[2] + at, a[3] - at < 4096 ? a[3] - at : 4096);
  }}
  noc_wait();
}}""")

def columns_kernel(dtype):
  """Store each tile column of a matrix or batch together, as used by Tensor.columns.
  BRISC copies one tile per work unit.

  args: [first unit, units, out, input address, input batch stride, input row stride, input tile offset,
  input tile columns, tiles per matrix, runs per matrix per bank, tiles per run].

  Tile column j forms a run in bank j % 8. Read each tile in full, then write it at its position in
  that run. For block floats, write exponents into the run's exponent region and values into its
  value region."""
  tb = tile_bytes(dtype)
  put = (f"noc_write(IN, BANK(j), run + 64 * i, 64); noc_write(IN + 64, BANK(j), run + 64 * a[10] + {BFP[dtype]} * i, {BFP[dtype]});"
         if dtype in BFP else f"noc_write(IN, BANK(j), run + i * {tb}, {tb});")
  return compile_kernel(f"""
#define IN {DATA:#x}
{BANKS}
void kmain(volatile unsigned *a) {{
  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned t = a[0] + u, b = t / a[8], i = t % a[8] / a[7], j = t % a[7], from = b * a[4] + i * a[5] + j + a[6];
    unsigned run = a[2] + (b * a[9] + j / {len(bh.DRAM)}) * a[10] * {tb};
    noc_get(BANK(from), AT(a[3], from, {tb}), IN, {tb});
    noc_wait();
    {put}
    noc_wait();
  }}
}}""")

def storage_kernel(dtype, runs=False):
  """Copy column runs or decode block floats into ordinary bf16 tiles, one tile per work unit.

  args: [first unit, units, out, input address, batch stride, row stride, tile offset, tile columns,
  tiles per matrix, repeat]. With runs, the three layout fields describe column runs, as in readback_kernel.
  Decode sign-magnitude values exactly as the unpacker does, including its negative-zero encoding of -inf."""
  n = tile_bytes(dtype)
  place = "from = b / a[9] * a[4] + i * a[5] + j + a[6]"
  get = f"noc_get(BANK(from), AT(a[3], from, {n}), IN, {n});"
  if runs:
    place = f"run = a[3] + (b / a[9] * a[4] + j / {len(bh.DRAM)}) * a[5] * {n}, m = b / a[9] * a[6] + i"
    get = (f"noc_get(BANK(j), run + 64 * m, IN, 64); noc_get(BANK(j), run + 64 * a[5] + {BFP[dtype]} * m, IN + 64, {BFP[dtype]});"
           if dtype in BFP else f"noc_get(BANK(j), run + m * {n}, IN, {n});")
  value = ("((volatile unsigned char *)(IN + 64))[q]" if dtype == "bfp8" else
           "(((volatile unsigned char *)(IN + 64))[q / 2] >> (4 * (q % 2)) & 15) << 4")
  decode = f"""
    for (unsigned k = 0; k < 1024; k += 2) {{
      unsigned word = 0;
      for (unsigned h = 0; h < 2; h++) {{
        unsigned q = k + h, x = {value}, e = ((volatile unsigned char *)IN)[q / 16], v = x >> 7 ? 0xff80 : 0;
        if (x & 127) {{
          unsigned m = (x & 127) << 1, shift = 0;
          while (!(m & 128)) {{ m <<= 1; shift++; }}
          v = (x >> 7) << 15 | ((e - shift) & 255) << 7 | (m & 126);
        }}
        word |= v << (16 * h);
      }}
      ((volatile unsigned *)OUT)[k / 2] = word;
    }}""" if dtype in BFP else ""
  return compile_kernel(f"""
#define IN {DATA:#x}
#define OUT {DATA + 2048:#x}
{BANKS}
void kmain(volatile unsigned *a) {{
  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned t = a[0] + u, b = t / a[8], i = t % a[8] / a[7], j = t % a[7], {place};
    {get}
    noc_wait();{decode}
    noc_write({"OUT" if dtype in BFP else "IN"}, BANK(t), AT(a[2], t, 2048), 2048);
    noc_wait();
  }}
}}""")

def bfp4_kernel():
  """Convert bf16 tiles to bfp4. BRISC does it alone, a tile a unit.

  args: [first unit, units, out, x's address, tile strides for batch and row, tile offset, x's tile columns, tiles a
  matrix].

  Each run of 16 values, a row of a face, shares its largest exponent. Each value keeps the 3 bits below it, rounded to
  the nearest with ties up, the same way Tenstorrent's host conversion rounds. The packer's own bfp4 is its bfp8 with
  the low 4 bits dropped, which rounds toward zero. This kernel reads and writes L1 a 32-bit word at a time."""
  return compile_kernel(f"""
#define IN {DATA:#x}
#define OUT {DATA + 2048:#x}
{BANKS}
{NIBBLE}
void kmain(volatile unsigned *a) {{
  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned t = a[0] + u, b = t / a[8], i = t % a[8] / a[7], j = t % a[7], from = b * a[4] + i * a[5] + j + a[6], tops = 0;
    noc_get(BANK(from), AT(a[3], from, {tile_bytes("bf16")}), IN, {tile_bytes("bf16")});
    noc_wait();
    for (unsigned r = 0; r < 64; r++) {{  // a run: 8 words of 2 values, its largest exponent, then 2 words of 8 nibbles
      unsigned v[8], top = 0;
      for (unsigned k = 0; k < 8; k++) {{
        v[k] = ((volatile unsigned *)(IN + 32 * r))[k];
        unsigned e0 = v[k] >> 7 & 255, e1 = v[k] >> 23 & 255;
        if (e0 > top) top = e0;
        if (e1 > top) top = e1;
      }}
      tops |= top << 8 * (r % 4);
      if (r % 4 == 3) {{ ((volatile unsigned *)OUT)[r / 4] = tops; tops = 0; }}
      for (unsigned w = 0; w < 2; w++) {{
        unsigned word = 0;
        for (unsigned k = 0; k < 4; k++) word |= (nibble(v[4 * w + k], top) | nibble(v[4 * w + k] >> 16, top) << 4) << 8 * k;
        ((volatile unsigned *)(OUT + 64 + 8 * r))[w] = word;
      }}
    }}
    noc_write(OUT, BANK(t), AT(a[2], t, {tile_bytes("bfp4")}), {tile_bytes("bfp4")});
    noc_wait();
  }}
}}""")

def transpose_kernel(rows, cols):
  """x.T for each (rows, cols) matrix x of a batch.

  args: [first unit, units, out, x's address, tile strides for batch and row, tile offset].

  Result tile (i, j) is tile (j, i) of x, transposed. BRISC fetches that tile with its off-diagonal faces swapped. Then
  unpacker 0 transposes each face on its way into SrcA, because Haloize_mode stays on in this kernel's config. x + 0
  through the FPU leaves the transpose in Dst. A matmul that reads B transposed also turns the mode on around its K
  loop (haloize), but reads B's tiles as they lie and picks their faces by address (matmul_phase)."""
  # ct is the result's tile columns. frame(f) gives the slots of x and of the result in frame f.
  ct, frame = -(-rows // 32), lambda f: (FRAMES + 2 * f, FRAMES + 2 * f + 1)
  per = ct * -(-cols // 32)  # tiles in a matrix
  def loop(*per_frame):  # the loop over result tiles t that all four programs share, with the two frames taking turns
    return f"""  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned t = a[0] + u;
    if (u & 1) {{ {per_frame[1]} }} else {{ {per_frame[0]} }}
  }}"""
  return compile_kernel(
    brisc_program("a[1]", loop(*[f"reached(frame_done[{f}]); flipped(a[3], t / {per} * a[4] + t % {ct} * a[5] + t % {per} / {ct} + a[6], "
                          f"{frame(f)[0]}); ready(); frame_done[{f}] = ++events;" for f in (0, 1)])),
    unpack_program(loop(*[f"fetched(); {tt(unpack(frame(f)[0], ZERO_SLOT) + REPORT)}" for f in (0, 1)]), kernel_config(transposing=True)),
    math_program({}, loop(*[tt(math_begin(False) + elementwise("add", {}) + MATH_END) + f" h ^= {DST_HALF};"] * 2)),  # padding stays 0
    pack_program(loop(*[f"output({f}, {frame(f)[1]}, t); " + tt(pack("h", frame(f)[1], SEM_PACKED)) + f" h ^= {DST_HALF};"
                   for f in (0, 1)])))

# move_kernel caches x's tiles in L1, tile t in slot t % MOVE_SLOTS. A prime spreads out power-of-two strides.
MOVE_SLOTS = 127

def move_kernel(dtype):
  """Move a tensor around: reshape, permute, expand, pad, slice. BRISC does it alone.

  args: [first unit, units, out, x's address, tile strides for batch and row, tile offset, x's rows and columns, the
  result's rows and columns, then the map: an offset and strides sb, sr and sc for batch, row and column, then the
  bounds: lo for batch, row and column, then hi].

  Element (b, r, c) of the result is element offset + b sb + r sr + c sc of x, counted in row-major order, if it's
  inside the bounds. Outside them it's 0.

  NoC alignment rules prevent arbitrary two-byte reads, so fetch complete input tiles into an L1 cache.
  If an output tile is an unchanged input tile within the bounds, copy it directly, bypassing the cache.
  Build transposed tiles face by face from cached data. For other tiles, start with zeros and copy valid
  values in runs along rows or columns, whichever are contiguous in x. Stop each run at a face boundary.
  When neither axis forms a run, copy individual values without dividing at each step.
  Repeat a single output column across its tile. Repeat a single row using NoC copies within L1."""
  tb = tile_bytes(dtype)
  return compile_kernel(f"""
#define OUT {DATA:#x}
#define CACHE {DATA + tb:#x}
typedef {"unsigned short" if ITEMSIZE[dtype] == 2 else "unsigned"} datum;
{ROWS}
// What each cache slot holds: x's tile number and the launch it came in. They live in .data, so unlike the other
// statics they aren't zeroed at each launch, which saves a store a slot. The launch number makes tiles that earlier
// launches cached miss.
__attribute__((section(".data"))) static unsigned held[{MOVE_SLOTS}], came[{MOVE_SLOTS}], launch;
static volatile datum *fetch(unsigned base, unsigned t) {{  // x's tile t, in L1
  unsigned s = t % {MOVE_SLOTS};
  if (held[s] == t && came[s] == launch) return (volatile datum *)(CACHE + s * {tb});
  noc_get(BANK(t), AT(base, t, {tb}), CACHE + s * {tb}, {tb});
  noc_wait();
  held[s] = t, came[s] = launch;
  return (volatile datum *)(CACHE + s * {tb});
}}
// Copy n datums from s to d, with steps ss and dd. Steps of 1 copy four at a time, and bf16 two to a word.
static void run(volatile datum *d, volatile datum *s, unsigned n, unsigned dd, unsigned ss) {{
  if (dd == 1 && ss == 1) {{
    if (sizeof(datum) == 2 && !(((unsigned)d | (unsigned)s) & 3)) {{
      volatile unsigned *dw = (volatile unsigned *)d, *sw = (volatile unsigned *)s;
      for (; n >= 8; n -= 8, dw += 4, sw += 4) dw[0] = sw[0], dw[1] = sw[1], dw[2] = sw[2], dw[3] = sw[3];
      d = (volatile datum *)dw, s = (volatile datum *)sw;
    }}
    for (; n >= 4; n -= 4, d += 4, s += 4) d[0] = s[0], d[1] = s[1], d[2] = s[2], d[3] = s[3];
  }}
  for (; n; n--, d += dd, s += ss) *d = *s;
}}
// The positions [from, to) in tile k, along an axis n long, whose elements are inside [lo, hi).
// A dimension of 1 fills its tile.
static int from(int k, int n, int lo) {{ return n == 1 || lo <= 32 * k ? 0 : lo - 32 * k; }}
static int to(int k, int n, int lo, int hi) {{
  if (n == 1) return lo <= 0 && 0 < hi ? 32 : 0;
  int end = (hi < n ? hi : n) - 32 * k;
  return end < 32 ? end : 32;
}}
void kmain(volatile unsigned *a) {{
  unsigned base = a[3], bstride = a[4], stride = a[5], offset = a[6], xr = a[7], xc = a[8], rows = a[9], cols = a[10];
  launch++;
  unsigned tr = (rows + 31) / 32, tc = (cols + 31) / 32;
  int e0 = a[11], sb = a[12], sr = a[13], sc = a[14], lo_b = a[15], lo_r = a[16], lo_c = a[17], hi_b = a[18], hi_r = a[19], hi_c = a[20];
  // The datums go in lines along the result's rows, or down its columns. Down is for when the columns lie in x as runs
  // (along a row of x, or down a column) and the rows don't or are shorter, and always for a single column. step is
  // the step along a line, in x's elements, split into x's matrices, rows and columns so stepping needs no division.
  int run_along = cols > 1 && (sc == 1 || sc == (int)xc), run_down = sr == 1 || sr == (int)xc;
  int down = rows > 1 && (cols == 1 || run_down && (!run_along || (rows < 32 ? rows : 32) > (cols < 32 ? cols : 32)));
  int step = down ? sr : cols == 1 ? 0 : sc;
  unsigned sq = step >= 0 ? step / xc : 0, sm = step >= 0 ? step % xc : 0, qb = sq / xr, qi = sq % xr;
  unsigned whole = rows > 1 && cols > 1 && sc == 1 && sr == (int)xc, turned = rows > 1 && cols > 1 && sr == 1 && sc == (int)xc;
  volatile datum *out = (volatile datum *)OUT, *tile = 0;
  unsigned held_t = ~0u;
  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned t = a[0] + u;
    int b = t / (tr * tc), i = t % (tr * tc) / tc, j = t % tc;
    int r0 = from(i, rows, lo_r), r1 = lo_b <= b && b < hi_b ? to(i, rows, lo_r, hi_r) : 0, c0 = from(j, cols, lo_c), c1 = to(j, cols, lo_c, hi_c);
    // where the tile starts in x
    unsigned e = (unsigned)e0 + (unsigned)b * sb + (unsigned)i * 32 * sr + (unsigned)j * 32 * sc, row = e / xc % xr, col = e % xc;
    unsigned src = e / xc / xr * bstride + row / 32 * stride + col / 32 + offset;
    if ((whole || turned) && r0 == 0 && r1 == 32 && c0 == 0 && c1 == 32 && row % 32 == 0 && row + 32 <= xr && col % 32 == 0 && col + 32 <= xc) {{
      if (whole) {{ noc_get(BANK(src), AT(base, src, {tb}), OUT, {tb}); noc_wait(); }}  // x's tile src as it is
      else {{  // x's tile src transposed: face (fr, fc) is its face (fc, fr) transposed
        tile = fetch(base, src);  // keep held_t in step with tile, as the lines below expect
        held_t = src;
        for (unsigned f = 0; f < 4; f++)
          for (unsigned r = 0; r < 16; r++)
            for (unsigned c = 0; c < 16; c++) out[f * 256 + r * 16 + c] = tile[(f % 2 * 2 + f / 2) * 256 + c * 16 + r];
      }}
    }} else {{
      if (rows == 1 && r1) {{  // a single row is made once and doubled below, so the rest of the tile needn't be zeroed
        for (unsigned c = 0; c < 16; c++) out[c] = 0, out[256 + c] = 0;
        r1 = 1;
      }} else {{ copy({ZEROS:#x}, OUT, {tb}); noc_wait(); }}  // padding and what lies outside the bounds stay 0
      int lines0 = down ? c0 : r0, lines1 = down ? (cols == 1 && c1 ? 1 : c1) : r1, k0 = down ? r0 : c0, k1 = down ? r1 : c1;
      for (int line = lines0; line < lines1; line++) {{
        int r = down ? k0 : line, c = down ? line : k0;
        unsigned e_line = (unsigned)e0 + (unsigned)b * sb + (rows == 1 ? 0u : i * 32u + r) * sr + (cols == 1 ? 0u : j * 32u + c) * sc;
        unsigned xrow = e_line / xc, xb = xrow / xr, xi = xrow % xr, xj = e_line % xc;
        for (int k = k0; k < k1;) {{
          unsigned xt = xb * bstride + xi / 32 * stride + xj / 32 + offset;
          if (xt != held_t) {{ tile = fetch(base, xt); held_t = xt; }}
          if (step == 1 || step == (int)xc) {{  // a run along a row or column of x, to the end of a face's line or of x
            unsigned in_row = step == 1, n = k1 - k, m = 16 - k % 16, f = 16 - (in_row ? xj : xi) % 16, left = in_row ? xc - xj : xr - xi;
            n = n < m ? n : m; n = n < f ? n : f; n = n < left ? n : left;
            run(out + at(r, c), tile + at(xi % 32, xj % 32), n, down ? 16 : 1, in_row ? 1 : 16);
            k += n;
            if (down) r += n; else c += n;
            if (in_row && (xj += n) == xc) {{ xj = 0; xi++; }}
            if (!in_row) xi += n;
            if (xi == xr) {{ xi = 0; xb++; }}
            continue;
          }}
          out[at(r, c)] = tile[at(xi % 32, xj % 32)];
          if (++k >= k1) break;
          if (down) r++; else c++;
          if (step < 0) {{  // a negative step finds its place in x again by division
            unsigned ek = e_line + (k - k0) * step;
            xrow = ek / xc; xb = xrow / xr; xi = xrow % xr; xj = ek % xc;
            continue;
          }}
          xj += sm; xi += qi; xb += qb;
          if (xj >= xc) {{ xj -= xc; xi++; }}
          if (xi >= xr) {{ xi -= xr; xb++; }}
        }}
      }}
      if (down && cols == 1)  // a single column is made once, then copied across both faces of each row
        for (int r = r0; r < r1; r++) {{
          volatile datum *left = out + at(r, 0), *right = out + at(r, 16);
          datum v = *left;
          for (unsigned c = 1; c < 16; c++) left[c] = v;
          for (unsigned c = 0; c < 16; c++) right[c] = v;
        }}
      if (rows == 1 && r1) row_down(OUT, {ITEMSIZE[dtype]});
    }}
    noc_write(OUT, BANK(t), AT(a[2], t, {tb}), {tb});
    noc_wait();
  }}
}}""")
