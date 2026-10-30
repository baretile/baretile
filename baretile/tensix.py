# Tensix instructions and kernel programs, implemented from the ISA without LLK.
# Tensor kernels use four programs per worker. BRISC reads operand tiles into L1 and writes results to DRAM.
# The three TRISCs supply instructions to the unpack, math and pack threads, which process
# separate tiles concurrently.
#
# A 32x32 tile has four 16x16 faces: top-left, top-right, bottom-left, bottom-right. Each face
# is row-major. Datum (r, c) is at (r // 16 * 2 + c // 16) * 256 + r % 16 * 16 + c % 16.
# Faces match the FPU's 16-column operands; each MVMUL computes eight output rows. The ISA doesn't define tiles.
# Unpacking puts face f in rows 16f..16f+15 of SrcA or SrcB. Dst uses the same layout.
import math, struct
from baretile import bh
from baretile.runtime import DATA, FLAGS, SYNC, SYNC_NAMES

ITEMSIZE = {"bf16": 2, "fp32": 4, "int32": 4}  # bytes per value of each dtype. a tile is 1024 values
# A bf16 tile fills a SrcA or SrcB bank (64 rows of 16), or 64 Dst rows. A block-float tile is 64 exponents, then the
# 1024 values, and the 16 values in a row of a face share an exponent. BFP has the size of those values in bytes. In
# bfp8 a value is a sign and 7 bits of magnitude m, worth m * 2^(exponent - 133). In bfp4 it's a sign and 3 bits, two
# to a byte with the first in the low nibble. Those 3 bits are the top 3 of bfp8's, and the unpacker shifts them back.
BFP = {"bfp8": 1024, "bfp4": 512}
def tile_bytes(dtype): return 64 + BFP[dtype] if dtype in BFP else 1024 * ITEMSIZE[dtype]
def ext(shape, ax): return shape[ax] if ax == 0 else -(-shape[ax] // 32)  # extent along ax: matrices for 0, else tiles
# The L1 tile slots of a kernel that computes in dtype. Slot i is at base + i * tile_bytes(dtype), so slot 0 is the
# runtime's zeros, or part of them. The Z counter of an unpacker and the channel 1 Y counter of the packer pick a slot.
def slot_base(dtype): return DATA - tile_bytes(dtype)
# Where a kernel packs its results when it packs in a different dtype than it computes in. There's a 4 KiB tile for
# each of the two frames.
OUTS = bh.L1_SIZE - 2 * 4096
# Dst rows for an fp32 result. Kernels use two such halves, rows 0-127, so math fills one while pack drains the other
DST_HALF = 64
# SEM_DST counts results in Dst waiting to be packed, SEM_PACKED outputs packed but not reported yet, and SEM_L1
# intermediates packed into L1 that nobody has waited for yet.
SEM_DST, SEM_PACKED, SEM_L1 = 1 << 0, 1 << 1, 1 << 2
SEM_UNPACKED = 1 << 5  # frames and matmul steps unpacked but not reported yet
MAX_OPS = 15  # the most ops a fused kernel can have, since SEM_L1 counts their intermediates
# Reductions accumulate at these Dst rows, with scratch 128 rows further on. Each has a semaphore that is 1 while the
# accumulator is free. The first step of a reduction takes it, and the pack gives it back.
ACCS = ((256, 1 << 3), (320, 1 << 4))

def ins(name, **fields):
  opcode, layout = bh.TENSIX_OPS[name]
  word = opcode << 24
  for k, v in fields.items():
    first, width = layout[k]
    if not (0 <= v < 1 << width): raise ValueError(f"{name} {k}={v} does not fit in {width} bits")
    word |= v << first
  return word

def config(**fields):  # backend config as {word index: value}. fields left out of a word we write are 0
  regs = {}
  for k, v in fields.items():
    index, first, width = bh.TENSIX_CFG_FIELDS[k]
    if not (0 <= v < 1 << width): raise ValueError(f"{k}={v} does not fit in {width} bits")
    regs[index] = regs.get(index, 0) | v << first
  return regs

FORMAT = {"bf16": bh.BF16, "fp32": bh.FP32, "int32": bh.INT32, "bfp8": bh.BFP8, "bfp4": bh.BFP4}
def kernel_config(dtype="bf16", out="bf16", transposing=False):
  """Build the unpacker and packer configuration for a kernel.

  UNPACR reads 1024 values per tile. Its Z counter selects an L1 slot with a stride of one tile.
  bf16 values enter SrcA or SrcB. fp32 values bypass those banks and enter Dst through unpacker 0's
  context 1, where channel 1's Y counter selects the 32-bit Dst row. The FPU writes Dst in fp32.

  Each PACR reads four Dst rows and writes them in format out, rounding when out is bf16. The Dst row
  stride uses bytes of out, or fp32 bytes for a block float. Channel 1's Y counter selects the output
  slot with a stride of one output tile.

  Base addresses use 16-byte units minus 1. Unpacker 0's rows have a bias of 4 (64 values).
  With transposing, unpacker 0 transposes each 16 x 16 face as it enters SrcA."""
  base, fmt, pack = slot_base(dtype) // 16 - 1, FORMAT[dtype], FORMAT[out]
  # The packer's own input format. Packing a block float from fp32 rounds each value once. From bfp8 it would round
  # twice.
  late = bh.FP32 if out in BFP else pack
  return config(Dstacc=late, Fp32_enabled=1, Zero_Flag_disabled_src=0, Read_32b_data=1, Edge_mask=0xFFFF,
                pack_reads_per_xy_plane=16,
                PCK0_Ystride=16 * ITEMSIZE["fp32" if late == bh.FP32 else out], PCK0_ZW=0,
                PCK0_Ystride_1=tile_bytes(out),
                PCK0_ZW_1=0, PCK0_Base_1=0, PCK_dest_offset=0, Pack_L1_Acc=0,
                L1_Dest_addr=(slot_base(dtype) if out == dtype else OUTS) // 16 - 1, Disable_zero_compress=1,
                Pack_out_format=pack,
                Exp_section_size=4,  # the 64 exponents of a block float, in units of 16 bytes
                Pack_in_format=late, UNP0_add_dest_addr_cntr=0, UNP0_Ystride_1=64, UNP0_ZW_1=0, UNP1_ZW_1=0,
                SEC0_format=fmt, SEC0_x_dim=1024, SEC0_y_dim=1, SEC0_z_dim=1, SEC0_out_format=fmt, SEC0_uncompressed=1,
                SEC0_base=base,
                SEC0_dest_addr=64, SEC0_tile_x_dim=1024, SEC0_offset=0, SEC0_Haloize_mode=int(transposing),
                SEC0_uncompressed_1=1, SEC0_to_dst_1=1, SEC0_base_1=base, SEC0_dest_addr_1=64, SEC0_tile_x_dim_1=1024,
                SEC0_offset_1=0,
                SEC1_format=fmt, SEC1_x_dim=1024, SEC1_y_dim=1, SEC1_z_dim=1, SEC1_out_format=fmt, SEC1_uncompressed=1,
                SEC1_base=base)

# Address modes of the math thread, as (SrcA step, SrcB step) and (Dst step, fidelity step). Modes 0-1 are for
# elementwise ops, 2-3 for SFPU loads and stores (the store steps Dst by 2), and 4-7 for matmul. Each kind of op has its
# own modes, so ops can mix without rewriting modes between them.
ADDR_MODES = {0: ((8, 8), (8, 0)), 1: ((0, 0), (0, 1)), 2: ((0, 0), (0, 0)), 3: ((0, 0), (2, 0)),
              4: ((16, 0), (0, 0)), 5: ((48, 8), (0, 0)), 6: ((16, 8), (0, 0)), 7: ((16, 8), (0, 1))}
# The config of a thread. We write all of it because the reset values aren't documented, except packer modes 1-3,
# since every PACR uses mode 0. With dst, the thread unpacks into Dst.
def thread(dst=False):
  return [ins("SETC16", setc16_reg=r, setc16_value=v) for r, v in
          ({r: 0 for r in bh.THREAD_CFG.values()} | {bh.ADDR_MOD_BIAS + m: 0 for m in range(8)} |
           {bh.ADDR_MOD_PACK: 4} |  # packer mode 0 moves on 4 Dst rows each PACR
           {bh.ADDR_MOD_AB + m: a | b << 8 for m, ((a, b), _) in ADDR_MODES.items()} |
           {bh.ADDR_MOD_DST + m: d | f << bh.FIDELITY_INCR for m, (_, (d, f)) in ADDR_MODES.items()} |
           ({bh.THREAD_CFG["UNPACK_MISC"]: 1} if dst else {bh.THREAD_CFG["SRCA_SET"]: bh.SRCA_SET_BY_ADDR})).items()]

# Each thread has a 32-entry replay buffer. REPLAY with load records the next instructions, and REPLAY without it plays
# them back, so a run that repeats costs its TRISC one word. The pack thread's buffer always holds 15 PACRs. A kernel
# divides the math thread's buffer among the sequences it uses, the most replayed first.
def replay(index, count): return ins("REPLAY", index=index, count=count)
def record(index, words): return [ins("REPLAY", load=1, index=index, count=len(words))] + words
PACRS = 0
class Loop(list):  # words that a TRISC pushes `times` times, in a C loop
  def __init__(self, times, words): super().__init__(words); self.times = times

# Thread 0 unpacks, thread 1 does the math and thread 2 packs, and TRISC i feeds thread i. This is how they sync:
#   unpack -> math: SrcA and SrcB each have two banks. The hardware makes UNPACR wait for a free bank, and the FPU
#     for a full one.
#   math -> pack: SEM_DST counts results in Dst waiting to be packed. Math waits while both halves are full, and pack
#     waits while there are none.
#   pack -> unpack, in a fused kernel: pack writes an intermediate into L1, and a later op unpacks it. Pack posts
#     SEM_L1 for each intermediate. Before unpack reads one, it takes SEM_L1 as many times as ops were packed since it
#     last looked.
#   unpack, pack -> BRISC: thread 0 posts SEM_UNPACKED once it has unpacked the tiles of a frame, or the pair of a
#     matmul step. The pack thread posts SEM_PACKED once it has packed a result. Their TRISCs add these up in SYNC,
#     where BRISC looks.
#   In an fp32 or int32 kernel the math thread does the unpacking into Dst, so it takes SEM_L1 and posts SEM_UNPACKED
#   in place of thread 0.
UNPACKERS = bh.UNPACK0 | bh.UNPACK1
def counters(mask, n):  # resets the address counters of these units to 0, with each instruction moving n datums
  return [ins("SETADCXY", cnt_set_mask=mask, bit_mask=0b1111), ins("SETADCZW", cnt_set_mask=mask, bit_mask=0b1111),
          ins("SETADCXX", cnt_set_mask=mask, x_end2=n - 1)]
SETUP_UNPACK = [ins("SEMINIT", sem_sel=SEM_DST, max_value=2), ins("SEMINIT", sem_sel=SEM_PACKED, max_value=15),
                ins("SEMINIT", sem_sel=SEM_L1, max_value=15), ins("SEMINIT", sem_sel=SEM_UNPACKED, max_value=15),
                *[ins("SEMINIT", sem_sel=sem, init_value=1, max_value=1) for _, sem in ACCS],
                *counters(UNPACKERS, 1024)]
SETUP_PACK, PACK_RUNS = counters(bh.PACK, 16), record(PACRS, [ins("PACR") for _ in range(15)])  # 16 datums a Dst row
# Puts v into the low field of word. v is a number, or a C expression whose value is added when the word is pushed at
# run time.
def field(word, v): return word + v if isinstance(v, int) else (word, v)
# Z is the slot
def setadc(unpacker, slot): return field(ins("SETADC", cnt_set_mask=unpacker, dimension_index=2, value=0), slot)
def unpacr(b, z):  # unpacks a tile into SrcA, or SrcB with b, and hands the bank to the FPU. Then Z moves on by z
  return ins("UNPACR", last=1, set_dat_valid=1, ovrd_thread_id=1, unpack_block_selection=b, addr_mode=z)
# slot a into SrcA, b into SrcB
def unpack(a, b): return [setadc(bh.UNPACK0, a), setadc(bh.UNPACK1, b), unpacr(0, 0), unpacr(1, 0)]
REPORT = [ins("STALLWAIT", stall_res=bh.STALL_SYNC, wait_res=bh.WAIT_UNPACK), ins("SEMPOST", sem_sel=SEM_UNPACKED)]
# Unpacks an fp32 tile from an L1 slot into Dst, from row `row` on, bypassing SrcA. The math thread does this, with
# unpacker 0 in context 1.
def fill(slot, row):
  return [setadc(bh.UNPACK0, slot),
          field(ins("SETADC", cnt_set_mask=bh.UNPACK0, dimension_index=1, channel_index=1, value=0), row),
          ins("UNPACR", last=1, ovrd_thread_id=1)]
# the SFPU and the sync unit wait for the fills
FILLED = [ins("STALLWAIT", stall_res=bh.STALL_SFPU | bh.STALL_SYNC, wait_res=bh.WAIT_UNPACK0)]
def reconfigure(**fields):  # rewrites config fields of the unpackers from thread 0, a byte at a time
  # RMWCIB keeps the rest of the word. The writes wait for unpacker 0 to finish its unpacks so far, or for both
  # unpackers when a field belongs to unpacker 1. The next unpack waits for the writes.
  writes = [ins(f"RMWCIB{b}", cfg_reg_addr=word, data=v << first >> 8 * b & 255,
                mask=(1 << width) - 1 << first >> 8 * b & 255)
            for (word, first, width), v in ((bh.TENSIX_CFG_FIELDS[k], v) for k, v in fields.items())
            for b in range(first // 8, (first + width + 7) // 8)]
  idle = bh.WAIT_UNPACK if any(k.startswith("SEC1") for k in fields) else bh.WAIT_UNPACK0
  return ([ins("STALLWAIT", stall_res=bh.STALL_CFG, wait_res=idle)] + writes +
          [ins("STALLWAIT", stall_res=bh.STALL_UNPACK, wait_res=bh.WAIT_CFG)])
def repack(old, new):  # switches the packer's config from old to new (both from kernel_config), from the pack thread
  # Only the bytes that differ get written. They wait for the packer to go idle, and its next pack waits for them.
  writes = [ins(f"RMWCIB{b}", cfg_reg_addr=w, data=new[w] >> 8 * b & 255,
                mask=255) for w in sorted(new) for b in range(4)
            if (old[w] ^ new[w]) >> 8 * b & 255]
  return ([ins("STALLWAIT", stall_res=bh.STALL_CFG, wait_res=bh.WAIT_PACK)] + writes +
          [ins("STALLWAIT", stall_res=bh.STALL_PACK, wait_res=bh.WAIT_CFG)])
def haloize(on): return reconfigure(SEC0_Haloize_mode=on)  # unpacker 0 transposes each 16x16 face into SrcA, or stops
def accumulating(at, onto):  # config for the packer to write fp32 tiles at `at`, its channel 1 Y picking the tile
  # With onto, it adds what it packs to what's already there (Pack_L1_Acc). Attention uses this for its output, and a
  # matmul to sum the parts of its K.
  return kernel_config("bf16", "fp32") | config(L1_Dest_addr=at // 16 - 1, Pack_L1_Acc=int(onto))
def reload(at, z):  # fp32 tile z of the accumulator at `at` into SrcA, with zeros in SrcB, for math to add into Dst
  # Unpacker 0 switches to the accumulator and back to the bf16 slots. SrcA holds the tile as TF32, with 10 bits of
  # mantissa. ttsim truncates it.
  return (reconfigure(SEC0_format=bh.FP32, SEC0_out_format=bh.TF32, SEC0_base=at // 16 - 1) + unpack(z, ZERO_SLOT) +
          reconfigure(SEC0_format=bh.BF16, SEC0_out_format=bh.BF16, SEC0_base=slot_base("bf16") // 16 - 1))
# Configure unpacker 0 for block floats in a region of `slots` tiles, at base or B8 + region * region size.
# The region stores all exponents before the values. The unpacker converts each value to bf16 in SrcA.
# With 0 slots, restore the ordinary bf16 configuration.
def bfp_b(wt, slots, region=0, base=None):
  fmt, base = (FORMAT[wt], base or B8 + region * slots * tile_bytes(wt)) if slots else (bh.BF16, slot_base("bf16"))
  return reconfigure(SEC0_format=fmt, SEC0_out_format=fmt, SEC0_z_dim=max(slots, 1), SEC0_base=base // 16 - 1)
def step_unpack(s):  # a matmul step. B's tile goes from stream slot s into SrcA, and the next tile of A's row into SrcB
  return [setadc(bh.UNPACK0, s), unpacr(0, 0), unpacr(1, 1)] + REPORT
# PAIR unpacks a tile into SrcA and one into SrcB, then moves each unpacker's Z on by 1. WALK is a matmul step along
# B's column and A's row when both are kept in L1.
PAIR = [unpacr(0, 1), unpacr(1, 1)]
WALK = PAIR + REPORT
# Aims the FPU and SFPU at Dst row `row` on. The write waits for this thread's FPU and SFPU work first, as tt-llk's
# dest_section_flip does, since work still in their pipelines would see the new offset. TTSIM_CHAOS_MATH tests this.
def aim(row):
  return [ins("STALLWAIT", stall_res=bh.STALL_CFG, wait_res=bh.WAIT_FPU | bh.WAIT_SFPU),
          field(ins("SETC16", setc16_reg=bh.THREAD_CFG["MATH_Offset"], setc16_value=0), row)]
# In the C programs, h is the first Dst row of the half that the next op gets, 0 or DST_HALF. h ^= DST_HALF moves on.
DST_FREE = ins("SEMWAIT", stall_res=bh.STALL_ALL, sem_sel=SEM_DST, wait_sem_cond=2)  # waits until a Dst half is free
# Waits for a free Dst half and aims the FPU at it. If the math accumulates, it also zeroes the half, 16 rows at a time.
def math_begin(zero):
  return [DST_FREE, *aim("h")] + [field(ins("ZEROACC", clear_mode=1, use_32_bit_mode=1, where=b),
                                        "h / 16") for b in range(4) if zero]
MATH_END = [ins("STALLWAIT", stall_res=bh.STALL_SYNC, wait_res=bh.WAIT_FPU | bh.WAIT_SFPU),
            ins("SEMPOST", sem_sel=SEM_DST)]
# packs 64 Dst rows from `row` into an L1 slot, then frees them and posts the semaphores in done
def pack(row, slot, *done):
  return [ins("SEMWAIT", stall_res=bh.STALL_ALL, sem_sel=SEM_DST, wait_sem_cond=1),
          field(ins("SETADC", cnt_set_mask=bh.PACK, dimension_index=1, value=0), row),  # channel 0 Y: Dst row
          # channel 1 Y: L1 slot
          field(ins("SETADC", cnt_set_mask=bh.PACK, dimension_index=1, channel_index=1, value=0), slot),
          replay(PACRS, 15), ins("PACR", last=1),  # 16 PACRs, 4 Dst rows each through all four read ports
          ins("STALLWAIT", stall_res=bh.STALL_SYNC, wait_res=bh.WAIT_PACK), ins("SEMGET", sem_sel=SEM_DST)] + \
         [ins("SEMPOST", sem_sel=d) for d in done]
# Waits until the semaphore is above 0, then takes one.
def take(sem): return [ins("SEMWAIT", stall_res=bh.STALL_ALL, sem_sel=sem, wait_sem_cond=1), ins("SEMGET", sem_sel=sem)]
TAKE_L1 = take(SEM_L1)

def matmul_phase(transposed=False):
  """One fidelity phase of Dst += A @ B on a pair of tiles, with A in SrcB and B in SrcA.

  MVMUL does Dst[8x16] += SrcB[8x16] @ SrcA[16x16]. Output face (i, j) adds A face (i, k) times B face (k, j) for
  k = 0, 1, in two MVMULs of 8 rows each (h). In the order below, SrcA steps by 16, -16, 16 and 16 rows, then again,
  and SrcB by 0, 8, 0 and 8 rows, so four address modes (4-7) cover it. The last one wraps both back to row 0, since the counters
  wrap at 64, and moves on to the next fidelity phase.

  With transposed, B came in as it lies and unpacker 0 transposed each face (haloize). Face (k, j) of B^T is then face
  2j + k of SrcA. SrcA steps by 32 for the next j or h, by -16 for the next k and by 16 for the next i. That takes five
  modes instead of four (TRANSPOSED rewrites slots 4-6 and 2 around the matmul's steps), but no read has to be split
  to move faces around."""
  phase = []
  for i, k, h, j in [(i, k, h, j) for i in (0, 1) for k in (0, 1) for h in (0, 1) for j in (0, 1)]:
    last = (i, k, h, j) == (1, 1, 1, 1)
    mode = 4 + (0 if j == 0 else 1 if h == 0 else 3 if last else 2)
    if transposed and mode == 6 and k == 1: mode = 2  # mode 2 moves to the next i. mode 6 moves to the next k
    phase.append(ins("MVMUL", dst=32 * i + 16 * j + 8 * h, addr_mode=mode))
  return phase

def round_bf16(bits):  # fp32 bits to bf16, preserving infinities and a quiet NaN payload
  if bits & 0x7FFFFFFF > 0x7F800000: return bits >> 16 | 0x40
  return (bits + 0x7FFF + (bits >> 16 & 1)) >> 16  # round to nearest even
def to_bf16(x): return round_bf16(struct.unpack("<I", struct.pack("<f", x))[0])
def from_bf16(b): return struct.unpack("<f", struct.pack("<I", b << 16))[0]

# An SFPU program is a setup, then a body that runs 32 times. Each pass loads 32 datums of the Dst half into LReg 0
# (4 rows of 8 even or odd columns) and stores the result back. The store steps Dst by 2 (address modes 2 and 3).
# Loads and stores use mod0 3 because Dst is fp32.
def sfp_load(r=0, at=0): return ins("SFPLOAD", vd=r, mod0=3, addr_mode=2, imm10=at)
def sfp_store(r, at=0, step=True, mod0=3): return ins("SFPSTORE", vd=r, mod0=mod0, addr_mode=3 if step else 2, imm10=at)
# The start of an SFPU program. The SFPU waits for the FPU's writes, and the Dst counter goes back to the row that aim
# set.
SFPU_START = [ins("STALLWAIT", stall_res=bh.STALL_SFPU, wait_res=bh.WAIT_FPU), ins("SETRWC", bit_mask=0b0100)]
# SFPSTORE's INT32 mode stores the bits as they are. Its FP32 mode flushes denormals, and as int32 those are the small
# positive integers.
RAW = 4
def sfp_bits(r, bits): return [ins("SFPLOADI", vd=r, mod0=8, imm16=bits >> 16),
                               ins("SFPLOADI", vd=r, mod0=10, imm16=bits & 0xFFFF)]
# a full fp32 constant into LReg r
def sfp_const(r, x): return sfp_bits(r, struct.unpack("<I", struct.pack("<f", x))[0])
def sfp_bf16(r, x):  # one word, for a constant that bf16 holds exactly
  assert from_bf16(to_bf16(x)) == x, f"{x} is not a bf16"
  return ins("SFPLOADI", vd=r, mod0=0, imm16=to_bf16(x))
def mad(d, a, b, c, neg=0): return ins("SFPMAD", vd=d, va=a, vb=b, vc=c, mod1=neg)  # d = a * b + c. neg 1: -a, 2: -c
def mul(d, a, b): return ins("SFPMUL", vd=d, va=a, vb=b, vc=9)
CONSTS = bh.LREG_CONSTS  # constant LRegs
ZERO, ONE, MINUS_ONE = CONSTS[0.0], CONSTS[1.0], CONSTS[-1.0]
# lane flags: on, all set, off
CC_ON, CC_ALL, CC_OFF = ins("SFPENCC", mod1=10, imm2=3), ins("SFPENCC", mod1=0), ins("SFPENCC", mod1=10, imm2=2)
NAN, INF = 0x7FC00000, 0x7F800000
def nans(x, t, inf):  # sets the lane flags where LReg x is NaN, using LRegs t and inf
  # x + 0 makes any NaN 0x7FC00000, even a negative one that SFPABS would leave at the bottom of SFPGT's order. A lane
  # is NaN where that is above the bits of inf (which LReg inf gets). An SFPMOV between CC_ON and CC_OFF can then give
  # those lanes NaN.
  return [mad(t, x, ONE, ZERO), *sfp_bits(inf, INF), ins("SFPGT", vd=t, vc=inf, mod1=1)]
# with flags on, keeps lanes where int LReg e >= n
def at_least(e, n, t): return ins("SFPIADD", vd=t, vc=e, mod1=9, imm12=-n & 0xFFF)
def below(e, n, t): return ins("SFPIADD", vd=t, vc=e, mod1=1, imm12=-n & 0xFFF)  # or < n. both overwrite LReg t
# LReg d = cs[0] + cs[1] x + ..., by Horner. Each coefficient goes into LReg temp when it's needed, except 0 and +-1,
# which have LRegs of their own.
def poly(d, x, cs, temp):
  words = sfp_const(d, cs[-1])
  for c in cs[-2::-1]: words += ([] if c in CONSTS else sfp_const(temp, c)) + [mad(d, d, x, CONSTS.get(c, temp))]
  return words

# The transcendentals are accurate to about 1e-7, the precision of fp32, so an fp32 tensor matches numpy's float32 to
# within a step or two. They reduce x to a small range in exact steps, and a polynomial does the rest. The polynomials
# are minimax fits, made offline.
EXP_P = [1.0, 1.0, 0.49999991, 0.16666421, 0.041668355, 0.008374773, 0.001382954]  # e^r on [-ln2/2, ln2/2], 8e-8
# 1.3e-7
LOG2_Q = [1.442695, -0.72134733, 0.4809106, -0.36070517, 0.2879189, -0.23891294, 0.21565668, -0.20746617, 0.12623915]
SIN_P = [1.0, -0.16666658, 0.0083330506, -0.00019809045, 2.6051646e-06]  # sin(r) / r in r^2, r in [-pi/2, pi/2], 1.4e-7

def exp(base2, nan=False):
  """2^x, or e^x when base2 is False.

  t is x for 2^x and x log2(e) for e^x. x is first clamped so that |t| is at most 150.04, which sends a NaN to one end.
  n = round(t). Adding and then subtracting 1.5 * 2^23 rounds to an integer and leaves its bits in the low bits of the
  sum. r = (x - n) ln 2 for 2^x. For e^x, r = x - n ln 2, in two parts (Cody-Waite: n times 0.693359375 is exact).
  A polynomial gives e^r, which gets multiplied by 2^floor(n/2) and 2^ceil(n/2), each built from its exponent bits.
  Those multiplies overflow to inf, or flush to 0, where the true result does.

  nan is for fp32 kernels. It loads x from Dst again and puts the NaN back. In bf16 a NaN is already inf before the
  SFPU sees it."""
  (lim, t), ln2 = (150.0, 0) if base2 else (104.0, 5), math.log(2)
  return (sfp_const(1, ln2) if base2 else
          sfp_const(1, 1 / ln2) + sfp_const(2, 0.693359375) + sfp_const(3, ln2 - 0.693359375),
          [sfp_load(), sfp_bf16(4, -lim), ins("SFPSWAP", vd=4, vc=0, mod1=1), sfp_bf16(4, lim),
           ins("SFPSWAP", vd=0, vc=4, mod1=1),  # x within +-lim
           # LReg 7: n
           *([] if base2 else [mul(5, 0, 1)]), sfp_bf16(6, 1.5 * 2**23), mad(4, t, ONE, 6), mad(7, 4, ONE, 6, neg=2),
           ins("SFPIADD", vd=6, vc=4, mod1=6),  # LReg 6: n as an integer, bits(t + 1.5 2^23) - bits(1.5 2^23)
           # LReg 5: r
           *([mad(5, 7, ONE, 0, neg=1), mul(5, 5, 1)] if base2 else [mad(5, 7, 2, 0, neg=1), mad(5, 7, 3, 5, neg=1)]),
           *poly(4, 5, EXP_P, 7), ins("SFPSHFT", vd=0, vc=6, mod1=7, imm12=-1 & 0xFFF),
           ins("SFPIADD", vd=6, vc=6, mod1=5, imm12=1),
           # LReg 0, 6: floor(n/2) and ceil(n/2). then 2^ each, from its bits
           ins("SFPSHFT", vd=6, mod1=3, imm12=-1 & 0xFFF),
           *[w for r in (0, 6) for w in (ins("SFPIADD", vd=r, vc=r, mod1=5, imm12=127),
                                         ins("SFPSHFT", vd=r, mod1=1, imm12=23), mul(4, 4, r))],
           *([sfp_load(0), CC_ON, *nans(0, 6, 7), ins("SFPMOV", vd=4, vc=0), CC_OFF] if nan else []), sfp_store(4)])

def recip(nan=False):
  """1/x: the 7-bit estimate y from SFPARECIP, then two Newton steps y += y e, with e = 1 - x y. The estimate is inf
  for 0 and denormals and 0 for inf, where e is 0 * inf = NaN. The steps only run where the first e is below 1 in
  magnitude, which its exponent tells, so those lanes keep the estimate. So does an x past 2^126, whose estimate is
  the 0 that the steps would leave. With nan, for fp32, a NaN gets NaN back instead of its estimate of 0."""
  return ([],
          [sfp_load(), ins("SFPARECIP", vd=1, vc=0), mad(2, 0, 1, ONE, neg=1), CC_ON,
           ins("SFPEXEXP", vd=3, vc=2, mod1=2), mad(1, 1, 2, 1), mad(2, 0, 1, ONE, neg=1), mad(1, 1, 2, 1),
           *([CC_ALL, *nans(0, 3, 2), ins("SFPMOV", vd=1, vc=0)] if nan else []), CC_OFF, sfp_store(1)])

def sqrt():
  """sqrt(x) by Goldschmidt's iteration. y = 0x5F3759DF minus half the bits of x is 1/sqrt(x) within 3.5%, and g = x y
  and h = y / 2 approach sqrt(x) and 1/(2 sqrt(x)). A step multiplies both by 1 + r, with r = 1/2 - g h, and the second
  step only g. The last is Newton's g += h (x - g^2), as g (1 - g h) + x h, which also corrects the rounding that the
  first two leave between g and h. Every value stays near sqrt(x), 1/sqrt(x) or 1/2, so none underflows or overflows
  for normal x, and the result is within a step or two of fp32's.

  x y and x h subtract 0 rather than add it, so that a zero keeps its sign and comes back as it is, like a denormal,
  which is flushed to one. The result is x where x is above the largest float in SFPGT's order (inf and NaN), and NaN
  where it's below minus the largest denormal."""
  return (sfp_bits(4, 0x5F3759DF) + sfp_const(5, 0.5) + sfp_bits(6, 0x7F7FFFFF) + sfp_bits(7, 0x807FFFFF),
          # LReg 1: y
          [sfp_load(), ins("SFPSHFT", vd=1, vc=0, mod1=5, imm12=-1 & 0xFFF), ins("SFPIADD", vd=1, vc=4, mod1=6),
           mad(2, 0, 1, ZERO, neg=2), mul(1, 1, 5),  # LReg 2: g, 1: h
           # LReg 3: r
           mad(3, 2, 1, 5, neg=1), mad(2, 2, 3, 2), mad(1, 1, 3, 1), mad(3, 2, 1, 5, neg=1), mad(2, 2, 3, 2),
           mad(3, 2, 1, ONE, neg=1), mad(1, 0, 1, ZERO, neg=2), mad(2, 2, 3, 1),  # g (1 - g h) + x h
           CC_ON, ins("SFPGT", vd=0, vc=6, mod1=1), ins("SFPMOV", vd=2, vc=0),
           CC_ALL, ins("SFPGT", vd=7, vc=0, mod1=1), *sfp_bits(2, NAN), CC_OFF, sfp_store(2)])

def log(scale):
  """log2(x) * scale. Write x = 2^e m with m in [sqrt(1/2), sqrt(2)). m starts in [1, 2), and where it's above sqrt 2
  we halve it and add 1 to e. Then log2(x) = e + u q(u), with u = m - 1 and q a polynomial. The error stays relative
  near 1, where u is small and exact. inf and NaN come back unchanged, and a negative number gives NaN. Zero gives
  -inf, and so does a denormal, since it gets flushed."""
  return (sfp_const(5, math.sqrt(2)) + (sfp_const(7, scale) if scale != 1 else []),
          # LReg 1: biased e, 2: m
          [sfp_load(), ins("SFPEXEXP", vd=1, vc=0, mod1=1), ins("SFPSETEXP", vd=2, vc=0, mod1=1, imm12=127), CC_ON,
           ins("SFPGT", vd=2, vc=5, mod1=1), ins("SFPSETEXP", vd=2, vc=2, mod1=1, imm12=126),
           ins("SFPIADD", vd=1, vc=1, mod1=5, imm12=1), CC_OFF, ins("SFPCAST", vd=1, vc=1), sfp_bf16(6, -127.0),
           mad(1, 1, ONE, 6), mad(2, 2, ONE, MINUS_ONE), *poly(3, 2, LOG2_Q, 6), mad(3, 3, 2, 1),
           *([mul(3, 3, 7)] if scale != 1 else []), ins("SFPEXEXP", vd=1, vc=0, mod1=1), CC_ON, at_least(1, 255, 4),
           ins("SFPMOV", vd=3, vc=0), CC_ALL, ins("SFPSETCC", vc=0), *sfp_bits(3, NAN), CC_ALL, below(1, 1, 4),
           *sfp_bits(3, INF | 1 << 31), CC_OFF, sfp_store(3)])

def sin(phase):
  """sin(x + pi phase), with phase 0 or 0.5 for cosine, for finite |x| < 2^24.

  Split the integer k = round(|x| / pi + phase) at bit 12, and pi into four parts. The first three parts
  have at most 12 significant bits, so their products with each part of k are exact. Subtract them largest
  first. Correct a possible off-by-one k after reduction, then use sin(r) = r p(r^2) on [-pi/2, pi/2].
  Outside the supported range, including infinity, return NaN instead of an inaccurate finite value."""
  pi = (3.140625, 0.0009675025939941406, 1.5099067240953445e-7, 5.126565838509123e-12)
  half, whole = float.fromhex("0x1.921fb6p+0"), float.fromhex("0x1.921fb6p+1")
  correct = []
  for sign in (1, -1):
    correct += [CC_ALL, *sfp_const(2, sign * half), ins("SFPMOV", vd=7, vc=6 if sign == 1 else 2),
                ins("SFPGT", vd=7, vc=2 if sign == 1 else 6, mod1=1)]
    correct += [*sfp_const(4, sign * whole), mad(6, 6, ONE, 4, neg=2),
                *sfp_const(4, sign * (math.pi - whole)), mad(6, 6, ONE, 4, neg=2), ins("SFPXOR", vd=3, vc=5)]
  return (sfp_const(1, 1 / math.pi),
          [sfp_load(), ins("SFPABS", vd=0, vc=0, mod1=1), sfp_bf16(7, phase + 0.5), mad(6, 0, 1, 7),
           ins("SFPEXEXP", vd=2, vc=6), ins("SFPEXMAN", vd=3, vc=6),
           ins("SFPIADD", vd=2, vc=2, mod1=5, imm12=-23 & 0xFFF),
           ins("SFPSHFT", vd=3, vc=2), ins("SFPMOV", vd=4, vc=3), ins("SFPLOADI", vd=2, mod0=2, imm16=4095),
           ins("SFPAND", vd=4, vc=2), ins("SFPSHFT", vd=5, vc=3, mod1=5, imm12=-12 & 0xFFF),
           ins("SFPSHFT", vd=5, mod1=1, imm12=12),
           ins("SFPCAST", vd=4, vc=4), ins("SFPCAST", vd=5, vc=5), ins("SFPSHFT", vd=3, mod1=1, imm12=31),
           ins("SFPMOV", vd=6, vc=0),
           *[w for p in pi for w in (*sfp_const(2, p), mad(6, 5, 2, 6, neg=1), mad(6, 4, 2, 6, neg=1))],
           *([*sfp_const(2, half), mad(6, 6, ONE, 2), *sfp_const(2, math.pi / 2 - half),
              mad(6, 6, ONE, 2)] if phase else []),
           *sfp_bits(5, 0x80000000), CC_ON, *correct, CC_OFF,
           mul(7, 6, 6), *poly(4, 7, SIN_P, 2), mul(4, 4, 6), ins("SFPXOR", vd=4, vc=3),
           sfp_load(), ins("SFPEXEXP", vd=2, vc=0, mod1=1),
           *([ins("SFPAND", vd=0, vc=5), ins("SFPXOR", vd=4, vc=0)] if not phase else []),
           CC_ON, at_least(2, 127 + 24, 5), *sfp_bits(4, NAN), CC_OFF, sfp_store(4)])

SFPU = {
  # max(x, 0). SFPSWAP's min/max mode leaves the smaller value in vd and the larger in vc. It only picks a value, so the
  # raw store works for int32 too.
  "relu": ([], [sfp_load(), ins("SFPMOV", vd=1, vc=ZERO), ins("SFPSWAP", vd=1, vc=0, mod1=1), sfp_store(0, mod0=RAW)]),
  "exp": exp(False), "exp2": exp(True), "log": log(math.log(2)), "log2": log(1), "sin": sin(0.0), "cos": sin(0.5),
  "fexp": exp(False, True), "fexp2": exp(True, True), "recip": recip(), "frecip": recip(True), "sqrt": sqrt(),
}
# Binary ops on the SFPU. x and y reach Dst by being added to zero, x at the op's Dst half and y 128 rows further on.
# They load into LRegs 0 and 1, and the result is stored over x. Comparisons give 1.0 or 0: the all ones or 0 from
# SFPGT, ANDed with the bits of 1.0. x + 0 first turns -0 into 0. keep and drop are the two halves of where.
# keep(c, x) is x if c else 0, and drop(c, y) is y if not c else 0, where c is true when it isn't 0. They work on the
# bits, so an inf in the value that isn't picked doesn't make a NaN, as c * x would. c's test is on its bits too: an
# integer SFPABS makes every c but +-0 positive, a negative NaN included, which a float one leaves negative. An int32 c
# comes as 0 or 1 (Tensor.where), since its lowest value would count as -0.
def binary(*body): return [sfp_load(0), sfp_load(1, 128), *body]
# fp32 comparisons start from x + 0 and y + 0, which turn -0 into 0 and a NaN into 0x7FC00000, the top of SFPGT's
# order. -(-(y + 0) - 0) then gives y + 0 back, but its NaN negated, at the bottom: an SFPMAD's NaN is always
# 0x7FC00000, and subtracting 0 keeps a zero's sign. Where either is NaN, x > y holds and y > x doesn't, so one SFPGT
# makes each comparison.
UNORDERED = [mad(0, 0, ONE, ZERO), mad(1, 1, ONE, ZERO), mad(1, 1, MINUS_ONE, ZERO, neg=2),
             ins("SFPMOV", vd=1, vc=1, mod1=1)]
# Map two's complement to SFPGT's sign-magnitude order, including INT32_MIN: XOR a negative with its sign spread over
# the magnitude bits, by an arithmetic shift right by 31 and then a logical one by 1. -1 becomes negative zero, below
# positive zero. Applying the same map restores the input.
def integer_order(*regs):
  return [w for r in regs for w in (ins("SFPSHFT", vd=2, vc=r, mod1=7, imm12=-31 & 0xFFF),
                                    ins("SFPSHFT", vd=2, mod1=1, imm12=-1 & 0xFFF), ins("SFPXOR", vd=r, vc=2))]
BINARY = {"max", "cmplt", "cmple", "cmpne", "keep", "drop", "join"}
# In an fp32 kernel the SFPU also does add, sub and mul, rounding once as fp32 does. There, exp, the reciprocal and
# the comparisons handle NaN the IEEE way. The bf16 versions skip that work, since a bf16 NaN is already inf before the
# SFPU sees it.
FP32_OPS = {"add": "fadd", "sub": "fsub", "mul": "fmul", "exp": "fexp", "exp2": "fexp2", "recip": "frecip",
            "cmplt": "fcmplt", "cmple": "fcmple", "cmpne": "fcmpne"}
# int32's own programs
INT32_OPS = {"add": "iadd", "sub": "isub", "max": "imax", "cmplt": "icmplt", "cmple": "icmple", "cmpne": "icmpne",
             "keep": "keep", "drop": "drop", "join": "join"}
# the SFPU program for a cast
CASTS = {("int32", "fp32"): "itof", ("int32", "bf16"): "itof", ("fp32", "int32"): "ftoi", ("bf16", "int32"): "ftoi"}
SFPU |= {
  "max": ([], binary(ins("SFPSWAP", vd=1, vc=0, mod1=1), sfp_store(0))),
  "cmplt": ([],
            binary(mad(0, 0, ONE, ZERO), mad(1, 1, ONE, ZERO), ins("SFPGT", vd=1, vc=0, mod1=8),
                   ins("SFPAND", vd=1, vc=ONE), sfp_store(1))),
  "cmple": ([],
            binary(mad(0, 0, ONE, ZERO), mad(1, 1, ONE, ZERO), ins("SFPGT", vd=0, vc=1, mod1=8),
                   ins("SFPNOT", vd=0, vc=0), ins("SFPAND", vd=0, vc=ONE), sfp_store(0))),
  "cmpne": ([],
            binary(mad(0, 0, ONE, ZERO), mad(1, 1, ONE, ZERO), ins("SFPMOV", vd=2, vc=0),
                   ins("SFPGT", vd=2, vc=1, mod1=8), ins("SFPGT", vd=1, vc=0, mod1=8), ins("SFPOR", vd=1, vc=2),
                   ins("SFPAND", vd=1, vc=ONE), sfp_store(1))),
  "fcmplt": ([], binary(*UNORDERED, ins("SFPGT", vd=1, vc=0, mod1=8), ins("SFPAND", vd=1, vc=ONE), sfp_store(1))),
  "fcmple": ([],
             binary(*UNORDERED, ins("SFPGT", vd=0, vc=1, mod1=8), ins("SFPNOT", vd=0, vc=0),
                    ins("SFPAND", vd=0, vc=ONE), sfp_store(0))),
  "fcmpne": ([],
             binary(*UNORDERED, ins("SFPMOV", vd=2, vc=0), ins("SFPGT", vd=2, vc=1, mod1=8),
                    ins("SFPGT", vd=1, vc=0, mod1=8), ins("SFPOR", vd=1, vc=2), ins("SFPAND", vd=1, vc=ONE),
                    sfp_store(1))),
  "keep": ([],
           binary(ins("SFPABS", vd=2, vc=0), ins("SFPGT", vd=2, vc=ZERO, mod1=8), ins("SFPAND", vd=1, vc=2),
                  sfp_store(1, mod0=RAW))),
  "drop": ([], binary(ins("SFPABS", vd=2, vc=0), ins("SFPGT", vd=2, vc=ZERO, mod1=8), ins("SFPNOT", vd=2, vc=2),
                      ins("SFPAND", vd=1, vc=2), sfp_store(1, mod0=RAW))),
  # The two disjoint masks of where combine without arithmetic, preserving a selected negative zero.
  "join": ([], binary(ins("SFPOR", vd=0, vc=1), sfp_store(0, mod0=RAW))),
  "fadd": ([], binary(mad(0, 0, ONE, 1), sfp_store(0))), "fsub": ([], binary(mad(0, 1, ONE, 0, neg=1), sfp_store(0))),
  "fmul": ([], binary(mul(0, 0, 1), sfp_store(0))),
  # int32 stays two's complement in L1, Dst and LRegs. Ordering uses the reversible map above. A comparison gives 1, not
  # 1.0: 0 minus SFPGT's all ones, or for <=, x > y's all ones plus 1.
  "iadd": ([], binary(ins("SFPIADD", vd=0, vc=1, mod1=4), sfp_store(0, mod0=RAW))),
  "isub": ([], binary(ins("SFPIADD", vd=1, vc=0, mod1=6), sfp_store(1, mod0=RAW))),
  "imax": ([],
           binary(*integer_order(0, 1), ins("SFPSWAP", vd=1, vc=0, mod1=1), *integer_order(0), sfp_store(0, mod0=RAW))),
  "icmplt": ([], binary(*integer_order(0, 1), ins("SFPGT", vd=1, vc=0, mod1=8), ins("SFPIADD", vd=1, vc=ZERO, mod1=6),
                        sfp_store(1, mod0=RAW))),
  "icmple": ([],
             binary(*integer_order(0, 1), ins("SFPGT", vd=0, vc=1, mod1=8), ins("SFPIADD", vd=0, vc=0, mod1=5, imm12=1),
                    sfp_store(0, mod0=RAW))),
  "icmpne": ([],
             binary(ins("SFPMOV", vd=2, vc=0), ins("SFPGT", vd=2, vc=1, mod1=8), ins("SFPGT", vd=1, vc=0, mod1=8),
                    ins("SFPOR", vd=1, vc=2), ins("SFPIADD", vd=1, vc=ZERO, mod1=6), sfp_store(1, mod0=RAW))),
  # Casts. itof turns an integer into sign-magnitude, then into a float, rounding to nearest (exact below 2^24). ftoi
  # rounds a float toward 0 by shifting its mantissa, hidden bit included, by its exponent - 23. It gives 0 where the
  # exponent is negative, and negates the result for a negative float.
  "itof": ([], [sfp_load(), *sfp_bits(1, 0x80000000), ins("SFPXOR", vd=1, vc=0),
                ins("SFPCAST", vd=0, vc=0, mod1=3), ins("SFPCAST", vd=0, vc=0),
                CC_ON, ins("SFPSETCC", vc=1, mod1=6), *sfp_const(0, -2**31), CC_OFF, sfp_store(0)]),
  "ftoi": ([],
           [sfp_load(), ins("SFPEXEXP", vd=1, vc=0), ins("SFPEXMAN", vd=2, vc=0),
            ins("SFPIADD", vd=1, vc=1, mod1=5, imm12=-23 & 0xFFF), ins("SFPSHFT", vd=2, vc=1), CC_ON, below(1, -23, 4),
            ins("SFPMOV", vd=2, vc=ZERO), CC_ALL, ins("SFPSETCC", vc=0), ins("SFPIADD", vd=2, vc=ZERO, mod1=6), CC_OFF,
            sfp_store(2, mod0=RAW)]),
  "one": ([], [sfp_store(ONE)]),  # 1.0 everywhere (a tile of ones)
  "none": (sfp_const(1, -math.inf), [sfp_store(1)]),  # -inf everywhere (a mask that hides all keys)
  # rounds an fp32 value to bf16 the way the packer does, half up: add 2^15 to the bits and clear the low 16. the
  # result stays fp32
  "ftob": ([ins("SFPLOADI", vd=2, mod0=2, imm16=0x8000), *sfp_bits(3, 0xFFFF0000)],
           [sfp_load(), ins("SFPIADD", vd=0, vc=2, mod1=4), ins("SFPAND", vd=0, vc=3), sfp_store(0, mod0=RAW)]),
}
# Reductions along an axis. Axis 0 reduces the batch to a matrix, axis 1 the rows to a row, and axis 2 the columns to
# a column. The tiles along the axis go into Dst as x + 0. The first lands in the reduction's accumulator, and the rest
# land 128 rows past it, where the SFPU folds them in. Then the rows or columns of the tile are folded across their 32
# values, and the result is written to all 32, so the tile broadcasts along that axis as it is. Over the batch, the
# tiles were already folded as they came, so there's nothing left to do.
FOLDS = {f"{f}{ax}": ax for f in ("sum", "max") for ax in range(3)}  # kind -> the axis it folds
def fold(kind, d, t):  # LReg d = d op t (for max, t is left with the min)
  return ins("SFPSWAP", vd=t, vc=d, mod1=1) if kind[:3] == "max" else mad(d, t, ONE, d)
COMBINE = {k: binary(fold(k, 0, 1), sfp_store(0)) for k in FOLDS}
def across_rows(kind):
  """Fold the rows of the tile across their 32 columns. Columns 0-15 of row r are in Dst row (r // 16) * 32 + r % 16,
  and columns 16-31 are 16 rows further on. An SFPLOAD takes the even or odd columns of 4 such rows, one row per 8-lane
  group. Folding the four loads leaves 8 lanes for each row. Rotating the group by 1, 2 and 4 lanes and folding each
  time leaves the row's result in all 8 lanes, and the stores write it back to all four positions.

  A sum's fold is an SFPMAD, whose result takes two cycles, and the stall logic doesn't see that SFPSHFT2's rotation
  reads it, so an SFPNOP goes between them (ISA SFPMAD.md, "Instruction scheduling"). A max's SFPSWAP stalls by
  itself."""
  rot = lambda d, s: ins("SFPSHFT2", vd=d, vc=s, mod1=3)
  gap = [ins("SFPNOP")] * (kind[:3] == "sum")
  p = list(SFPU_START)
  for base in (0, 4, 8, 12, 32, 36, 40, 44):
    at = [base, base + 2, base + 16, base + 18]  # even and odd columns, left and right face
    p += [sfp_load(r, a) for r, a in enumerate(at)]
    p += [fold(kind, 0, 1), fold(kind, 2, 3), fold(kind, 0, 2), *gap]
    p += [rot(1, 0), fold(kind, 0, 1), *gap, rot(1, 0), rot(1, 1), fold(kind, 0, 1), *gap,
          rot(1, 0), rot(1, 1), rot(1, 1), rot(1, 1), fold(kind, 0, 1)]
    p += [sfp_store(0, at=a, step=False) for a in at]
  return p
def across_cols(kind):
  """Fold the columns of the tile across their 32 rows. Column c is lane c % 16 // 2 of the loads at the even
  addresses (for even c) or the odd ones of its two faces, the left faces for c < 16. An 8-lane group holds a row of
  the column. The 8 loads fold into one register. Three copies and an SFPTRANSP move its 4 lane groups into 4
  registers, and folding those leaves the column's result in its lane of all 4 groups. The stores write it back to all
  8 positions."""
  p = list(SFPU_START)
  for half, odd in [(h, o) for h in (0, 16) for o in (0, 2)]:
    # rows 0-15 of the top face, then the bottom face
    at = [half + odd + base for base in (0, 4, 8, 12, 32, 36, 40, 44)]
    p += [sfp_load(0, at[0])]
    p += [w for a in at[1:] for w in (sfp_load(1, a), fold(kind, 0, 1))]
    p += [ins("SFPMOV", vd=r, vc=0) for r in (1, 2, 3)] + [ins("SFPTRANSP")] + [fold(kind, 0, r) for r in (1, 2, 3)]
    p += [sfp_store(0, at=a, step=False) for a in at]
  return p
ACROSS = {0: lambda kind: [], 1: across_cols, 2: across_rows}
def masking(kind):  # an SFPU start and the reduction's identity: 0 in LReg 9, or -inf in LReg 5 for max
  return (list(SFPU_START), ZERO) if kind[:3] == "sum" else (SFPU_START + sfp_const(5, -math.inf), 5)
def mask_columns(kind, n, d=0):
  """Set columns n and up of the tile at Dst row d to the reduction's identity, 0 or -inf for max, so the padding past
  a tensor's last column doesn't count. An SFPSTORE writes 4 rows of 8 even or odd columns, with lane l at column
  2 (l % 8), plus 1 if odd. LReg 15 holds 2l, so LReg 7 = LReg 15 & 14 is twice the lane's position in its row. Each
  store goes only to the lanes whose column is n or more."""
  p, identity = masking(kind)
  p += sfp_bits(7, 14) + [ins("SFPAND", vd=7, vc=15)]
  for right, odd in [(r, o) for r in (0, 1) for o in (0, 1)]:
    t = max(0, min(8, -(-(n - 16 * right - odd) // 2)))  # the first of the 8 lanes at column n or more
    if t < 8:  # store to lanes t and up at the face's rows. a right face is 16 rows further on. stores keep the flags
      p += [CC_ON, at_least(7, 2 * t, 6)] + [sfp_store(identity, at=d + b + 16 * right + 2 * odd, step=False)
                                              for b in (0, 4, 8, 12, 32, 36, 40, 44)]
  return p + [CC_OFF]
def mask_rows(kind, m, d=0):
  """Set rows m and up of the tile at Dst row d to the identity, past a tensor's last row. A store covers 4 rows, with
  lane l in row l // 8 of them. LReg 15 holds 2l, so LReg 15 - 16k >= 0 picks the rows from k on."""
  p, identity = masking(kind)
  for r in range(0, 32, 4):  # rows r..r+3 start at Dst row (r // 16) * 32 + r % 16 in the left face, 16 on in the right
    k = max(0, m - r)
    if k < 4:
      p += [CC_ON, at_least(15, 16 * k, 6)] + [sfp_store(identity, at=d + r // 16 * 32 + r % 16 + o, step=False)
                                               for o in (0, 2, 16, 18)]
  return p + [CC_OFF]
def mask_rows_at(m):
  """Generate mask_rows("sum", m) with m supplied as a C expression at run time. m is the number of valid rows
  in the final tile, from 1 to 31. Compute instruction immediates for each group of four rows before pushing
  them."""
  p, identity = masking("sum")
  cut, store = at_least(15, 0, 6), sfp_store(identity, step=False)
  imm12, imm10 = bh.TENSIX_OPS["SFPIADD"][1]["imm12"][0], bh.TENSIX_OPS["SFPSTORE"][1]["imm10"][0]
  return (f"{tt(p)} for (unsigned r = 0; r < 32; r += 4) if ({m} < r + 4) {{ unsigned k = {m} > r ? {m} - r : 0; "
          f"push_word({CC_ON:#x}); push_word({cut:#x} | (-16 * k & 0xFFF) << {imm12}); for (unsigned o = 0; "
          f"o < 4; o++) "
          f"push_word({store:#x} | (r / 16 * 32 + r % 16 + o % 2 * 2 + o / 2 * 16) << {imm10}); }} {tt([CC_OFF])}")
MASK = {1: mask_rows, 2: mask_columns}  # masks the padding past the tensor's end, in its last tile along an axis
# mode 1 steps the phase
MUL_PHASE = [ins("SETRWC", bit_mask=0b0111)] + [ins("ELWMUL", addr_mode=int(i == 7)) for i in range(8)]
MATH_RUNS = {"mm": matmul_phase(), "mmT": matmul_phase(True),
             "mul": MUL_PHASE} | {u: body for u, (_, body) in SFPU.items()} | COMBINE

def allocate(names):
  """Assign runs to the math thread's replay buffer, as {name: (index, words recorded)}. Prioritize SFPU bodies
  because they repeat 32 times per tile, a reduction's folds first, since they run on every tile it sweeps and the
  ops on its result once. Then allocate the longest first. Matmul phases repeat up to four times a step and mul's
  four times a tile, and they get the remaining space. The first run that doesn't fit records the words that still
  do, so its TRISC pushes only the rest, after a REPLAY. Other runs that don't fit are pushed one word at a time."""
  slots, at = {}, 0
  # the name breaks ties, so different processes build the same kernel
  order = sorted(names, key=lambda n: (n in ("mm", "mmT", "mul"), n not in COMBINE, -len(MATH_RUNS[n]), n))
  for name in order:
    if at + len(MATH_RUNS[name]) <= 32: slots[name], at = (at, len(MATH_RUNS[name])), at + len(MATH_RUNS[name])
  rest = next((name for name in order if name not in slots), None)
  if rest and at < 32: slots[rest] = (at, 32 - at)
  return slots
def run(slots, name, count=None):  # the first count words of a run: a REPLAY of those recorded, then the others
  words = MATH_RUNS[name][:count]
  if name not in slots: return words
  at, n = slots[name]
  return [replay(at, min(n, len(words)))] + words[n:]
# a C loop is fine, since the math TRISC runs ahead of its pushes
def runs(slots, name, times): return [Loop(times, run(slots, name))]
# recorded at setup
def math_runs(slots): return [w for name, (at, n) in slots.items() for w in record(at, MATH_RUNS[name][:n])]

def release(word):  # the same ELW*/MVMUL with clear_dvalid set, so both Src banks go back to the unpackers
  first, width = bh.TENSIX_OPS["MVMUL"][1]["clear_dvalid"]
  return word | ((1 << width) - 1) << first
def phased(slots, name, phases):  # a run of MATH_RUNS for each fidelity phase. its last word releases both Src banks
  words = MATH_RUNS[name]
  return ([ins("SETRWC", bit_mask=0b1111)] + run(slots, name) * (phases - 1) + run(slots, name, len(words) - 1)
          + [release(words[-1])])
# The address mode slots that the walk of a transposed B rewrites (see matmul_phase), as (SrcA step, SrcB step). The
# rest of the time they hold what ADDR_MODES says.
TRANSPOSED = {4: (32, 0), 5: (32, 8), 6: (48, 8), 2: (16, 8)}
def walk(transposed):  # sets the math thread's address modes for the steps of a transposed B, or back
  return [ins("SETC16", setc16_reg=bh.ADDR_MOD_AB + m, setc16_value=a | b << 8)
          for m, (a, b) in ((m, TRANSPOSED[m] if transposed else ADDR_MODES[m][0]) for m in sorted(TRANSPOSED))]

def matmul(slots, phases=4, transposed=False):
  """One matmul step with its first `phases` fidelity phases. SETRWC starts each step at phase 0.

  A phase multiplies some of the 8 significand bits of A (in SrcB) by some of B's (in SrcA) (ISA SrcASrcB.md):
    phase 0: A's top 7 bits by B's top 5
    phase 1: A's top 7 bits by B's last 3
    phase 2: A's last bit by B's top 5
    phase 3: A's last bit by B's last 3
  Four phases leave nothing out, and three leave out under 2^-11 of the product. Two drop A's last bit, which truncates
  by 2^-7 at most. One keeps only 5 bits of B."""
  return phased(slots, "mmT" if transposed else "mm", phases)

def elementwise(op, slots, dst=0):
  """Math for a tile of SrcA op SrcB at Dst row dst, 8 rows of 16 per instruction. Address mode 0 steps 8 rows.
  ELWMUL always accumulates, and each fidelity phase multiplies a slice of the mantissas, so mul runs all 4 phases on
  a zeroed Dst."""
  if op == "mul": return phased(slots, "mul", 4)
  fpu = {"add": "ELWADD", "sub": "ELWSUB"}[op]
  return [ins("SETRWC", bit_mask=0b1111)] + [ins(fpu, dst=dst) for _ in range(7)] + [release(ins(fpu, dst=dst))]

def reduce_step(kind, slots, acc, first, edge=0, src=None):
  """Accumulate one tile of a reduction. Load bf16 through SrcA plus zeros in SrcB, or fill fp32 from L1 slot
  src. Put the first tile in the accumulator after waiting for it to become free. Load subsequent tiles into
  scratch and combine them with the SFPU. For a partial final tile, edge gives the number of valid rows or
  columns along the reduction axis."""
  row, sem = ACCS[acc]
  mask = lambda d: MASK[FOLDS[kind]](kind, edge, d) if edge else []
  put = lambda d: fill(src, row + d) + FILLED if src is not None else elementwise("add", slots, dst=d)
  if first: return take(sem) + aim(row) + put(0) + mask(0)
  return aim(row) + put(128) + mask(128) + SFPU_START + runs(slots, kind, 32)
# folds across the tile and runs the SFPU ops on the result, before MATH_END
def reduce_finish(kind, slots, acc, posts):
  return [DST_FREE, *aim(ACCS[acc][0])] + ACROSS[FOLDS[kind]](kind) + sfpu(posts, slots)

def sfpu(posts, slots):  # runs the SFPU ops in order, over the Dst half that the FPU has written
  return [w for u in posts for w in SFPU_START + SFPU[u][0] + runs(slots, u, 32)]

def table(name, words): return f"static const unsigned {name}[] = {{{', '.join(f'{w:#x}' for w in words)}}};"
BANKS = table("bank", [bh.noc_xy(x, y) for x, y in bh.DRAM]) + f"""
#define BANK(t) bank[(t) % {len(bh.DRAM)}] & 63, bank[(t) % {len(bh.DRAM)}] >> 6  // x, y of the bank that holds tile t
#define AT(base, t, size) ((base) + (t) / {len(bh.DRAM)} * (size))  // tile t's address in that bank, for tiles of size bytes
static inline unsigned at(unsigned r, unsigned c) {{ return (r / 16 * 2 + c / 16) * 256 + r % 16 * 16 + c % 16; }}  // datum (r, c) in a tile"""
# Runtime.readers gives some workers a DRAM bank and a port to read it through. NEAR makes BANK use a copy of the bank
# table, and the kernel first points the entry for the worker's bank at that port.
NEAR = f"""static unsigned near[{len(bh.DRAM)}];
#undef BANK
#define BANK(t) near[(t) % {len(bh.DRAM)}] & 63, near[(t) % {len(bh.DRAM)}] >> 6"""
# C shared by the kernels that run on BRISC alone: the banks, copies within this L1, and copying a row down its tile
ROWS = BANKS + """
// A copy within this L1. This tile's coordinates take an NIU register read, so it reads them once a launch.
static void copy(unsigned from, unsigned to, unsigned n) {
  static unsigned me;
  if (!me) me = noc_me();
  noc_get(me & 63, me >> 6, from, to, n);
}
// copies row 0 of a tile of item-byte values down the tile: down the 16 rows of faces 0 and 1, then faces 0-1 onto 2-3
static void row_down(unsigned tile, unsigned item) {
  unsigned face = 256 * item;
  l1_settle(tile);
  for (unsigned n = 16 * item; n < face; n *= 2) { copy(tile, tile + n, n); copy(tile + face, tile + face + n, n); noc_wait(); }
  copy(tile, tile + 2 * face, 2 * face);
  noc_wait();
}"""
# tile (b, r, c) of the tensor at a[i]
def tile_of(i, b, r, c, a="a"): return f"({b}) * {a}[{i} + 1] + ({r}) * {a}[{i} + 2] + ({c}) + {a}[{i} + 3]"

def tt(words):
  """C for a TRISC that pushes words to its thread as .ttinsn instructions. Each word is rotated left by 2 and written
  into the code, so the TRISC stalls while the thread's FIFO is full. A Loop becomes a C loop. A (word, expression)
  becomes a store of the word plus the value of the expression, worked out at run time."""
  c, pending = [], []
  for w in list(words) + [None]:
    if isinstance(w, int):
      if w >= 0xC0000000: raise ValueError(f"{w:#x} isn't a Tensix instruction")
      pending.append(f"{(w << 2 | w >> 30) & 0xFFFFFFFF:#x}")
      continue
    if pending: c.append(f'asm volatile(".word {", ".join(pending)}" ::: "memory");')
    pending = []
    if isinstance(w, tuple): c.append(f"push_word({w[0]:#x} + ({w[1]}));")
    elif w is not None:
      c.append(f'_Pragma("clang loop unroll(disable)") for (unsigned r_ = 0; r_ < {w.times}; r_++) {{ {tt(w)} }}')
  return " ".join(c)

FLAGS_B = 16  # FLAGS has A's flags, for the workers in a grid row, then from here B's, for the workers in a grid column
# The most matmul steps that can have reads in flight. Each reads into a stream slot of B, and of A if A streams.
DEPTH = 4
# Streaming a B laid out by columns, for a token's row: chunks of at most CHUNK bytes of values (a bfp4 chunk has 8 KiB,
# since B8_SLOTS caps its tiles), with CHUNKS chunks in flight
CHUNK, CHUNKS = bh.MAX_PACKET, 4
def chunk_tiles(wt):  # tiles in a chunk: 8 of bf16, or 16 of a block float
  # CHUNKS chunks make a region, and the first datums of the region's slots must stay below BFP_FIRST_DATUM
  return min(CHUNK // (BFP[wt] if wt else tile_bytes("bf16")), B8_SLOTS // CHUNKS)
# Slot 0 is a zero tile, then come the stream slots of B and of A. From FRAMES on come two frames, the narrow ops, and
# what stays in L1: A's rows, B's column or buffers, and kept inputs.
ZERO_SLOT, FRAMES = 0, 1 + 2 * DEPTH
UNPACKABLE = bh.UNPACKABLE  # the slots an unpacker reads are below this
# A matmul's block-float B goes past the slots, in a region, or in three for the buffers of mc. Each region holds the
# exponents of its slots, then their values, which is the layout the unpacker wants. A region has at most B8_SLOTS
# slots, because the unpacker finds a slot's exponents from its first datum, which must stay below bh.BFP_FIRST_DATUM.
B8 = slot_base("bf16") + UNPACKABLE * tile_bytes("bf16")
B8_SLOTS = bh.BFP_FIRST_DATUM // 1024

# SYNC holds what BRISC and the TRISCs tell each other. Each word has one writer, and the firmware zeroes the counters
# for each launch.
#   FETCHED: frames that BRISC has fetched into L1
#   ISSUED: matmul steps whose reads BRISC has issued
#   UNPACKED: frames and steps that thread 0 has unpacked, written by TRISC0
#   READY: set by TRISC0 once the backend config and semaphores are set up
#   PACKED: results that the pack thread has packed, written by TRISC2. RING says where they go, in four [slot, tile]
#     entries.
#   SENT: packed results that BRISC has written to DRAM
SYNC_WORDS = f"""
enum {{ {", ".join(name.upper() for name in SYNC_NAMES)}, RING = 8 }};
#define SYNC ((volatile unsigned *){SYNC:#x})
static unsigned peek(unsigned i) {{ asm volatile("fence" ::: "memory"); return SYNC[i]; }}  // fenced: L0 cache doesn't see other cores' writes
"""
TRISC = SYNC_WORDS + f"""
#define SEM(s) (((volatile unsigned *){bh.TENSIX_SEM:#x})[__builtin_ctz(s)])  // load: its value. store 0: post, 1: take one
// Words that REPLAY records go through stores, since fused .ttinsn instructions can corrupt a recording. Words worked
// out at run time go through stores too.
static inline void push_word(unsigned w) {{ *(volatile unsigned *){bh.TENSIX_PUSH:#x} = w; }}
static unsigned posted, drained;  // intermediates that pack will post to L1, and those waited for
static void drain(void) {{ for (; drained < posted; drained++) {{ {tt(TAKE_L1)} }} }}  // waits for everything packed so far to land
"""
# The programs of a Tensix kernel read their arguments through a pointer that isn't volatile. The arguments are fixed
# for the launch, and this core sees them once the firmware has fenced, so the compiler can keep them in registers and
# out of loops.
KMAIN = "void kmain(volatile unsigned *args) {\n  const unsigned *a = (const unsigned *)args;"
def _pushes(words): return " ".join(f"push_word({w:#x});" for w in words)  # unrolled, with no table to load from

def brisc_program(total, body, dtype="bf16", out="bf16", decls="", tail="", b8=0, wt="bfp8", outs=OUTS, b8_at=None,
                  near=False):
  # BRISC reads inputs into L1 before unpacking and writes results to DRAM after packing.
  #   decls: statics and tables that the body needs. tail: the body's last words.
  #   b8: slots of block floats wt, a matmul's B, at b8_at (B8 by default).
  #   outs: where results in another dtype are packed. near: BANK goes through NEAR.
  tb, to, banks = tile_bytes(dtype), tile_bytes(out), BANKS + ("\n" + NEAR if near else "")
  return SYNC_WORDS + f"""
#define SLOT(i) ({slot_base(dtype):#x} + (i) * {tb})  // where the unpackers read slot i
#define OUT_SLOT(i) ({slot_base(dtype) if out == dtype else outs:#x} + (i) * {to})  // and where the packer writes it
{banks}
static unsigned fetched, events, frame_done[2], stream_done[{DEPTH}], steps, sent, out;
static unsigned rows[16];  // for each place in L1, i + 1 of the row it holds (see fresh)
// fetches tile t of a tensor at DRAM address base (see BANK and AT)
static void get(unsigned base, unsigned t, unsigned slot) {{ noc_get(BANK(t), AT(base, t, {tb}), SLOT(slot), {tb}); }}
// the same, with its two off-diagonal faces swapped. that's half of a transpose, and unpacker 0 does the rest
static void flipped(unsigned base, unsigned t, unsigned slot) {{
  for (unsigned f = 0; f < 4; f++)
    noc_get(BANK(t), AT(base, t, {tb}) + f * {tb // 4}, SLOT(slot) + (f == 1 ? 2 : f == 2 ? 1 : f) * {tb // 4}, {tb // 4});
}}
static void ready(void) {{ noc_wait(); SYNC[FETCHED] = ++fetched; }}
static void send(void) {{  // write completed results to DRAM in pack order
  for (; sent < peek(PACKED); SYNC[SENT] = ++sent) {{
    unsigned slot = peek(RING + sent % 4 * 2), t = peek(RING + sent % 4 * 2 + 1);
    noc_put(OUT_SLOT(slot), BANK(t), AT(out, t, {to}), {to}); noc_wait();
  }}
}}
// A slot gets fetched into again once thread 0 has unpacked what was there. For a frame that's the frame two back,
// and for a stream slot the step DEPTH steps back. Thread 0 reports each frame and step in turn, and events counts them
// the same way. BRISC writes results to DRAM while waiting.
static void reached(unsigned n) {{ while (peek(UNPACKED) < n) send(); }}
// Fetches block-float tile t into slot s past B8, which is slot s % b8 of region s / b8. Its 64 exponents go among the
// region's exponents, and its values among their values.
static void getb(unsigned base, unsigned t, unsigned s) {{
  unsigned at = AT(base, t, {tile_bytes(wt)}), r = {b8_at or B8:#x} + s / {max(b8, 1)} * {b8 * tile_bytes(wt)}, z = s % {max(b8, 1)};
  noc_get(BANK(t), at, r + 64 * z, 64);
  noc_get(BANK(t), at + 64, r + {64 * b8} + {BFP[wt]} * z, {BFP[wt]});
}}
// A matmul step fetches B's tile (for SrcA) into b_slot and A's into a_slot. Each is a stream slot, or the tile's place
// in a row or column kept in L1. KEPT means the tile is there already. With bfp, B is a block float (getb). Otherwise
// it's bf16, and a transposed B's tile comes as it lies. The reads of step k carry transaction id k % DEPTH + 1, the id
// of its stream slots. BRISC counts the steps whose reads are out (ISSUED), and it's TRISC0 that waits for the reads
// with that id to land, so up to DEPTH steps can have reads in flight. A frame's reads carry id 0.
#define KEPT 0xFFFFFFFF
static unsigned read_id;  // the transaction id that initiator 1 puts on reads
static void tagged(unsigned id) {{ if (id != read_id) {{ while (NIU1(CMD) & 1) {{}} NIU1(PACKET_TAG) = (read_id = id) << 10; }} }}
static void step(unsigned bfp, unsigned b, unsigned b_tile, unsigned b_slot, unsigned a, unsigned a_tile, unsigned a_slot) {{
  unsigned s = steps % {DEPTH};
  reached(stream_done[s]);  // waits until the step that last used slot s has been unpacked
  tagged(s + 1);
  if (b_slot != KEPT) {{ if (bfp) getb(b, b_tile, b_slot); else get(b, b_tile, b_slot); }}
  if (a_slot != KEPT) get(a, a_tile, a_slot);
  (void)NIU1(CMD); asm volatile("fence" ::: "memory");  // the NIU must count the reads before TRISC0 sees ISSUED
  stream_done[s] = ++events;
  SYNC[ISSUED] = ++steps;
}}
// B by columns (Tensor.columns). getc and getbc fetch n tiles, from tile k on, of the run at `run` (len tiles long) in
// bank j % 8. They go into slots from `slot`, or from s for a block float, whose exponents and values come from the two
// parts of the run. No read is over MAX_PACKET. The n tiles are a matmul step's tile, a chunk of the stream for a
// token's row, or a grid worker's share of a column.
static void noc_gets(unsigned x, unsigned y, unsigned addr, unsigned l1, unsigned n) {{
  for (unsigned k; n; addr += k, l1 += k, n -= k) noc_get(x, y, addr, l1, k = n < MAX_PACKET ? n : MAX_PACKET);
}}
static void getc(unsigned j, unsigned run, unsigned k, unsigned slot, unsigned n) {{ noc_gets(BANK(j), run + k * {tb}, SLOT(slot), n * {tb}); }}
static void getbc(unsigned j, unsigned run, unsigned len, unsigned k, unsigned s, unsigned n) {{
  unsigned r = {b8_at or B8:#x} + s / {max(b8, 1)} * {b8 * tile_bytes(wt)}, z = s % {max(b8, 1)};
  noc_gets(BANK(j), run + 64 * k, r + 64 * z, 64 * n);
  noc_gets(BANK(j), run + 64 * len + {BFP[wt]} * k, r + {64 * b8} + {BFP[wt]} * z, {BFP[wt]} * n);
}}
// step, with its B tile from a B by columns
static void stepc(unsigned bfp, unsigned j, unsigned run, unsigned len, unsigned k, unsigned b_slot, unsigned a, unsigned a_tile, unsigned a_slot) {{
  if (b_slot != KEPT) {{
    reached(stream_done[steps % {DEPTH}]);  // like step: wait for the slot and set the step's id before its reads
    tagged(steps % {DEPTH} + 1);
    if (bfp) getbc(j, run, len, k, b_slot, 1); else getc(j, run, k, b_slot, 1);
  }}
  step(bfp, 0, 0, KEPT, a, a_tile, a_slot);
}}
// A chunk of a chunked stream. chunk_part waits until the unpacker is done with the chunk that last used the next part
// of the ring, has reads carry that part's transaction id, and returns the part. Once the chunk's reads are out,
// chunk_issued marks its k steps issued.
static unsigned part_done[{CHUNKS}], chunks;
static unsigned chunk_part(void) {{ unsigned p = chunks++ % {CHUNKS}; reached(part_done[p]); tagged(p + 1); return p; }}
static void chunk_issued(unsigned p, unsigned k) {{
  (void)NIU1(CMD); asm volatile("fence" ::: "memory");  // the NIU must count the reads before TRISC0 sees ISSUED
  part_done[p] = events += k;
  SYNC[ISSUED] = steps += k;
}}
// A row of A, or a column of B, stays in L1 at place q while the result tiles use it. When i, the row it holds,
// changes, the new one comes in once the old one is unpacked.
static unsigned fresh(unsigned q, unsigned i) {{
  if (rows[q] == i + 1) return 0;
  reached(events);
  return rows[q] = i + 1;
}}
static void kept_step(void) {{ SYNC[ISSUED] = ++steps; ++events; }}  // a matmul step whose tiles are in L1 already
// Tiles shared between workers by multicast. FLAGS counts the shares that the workers in this one's grid row have sent
// it (A's rows), then those from the workers in its grid column (B's columns). Workers zero their flags at the end
// (unflag). By then all the flags of the launch have arrived, and nobody writes one again before the next go.
#define FLAGS ((volatile unsigned *){FLAGS:#x})
static unsigned col_end[3];  // events at the end of the last column in each B buffer
static void await(unsigned i, unsigned n, unsigned v) {{  // wait for flags i..i+n-1 to reach v while writing results to DRAM
  for (unsigned k = i; k < i + n; k++)
    for (;;) {{ asm volatile("fence" ::: "memory"); if (FLAGS[k] >= v) break; send(); }}
}}
static void unflag(unsigned a, unsigned b) {{
  for (unsigned k = 0; k < a; k++) FLAGS[k] = 0;
  for (unsigned k = 0; k < b; k++) FLAGS[{FLAGS_B} + k] = 0;
  l1_settle({FLAGS:#x});
}}
// Multicasts n bytes of this L1 to the same address in the other `tiles` workers in rect (end | start << 12), and waits
// until they all have them.
static void mcast(unsigned l1, unsigned n, unsigned rect, unsigned tiles) {{
  noc_wait();
  mcast_begin();
  unsigned acks = NIU(WR_ACKS) + (n + MAX_PACKET - 1) / MAX_PACKET * tiles;
  l1_settle(l1);
  for (unsigned k; n; l1 += k, n -= k) noc_issue(l1, noc_me(), l1, rect, MCAST, k = n < MAX_PACKET ? n : MAX_PACKET);
  mcast_done(acks);
}}
// Sends this worker's part of what a row or column of the grid shares: n bytes at l1, already read in. They go to the
// same address in the other `tiles` workers in rect, and then flag, set to v, goes the same way. With no other workers, it
// waits for the reads and sets the flag.
static void share(unsigned l1, unsigned n, unsigned rect, unsigned tiles, unsigned flag, unsigned v) {{
  if (tiles && n) mcast(l1, n, rect, tiles);
  else noc_wait();
  FLAGS[flag] = v;
  if (tiles) mcast((unsigned)&FLAGS[flag], 4, rect, tiles);
}}
{decls}
{KMAIN}
  out = a[2];
  unsigned tiles = 0;
{body}
  {tail}
  while (sent < {total}) send();
}}"""

# TRISC0 sets up the backend config and semaphores, then unpacks tiles as BRISC fetches them.
def unpack_program(body, cfg, decls=""):
  return TRISC + decls + f"""
#define CFG ((volatile unsigned *){bh.TENSIX_CFG:#x})
static unsigned needed, relayed, stepped;
static void relay(void) {{ for (; SEM({SEM_UNPACKED}); SYNC[UNPACKED] = ++relayed) SEM({SEM_UNPACKED}) = 1; }}  // hands unpack reports to BRISC
static void fetched(void) {{ needed++; do relay(); while (peek(FETCHED) < needed); }}  // waits for BRISC to fetch the next frame
// Waits until the next matmul step's tiles are in, and returns its stream slot. BRISC must have issued the step's
// reads, and the reads with its transaction id must have landed. Also, at most 15 reports, the most that SEM_UNPACKED
// holds, may be pushed but not relayed. A step kept in L1 waits for no fetch, so thread 0 could fall further behind,
// and a post onto a full semaphore breaks the contract.
static unsigned landed(void) {{
  unsigned z = stepped++ % {DEPTH};
  do relay(); while (peek(ISSUED) < stepped || needed + stepped - relayed > 15);
  while (NIU(OUTSTANDING + 4 + 4 * z)) relay();
  return z;
}}
// The same for step m of a chunked stream, a column of n-tile chunks. At the first step of a chunk, it also waits for
// the chunk's reads, with the transaction id of its ring part, to land. Returns the step's ring slot.
static unsigned part, parts;
static unsigned landed_chunk(unsigned m, unsigned n) {{
  if (m % n == 0) part = parts++ % {CHUNKS};
  stepped++;
  do relay(); while (peek(ISSUED) < stepped || needed + stepped - relayed > 15);
  if (m % n == 0) while (NIU(OUTSTANDING + 4 + 4 * part)) relay();
  return part * n + m % n;
}}
{KMAIN}
  {" ".join(f"CFG[{i}] = {v:#x};" for i, v in cfg.items())}
  l1_settle((unsigned)&CFG[{next(iter(cfg))}]);  // the config writes are in, so no instruction pushed below sees them stale
  {tt(thread() + SETUP_UNPACK)}
  tensix_idle();  // the semaphores are set before the other threads can touch them
  SYNC[READY] = 1;
  unsigned tiles = 0;
{body}
  while (relayed < needed) relay();  // takes the remaining reports, whichever thread unpacked
}}"""

# TRISC1. With dst, it does the unpacking into Dst, so it waits for BRISC's fetches too.
def math_program(slots, body, dst=False, decls=""):
  return TRISC + decls + f"""
static unsigned needed;
static void fetched(void) {{ needed++; while (peek(FETCHED) < needed) {{}} }}  // waits for BRISC to fetch the next tiles
{KMAIN}
  {tt(thread(dst) + (counters(bh.UNPACK0, 1024) if dst else []) + [CC_OFF])} {_pushes(math_runs(slots))}
  while (!peek(READY)) {{}}  // waits for TRISC0 to set up the backend config and the semaphores
  unsigned tiles = 0, h = 0;  // h: the first row of the Dst half that the next op gets
{body}
}}"""

# TRISC2. Reuse an output slot after BRISC has written its previous result to DRAM.
def pack_program(body, slots=2, decls=""):
  return TRISC + decls + f"""
static unsigned outputs, relayed, used[{slots}];
static void relay(void) {{ for (; SEM({SEM_PACKED}); SYNC[PACKED] = ++relayed) SEM({SEM_PACKED}) = 1; }}  // hands packed results to BRISC
// Queue the result that the caller packs next into L1 slot, tracked as output slot s, for DRAM tile t. Frames may share an output slot.
// Wait until BRISC has written the previous result from this slot and RING has space, then queue the next result.
// RING holds at most four results awaiting transfer.
static void output(unsigned s, unsigned slot, unsigned t) {{
  while (peek(SENT) < used[s] || peek(SENT) + 4 <= outputs) relay();
  SYNC[RING + outputs % 4 * 2] = slot; SYNC[RING + outputs % 4 * 2 + 1] = t;
  used[s] = ++outputs;
}}
{KMAIN}
  {tt(thread() + SETUP_PACK)} {_pushes(PACK_RUNS)}
  while (!peek(READY)) {{}}
  unsigned tiles = 0, h = 0;  // h: the first row of the Dst half that the next op gets
{body}
  while (relayed < outputs) relay();
}}"""
