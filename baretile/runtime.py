# Load and launch kernels on Tensix tiles. Resident firmware starts each kernel without a core reset.
# A dispatcher reads commands from a ring in host memory and sends them to the workers.
import atexit, bisect, collections, functools, hashlib, operator, os, pathlib, shutil, struct, subprocess, sys, tempfile, time, weakref
from baretile import bh, pci

# The L1 layout, which is ours to choose. Firmware sits at 0 and starts with a jump for each core: BRISC's at 0 and
# TRISC i's at 4 + 4i. A worker has, in order:
# - MAILBOX: [go, kernel, arguments, workers] [done, dispatcher's x | y << 6, scratch, index]. The dispatcher
#   multicasts the first half.
# - TRISCS: [go phase, done 0, 1, 2], alternating 0/1 for each launch that this worker runs
# - MCAST_STATE: [cancel requested, multicast active], used to stop between acknowledged multicasts
# - SYNC: words that the cores of a kernel talk through. Its counters are zeroed before each launch.
# - FLAGS: 32 words that the workers of a kernel use to signal each other over the NoC. Each word is written by one
#   worker, and zeroed by the worker that holds it at the end.
# - ARGS: two buffers of kernel arguments. Launches alternate between them, so the next launch's arguments go in
#   while this one runs.
# - KERNELS: the kernels it holds
# - ZEROS: a tile of zeros (4 KiB, an fp32 tile) that kernels read but don't write
# - DATA: free space
# The dispatcher has its config at DISPATCH: [doorbell, ring lo, hi, ring size, status lo, hi, workers' rectangle, how
# many]. OUTBOX holds [go, kernel, arguments, workers] [rptr, launches done], and at +32 a launch's [go, last report,
# flits read]. DONE holds the count of the workers' reports and the sum of what they read, and STAGING holds up to
# STAGE bytes of commands. Those two sit above the kernels, since loading a kernel writes the dispatcher's tile too.
MAILBOX, TRISCS, SYNC, FLAGS, ARGS, KERNELS, ZEROS, DATA = 0x1000, 0x1020, 0x1040, 0x1080, 0x1100, 0x2000, 0x7F000, 0x80000
MCAST_STATE = 0x1030  # two words shared by workers and dispatcher, between their config/mailbox and SYNC
ARGS_SIZE = (KERNELS - ARGS) // 2  # the size of each argument buffer
SYNC_COUNTS = 8  # the first 8 words of SYNC are zeroed before each launch. kernels write the rest before reading them
SYNC_NAMES = ("fetched", "unpacked", "ready", "packed", "sent", "issued")  # the first SYNC counters in compiled kernels
DISPATCH, OUTBOX, DONE, STAGING = 0x1000, 0x1400, DATA, DATA + 64
STAGE = 1 << 16  # the biggest read that the dispatcher makes from the ring into STAGING, in bytes

# compile_rv32 puts DEVICE before the C it compiles. Its NoC transfers all go through NIU 0 (bh.NIU_REGS,
# bh.NOC_CMD). noc_read and noc_write use request initiator 0. noc_get and noc_put use initiators 1 and 2, which the
# firmware sets up at boot (noc_setup).
DEVICE = f"""
#define L1_DATA {DATA:#x}
#define NIU(reg) (((volatile unsigned *){bh.NIU:#x})[(reg) / 4])
enum {{ {", ".join(f"{k} = {v:#x}" for k, v in (bh.NIU_REGS | bh.NOC_CMD | {"MAX_PACKET": bh.MAX_PACKET}).items())} }};
static inline unsigned noc_me(void) {{ return NIU(ID_LOGICAL) & 0xFFF; }}  // this tile's translated x | y << 6
static inline void noc_issue(unsigned long long targ, unsigned targ_xy, unsigned long long ret, unsigned ret_xy,
                             unsigned ctrl, unsigned n) {{
  while (NIU(CMD) & 1) {{}}  // the initiator still holds the previous request
  NIU(TARG_LO) = targ; NIU(TARG_MID) = targ >> 32; NIU(TARG_HI) = targ_xy;
  NIU(RET_LO) = ret; NIU(RET_MID) = ret >> 32; NIU(RET_HI) = ret_xy;
  NIU(PACKET_TAG) = 0; NIU(CTRL) = ctrl; NIU(LEN) = n;
  NIU(CMD) = 1;
}}
// Wait until this core's earlier stores to L1 are processed: a load from addr behind them only returns once they're
// out. To settle stores to the config bank, pass an address there.
static inline void l1_settle(unsigned addr) {{
  asm volatile("fence" ::: "memory");
  (void)*(volatile unsigned *)addr;
  asm volatile("fence" ::: "memory");
}}
// Copy n bytes between (x, y, addr) and this tile's L1. The low 4 bits of the two addresses must match (6 for DRAM).
static inline void noc_read(unsigned x, unsigned y, unsigned long long addr, unsigned l1, unsigned n) {{
  for (unsigned k; n; addr += k, l1 += k, n -= k) noc_issue(addr, x | y << 6, l1, noc_me(), READ, k = n < MAX_PACKET ? n : MAX_PACKET);
}}
// noc_get reads into this L1 and noc_put writes from it, at most MAX_PACKET bytes, through request initiators 1 and 2.
// The firmware sets those up once (noc_setup), so each call writes only four registers and the go, where noc_read
// and noc_write write all nine. noc_put doesn't settle this core's stores. It's meant for data that the Tensix
// wrote, where there are none.
#define NIU1(reg) NIU(INITIATOR + (reg))
#define NIU2(reg) NIU(2 * INITIATOR + (reg))
static inline void noc_setup(void) {{
  NIU1(TARG_MID) = 0; NIU1(RET_MID) = 0; NIU1(RET_HI) = noc_me(); NIU1(PACKET_TAG) = 0; NIU1(CTRL) = READ;
  NIU2(TARG_MID) = 0; NIU2(TARG_HI) = noc_me(); NIU2(RET_MID) = 0; NIU2(PACKET_TAG) = 0; NIU2(CTRL) = WRITE;
}}
static inline void noc_get(unsigned x, unsigned y, unsigned addr, unsigned l1, unsigned n) {{
  while (NIU1(CMD) & 1) {{}}
  NIU1(TARG_LO) = addr; NIU1(TARG_HI) = x | y << 6; NIU1(RET_LO) = l1; NIU1(LEN) = n; NIU1(CMD) = 1;
}}
static inline void noc_put(unsigned l1, unsigned x, unsigned y, unsigned addr, unsigned n) {{
  while (NIU2(CMD) & 1) {{}}
  NIU2(TARG_LO) = l1; NIU2(RET_LO) = addr; NIU2(RET_HI) = x | y << 6; NIU2(LEN) = n; NIU2(CMD) = 1;
}}
static inline void noc_write(unsigned l1, unsigned x, unsigned y, unsigned long long addr, unsigned n) {{
  l1_settle(l1);  // the NIU doesn't see this core's stores in flight, so they must land first
  for (unsigned k; n; addr += k, l1 += k, n -= k) noc_issue(l1, noc_me(), addr, x | y << 6, WRITE, k = n < MAX_PACKET ? n : MAX_PACKET);
}}
// Publish the active region before checking cancellation. Once the host requests cancellation and sees no active
// region, no later multicast can issue. Ordinary requests can still issue until reset; shutdown drains their counters.
#define MCAST_STATE ((volatile unsigned *){MCAST_STATE:#x})
static inline void mcast_begin(void) {{
  MCAST_STATE[1] = 1;
  l1_settle({MCAST_STATE + 4:#x});
  if (MCAST_STATE[0]) {{
    MCAST_STATE[1] = 0;
    l1_settle({MCAST_STATE + 4:#x});
    for (;;) {{}}  // cancelled: the host will reset this core
  }}
}}
// Wait until a multicast has all its acknowledgements, which is when the ack counter reaches acks. A multicast leaves
// the outstanding count wrong, up once and down once per tile, so no other request may be in flight. We clear it after.
static inline void mcast_done(unsigned acks) {{
  while (NIU(WR_ACKS) != acks) {{}}
  NIU(CLEAR_OUTSTANDING) = 0xFFFF;
  l1_settle({bh.NIU + bh.NIU_REGS["OUTSTANDING"]:#x});
  MCAST_STATE[1] = 0;
  l1_settle({MCAST_STATE + 4:#x});
}}
static inline void noc_wait(void) {{  // wait until reads have landed and writes are acknowledged
  (void)NIU(CMD); (void)NIU1(CMD); (void)NIU2(CMD);  // reading the initiators back keeps the counter read behind their last requests
  while (NIU(OUTSTANDING)) {{}}
  asm volatile("fence" ::: "memory");  // drop L0 data cache lines that the NoC may have overwritten
}}
// TRISC only. Wait until its Tensix thread has run everything pushed so far. The store keeps the load behind those
// pushes, and nothing may load from the same region before the answer is back (ISA BabyRISCV/ManualTTSync.md).
static inline void tensix_idle(void) {{
  asm volatile("sw zero, 0(%0)\\n lw t0, 0(%0)\\n andi t0, t0, 0" :: "r"({bh.TENSIX_DONE:#x}) : "t0", "memory");
}}
#define CLOCK (*(volatile unsigned *){bh.WALL_CLOCK:#x})  // the tile's wall clock, which counts cycles
"""

CLANG = ["clang", "--target=riscv32-unknown-elf", "-march=rv32im", "-mabi=ilp32", "-mcmodel=medany", "-mno-relax", "-fno-jump-tables",
         "-O2", "-ffreestanding", "-nostdlib", "-ffunction-sections", "-fuse-ld=lld", "-Wl,--emit-relocs,-e,0"]
CACHE = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache"), "baretile")  # compiled images, by what made them

@functools.cache
def _clang_version(): return subprocess.run(["clang", "--version"], capture_output=True, check=True).stdout

@functools.cache
def _prune():
  # Once per process, before the first compile, delete images unused for a week. Each use touches its file. This keeps
  # images from older code from piling up.
  for f in CACHE.glob("*"):
    try:
      if f.is_file() and f.stat().st_mtime < time.time() - 7 * 86400: f.unlink()
    except FileNotFoundError: pass  # another process deleted it first

def _position_independent_image(elf):
  # Keep only the linked image, but inspect its ELF relocations first. PC-relative code does not make absolute
  # pointer initializers relocatable. ELF32 little-endian is fixed by our rv32 target and linker script.
  shoff = struct.unpack_from("<I", elf, 32)[0]
  shsize, count = struct.unpack_from("<2H", elf, 46)
  sections = [struct.unpack_from("<10I", elf, shoff + i * shsize) for i in range(count)]
  image = bytearray(max((s[3] + s[5] for s in sections if s[2] & 2), default=0))  # SHF_ALLOC
  for _, kind, flags, addr, off, size, _, target, _, entry in sections:
    if flags & 2 and kind != 8: image[addr:addr + size] = elf[off:off + size]  # SHT_NOBITS stays zero
    if kind != 4 or not sections[target][2] & 2: continue  # SHT_RELA against an allocated section
    for at in range(off, off + size, entry):
      relocation = struct.unpack_from("<I", elf, at + 4)[0] & 255
      if relocation not in (0, 16, 17, 18, 19, 23, 24, 25):  # NONE, branches/calls and PC-relative HI20/LO12
        raise ValueError(f"kernel relocation {relocation} is not position-independent; use offsets instead of static pointers")
  return bytes(image)

def compile_rv32(src):
  """Prepend DEVICE to C source, compile for rv32, and link a flat image at address 0. Put .start first if
  present, otherwise kmain. PC-relative addressing lets the image run wherever it's loaded; this requires
  medany, no linker relaxation and no jump tables. Reject absolute relocations, including static pointer
  initializers; use offsets into a const array instead. Cache images by compiler version, flags, linker script
  and source. Entries expire after a week without use."""
  link = ("SECTIONS { .text 0 : { *(.start) *(.text.kmain) *(.text*) *(.rodata*) *(.srodata*) *(.data*) *(.sdata*) "
          ". = ALIGN(4); __bss_start = .; *(.sbss*) *(.bss*) . = ALIGN(4); __bss_end = .; } }")  # one section: .bss is zeros in the image
  cached = CACHE / hashlib.sha256(_clang_version() + repr(CLANG).encode() + link.encode() + (DEVICE + src).encode()).hexdigest()
  _prune()
  try:
    os.utime(cached)
    return cached.read_bytes()
  except FileNotFoundError: pass  # not compiled yet, or another process pruned it
  if not shutil.which("ld.lld"): raise RuntimeError("clang links with lld, and ld.lld is not on PATH: install lld")
  with tempfile.TemporaryDirectory() as d:
    (ld := pathlib.Path(d, "link.ld")).write_text(link)
    out = pathlib.Path(d, "prog.elf")
    subprocess.run(CLANG + [f"-Wl,-T,{ld}", "-x", "c", "-", "-o", str(out)], input=(DEVICE + src).encode(), check=True)
    image = _position_independent_image(out.read_bytes())
  CACHE.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(dir=CACHE, prefix=cached.name + ".", delete=False) as f:
    part = pathlib.Path(f.name)
    try:
      f.write(image)
      f.close()
      os.replace(part, cached)  # atomic; threads and processes may compile the same image concurrently
    finally: part.unlink(missing_ok=True)
  return image

# Kernel programs start here. A kernel stays loaded between launches, so kentry zeroes its .bss at the start of a
# launch. Statics with any other starting value must be const.
KENTRY = """
extern unsigned __bss_start[], __bss_end[];
void kmain(volatile unsigned *args);
__attribute__((section(".start"))) void kentry(volatile unsigned *args) {
  for (volatile unsigned *p = __bss_start; p < __bss_end; p++) *p = 0;
  kmain(args);
}
"""

def compile_kernel(*programs):
  """Build an image from C programs for BRISC and TRISC0-2, in that order. Each program defines void
  kmain(volatile unsigned *args). Omitted or None programs leave that core idle. The header gives four entry
  offsets, with 0 for an absent program, followed by the programs. The image can run at any load address;
  absolute address initializers are rejected during compilation."""
  if len(programs) > 4: raise ValueError("a kernel has at most four programs: BRISC and TRISC0-2")
  image, entry = b"", []
  for src in programs + (None,) * (4 - len(programs)):
    entry.append(16 + len(image) if src else 0)
    if src:
      program = compile_rv32(KENTRY + src)
      image += program.ljust(-(-len(program) // 16) * 16, b"\0")
  return struct.pack("<4I", *entry) + image

# the first words of firmware. BRISC starts at 0, and TRISC i at its reset PC, 4 + 4i
def jumps(brisc, trisc0, trisc1, trisc2):
  return f'__attribute__((naked, section(".start"))) void _start(void) {{ asm("j {brisc}\\n j {trisc0}\\n j {trisc1}\\n j {trisc2}"); }}'

# BRISC receives go, the kernel and argument addresses, and the worker count n in its mailbox. Workers with index
# n or higher skip the launch. BRISC divides the work into [first unit, unit count], clears SYNC, and invalidates
# all four instruction caches because a different kernel may have occupied the same addresses. It then starts
# the programs named in the kernel header. Each TRISC waits for its Tensix thread to finish before returning.
# Once all four cores finish, BRISC records completion in its mailbox and increments the dispatcher's report
# count with an acknowledged NoC atomic. Its transaction counter lets shutdown wait for completion after the
# core stops. With COUNTED enabled, BRISC first adds the launch's read-data count, in 64-byte flits, to the
# dispatcher's sum. Both atomics use the same static VC, so the byte count arrives before the completion report.
FIRMWARE = f"""
#define MAILBOX ((volatile unsigned *){MAILBOX:#x})
#define TRISCS ((volatile unsigned *){TRISCS:#x})
#define CFG ((volatile unsigned *){bh.TENSIX_CFG:#x})
{jumps("brisc", "trisc0", "trisc1", "trisc2")}
__attribute__((naked)) void brisc(void) {{ asm("li sp, {bh.BRISC_STACK_TOP:#x}\\n j brisc_loop"); }}
__attribute__((naked)) void trisc0(void) {{ asm("li sp, {bh.TRISC_STACK_TOP:#x}\\n li a0, 0\\n j trisc_loop"); }}
__attribute__((naked)) void trisc1(void) {{ asm("li sp, {bh.TRISC_STACK_TOP:#x}\\n li a0, 1\\n j trisc_loop"); }}
__attribute__((naked)) void trisc2(void) {{ asm("li sp, {bh.TRISC_STACK_TOP:#x}\\n li a0, 2\\n j trisc_loop"); }}
static void run(unsigned core) {{  // runs the kernel's program for this core, if it has one
  unsigned kernel = MAILBOX[1], entry = ((volatile unsigned *)kernel)[core];
  if (entry) ((void (*)(volatile unsigned *))(kernel + entry))((volatile unsigned *)MAILBOX[2]);
}}
void brisc_loop(void) {{
  noc_setup();
  for (unsigned seen = 0, phase = 0;;) {{
    while (MAILBOX[0] == seen) asm volatile("fence" ::: "memory");  // the L0 data cache misses NoC writes, and fence drops it
    seen = MAILBOX[0];
    asm volatile("fence" ::: "memory");  // the rest of the mailbox and the arguments came over the NoC before go
    unsigned me = MAILBOX[7], n = MAILBOX[3];
    if (me >= n) continue;
#ifdef COUNTED
    unsigned words = NIU(WORDS);
#endif
    volatile unsigned *a = (volatile unsigned *)MAILBOX[2];
    unsigned units = a[0];
    unsigned q = units / n, r = units % n;
    a[0] = me * q + me * r / n;
    a[1] = q + (me + 1) * r / n - me * r / n;  // products stay in range even when units is UINT32_MAX
    for (unsigned i = 0; i < {SYNC_COUNTS}; i++) ((volatile unsigned *){SYNC:#x})[i] = 0;
    CFG[{bh.CFG_IC_INVALIDATE}] = 0xF;  // drop the cached instructions of BRISC and the TRISCs
    l1_settle({bh.TENSIX_CFG:#x});
    l1_settle({SYNC:#x});  // the SYNC counters are zero before any core of the kernel looks at them
    TRISCS[0] = phase ^= 1;  // a worker can first participate at launch zero, or skip a whole launch-number cycle
    run(0);
    noc_wait();  // what the kernel sent has landed before we report done
    for (unsigned i = 1; i <= 3; i++) while (TRISCS[i] != phase) asm volatile("fence" ::: "memory");
    MAILBOX[4] = seen;
#ifdef COUNTED
    while (NIU(CMD) & 1) {{}}
    NIU(AT_DATA) = NIU(WORDS) - words;
    noc_issue({DONE + 4:#x}, MAILBOX[5], {MAILBOX + 24:#x}, noc_me(), ATOMIC, INCREMENT | 1);
#endif
    while (NIU(CMD) & 1) {{}}
    NIU(AT_DATA) = 1;
    noc_issue({DONE:#x}, MAILBOX[5], {MAILBOX + 24:#x}, noc_me(), ATOMIC, INCREMENT);
  }}
}}
void trisc_loop(unsigned i) {{
  for (unsigned seen = 0;;) {{
    while (TRISCS[0] == seen) asm volatile("fence" ::: "memory");
    seen = TRISCS[0];
    asm volatile("fence" ::: "memory");  // the mailbox and the kernel came over the NoC, after anything we may have cached
    run(1 + i);
    tensix_idle();
    TRISCS[1 + i] = seen;
  }}
}}
"""

# Commands contain a 16-byte header [length, kernel address, argument bytes, workers], followed by arguments and
# padding to a multiple of 64 bytes. Length 0 marks the unused tail of the ring. Runtime.load has already put
# the kernel in worker L1.
# To reduce PCIe reads, the dispatcher fetches new commands in one noc_read, limited by STAGE and the ring's end.
# It multicasts arguments and go to every worker, including those outside this launch's worker count. Multicast
# completion uses acknowledgements: the request counter increments once but decrements for each recipient, so
# the dispatcher clears that counter afterwards. Workers report completion by incrementing DONE.
# Only one launch runs at a time. The dispatcher prepares the next launch's arguments in the alternate buffer
# while the current launch runs. After completion, it captures the timing and read-data counts, then starts
# the next launch before reporting to the host. Timing records use a TIMES-entry ring in host memory.
# Each record holds [go, last report, flits read]; flits are the increase in the shared read-data count.
# The dispatcher writes the record before [read pointer, launches done] on the same static VC, preserving their
# arrival order. Both writes are acknowledged, so shutdown can drain them before releasing the host DMA page.
TIMES = 1024
DISPATCHER = f"""
#define CONFIG ((volatile unsigned *){DISPATCH:#x})
#define OUT ((volatile unsigned *){OUTBOX:#x})
{jumps("brisc", "park", "park", "park")}  // its TRISCs have nothing to do
__attribute__((naked)) void park(void) {{ asm("1: j 1b"); }}
__attribute__((naked)) void brisc(void) {{ asm("li sp, {bh.BRISC_STACK_TOP:#x}\\n j loop"); }}
// n bytes from l1 to addr in the workers, and wait until they all have them
static void multicast(unsigned l1, unsigned addr, unsigned n) {{
  mcast_begin();
  unsigned acks = NIU(WR_ACKS) + CONFIG[7];
  l1_settle(l1);
  noc_issue(l1, noc_me(), addr, CONFIG[6], MCAST, n);  // the tiles of the rectangle but this one, which is in it
  mcast_done(acks);
}}
// The go copies the first 16 bytes of the outbox to the workers' mailboxes. It's always the same multicast, so request
// initiator 1 holds it and a launch starts with one store.
static void go(void) {{ mcast_begin(); unsigned acks = NIU(WR_ACKS) + CONFIG[7]; NIU1(CMD) = 1; mcast_done(acks); }}
void loop(void) {{
  unsigned long long ring = CONFIG[1] | (unsigned long long)CONFIG[2] << 32, status = CONFIG[4] | (unsigned long long)CONFIG[5] << 32;
  unsigned size = CONFIG[3], rptr = 0, launches = 0, me = noc_me();
  unsigned ready = 0, ready_n = 0, ready_end = 0, prepared = 0;  // the next launch, and whether its arguments are ready
  // the launch the workers are on, the ring pointer past it, the report count once it's done, and when its go went
  unsigned running = 0, running_end = 0, reports = 0, began = 0, active = 0;
  unsigned long long times = ring + size;  // launch n's [go, last report, flits read] at times + n % TIMES * 16
  unsigned words = 0;  // the flits the workers read, up to the last launch done
  unsigned staged = 0, first = 0;  // the ring's bytes [first, staged) are in STAGING
  NIU1(TARG_LO) = {OUTBOX:#x}; NIU1(TARG_MID) = 0; NIU1(TARG_HI) = noc_me(); NIU1(RET_LO) = {MAILBOX:#x}; NIU1(RET_MID) = 0;
  NIU1(RET_HI) = CONFIG[6]; NIU1(PACKET_TAG) = 0; NIU1(CTRL) = MCAST; NIU1(LEN) = 16;
  for (;;) {{
    asm volatile("fence" ::: "memory");  // the host rings the doorbell with its write pointer
    if (!prepared && CONFIG[0] != rptr) {{  // the next command, while the last launch runs
      if (rptr == staged) {{  // read everything new in the ring, up to its end and at most STAGE bytes
        unsigned n = CONFIG[0] - rptr, left = size - rptr % size;
        n = n < left ? n : left;
        n = n < {STAGE} ? n : {STAGE};
        noc_read({bh.PCIE[0]}, {bh.PCIE[1]}, ring + rptr % size, {STAGING:#x}, n);
        noc_wait();
        first = rptr; staged = rptr + n;
      }}
      volatile unsigned *cmd = (volatile unsigned *)({STAGING:#x} + rptr - first);
      if (cmd[0] == 0) {{ staged = rptr += size - rptr % size; continue; }}
      if (cmd[0] > staged - rptr) {{ staged = rptr; continue; }}  // cut short by STAGE, so read again from its start
      ready = ++launches;  // its arguments go into the buffer that the running launch isn't using
      unsigned args = {ARGS:#x} + ready % 2 * {ARGS_SIZE};
      multicast((unsigned)cmd + 16, args, cmd[2]);
      ready_n = cmd[3]; ready_end = rptr += cmd[0];
      // write its go now, since the go of the running launch has already gone out
      OUT[0] = ready; OUT[1] = cmd[1]; OUT[2] = args; OUT[3] = ready_n;
      l1_settle({OUTBOX:#x});
      prepared = 1;
    }}
    if (prepared && active)  // the next launch is ready, so spin on DONE until the last report of this one
      while (*(volatile unsigned *){DONE:#x} != reports) asm volatile("fence" ::: "memory");
    asm volatile("fence" ::: "memory");
    unsigned complete = active && *(volatile unsigned *){DONE:#x} == reports, done = running, end = running_end;
    unsigned took = began, finished = complete ? CLOCK : 0, read = complete ? *(volatile unsigned *){DONE + 4:#x} - words : 0;
    words += read;
    if (complete) active = 0;
    if (prepared && !active) {{  // the workers have finished the last launch
      began = CLOCK;
      go();
      running = ready; running_end = ready_end; reports += ready_n; active = 1; prepared = 0;
    }}
    if (complete) {{  // after the next go: better the host hears late than the next launch starts late
      while (NIU(OUTGOING)) {{}}  // wait until the last launch's times and status have left the outbox
      OUT[4] = end; OUT[5] = done; OUT[8] = took; OUT[9] = finished; OUT[10] = read;
      l1_settle({OUTBOX + 16:#x});
      // its times, then the status that says it's done, in that order
      noc_issue({OUTBOX + 32:#x}, me, times + done % {TIMES} * 16, {bh.noc_xy(*bh.PCIE)}, WRITE | STATIC, 16);
      noc_issue({OUTBOX + 16:#x}, me, status, {bh.noc_xy(*bh.PCIE)}, WRITE | STATIC, 8);
      noc_wait();  // a later multicast must not count these acknowledgements as worker acknowledgements
    }}
  }}
}}
"""

def _merge_holes(holes):
  holes.sort()
  for i in range(len(holes) - 1, 0, -1):
    (a, n), (b, m) = holes[i - 1], holes[i]
    if a + n == b: holes[i - 1:i + 1] = [(a, n + m)]

# _open holds the resources of every runtime that hasn't stopped, and _stranded those of runtimes whose cores stopped
# before their transfers could be shown finished. Holding them keeps host DMA pages mapped and DRAM reserved while the
# card may still use them. Collecting a Runtime doesn't stop it: closing goes through the shared TLB window, and a
# collection can run in the middle of another access to that window. The next Runtime on the chip, an allocation
# that finds no room, or process exit stops it instead.
_open, _stranded = set(), set()

def _reap(chip):  # stop the runtimes on chip that were collected without close(). True if any memory came back
  freed = False
  for resources in [r for r in _open if r.chip is chip and r.owner() is None]:
    try: freed |= resources.close()
    except Exception: pass  # still open, and a later reap, takeover or exit tries again
  return freed

@atexit.register
def _exit():  # stop every runtime still running when Python exits, since the card would outlive the host pages
  for resources in list(_open):
    try: resources.close()
    except Exception as e: print(f"baretile: could not stop the runtime from {resources.start} to {resources.end}: {e}", file=sys.stderr)

class _Resources:
  """Own a rectangle, its host DMA views and its DRAM without retaining the Runtime that uses them."""
  def __init__(self, rt, views):
    self.chip, self.start, self.end, self.cores = rt.chip, rt.start, rt.end, {*rt.cores, rt.dispatcher}
    self.owner, self.views, self.used = weakref.ref(rt), views, rt.used
    self.active, self.closed, self.started = True, False, False
    _open.add(self)
  def close(self):
    """Stop the rectangle and release its memory, returning whether the memory came back. If a multicast never
    finishes or the NoC never drains, stop the cores anyway and keep the memory for the rest of the process, so
    that another runtime can still take the tiles. If halting fails, the memory stays reserved and close can be retried."""
    if self.closed: return False
    self.active, settled = False, True
    if self.started:
      try:
        self.chip.write32(*self.end, MCAST_STATE, 1, self.start)
        self.chip.wait(lambda: all(self.chip.read32(x, y, MCAST_STATE) == 1 and self.chip.read32(x, y, MCAST_STATE + 4) == 0
                                  for x, y in self.cores),
                       "finish the runtime's active multicasts")
      except TimeoutError: settled = False  # the multicast's counters are wrong until it finishes, so draining can't tell
    self.chip.halt(*self.end, self.start)
    try:
      # Read reset back before polling the NoC: requests issued before the cores stopped can still be in flight.
      for x, y in self.cores: self.chip.read32(x, y, bh.SOFT_RESET)
      # Every initiator must finish accepting its command before the transaction counters can be trusted.
      if settled:
        self.chip.wait(lambda: all(not self.chip.read32(x, y, bh.NIU + i * bh.NIU_REGS["INITIATOR"] + bh.NIU_REGS["CMD"]) & 1
                                  for x, y in self.cores for i in range(4)), "stop issuing the runtime's NoC requests")
        self.chip.wait(lambda: all(self.chip.read32(x, y, bh.NIU + bh.NIU_REGS[kind] + 4 * i) == 0
                                  for x, y in self.cores for kind in ("OUTSTANDING", "OUTGOING") for i in range(16)),
                       "drain the stopped runtime's NoC requests")
    except TimeoutError: settled = False
    del self.chip.taken[self.start, self.end]
    _open.discard(self)
    self.closed = True
    if not settled:
      _stranded.add(self)
      return False
    self.chip.dram_holes.extend(self.used.items())
    _merge_holes(self.chip.dram_holes)
    self.used.clear()
    for view in self.views: view.release()
    self.views = ()
    return True

class Runtime:
  """Run kernels on the Tensix rectangle from start to end, using the available tiles by default. The last tile
  dispatches work to the others. Kernels stay in L1 until the kernel area fills. The dispatcher reads
  commands from host memory and executes them in order. A new Runtime halts any older runtime that overlaps
  its rectangle. Close a runtime to stop its workers and release its memory; a closed runtime cannot be reused."""
  def __init__(self, chip, start=None, end=None, ring=1 << 20):
    start, end = start or chip.tensix[0], end or chip.tensix[-1]
    if ring < 64 or ring & (ring - 1): raise ValueError("ring size must be a power of two, at least 64 bytes")
    back = 64 + ring + 16 * TIMES
    if back + 4096 > 1 << 21: raise ValueError("the ring leaves no room in the hugepage to read a tile back")
    if start not in chip.tensix or end not in chip.tensix: raise ValueError("runtime endpoints must be available Tensix tiles")
    if chip.banks != list(range(len(bh.DRAM))): raise RuntimeError(f"the runtime needs all eight DRAM banks, and the card has {chip.banks}")
    cores = [(x, y) for x, y in chip.tensix if start[0] <= x <= end[0] and start[1] <= y <= end[1]]
    if len(cores) < 2: raise ValueError("a rectangle needs a dispatcher and at least one worker")
    self.chip, self.launches, self.wptr = chip, 0, 0
    self.queued = collections.deque()  # (launch, workers) for each launch that may not be done yet
    # DRAM: holes lists free (address, size) ranges; used maps address to size.
    # freed pairs an address with the last queued launch that could use it.
    self.holes, self.used, self.freed = chip.dram_holes, {}, []
    # loaded maps each kernel image in L1 to its address, and room is where the next one goes
    self.start, self.end, self.loaded, self.room = start, end, {}, KERNELS
    # how many times the loaded kernels were evicted. A recorded command names its kernels by address, so a replay
    # checks this first.
    self.evictions = 0
    *self.cores, self.dispatcher = cores  # the dispatcher is end, the last tile of the rectangle
    # The workers as a grid, for kernels that multicast along rows and columns. The grid keeps the rows that have no
    # tile missing, and only the dispatcher's row can be short. grid is (x of each column, y of each row), and worker
    # gy * columns + gx is at (x[gx], y[gy]).
    xs = sorted({x for x, _ in self.cores})
    ys = [y for y in sorted({y for _, y in self.cores}) if all((x, y) in self.cores for x in xs)]
    assert self.cores[:len(xs) * len(ys)] == [(x, y) for y in ys for x in xs], "the grid's rows must come first"
    self.grid = tuple(xs), tuple(ys)
    # Give each grid worker a DRAM bank and a port to read it through, for kernels where workers read from their own
    # bank. We go through the (worker, port) pairs by the NoC0 links that the data takes back (bh.hops), fewest first.
    # A pair is taken while the worker has no port yet, the bank has fewer than its share of workers, and the port
    # fewer than a third of that. This keeps any row's links from carrying the reads of many workers.
    n, ports = len(xs) * len(ys), [(b, (x, y + p)) for b, (x, y) in enumerate(bh.DRAM) for p in range(3)]
    pairs = sorted((bh.hops(chip.raw(*port), chip.raw(x, y)), i, b, port) for i, (x, y) in enumerate(self.cores[:n]) for b, port in ports)
    readers, count = [None] * n, collections.Counter()  # count: how many readers each bank and each port has
    for _, i, b, port in pairs:
      if readers[i] is None and count[b] < n // len(bh.DRAM) + (b < n % len(bh.DRAM)) and count[port] < -(-n // len(ports)):
        readers[i], count[b], count[port] = (b, port), count[b] + 1, count[port] + 1
    self.readers = tuple(readers)  # (bank, translated (x, y) of the port) for each grid worker, in grid order
    self.counts = chip.pci.counts  # whether profile knows the bytes each launch read
    worker, dispatcher = compile_rv32("#define COUNTED\n" * self.counts + FIRMWARE), compile_rv32(DISPATCHER)
    assert max(len(worker), len(dispatcher)) <= MAILBOX, "firmware overlaps the mailbox"
    # Close the whole old rectangle, including workers we don't take. The old object stays invalid even if a
    # later runtime uses exactly the same rectangle. Disjoint rectangles keep running and share the DRAM holes.
    _reap(chip)
    for theirs in list(chip.taken.values()):
      if theirs.cores & set(cores): theirs.close()
    chip.halt(*end, start)
    buf, bus = chip.pci.alloc(1 << 21)  # status, command ring, times, then transfer staging
    self.status, self.ring, self.times, self.back, self.back_bus = buf[:64], buf[64:64 + ring], buf[64 + ring:back], buf[back:], bus + back
    self.resources = _Resources(self, (self.status, self.ring, self.times, self.back, buf))
    chip.taken[start, end] = self.resources
    self.sent = [("", 0, b"")] * TIMES  # (name, workers, kernel) of the last TIMES launches, for profile and baretile.viz
    self.numbers = {}  # (repr(value), dtype) -> a tensor holding that number, uploaded once by Tensor.number
    # while tensor.jit records a call: its launches, and the DRAM it took, which dram_free won't give back
    self.log, self.pinned, self.pinned_freed = None, collections.Counter(), set()
    self.recorded, self.recorded_bases = None, []  # active recording pins and existing allocation bases
    self.written = None  # while tensor.jit records a call: the allocations it writes in place
    try:
      (dx, dy), n = self.dispatcher, len(self.cores)
      chip.write(*end, 0, worker, start)
      chip.write(*end, ZEROS, bytes(DATA - ZEROS), start)
      # zero the mailbox, the TRISCs' go and done, SYNC and FLAGS, except whom to report to and the worker's index
      for i, (x, y) in enumerate(self.cores):
        chip.write(x, y, MAILBOX, struct.pack("<8I", 0, 0, 0, 0, 0, bh.noc_xy(dx, dy), 0, i).ljust(ARGS - MAILBOX, b"\0"))
      chip.write(dx, dy, 0, dispatcher)
      rect = bh.noc_rect(start, end)
      chip.write(dx, dy, DISPATCH, struct.pack("<8I", 0, (bus + 64) & 0xFFFFFFFF, (bus + 64) >> 32, ring, bus & 0xFFFFFFFF, bus >> 32, rect, n))
      chip.write(dx, dy, DONE, bytes(8))
      chip.write(*end, MCAST_STATE, bytes(8), start)
      chip.write(dx, dy, OUTBOX, bytes(64))  # clear completion status left by the previous dispatcher
      self.doorbell = chip.map(dx, dy, DISPATCH)  # its own TLB, so ringing the doorbell doesn't have to move a window first
      for i in range(3): chip.write32(*end, bh.TRISC_RESET_PC + 4 * i, 4 + 4 * i, start)
      chip.write32(*end, bh.TRISC_RESET_PC_OVERRIDE, 0b111, start)
      self.resources.started = True
      chip.write32(*end, bh.SOFT_RESET, bh.NCRISC, start)  # BRISC and the TRISCs go, and NCRISC stays in reset
    except BaseException:
      self.close()
      raise

  def check(self):
    """Reject work after explicit shutdown or takeover by another runtime."""
    if not self.resources.active: raise RuntimeError("runtime is closed or was replaced by an overlapping runtime")

  def close(self):
    """Stop all workers, cancel queued work and release memory. Call wait() first to finish queued work.
    Closing twice is harmless; tensors and recordings from this runtime can no longer run."""
    self.resources.close()

  def __enter__(self): self.check(); return self
  def __exit__(self, *_): self.close()

  def load(self, kernel):
    """Load a kernel into the workers' L1 and return its address, reusing an existing copy if present. Loading
    multicasts over the rectangle, including the dispatcher. If space runs out, wait for queued work before
    evicting kernels. Strict TLB ordering makes the kernel arrive before a later doorbell write."""
    self.check()
    if kernel not in self.loaded:
      if len(kernel) > ZEROS - KERNELS: raise ValueError(f"kernel is {len(kernel)} bytes, and kernel space holds {ZEROS - KERNELS}")
      if self.evicts(kernel): self.evict()
      self.chip.write(*self.end, self.room, kernel, self.start)
      self.loaded[kernel], self.room = self.room, self.room + len(kernel)  # compile_kernel makes it a multiple of 16
    return self.loaded[kernel]
  # whether loading kernel would evict the others
  def evicts(self, kernel): self.check(); return kernel not in self.loaded and self.room + len(kernel) > ZEROS

  def evict(self):  # forget the loaded kernels, once the launches that use them are done
    self.wait()
    self.loaded, self.room, self.evictions = {}, KERNELS, self.evictions + 1

  def launch(self, kernel, args=(), units=None, name=""):
    """Queue a compiled kernel, loading it if needed. Divide units between workers in order; the default is one
    unit per worker. Workers receive [first unit, unit count, *args]. name identifies the launch in
    profiles."""
    self.check()
    if self.log is not None:
      self.log.append((kernel, args, units, name))
      # Raw kernel arguments can contain pointers inside an allocation. Conservatively retain any allocation
      # they address. New allocations are already retained by dram_alloc; only older ones need this lookup.
      for value in args if self.recorded is not None else ():
        i = bisect.bisect_right(self.recorded_bases, value) - 1
        if i >= 0:
          addr = self.recorded_bases[i]
          if addr <= value < addr + self.used.get(addr, 0): self.pin(addr)
    self.send(*self.block([(kernel, args, units, name)]))

  def block(self, launches):
    """Convert [(kernel, args, units, name)] to commands for send, plus (name, worker count, kernel) per launch. Load
    the kernels first, since commands contain their L1 addresses. Commands can be replayed until a kernel is
    evicted. If loading requires eviction, send the commands built so far before waiting for space."""
    self.check()
    cmds, sent = [], []
    for kernel, args, units, name in launches:
      if cmds and self.evicts(kernel): self.send(b"".join(cmds), sent); cmds, sent = [], []
      cmd, n = self.command(kernel, args, units)
      cmds.append(cmd); sent.append((name, n, kernel))
    return b"".join(cmds), sent

  def command(self, kernel, args, units):  # a launch as the ring holds it: its bytes, and how many workers it takes
    self.check()
    units = len(self.cores) if units is None else units
    try: units = operator.index(units)
    except TypeError: raise ValueError("launch units must be an integer in [1, 2**32 - 1]") from None
    if not 1 <= units < 2**32: raise ValueError("launch units must be in [1, 2**32 - 1]")
    n = min(units, len(self.cores))
    body = struct.pack(f"<{2 + len(args)}I", units, 0, *args)
    if len(body) > ARGS_SIZE: raise ValueError(f"{len(body)} bytes of arguments, and an argument buffer holds {ARGS_SIZE}")
    length = -(-(16 + len(body)) // 64) * 64
    if length > len(self.ring): raise ValueError(f"a {length}-byte command does not fit in a {len(self.ring)}-byte ring")
    return struct.pack("<4I", length, self.load(kernel), len(body), n) + body.ljust(length - 16, b"\0"), n

  def send(self, cmds, launches):
    """Write commands back to back into the ring, then ring the doorbell. A block uses one PCIe write when it fits
    with any wrap padding; larger blocks are sent in pieces. launches has (name, workers, kernel) for each."""
    self.check()
    size, at, length = len(self.ring), self.wptr % len(self.ring), len(cmds)
    skip = size - at if at + length > size else 0  # commands don't wrap, so mark the tail unused and start over
    if not length: return
    while length + skip > size:
      # Keep the whole block replayable, even when it is bigger than the ring. Split only between commands.
      end, count = 0, 0
      while end < length:
        n = struct.unpack_from("<I", cmds, end)[0]
        if not 0 < n <= size: raise ValueError(f"a {n}-byte command does not fit in a {size}-byte ring")
        if end + n > size - at: break
        end, count = end + n, count + 1
      if end:
        self.send(cmds[:end], launches[:count])
      else:
        self.until(lambda: (self.wptr - self.read_status(0)) % 2**32 + skip <= size, "free space in the ring")
        self.ring[at:at + 4] = bytes(4)
        self.wptr += skip
        self.chip.pci.write(self.doorbell, (self.wptr % 2**32).to_bytes(4, "little"))
      cmds, launches = cmds[end:], launches[count:]
      at, length = self.wptr % size, len(cmds)
      skip = size - at if at + length > size else 0
    self.until(lambda: (self.wptr - self.read_status(0)) % 2**32 + skip + length <= size, "free space in the ring")
    if skip:
      self.ring[at:at + 4] = bytes(4)
      self.wptr, at = self.wptr + skip, 0
    self.ring[at:at + length] = cmds
    self.wptr += length
    for launch in launches:
      self.launches += 1
      self.sent[self.launches % TIMES] = launch
      self.queued.append((self.launches, launch[1]))
    self.chip.pci.write(self.doorbell, (self.wptr % 2**32).to_bytes(4, "little"))

  def profile(self, since):
    """Wait for launches after since and return [(name, launches, cycles, bytes)], grouped by name with the
    longest first. The dispatcher times from go to the final worker report. Bytes count NoC reads in 64-byte
    flits, including DRAM, other tiles' L1 and local copies. Bytes are None without NIU counters. Only the
    last TIMES launches are available."""
    self.wait()
    if self.launches - since > TIMES: raise ValueError(f"{self.launches - since} launches to profile, but only the last {TIMES} are kept")
    totals = collections.defaultdict(lambda: [0, 0, 0])
    for n in range(since + 1, self.launches + 1):
      go, done, flits = struct.unpack_from("<3I", self.times, n % TIMES * 16)
      for i, v in enumerate((1, (done - go) % 2**32, 64 * flits)): totals[self.sent[n % TIMES][0]][i] += v
    return sorted(((name, n, cycles, read if self.counts else None) for name, (n, cycles, read) in totals.items()), key=lambda r: -r[2])

  def args(self):  # where the last launch's arguments are on each of its workers. some kernels write results there
    self.check()
    return ARGS + self.launches % 2 * ARGS_SIZE

  def dram_alloc(self, size):
    """Allocate size bytes in each of the 8 DRAM banks, at the same address, from the first free hole that's big
    enough."""
    self.check()
    size = max(64, -(-size // 64) * 64)  # NoC transfers to DRAM want 64-byte alignment
    for i, (addr, room) in enumerate(self.holes):
      if room >= size:
        self.holes[i:i + 1] = [(addr + size, room - size)] if room > size else []
        self.used[addr] = size
        self.pin(addr)
        return addr
    for resources in self.chip.taken.values():  # another runtime's completed launches can free shared space too
      if resources.active and (rt := resources.owner()) is not None and rt.freed:
        rt.wait()
        return self.dram_alloc(size)
    if _reap(self.chip): return self.dram_alloc(size)  # and so can stopping runtimes that were collected
    raise MemoryError(f"no {size} bytes free in DRAM")

  def dram_free(self, addr):
    """Release an allocation after the launches queued so far have finished, then merge adjacent free ranges.
    Keep allocations made during recording, since replay uses the same addresses."""
    if not self.resources.active: return  # shutdown already released all allocations after stopping the workers
    if addr in self.pinned: self.pinned_freed.add(addr); return
    self.freed.append((self.launches, addr))
    self.reclaim()

  def pin(self, addr):
    """Keep an allocation for the active recording, once per recording."""
    if self.recorded is not None and addr not in self.recorded:
      self.recorded.add(addr)
      self.pinned[addr] += 1

  def unpin(self, addresses):
    """Release one recording's ownership, preserving live tensors and queued launches."""
    for addr in addresses:
      if addr not in self.pinned: continue  # runtime shutdown may have already released all of its allocations
      self.pinned[addr] -= 1
      if not self.pinned[addr]:
        del self.pinned[addr]
        if addr in self.pinned_freed:
          self.pinned_freed.remove(addr)
          self.dram_free(addr)

  def finished(self): return self.launches - (self.launches - self.read_status(1)) % 2**32  # launches done, from the status

  def reclaim(self):  # freed DRAM that no unfinished launch can use goes back into the holes
    done = self.finished()
    while self.queued and self.queued[0][0] <= done: self.queued.popleft()
    for launches, addr in [f for f in self.freed if f[0] <= done]:
      self.freed.remove((launches, addr))
      self.holes.append((addr, self.used.pop(addr)))
      _merge_holes(self.holes)

  def read_status(self, i): self.check(); return int.from_bytes(self.status[4 * i:4 * i + 4], "little")

  def until(self, cond, what, timeout=60):
    """Poll until cond() is true. Raise TimeoutError if no launch completes within timeout seconds.
    Each completed launch restarts the timeout, allowing long queues to make progress on the simulator."""
    while not cond():
      done = self.read_status(1)
      self.chip.wait(lambda: cond() or self.read_status(1) != done, what, timeout)

  def wait(self, timeout=60, launch=None):
    """Poll completion status in host memory until launch number `launch` is done, by default the last one
    queued, then reclaim memory freed by completed launches. On timeout, report the current launch and its
    unfinished workers."""
    launch = self.launches if launch is None else launch
    try: self.until(lambda: self.finished() >= launch, f"finish launch {launch}", timeout)
    except TimeoutError as e:
      current = (self.chip.read32(*self.dispatcher, OUTBOX + 20) + 1) % 2**32  # after the last it saw finish (OUT[5])
      if current == (self.launches + 1) % 2**32: raise TimeoutError(f"{e}: all done, but the status never reached the host") from None
      n = next((n for launch, n in self.queued if launch % 2**32 == current), 0)
      stuck = [c for c in self.cores[:n] if self.chip.read32(*c, MAILBOX + 16) != current]
      reports = "".join(f"\n  {self.report(c, current)}" for c in stuck[:4])
      raise TimeoutError(f"{e}: the dispatcher is on launch {current}, still running on {stuck}{reports}") from None
    self.reclaim()

  def report(self, core, launch):
    """Describe an unfinished worker: active cores, SYNC and FLAGS, and outstanding NoC requests by transaction
    ID. On hardware, include each core's program counter."""
    self.check()
    x, y = core
    words = lambda addr, n: struct.unpack(f"<{n}I", self.chip.read(x, y, addr, 4 * n))
    niu = lambda off: self.chip.read32(x, y, bh.NIU + off)
    mail, triscs, sync, flags = words(MAILBOX, 8), words(TRISCS, 4), words(SYNC, len(SYNC_NAMES)), words(FLAGS, 32)
    if mail[0] != launch: return f"{core}: never saw the go of launch {launch} (its last: {mail[0]})"
    left = ["BRISC"] if mail[4] != launch else []
    left += [name for name, done in zip(("TRISC0", "TRISC1", "TRISC2"), triscs[1:]) if done != triscs[0]]
    lines = [f"{core}: launch {launch} not finished by {', '.join(left) or 'none: its report to the dispatcher is lost'}",
             "SYNC " + ", ".join(f"{name} {v}" for name, v in zip(SYNC_NAMES, sync)) + "; FLAGS set: "
             + (", ".join(f"{i}={v}" for i, v in enumerate(flags) if v) or "none"),
             f"NoC requests in flight by id: {({i: n for i in range(16) if (n := niu(bh.NIU_REGS['OUTSTANDING'] + 4 * i))}) or 'none'}; "
             f"write acknowledgements {niu(bh.NIU_REGS['WR_ACKS'])}, read responses {niu(bh.NIU_REGS['READS'])}"]
    if isinstance(self.chip.pci, pci.Card):  # ttsim doesn't model the pc snapshots, and reading one stops it
      lines.append("pc " + ", ".join(f"{name} {self.chip.read32(x, y, bh.PC_SNAPSHOT + 4 * i):#x}"
                                     for i, name in enumerate(("BRISC", "NCRISC", "TRISC0", "TRISC1", "TRISC2"))))
    return "\n    ".join(lines)
