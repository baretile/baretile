# The runtime. Firmware stays loaded on the Tensix cores, and kernels launch without resetting anything. One core, the
# dispatcher, pulls commands from a ring in host memory.
import atexit, collections, functools, hashlib, os, pathlib, shutil, struct, subprocess, tempfile, time
from baretile import bh, pci

# The L1 layout, which is ours to choose. Firmware sits at 0 and starts with a jump for each core: BRISC's at 0 and
# TRISC i's at 4 + 4i. A worker has, in order:
# - MAILBOX: [go, kernel, arguments, workers] [done, dispatcher's x | y << 6, scratch, index]. The dispatcher
#   multicasts the first half.
# - TRISCS: [go, done 0, 1, 2] for the TRISCs
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
// Wait until a multicast has all its acknowledgements, which is when the ack counter reaches acks. A multicast leaves
// the outstanding count wrong, up once and down once per tile, so no other request may be in flight. We clear it after.
static inline void mcast_done(unsigned acks) {{
  while (NIU(WR_ACKS) != acks) {{}}
  NIU(CLEAR_OUTSTANDING) = 0xFFFF;
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
         "-O2", "-ffreestanding", "-nostdlib", "-ffunction-sections", "-fuse-ld=lld"]
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

def compile_rv32(src):
  """Prepend DEVICE to C source, compile for rv32, and link a flat image at address 0. Put .start first if
  present, otherwise kmain. PC-relative addressing lets the image run wherever it's loaded; this requires
  medany, no linker relaxation and no jump tables. Cache images by compiler version, flags, linker script
  and source. Entries expire after a week without use."""
  link = ("SECTIONS { .text 0 : { *(.start) *(.text.kmain) *(.text*) *(.rodata*) *(.srodata*) *(.data*) *(.sdata*) "
          ". = ALIGN(4); __bss_start = .; *(.sbss*) *(.bss*) . = ALIGN(4); __bss_end = .; } }")  # one section: .bss is zeros in the image
  cached = CACHE / hashlib.sha256(_clang_version() + repr(CLANG).encode() + link.encode() + (DEVICE + src).encode()).hexdigest()
  if cached.exists():
    os.utime(cached)
    return cached.read_bytes()
  _prune()
  if not shutil.which("ld.lld"): raise RuntimeError("clang links with lld, and ld.lld is not on PATH: install lld")
  with tempfile.TemporaryDirectory() as d:
    (ld := pathlib.Path(d, "link.ld")).write_text(link)
    out = pathlib.Path(d, "prog.bin")
    subprocess.run(CLANG + [f"-Wl,-T,{ld},--oformat=binary", "-x", "c", "-", "-o", str(out)], input=(DEVICE + src).encode(), check=True)
    image = out.read_bytes()
  CACHE.mkdir(parents=True, exist_ok=True)
  (part := cached.with_suffix(f".{os.getpid()}")).write_bytes(image)
  os.replace(part, cached)  # atomic, since two processes may compile the same image concurrently
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
  offsets, with 0 for an absent program, followed by the programs. The image can run at any load address."""
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

# A launch on a worker. BRISC sees go in its mailbox, with the addresses of the kernel and its arguments, and n, the
# number of workers it's for. Workers n and up sit it out. The arguments start with a number of units to share, which
# BRISC turns into this worker's [first unit, how many]. Then BRISC zeroes the SYNC counters and drops the cached
# instructions of all four cores, since the kernel may sit where another one was. It tells the TRISCs to go. The cores
# run their programs from the kernel's header, and a TRISC then waits for its Tensix thread to finish what it pushed.
# Once all four cores are done, BRISC reports. It notes the launch in its own mailbox, for the host to read if a
# launch hangs. Then it adds 1 to the dispatcher's count with a posted NoC atomic increment: nothing comes back, and no
# counter moves at either end. Where the NIU counts read data (COUNTED: a card or ttsim-chaos), BRISC first adds the
# 64-byte flits that the launch's reads brought in to the dispatcher's sum, the same way. Both go on the same static
# VC, so the sum is complete once the count is.
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
  for (unsigned seen = 0;;) {{
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
    a[0] = me * units / n; a[1] = (me + 1) * units / n - a[0];
    for (unsigned i = 0; i < {SYNC_COUNTS}; i++) ((volatile unsigned *){SYNC:#x})[i] = 0;
    CFG[{bh.CFG_IC_INVALIDATE}] = 0xF;  // drop the cached instructions of BRISC and the TRISCs
    l1_settle({bh.TENSIX_CFG:#x});
    l1_settle({SYNC:#x});  // the SYNC counters are zero before any core of the kernel looks at them
    TRISCS[0] = seen;
    run(0);
    noc_wait();  // what the kernel sent has landed before we report done
    for (unsigned i = 1; i <= 3; i++) while (TRISCS[i] != seen) asm volatile("fence" ::: "memory");
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

# A command is a 16-byte header [length, kernel address, argument bytes, workers], then the arguments, padded to a
# multiple of 64 bytes. Length 0 means the rest of the ring is unused. The kernel is already in the workers' L1
# (Runtime.load).
# A read of host memory waits on PCIe, so the dispatcher reads everything new in the ring with one noc_read, up to the
# ring's end and at most STAGE bytes. It runs the commands from its L1. The arguments and the go each reach all the
# workers in a multicast, even when the launch uses fewer of them. The workers report done by adding to a shared word,
# DONE. A multicast gets an acknowledgement from each worker, so the dispatcher waits on the acknowledgement counter.
# The request counter goes up once and down once per worker, so it's cleared after.
# One launch runs at a time, but the next one's arguments go into the other buffer while it runs, so it starts the
# moment the last worker reports. When a launch is done, the dispatcher writes its [go, last report] times on its wall
# clock and the 64-byte flits its workers read into a ring of TIMES entries in host memory, after the commands. The
# flits are how much the sum grew, read before the next launch goes. The host gets a profile of the launches, timed
# by the chip's clock. Then the dispatcher writes [read pointer, launches done] into the status. Both are posted
# writes on the same static VC, so they arrive in that order. The dispatcher doesn't wait for PCIe, only for their
# bytes to leave its outbox.
TIMES = 1024
DISPATCHER = f"""
#define CONFIG ((volatile unsigned *){DISPATCH:#x})
#define OUT ((volatile unsigned *){OUTBOX:#x})
{jumps("brisc", "park", "park", "park")}  // its TRISCs have nothing to do
__attribute__((naked)) void park(void) {{ asm("1: j 1b"); }}
__attribute__((naked)) void brisc(void) {{ asm("li sp, {bh.BRISC_STACK_TOP:#x}\\n j loop"); }}
// n bytes from l1 to addr in the workers, and wait until they all have them
static void multicast(unsigned l1, unsigned addr, unsigned n) {{
  unsigned acks = NIU(WR_ACKS) + CONFIG[7];
  l1_settle(l1);
  noc_issue(l1, noc_me(), addr, CONFIG[6], MCAST, n);  // the tiles of the rectangle but this one, which is in it
  mcast_done(acks);
}}
// The go copies the first 16 bytes of the outbox to the workers' mailboxes. It's always the same multicast, so request
// initiator 1 holds it and a launch starts with one store.
static void go(void) {{ unsigned acks = NIU(WR_ACKS) + CONFIG[7]; NIU1(CMD) = 1; mcast_done(acks); }}
void loop(void) {{
  unsigned long long ring = CONFIG[1] | (unsigned long long)CONFIG[2] << 32, status = CONFIG[4] | (unsigned long long)CONFIG[5] << 32;
  unsigned size = CONFIG[3], rptr = 0, launches = 0, me = noc_me();
  unsigned ready = 0, ready_n = 0, ready_end = 0;  // a launch whose arguments and go are in place, or 0
  // the launch the workers are on, the ring pointer past it, the report count once it's done, and when its go went
  unsigned running = 0, running_end = 0, reports = 0, began = 0;
  unsigned long long times = ring + size;  // launch n's [go, last report, flits read] at times + n % TIMES * 16
  unsigned words = 0;  // the flits the workers read, up to the last launch done
  unsigned staged = 0, first = 0;  // the ring's bytes [first, staged) are in STAGING
  NIU1(TARG_LO) = {OUTBOX:#x}; NIU1(TARG_MID) = 0; NIU1(TARG_HI) = noc_me(); NIU1(RET_LO) = {MAILBOX:#x}; NIU1(RET_MID) = 0;
  NIU1(RET_HI) = CONFIG[6]; NIU1(PACKET_TAG) = 0; NIU1(CTRL) = MCAST; NIU1(LEN) = 16;
  for (;;) {{
    asm volatile("fence" ::: "memory");  // the host rings the doorbell with its write pointer
    if (!ready && CONFIG[0] != rptr) {{  // the next command, while the last launch runs
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
    }}
    if (ready && running)  // the next launch is ready, so spin on DONE until the last report of this one
      while (*(volatile unsigned *){DONE:#x} != reports) asm volatile("fence" ::: "memory");
    asm volatile("fence" ::: "memory");
    unsigned done = running && *(volatile unsigned *){DONE:#x} == reports ? running : 0, end = running_end;
    unsigned took = began, finished = done ? CLOCK : 0, read = done ? *(volatile unsigned *){DONE + 4:#x} - words : 0;
    words += read;
    if (done) running = 0;
    if (ready && !running) {{  // the workers have finished the last launch
      began = CLOCK;
      go();
      running = ready; running_end = ready_end; reports += ready_n; ready = 0;
    }}
    if (done) {{  // after the next go: better the host hears late than the next launch starts late
      while (NIU(OUTGOING)) {{}}  // wait until the last launch's times and status have left the outbox
      OUT[4] = end; OUT[5] = done; OUT[8] = took; OUT[9] = finished; OUT[10] = read;
      l1_settle({OUTBOX + 16:#x});
      // its times, then the status that says it's done, in that order
      noc_issue({OUTBOX + 32:#x}, me, times + done % {TIMES} * 16, {bh.noc_xy(*bh.PCIE)}, POST, 16);
      noc_issue({OUTBOX + 16:#x}, me, status, {bh.noc_xy(*bh.PCIE)}, POST, 8);
    }}
  }}
}}
"""

class Runtime:
  """Run kernels on the Tensix rectangle from start to end, using the available tiles by default. The last tile
  dispatches work to the others. Kernels stay in L1 until the kernel area fills. The dispatcher reads
  commands from host memory and executes them in order. A new Runtime halts any older runtime that overlaps
  its rectangle."""
  def __init__(self, chip, start=None, end=None, ring=1 << 20):
    start, end = start or chip.tensix[0], end or chip.tensix[-1]
    assert ring & (ring - 1) == 0 and ring % 64 == 0, "ring size must be a power of two: pointers wrap at 2**32"
    cores = [(x, y) for x, y in chip.tensix if start[0] <= x <= end[0] and start[1] <= y <= end[1]]
    assert len(cores) >= 2, "a rectangle needs a dispatcher and at least one worker"
    self.chip, self.launches, self.wptr = chip, 0, 0
    self.queued = collections.deque()  # (launch, workers) for each launch that may not be done yet
    # DRAM: holes lists free (address, size) ranges; used maps address to size.
    # freed pairs an address with the last queued launch that could use it.
    self.holes, self.used, self.freed = [(0, bh.DRAM_BANK_SIZE)], {}, []
    # loaded maps each kernel image in L1 to its address, and room is where the next one goes
    self.start, self.end, self.loaded, self.room = start, end, {}, KERNELS
    # how many times the loaded kernels were evicted. A recorded command names its kernels by address, so a replay
    # checks this first.
    self.evictions = 0
    *self.cores, self.dispatcher = cores  # the dispatcher is end, the last core of the rectangle
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
    # Halt all the cores of an older runtime that overlaps this one, including those we don't take. Idle cores cost
    # nothing on a card, but ttsim simulates the cores that are running, so they slow it down.
    for (s, e), theirs in list(chip.taken.items()):
      if theirs & set(cores): chip.halt(*e, s); del chip.taken[s, e]
    chip.taken[start, end] = set(cores)
    # Halt the rectangle now, and again at exit. A dispatcher that's still running would read and write its hugepage
    # after the process that owned it is gone, and the next process may get the same page.
    chip.halt(*end, start)
    atexit.register(chip.halt, *end, start)
    buf, bus = chip.pci.alloc(1 << 21)  # a hugepage for the status, the ring, the times, then tensors read back
    back = 64 + ring + 16 * TIMES
    assert back + 4096 <= 1 << 21, "the ring leaves no room in the hugepage to read a tile back"
    self.status, self.ring, self.times, self.back, self.back_bus = buf[:64], buf[64:64 + ring], buf[64 + ring:back], buf[back:], bus + back
    self.sent = [("", 0, b"")] * TIMES  # (name, workers, kernel) of the last TIMES launches, for profile and baretile.viz
    self.numbers = {}  # (value, dtype) -> a tensor holding that number, uploaded once for tensor.py's constants
    # while tensor.jit records a call: its launches, and the DRAM it took, which dram_free won't give back
    self.log, self.pinned = None, set()
    self.counts = chip.pci.counts  # whether profile knows the bytes each launch read
    worker, dispatcher = compile_rv32("#define COUNTED\n" * self.counts + FIRMWARE), compile_rv32(DISPATCHER)
    assert max(len(worker), len(dispatcher)) <= MAILBOX, "firmware overlaps the mailbox"
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
    chip.write(dx, dy, OUTBOX, bytes(64))  # clear completion status left by the previous dispatcher
    self.doorbell = chip.map(dx, dy, DISPATCH)  # its own TLB, so ringing the doorbell doesn't have to move a window first
    for i in range(3): chip.write32(*end, bh.TRISC_RESET_PC + 4 * i, 4 + 4 * i, start)
    chip.write32(*end, bh.TRISC_RESET_PC_OVERRIDE, 0b111, start)
    chip.write32(*end, bh.SOFT_RESET, bh.NCRISC, start)  # BRISC and the TRISCs go, and NCRISC stays in reset

  def load(self, kernel):
    """Load a kernel into the workers' L1 and return its address, reusing an existing copy if present. Loading
    multicasts over the rectangle, including the dispatcher. If space runs out, wait for queued work before
    evicting kernels. Strict TLB ordering makes the kernel arrive before a later doorbell write."""
    if kernel not in self.loaded:
      assert len(kernel) <= ZEROS - KERNELS, f"kernel is {len(kernel)} bytes, and kernel space holds {ZEROS - KERNELS}"
      if self.evicts(kernel): self.evict()
      self.chip.write(*self.end, self.room, kernel, self.start)
      self.loaded[kernel], self.room = self.room, self.room + len(kernel)  # compile_kernel makes it a multiple of 16
    return self.loaded[kernel]
  # whether loading kernel would evict the others
  def evicts(self, kernel): return kernel not in self.loaded and self.room + len(kernel) > ZEROS

  def evict(self):  # forget the loaded kernels, once the launches that use them are done
    self.wait()
    self.loaded, self.room, self.evictions = {}, KERNELS, self.evictions + 1

  def launch(self, kernel, args=(), units=None, name=""):
    """Queue a compiled kernel, loading it if needed. Divide units between workers in order; the default is one
    unit per worker. Workers receive [first unit, unit count, *args]. name identifies the launch in
    profiles."""
    if self.log is not None: self.log.append((kernel, args, units, name))
    self.send(*self.block([(kernel, args, units, name)]))

  def block(self, launches):
    """Convert [(kernel, args, units, name)] to commands for send, plus (name, worker count, kernel) per launch. Load
    the kernels first, since commands contain their L1 addresses. Commands can be replayed until a kernel is
    evicted. If loading requires eviction, send the commands built so far before waiting for space."""
    cmds, sent = [], []
    for kernel, args, units, name in launches:
      if cmds and self.evicts(kernel): self.send(b"".join(cmds), sent); cmds, sent = [], []
      cmd, n = self.command(kernel, args, units)
      cmds.append(cmd); sent.append((name, n, kernel))
    return b"".join(cmds), sent

  def command(self, kernel, args, units):  # a launch as the ring holds it: its bytes, and how many workers it takes
    kernel, n = self.load(kernel), min(units or len(self.cores), len(self.cores))
    body = struct.pack(f"<{2 + len(args)}I", units or n, 0, *args)
    assert len(body) <= ARGS_SIZE, f"{len(body)} bytes of arguments, and an argument buffer holds {ARGS_SIZE}"
    length = -(-(16 + len(body)) // 64) * 64
    return struct.pack("<4I", length, kernel, len(body), n) + body.ljust(length - 16, b"\0"), n

  def send(self, cmds, launches):
    """Write commands back to back into the ring, then ring the doorbell. The doorbell is the host's one PCIe write,
    however many commands there are. launches has (name, workers, kernel) for each."""
    size, at, length = len(self.ring), self.wptr % len(self.ring), len(cmds)
    skip = size - at if at + length > size else 0  # commands don't wrap, so mark the tail unused and start over
    assert length + skip <= size, f"{length} bytes of commands don't fit in a ring of {size}"
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
    assert self.launches - since <= TIMES, f"{self.launches - since} launches to profile, but only the last {TIMES} are kept"
    totals = collections.defaultdict(lambda: [0, 0, 0])
    for n in range(since + 1, self.launches + 1):
      go, done, flits = struct.unpack_from("<3I", self.times, n % TIMES * 16)
      for i, v in enumerate((1, (done - go) % 2**32, 64 * flits)): totals[self.sent[n % TIMES][0]][i] += v
    return sorted(((name, n, cycles, read if self.counts else None) for name, (n, cycles, read) in totals.items()), key=lambda r: -r[2])

  def args(self):  # where the last launch's arguments are on each of its workers. some kernels write results there
    return ARGS + self.launches % 2 * ARGS_SIZE

  def dram_alloc(self, size):
    """Allocate size bytes in each of the 8 DRAM banks, at the same address, from the first free hole that's big
    enough."""
    size = max(64, -(-size // 64) * 64)  # NoC transfers to DRAM want 64-byte alignment
    for i, (addr, room) in enumerate(self.holes):
      if room >= size:
        self.holes[i:i + 1] = [(addr + size, room - size)] if room > size else []
        self.used[addr] = size
        if self.log is not None: self.pinned.add(addr)
        return addr
    if self.freed:  # what queued launches still hold may be enough once they finish
      self.wait()
      return self.dram_alloc(size)
    raise MemoryError(f"no {size} bytes free in DRAM")

  def dram_free(self, addr):
    """Release an allocation after the launches queued so far have finished, then merge adjacent free ranges.
    Keep allocations made during recording, since replay uses the same addresses."""
    if addr in self.pinned: return
    self.freed.append((self.launches, addr))
    self.reclaim()

  def reclaim(self):  # freed DRAM that no unfinished launch can use goes back into the holes
    done = self.launches - (self.launches - self.read_status(1)) % 2**32
    while self.queued and self.queued[0][0] <= done: self.queued.popleft()
    for launches, addr in [f for f in self.freed if f[0] <= done]:
      self.freed.remove((launches, addr))
      self.holes = sorted(self.holes + [(addr, self.used.pop(addr))])
      for i in range(len(self.holes) - 1, 0, -1):
        (a, n), (b, m) = self.holes[i - 1], self.holes[i]
        if a + n == b: self.holes[i - 1:i + 1] = [(a, n + m)]

  def read_status(self, i): return int.from_bytes(self.status[4 * i:4 * i + 4], "little")

  def until(self, cond, what, timeout=60):
    """Wait until cond(), as long as launches keep finishing. If none finishes in timeout seconds, the chip has hung. A
    launch can take seconds on the simulator."""
    while not cond():
      done = self.read_status(1)
      self.chip.wait(lambda: cond() or self.read_status(1) != done, what, timeout)

  def wait(self, timeout=60):
    """Poll completion status in host memory, then reclaim memory freed by completed launches. On timeout,
    report the current launch and its unfinished workers."""
    try: self.until(lambda: self.read_status(1) == self.launches % 2**32, f"finish launch {self.launches}", timeout)
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
    x, y = core
    words = lambda addr, n: struct.unpack(f"<{n}I", self.chip.read(x, y, addr, 4 * n))
    niu = lambda off: self.chip.read32(x, y, bh.NIU + off)
    mail, triscs, sync, flags = words(MAILBOX, 8), words(TRISCS, 4), words(SYNC, len(SYNC_NAMES)), words(FLAGS, 32)
    if mail[0] != launch: return f"{core}: never saw the go of launch {launch} (its last: {mail[0]})"
    left = [name for name, done in (("BRISC", mail[4]), *zip(("TRISC0", "TRISC1", "TRISC2"), triscs[1:])) if done != launch]
    lines = [f"{core}: launch {launch} not finished by {', '.join(left) or 'none: its report to the dispatcher is lost'}"
             + ("" if triscs[0] == launch else f" (BRISC never started the TRISCs: their go is {triscs[0]})"),
             "SYNC " + ", ".join(f"{name} {v}" for name, v in zip(SYNC_NAMES, sync)) + "; FLAGS set: "
             + (", ".join(f"{i}={v}" for i, v in enumerate(flags) if v) or "none"),
             f"NoC requests in flight by id: {({i: n for i in range(16) if (n := niu(bh.NIU_REGS['OUTSTANDING'] + 4 * i))}) or 'none'}; "
             f"write acknowledgements {niu(bh.NIU_REGS['WR_ACKS'])}, read responses {niu(bh.NIU_REGS['READS'])}"]
    if isinstance(self.chip.pci, pci.Card):  # ttsim doesn't model the pc snapshots, and reading one stops it
      lines.append("pc " + ", ".join(f"{name} {self.chip.read32(x, y, bh.PC_SNAPSHOT + 4 * i):#x}"
                                     for i, name in enumerate(("BRISC", "NCRISC", "TRISC0", "TRISC1", "TRISC2"))))
    return "\n    ".join(lines)
