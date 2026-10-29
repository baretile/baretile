# NoC access through PCIe TLB windows. Chip also reads ARC telemetry and coordinate tables,
# resets Tensix cores, and transfers data through Ethernet links.
import hashlib, importlib.util, pathlib, shutil, struct, subprocess, time
from baretile import bh, pci

def tlb_config(**fields):
  cfg = 0
  for name, val in fields.items():
    first, width = bh.TLB_FIELDS[name]
    if not (0 <= val < 1 << width): raise ValueError(f"TLB {name}={val:#x} does not fit in {width} bits")
    cfg |= val << first
  return cfg

class Chip:
  # boot is how long to wait for ARC firmware, in seconds. After a reset it takes minutes, and umd allows 300.
  def __init__(self, pci, boot=2.0):
    if (ids := pci.cfg_read(0)) != bh.VENDOR | bh.DEVICE << 16: raise RuntimeError(f"not a Blackhole: {ids:#x}")
    self.pci, self.target = pci, None  # TLB 0 is the shared window, and map hands out the others
    self.taken = {}  # (start, end) -> runtime resources; these do not keep the Runtime alive
    self.dram_holes = [(0, bh.DRAM_BANK_SIZE)]  # all runtimes on this chip share its DRAM
    self.free_tlbs, self.kept = iter(range(1, bh.TLB_COUNT)), {}
    self.wait(lambda: self.read32(*bh.ARC, bh.ARC_BOOT_STATUS) & 7 == bh.ARC_READY, "report ARC firmware ready", boot)
    if not (int.from_bytes(pci.read(bh.NIU_CFG_0, 4), "little") & bh.NOC_TRANSLATED):
      raise RuntimeError("NoC coordinate translation is off, but bh.py assumes it")
    self.arc_msg(bh.ASIC_STATE0)  # leave the state that tt-kmd puts the chip in when it lets go of it
    self.arc_msg(bh.POWER_ON)  # tt-kmd clock-gates Tensix whenever nobody has the card open
    self.wait(lambda: self.read32(*bh.ARC, bh.ARC_TELEMETRY[0]), "publish its telemetry", boot)
    # Ask for the busy AI clock and, like UMD, wait up to 0.2 s for it to come within 5% of bh.AICLK. If it doesn't,
    # we use the clock it reports. ttsim reports 1000 MHz.
    self.arc_msg(bh.AICLK_GO_BUSY)
    busy, end = bh.AICLK / 1e6, time.monotonic() + 0.2
    while abs(self.telemetry().get(bh.TAG_AICLK, 0) - busy) > 0.05 * busy and time.monotonic() < end: self.pci.tick()
    t = self.telemetry()
    missing = {bh.TAG_AICLK, bh.TAG_GDDR_STATUS, bh.TAG_TENSIX_COLS, bh.TAG_GDDR_ENABLED} - t.keys()
    if missing: raise RuntimeError(f"ARC telemetry is missing tags {sorted(missing)}. Is the firmware too old?")
    self.aiclk = t[bh.TAG_AICLK]  # MHz, after asking for busy
    self.tensix = [(x, y) for y in bh.TENSIX_Y for x in bh.TENSIX_X[:t[bh.TAG_TENSIX_COLS].bit_count()]]
    table = lambda at: [int.from_bytes(pci.read(at + i // 6 * 4, 4), "little") >> i % 6 * 5 & 31 for i in range(32)]
    # how firmware translates around harvesting: the raw x of each translated x, and the raw y of each translated y
    self.noc_x, self.noc_y = map(table, bh.NOC_TABLES)
    self.banks = [b for b in range(len(bh.DRAM)) if t[bh.TAG_GDDR_ENABLED] >> b & 1]  # the DRAM banks firmware enabled
    # the GDDR status of each bank: 0 still training, 1 done, 2 or 3 failed
    training = lambda: [self.telemetry()[bh.TAG_GDDR_STATUS] >> 2 * b & 3 for b in self.banks]
    self.wait(lambda: 0 not in training(), "finish GDDR training", 300)
    bad = [b for b, s in zip(self.banks, training()) if s != 1]
    if bad: raise RuntimeError(f"GDDR banks {bad} failed training (UMD retrains with ARC message 0xB6)")

  def raw(self, x, y): return (x if y < 2 else self.noc_x[x]), self.noc_y[y]  # tile (x, y) on the raw NoC0 grid, for bh.hops

  def map(self, x, y, addr, start=None, tlb=None):
    """Map the 2 MiB block containing (x, y, addr) and return its BAR0 offset. With start, writes multicast
    over the rectangle from start to (x, y). If tlb is omitted, reserve a window and reuse it on later calls
    for the same block. Runtimes with the same dispatcher therefore share a doorbell mapping."""
    base = addr & ~(bh.TLB_WINDOW - 1)
    if tlb is None:
      if (x, y, base, start) not in self.kept:
        if (free := next(self.free_tlbs, None)) is None: raise RuntimeError(f"all {bh.TLB_COUNT} TLBs are taken")
        self.kept[x, y, base, start] = free
      tlb = self.kept[x, y, base, start]
    mcast = {} if start is None else dict(x_start=start[0], y_start=start[1], mcast=1, static_vc=1, vc_class=bh.MCAST_VC)
    cfg = tlb_config(addr=base >> 21, x_end=x, y_end=y, ordering=bh.STRICT, **mcast)
    if not 0 <= tlb < bh.TLB_COUNT: raise ValueError(f"TLB index must be in [0, {bh.TLB_COUNT})")
    for i in range(3): self.pci.write(bh.TLB_CONFIG + 12 * tlb + 4 * i, (cfg >> 32 * i & 0xFFFFFFFF).to_bytes(4, "little"))
    if tlb == 0: self.target = (x, y, base, start)
    return tlb * bh.TLB_WINDOW + addr - base

  def window(self, x, y, addr, start=None):  # TLB 0 is shared, and moves only when an access needs another block
    base = addr & ~(bh.TLB_WINDOW - 1)
    if self.target != (x, y, base, start): self.map(x, y, base, start, 0)
    return addr - base

  def chunks(self, x, y, addr, n, start=None):  # [addr, addr+n) as (BAR0 offset, size) pieces, split where the window moves
    if addr < 0 or n < 0: raise ValueError("addresses and transfer sizes must be nonnegative")
    while n:
      size = min(n, bh.TLB_WINDOW - addr % bh.TLB_WINDOW)
      yield self.window(x, y, addr, start), size
      addr, n = addr + size, n - size

  def read(self, x, y, addr, n): return b"".join(self.pci.read(off, size) for off, size in self.chunks(x, y, addr, n))
  def write(self, x, y, addr, data, start=None):
    """Write data at NoC address (x, y, addr). With start, multicast to the Tensix tiles from start to (x, y)."""
    done = 0
    for off, size in self.chunks(x, y, addr, len(data), start):
      self.pci.write(off, data[done:done + size])
      done += size
  def read32(self, x, y, addr): return int.from_bytes(self.read(x, y, addr, 4), "little")
  def write32(self, x, y, addr, val, start=None): self.write(x, y, addr, val.to_bytes(4, "little"), start)

  def wait(self, cond, what, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not cond():
      if time.monotonic() > deadline: raise TimeoutError(f"chip did not {what} within {timeout}s")
      self.pci.tick()

  def arc_msg(self, header):
    """Send ARC a message and return the 8-word response. The low byte of header is the message code."""
    (x, y), entry = bh.ARC, bh.ARC_QUEUE_ENTRY
    qcb = self.read32(x, y, bh.ARC_QUEUES)
    base, n = self.read32(x, y, qcb), self.read32(x, y, qcb + 4) & 0xFF
    q = base + bh.APP_QUEUE * (entry + 2 * n * entry)
    req_w, resp_r, req_r, resp_w = (q + at for at in bh.ARC_QUEUE_PTRS)
    wptr, rptr = self.read32(x, y, req_w), self.read32(x, y, resp_r)
    # drop any response that nobody read, left by a client that died. kmd does the same
    if (left := self.read32(x, y, resp_w)) != rptr: self.write32(x, y, resp_r, rptr := left)
    if (wptr - self.read32(x, y, req_r)) % (2 * n) >= n: raise RuntimeError("ARC message queue full")
    self.write(x, y, q + entry + wptr % n * entry, struct.pack("<8I", header, *[0] * 7))
    self.write32(x, y, req_w, (wptr + 1) % (2 * n))
    self.write32(x, y, *bh.ARC_IRQ)
    self.wait(lambda: self.read32(x, y, resp_w) != rptr, f"answer ARC message {header & 0xFF:#x}", 1.0)
    resp = struct.unpack("<8I", self.read(x, y, q + entry * (1 + n + rptr % n), entry))
    self.write32(x, y, resp_r, (rptr + 1) % (2 * n))
    if resp[0] & 0xFF >= bh.ARC_FAILED: raise RuntimeError(f"ARC message {header & 0xFF:#x} failed: {resp[0]:#x}")
    return resp

  def telemetry(self):
    """{tag: value} from the table ARC keeps up to date."""
    x, y = bh.ARC
    table, values = (self.read32(x, y, a) for a in bh.ARC_TELEMETRY)
    n = self.read32(x, y, table + 4)
    tags = struct.unpack(f"<{n}I", self.read(x, y, table + 8, 4 * n))
    vals = struct.unpack(f"<{n}I", self.read(x, y, values, 4 * n))
    return {t & 0xFFFF: vals[t >> 16] for t in tags}

  def halt(self, x, y, start=None): self.write32(x, y, bh.SOFT_RESET, bh.ALL_CORES, start)  # hold BRISC, NCRISC and the TRISCs in reset

  def links(self):
    """Return (ASIC id, links), where links maps a channel to (peer ASIC id, peer channel). Include links that
    firmware reports as up. The chip_info records for both ends are in each Ethernet tile's L1."""
    enabled, me, up = self.telemetry().get(bh.TAG_ETH_ENABLED, 0), None, {}
    for ch, (x, y) in enumerate(bh.ETH):
      if enabled >> ch & 1 and self.read32(x, y, bh.ETH_PORT_STATUS) == bh.ETH_PORT_UP:
        local, remote = self.read(x, y, bh.ETH_LOCAL, 32), self.read(x, y, bh.ETH_REMOTE, 32)
        me, up[ch] = struct.unpack_from("<2I", local, 20), (struct.unpack_from("<2I", remote, 20), remote[2])
    return me, up

  def send(self, ch, src, dst, n, timeout=1.0):
    """Copy n bytes from src in Ethernet channel ch's L1 to dst in the peer's L1, using TX queue 0. TT-link
    delivers writes once and in order, so a flag sent after the data confirms its arrival. The queue may
    still be reading src when this returns. Leave src unchanged until the queue is free; the next send waits
    for it."""
    (x, y), reg = bh.ETH[ch], lambda name: bh.ETH_TXQ0 + bh.ETH_TXQ[name]
    if not (min(src, dst) >= 0 and n > 0 and src % 16 == dst % 16 == n % 16 == 0 and max(src, dst) + n <= bh.ETH_FIRMWARE):
      raise ValueError("Ethernet addresses and size must be 16-byte aligned and stay below firmware L1")
    def free():  # the busy bit of STATUS is wrong if polled too soon after a send, so read another register first (BH-55)
      self.read32(x, y, reg("CMD"))
      return not self.read32(x, y, reg("STATUS")) & bh.ETH_BUSY
    self.wait(free, f"free Ethernet channel {ch}'s TX queue", timeout)
    for name, v in (("START", src), ("SIZE", n), ("DEST", dst), ("CMD", bh.ETH_TT_L1_WRITE)): self.write32(x, y, reg(name), v)

def check():
  """Check what baretile needs, without opening the device: clang and lld to build kernels, NumPy for the models, and
  the simulator or a card, which pci.preflight checks. Return [(level, message)], where level is "ok", "warn" or
  "fail". baretile can't run after a fail."""
  targets = subprocess.run(["clang", "--print-targets"], capture_output=True, text=True).stdout if shutil.which("clang") else ""
  found = [("ok", "clang builds for RISC-V") if "riscv32" in targets else
           ("fail", "baretile builds kernels with clang for riscv32. Install clang, with its RISC-V target"),
           ("ok", "ld.lld links the kernels") if shutil.which("ld.lld") else ("fail", "ld.lld isn't on PATH. Install lld"),
           ("ok", "NumPy is there for baretile.models") if importlib.util.find_spec("numpy") else
           ("warn", "baretile.models needs NumPy: pip install 'baretile[models]'")]
  if path := pci.simulator():
    try: tested = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest() == pci.TTSIM_SHA256
    except OSError as e: return found + [("fail", f"can't read the simulator at {path}: {e.strerror}")]
    return found + [("ok", f"the simulator at {path} is the tested ttsim release") if tested else
                    ("warn", f"the simulator at {path} isn't the tested release, {pci.TTSIM_URL}")]
  try: bdf = pci.find_cards()[0]
  except RuntimeError as e: return found + [("fail", str(e))]
  return found + [(level, f"{bdf}: {message}") for level, message in pci.preflight(bdf)]

# Report firmware, clock, Tensix tiles, DRAM banks and ARC telemetry. Run this before loading kernels on a new card,
# or on the simulator. check runs check() without opening the device, and ttsim downloads the tested simulator
# (pci.fetch_ttsim). For a card, release gives it back to tt-kmd (pci.release), and reset calls pci.reset before
# opening it.
if __name__ == "__main__":
  import os, sys
  command = sys.argv[1] if len(sys.argv) > 1 else None
  if command not in (None, "check", "ttsim", "release", "reset"): sys.exit("usage: python3 -m baretile.chip [check | ttsim | release | reset]")
  if command in ("release", "reset") and "TTSIM" in os.environ:
    sys.exit(f"{command} is for a card. ttsim has no driver, and a new process starts it fresh")
  if command == "ttsim":
    print(f"ttsim is at {pci.fetch_ttsim()}. baretile uses it when TTSIM isn't set and there's no card")
    sys.exit()
  if command == "check":
    found = check()
    for level, message in found: print(f"{level:5s}{message}")
    sys.exit(any(level == "fail" for level, _ in found))
  if command == "release":
    bdf = pci.find_cards()[0]
    print(f"{bdf}: {pci.release(bdf)} has the card")
    sys.exit()
  if reset := command == "reset":
    pci.reset(bdf := pci.find_cards()[0])
    dev = pci.Card(bdf)
  else: dev = pci.open_device()
  chip = Chip(dev, 300 if reset else 2)
  t = chip.telemetry()
  fw = ".".join(str(t.get(bh.TAG_FIRMWARE, 0) >> s & 255) for s in (24, 16, 8))
  print(f"{getattr(dev, 'sys', 'ttsim')}: firmware bundle {fw}. ARC is up and Tensix is powered. The AI clock is {t.get(bh.TAG_AICLK, 0)} MHz. "
        f"{len(chip.tensix)} Tensix tiles in raw columns {sorted({chip.raw(x, y)[0] for x, y in chip.tensix})}. DRAM banks {chip.banks} trained.")
  print("telemetry:", " ".join(f"{tag}={value:#x}" for tag, value in sorted(t.items())))
