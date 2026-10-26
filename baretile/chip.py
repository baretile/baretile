# A Blackhole, reached through its PCIe device interface. NoC reads and writes go through a shared 2 MiB TLB window,
# or through a TLB of their own (map). Chip also waits with a timeout, finds each tile on the raw grid (raw), talks to
# ARC and reads its telemetry, resets cores, and drives the Ethernet links (links, send).
import struct, time
from baretile import bh

def tlb_config(**fields):
  cfg = 0
  for name, val in fields.items():
    first, width = bh.TLB_FIELDS[name]
    assert 0 <= val < 1 << width, f"TLB {name}={val:#x} does not fit in {width} bits"
    cfg |= val << first
  return cfg

class Chip:
  # boot is how long to wait for ARC firmware, in seconds. After a reset it takes minutes, and umd allows 300.
  def __init__(self, pci, boot=2.0):
    assert (ids := pci.cfg_read(0)) == bh.VENDOR | bh.DEVICE << 16, f"not a Blackhole: {ids:#x}"
    self.pci, self.target = pci, None  # TLB 0 is the shared window, and map hands out the others
    self.taken = {}  # (start, end) -> the cores of each rectangle that a Runtime has taken
    self.free_tlbs, self.kept = iter(range(1, bh.TLB_COUNT)), {}
    self.wait(lambda: self.read32(*bh.ARC, bh.ARC_BOOT_STATUS) & 7 == bh.ARC_READY, "report ARC firmware ready", boot)
    assert int.from_bytes(pci.read(bh.NIU_CFG_0, 4), "little") & bh.NOC_TRANSLATED, "NoC coordinate translation is off, but bh.py assumes it"
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
    assert not missing, f"ARC telemetry is missing tags {sorted(missing)}. Is the firmware too old?"
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
    assert not bad, f"GDDR banks {bad} failed training (UMD retrains with ARC message 0xB6)"

  def raw(self, x, y): return (x if y < 2 else self.noc_x[x]), self.noc_y[y]  # tile (x, y) on the raw NoC0 grid, for bh.hops

  def map(self, x, y, addr, start=None, tlb=None):
    """Point a TLB at the 2 MiB block that holds NoC address (x, y, addr), and return the BAR0 offset that reaches
    addr. With start, writes go to that block in the Tensix cores of the rectangle from start to (x, y). Without tlb,
    the block keeps a TLB of its own, and later calls reuse it. A Runtime maps its doorbell this way, so a new Runtime
    with the same dispatcher doesn't take another TLB."""
    base = addr & ~(bh.TLB_WINDOW - 1)
    if tlb is None:
      if (x, y, base, start) not in self.kept:
        if (free := next(self.free_tlbs, None)) is None: raise RuntimeError(f"all {bh.TLB_COUNT} TLBs are taken")
        self.kept[x, y, base, start] = free
      tlb = self.kept[x, y, base, start]
    mcast = {} if start is None else dict(x_start=start[0], y_start=start[1], mcast=1, static_vc=1, vc_class=bh.MCAST_VC)
    cfg = tlb_config(addr=base >> 21, x_end=x, y_end=y, ordering=bh.STRICT, **mcast)
    for i in range(3): self.pci.write(bh.TLB_CONFIG + 12 * tlb + 4 * i, (cfg >> 32 * i & 0xFFFFFFFF).to_bytes(4, "little"))
    return tlb * bh.TLB_WINDOW + addr - base

  def window(self, x, y, addr, start=None):  # TLB 0 is shared, and moves only when an access needs another block
    base = addr & ~(bh.TLB_WINDOW - 1)
    if self.target != (x, y, base, start): self.map(x, y, base, start, 0)
    self.target = (x, y, base, start)
    return addr - base

  def chunks(self, x, y, addr, n, start=None):  # [addr, addr+n) as (BAR0 offset, size) pieces, split where the window moves
    while n:
      size = min(n, bh.TLB_WINDOW - addr % bh.TLB_WINDOW)
      yield self.window(x, y, addr, start), size
      addr, n = addr + size, n - size

  def read(self, x, y, addr, n): return b"".join(self.pci.read(off, size) for off, size in self.chunks(x, y, addr, n))
  def write(self, x, y, addr, data, start=None):
    """Write data at NoC address (x, y, addr). With start, multicast to the Tensix cores from start to (x, y)."""
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
    assert (wptr - self.read32(x, y, req_r)) % (2 * n) < n, "ARC message queue full"
    self.write(x, y, q + entry + wptr % n * entry, struct.pack("<8I", header, *[0] * 7))
    self.write32(x, y, req_w, (wptr + 1) % (2 * n))
    self.write32(x, y, *bh.ARC_IRQ)
    self.wait(lambda: self.read32(x, y, resp_w) != rptr, f"answer ARC message {header & 0xFF:#x}", 1.0)
    resp = struct.unpack("<8I", self.read(x, y, q + entry * (1 + n + rptr % n), entry))
    self.write32(x, y, resp_r, (rptr + 1) % (2 * n))
    assert resp[0] & 0xFF < bh.ARC_FAILED, f"ARC message {header & 0xFF:#x} failed: {resp[0]:#x}"
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
    """Return this chip's ASIC id, and its Ethernet links that are up as {channel: (peer's ASIC id, peer's channel)}.
    Tenstorrent's Ethernet firmware reports them in the L1 of each Ethernet tile, in chip_info for both ends."""
    enabled, me, up = self.telemetry().get(bh.TAG_ETH_ENABLED, 0), None, {}
    for ch, (x, y) in enumerate(bh.ETH):
      if enabled >> ch & 1 and self.read32(x, y, bh.ETH_PORT_STATUS) == bh.ETH_PORT_UP:
        local, remote = self.read(x, y, bh.ETH_LOCAL, 32), self.read(x, y, bh.ETH_REMOTE, 32)
        me, up[ch] = struct.unpack_from("<2I", local, 20), (struct.unpack_from("<2I", remote, 20), remote[2])
    return me, up

  def send(self, ch, src, dst, n, timeout=1.0):
    """Copy n bytes from src in the L1 of channel ch's tile to dst in the L1 of the tile at the other end of the link.
    It's a TT-link write by TX queue 0, which the link delivers once and in order: a write sent after it lands after
    it, and that's how you know it has landed. send returns while the queue may still be reading src, so don't change
    src until the queue is free again. The next send waits for that."""
    (x, y), reg = bh.ETH[ch], lambda name: bh.ETH_TXQ0 + bh.ETH_TXQ[name]
    assert src % 16 == dst % 16 == n % 16 == 0 and max(src, dst) + n <= bh.ETH_FIRMWARE, "16-byte aligned, below the firmware's L1"
    def free():  # the busy bit of STATUS is wrong if polled too soon after a send, so read another register first (BH-55)
      self.read32(x, y, reg("CMD"))
      return not self.read32(x, y, reg("STATUS")) & bh.ETH_BUSY
    self.wait(free, f"free Ethernet channel {ch}'s TX queue", timeout)
    for name, v in (("START", src), ("SIZE", n), ("DEST", dst), ("CMD", bh.ETH_TT_L1_WRITE)): self.write32(x, y, reg(name), v)

# Prints what a card reports: its firmware, clock, cores, DRAM banks and telemetry. It's the first thing to run on a new
# card, or on ttsim with TTSIM set. With `reset` it resets the card first (pci.reset), for a card whose NoC or firmware
# hangs.
if __name__ == "__main__":
  import os, sys
  from baretile import pci
  if reset := sys.argv[1:] == ["reset"]:
    assert "TTSIM" not in os.environ, "ttsim has no reset. A new process starts it fresh"
    pci.reset(bdf := pci.find_cards()[0])
    dev = pci.Card(bdf)
  else: dev = pci.open_device()
  chip = Chip(dev, 300 if reset else 2)
  t = chip.telemetry()
  fw = ".".join(str(t.get(bh.TAG_FIRMWARE, 0) >> s & 255) for s in (24, 16, 8))
  print(f"{getattr(dev, 'sys', 'ttsim')}: ARC firmware {fw} is up and Tensix is powered. The AI clock is {t.get(bh.TAG_AICLK, 0)} MHz. "
        f"{len(chip.tensix)} Tensix cores in raw columns {sorted({chip.raw(x, y)[0] for x, y in chip.tensix})}. DRAM banks {chip.banks} trained.")
  print("telemetry:", " ".join(f"{tag}={value:#x}" for tag, value in sorted(t.items())))
