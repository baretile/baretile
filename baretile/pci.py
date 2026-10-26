# The device at the PCIe level: config space, BAR0, and host memory that the device can DMA to. There are two backends
# with the same methods. Sim drives ttsim's libttsim.so, and Card drives a card through sysfs. reset() is a chip-level
# reset, for a card that hangs.
import array, ctypes, gc, mmap, os, time
from baretile import bh

SYS = "/sys/bus/pci"  # where the kernel shows the PCI bus. a test points it at a made-up one

class Sim:
  """A virtual Blackhole in libttsim.so, driven by forwarding config and BAR0 accesses to it. bdf is its PCI address,
  since a multi-chip build has several. There's one library per process, so Sims share it, its clock and the host
  memory it reaches. counts says whether it models NIU_MST_RD_DATA_WORD_RECEIVED, the count of 64-byte flits that
  a core's reads bring in, for Runtime.profile. ttsim-chaos does, and ttsim as released stops on it."""
  lib, dma, clocks, counts = None, [], 0, False  # dma: (bus address, size, host address, buffer). clocks: chip cycles so far
  def __init__(self, path, bdf=0):
    Sim.load(path)
    self.lib, self.bdf = Sim.lib, bdf
    self.bar0 = self.cfg_read(0x10) & ~0xF | self.cfg_read(0x14) << 32  # the TLB windows and registers that baretile uses

  @staticmethod
  def load(path):  # loads the library once per process
    if Sim.lib is None:
      Sim.lib = lib = ctypes.CDLL(path)
      lib.libttsim_pci_config_rd32.restype = ctypes.c_uint32
      lib.libttsim_pci_config_rd32.argtypes = [ctypes.c_uint32] * 2
      lib.libttsim_pci_config_wr32.argtypes = [ctypes.c_uint32] * 3
      lib.libttsim_pci_mem_rd_bytes.argtypes = lib.libttsim_pci_mem_wr_bytes.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint32]
      lib.libttsim_clock.argtypes = [ctypes.c_uint32]
      # the devices reach host memory through these callbacks, so they must stay referenced as long as the library lives
      cb = ctypes.CFUNCTYPE(None, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint32)
      Sim.cbs = (cb(lambda bus, p, n: ctypes.memmove(p, Sim.host(bus, n), n)), cb(lambda bus, p, n: ctypes.memmove(Sim.host(bus, n), p, n)))
      lib.libttsim_set_pci_dma_mem_callbacks(*Sim.cbs)
      lib.libttsim_init()
      Sim.counts = hasattr(lib, "libttsim_chaos")  # a marker that test/ttsim-chaos.patch adds

  @staticmethod
  def chips(path):  # PCI addresses of a build's chips. we scan the slots, since the chips may not be adjacent
    Sim.load(path)
    return [bdf for bdf in range(0, 1 << 16, 8) if Sim.lib.libttsim_pci_config_rd32(bdf, 0) == bh.DEVICE << 16 | bh.VENDOR]

  def cfg_read(self, off): return self.lib.libttsim_pci_config_rd32(self.bdf, off)
  def cfg_write(self, off, val): self.lib.libttsim_pci_config_wr32(self.bdf, off, val)

  def read(self, off, n):
    buf = ctypes.create_string_buffer(n)
    self.lib.libttsim_pci_mem_rd_bytes(self.bar0 + off, buf, n)
    return buf.raw
  def write(self, off, data): self.lib.libttsim_pci_mem_wr_bytes(self.bar0 + off, bytes(data), len(data))

  def alloc(self, size):
    """Allocate host memory that the device can reach. Returns (memoryview, bus address). Sim makes up the bus
    addresses, 4 GiB apart."""
    buf, bus = mmap.mmap(-1, size), (len(Sim.dma) + 1) << 32
    Sim.dma.append((bus, size, ctypes.addressof(ctypes.c_char.from_buffer(buf)), buf))
    return memoryview(buf), bus
  @staticmethod
  def host(bus, n):
    for base, size, addr, _ in Sim.dma:
      if base <= bus and bus + n <= base + size: return addr + bus - base
    raise RuntimeError(f"device DMA to {bus:#x}+{n}, which is not host memory we allocated")

  TICK = 10  # chip clocks per poll, so waits and cycle counts round up to a multiple of TICK
  def tick(self):  # the simulator runs only when we tick it, and a tick advances all of its chips
    self.lib.libttsim_clock(self.TICK)
    Sim.clocks += self.TICK
  def now(self): return Sim.clocks / bh.AICLK  # seconds of chip time by ttsim's clock, which doesn't match silicon

class Card:
  """A card, driven from userspace. Config space and BAR0 go through sysfs, and DMA memory comes from locked
  hugepages. It needs root, and the IOMMU off or in passthrough so that bus addresses are physical addresses. It
  unbinds the kernel driver if one is bound."""
  counts = True  # a card's NIUs count what their reads bring in (see Sim)
  def __init__(self, bdf):
    if os.geteuid() != 0: raise PermissionError("Card needs root for config space, the BARs, and physical addresses for DMA")
    self.sys = f"{SYS}/devices/{bdf}"
    if os.path.exists(f"{self.sys}/iommu_group/type") and (kind := open(f"{self.sys}/iommu_group/type").read().strip()) != "identity":
      raise RuntimeError(f"the IOMMU translates DMA from {bdf} ({kind}), but Card gives the card physical addresses. Boot with iommu=pt")
    if os.path.exists(f"{self.sys}/driver"): _write(f"{self.sys}/driver/unbind", bdf)
    _write(f"{self.sys}/enable", "1")
    self.cfg = os.open(f"{self.sys}/config", os.O_RDWR | os.O_SYNC)
    self.cfg_write(0x04, self.cfg_read(0x04) | 0x6)  # memory decode, and bus mastering so the device can reach host memory
    # Set Max Read Request Size in the PCI Express capability to 4 KiB, so the device reads host memory in requests of
    # up to 4 KiB. kmd sets it up the same way.
    if at := self.capability(0x10):
      os.pwrite(self.cfg, (self.cfg_read(at + 8) & 0x8FFF | 5 << 12).to_bytes(2, "little"), at + 8)
    # Map only BAR0, which has the TLB windows and registers that baretile uses. BAR4 may not be all there.
    fd = os.open(f"{self.sys}/resource0", os.O_RDWR | os.O_SYNC)
    self.map = mmap.mmap(fd, os.fstat(fd).st_size)
    self.bar0 = ctypes.addressof(ctypes.c_char.from_buffer(self.map))
    os.close(fd)
    # Zero the strided multicast words of TLBs 0-31, as kmd leaves them, so a multicast skips no tiles of its rectangle.
    # An earlier user may have set them.
    for i in range(32): self.write(bh.TLB_STRIDED + 4 * i, bytes(4))
  def cfg_read(self, off): return int.from_bytes(os.pread(self.cfg, 4, off), "little")
  def cfg_write(self, off, val): os.pwrite(self.cfg, val.to_bytes(4, "little"), off)
  def capability(self, cap):  # the offset of a PCI capability in config space, or 0
    at = self.cfg_read(0x34) & 0xFC
    while at and self.cfg_read(at) & 0xFF != cap: at = self.cfg_read(at) >> 8 & 0xFC
    return at

  def read(self, off, n): return _load(self.bar0 + off, n)
  def write(self, off, data): _store(self.bar0 + off, data)

  def alloc(self, size):
    """Allocate a locked 2 MiB hugepage, which is physically contiguous. Returns (memoryview, bus address). The page
    goes back to the kernel's pool once nothing refers to the memoryview. A Runtime keeps its ring and status in it, so
    the page lives as long as the Runtime does."""
    assert size <= 1 << 21, "Card.alloc gives at most one 2 MiB hugepage"
    MAP_HUGETLB, MAP_LOCKED, MAP_HUGE_2MB = 0x40000, 0x2000, 21 << 26  # 2 MiB pages, even if the host's default differs
    # A dropped Runtime can sit in a reference cycle, since the number tensors it caches refer back to it. It then
    # holds its page until gc collects it, so when no page is free we collect and try once more.
    for tries in (1, 0):
      try: buf = mmap.mmap(-1, 1 << 21, mmap.MAP_SHARED | mmap.MAP_ANONYMOUS | mmap.MAP_POPULATE | MAP_HUGETLB | MAP_LOCKED | MAP_HUGE_2MB); break
      except OSError:
        if not tries: raise MemoryError("no free 2 MiB hugepage for DMA memory. Try: echo 4 > /proc/sys/vm/nr_hugepages") from None
        gc.collect()
    with open("/proc/self/pagemap", "rb") as f:
      f.seek(ctypes.addressof(ctypes.c_char.from_buffer(buf)) // mmap.PAGESIZE * 8)
      pfn = int.from_bytes(f.read(8), "little") & ((1 << 55) - 1)
    assert pfn, "pagemap hides physical addresses without CAP_SYS_ADMIN"
    return memoryview(buf)[:size], pfn * mmap.PAGESIZE

  def tick(self): pass  # a card runs without being ticked
  def now(self): return time.monotonic()

# A BAR is mapped uncached, where libc's memcpy may copy a byte at a time (rep movsb), and each byte is its own PCIe
# transaction. _load and _store copy words, the widest that the address and the length allow, so a register's 4 bytes
# go in one access.
WORDS = {8: ("Q", ctypes.c_uint64), 4: ("I", ctypes.c_uint32)}
def _width(addr, n): return next((w for w in WORDS if addr % w == 0 and n % w == 0), None)
def _load(addr, n):
  if (w := _width(addr, n)) is None: return ctypes.string_at(addr, n)
  code, word = WORDS[w]
  return array.array(code, (word * (n // w)).from_address(addr)).tobytes()
def _store(addr, data):
  if (w := _width(addr, len(data))) is None: return ctypes.memmove(addr, bytes(data), len(data))
  code, word = WORDS[w]
  (word * (len(data) // w)).from_address(addr)[:] = memoryview(data).cast("B").cast(code)

def _write(path, s):
  with open(path, "w") as f: f.write(s)

def find_cards():
  """List the Blackholes on the PCI bus by their sysfs names (domain:bus:device.function)."""
  ids = lambda bdf: [int(open(f"{SYS}/devices/{bdf}/{f}").read(), 16) for f in ("vendor", "device")]
  if cards := [bdf for bdf in sorted(os.listdir(f"{SYS}/devices")) if ids(bdf) == [bh.VENDOR, bh.DEVICE]]: return cards
  raise RuntimeError("no Blackhole on the PCI bus. Set TTSIM=/path/to/libttsim.so to use the simulator")

def open_device(n=0):
  """Open chip n. With TTSIM=/path/to/libttsim.so that's chip n of the simulator, which has several in a multi-chip
  build. Otherwise it's the n-th Blackhole on the PCI bus."""
  return Sim(os.environ["TTSIM"], Sim.chips(os.environ["TTSIM"])[n]) if "TTSIM" in os.environ else Card(find_cards()[n])

def reset(bdf, timeout=10.0):
  """Reset the chip the way tt-kmd's ASIC reset does, for a card whose NoC or firmware hangs. A hung kernel doesn't
  need it, since the next Runtime resets the cores it uses.

  The parity bit of PCI_COMMAND marks the reset, and two config writes fire the interface timer of the PCIe controller.
  The reset is over when the bit clears, or when the card has left the bus and come back. Then a remove and rescan
  restores the config space as it was at boot. Nothing may have the card open. Untested on silicon (NOTES.md)."""
  if os.geteuid() != 0: raise PermissionError("reset needs root to write the card's config space and sysfs")
  path, deadline, gone = f"{SYS}/devices/{bdf}", time.monotonic() + timeout, False
  fd = os.open(f"{path}/config", os.O_RDWR | os.O_SYNC)
  command = int.from_bytes(os.pread(fd, 2, 0x04), "little")
  os.pwrite(fd, (command | bh.RESET_MARKER).to_bytes(2, "little"), 0x04)
  for off, val in bh.RESET_TIMER: os.pwrite(fd, val.to_bytes(4, "little"), off)
  os.close(fd)
  def back():  # whether the reset is over
    nonlocal gone
    if not os.path.exists(path): gone = True; return False
    try:
      with open(f"{path}/config", "rb") as f: f.seek(0x04); return gone or not int.from_bytes(f.read(2), "little") & bh.RESET_MARKER
    except OSError: return False  # its config space doesn't answer yet
  def until(cond, what):
    while not cond():
      if time.monotonic() > deadline: raise TimeoutError(f"{bdf} did not come back from {what} within {timeout}s")
      time.sleep(0.1)
  until(back, "its reset")
  _write(f"{path}/remove", "1")
  _write(f"{SYS}/rescan", "1")
  until(lambda: os.path.exists(f"{path}/config"), "a rescan")
