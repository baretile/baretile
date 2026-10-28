# PCIe access through ttsim (Sim) or Linux sysfs (Card). Both backends expose config space,
# BAR0 and memory that the device can reach by DMA. reset() resets a card whose NoC or firmware has hung.
import array, ctypes, gc, mmap, os, pathlib, time
from baretile import bh

SYS = "/sys/bus/pci"  # sysfs PCI device directory; tests replace it with a temporary directory
HUGEPAGES = "/sys/kernel/mm/hugepages/hugepages-2048kB"  # the kernel's pool of 2 MiB pages; tests replace it too
DRIVER = "tenstorrent"  # tt-kmd's PCI driver, which Card unbinds and release binds again

class Sim:
  """A Blackhole in libttsim.so. bdf selects the PCI device in builds with several chips. Sim instances share
  one library, clock and set of DMA buffers per process. counts is true when the library implements
  NIU_MST_RD_DATA_WORD_RECEIVED for profiling. ttsim-chaos implements it; released ttsim stops if you read
  it."""
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
      Sim.cbs = (cb(lambda bus, p, n: Sim.dma_copy(bus, p, n, False)), cb(lambda bus, p, n: Sim.dma_copy(bus, p, n, True)))
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
  fault = None  # an error in a DMA callback, which ctypes can't raise through C. tick raises it
  @staticmethod
  def dma_copy(bus, p, n, to_host):  # a device reads (to_host False) or writes host memory at bus
    for base, size, addr, _ in Sim.dma:
      if base <= bus and bus + n <= base + size:
        at = addr + bus - base
        ctypes.memmove(*((at, p) if to_host else (p, at)), n)
        return
    Sim.fault = Sim.fault or RuntimeError(f"device DMA to {bus:#x}+{n}, which is not host memory we allocated")

  TICK = 10  # chip clocks per poll, so waits and cycle counts round up to a multiple of TICK
  def tick(self):  # the simulator runs only when we tick it, and a tick advances all of its chips
    if Sim.fault: fault, Sim.fault = Sim.fault, None; raise fault
    self.lib.libttsim_clock(self.TICK)
    Sim.clocks += self.TICK
  def now(self): return Sim.clocks / bh.AICLK  # seconds of chip time by ttsim's clock, which doesn't match silicon

class Card:
  """Access a card through sysfs, using locked hugepages for DMA. Requires root and the IOMMU disabled or in
  passthrough mode, since the card uses physical addresses. Opening the card unbinds its kernel driver."""
  counts = True  # a card's NIUs count what their reads bring in (see Sim)
  def __init__(self, bdf):
    if os.geteuid() != 0: raise PermissionError("Card needs root for config space, the BARs, and physical addresses for DMA")
    self.sys = f"{SYS}/devices/{bdf}"
    if os.path.exists(f"{self.sys}/iommu_group/type") and (kind := pathlib.Path(self.sys, "iommu_group/type").read_text().strip()) != "identity":
      raise RuntimeError(f"the IOMMU translates DMA from {bdf} ({kind}), but Card gives the card physical addresses. Boot with iommu=pt")
    if os.path.exists(f"{self.sys}/driver"): _write(f"{self.sys}/driver/unbind", bdf)
    _write(f"{self.sys}/enable", "1")
    self.cfg = os.fdopen(os.open(f"{self.sys}/config", os.O_RDWR | os.O_SYNC), "r+b", buffering=0)
    self.cfg_write(0x04, self.cfg_read(0x04) | 0x6)  # memory decode, and bus mastering so the device can reach host memory
    # Set Max Read Request Size in the PCI Express capability to 4 KiB, so the device reads host memory in requests of
    # up to 4 KiB. kmd sets it up the same way.
    if at := self.capability(0x10):
      os.pwrite(self.cfg.fileno(), (self.cfg_read(at + 8) & 0x8FFF | 5 << 12).to_bytes(2, "little"), at + 8)
    # Map only BAR0, which has the TLB windows and registers that baretile uses. BAR4 may not be all there.
    fd = os.open(f"{self.sys}/resource0", os.O_RDWR | os.O_SYNC)
    try: self.map = mmap.mmap(fd, os.fstat(fd).st_size)
    finally: os.close(fd)
    self.bar0 = ctypes.addressof(ctypes.c_char.from_buffer(self.map))
    # Zero the strided multicast words of TLBs 0-31, as kmd leaves them, so a multicast skips no tiles of its rectangle.
    # An earlier user may have set them.
    for i in range(32): self.write(bh.TLB_STRIDED + 4 * i, bytes(4))
  def cfg_read(self, off): return int.from_bytes(os.pread(self.cfg.fileno(), 4, off), "little")
  def cfg_write(self, off, val): os.pwrite(self.cfg.fileno(), val.to_bytes(4, "little"), off)
  def capability(self, cap):  # the offset of a PCI capability in config space, or 0
    at, seen = self.cfg_read(0x34) & 0xFC, set()
    while at:
      if at < 0x40 or at in seen: raise RuntimeError(f"invalid PCI capability chain at {at:#x}")
      seen.add(at)
      word = self.cfg_read(at)
      if word & 0xFF == cap: return at
      at = word >> 8 & 0xFC
    return 0

  def read(self, off, n): return _load(self.bar0 + off, n)
  def write(self, off, data): _store(self.bar0 + off, data)

  def alloc(self, size):
    """Allocate a locked, physically contiguous 2 MiB hugepage and return (memoryview, bus address). The page
    returns to the kernel's pool when the memoryview is no longer referenced. A Runtime keeps one for its
    status, ring, launch times and transfer staging."""
    if size > 1 << 21: raise ValueError("Card.alloc gives at most one 2 MiB hugepage")
    MAP_HUGETLB, MAP_LOCKED, MAP_HUGE_2MB = 0x40000, 0x2000, 21 << 26  # 2 MiB pages, even if the host's default differs
    # Cached scalar tensors refer back to their Runtime, so dropping a Runtime can leave a reference cycle.
    # Collect those cycles before giving up: an unused Runtime may still hold a hugepage.
    for tries in (1, 0):
      try: buf = mmap.mmap(-1, 1 << 21, mmap.MAP_SHARED | mmap.MAP_ANONYMOUS | mmap.MAP_POPULATE | MAP_HUGETLB | MAP_LOCKED | MAP_HUGE_2MB); break
      except OSError:
        if not tries: raise MemoryError("no free 2 MiB hugepage for DMA memory. Try: echo 4 > /proc/sys/vm/nr_hugepages") from None
        gc.collect()
    with open("/proc/self/pagemap", "rb") as f:
      f.seek(ctypes.addressof(ctypes.c_char.from_buffer(buf)) // mmap.PAGESIZE * 8)
      pfn = int.from_bytes(f.read(8), "little") & ((1 << 55) - 1)
    if not pfn: raise PermissionError("pagemap hides physical addresses without CAP_SYS_ADMIN")
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
  ids = lambda bdf: [int(pathlib.Path(SYS, "devices", bdf, f).read_text(), 16) for f in ("vendor", "device")]
  if cards := [bdf for bdf in sorted(os.listdir(f"{SYS}/devices")) if ids(bdf) == [bh.VENDOR, bh.DEVICE]]: return cards
  raise RuntimeError("no Blackhole on the PCI bus. Set TTSIM=/path/to/libttsim.so to use the simulator")

def preflight(bdf):
  """Check what Card needs, from sysfs, without touching the card. Return [(level, message)], where level is "ok",
  "warn" or "fail". Card can't run after a fail."""
  dev, found = pathlib.Path(SYS, "devices", bdf), []
  read = lambda path: path.read_text().strip() if path.exists() else None
  found.append(("ok", "running as root") if os.geteuid() == 0 else
               ("fail", "Card needs root for config space, the BARs and physical addresses for DMA"))
  kind = read(dev / "iommu_group/type")
  found.append(("ok", "the IOMMU passes DMA addresses through" if kind else "no IOMMU group") if kind in (None, "identity") else
               ("fail", f"the IOMMU translates DMA addresses ({kind}), but Card gives the card physical ones. Boot with iommu=pt"))
  free = int(read(pathlib.Path(HUGEPAGES, "free_hugepages")) or 0)
  found.append(("ok", f"{free} free 2 MiB hugepages; a Runtime takes one") if free else
               ("fail", "no free 2 MiB hugepage for DMA memory. Try: echo 4 > /proc/sys/vm/nr_hugepages"))
  link = [read(dev / f) for f in ("current_link_speed", "current_link_width", "max_link_speed", "max_link_width")]
  if None in link: found.append(("warn", "sysfs doesn't say how fast the PCIe link is"))
  else: found.append(("ok", f"PCIe link at {link[0]} x{link[1]}") if link[:2] == link[2:] else
                     ("warn", f"the PCIe link runs at {link[0]} x{link[1]}, below the card's {link[2]} x{link[3]}. Check the slot"))
  if (dev / "driver").exists():
    name, fw = pathlib.Path(os.path.realpath(dev / "driver")).name, next(dev.glob("tenstorrent/*/tt_fw_bundle_ver"), None)
    found.append(("ok", f"{name} drives the card now. Card unbinds it, and `python3 -m baretile.chip release` binds {DRIVER} again"))
    if fw: found.append(("ok", f"firmware bundle {read(fw)}, as tt-kmd reads it"))
  else: found.append(("ok", "no driver has the card"))
  return found

def release(bdf):
  """Bind the card to tt-kmd's driver again, so TT-NN and tt-smi can use it after Card unbound it. Run it once no
  process has the card open through baretile. A card that a driver has is left alone. Return the name of the driver
  that has the card."""
  if os.geteuid() != 0: raise PermissionError("release needs root to bind a driver through sysfs")
  if os.path.exists(at := f"{SYS}/devices/{bdf}/driver"): return pathlib.Path(os.path.realpath(at)).name
  if not os.path.exists(f"{SYS}/drivers/{DRIVER}"): raise RuntimeError(f"no {DRIVER} driver is loaded. Load tt-kmd: modprobe {DRIVER}")
  _write(f"{SYS}/drivers/{DRIVER}/bind", bdf)
  return DRIVER

def open_device(n=0):
  """Open chip n. With TTSIM=/path/to/libttsim.so that's chip n of the simulator, which has several in a multi-chip
  build. Otherwise it's the n-th Blackhole on the PCI bus."""
  return Sim(os.environ["TTSIM"], Sim.chips(os.environ["TTSIM"])[n]) if "TTSIM" in os.environ else Card(find_cards()[n])

def reset(bdf, timeout=10.0):
  """Reset a card whose NoC or firmware has hung, following tt-kmd's ASIC reset. Close all processes using the
  card first. For a hung kernel, creating a new Runtime is enough to reset the cores. This chip reset is
  untested on hardware.

  PCI_COMMAND's parity bit marks the reset. Two config writes trigger the PCIe interface timer. Once the
  marker clears or the device leaves the bus and returns, remove and rescan it to restore config space. See
  docs/bring-up.md."""
  if os.geteuid() != 0: raise PermissionError("reset needs root to write the card's config space and sysfs")
  path, deadline, gone = f"{SYS}/devices/{bdf}", time.monotonic() + timeout, False
  with os.fdopen(os.open(f"{path}/config", os.O_RDWR | os.O_SYNC), "r+b", buffering=0) as cfg:
    command = int.from_bytes(os.pread(cfg.fileno(), 2, 0x04), "little")
    os.pwrite(cfg.fileno(), (command | bh.RESET_MARKER).to_bytes(2, "little"), 0x04)
    for off, val in bh.RESET_TIMER: os.pwrite(cfg.fileno(), val.to_bytes(4, "little"), off)
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
