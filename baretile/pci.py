# PCIe access through ttsim (Sim), tt-kmd's device file (Kmd) or Linux sysfs (Card). Each backend exposes config space,
# BAR0, TLB windows and memory that the device can reach by DMA. reset() resets a card whose NoC or firmware has hung.
import array, ctypes, fcntl, gc, hashlib, mmap, os, pathlib, struct, time, urllib.request, weakref
from baretile import bh

# what baretile builds and downloads
CACHE = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache"), "baretile")
# The ttsim release that baretile is tested with: its Blackhole library, the library's SHA-256, and where fetch_ttsim
# puts it. open_device uses it when TTSIM isn't set and there's no card.
TTSIM_URL = "https://github.com/tenstorrent/ttsim/releases/download/v1.11.5/libttsim_bh.so"
TTSIM_SHA256 = "9471430de59a0316facdf7716c2e526727a9d0ee397c204cc8eece85b6a52432"
TTSIM_PATH = CACHE / "ttsim" / "v1.11.5" / "libttsim_bh.so"

SYS = "/sys/bus/pci"  # sysfs PCI device directory; tests replace it with a temporary directory
DEV = "/dev/tenstorrent"  # tt-kmd's device files, one for each card it drives; tests replace it too
MODULE = "/sys/module/tenstorrent"  # tt-kmd's module, with its version; tests replace it too
HUGEPAGES = "/sys/kernel/mm/hugepages/hugepages-2048kB"  # the kernel's pool of 2 MiB pages; tests replace it too
DRIVER = "tenstorrent"  # tt-kmd's PCI driver, which Card unbinds and release binds again

def tlb_config(**fields):
  """A 2 MiB TLB's 96-bit config from its fields (bh.TLB_FIELDS)."""
  cfg = 0
  for name, val in fields.items():
    first, width = bh.TLB_FIELDS[name]
    if not (0 <= val < 1 << width): raise ValueError(f"TLB {name}={val:#x} does not fit in {width} bits")
    cfg |= val << first
  return cfg

def _write_tlb(dev, index, fields):  # write TLB index's config registers through BAR0, as Sim and Card do
  cfg = tlb_config(**fields)
  for i in range(3): dev.write(bh.TLB_CONFIG + 12 * index + 4 * i, (cfg >> 32 * i & 0xFFFFFFFF).to_bytes(4, "little"))

class Sim:
  """A Blackhole in libttsim.so. bdf selects the PCI device in builds with several chips. Sim instances share
  one library, clock and set of DMA buffers per process. counts is true when the library implements
  NIU_MST_RD_DATA_WORD_RECEIVED for profiling. ttsim-chaos implements it; released ttsim stops if you read
  it."""
  # dma: (bus address, size, host address, weak buffer reference). clocks: chip cycles so far
  lib, dma, clocks, counts = None, [], 0, False
  next_bus = 1 << 32  # where alloc puts the next page: never an address a released page had
  opened = weakref.WeakValueDictionary()  # bdf: its Sim. two Chips on one chip would hand out the same TLBs and DRAM
  def __init__(self, path, bdf=0):
    Sim.load(path)
    _claim(lambda: Sim.opened.setdefault(bdf, self) is self, f"ttsim chip {bdf:#x}")
    self.lib, self.bdf = Sim.lib, bdf
    # the TLB windows and registers that baretile uses
    self.bar0 = self.cfg_read(0x10) & ~0xF | self.cfg_read(0x14) << 32

  @staticmethod
  def load(path):  # loads the library once per process
    if Sim.lib is None:
      Sim.lib = lib = ctypes.CDLL(path)
      lib.libttsim_pci_config_rd32.restype = ctypes.c_uint32
      lib.libttsim_pci_config_rd32.argtypes = [ctypes.c_uint32] * 2
      lib.libttsim_pci_mem_rd_bytes.argtypes = lib.libttsim_pci_mem_wr_bytes.argtypes = [
        ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint32]
      lib.libttsim_clock.argtypes = [ctypes.c_uint32]
      # the devices reach host memory through these callbacks, so they must stay referenced as long as the library lives
      cb = ctypes.CFUNCTYPE(None, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint32)
      Sim.cbs = (cb(lambda bus, p, n: Sim.dma_copy(bus, p, n, False)),
                 cb(lambda bus, p, n: Sim.dma_copy(bus, p, n, True)))
      lib.libttsim_set_pci_dma_mem_callbacks(*Sim.cbs)
      lib.libttsim_init()
      Sim.counts = hasattr(lib, "libttsim_chaos")  # a marker that test/ttsim-chaos.patch adds

  @staticmethod
  def chips(path):  # PCI addresses of a build's chips. we scan the slots, since the chips may not be adjacent
    Sim.load(path)
    return [bdf for bdf in range(0, 1 << 16, 8)
            if Sim.lib.libttsim_pci_config_rd32(bdf, 0) == bh.DEVICE << 16 | bh.VENDOR]

  def cfg_read(self, off): return self.lib.libttsim_pci_config_rd32(self.bdf, off)

  def read(self, off, n):
    buf = ctypes.create_string_buffer(n)
    self.lib.libttsim_pci_mem_rd_bytes(self.bar0 + off, buf, n)
    return buf.raw
  def write(self, off, data): self.lib.libttsim_pci_mem_wr_bytes(self.bar0 + off, bytes(data), len(data))

  def alloc(self, size):
    """Allocate host memory that the device can reach. Returns (memoryview, bus address). Sim makes up the bus
    addresses, 4 GiB apart. The last memoryview releases the mapping, as it does on a card."""
    buf, bus = mmap.mmap(-1, size), Sim.next_bus
    Sim.next_bus += 1 << 32
    # a new list, since dma_copy may be going through the old one
    def release(_): Sim.dma = [entry for entry in Sim.dma if entry[0] != bus]
    Sim.dma.append((bus, size, ctypes.addressof(ctypes.c_char.from_buffer(buf)), weakref.ref(buf, release)))
    return memoryview(buf), bus
  fault = None  # an error in a DMA callback, which ctypes can't raise through C. tick raises it
  @staticmethod
  def dma_copy(bus, p, n, to_host):  # a device reads (to_host False) or writes host memory at bus
    for base, size, addr, ref in Sim.dma:
      buf = ref()  # keep the mapping alive through memmove
      if buf is not None and base <= bus and bus + n <= base + size:
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
  def tlb(self, index, fields): _write_tlb(self, index, fields)  # point TLB window index at fields (bh.TLB_FIELDS)

class Card:
  """Access a card through sysfs, using locked hugepages for DMA. Requires root and the IOMMU disabled or in
  passthrough mode, since the card uses physical addresses. Opening the card unbinds its kernel driver."""
  counts = True  # a card's NIUs count what their reads bring in (see Sim)
  def __init__(self, bdf):
    if os.geteuid() != 0:
      raise PermissionError("Card needs root for config space, the BARs, and physical addresses for DMA")
    self.sys = f"{SYS}/devices/{bdf}"
    if (os.path.exists(f"{self.sys}/iommu_group/type")
        and (kind := pathlib.Path(self.sys, "iommu_group/type").read_text().strip()) != "identity"):
      raise RuntimeError(f"the IOMMU translates DMA from {bdf} ({kind}), but Card gives the card physical addresses. "
                         f"Boot with iommu=pt")
    # The lock on config space makes this the card's only Card, in any process. It goes when cfg closes
    self.cfg = os.fdopen(os.open(f"{self.sys}/config", os.O_RDWR | os.O_SYNC), "r+b", buffering=0)
    _claim(lambda: _flock(self.cfg), bdf)
    if os.path.exists(f"{self.sys}/driver"): _write(f"{self.sys}/driver/unbind", bdf)
    _write(f"{self.sys}/enable", "1")
    # Memory decode, and bus mastering so the device can reach host memory. Two bytes, since writing Status back would
    # clear its error bits
    os.pwrite(self.cfg.fileno(), (self.cfg_read(0x04) & 0xFFFF | 0x6).to_bytes(2, "little"), 0x04)
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
    try: buf = mmap.mmap(-1, 1 << 21,
                         mmap.MAP_SHARED | mmap.MAP_ANONYMOUS | mmap.MAP_POPULATE | MAP_HUGETLB | MAP_LOCKED
                         | MAP_HUGE_2MB)
    except OSError: raise MemoryError("no free 2 MiB hugepage for DMA memory. Try: "
                                      "echo 4 > /proc/sys/vm/nr_hugepages") from None
    with open("/proc/self/pagemap", "rb") as f:
      f.seek(ctypes.addressof(ctypes.c_char.from_buffer(buf)) // mmap.PAGESIZE * 8)
      pfn = int.from_bytes(f.read(8), "little") & ((1 << 55) - 1)
    if not pfn: raise PermissionError("pagemap hides physical addresses without CAP_SYS_ADMIN")
    return memoryview(buf)[:size], pfn * mmap.PAGESIZE

  def tlb(self, index, fields): _write_tlb(self, index, fields)  # point TLB window index at fields (bh.TLB_FIELDS)
  def tick(self): pass  # a card runs without being ticked
  def now(self): return time.monotonic()

# tt-kmd's ioctls, _IO(0xFA, n) in its ioctl.h, and their arguments as (C type, [(member, struct code)]), with the
# padding the C compiler adds. PIN_PAGES asks for the extended output, which carries the NoC address. test_kmd_layouts
# checks each member's offset against the header.
KMD = {"GET_DEVICE_INFO": 0, "QUERY_MAPPINGS": 2, "PIN_PAGES": 7, "UNPIN_PAGES": 10, "ALLOCATE_TLB": 11, "FREE_TLB": 12,
       "CONFIGURE_TLB": 13, "SMC_MSG": 17}
KMD_ARGS = {
  "GET_DEVICE_INFO": ("struct tenstorrent_get_device_info",
                      [("in.output_size_bytes", "I"), ("out.output_size_bytes", "I"), ("out.vendor_id", "H"),
                       ("out.device_id", "H"), ("out.subsystem_vendor_id", "H"), ("out.subsystem_id", "H"),
                       ("out.bus_dev_fn", "H"), ("out.max_dma_buf_size_log2", "H"), ("out.pci_domain", "H"),
                       ("out.reserved", "H")]),
  "QUERY_MAPPINGS": ("struct tenstorrent_query_mappings", [("in.output_mapping_count", "I"), ("in.reserved", "I")]),
  "MAPPING": ("struct tenstorrent_mapping",
              [("mapping_id", "I"), ("reserved", "I"), ("mapping_base", "Q"), ("mapping_size", "Q")]),
  "PIN_PAGES": ("struct { struct tenstorrent_pin_pages_in in; struct tenstorrent_pin_pages_out_extended out; }",
                [("in.output_size_bytes", "I"), ("in.flags", "I"), ("in.virtual_address", "Q"), ("in.size", "Q"),
                 ("out.physical_address", "Q"), ("out.noc_address", "Q")]),
  "UNPIN_PAGES": ("struct tenstorrent_unpin_pages",
                  [("in.virtual_address", "Q"), ("in.size", "Q"), ("in.reserved", "Q")]),
  "ALLOCATE_TLB": ("struct tenstorrent_allocate_tlb",
                   [("in.size", "Q"), ("in.reserved", "Q"), ("out.id", "I"), ("out.reserved0", "I"),
                    ("out.mmap_offset_uc", "Q"), ("out.mmap_offset_wc", "Q"), ("out.reserved1", "Q")]),
  "FREE_TLB": ("struct tenstorrent_free_tlb", [("in.id", "I")]),
  "CONFIGURE_TLB": ("struct tenstorrent_configure_tlb",
                    [("in.id", "I"), ("in.reserved", "I"), ("in.config.addr", "Q"), ("in.config.x_end", "H"),
                     ("in.config.y_end", "H"), ("in.config.x_start", "H"), ("in.config.y_start", "H"),
                     ("in.config.noc", "B"), ("in.config.mcast", "B"), ("in.config.ordering", "B"),
                     ("in.config.linked", "B"), ("in.config.static_vc", "B"), ("in.config.reserved0", "3s"),
                     ("in.config.reserved1", "8s"), ("out.reserved", "Q")]),
  "SMC_MSG": ("struct tenstorrent_smc_msg",
              [("argsz", "I"), ("flags", "I"), ("queue_index", "I"), ("reserved0", "I"), ("message", "8I")]),
}
KMD_MAPPING_BAR0 = 1  # TENSTORRENT_MAPPING_RESOURCE0_UC: BAR0, uncached
KMD_PIN_CONTIGUOUS, KMD_PIN_NOC_DMA = 1, 2  # the pages are physically contiguous; the card reaches them through the NoC
KMD_SMC_POST, KMD_SMC_POLL = 1, 2

def _kmd_layout(name): return "<" + "".join(code for _, code in KMD_ARGS[name][1])

class Kmd:
  """A card through tt-kmd's device file, /dev/tenstorrent/N, which any user can open. kmd maps DMA memory for the
  card, through the IOMMU when it translates, and hands out TLB windows, so this needs neither root nor iommu=pt, and
  tt-smi can use the card at the same time. Opening the file without O_APPEND has kmd power the chip up for us.
  Needs tt-kmd 2.0 or later, and 2.11 for arc_msg."""
  counts = True  # see Sim
  def __init__(self, bdf, path):
    self.fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
    try:
      # one baretile opener at a time, as for Card. Other kmd clients don't take it
      _claim(lambda: _flock(self.fd), path)
      self.sys = f"{SYS}/devices/{bdf}"
      bar0 = next((m for m in self.mappings() if m["mapping_id"] == KMD_MAPPING_BAR0), None)
      if bar0 is None: raise RuntimeError(f"tt-kmd offers no BAR0 mapping for {bdf}")
      self.map = mmap.mmap(self.fd, bar0["mapping_size"], offset=bar0["mapping_base"])
    except BaseException:
      os.close(self.fd)
      raise
    self.bar0 = ctypes.addressof(ctypes.c_char.from_buffer(self.map))
    self.windows = {}  # baretile's TLB index: (kmd's id for it, its mapping, the mapping's address)
    # Closing the file gives back its lock, and kmd frees its TLBs and pins. The mappings hold their own references
    weakref.finalize(self, os.close, self.fd)

  def call(self, name, **values):
    """Run tt-kmd's ioctl name with its argument's members set from values, the rest zero, and return the
    members as kmd left them."""
    members, args = KMD_ARGS[name][1], []
    for member, code in members:
      value = values.get(member)
      if code.endswith("s"): args.append(value or b"")
      elif code[0].isdigit(): args.extend(value or (0,) * int(code[:-1]))
      else: args.append(value or 0)
    buf = bytearray(struct.pack(_kmd_layout(name), *args))
    fcntl.ioctl(self.fd, 0xFA << 8 | KMD[name], buf)
    out, at = {}, 0
    for member, code in members:
      field = struct.unpack_from("<" + code, buf, at)
      out[member], at = (field if code[0].isdigit() and not code.endswith("s") else field[0],
                         at + struct.calcsize("<" + code))
    return out

  def mappings(self):  # what this device file can map: [{mapping_id, mapping_base, mapping_size}]
    n, size = 8, struct.calcsize(_kmd_layout("MAPPING"))
    buf = bytearray(struct.pack(_kmd_layout("QUERY_MAPPINGS"), n, 0) + bytes(n * size))
    fcntl.ioctl(self.fd, 0xFA << 8 | KMD["QUERY_MAPPINGS"], buf)
    names = [m for m, _ in KMD_ARGS["MAPPING"][1]]
    found = [dict(zip(names, struct.unpack_from(_kmd_layout("MAPPING"), buf, 8 + i * size))) for i in range(n)]
    return [m for m in found if m["mapping_id"]]

  def cfg_read(self, off):  # the standard header of config space, which sysfs lets any user read
    with open(f"{self.sys}/config", "rb") as f: return int.from_bytes(os.pread(f.fileno(), 4, off), "little")

  # the address of BAR0 offset off: in a TLB window that kmd gave us, or among the registers above them
  def _at(self, off):
    if off >= bh.TLB_COUNT * bh.TLB_WINDOW: return self.bar0 + off
    return self.windows[off // bh.TLB_WINDOW][2] + off % bh.TLB_WINDOW
  def read(self, off, n): return _load(self._at(off), n)
  def write(self, off, data): _store(self._at(off), data)

  def tlb(self, index, fields):
    """Point TLB window index at fields (bh.TLB_FIELDS). kmd gives each index a window of its own on first use, and
    the window's BAR0 offset becomes index's. CONFIGURE_TLB has no field for the static VC class, which a multicast
    must set to bh.MCAST_VC, so we set those two bits in the window's registers after kmd writes them."""
    if index not in self.windows:
      got = self.call("ALLOCATE_TLB", **{"in.size": bh.TLB_WINDOW})
      window = mmap.mmap(self.fd, bh.TLB_WINDOW, offset=got["out.mmap_offset_uc"])
      self.windows[index] = got["out.id"], window, ctypes.addressof(ctypes.c_char.from_buffer(window))
    tlb_config(**fields)  # the same checks as Card's
    id = self.windows[index][0]
    config = {f"in.config.{k}": fields.get(k, 0) for k in ("x_end", "y_end", "x_start", "y_start", "mcast", "ordering",
                                                           "static_vc")}
    self.call("CONFIGURE_TLB", **{"in.id": id, "in.config.addr": fields.get("addr", 0) << 21}, **config)
    # skip no tiles of a multicast, as Card leaves them
    if id < 32: _store(self.bar0 + bh.TLB_STRIDED + 4 * id, bytes(4))
    if fields.get("vc_class"):
      first, width = bh.TLB_FIELDS["vc_class"]
      at = self.bar0 + bh.TLB_CONFIG + 12 * id + 8  # the high word holds bits 64 to 95
      word = (int.from_bytes(_load(at, 4), "little") & ~((1 << width) - 1 << first - 64)
              | fields["vc_class"] << first - 64)
      _store(at, word.to_bytes(4, "little"))

  def alloc(self, size):
    """Allocate 2 MiB that the card can reach and return (memoryview, the address the card uses on the NoC). kmd pins
    a hugepage, physically contiguous, or with the IOMMU translating, ordinary pages, which it maps contiguously. The
    pages go back when the memoryview is no longer referenced."""
    if size > 1 << 21: raise ValueError("Kmd.alloc gives at most 2 MiB")
    MAP_HUGETLB, MAP_HUGE_2MB = 0x40000, 21 << 26
    flags = mmap.MAP_SHARED | mmap.MAP_ANONYMOUS | mmap.MAP_POPULATE
    try: buf, pin = mmap.mmap(-1, 1 << 21, flags | MAP_HUGETLB | MAP_HUGE_2MB), KMD_PIN_CONTIGUOUS | KMD_PIN_NOC_DMA
    except OSError: buf, pin = mmap.mmap(-1, 1 << 21, flags), KMD_PIN_NOC_DMA  # contiguous only if the IOMMU translates
    va = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    try: got = self.call("PIN_PAGES",
                         **{"in.output_size_bytes": 16, "in.flags": pin, "in.virtual_address": va, "in.size": 1 << 21})
    except OSError as e:
      raise MemoryError(f"tt-kmd couldn't pin 2 MiB for DMA ({e.strerror}). Without an IOMMU translating, "
                        f"reserve a hugepage: "
                        "echo 4 > /proc/sys/vm/nr_hugepages") from None
    fd = self.fd
    def unpin():
      try: fcntl.ioctl(fd, 0xFA << 8 | KMD["UNPIN_PAGES"], struct.pack(_kmd_layout("UNPIN_PAGES"), va, 1 << 21, 0))
      except OSError: pass  # the device file closed first, and kmd unpinned everything then
    weakref.finalize(buf, unpin)
    return memoryview(buf)[:size], got["out.noc_address"]

  def message(self, header):
    """Send ARC a message through kmd, which shares ARC's queue among its clients, and return the 8-word response."""
    try: self.call("SMC_MSG", argsz=48, flags=KMD_SMC_POST, message=(header, *[0] * 7))
    except OSError as e: raise RuntimeError(f"ARC messages through tt-kmd need tt-kmd 2.11 "
                                            f"or later ({e.strerror})") from None
    deadline = time.monotonic() + 1.0
    while True:
      try: return self.call("SMC_MSG", argsz=48, flags=KMD_SMC_POLL)["message"]
      except BlockingIOError:
        if time.monotonic() > deadline:
          raise TimeoutError(f"ARC did not answer message {header & 0xFF:#x} within 1.0s") from None
        time.sleep(0.001)

  def tick(self): pass
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

def _flock(f):  # whether this process now holds f's exclusive lock
  try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
  except BlockingIOError: return False
  return True

def _claim(take, name):  # take() claims a device for one opener, or returns False if another opener has it
  if take(): return
  gc.collect()  # a Chip dropped in this process may still hold the device from a reference cycle
  if not take():
    raise RuntimeError(f"{name} is already open, in this process or another. Share its Chip, "
                       f"or close its runtimes and drop it")

def kmd_device(bdf):
  """tt-kmd's device file for card bdf, such as /dev/tenstorrent/0, or None when tt-kmd doesn't drive the card.
  sysfs names the device under the card's PCI device, with ! for the slash."""
  found = sorted(pathlib.Path(SYS, "devices", bdf).glob("tenstorrent/tenstorrent!*"))
  return f"{DEV}/{found[0].name.split('!', 1)[1]}" if found else None

def find_cards():
  """List the Blackholes on the PCI bus by their sysfs names (domain:bus:device.function)."""
  ids = lambda bdf: [int(pathlib.Path(SYS, "devices", bdf, f).read_text(), 16) for f in ("vendor", "device")]
  devices = sorted(os.listdir(f"{SYS}/devices")) if os.path.isdir(f"{SYS}/devices") else []
  if cards := [bdf for bdf in devices if ids(bdf) == [bh.VENDOR, bh.DEVICE]]: return cards
  raise RuntimeError("no Blackhole on the PCI bus. To use the simulator instead, run python3 -m baretile.chip ttsim, "
                     "or set TTSIM=/path/to/libttsim.so")

def fetch_ttsim():
  """Download the tested ttsim release's Blackhole library to TTSIM_PATH, unless it's there already, and return the
  path. The download must have TTSIM_SHA256."""
  if TTSIM_PATH.exists(): return TTSIM_PATH
  with urllib.request.urlopen(TTSIM_URL, timeout=60) as r: data = r.read()
  if (got := hashlib.sha256(data).hexdigest()) != TTSIM_SHA256:
    raise RuntimeError(f"{TTSIM_URL} has SHA-256 {got}, not {TTSIM_SHA256}")
  TTSIM_PATH.parent.mkdir(parents=True, exist_ok=True)
  part = TTSIM_PATH.with_name(TTSIM_PATH.name + ".part")
  part.write_bytes(data)
  os.replace(part, TTSIM_PATH)  # whole or not at all, if two processes fetch it at once
  return TTSIM_PATH

def simulator():
  """The libttsim.so that open_device opens: TTSIM's, else the one fetch_ttsim downloaded if there's no card. None
  means a card."""
  if "TTSIM" in os.environ: return os.environ["TTSIM"]
  if not TTSIM_PATH.exists(): return None
  try: find_cards()
  except RuntimeError: return str(TTSIM_PATH)
  return None

def preflight(bdf):
  """Check what opening card bdf needs, from sysfs, without touching the card. Return [(level, message)], where level
  is "ok", "warn" or "fail". open_device can't open the card after a fail. Through tt-kmd (Kmd) that's access to its
  device file, and the memory kmd pins for DMA; through sysfs (Card), root, the IOMMU passing addresses through, and
  a hugepage."""
  dev, found = pathlib.Path(SYS, "devices", bdf), []
  read = lambda path: path.read_text().strip() if path.exists() else None
  kind, free = read(dev / "iommu_group/type"), int(read(pathlib.Path(HUGEPAGES, "free_hugepages")) or 0)
  if path := kmd_device(bdf):
    version = read(pathlib.Path(MODULE, "version"))
    found.append(("ok", f"tt-kmd {version or '(version unknown)'} drives the card, "
                        f"and baretile opens it through {path}, without root"))
    found.append(("ok", f"this user can open {path}") if os.access(path, os.R_OK | os.W_OK) else
                 ("fail", f"this user can't open {path}. tt-kmd's udev rule makes it 0666: reinstall tt-kmd, "
                          f"or check /etc/udev/rules.d"))
    if kind not in (None, "identity"): found.append(("ok", "the IOMMU maps the memory that tt-kmd pins for DMA"))
    else: found.append(("ok", f"{free} free 2 MiB hugepages; a Runtime pins one") if free else
                       ("fail", "without an IOMMU translating, tt-kmd needs a hugepage to pin for DMA. Try: "
                                "echo 4 > /proc/sys/vm/nr_hugepages"))
  else:
    found.append(("ok", "running as root") if os.geteuid() == 0 else
                 ("fail", "tt-kmd doesn't drive the card, and opening it through sysfs needs root. Install tt-kmd, "
                          "or run as root"))
    found.append(("ok",
                  "the IOMMU passes DMA "
                  "addresses through" if kind else "no IOMMU group") if kind in (None, "identity") else
                 ("fail", f"the IOMMU translates DMA addresses ({kind}), but Card gives the card physical ones. "
                          f"Boot with iommu=pt"))
    found.append(("ok", f"{free} free 2 MiB hugepages; a Runtime takes one") if free else
                 ("fail", "no free 2 MiB hugepage for DMA memory. Try: echo 4 > /proc/sys/vm/nr_hugepages"))
  link = [read(dev / f) for f in ("current_link_speed", "current_link_width", "max_link_speed", "max_link_width")]
  if None in link: found.append(("warn", "sysfs doesn't say how fast the PCIe link is"))
  else: found.append(("ok", f"PCIe link at {link[0]} x{link[1]}") if link[:2] == link[2:] else
                     ("warn", f"the PCIe link runs at {link[0]} x{link[1]}, below the card's {link[2]} x{link[3]}. "
                              f"Check the slot"))
  if fw := next(dev.glob("tenstorrent/*/tt_fw_bundle_ver"), None):
    found.append(("ok", f"firmware bundle {read(fw)}, as tt-kmd reads it"))
  if path: return found
  if (dev / "driver").exists():
    found.append(("ok", f"{pathlib.Path(os.path.realpath(dev / 'driver')).name} has the card. Card unbinds it, "
                        f"and `python3 -m baretile.chip release` binds {DRIVER} again"))
  else: found.append(("ok", "no driver has the card"))
  return found

def release(bdf):
  """Bind the card to tt-kmd's driver again, so TT-NN and tt-smi can use it after Card unbound it. Run it once no
  process has the card open through baretile. A card that a driver has is left alone. Return the name of the driver
  that has the card."""
  if os.geteuid() != 0: raise PermissionError("release needs root to bind a driver through sysfs")
  if os.path.exists(at := f"{SYS}/devices/{bdf}/driver"): return pathlib.Path(os.path.realpath(at)).name
  if not os.path.exists(f"{SYS}/drivers/{DRIVER}"):
    raise RuntimeError(f"no {DRIVER} driver is loaded. Load tt-kmd: modprobe {DRIVER}")
  _write(f"{SYS}/drivers/{DRIVER}/bind", bdf)
  return DRIVER

def open_device(n=0):
  """Open chip n. With TTSIM=/path/to/libttsim.so that's chip n of the simulator, which has several in a multi-chip
  build. Otherwise it's the n-th Blackhole on the PCI bus, through tt-kmd when it drives the card (Kmd) and through
  sysfs as root when it doesn't (Card), or with no card there, the simulator that fetch_ttsim downloaded. Only one
  Chip at a time may have a device open."""
  sim = simulator()
  found = Sim.chips(sim) if sim else find_cards()
  if not 0 <= n < len(found): raise ValueError(f"open_device({n}): there are {len(found)} chips")
  if sim: return Sim(sim, found[n])
  return Kmd(found[n], path) if (path := kmd_device(found[n])) else Card(found[n])

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
      with open(f"{path}/config", "rb") as f:
        f.seek(0x04); return gone or not int.from_bytes(f.read(2), "little") & bh.RESET_MARKER
    except OSError: return False  # its config space doesn't answer yet
  def until(cond, what):
    while not cond():
      if time.monotonic() > deadline: raise TimeoutError(f"{bdf} did not come back from {what} within {timeout}s")
      time.sleep(0.1)
  until(back, "its reset")
  _write(f"{path}/remove", "1")
  _write(f"{SYS}/rescan", "1")
  until(lambda: os.path.exists(f"{path}/config"), "a rescan")
