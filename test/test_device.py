# Device and runtime tests (layers 0 and 1), plus import and layer checks. See conftest.py for simulator setup.
import ast, collections, ctypes, gc, hashlib, io, math, mmap, os, pathlib, random, re, shutil, struct, subprocess, sys, threading, time, tokenize
import weakref, pytest
from baretile import bh, pci, runtime
from baretile.chip import check
from baretile.hello import hello
from baretile.runtime import DATA, KERNELS, MAILBOX, MCAST_STATE, Runtime, compile_kernel

def test_layers():
  # Modules in baretile/ import only the standard library and the layers below their own. Layer 4, in baretile/models
  # and baretile/adapters, also uses NumPy, and the adapter tinygrad; tiktoken and PyTorch only where a command needs
  # them. The models don't use the adapters. A layer's tests, and the helpers in conftest, import nothing above that
  # layer. No file uses the private names of another file, whether it imports them or uses them as attributes that
  # only the other file defines.
  root = pathlib.Path(__file__).parent.parent
  order = ["bh", "pci", "chip", "runtime", "hello", "tensix", "kernels", "tensor", "bench", "probe", "viz"]  # layers and their commands
  trees = {p: ast.parse(p.read_text()) for d in ("baretile", "baretile/models", "baretile/adapters") for p in sorted((root / d).glob("*.py"))}
  assert sorted(order) == sorted(p.stem for p in trees if p.parent.name == "baretile"), "every module in baretile/ needs a layer"
  def imports(name, tree, below, also=()):  # each import is from the standard library, from also, or a baretile module in below
    for n in ast.walk(tree):
      if not isinstance(n, (ast.Import, ast.ImportFrom)): continue
      for m in [a.name for a in n.names] if isinstance(n, ast.Import) else [f"{n.module}.{a.name}" for a in n.names]:
        top, _, sub = m.partition(".")
        assert sub.split(".")[0] in below if top == "baretile" else top in sys.stdlib_module_names or top in also, f"{name} imports {m}"
  for p, tree in trees.items():
    if p.parent.name == "baretile": imports(p.name, tree, order[:order.index(p.stem)])
    if p.parent.name == "models": imports(p.name, tree, order + ["models"], ("numpy", "tiktoken", "torch"))
    if p.parent.name == "adapters": imports(p.name, tree, order + ["models"], ("numpy", "tinygrad"))
  for name, top in (("conftest", "hello"), ("test_device", "hello"), ("test_kernels", "viz")):
    imports(name, ast.parse((root / "test" / f"{name}.py").read_text()), order[:order.index(top) + 1], ("numpy", "pytest", "conftest"))
  ours, private = {p.stem for p in trees} | {"baretile"}, lambda name: name.startswith("_") and not name.endswith("__")
  defined, used = {p: set() for p in trees}, {p: set() for p in trees}
  for p, tree in trees.items():
    for n in ast.walk(tree):
      if isinstance(n, (ast.FunctionDef, ast.ClassDef)): defined[p].add(n.name)
      elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store): defined[p].add(n.id)
      elif isinstance(n, ast.Attribute): (defined[p] if isinstance(n.ctx, ast.Store) else used[p]).add(n.attr)
      elif isinstance(n, ast.ImportFrom) and n.module.split(".")[0] in ours: used[p] |= {a.name for a in n.names}
  for p in trees:
    theirs = {name for q in trees if q != p for name in defined[q] if private(name)} - defined[p]
    assert not used[p] & theirs, f"{p.name} uses another file's private {sorted(used[p] & theirs)}"

CORE, BUDGET, WIDTH = ("bh", "pci", "chip", "runtime", "tensix", "kernels", "tensor"), 4500, 120  # layers 0 to 3

def code_lines(text, width=WIDTH):
  """Count the lines of code in Python source: not its comments, blank lines or docstrings. A line counts once for
  each width characters it takes, so making lines longer doesn't make the count smaller. C in a string is code."""
  docs = {i for n in ast.walk(ast.parse(text)) if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(n) is not None
          for i in range(n.body[0].lineno, n.body[0].end_lineno + 1)}
  comments = {t.start[0]: t.start[1] for t in tokenize.generate_tokens(io.StringIO(text).readline) if t.type == tokenize.COMMENT}
  return sum(math.ceil(len(code) / width) for i, line in enumerate(text.splitlines(), 1)
             if i not in docs and (code := line[:comments.get(i, len(line))].rstrip()))

def test_size():
  # The core, from the PCIe driver to Tensor, stays under its budget, and the README says what that is. Long lines and
  # comments can't move the count: lines count by width, and comments and docstrings don't count.
  root = pathlib.Path(__file__).parent.parent
  sample = ["x = 1  # a comment", "", "def f():", "  'what f does'", "  s = '#define ONE 1'  # C in a string is code",
            "  return " + " + ".join("1" * 60)]
  assert code_lines("\n".join(sample)) == 1 + 1 + 1 + 3, "the return takes 246 columns, so three lines"
  lines = sum(code_lines((root / "baretile" / f"{m}.py").read_text()) for m in CORE)
  assert lines <= BUDGET, f"the core is {lines:,} lines at {WIDTH} columns, over its budget of {BUDGET:,}. Remove or reuse something first"
  assert re.search(rf"under {BUDGET:,} lines of code at {WIDTH} columns", (root / "README.md").read_text()), "the README states the budget"

def test_open_device(chip):
  # A device has one opener at a time, since two Chips on one chip would hand out the same TLBs and DRAM.
  with pytest.raises(RuntimeError, match="already open"): pci.open_device()
  with pytest.raises(ValueError, match="chips"): pci.open_device(64)

def test_config_space(chip):
  assert chip.pci.cfg_read(0x00) == bh.VENDOR | bh.DEVICE << 16
  assert chip.pci.cfg_read(0x04) & 0x2, "memory decode is off"
  assert chip.pci.cfg_read(0x10) & ~0xF | chip.pci.cfg_read(0x14) << 32, "BAR0 has no address"

def test_arc(chip):
  # ARC messages and telemetry. A client that died can leave a response unread in the queue, and the next message must
  # still get its own answer. We fake that by overwriting the slot of TEST's response and marking it unread.
  chip.arc_msg(0x90)  # TEST
  (x, y), entry = bh.ARC, bh.ARC_QUEUE_ENTRY
  qcb = chip.read32(x, y, bh.ARC_QUEUES)
  base, n = chip.read32(x, y, qcb), chip.read32(x, y, qcb + 4) & 0xFF
  q = base + bh.APP_QUEUE * (entry + 2 * n * entry)
  rptr = (chip.read32(x, y, q + bh.ARC_QUEUE_PTRS[1]) - 1) % (2 * n)
  chip.write32(x, y, q + entry * (1 + n + rptr % n), 0xABCD0000)
  chip.write32(x, y, q + bh.ARC_QUEUE_PTRS[1], rptr)  # mark it unread
  assert chip.arc_msg(0x90)[0] != 0xABCD0000, "got the stale response left in the queue"
  t = chip.telemetry()
  assert len(chip.tensix) in (120, 140) and t[bh.TAG_AICLK] > 0, "expected 140 Tensix, or 120 from firmware 19.5 on, and a nonzero AICLK"

def test_dma(chip):
  # The PCIe tile forwards NoC addresses below 2**58 to the host as bus addresses. Point a TLB there and the device
  # reads and writes host memory.
  buf, bus = chip.pci.alloc(4096)
  buf[:16] = data = random.randbytes(16)
  assert chip.read(*bh.PCIE, bus, 16) == data
  chip.write(*bh.PCIE, bus + 1024, data[::-1])
  assert bytes(buf[1024:1040]) == data[::-1]

def test_sim_dma_fault(chip):
  # ttsim reaches host memory through a C callback, which can't pass a Python error on. A DMA outside the memory
  # baretile allocated fails the next tick instead, once.
  if not isinstance(chip.pci, pci.Sim): pytest.skip("ttsim's DMA callbacks")
  chip.write(*bh.PCIE, 0x7 << 40, bytes(16))
  with pytest.raises(RuntimeError, match="not host memory"): chip.pci.tick()
  chip.pci.tick()

def test_l1(chip):
  # Write both targets before reading either back, so targets that alias each other fail. Then map a TLB onto the first
  # block and read through it. Mapping the same block again gives back the same TLB, so later runtimes don't take a
  # new TLB for their doorbell.
  addr, data = bh.L1_SIZE - 65536, {xy: random.randbytes(65536) for xy in (chip.tensix[0], chip.tensix[-1])}
  for (x, y), d in data.items(): chip.write(x, y, addr, d)
  for (x, y), d in data.items(): assert chip.read(x, y, addr, len(d)) == d, f"Tensix {x},{y}"
  (x, y), d = next(iter(data.items()))
  at = chip.map(x, y, addr)
  assert chip.map(x, y, addr) == at and chip.pci.read(at, 64) == d[:64]
  # Explicitly remapping the movable TLB must invalidate its cached target.
  assert chip.read(x, y, addr, 64) == d[:64]
  chip.map(*chip.tensix[-1], addr, tlb=0)
  assert chip.read(x, y, addr, 64) == d[:64]
  with pytest.raises(ValueError, match="TLB index"): chip.map(x, y, addr, tlb=-1)
  with pytest.raises(ValueError, match="nonnegative"): chip.read(x, y, addr, -4)

def test_dram(chip):
  # Every bank through each of its three ports: across two 2 MiB windows, and at the top of what the allocator uses
  for addr in (bh.TLB_WINDOW - 2048, bh.DRAM_BANK_SIZE - 4096):
    data = [random.randbytes(4096) for _ in bh.DRAM]
    for (x, y), d in zip(bh.DRAM, data): chip.write(x, y, addr, d)
    for bank, ((x, y), d) in enumerate(zip(bh.DRAM, data)):
      for port in range(3): assert chip.read(x, y + port, addr, len(d)) == d, f"bank {bank} port {port} at {addr:#x}"

def test_raw_places(chip):
  # Each Tensix is where the translation tables of the PCIe tile put it, as its NIU reports. The DRAM ports are raw
  # columns 0 and 9, one per row. NoC0's links run right and down, and wrap around.
  for x, y in chip.tensix:
    niu = lambda reg: chip.read32(x, y, bh.NIU + bh.NIU_REGS[reg]) & 0xFFF
    assert (niu("NODE_ID"), niu("ID_LOGICAL")) == (bh.noc_xy(*chip.raw(x, y)), bh.noc_xy(x, y)), f"Tensix {x},{y}"
  assert sorted(chip.raw(x, y + p) for x, y in bh.DRAM for p in range(3)) == [(x, y) for x in (0, 9) for y in range(12)]
  assert [bh.hops((1, 2), b) for b in ((2, 2), (0, 2), (1, 1), (0, 1))] == [1, 16, 11, 27]

def test_cache_prune(tmp_path, monkeypatch):
  # The first compile removes cached images unused for a week. Reusing an image refreshes its timestamp.
  # Preserve subdirectories, which may contain caches from older checkouts.
  import baretile.runtime as runtime
  monkeypatch.setattr(runtime, "CACHE", tmp_path)
  old, new, folder = tmp_path / "old", tmp_path / "new", tmp_path / "folder"
  old.write_bytes(b"x"), new.write_bytes(b"y"), folder.mkdir()
  for f in (old, folder): os.utime(f, (0, time.time() - 8 * 86400))
  runtime._prune.__wrapped__()
  assert not old.exists() and new.exists() and folder.exists()

def test_compile_concurrent(tmp_path, monkeypatch):
  # Both threads finish compiling before either publishes the same cache entry. Temporary files must not collide.
  import concurrent.futures
  import baretile.runtime as runtime
  monkeypatch.setattr(runtime, "CACHE", tmp_path)
  barrier, replace = threading.Barrier(2, timeout=10), os.replace
  def publish(src, dst):
    barrier.wait()
    return replace(src, dst)
  monkeypatch.setattr(os, "replace", publish)
  with concurrent.futures.ThreadPoolExecutor(2) as pool:
    a, b = pool.map(runtime.compile_rv32, ["void kmain(volatile unsigned *a) { a[0] = 42; }"] * 2)
  assert a == b and len(list(tmp_path.iterdir())) == 1 and next(tmp_path.iterdir()).read_bytes() == a

def test_kernel_pointer_initializers_are_rejected(tmp_path, monkeypatch):
  import baretile.runtime as runtime
  monkeypatch.setattr(runtime, "CACHE", tmp_path)
  data = """
static const unsigned table[] = {111, 222};
static const unsigned * const pointers[] = {table, table + 1};
void kmain(volatile unsigned *a) { a[0] = *pointers[a[2] & 1]; }
"""
  function = """
static unsigned first(unsigned x) { return x + 111; }
static unsigned second(unsigned x) { return x + 222; }
static unsigned (* const pointers[])(unsigned) __attribute__((used)) = {first, second};
void kmain(volatile unsigned *a) { a[0] = pointers[a[2] & 1](a[3]); }
"""
  for src in (data, function):
    with pytest.raises(ValueError, match="position-independent"): compile_kernel(src)
  assert not list(tmp_path.iterdir()), "unsupported code must not enter the image cache"

def test_kernel_const_offsets_at_different_addresses(chip):
  kernel = compile_kernel("""
static const unsigned table[] = {111, 222};
static const unsigned offsets[] = {1, 0};
void kmain(volatile unsigned *a) { a[0] = table[offsets[a[2] & 1]]; }
""")
  with Runtime(chip, end=chip.tensix[1]) as rt:
    for padding in (16, 4096):
      rt.evict()
      rt.load(bytes(padding))
      assert rt.load(kernel) == KERNELS + padding
      for index, expected in enumerate((222, 111)):
        rt.launch(kernel, (index,))
        rt.wait()
        assert chip.read32(*rt.cores[0], rt.args()) == expected

def test_card_copies():
  # A card's BARs are read and written a word at a time, since libc's memcpy may go byte by byte on uncached memory. A
  # word is 8 bytes, or 4 where the address and length don't allow 8, and a byte where neither fits. This test runs on
  # ordinary memory.
  buf = (ctypes.c_char * 256)()
  for off, n in ((0, 64), (8, 16), (4, 12), (4, 4), (1, 7), (0, 256)):
    data = random.randbytes(n)
    pci._store(ctypes.addressof(buf) + off, data)
    assert pci._load(ctypes.addressof(buf) + off, n) == data == bytes(buf[off:off + n]), (off, n)
  assert [pci._width(a, n) for a, n in ((0, 64), (4, 12), (1, 7))] == [8, 4, None]

def test_card_wants_root():
  # pci.Card, pci.reset and pci.release check for root before they touch the card, so this test needs no card.
  # Without root they raise PermissionError before any sysfs write can fail.
  if os.geteuid() == 0: pytest.skip("as root, Card would go on to open the device")
  with pytest.raises(PermissionError, match="root"): pci.Card("0000:00:00.0")
  with pytest.raises(PermissionError, match="root"): pci.reset("0000:00:00.0")
  with pytest.raises(PermissionError, match="root"): pci.release("0000:00:00.0")

def test_capability_chain():
  card = object.__new__(pci.Card)
  config = {0x34: 0x40, 0x40: 0x4801, 0x48: 0x10}
  card.cfg_read = config.__getitem__
  assert card.capability(0x10) == 0x48 and card.capability(0xFF) == 0
  config[0x48] = 0x4002
  with pytest.raises(RuntimeError, match="capability chain"): card.capability(0x10)
  config[0x34] = 0x20
  with pytest.raises(RuntimeError, match="capability chain"): card.capability(0x10)

def test_card_sysfs(tmp_path, monkeypatch):
  # This runs pci.Card and reset against a fake sysfs, so no card is needed. Card locks config space, unbinds the
  # driver, enables the device, turns on memory decode and bus mastering, and sets Max Read Request Size to 4 KiB. It maps BAR0, clears the
  # strided multicast of TLBs 0-31 there, and refuses an IOMMU that translates. find_cards finds the card by its ids.
  # reset fires the interface timer, waits for the card to clear the marker, and has the kernel enumerate it again.
  bdf = "0000:01:00.0"
  dev = tmp_path / "devices" / bdf
  for d in ("driver", "iommu_group"): (dev / d).mkdir(parents=True)
  for name, text in (("vendor", f"{bh.VENDOR:#x}\n"), ("device", f"{bh.DEVICE:#x}\n"), ("enable", "0"), ("driver/unbind", ""),
                     ("iommu_group/type", "identity\n"), ("remove", "")): (dev / name).write_text(text)
  config = bytearray(4096)  # PCIe capability at 0x40. Device control at 0x48 asks for reads of 512 bytes
  config[6:8] = (0xF900).to_bytes(2, "little")  # sticky PCI error bits: writing a 1 clears that bit
  config[0x34], config[0x40], config[0x48:0x4A] = 0x40, 0x10, (0x2810).to_bytes(2, "little")
  (dev / "config").write_bytes(config)
  pwrite = os.pwrite
  def config_write(fd, data, off):
    if off == 4 and len(data) == 4:
      status = int.from_bytes(os.pread(fd, 2, 6), "little") & ~int.from_bytes(data[2:], "little")
      data = data[:2] + status.to_bytes(2, "little")
    return pwrite(fd, data, off)
  monkeypatch.setattr(pci.os, "pwrite", config_write)
  # BAR0, with TLBs left set by an earlier user
  with open(dev / "resource0", "wb") as f: f.truncate(1 << 29), f.seek(bh.TLB_STRIDED), f.write(b"\xff" * 128)
  monkeypatch.setattr(pci, "SYS", str(tmp_path))
  monkeypatch.setattr(pci.os, "geteuid", lambda: 0)
  assert pci.find_cards() == [bdf]
  card, data = pci.Card(bdf), random.randbytes(64)
  with pytest.raises(RuntimeError, match="already open"): pci.Card(bdf)  # config space stays locked while card is open
  card.write(0x1000, data)
  with open(dev / "resource0", "rb") as f: written = f.read(0x1040)[0x1000:]
  assert (dev / "driver/unbind").read_text() == bdf and (dev / "enable").read_text() == "1" and written == data
  assert card.cfg_read(0x04) & 6 == 6 and card.cfg_read(0x48) & 0xFFFF == 0x5810 and card.read(bh.TLB_STRIDED, 128) == bytes(128)
  assert card.cfg_read(0x04) >> 16 == 0xF900, "enabling memory decode and bus mastering preserves PCI error status"
  (dev / "iommu_group/type").write_text("DMA\n")
  with pytest.raises(RuntimeError, match="IOMMU"): pci.Card(bdf)
  def back():  # the card clears the marker when its reset is done
    for _ in range(500):
      if (command := (dev / "config").read_bytes()[4]) & bh.RESET_MARKER:
        with open(dev / "config", "r+b") as f: f.seek(4), f.write(bytes([command & ~bh.RESET_MARKER]))
        return
      time.sleep(0.01)
  threading.Thread(target=back, daemon=True).start()
  (tmp_path / "rescan").write_text("")
  pci.reset(bdf, timeout=5)
  cfg = (dev / "config").read_bytes()
  assert [int.from_bytes(cfg[off:off + 4], "little") for off, _ in bh.RESET_TIMER] == [v for _, v in bh.RESET_TIMER]
  assert (dev / "remove").read_text() == "1" and (tmp_path / "rescan").read_text() == "1"

def test_preflight(tmp_path, monkeypatch):
  # preflight reads what opening a card needs from a fake sysfs. With tt-kmd driving it, that's its device file, which
  # this user can open, and memory to pin for DMA: an IOMMU that translates, or a hugepage. Without tt-kmd, it's root,
  # an IOMMU that passes addresses through, and a free hugepage. Both check the PCIe link, and each problem shows.
  bdf = "0000:01:00.0"
  dev, pages, module, files = tmp_path / "devices" / bdf, tmp_path / "hugepages", tmp_path / "module", tmp_path / "dev"
  for d in (dev / "iommu_group", dev / "tenstorrent" / "tenstorrent!0", pages, module, files): d.mkdir(parents=True)
  for name, text in (("iommu_group/type", "identity\n"), ("current_link_speed", "32.0 GT/s PCIe\n"), ("current_link_width", "16\n"),
                     ("max_link_speed", "32.0 GT/s PCIe\n"), ("max_link_width", "16\n"),
                     ("tenstorrent/tenstorrent!0/tt_fw_bundle_ver", "18.12.0.0\n")):
    (dev / name).write_text(text)
  (pages / "free_hugepages").write_text("4\n")
  (module / "version").write_text("2.11.0\n")
  (files / "0").write_text("")
  for name, value in (("SYS", str(tmp_path)), ("HUGEPAGES", str(pages)), ("MODULE", str(module)), ("DEV", str(files))):
    monkeypatch.setattr(pci, name, value)
  monkeypatch.setattr(pci.os, "geteuid", lambda: 1000)
  assert pci.kmd_device(bdf) == f"{files}/0"
  found = pci.preflight(bdf)
  assert {level for level, _ in found} == {"ok"}, found
  text = " ".join(message for _, message in found)
  assert all(s in text for s in ("tt-kmd 2.11.0", "without root", "4 free", "32.0 GT/s PCIe x16", "18.12.0.0")), text
  (pages / "free_hugepages").write_text("0\n")
  assert [level for level, _ in pci.preflight(bdf)].count("fail") == 1, "no IOMMU translating and no hugepage to pin"
  (dev / "iommu_group/type").write_text("DMA\n")
  monkeypatch.setattr(pci.os, "access", lambda path, mode: False)
  levels = collections.Counter(level for level, _ in pci.preflight(bdf))
  assert levels == {"fail": 1, "ok": 4}, levels  # the IOMMU maps the pinned memory, but the device file is out of reach
  # Without tt-kmd, through sysfs as root
  monkeypatch.undo()
  for name, value in (("SYS", str(tmp_path)), ("HUGEPAGES", str(pages))): monkeypatch.setattr(pci, name, value)
  shutil.rmtree(dev / "tenstorrent")
  (tmp_path / "drivers" / "vfio-pci").mkdir(parents=True)
  (dev / "driver").symlink_to(tmp_path / "drivers" / "vfio-pci")
  (dev / "iommu_group/type").write_text("identity\n")
  (pages / "free_hugepages").write_text("4\n")
  monkeypatch.setattr(pci.os, "geteuid", lambda: 0)
  assert pci.kmd_device(bdf) is None
  found = pci.preflight(bdf)
  assert {level for level, _ in found} == {"ok"}, found
  assert "vfio-pci has the card" in " ".join(message for _, message in found)
  (dev / "current_link_width").write_text("8\n")
  (dev / "iommu_group/type").write_text("DMA\n")
  (pages / "free_hugepages").write_text("0\n")
  monkeypatch.setattr(pci.os, "geteuid", lambda: 1000)
  levels = collections.Counter(level for level, _ in pci.preflight(bdf))
  assert levels == {"fail": 3, "warn": 1, "ok": 1}, levels

class FakeKmd:
  """tt-kmd as Kmd sees it: a file for the device, BAR0 at its start and TLB windows after it, and ioctls that answer as
  kmd's do. CONFIGURE_TLB writes the registers as kmd's blackhole.c does, without the static VC class."""
  def __init__(self, tmp_path):
    self.fd = os.memfd_create("tenstorrent")
    os.ftruncate(self.fd, 1 << 30)  # BAR0's 512 MiB, then room for the windows; sparse, so nothing is written yet
    self.path, self.tlbs, self.unpinned, self.polls = f"/proc/self/fd/{self.fd}", 0, [], 0
  def ioctl(self, fd, request, buf):
    name = {0xFA << 8 | n: name for name, n in pci.KMD.items()}[request]
    if name == "QUERY_MAPPINGS": struct.pack_into("<IIQQ", buf, 8, pci.KMD_MAPPING_BAR0, 0, 0, 512 << 20)
    elif name == "ALLOCATE_TLB":
      self.tlbs += 1
      struct.pack_into("<IIQQQ", buf, 16, self.tlbs, 0, (512 << 20) + self.tlbs * bh.TLB_WINDOW, 0, 0)
    elif name == "CONFIGURE_TLB":
      id, _, addr, x_end, y_end, x_start, y_start, noc, mcast, ordering, linked, static_vc = struct.unpack_from("<IIQ4H5B", buf)
      reg = (addr >> 21 | x_end << 43 | y_end << 49 | x_start << 55 | y_start << 61 | noc << 67 | mcast << 69 | ordering << 70
             | linked << 72 | static_vc << 73)
      os.pwrite(self.fd, reg.to_bytes(12, "little"), bh.TLB_CONFIG + 12 * id)
    elif name == "PIN_PAGES": struct.pack_into("<QQ", buf, 24, 0x1234000, 1 << 60 | 0x1234000)
    elif name == "UNPIN_PAGES": self.unpinned.append(struct.unpack_from("<QQ", buf))
    elif name == "SMC_MSG" and struct.unpack_from("<I", buf, 4)[0] == pci.KMD_SMC_POLL:
      self.polls += 1
      if self.polls == 1: raise BlockingIOError(11, "the message is still pending")
      struct.pack_into("<8I", buf, 16, *range(8))
    return 0

def test_kmd(tmp_path, monkeypatch):
  # Kmd maps BAR0 and the TLB windows kmd gives it, and reads and writes go to the right one. A window kmd configures has
  # the 96 bits that Card writes, including the static VC class that CONFIGURE_TLB leaves out. Pinned DMA memory goes
  # back when its memoryview does, and an ARC message waits for kmd's answer. One Kmd has a device at a time, and a
  # dropped one lets the next one open it.
  kmd = FakeKmd(tmp_path)
  monkeypatch.setattr(pci.fcntl, "ioctl", kmd.ioctl)
  card = pci.Kmd("0000:01:00.0", kmd.path)
  with pytest.raises(RuntimeError, match="already open"): pci.Kmd("0000:01:00.0", kmd.path)
  del card
  card = pci.Kmd("0000:01:00.0", kmd.path)  # the dropped one closed its file, and its lock with it
  read_reg = lambda id: int.from_bytes(os.pread(kmd.fd, 12, bh.TLB_CONFIG + 12 * id), "little")
  unicast = dict(addr=5, x_end=3, y_end=4, ordering=bh.STRICT)
  multicast = unicast | dict(x_start=1, y_start=2, mcast=1, static_vc=1, vc_class=bh.MCAST_VC)
  card.tlb(7, unicast)
  assert read_reg(1) == pci.tlb_config(**unicast)
  card.tlb(9, multicast)
  assert read_reg(2) == pci.tlb_config(**multicast), "the class bits that CONFIGURE_TLB leaves out"
  card.tlb(7, multicast)
  assert kmd.tlbs == 2 and read_reg(1) == pci.tlb_config(**multicast), "an index keeps its window"
  card.write(9 * bh.TLB_WINDOW + 16, b"window 9")
  assert os.pread(kmd.fd, 8, (512 << 20) + 2 * bh.TLB_WINDOW + 16) == b"window 9" and card.read(9 * bh.TLB_WINDOW + 16, 8) == b"window 9"
  assert card.read(bh.TLB_CONFIG + 24, 12) == pci.tlb_config(**multicast).to_bytes(12, "little"), "registers above the windows"
  with pytest.raises(ValueError, match="does not fit"): card.tlb(7, unicast | dict(x_end=64))
  buf, address = card.alloc(64)
  assert len(buf) == 64 and address == 1 << 60 | 0x1234000, "the NoC address kmd returns"
  del buf
  gc.collect()
  assert len(kmd.unpinned) == 1 and kmd.unpinned[0][1] == 1 << 21
  assert card.message(0x90) == tuple(range(8)) and kmd.polls == 2

@pytest.mark.skipif("KMD_IOCTL" not in os.environ, reason="set KMD_IOCTL to tt-kmd's ioctl.h")
def test_kmd_layouts(tmp_path):
  # Each ioctl number and each member's offset in pci.KMD_ARGS match tt-kmd's ioctl.h, compiled by clang for this host,
  # and each argument has the size that kmd expects.
  lines = ["#include <stddef.h>", "#include <stdio.h>", f'#include "{os.environ["KMD_IOCTL"]}"', "int main(void) {"]
  for name, (ctype, members) in pci.KMD_ARGS.items():
    lines.append(f"  {{ typedef {ctype} T; printf(\"{name} %zu\\n\", sizeof(T));")
    lines += [f'    printf("{name}.{member} %zu\\n", offsetof(T, {member}));' for member, _ in members]
    lines.append("  }")
  lines += [f'  printf("{name} = %u\\n", TENSTORRENT_IOCTL_{name});' for name in pci.KMD] + ["}"]
  (tmp_path / "layouts.c").write_text("\n".join(lines))
  subprocess.run(["clang", "-o", tmp_path / "layouts", tmp_path / "layouts.c"], check=True)
  got = subprocess.run([tmp_path / "layouts"], capture_output=True, text=True, check=True).stdout.splitlines()
  want = []
  for name, (_, members) in pci.KMD_ARGS.items():
    want.append(f"{name} {struct.calcsize(''.join(['<'] + [code for _, code in members]))}")
    want += [f"{name}.{member} {struct.calcsize(''.join(['<'] + [code for _, code in members[:i]]))}" for i, (member, _) in enumerate(members)]
  want += [f"{name} = {0xFA << 8 | n}" for name, n in pci.KMD.items()]
  assert got == want, [(g, w) for g, w in zip(got, want) if g != w]

def test_fetch_ttsim(tmp_path, monkeypatch):
  # fetch_ttsim downloads the simulator once, and only with the expected SHA-256. A server in a thread stands in for
  # GitHub.
  import http.server
  library, gets = os.urandom(1024), []
  class Release(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
      gets.append(self.path)
      self.send_response(200), self.send_header("Content-Length", str(len(library))), self.end_headers()
      self.wfile.write(library)
    def log_message(self, *_): pass
  server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Release)
  threading.Thread(target=server.serve_forever, daemon=True).start()
  try:
    path = tmp_path / "ttsim" / "libttsim_bh.so"
    monkeypatch.setattr(pci, "TTSIM_URL", f"http://127.0.0.1:{server.server_address[1]}/libttsim_bh.so")
    monkeypatch.setattr(pci, "TTSIM_PATH", path)
    monkeypatch.setattr(pci, "TTSIM_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="SHA-256"): pci.fetch_ttsim()
    assert not path.parent.exists() or not any(path.parent.iterdir()), "a download that doesn't check out leaves nothing"
    monkeypatch.setattr(pci, "TTSIM_SHA256", hashlib.sha256(library).hexdigest())
    assert pci.fetch_ttsim() == pci.fetch_ttsim() == path and path.read_bytes() == library and len(gets) == 2
  finally: server.shutdown(); server.server_close()

def test_setup_checks(tmp_path, monkeypatch):
  # open_device takes TTSIM's simulator, then a card, then the simulator that fetch_ttsim downloaded. check says what's
  # missing for each, without opening the device.
  bdf, downloaded = "0000:01:00.0", tmp_path / "libttsim_bh.so"
  (tmp_path / "devices").mkdir()
  downloaded.write_bytes(b"ttsim")
  monkeypatch.setattr(pci, "SYS", str(tmp_path))
  monkeypatch.setattr(pci, "TTSIM_PATH", downloaded)
  monkeypatch.setattr(pci, "TTSIM_SHA256", hashlib.sha256(b"ttsim").hexdigest())
  monkeypatch.setenv("TTSIM", "/elsewhere/libttsim.so")
  assert pci.simulator() == "/elsewhere/libttsim.so"
  monkeypatch.delenv("TTSIM")
  assert pci.simulator() == str(downloaded)
  assert check()[-1] == ("ok", f"the simulator at {downloaded} is the tested ttsim release")
  downloaded.write_bytes(b"another build")
  assert check()[-1][0] == "warn"
  downloaded.unlink()
  assert pci.simulator() is None
  level, message = check()[-1]
  assert level == "fail" and "python3 -m baretile.chip ttsim" in message, message
  card = tmp_path / "devices" / bdf
  card.mkdir()
  for name, text in (("vendor", f"{bh.VENDOR:#x}\n"), ("device", f"{bh.DEVICE:#x}\n")): (card / name).write_text(text)
  downloaded.write_bytes(b"ttsim")
  assert pci.simulator() is None, "a card comes before the downloaded simulator"
  assert all(message.startswith(bdf) for _, message in check()[3:]), "then check runs the card's preflight"
  monkeypatch.setattr(shutil, "which", lambda name: None)
  assert [level for level, _ in check()[:2]] == ["fail", "fail"], "no clang, no lld"

def test_release(tmp_path, monkeypatch):
  # release binds tt-kmd to a card that no driver has, through a fake sysfs. It leaves a driven card alone, and asks
  # for tt-kmd when its driver isn't loaded.
  bdf = "0000:01:00.0"
  dev, kmd = tmp_path / "devices" / bdf, tmp_path / "drivers" / pci.DRIVER
  dev.mkdir(parents=True)
  monkeypatch.setattr(pci, "SYS", str(tmp_path))
  monkeypatch.setattr(pci.os, "geteuid", lambda: 0)
  with pytest.raises(RuntimeError, match="modprobe"): pci.release(bdf)
  kmd.mkdir(parents=True)
  (kmd / "bind").write_text("")
  assert pci.release(bdf) == pci.DRIVER and (kmd / "bind").read_text() == bdf
  (kmd / "bind").write_text("")
  (dev / "driver").symlink_to(kmd)
  assert pci.release(bdf) == pci.DRIVER and (kmd / "bind").read_text() == ""

def test_hello(chip):
  assert set(hello(chip, chip.tensix[0], chip.tensix[-1], 30).values()) == {832040}

def test_every_core(chip):
  # cores other than the dispatcher write their coordinates, which they read from their NIU
  rt = Runtime(chip)
  rt.launch(compile_kernel("void kmain(volatile unsigned *a) { a[0] = noc_me(); }"))
  rt.wait()
  assert len(rt.cores) == len(chip.tensix) - 1  # plus the dispatcher
  for x, y in rt.cores: assert chip.read32(x, y, rt.args()) == x | y << 6

def test_takeover(chip):
  # A runtime that overlaps an earlier one stops it. The old runtime's cores outside the new rectangle are held in
  # reset too, since idling in firmware still costs ttsim time. A runtime beside the new one keeps running.
  full = Runtime(chip)
  first = Runtime(chip, end=chip.tensix[2])
  assert all(chip.read32(x, y, bh.SOFT_RESET) == bh.ALL_CORES for x, y in full.cores[3:] + [full.dispatcher])
  beside, me = Runtime(chip, start=chip.tensix[3], end=chip.tensix[5]), compile_kernel("void kmain(volatile unsigned *a) { a[0] = noc_me(); }")
  for rt in (first, beside):
    rt.launch(me), rt.wait()
    assert [chip.read32(x, y, rt.args()) for x, y in rt.cores] == [x | y << 6 for x, y in rt.cores]

def test_runtime_shared_dram(chip):
  a = Runtime(chip, end=chip.tensix[1])
  b = Runtime(chip, start=chip.tensix[2], end=chip.tensix[3])
  free = list(chip.dram_holes)  # runtimes of earlier tests elsewhere on the chip can still hold some
  first, second = a.dram_alloc(64), b.dram_alloc(128)
  assert first != second and a.holes is b.holes
  for xy in bh.DRAM:
    chip.write(*xy, first, bytes([11]) * 64)
    chip.write(*xy, second, bytes([22]) * 128)
    assert chip.read(*xy, first, 64) == bytes([11]) * 64
  a.close()
  assert not a.used and b.used == {second: 128}
  c = Runtime(chip, end=chip.tensix[1])
  assert c.dram_alloc(64) == first, "a stopped runtime's allocations can be reused"
  a.dram_free(first)  # a late tensor finalizer must not release c's allocation
  a.close()  # old cleanup must not halt c, even when its rectangle is identical
  c.launch(compile_kernel("void kmain(volatile unsigned *a) { a[0] = 111; }"))
  c.wait()
  assert chip.read32(*c.cores[0], c.args()) == 111
  assert chip.read(*bh.DRAM[0], second, 128) == bytes([22]) * 128
  c.close(), b.close()
  assert a.holes == free

def test_takeover_during_collection(chip, monkeypatch, no_gc):
  # Takeover owns only the old resources. Collecting the Runtime cycle during its drain must not close it again in
  # the middle of the same close, release allocations twice, or remove the rectangle twice.
  old = Runtime(chip, end=chip.tensix[1])
  free = list(chip.dram_holes)
  addr = old.dram_alloc(64)
  old.cycle = old
  owner, resources = weakref.ref(old), old.resources
  del old
  wait, halt, collected, stopped = chip.wait, chip.halt, False, 0
  def poll(cond, what, timeout=2.0):
    nonlocal collected
    if not collected:
      collected = True
      gc.collect()
    return wait(cond, what, timeout)
  def stop(*args):
    nonlocal stopped
    if not resources.closed: stopped += 1
    return halt(*args)
  with monkeypatch.context() as patch:
    patch.setattr(chip, "wait", poll)
    patch.setattr(chip, "halt", stop)
    replacement = Runtime(chip, end=chip.tensix[1])
  gc.collect()
  assert collected and stopped == 1 and owner() is None and resources.closed
  assert replacement.dram_alloc(64) == addr
  replacement.close()
  assert chip.dram_holes == free

def test_displaced_runtime_rejects_work(chip):
  a = Runtime(chip, end=chip.tensix[1])
  original = compile_kernel("void kmain(volatile unsigned *a) { a[0] = 111; }")
  overwritten = compile_kernel("void kmain(volatile unsigned *a) { a[0] = 222; }")
  block = a.block([(original, (), 1, "old")])
  b = Runtime(chip, end=chip.tensix[1])
  b.load(original)
  for work in (lambda a=a: a.load(overwritten), lambda a=a: a.launch(original), lambda a=a: a.send(*block),
               lambda a=a: a.dram_alloc(64), a.wait):
    with pytest.raises(RuntimeError, match="closed"): work()
  a.close()
  del work, a
  gc.collect()
  b.launch(original)
  b.wait()
  assert chip.read32(*b.cores[0], b.args()) == 111
  b.close()

def test_collected_runtime_keeps_dma_until_stopped(chip, monkeypatch):
  if not isinstance(chip.pci, pci.Sim): pytest.skip("mock DMA addresses cannot run on a physical card")
  # A Card-like allocator owns no references to its mmap. No simulator tick is needed: the queued command is
  # deliberately unfinished when the runtime and its reference cycle are collected. Collection leaves the card alone,
  # and the next runtime on the chip stops the collected one before its page goes.
  pages, halted, halt = [], [], chip.halt
  def alloc(size):
    page = mmap.mmap(-1, size)
    pages.append(weakref.ref(page))
    return memoryview(page), 1 << 48
  def stop(*args):
    if pages: halted.append(pages[0]() is not None)
    halt(*args)
  monkeypatch.setattr(chip.pci, "alloc", alloc)
  monkeypatch.setattr(chip.pci, "tick", lambda: pytest.fail("this command must stay queued"))
  monkeypatch.setattr(chip, "halt", stop)
  rt = Runtime(chip, end=chip.tensix[1])
  rt.launch(compile_kernel("void kmain(volatile unsigned *a) {}"))
  rt.cycle = rt
  assert rt.launches == 1 and rt.finished() == 0
  assert chip.read32(*rt.dispatcher, bh.SOFT_RESET) == bh.NCRISC
  ref, cores = weakref.ref(rt), [*rt.cores, rt.dispatcher]
  halted.clear()
  del rt
  gc.collect()
  assert ref() is None and pages[0]() is not None and halted == [], "collection must not touch the card"
  assert chip.read32(*cores[-1], bh.SOFT_RESET) == bh.NCRISC
  with Runtime(chip, start=chip.tensix[2], end=chip.tensix[3]):
    assert pages[0]() is None and halted[0] is True
    assert all(chip.read32(*xy, bh.SOFT_RESET) == bh.ALL_CORES for xy in cores)

def test_collection_during_another_runtimes_access(chip, monkeypatch, no_gc):
  # Python can collect a dropped runtime's reference cycle in the middle of another runtime's chip access. Stopping
  # it there would move the shared TLB window under that access and send its bytes to the collected runtime's tiles.
  dropped = Runtime(chip, end=chip.tensix[1])
  dropped.cycle, resources = dropped, dropped.resources
  with Runtime(chip, start=chip.tensix[2], end=chip.tensix[3]) as rt:
    del dropped
    x, y = rt.cores[0]
    chip.write(x, y, DATA, bytes(16))
    write, collected = chip.pci.write, []
    def collect_first(off, data):  # where CPython may run a pending collection: entering the next Python call
      if not collected and off < bh.TLB_WINDOW:
        collected.append(1)
        gc.collect()
      return write(off, data)
    with monkeypatch.context() as patch:
      patch.setattr(chip.pci, "write", collect_first)
      chip.write(x, y, DATA, bytes([0x55]) * 16)
    assert collected and chip.read(x, y, DATA, 16) == bytes([0x55]) * 16
    assert not resources.closed, "the collected runtime waits for a safe point"
  with Runtime(chip, start=chip.tensix[2], end=chip.tensix[3]):
    assert resources.closed and resources not in chip.taken.values()

def test_dropped_runtime_gives_memory_back(chip, no_gc):
  # A dropped runtime in a reference cycle still holds its DRAM until Python collects it. An allocation that finds no
  # room collects, stops it and tries again.
  dropped = Runtime(chip, end=chip.tensix[1])
  dropped.cycle = dropped
  with Runtime(chip, start=chip.tensix[2], end=chip.tensix[3]) as rt:
    room = max(n for _, n in chip.dram_holes)
    dropped.dram_alloc(room)
    del dropped
    rt.dram_alloc(room)  # MemoryError if the dropped runtime still held it

def test_takeover_after_hung_multicast(chip):
  # A kernel that never sees its multicast's acknowledgements never leaves the multicast region. Closing it can't
  # show that its transfers finished, so it resets the cores anyway and keeps its memory. A new runtime can still take
  # the tiles, as after any other hung kernel.
  rt = Runtime(chip, end=chip.tensix[1])
  addr, page = rt.dram_alloc(64), weakref.ref(rt.status.obj)
  rt.launch(compile_kernel("void kmain(volatile unsigned *a) { mcast_begin(); mcast_done(NIU(WR_ACKS) + 1); }"), units=1)
  with pytest.raises(TimeoutError): rt.wait(timeout=0.5)
  cores = [*rt.cores, rt.dispatcher]
  for attempt in range(2):
    with Runtime(chip, end=chip.tensix[1]) as fresh:
      assert all(chip.read32(*xy, MCAST_STATE + 4) == 0 for xy in cores)
      fresh.launch(compile_kernel("void kmain(volatile unsigned *a) { a[0] = 7; }"))
      fresh.wait()
      assert chip.read32(*fresh.cores[0], fresh.args()) == 7
      assert fresh.dram_alloc(64) != addr, "the hung runtime's DRAM may still be written"
  assert page() is not None and rt.used == {addr: 64}, "nor may its host page be released"
  with pytest.raises(RuntimeError, match="closed"): rt.launch(bytes(16))
  rt.resources.release()  # the kernel never issued its multicast, so the tests that follow can have the memory back

@pytest.mark.parametrize("kind,tag", [("OUTSTANDING", 1), ("OUTSTANDING", 15), ("OUTGOING", 3), ("CMD", 2)])
def test_runtime_close_drains_all_transactions(chip, monkeypatch, kind, tag):
  rt = Runtime(chip, end=chip.tensix[1])
  addr, page = rt.dram_alloc(64), weakref.ref(rt.status.obj)
  read, pending, polls = chip.read32, 2, 0
  watched = bh.NIU + bh.NIU_REGS[kind] + tag * (bh.NIU_REGS["INITIATOR"] if kind == "CMD" else 4)
  def delayed(x, y, at):
    nonlocal pending, polls
    if (x, y, at) == (*rt.cores[0], watched):
      polls += 1
      if pending:
        pending -= 1
        assert page() is not None and rt.used == {addr: 64}, "DMA and DRAM must survive an unfinished transaction"
        return 1
    return read(x, y, at)
  monkeypatch.setattr(chip, "read32", delayed)
  rt.close()
  assert polls >= 3 and page() is None and not rt.used

def test_runtime_close_during_dispatcher_multicast(chip, monkeypatch):
  if not isinstance(chip.pci, pci.Sim): pytest.skip("stop at a particular simulated cycle")
  rt = Runtime(chip, end=chip.tensix[2])  # two acknowledgements decrement a counter incremented only once
  addr, page = rt.dram_alloc(64), weakref.ref(rt.status.obj)
  rt.launch(compile_kernel("void kmain(volatile unsigned *a) {}"))
  with monkeypatch.context() as patch:
    patch.setattr(chip.pci, "TICK", 1)
    chip.wait(lambda: chip.read32(*rt.dispatcher, bh.NIU + bh.NIU_REGS["OUTSTANDING"]) == 255,
              "reach the dispatcher's multicast counter underflow")
  rt.close()
  assert page() is None and addr not in rt.used
  with Runtime(chip, end=chip.tensix[2]) as fresh:
    assert all(chip.read32(*xy, MCAST_STATE) == 0 for xy in [*fresh.cores, fresh.dispatcher])
    fresh.launch(compile_kernel("void kmain(volatile unsigned *a) {}"))
    fresh.wait()

@pytest.mark.parametrize("received", [2, 4, 6])
def test_runtime_close_during_worker_multicast(chip, monkeypatch, received):
  if not isinstance(chip.pci, pci.Sim): pytest.skip("stop at a particular simulated cycle")
  rt = Runtime(chip, end=chip.tensix[3])
  first, last = rt.cores[0], rt.cores[-1]
  addr, page = rt.dram_alloc(64), weakref.ref(rt.status.obj)
  source = bytes(range(64))
  chip.write(*first, DATA, source)
  acks = chip.read32(*first, bh.NIU + bh.NIU_REGS["WR_ACKS"])
  rt.launch(compile_kernel("""void kmain(volatile unsigned *a) {
    noc_wait();
    mcast_begin();
    unsigned acks = NIU(WR_ACKS) + 6;
    for (unsigned packet = 0; packet < 3; packet++) {
      noc_issue(L1_DATA, noc_me(), L1_DATA + packet * MAX_PACKET, a[2], MCAST, MAX_PACKET);
      for (volatile unsigned spin = 0; spin < 64; spin++) {}  // pause before issuing the next packet or clearing the counter
    }
    mcast_done(acks);
  }"""), (bh.noc_rect(first, last),), units=1)
  with monkeypatch.context() as patch:
    patch.setattr(chip.pci, "TICK", 1)
    chip.wait(lambda: (chip.read32(*first, bh.NIU + bh.NIU_REGS["WR_ACKS"]) - acks) % 2**32 >= received,
              "reach the worker's multicast acknowledgements")
  assert chip.read32(*first, MCAST_STATE + 4) == 1
  halt = chip.halt
  def stopped(*args):
    assert page() is not None and rt.used == {addr: 64}, "keep memory alive while the multicast finishes"
    assert chip.read32(*first, MCAST_STATE + 4) == 0
    assert chip.read32(*first, bh.NIU + bh.NIU_REGS["WR_ACKS"]) == (acks + 6) % 2**32
    return halt(*args)
  with monkeypatch.context() as patch:
    patch.setattr(chip, "halt", stopped)
    rt.close()
  assert page() is None and addr not in rt.used
  assert chip.read32(*first, bh.NIU + bh.NIU_REGS["WR_ACKS"]) == (acks + 6) % 2**32
  for xy in rt.cores[1:]:
    for packet in range(3): assert chip.read(*xy, DATA + packet * bh.MAX_PACKET, len(source)) == source

def test_runtime_startup_and_shutdown_failures(chip, monkeypatch):
  # Failed startup releases the new DMA page. Failed shutdown keeps its page and allocations until a retry.
  old = Runtime(chip, end=chip.tensix[1])
  old.close()
  write = chip.write
  def fail_start(x, y, addr, data, start=None):
    if addr == 0: raise RuntimeError("startup failed")
    return write(x, y, addr, data, start)
  with monkeypatch.context() as patch:
    patch.setattr(chip, "write", fail_start)
    with pytest.raises(RuntimeError, match="startup failed"): Runtime(chip, end=chip.tensix[1])
  assert not chip.taken
  rt = Runtime(chip, end=chip.tensix[1])
  addr, page = rt.dram_alloc(64), weakref.ref(rt.status.obj)
  with monkeypatch.context() as patch:
    def fail_stop(*_): raise RuntimeError("halt failed")
    patch.setattr(chip, "halt", fail_stop)
    with pytest.raises(RuntimeError, match="halt failed"): rt.close()
    assert page() is not None and rt.used == {addr: 64}
    with pytest.raises(RuntimeError, match="closed"): rt.launch(bytes(16))
  rt.close()
  assert page() is None and not rt.used and not chip.taken
  with Runtime(chip, end=chip.tensix[1]) as rt:
    rt.dram_alloc(64)
  assert not rt.used

def test_sim_dma_pages_are_released(chip):
  if not isinstance(chip.pci, pci.Sim): pytest.skip("ttsim's DMA registry")
  before = len(pci.Sim.dma)
  page, bus = chip.pci.alloc(4096)
  view = page[128:256]
  assert len(pci.Sim.dma) == before + 1
  del page
  gc.collect()
  assert len(pci.Sim.dma) == before + 1, "a derived view still owns the mapping"
  del view
  gc.collect()
  assert len(pci.Sim.dma) == before
  page, next_bus = chip.pci.alloc(4096)
  assert next_bus > bus, "released DMA addresses must not alias a later page"

def test_readers(chip):
  # Grid workers read their bank through one of its three ports. The banks split the grid evenly, and no port serves
  # more than a third of a bank's share. A reader's data crosses under half the NoC0 links that a random port would.
  rt = Runtime(chip)
  n, links = len(rt.grid[0]) * len(rt.grid[1]), lambda port, w: bh.hops(chip.raw(*port), chip.raw(*w))
  banks, ports = collections.Counter(b for b, _ in rt.readers), collections.Counter(p for _, p in rt.readers)
  assert len(rt.readers) == n and sorted(banks.values()) == sorted(n // 8 + (b < n % 8) for b in range(8))
  assert max(ports.values()) <= -(-n // 24) and all(p[0] == bh.DRAM[b][0] and 0 <= p[1] - bh.DRAM[b][1] < 3 for b, p in rt.readers)
  every = [(x, y + p) for x, y in bh.DRAM for p in range(3)]
  assert sum(links(p, w) for w, (_, p) in zip(rt.cores, rt.readers)) < sum(links(p, w) for w in rt.cores[:n] for p in every) / len(every) / 2

def test_relaunch_without_reset(chip):
  # Different kernels run back to back from the same address, each loaded after the one before was evicted. A stale
  # copy of the previous kernel would give the wrong answer.
  rt = Runtime(chip, end=chip.tensix[2])
  cores = rt.cores
  for i, (op, f) in enumerate([("*", int.__mul__), ("+", int.__add__)] * 2):
    rt.evict()
    rt.launch(compile_kernel(f"void kmain(volatile unsigned *a) {{ a[3] = (noc_me() + a[2]) {op} 3; }}"), (i, 0))
    rt.wait()
    for x, y in cores: assert chip.read32(x, y, rt.args() + 12) == f((x | y << 6) + i, 3), f"launch {i} on {x},{y}"
  # Kernels stay loaded until their area of L1 is full. Then the next load evicts them all, once the queue is done.
  big = [compile_kernel(f"void kmain(volatile unsigned *a) {{ a[1] = a[0] + {k}; }}").ljust(300_000, b"\0") for k in (1, 2)]
  rt.evict()
  for k, image in enumerate(big + big):
    rt.launch(image)
    rt.wait()
    assert rt.loaded == {image: KERNELS} and all(chip.read32(x, y, rt.args() + 4) == i + k % 2 + 1 for i, (x, y) in enumerate(cores))

def test_host_memory(chip):
  # A worker reads 64 KiB of host memory through the PCIe tile and writes it back to another part of the hugepage, as
  # the dispatcher's command reads, uploads and readback do. That's several NoC packets and PCIe requests each way.
  rt, n = Runtime(chip, end=chip.tensix[1]), 64 * 1024
  rt.back[:n], rt.back[n:2 * n] = (data := random.randbytes(n)), bytes(n)
  src, dst = rt.back_bus, rt.back_bus + n
  rt.launch(compile_kernel("""
void kmain(volatile unsigned *a) {  // a: [index, 1, PCIe x, y, source low, high, destination low, high, bytes]
  noc_read(a[2], a[3], a[4] | (unsigned long long)a[5] << 32, L1_DATA, a[8]);
  noc_wait();
  noc_write(L1_DATA, a[2], a[3], a[6] | (unsigned long long)a[7] << 32, a[8]);
}"""), (*bh.PCIE, src & 0xFFFFFFFF, src >> 32, dst & 0xFFFFFFFF, dst >> 32, n))
  rt.wait()
  assert bytes(rt.back[n:2 * n]) == data

def test_kernel_moves_data(chip):
  # Cores pull their 40 KiB slice of DRAM into L1 in 3 NoC packets, add their index to each word, and write the slice
  # back elsewhere in DRAM. Then they send the first 64 bytes to the L1 of the next core.
  rt, n = Runtime(chip, end=chip.tensix[4]), 40 * 1024  # four workers in the first row
  cores, (bx, by), src, dst = rt.cores, bh.DRAM[3], 1 << 20, 2 << 20
  data = [random.randbytes(n) for _ in cores]
  for i, d in enumerate(data): chip.write(bx, by, src + i * n, d)
  rt.launch(compile_kernel("""
void kmain(volatile unsigned *a) {  // a: [index, 1, bank x, bank y, src, dst, n, workers, each worker's x | y << 6]
  unsigned i = a[0], *buf = (unsigned *)L1_DATA, next = a[8 + (i + 1) % a[7]];
  noc_read(a[2], a[3], a[4] + i * a[6], L1_DATA, a[6]);
  noc_wait();
  for (unsigned k = 0; k < a[6] / 4; k++) buf[k] += i;
  noc_write(L1_DATA, a[2], a[3], a[5] + i * a[6], a[6]);
  noc_write(L1_DATA, next & 63, next >> 6, L1_DATA + a[6], 64);
}"""), (bx, by, src, dst, n, len(cores), *[x | y << 6 for x, y in cores]))
  rt.wait()
  out = [struct.pack(f"<{n // 4}I", *[(w + i) & 0xFFFFFFFF for w in struct.unpack(f"<{n // 4}I", d)]) for i, d in enumerate(data)]
  for i, o in enumerate(out):
    assert chip.read(bx, by, dst + i * n, n) == o, f"DRAM slice {i}"
    assert chip.read(*cores[(i + 1) % len(cores)], DATA + n, 64) == o[:64], f"L1 of the core after {i}"

def test_queue(chip):
  # 100 launches go back to back through a ring that holds 32 launches of 64 bytes. The ring fills, the host waits for
  # space, and the dispatcher wraps around three times. Launches add their index to a word in the workers' L1.
  rt = Runtime(chip, end=chip.tensix[2], ring=2048)
  for x, y in rt.cores: chip.write32(x, y, DATA, 0)
  add = compile_kernel("void kmain(volatile unsigned *a) { *(volatile unsigned *)L1_DATA += a[2]; }")
  rt.load(add)  # load it once up front, and it stays
  writes, write = [], chip.pci.write
  chip.pci.write = lambda *a: (writes.append(a), write(*a))
  try:
    for i in range(100):
      before = len(writes)
      rt.launch(add, (i,))
      assert len(writes) - before == 1, "a launch is one PCIe write: the doorbell"
  finally: chip.pci.write = write
  rt.wait()
  for x, y in rt.cores: assert chip.read32(x, y, DATA) == sum(range(100))

def test_queue_blocks(chip):
  # A block can fill the ring from any starting offset, or span several turns. Variable command lengths also
  # leave a tail too short for the next command. Replay must execute every launch in order each time.
  rt = Runtime(chip, end=chip.tensix[1], ring=256)
  chip.write32(*rt.cores[0], DATA, 0)
  add = compile_kernel("void kmain(volatile unsigned *a) { *(volatile unsigned *)L1_DATA += a[2]; }")
  rt.launch(add, (1,))
  block = rt.block([(add, (i,) + (0,) * (20 if i % 2 else 0), 1, "add") for i in range(1, 10)])
  for _ in range(3): rt.send(*block)
  rt.wait()
  assert rt.launches == 28 and chip.read32(*rt.cores[0], DATA) == 1 + 3 * sum(range(1, 10))

@pytest.mark.parametrize("workers", [1, 3, 8])
def test_large_work_partitions(chip, workers):
  kernel = compile_kernel("void kmain(volatile unsigned *a) {}")
  with Runtime(chip, end=chip.tensix[workers]) as rt:
    for units in (1, workers, workers + 1, 2**31 - 1, 2**31, 2**31 + 1, 2**32 - 1):
      rt.launch(kernel, units=units)
      rt.wait()
      n = min(workers, units)
      got = [struct.unpack("<2I", chip.read(*xy, rt.args(), 8)) for xy in rt.cores[:n]]
      expected = [(i * units // n, (i + 1) * units // n - i * units // n) for i in range(n)]
      assert got == expected and sum(count for _, count in got) == units

def test_launch_number_wrap(chip, monkeypatch):
  import baretile.runtime as runtime
  # Accelerate the real dispatcher's counter; the host keeps an unbounded count across its 32-bit status wrap.
  start = 2**32 - 2
  assert "launches = 0," in runtime.DISPATCHER
  monkeypatch.setattr(runtime, "DISPATCHER", runtime.DISPATCHER.replace("launches = 0,", f"launches = {start}u,"))
  with Runtime(chip, end=chip.tensix[2], ring=256) as rt:
    rt.launches = start
    rt.status[4:8] = start.to_bytes(4, "little")
    for xy in rt.cores: chip.write32(*xy, DATA, 0)
    kernel = compile_kernel("void kmain(volatile unsigned *a) { *(volatile unsigned *)L1_DATA += a[2]; }")
    addr = rt.dram_alloc(64)
    rt.launch(kernel, (1,), units=1, name="wrapped")
    rt.dram_free(addr)
    block = rt.block([(kernel, (i,), 1 if i == 2 else None, "wrapped") for i in range(2, 6)])
    rt.send(*block)  # both command-ring wrap and launch-number wrap, including a ready launch numbered zero
    rt.wait()
    assert rt.launches == start + 5 and rt.finished() == start + 5 and not rt.queued and not rt.used
    assert [chip.read32(*xy, DATA) for xy in rt.cores] == [15, 12], "the second worker first runs at launch zero"
    profile = rt.profile(start)
    assert len(profile) == 1 and profile[0][:2] == ("wrapped", 5)
    rt.launch(kernel, (6,))
    rt.wait()
    assert [chip.read32(*xy, DATA) for xy in rt.cores] == [21, 18]

def test_runtime_inputs(chip):
  taken = chip.taken.copy()
  for ring in (0, -64, 32, 192, 1 << 21):
    with pytest.raises(ValueError, match="ring"): Runtime(chip, ring=ring)
  with pytest.raises(ValueError, match="endpoints"): Runtime(chip, start=(0, 0))
  assert chip.taken == taken, "reject invalid settings before stopping an existing runtime"
  with pytest.raises(ValueError, match="four programs"): compile_kernel(*[None] * 5)
  rt = Runtime(chip, end=chip.tensix[1], ring=64)
  for units in (0, -1, 2**32, 1.5):
    with pytest.raises(ValueError, match="unit"): rt.launch(bytes(16), units=units)
  for args in ((-1,), (2**32,), (1.5,)):
    with pytest.raises(ValueError, match="arguments"): rt.launch(bytes(16), args)
  with pytest.raises(ValueError, match="command"): rt.launch(bytes(16), (0,) * 20)
  assert not rt.loaded and not rt.launches
  with pytest.raises(ValueError, match="never queued"): rt.wait(launch=1)

def test_dump(chip, tmp_path, monkeypatch):
  # With DUMP set, a runtime writes each kernel it launches once, a file for each core that has a program. With
  # device.h in place of its first line, each file is what clang compiled for that core: compile_rv32 makes the program
  # that sits at the core's entry in the image.
  monkeypatch.setattr(runtime, "DUMP", str(tmp_path))
  rt = Runtime(chip)
  kernel = compile_kernel("void kmain(volatile unsigned *a) { *(volatile unsigned *)a[2] = 1; }", None, "void kmain(volatile unsigned *a) {}")
  for _ in range(2): rt.launch(kernel, (DATA,), name="dump test")
  rt.wait()
  [at] = [d for d in tmp_path.iterdir() if d.is_dir()]
  assert at.name == f"dump-test-{hashlib.sha256(kernel).hexdigest()[:8]}" and sorted(f.name for f in at.iterdir()) == ["brisc.c", "trisc1.c"]
  device = (tmp_path / "device.h").read_text()
  assert device.startswith(runtime.DEVICE)
  entries = struct.unpack_from("<4I", kernel)
  for core, entry in (("brisc", entries[0]), ("trisc1", entries[2])):
    first, rest = (at / f"{core}.c").read_text().split("\n", 1)
    assert first == '#include "../device.h"'
    program = runtime.compile_rv32((device + rest).removeprefix(runtime.DEVICE))
    assert kernel[entry:entry + len(program)] == program, core

def test_hung_kernel_times_out(chip):
  # A timeout reports unfinished workers, active cores, synchronization counters and outstanding NoC requests.
  # Check a stuck BRISC, a stuck TRISC1 that leaves BRISC waiting for completion, and a report the dispatcher missed.
  rt = Runtime(chip, end=chip.tensix[1])
  rt.launch(compile_kernel("void kmain(volatile unsigned *a) { for (;;); }"))
  with pytest.raises(TimeoutError, match=r"launch 1, still running on \[\(1, 2\)\]\n  \(1, 2\): launch 1 not finished by "
                     r"BRISC\n    SYNC fetched 0, .*; FLAGS set: none\n    NoC requests in flight by id: none"): rt.wait(timeout=0.2)
  rt = Runtime(chip, end=chip.tensix[1])
  rt.launch(compile_kernel(None, None, "void kmain(volatile unsigned *a) { for (;;); }"))
  with pytest.raises(TimeoutError, match=r"\(1, 2\): launch 1 not finished by BRISC, TRISC1\n"): rt.wait(timeout=0.2)
  rt = Runtime(chip, end=chip.tensix[1])
  rt.launch(compile_kernel(f"void kmain(volatile unsigned *a) {{ ((volatile unsigned *){MAILBOX:#x})[5] = noc_me(); }}"))  # reports to itself
  with pytest.raises(TimeoutError, match=r"launch 1, whose workers all finished, so it missed a report$"): rt.wait(timeout=0.2)
  rt.close()  # a hung ordinary kernel holds cancellation up for at most the launch's two seconds
  assert rt.resources.closed
