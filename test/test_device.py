# Device and runtime tests (layers 0 and 1), plus import and layer checks. See conftest.py for simulator setup.
import ast, collections, ctypes, os, pathlib, random, struct, sys, threading, time, pytest
from baretile import bh, pci
from baretile.hello import hello
from baretile.runtime import DATA, KERNELS, Runtime, compile_kernel

def test_layers():
  # Modules in baretile/ import only the standard library and the layers below their own. A layer's tests, and the
  # helpers in conftest, import nothing above that layer. Outside the tests, no file uses the private names of another
  # file, whether it imports them or uses them as attributes that only the other file defines.
  root = pathlib.Path(__file__).parent.parent
  order = ["bh", "pci", "chip", "runtime", "hello", "tensix", "kernels", "tensor", "bench", "probe"]  # layers 0-1, then 2-3, each with its commands
  trees = {p: ast.parse(p.read_text()) for d in ("baretile", "models", "adapters") for p in sorted((root / d).glob("*.py"))}
  assert sorted(order) == sorted(p.stem for p in trees if p.parent.name == "baretile"), "every module in baretile/ needs a layer"
  def imports(name, tree, below, also=()):  # each import is from the standard library, from also, or a baretile module in below
    for n in ast.walk(tree):
      if not isinstance(n, (ast.Import, ast.ImportFrom)): continue
      for m in [a.name for a in n.names] if isinstance(n, ast.Import) else [f"{n.module}.{a.name}" for a in n.names]:
        top, _, sub = m.partition(".")
        assert sub.split(".")[0] in below if top == "baretile" else top in sys.stdlib_module_names or top in also, f"{name} imports {m}"
  for p, tree in trees.items():
    if p.parent.name == "baretile": imports(p.name, tree, order[:order.index(p.stem)])
  for name, top in (("conftest", "hello"), ("test_device", "hello"), ("test_kernels", "probe")):
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

def test_dram(chip):
  addr, data = bh.TLB_WINDOW - 2048, [random.randbytes(4096) for _ in bh.DRAM]  # straddles two 2 MiB windows
  for (x, y), d in zip(bh.DRAM, data): chip.write(x, y, addr, d)
  for bank, ((x, y), d) in enumerate(zip(bh.DRAM, data)):
    for port in range(3): assert chip.read(x, y + port, addr, len(d)) == d, f"bank {bank} port {port}"

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
  # pci.Card and pci.reset check for root before they touch the card, so this test needs no card. Without root they
  # raise PermissionError before any sysfs write can fail.
  if os.geteuid() == 0: pytest.skip("as root, Card would go on to open the device")
  with pytest.raises(PermissionError, match="root"): pci.Card("0000:00:00.0")
  with pytest.raises(PermissionError, match="root"): pci.reset("0000:00:00.0")

def test_card_sysfs(tmp_path, monkeypatch):
  # This runs pci.Card and reset against a fake sysfs, so no card is needed. Card unbinds the driver, enables the
  # device, turns on memory decode and bus mastering, and sets Max Read Request Size to 4 KiB. It maps BAR0, clears the
  # strided multicast of TLBs 0-31 there, and refuses an IOMMU that translates. find_cards finds the card by its ids.
  # reset fires the interface timer, waits for the card to clear the marker, and has the kernel enumerate it again.
  bdf = "0000:01:00.0"
  dev = tmp_path / "devices" / bdf
  for d in ("driver", "iommu_group"): (dev / d).mkdir(parents=True)
  for name, text in (("vendor", f"{bh.VENDOR:#x}\n"), ("device", f"{bh.DEVICE:#x}\n"), ("enable", "0"), ("driver/unbind", ""),
                     ("iommu_group/type", "identity\n"), ("remove", "")): (dev / name).write_text(text)
  config = bytearray(4096)  # PCIe capability at 0x40. Device control at 0x48 asks for reads of 512 bytes
  config[0x34], config[0x40], config[0x48:0x4A] = 0x40, 0x10, (0x2810).to_bytes(2, "little")
  (dev / "config").write_bytes(config)
  # BAR0, with TLBs left set by an earlier user
  with open(dev / "resource0", "wb") as f: f.truncate(1 << 29), f.seek(bh.TLB_STRIDED), f.write(b"\xff" * 128)
  monkeypatch.setattr(pci, "SYS", str(tmp_path))
  monkeypatch.setattr(pci.os, "geteuid", lambda: 0)
  assert pci.find_cards() == [bdf]
  card, data = pci.Card(bdf), random.randbytes(64)
  card.write(0x1000, data)
  with open(dev / "resource0", "rb") as f: written = f.read(0x1040)[0x1000:]
  assert (dev / "driver/unbind").read_text() == bdf and (dev / "enable").read_text() == "1" and written == data
  assert card.cfg_read(0x04) & 6 == 6 and card.cfg_read(0x48) & 0xFFFF == 0x5810 and card.read(bh.TLB_STRIDED, 128) == bytes(128)
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

def test_hung_kernel_times_out(chip):
  # A launch that hangs times out with a report from each stuck worker: which of its cores are still in the kernel, the
  # kernel's counters, and its NoC requests in flight. The first kernel hangs BRISC. The second hangs TRISC1, and BRISC
  # waits for it.
  rt = Runtime(chip, end=chip.tensix[1])
  rt.launch(compile_kernel("void kmain(volatile unsigned *a) { for (;;); }"))
  with pytest.raises(TimeoutError, match=r"launch 1, still running on \[\(1, 2\)\]\n  \(1, 2\): launch 1 not finished by "
                     r"BRISC\n    SYNC fetched 0, .*; FLAGS set: none\n    NoC requests in flight by id: none"): rt.wait(timeout=0.2)
  rt = Runtime(chip, end=chip.tensix[1])
  rt.launch(compile_kernel(None, None, "void kmain(volatile unsigned *a) { for (;;); }"))
  with pytest.raises(TimeoutError, match=r"\(1, 2\): launch 1 not finished by BRISC, TRISC1\n"): rt.wait(timeout=0.2)
