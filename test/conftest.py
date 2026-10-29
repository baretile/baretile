# Shared chip fixture and tensor helpers. test_device.py covers the device and runtime; test_kernels.py covers
# code generation and tensors; test_models.py covers models, serving and adapters. They run on the simulator that
# python3 -m baretile.chip ttsim downloads, or the libttsim.so that TTSIM names. On a machine with a card and no TTSIM,
# they open the first Blackhole on the PCI bus as root (see docs/bring-up.md).
import functools, gc, os, pytest
import numpy as np
from baretile import pci
from baretile.chip import Chip
from baretile.runtime import DATA, compile_kernel

def bf16(a, packer=False):  # rounds to bf16 and back: to nearest even like our uploads, or half up like the packer
  b = np.asarray(a, np.float32).view(np.uint32)
  return ((b + (0x8000 if packer else 0x7FFF + (b >> 16 & 1))) >> 16 << 16).view(np.float32).astype(np.float64)

def value(t): return np.array(t.tolist()).reshape(t.shape)  # t as a numpy array of its shape

HELD = 'void kmain(volatile unsigned *a) { while (*(volatile unsigned *)L1_DATA != a[2]) asm volatile("fence" ::: "memory"); }'
@functools.cache
def _held(): return compile_kernel(HELD)
def hold(rt):
  """Queue a launch on rt's first worker that waits for the host, so that the launches queued after it wait too, on
  a card as on ttsim. Call the function this returns to let them go. Don't upload tensor.DMA_UPLOAD bytes or more
  in between, since such an upload waits for the queue."""
  rt.wait()  # nothing queued earlier may write the word it waits on
  x, y = rt.cores[0]
  rt.chip.write32(x, y, DATA, 0)
  rt.launch(_held(), (1,), units=1, name="hold")
  return lambda: rt.chip.write32(x, y, DATA, 1)

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):  # with SKIPS=fail, a test that skips fails, for a CI job that must run all it selects
  report = (yield).get_result()
  if report.skipped and os.environ.get("SKIPS") == "fail":
    report.outcome, report.longrepr = "failed", f"it skipped, with SKIPS=fail: {report.longrepr}"

@pytest.fixture
def no_gc():  # Python collects only where the test calls gc.collect()
  collecting = gc.isenabled()
  gc.disable()
  yield
  if collecting: gc.enable()

@pytest.fixture(scope="session")
def chip(): return Chip(pci.open_device())  # session-wide, since libttsim can only be initialized once a process
