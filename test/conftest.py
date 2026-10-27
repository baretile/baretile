# Shared chip fixture and tensor helpers. test_device.py covers the device and runtime; test_kernels.py covers
# code generation and tensors; test_models.py covers models, serving and adapters. Set TTSIM to libttsim.so to
# use the simulator. Otherwise tests open the first Blackhole on the PCI bus as root (see NOTES.md, "Bring-up").
import pytest
import numpy as np
from baretile import pci
from baretile.chip import Chip

def bf16(a, packer=False):  # rounds to bf16 and back: to nearest even like our uploads, or half up like the packer
  b = np.asarray(a, np.float32).view(np.uint32)
  return ((b + (0x8000 if packer else 0x7FFF + (b >> 16 & 1))) >> 16 << 16).view(np.float32).astype(np.float64)

def value(t): return np.array(t.tolist()).reshape(t.shape)  # t as a numpy array of its shape

@pytest.fixture(scope="session")
def chip(): return Chip(pci.open_device())  # session-wide, since libttsim can only be initialized once a process
