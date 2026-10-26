# The one-command demo. It boots firmware on the Tensix cores of a rectangle, launches a kernel built with upstream
# clang and lld on the workers, and reads back what they wrote.
#   TTSIM=/path/to/libttsim.so python3 -m baretile.hello
from baretile import pci
from baretile.chip import Chip
from baretile.runtime import Runtime, compile_kernel

FIB = """
void kmain(volatile unsigned *args) {  // args: [first unit, units, n], and fib(n) goes in args[3]
  unsigned a = 0, b = 1;
  for (unsigned i = 0; i < args[2]; i++) { unsigned t = a + b; a = b; b = t; }
  args[3] = a;
}
"""

def hello(chip, start, end, n):
  """Compute fib(n) on the workers in the rectangle from start to end. Returns {core: answer}."""
  rt = Runtime(chip, start, end)
  rt.launch(compile_kernel(FIB), (n, 0))
  rt.wait()
  return {(x, y): chip.read32(x, y, rt.args() + 12) for x, y in rt.cores}

if __name__ == "__main__":
  chip = Chip(pci.open_device())
  answers = hello(chip, chip.tensix[0], chip.tensix[-1], 30)
  print(f"{len(answers)} Tensix cores computed fib(30) = {set(answers.values())}, and one more core dispatched the work")
