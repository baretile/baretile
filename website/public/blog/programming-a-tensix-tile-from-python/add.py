import struct

from baretile import bh, pci
from baretile.chip import Chip
from baretile.runtime import Runtime, compile_kernel
from baretile.tensix import (
    MATH_END, from_bf16, kernel_config, math_begin,
    math_program, pack, pack_program, slot_base,
    to_bf16, tt, unpack, unpack_program,
)

chip = Chip(pci.open_device())
rt = Runtime(chip)
x, y = rt.cores[0]

a = [[float((r * 32 + c) % 7 - 3) for c in range(32)]
     for r in range(32)]
b = [[float((r + 2 * c) % 5 - 2) for c in range(32)]
     for r in range(32)]

def tile_index(r, c):
    face = (r // 16) * 2 + c // 16
    return face * 256 + (r % 16) * 16 + c % 16


def to_l1(slot, matrix):
    words = [0] * 1024
    for r in range(32):
        for c in range(32):
            words[tile_index(r, c)] = to_bf16(matrix[r][c])
    address = slot_base("bf16") + slot * 2048
    chip.write(x, y, address, struct.pack("<1024H", *words))


to_l1(1, a)
to_l1(2, b)

def ins(name, **fields):
    opcode, layout = bh.TENSIX_OPS[name]
    word = opcode << 24
    for name, value in fields.items():
        first, width = layout[name]
        assert 0 <= value < 1 << width
        word |= value << first
    return word


add = ins("ELWADD", dst=0)
print(hex(add))
print(tt([add]))

unpacks = unpack(1, 2)

adds = [ins("SETRWC", bit_mask=0b1111)]
adds += [ins("ELWADD", dst=0) for _ in range(7)]
adds += [ins("ELWADD", dst=0, clear_dvalid=3)]

maths = math_begin(False) + adds + MATH_END
packs = pack("h", 3)

kernel = compile_kernel(
    None,
    unpack_program(tt(unpacks), kernel_config()),
    math_program({}, tt(maths)),
    pack_program(tt(packs)),
)

rt.launch(kernel, (), 1)
rt.wait()

def from_l1(slot):
    address = slot_base("bf16") + slot * 2048
    words = struct.unpack("<1024H", chip.read(x, y, address, 2048))
    return [
        [from_bf16(words[tile_index(r, c)]) for c in range(32)]
        for r in range(32)
    ]


got = from_l1(3)
expected = [[a[r][c] + b[r][c] for c in range(32)]
            for r in range(32)]

assert got == expected
print("all 1024 sums match")

from baretile.tensor import Tensor

A = Tensor.of(rt, a)
B = Tensor.of(rt, b)
assert (A + B).tolist() == got
print("Tensor result matches the handwritten kernel")
