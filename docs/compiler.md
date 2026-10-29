# Compiler

The compiler turns tensor expressions into kernels. [tensix.py](../baretile/tensix.py)
encodes Tensix instructions and builds the four programs of a kernel.
[kernels.py](../baretile/kernels.py) builds the kernels from them, and
[tensor.py](../baretile/tensor.py) decides which kernels an expression needs.
[ramps/3_tensix.py](../ramps/3_tensix.py) writes the unpack, math and pack
programs of a small kernel by hand.

## Tiles and storage

A tile holds 32 by 32 elements in four 16 by 16 faces. Faces and their elements
are ordered by row. Element `(r, c)` has this offset within a tile:

```python
(r // 16 * 2 + c // 16) * 256 + (r % 16) * 16 + c % 16
```

Ordinary storage assigns tile `i` to bank `i % 8`. Tensor dimensions are padded
to tiles. A dimension of one is repeated across its tile for broadcasting.
Other padding is zeroed.
Tensor dimensions must be positive integers. `Tensor.of` accepts rectangular
matrices of the same shape, and `Tensor.frombytes` checks the byte count before
allocating memory. Operations require operands from the same runtime; use
`Tensor.to` to transfer data between chips.

`Tensor.columns()` rearranges a matrix for use as the right operand of a
matmul. Tile column `j` goes in bank `j % 8`, with consecutive tiles along the
dimension K that the matmul sums over. A worker can then read several tiles in
a single request. For block floats, exponents are stored together before the
values. Use this layout for matmul's B. You can also read it back to the host.
Cast to bf16 to use ordinary operations on the result.

| Format | Bytes per tile | Notes |
| --- | ---: | --- |
| bf16 | 2,048 | Default floating-point storage |
| fp32 | 4,096 | Elementwise math runs on the SFPU |
| int32 | 4,096 | Integer indices and arithmetic |
| bfp8 | 1,088 | 64 shared exponents and 1,024 value bytes |
| bfp4 | 576 | 64 shared exponents and 512 packed value bytes |

In a block float, sixteen values share an exponent. bfp8 has a sign and seven
magnitude bits per value. bfp4 has a sign and three. The unpacker converts these
values for the matrix unit. For bfp4, the hardware packer truncates. We use a
BRISC kernel to round instead.

## Fusion and reductions

A fused kernel allocates L1 slots for inputs and intermediate results. Two
frames alternate so BRISC can fetch the next tile while Tensix computes. The
semaphore that counts intermediates can hold fifteen reports, so a kernel can
have at most fifteen operations. The compiler splits larger expressions.

Reductions make passes over a line of tiles. Softmax first finds the maximum,
then adds up the exponentials, then divides by that sum. A kernel can reduce
along one axis. If the expression also reduces along another axis, that part
is computed first. Inputs can stay in L1 across passes when there is room.

Batch indexing, tile-aligned slices, `split` and `repeat_interleave` can share
storage through a view's batch, strides and offset. A shape-changing `reshape`
copies through `move_kernel`; an identical-shape reshape returns its input.
Other movement also copies when it cannot use a view.
`gather` reads rows by index, and `put` writes them. The models use these for
embeddings and KV caches. `put` can also update bfp8 and bfp4 caches.
Casting block floats to bf16 decodes them on the chip into ordinary tiles;
casting column storage to its own dtype also restores ordinary tile order.
Further casts and movement use those tiles. Column copies preserve gradients.
Block-float casts use a straight-through gradient in bf16: backward treats the
quantizer as an identity, while matmul uses the stored, quantized weights for
its input gradient. This approximates the quantizer's derivative.
Matmul with `dtype="int32"` accumulates in fp32 and then converts the result.
Expressions with several indexed matmuls or mixed block-float weights are split
into supported kernels before fusion.
`move` checks that its selected source positions stay inside the input's logical
shape. Its offset and strides must fit signed 32-bit integers. Padding outside
the selected output bounds reads no source data. Negative `pad` widths crop
elements; their gradients are zero. Padding accepts at most three integer pairs.
`assign` preserves the original source when source and destination views overlap
with different layouts. `realize(into=...)` also computes through a temporary
when its destination overlaps a matmul input or a differently laid out input.
`put` snapshots overlapping source rows and indices before writing, and keeps
singleton rows replicated across their physical tiles for later broadcasting.

Autograd walks the expression backwards and builds tensor operations for the
gradients. `backward()` queues the forward value and every leaf gradient before
returning, so sequential parameter updates use the same forward values.
`gradient()` returns lazy expressions; realize all of them before changing any
forward operand. `test_autograd` checks gradients against numerical differences.
[test/train.py](../test/train.py) compares a small GPT training run with PyTorch.

A single-element tensor converts to `bool` by waiting and reading its value.
The truth value of a larger tensor is ambiguous and raises `ValueError`, including
in Python conditions and container membership checks. A scalar truth test is a
host read and must stay outside a recorded function.

## Unpack, math and pack

`tensix.py` encodes instructions from the ISA. A word contains the opcode in its
top byte and the instruction fields below it. Constant instructions can use
`.ttinsn`, which rotates the word left by two bits in the RISC-V instruction
stream. Instructions with fields computed at runtime use stores instead.

TRISC0 sets the backend registers, thread configuration and semaphores before
marking the kernel ready. Each thread also sets its address counters. If BRISC
and a TRISC submit a Tensix instruction in the same cycle, the TRISC's instruction
can be dropped. That's why BRISC handles data movement and leaves Tensix to
the TRISCs.

The unpacker fills SrcA and SrcB. The math thread takes a free half of Dst,
computes, and posts `SEM_DST`. The packer writes that result to L1 and signals
either an intermediate in L1 or a completed output. There are two halves of Dst,
so math and pack can work at the same time.

BRISC can't read the Tensix semaphores. TRISC0 and TRISC2 relay progress into
`SYNC`, where the RISC-V cores can read it after a fence.

| Counter | Writer | What it tells the other cores |
| --- | --- | --- |
| `FETCHED` | BRISC | Input frames have arrived |
| `ISSUED` | BRISC | Matmul reads have been submitted |
| `UNPACKED` | TRISC0 | Input slots can be reused |
| `READY` | TRISC0 | Configuration is complete |
| `PACKED` | TRISC2 | Output is ready to send |
| `SENT` | BRISC | Output transfers have finished |

Semaphores hold at most fifteen reports. A thread must relay reports before
posting a sixteenth. A core still has work to do while it waits. BRISC sends
completed output, and the TRISCs relay progress. If you remove that work from
a wait loop, another core may be waiting for it, and both will stop.

## Matmul

The matrix unit computes `Dst[8x16] += SrcB[8x16] @ SrcA[16x16]`. The operand
names are easy to reverse: the left matrix A goes to SrcB, and B goes to SrcA.
A pair of 32 by 32 tiles takes sixteen `MVMUL` instructions per fidelity phase.

For a matrix with several tile rows, workers share A across grid rows and B down
grid columns. Three buffers let them rotate through inputs without a separate
message for each freed buffer. If K doesn't fit, the kernel splits it and keeps
partial sums in L1. If the output accumulators don't fit either, it processes
groups of columns and reads A again for each group.

A single tile row, as in token decoding, uses a different path. Workers take
output columns, share A, and stream B from DRAM. With column storage, workers
read through their assigned bank ports in requests of up to 16 KiB. The default
`CHUNKS` and `DEPTH` are both four. We'll need a card to tune them.

For a mixture of experts, matmul reads the expert indices from a tensor on the
chip. It uses those indices to select matrices from B. An index past the last
matrix gives zero. The host doesn't need to read the indices, so the same
recorded launches work when the next token selects different experts.

## Matmul precision

Four fidelity phases multiply all the bits of a bf16 pair. Two phases leave
out the lowest bit of A. One phase uses fewer bits of B too. The model's
precision mode sets both the weight formats and the number of phases. ttsim
takes the same time regardless of the phase count.

Four phases make bf16 products exact, but the sum is still rounded. The matrix
unit aligns groups of products before adding them. A partial sum reloaded
through SrcA loses bits in the conversion to TF32. For fp32 matmul, we split
each operand into a bf16 high part and a bf16 low part. We compute
`hi @ hi + hi @ lo + lo @ hi` and add those products in fp32.

fp32 elementwise operations use the SFPU, a 32-lane vector unit working on Dst.
It implements arithmetic and the polynomial approximations used for functions
such as exp and log. Packing a bf16 result rounds half up, while host uploads
use round-to-nearest-even. The two can differ by one bf16 step on the same input.

Integer comparisons, minimum and maximum cover the full int32 range, including
`-2**31`. Integer addition and subtraction wrap; conversion to fp32 rounds to
nearest. fp32 comparisons treat NaN as unordered: `<`, `<=`, `>`,
`>=` and `==` are false, and `!=` is true. fp32 `where` preserves the selected
value's bits, including the sign of zero.

Square root covers every positive normal fp32 input, including the smallest
normal value and the largest finite value. Its iteration keeps every
intermediate value near the root, its reciprocal or 1/2, so none underflows
or overflows. Subnormal arithmetic flushes to zero.
Sine and cosine support finite inputs with absolute value below `2**24`;
inputs outside that range return NaN. Their range reduction splits both the
integer turn count and pi to keep large-argument subtraction accurate.

## Attention

`Tensor.attend` computes the scores and weighted values in an attention kernel,
then divides by the softmax sums in another launch. BRISC reads the token
positions and sliding-window bounds to build the masks. A new position changes
the mask without changing the recorded launches.

The kernel sweeps keys twice: first to find row maxima, then to compute
exponentials and weighted values. Values are read on the second sweep. Keys
stay in L1 between sweeps when they fit. Scores stay local instead of being
written as a full attention matrix in DRAM.

Each unit handles at most 64 query rows, keeping their maxima in fp32 Dst.
The SFPU subtracts those maxima after the full scores have accumulated. This
avoids overflow from rounding a large maximum or subtracting it too early.
Longer prompts use more units and reread K/V for each group of 64 query rows;
the cost estimator counts those reads. This trades more prefill traffic for
stable scores. Heads whose frames do not fit in L1 raise `ValueError` before
allocation or launch. A supported frame with no space to retain keys streams
them again for the second sweep.

Decoding often has too few heads to occupy the workers. The implementation
splits the keys into up to eight parts and combines their maxima, sums and
weighted outputs afterwards. This adds launches. The Llama model code uses
ordinary tensor attention for heads narrower than 64 and for gpt-oss sinks.
