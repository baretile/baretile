# Working on baretile

The [README](README.md) has the first example. Here we'll follow a tensor
operation down to the chip, with the register maps near the end for reference.
We develop on ttsim. The card code still needs testing.

## Running and debugging

Run these from the checkout, with clang and ld.lld on `PATH`. Set `TTSIM` to a
Blackhole `libttsim.so`. You'll need NumPy and pytest for the tests.

```sh
export TTSIM=/absolute/path/to/libttsim.so
python3 -m baretile.chip
python3 -m baretile.hello
python3 -m pytest -s test/test_device.py
python3 -m pytest -s test/test_kernels.py
python3 -m pytest -s test/test_models.py
```

The tests follow the layers of the code. `test_device.py` starts at PCIe access
and goes through the runtime. `test_kernels.py` tests the instructions and the
tensor operations built from them. Models, the server and the adapters are in
`test_models.py`.

If pytest exits without explaining why, run it with `-s`. ttsim stops the
process on errors such as an invalid address or taking an empty semaphore.
pytest can otherwise hide that error message. The tests share a `Chip` because
the simulator can only be initialized once per process.

The runtime compiles device programs with clang for `riscv32-unknown-elf` and
links them with lld. Builds go in `~/.cache/baretile`, so you won't have to
compile them again on the next run. Entries expire after a week without use.

### ttsim doesn't measure hardware speed

Tensix instructions take no time in ttsim. NoC requests complete immediately,
and some `STALLWAIT` conditions are ignored. Most of the cycles you see come
from the RISC-V code that controls the kernel. A lower count can show that
you've reduced that work, but it won't tell you how fast the kernel is on a card.

### ttsim-chaos

[test/ttsim-chaos.patch](test/ttsim-chaos.patch) adds delayed requests and random
pauses to ttsim v1.11.5 at commit `464814f`. Apply it to a separate ttsim checkout
and build the Blackhole library there. Run with delays enabled to check whether
the kernels wait for their data:

```sh
TTSIM=/path/to/ttsim-chaos/src/_out/release_bh/libttsim.so \
  TTSIM_CHAOS_NOC=1,400 TTSIM_CHAOS_RISCV=0.2 TTSIM_CHAOS_TENSIX=0.2 \
  python3 -m pytest -s test
```

`TTSIM_CHAOS_NOC=lo,hi` delays a request by a random number of clocks in that
range. The RISC-V and Tensix settings give the probability of skipping a turn.
Set `TTSIM_CHAOS_SEED` to repeat a run. `TTSIM_CHAOS_UNHARVESTED=1` enables all
140 Tensix tiles, so you can check the code with a larger grid.

The patch also implements the NIU counter for bytes received by reads.
`pci.Sim.counts` detects the patched library. Released ttsim doesn't support that
counter, so profiles from it report `None` for bytes.

### When a launch hangs

`Runtime.wait()` times out after 60 seconds without a completed launch by
default. Its report identifies the current launch and unfinished workers. For
up to four workers, it prints core status, synchronization counters, flags and
outstanding NoC requests. On a card it also reads the program counters.

If a worker never received `go`, check dispatch and multicast first. If the
workers finished but the host is still waiting, check their atomic reports and
the status write to host memory. A read ID that's still outstanding means a
transfer hasn't finished. Use the [pipeline counters](#unpack-math-and-pack) to
see where a kernel is waiting.

Creating a new runtime resets its worker cores. If the NoC or ARC has hung,
see [bring-up](#bring-up-on-a-p150) for the chip reset command.

## From Python to a launch

The imports follow the hardware upwards. `bh.py` holds the constants, `pci.py`
opens the device, and `chip.py` reads and writes through the NoC. The runtime
uses those to load and launch programs. Above it are the Tensix instructions,
kernels and tensors. Models and adapters live outside `baretile/`.

`Tensor` builds an expression when you add, multiply or reduce. `realize()` turns
that expression into kernels and queues them. `tolist()` also waits and copies
the result back to Python. A matmul computes its input matrices first. It can
combine operations on its output into the matmul kernel, as with the ReLU in
the README.

A kernel contains four RISC-V programs. BRISC moves data. TRISC0 feeds the
unpacker, TRISC1 feeds the math unit, and TRISC2 feeds the packer. NCRISC stays in
reset. The three TRISCs submit Tensix instructions to separate hardware queues.

The runtime loads a kernel into L1, the local memory of each worker tile. It
then puts a command in host memory and writes a doorbell on the dispatcher tile.
The dispatcher reads the command, sends its arguments to the workers, starts
them and waits for their reports. The host sees completion in its own memory.

## Runtime

`Runtime(chip)` uses the available Tensix tiles. You can restrict it with `start`
and `end` coordinates. The last tile in that rectangle is the dispatcher. The
others are workers. A new runtime stops any older runtime whose tiles overlap.

Some kernels need a rectangular grid of workers. The dispatcher's row has a
tile missing, so `Runtime.grid` leaves that row out. With 120 tiles, you get
119 workers and a grid of 108, in nine rows of twelve. `Runtime.readers` assigns
the grid workers to DRAM banks and ports. It spreads the workers across banks
and prefers ports with shorter return paths on NoC0.

### Local memory

We divide a worker's L1 like this. The constants are in `runtime.py`.

| Address | Use on a worker |
| --- | --- |
| `0x00000` | Firmware and entry jumps for BRISC and the three TRISCs |
| `0x01000` | Mailbox: launch, kernel, arguments, worker count, completion and worker identity |
| `0x01020` | TRISC launch and completion words |
| `0x01040` | `SYNC`, 64 bytes shared by a kernel's cores |
| `0x01080` | `FLAGS`, 32 words used between workers |
| `0x01100` | Two buffers for kernel arguments |
| `0x02000` | Loaded kernel images, ending before `0x7F000` |
| `0x7F000` | A 4 KiB tile of zeros |
| `0x80000` | Kernel data |

The dispatcher uses `0x1000` for its configuration and doorbell, and `0x1400`
for outgoing messages. At `0x80000` it counts worker reports and read-data flits.
It stages commands at `0x80040`, with room for 64 KiB.

Kernel images start with four entry offsets. They can run at different L1
addresses. The runtime loads an image once and reuses it until the kernel area
fills. It then waits for queued work before evicting the loaded images. The
worker firmware clears a kernel's `.bss` at launch.

### Commands and memory

A command has a 16-byte header: length, kernel address, argument length and worker
count. Arguments follow, and the command is padded to 64 bytes. Commands never
straddle the end of the host ring. A zero length marks unused space at the end.

`Runtime.send()` writes commands into the ring and updates the doorbell. A block
of recorded commands uses the same doorbell write. The dispatcher executes
launches in order and alternates argument buffers so it can prepare the next
launch while workers run.

Each worker reports completion with a NoC atomic increment. The dispatcher
writes timing data before completion status, using ordered posted writes.
`Runtime.wait()` polls that status in host memory.

The runtime allocates the same address range in eight DRAM banks. It uses the
first hole that's big enough, rounded to 64 bytes. When you free a tensor,
queued kernels may still be using its memory. The runtime waits for them to
finish before reusing that space. A recording keeps its memory for later calls.

### Recording repeated work

`tensor.jit(f)` runs the first call normally and records the second. Later calls
copy inputs into the recorded buffers and resend the launches. If kernels have
been evicted, the recording reloads them and rebuilds its commands.

Keep input shapes and dtypes fixed. The function must not read tensors back to
the host or depend on changing Python state. Results reuse the same buffers, so
a later call overwrites an earlier result. Copy a result if you need to keep it.

Decode works this way because a token's position is tensor data. Moving to the
next position doesn't require a new shape or a new compiled program.

### Profiling

Save `rt.launches` before the work, then call `rt.profile(saved)`. It waits and
returns `(name, launches, cycles, bytes)` grouped by kernel name. The runtime
keeps the last 1,024 launch records.

The dispatcher times a launch from `go` to the last worker's report. Bytes are
counted in 64-byte units and include everything the workers read through the
NoC. That includes other tiles' L1 and local copies. You can't use the byte
count as a DRAM measurement unless you know where the kernel reads from.

`baretile.viz` shows the same records as a page you open in a browser. Run any
script under it:

```sh
python3 -m baretile.viz models/llama.py /path/to/SmolLM-135M "The meaning of life is" 6
```

The script runs as it would under `python3`. When it ends, or stops with an
error or Ctrl-C, `viz.html` is written in the current directory with the
finished launches of each runtime the script made. From Python, call
`viz.save(path, rt, since=saved)` after a wait. It doesn't wait itself, so it
also works after a hang.

The timeline shows each launch from `go` to its last report, one after
another on the dispatcher's clock. Scroll to zoom and drag to pan. The table
lists the kernels by total cycles, with a fused kernel's input shapes after
its name, since many kernels share a name like `C matmul`. Pick one to see its
launches and, for a fused kernel, the expression that `fused_kernel`
compiled, drawn with the inputs at the top. Each distinct subexpression is one
node, since the kernel computes it once. The graphs show where fusion
recomputes a matmul in several kernels, and which values were realized into
kernels of their own. A kernel made by `Tensor.run` shows its cache key, and
one launched with `Runtime.launch` alone shows only its name.

The page covers the last 1,024 launches, like `profile`. It doesn't connect
kernels to each other: launch arguments are plain numbers, so the runtime
can't tell which of them are the DRAM addresses a launch reads or writes.

## Tensors and kernels

### Tiles and storage

A tile holds 32 by 32 elements in four 16 by 16 faces. Faces and their elements
are ordered by row. Element `(r, c)` has this offset within a tile:

```python
(r // 16 * 2 + c // 16) * 256 + (r % 16) * 16 + c % 16
```

Ordinary storage assigns tile `i` to bank `i % 8`. Tensor dimensions are padded
to tiles. A dimension of one is repeated across its tile for broadcasting.
Other padding is zeroed.

`Tensor.columns()` rearranges a matrix for use as the right operand of a
matmul. Column `j` goes in bank `j % 8`, with consecutive tiles down K. A worker
can then read several tiles in a single request. For block floats, exponents
are stored together before the values. Use this layout for matmul's B. You can
also read it back to the host, but other operations may reject it.

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

### Fusion and reductions

A fused kernel allocates L1 slots for inputs and intermediate results. Two
frames alternate so BRISC can fetch the next tile while Tensix computes. The
semaphore that counts intermediates can hold fifteen reports, so a kernel can
have at most fifteen operations. The compiler splits larger expressions.

Reductions make passes over a line of tiles. Softmax first finds the maximum,
then adds up the exponentials, then divides by that sum. A kernel can reduce
along one axis. If the expression also reduces along another axis, that part
is computed first. Inputs can stay in L1 across passes when there is room.

Some slices and reshapes change only the view of a tensor: its batch, strides
and offset. When data has to move, `move_kernel` copies it to the new layout.
`gather` reads rows by index, and `put` writes them. The models use these for
embeddings and KV caches. `put` can also update bfp8 and bfp4 caches.

Autograd walks the expression backwards and builds lazy operations for the
gradients. `test_autograd` checks these against numerical differences.
[test/train.py](test/train.py) compares a small GPT training run with PyTorch.

### Unpack, math and pack

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

### Matmul

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

### Matmul precision

Four fidelity phases multiply all the bits of a bf16 pair. Two phases leave
out the lowest bit of A. One phase uses fewer bits of B too. The model's
precision mode sets both the weight formats and the number of phases. ttsim
takes the same time regardless of the phase count.

Even at four phases, the sum is rounded. The matrix unit aligns groups of
products before adding them, and a partial sum reloaded through SrcA loses
bits in the conversion to TF32. For fp32 matmul, we split each operand into a
bf16 high part and a bf16 low part. We compute `hi @ hi + hi @ lo + lo @ hi`
and add those products in fp32.

fp32 elementwise operations use the SFPU, a 32-lane vector unit working on Dst.
It implements arithmetic and the polynomial approximations used for functions
such as exp and log. Packing a bf16 result rounds half up, while host uploads
use round-to-nearest-even. The two can differ by one bf16 step on the same input.

### Attention

`Tensor.attend` computes the scores and weighted values in an attention kernel,
then divides by the softmax sums in another launch. BRISC reads the token
positions and sliding-window bounds to build the masks. A new position changes
the mask without changing the recorded launches.

The kernel sweeps keys twice: first to find row maxima, then to compute
exponentials and weighted values. Values are read on the second sweep. Keys
stay in L1 between sweeps when they fit. Scores stay local instead of being
written as a full attention matrix in DRAM.

Decoding often has too few heads to occupy the workers. The implementation
splits the keys into up to eight parts and combines their maxima, sums and
weighted outputs afterwards. This adds launches. The Llama model code uses
ordinary tensor attention for heads narrower than 64 and for gpt-oss sinks.

## Models and tests

`models/gpt2.py` and `models/llama.py` can run the same model operations with
baretile tensors or NumPy arrays. The generation scripts compare the two at
each step. They feed NumPy's chosen token to both versions on the next step,
so one disagreement won't make the rest of the comparison useless.

`llama.load` reads a local Hugging Face config and safetensors weights. It
handles dense Llama-family models, several mixtures of experts, and the text
models of Qwen3.5 and Qwen3.6. The tokenizer reads byte-level BPE data from
`tokenizer.json`. The standalone GPT-2 command also needs `tiktoken`.

Choose the precision as the fourth argument after the script name:

```sh
python3 models/llama.py /path/to/SmolLM-135M "Hello" 6 bfp8
python3 models/gpt2.py /path/to/gpt2/model.safetensors "Hello" 6 bf16
```

`bf16` is the default. `bfp8` stores weight matrices in bfp8. The Llama `bfp4`
mode puts gate and up projections in bfp4 and the other matrices in bfp8.
`tt` follows Tenstorrent's performance configuration, including its fidelity
settings. `experts` puts the MLP matrices in bfp4, the other matrices in bfp8,
and the KV cache in bfp8. `llama.precision` has the settings for each operation.

The loader fuses q, k and v projections, and gate and up projections, where
appropriate. Matmul weights are uploaded in column order. The KV cache is
allocated once, and each token writes its keys and values at its position.

### Experts and Gated DeltaNet

`models/qwen.py` contains the operations that extend a Llama block. Expert
routing uses `topk` on the chip. During decode, `touched` lists the selected
experts so tokens can share their weight reads. For longer prompts, `group`
sorts tokens by expert.

Qwen3.5 and Qwen3.6 combine attention with Gated DeltaNet. A DeltaNet block keeps
convolution history and a fixed-size recurrent state. Its decode kernel performs
the convolution, gates and state update in fp32. Prompt processing works in
pieces of 64 tokens. State updates are written in place so later replays see
the new state.

Models with a multi-token prediction layer can draft a token and verify it with
the next token. Verification keeps state after the first token and after both
tokens. The next step reads whichever state matches the accepted tokens.
Free server slots leave their recurrent states unchanged.

### Optional checks

Most model tests use small configurations. To compare against transformers,
install PyTorch and a transformers release that includes the model you're
testing. The MXFP4 test also needs safetensors. Check pytest's skipped tests if
you expected one of these comparisons to run.

To include pretrained checkpoint checks:

```sh
GPT2=/path/to/gpt2/model.safetensors python3 -m pytest -s test -k gpt2_model
LLAMA=/path/to/SmolLM-135M python3 -m pytest -s test -k llama_model
```

`LLAMA8B=1` enables a test at Llama-3.1-8B block dimensions. `FUZZ=n` sets the
number of random expressions in the kernel fuzz test. `TTSIM2=/path/to/libttsim.so`
enables the two-chip test. Build that library with ttsim's `release_bh_x2` target.

`llama.accuracy` compares generated choices with Tenstorrent reference tokens.
The command accepts a `.refpt` prompt file, a step count and a precision mode.
Append `numpy` to run that comparison on the host. Reading the reference file
requires PyTorch. The usual reference procedure has a 512-token prompt and
500 following steps. Top-1 counts how often baretile picks the reference's best
token. Top-5 counts how often its pick is among the reference's five best tokens.

## Serving

The server accepts a folder or Hugging Face model ID. Its positional arguments
are model, port, slots, positions, precision and optional KV format:

```sh
python3 models/serve.py HuggingFaceTB/SmolLM-135M 8000 4 512 bf16
```

Here, four requests can run at once, with 512 positions per slot. Choose a
multiple of 32 for that size, and leave room for the answer as well as the
prompt. The engine also reserves positions at the end for inactive slots.

Downloads go to `~/.cache/baretile/hub`. `HF_TOKEN` supplies access to a gated
model, and `HF_ENDPOINT` changes the download endpoint. Completed downloads are
reused. The server binds to all interfaces on the selected port and has no API
key check.

It serves `GET /v1/models`, `POST /v1/completions` and
`POST /v1/chat/completions`. For the example above:

```sh
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"SmolLM-135M","prompt":"The meaning of life is","max_tokens":6,"temperature":0}'
```

Set `stream` to true in the request to receive tokens as they're generated.
You can set temperature, top_k, top_p, min_p, seed, stop strings and repetition
penalties. Defaults come from the model's `generation_config.json` when present.

The engine gives each request a slot in a shared KV cache and decodes occupied
slots together. Prompts are processed in chunks between decode steps. A request
gets HTTP 429 when no slot is free. A non-streaming request that receives no text
for 600 seconds gets HTTP 504. A stream that times out releases its slot.

### Reusing a conversation

When a prompt arrives, the engine looks for a free slot that has already
processed the start of it. It can also restore a saved session with its tokens,
KV cache and recurrent state. DeltaNet must resume from the exact tokens that
produced its state.

Sessions are saved in spare DRAM on the chip. When that fills, the server copies
the least recently used sessions to host memory. The host cache has a size limit
too, after which sessions are dropped. The response's usage field counts the
prompt tokens that were reused.

### Sampling and API limits

For supported vocabulary shapes, the device returns the largest 64 logits and
the normalization sum. Greedy selection and top_k up to 64 can use that result.
Penalties, logprobs and constrained output need the full row. Some other
sampling requests do too, so they can read much more data back to the host.

Chat templates are written in Python, so a model with a new template may need
code in `serve.py`. Qwen tool calls are returned in `tool_calls`, and reasoning
goes in `reasoning_content`. `tool_choice` and JSON response formats constrain
which tokens can be sampled. `test_chat_tools` and `test_serve` have example
requests to start from when adding a model or client.

## tinygrad and PyTorch

Import `adapters.tinygrad_baretile` before using tinygrad's `BARETILE` device.
The adapter translates tinygrad's graph into baretile tensors before tinygrad
schedules kernels. It recognizes a broadcast multiply followed by a reduction
as a matmul. `TinyJit` uses baretile recordings when the call can be recorded.

You can run an existing tinygrad script through the adapter:

```sh
python3 adapters/tinygrad_baretile.py /path/to/script.py
```

Install tinygrad or put its checkout on `PYTHONPATH`. The tests also accept a
checkout in `TINYGRAD`. The adapter calls tinygrad's internal APIs, so a tinygrad
update can break it.

Unsupported operations fall back to tinygrad's CPU device. These include
64-bit integers, bit operations, random-number generation and symbolic shapes.

PyTorch uses tinygrad's `extra/torch_backend`, which needs a tinygrad checkout,
PyTorch, a C++ compiler and ninja. `test_torch_frontend` exercises that path.

## Hardware reference

The addresses and encodings in [bh.py](baretile/bh.py) come from Tenstorrent's
[Blackhole ISA documentation](https://github.com/tenstorrent/tt-isa-documentation),
[ttsim](https://github.com/tenstorrent/ttsim),
[tt-kmd](https://github.com/tenstorrent/tt-kmd) and
[tt-umd](https://github.com/tenstorrent/tt-umd).
The Ethernet code also uses definitions from
[tt-metal](https://github.com/tenstorrent/tt-metal).

ISA paths below start at `BlackholeA0/`, and simulator paths start at `src/`.

### PCIe and TLB windows

Blackhole uses vendor `0x1E52`, device `0xB140`. The PCIe tile's address map is
described in `PCIExpressTile/README.md`. For controller details that aren't in
the ISA, see the simulator's `pcie.cpp` and kmd.

| BAR | Size | Contents |
| --- | --- | --- |
| BAR0 | 512 MiB | 202 windows of 2 MiB, followed by configuration registers |
| BAR2 | 1 MiB | PCIe controller, including eDMA and outbound iATU |
| BAR4 | 32 GiB | Eight windows of 4 GiB |

`pci.Card` maps BAR0 through sysfs. It unbinds an existing kernel driver, enables
memory access and bus mastering, and sets the maximum read request size to
4 KiB. It copies registers a word at a time. Using `memcpy` could split a register
access into separate byte transactions. TLB 201 is reserved for tt-kmd.

A TLB maps a BAR window to a tile and address on the NoC. The configuration
starts at BAR0 `0x1FC00000`, with three 32-bit words per entry. Write those words
separately: odd entries aren't aligned to eight bytes. Strided multicast words
for entries 0 through 31 start at `0x1FC009D8`.

The 2 MiB format comes from `PCIExpressTile/HostToDeviceTLBs.md`:

| Bits | Meaning |
| --- | --- |
| 0–42 | Tile address shifted right by 21 |
| 43–48, 49–54 | Destination x and y |
| 55–60, 61–66 | Start x and y for multicast |
| 67–68 | NoC selection |
| 69 | Multicast enable |
| 70–71 | Ordering: relaxed, strict, posted or counted |
| 72 | Linked mode |
| 73 | Static virtual channel enable |
| 75 | Buddy |
| 76–77 | Virtual channel class |

For a 4 GiB window, the address occupies bits 0–31. The coordinate fields start
at bits 32, 38, 44 and 50. NoC selection starts at 56, multicast at 58, ordering
at 59, linked mode at 61 and static VC at 62.

`Chip` uses TLB 0 as a movable window with strict ordering. `Chip.map()` reserves
other windows for repeated access, including a runtime's doorbell. Unicast uses
a dynamic virtual channel. Multicast uses static VC class 2. UMD warns against
using a static VC for a window that both reads and writes.

### Coordinates and DRAM

The raw NoC0 grid is 17 by 12. Firmware translates coordinates around harvested
columns. `Chip` checks that translation is enabled and reads the tables at
BAR0 `0x1FD04118` and `0x1FD04130`. `Chip.raw()` converts a translated position
back to the raw grid for routing calculations.

| Translated coordinates | Tile |
| --- | --- |
| x `1..7, 10..16`, y `2..11` | Tensix; the available columns come from telemetry |
| `(17 + b//4, 12 + 3*(b%4) + p)` | DRAM bank b, port p, for b `0..7` and p `0..2` |
| `(19, 24)` | PCIe connected to the host |
| x `20..31`, y `25` | Ethernet |
| `(8, 0)` | ARC |

A p150 with twelve available columns has 120 Tensix tiles. The code also
handles fourteen columns. Use `chip.tensix` when you need the cores, since the
list can change with the firmware and harvesting configuration.

NoC0 routes along increasing x, then increasing y, wrapping at the grid edges.
NoC1 routes the opposite way, along y first. A reply uses the same NoC as its
request. A different DRAM port can shorten the route back to a worker. The
runtime uses raw coordinates to compare those routes when assigning readers. See
`NoC/Coordinates.md`, `RoutingPaths.md` and `bh.hops`.

Each of the eight DRAM banks has three ports addressing the same memory.
The allocator uses `0xFF000000` bytes per bank, matching ttsim's 4080 MiB.
Telemetry reports enabled banks and training status. `Chip` waits up to
300 seconds for training and raises an error if a bank reports failure.

### Host memory and ARC

The card backend uses locked 2 MiB hugepages for DMA. It reads physical addresses
from `/proc/self/pagemap` and passes them to the device. You'll need root for
this. The IOMMU must be disabled or in passthrough mode so the card can use
those addresses without translation.

Host memory is reached through the PCIe tile. Addresses below `2**58` use the
direct bus-address path. Bit 60 selects the outbound iATU path used by kmd and
UMD. baretile uses the direct path and doesn't program iATU windows. We still
need to check whether firmware sets that up after boot.

If it doesn't, we'll need to program the iATU. The definitions are in kmd's
`blackhole.c`. Region r starts at BAR2 `0x1000 + 0x200*r`: base at `+0x08/0x0C`,
limit at `+0x10/0x20`, target at `+0x14/0x18`, CTRL_1 `1 << 13`, and CTRL_2
`1 << 31`.

ARC runs the board management firmware. `Chip` waits for the low three bits at
`0x80030408` to equal five, then sends messages to leave the idle power state,
enable clocks and request the busy AI clock. It reads back the reported clock.
The target in `bh.py` is 1,350 MHz.

| ARC address or tag | Meaning |
| --- | --- |
| `0x8003042C` | Pointer to queue metadata |
| `0x80030434` | Pointer to telemetry tags |
| `0x80030430` | Pointer to telemetry values |
| `0x80030100`, bit 16 | Notify ARC of a queued message |
| Tag 14 | AI clock in MHz |
| Tag 22 | Two training-status bits per DRAM bank |
| Tag 28 | Firmware bundle version |
| Tag 34 | Available Tensix columns |
| Tag 35 | Available Ethernet channels |
| Tag 36 | Enabled DRAM banks |

Applications use ARC queue 3. A queue has a 32-byte header, N request slots and
N response slots. Each slot is 32 bytes. Header offsets `0`, `4`, `0x10` and
`0x14` hold the request write, response read, request read and response write
pointers. Pointers wrap at `2*N`. A response status of 240 or above is an error.
The implementation follows kmd's `msgqueue.c` and UMD's protocol.

### Reset and RISC-V cores

The tile address map comes from `TensixTile/BabyRISCV/README.md`:

| Address | Contents |
| --- | --- |
| `0x00000000..0x0017FFFF` | 1.5 MiB L1, including executable code |
| `0xFFB00000` | Private core RAM, 8 KiB for BRISC/NCRISC and 4 KiB per TRISC |
| `0xFFB12000..0xFFB12FFF` | Control and debug registers |
| `0xFFB13138` | Program-counter snapshots |
| `0xFFB20000`, `0xFFB30000` | NoC0 and NoC1 interfaces |
| `0xFFE00000` and above | Tensix interfaces |

Private RAM and the Tensix interfaces aren't accessible through the NoC. Use
four-byte accesses for MMIO registers.

`SOFT_RESET_0` at `0xFFB121B0` holds a core in reset when its bit is set.
BRISC uses bit 11, TRISC0–2 use bits 12–14, and NCRISC uses bit 18. Load code
while the cores are held. A released core starts from L1. There's no boot ROM.
The runtime overrides the TRISC entry points through `0xFFB12228` and enables
the overrides at `0xFFB12234`.

The compiler targets `rv32im`. `.ttinsn` uses the encoding space that would
normally hold compressed RISC-V instructions. Don't use `mhartid` to identify
these cores, and don't use `fence.i` to reload a kernel. Firmware invalidates the instruction caches
through Tensix configuration instead.

Unsupported instructions have undefined behavior instead of trapping. Use
instructions from the subset in the existing device programs.

The small RISC-V data cache isn't coherent with NoC writes. Poll shared L1 with
a fence. Before another unit reads data that a core has stored, `l1_settle`
uses a fence, a load and another fence. A fence alone doesn't settle stores
across memory regions.

### NoC transfers

The device helpers in `runtime.DEVICE` provide reads, writes and waits.
`noc_get` and `noc_put` use preconfigured initiators 1 and 2. General transfers
use initiator 0. The rules come from `NoC/MemoryMap.md` and `NoC/Counters.md`.

A request carries at most 16 KiB. Source and destination must agree in their
low four address bits for L1, or low six when DRAM is involved. Wait for an
initiator to become free before changing its registers, and read back its
command register before consulting completion counters. Reads need acknowledgements.

Multicast needs broadcast, path reservation and static VC class 2. Count write
acknowledgements from the recipients. The outstanding-request counter goes up
once for the send, then down once per recipient, so it won't tell you when the
multicast is done. `mcast_done` waits for acknowledgements and clears the
affected counters after transfers have finished.

### Ethernet links

`Chip.links` reads which links are up and which chips they connect to.
`Chip.send` writes through an Ethernet tile to its peer. `link_kernel` uses
these links to copy between the chips' DRAM, and `Tensor.to` copies a tensor
this way. `test_two_chips` runs the copy on the two-chip simulator.

An Ethernet tile has 512 KiB of L1 and two RISC-V cores. E0 runs Tenstorrent's
link firmware, which owns the top 64 KiB. Keep E0 running. The link status is
at `0x7CC04`, and the peer identity starts at `0x7CFE0`.

Transmit queue 0 starts at `0xFFB90000`. Command 2 writes to the peer's L1.
Addresses and sizes must be multiples of sixteen bytes. The host can reach
these registers through the NoC. The copy kernel uses a worker on each chip,
two buffers in Ethernet L1, completion flags and return credits.

The definitions are in `EthernetTile/` in the ISA and `eth_fw_api.h`,
`dev_mem_map.h` and `tunneling.h` in tt-metal.

## Bring-up on a p150

We haven't run these steps on a card yet. Start at the bottom of the stack and
stop at the first failure, since the later tests need the earlier ones to work.

Check that the card appears in `lspci -d 1e52:b140`, has the required power,
and has working board firmware. Check firmware with Tenstorrent's tools before
opening baretile, because `pci.Card` unbinds tt-kmd. Configure the IOMMU for
passthrough or disable it, and reserve a few 2 MiB hugepages as root:

```sh
echo 4 > /proc/sys/vm/nr_hugepages
```

Run the following commands as root from the checkout, with `TTSIM` unset.
Use the Python environment that has the test dependencies.

1. Run `python3 -m baretile.chip`. Check ARC status, the clock, available columns
   and DRAM training before loading worker programs.
2. Run `python3 -m pytest -s test/test_device.py -k 'config_space or arc'`, then
   `python3 -m pytest -s test/test_device.py -k 'dma or l1 or (dram and not freed)'`.
   These test register access and memory transfers.
3. Run `python3 -m baretile.hello`, then the rest of `test/test_device.py`.
   Investigate launch failures before testing Tensix math.
4. Run `python3 -m pytest -s test/test_kernels.py -k 'tile_math or tensor_math'`.
   This checks instruction encoding and basic unpack, math and pack behavior.
5. Run the rest of `test/test_kernels.py`, then the model tests. If a failure
   comes and goes, check the waits that ttsim doesn't model.
6. Run `python3 -m baretile.probe 8 probe.json` and
   `python3 -m baretile.bench`. The probe saves measurements for the cost model.

A new runtime resets the cores after a kernel hangs. If ARC or the NoC stops
responding, close processes using the card before trying
`python3 -m baretile.chip reset`. That writes PCIe configuration, then removes
and rescans the device. So far, we've tested the reset code against fake sysfs
files. We don't know yet whether it recovers a hung card.

## Performance and cost estimates

`baretile.probe` measures launch overhead, transfers, memory latency and Tensix
operations. `baretile.bench` runs a matmul. Use a card for performance numbers.

`test/benchmark.py` compares matmul, Llama decode and prefill with TT-NN. Its
fourth workload sends concurrent HTTP requests to baretile. The TT-NN runs need
tt-metal's Python environment. You'll also need PyTorch to load the reference
file, and `cloc` to count lines of code. The script lists its commands at the top.

`models/cost.py` estimates memory use and serving cost from a model config:

```sh
python3 models/cost.py /path/to/Qwen3.6-35B-A3B experts context=16384 accept=0.55
```

Add `probe=probe.json` once you've measured a card. Until then, the model assumes
that decode gets 70% of the specified 512 GB/s DRAM bandwidth and that a launch
takes one microsecond. Those two assumptions are guesses.

The model counts the weights, caches, states and activations that a step reads.
It estimates how long those reads and the matrix operations take, uses the
longer time, then adds the cost of launches and RISC-V control code. The number
of different experts selected and the fraction of drafts accepted will change
with your workload.

## What still needs checking

On a card, we need to check that DMA reaches host memory, ARC turns the clocks
on, and the status writes arrive in order. We also need to check that loading
a new kernel invalidates the old instructions and that the Tensix registers
are at the indices we use. The BAR2 offsets and parts of ARC's protocol come
from the drivers and simulator because the ISA doesn't document them.

The host relies on x86 ordering between writes to the ring and the doorbell.
There's no explicit barrier for other host architectures.

There are also ISA and simulator differences: `UNPACR` fields have different
names, `SETDVALID` is unavailable in ttsim, and some address-mode documentation
describes Wormhole. The code uses the encodings accepted by the Blackhole simulator.

We haven't measured Qwen3.6 token accuracy with all matrix weights in bfp8 or
with a bfp4 KV cache.

ttsim also decodes an unexpected `XMOV` in very long loops of inline Tensix
instructions. We don't know why. The DeltaNet kernel calls SFPU functions to
avoid generating a loop that long.
