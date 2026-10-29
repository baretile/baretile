# Runtime

The runtime loads programs onto the workers and launches them. It also
allocates DRAM, records launches for replay and profiles them. The code is in
[runtime.py](../baretile/runtime.py), and [ramps/2_runtime.py](../ramps/2_runtime.py)
runs C kernels through it.

The runtime loads a kernel into L1, the local memory of each worker tile. It
then puts a command in host memory and writes a doorbell on the dispatcher tile.
The dispatcher reads the command, sends its arguments to the workers, starts
them and waits for their reports. The host sees completion in its own memory.

## Workers and the dispatcher

`Runtime(chip)` uses the available Tensix tiles. You can restrict it with `start`
and `end` coordinates. The last tile in that rectangle is the dispatcher. The
others are workers. A new runtime closes any older runtime whose tiles overlap.
The displaced runtime, its tensors and its recordings reject further work.
Disjoint rectangles can run together and allocate from the same DRAM pool.
Both endpoints must be available Tensix tiles. The runtime requires all eight
DRAM banks. A custom command ring must be a power of two from 64 bytes through
1 MiB, leaving room for status, timing records and tensor data in its hugepage.

Call `rt.close()` when finished, or use `with Runtime(chip) as rt:`. Closing stops
all workers and cancels queued work; call `rt.wait()` first if it must finish.
Closing first requests cancellation and lets active multicasts finish their
acknowledgements; the multicast helpers check for cancellation before issuing a
new transfer. Closing then resets the cores and drains outstanding NoC requests
before releasing host DMA and DRAM memory. Closing again is harmless, and process
exit closes every runtime still running.

Collecting a runtime without closing it doesn't touch the card, since a
collection can interrupt another runtime's access to the shared TLB window. Its
cores keep running and its memory stays reserved until a safe point closes it:
the next runtime on the chip, once Python has collected the dropped one, or an
allocation of DRAM or a hugepage that finds no room, which collects first. A
dropped runtime is usually in a reference cycle through its cached number
tensors, so call `close()` to release it promptly. If a multicast never finishes
or the NoC never drains, closing resets the cores anyway and keeps the runtime's
host page and DRAM for the rest of the process, so a new runtime can still take
over a hung rectangle. If resetting the cores fails, the memory stays reserved
until a later close or takeover succeeds. Cleanup only stops its own rectangle.

Custom kernels must use acknowledged reads, writes and atomics, and wait for
completion before returning. Unacknowledged writes have no completion counter;
shutdown cannot establish their destination memory's lifetime. Custom multicast
code must drain earlier requests, call `mcast_begin()` before issuing any packets,
and call `mcast_done(expected_acks)` after issuing all packets. Only one core per
tile may issue requests during that region. The dispatcher and generated kernels
already use these helpers. Raw custom multicasts bypass cancellation tracking and
can leave cleanup unable to establish completion.

Some kernels need a rectangular grid of workers. `Runtime.grid` keeps rows that
contain every x coordinate still present among workers. In a multi-row runtime,
this excludes the dispatcher's incomplete row. A single-row runtime keeps its
shortened row, excluding the dispatcher's x coordinate. With 120 tiles, you get
119 workers and a grid of 108, in nine rows of twelve. `Runtime.readers` assigns
the grid workers to DRAM banks and ports. It spreads the workers across banks
and prefers ports with shorter return paths on NoC0.

## Local memory

We divide a worker's L1 like this. The constants are in `runtime.py`.

| Address | Use on a worker |
| --- | --- |
| `0x00000` | Firmware and entry jumps for BRISC and the three TRISCs |
| `0x01000` | Mailbox: launch, kernel, arguments, worker count, completion and worker identity |
| `0x01020` | TRISC go and completion phases, alternating per worker |
| `0x01030` | Cancellation request and multicast-active flag, also used by the dispatcher |
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
worker firmware clears a kernel's `.bss` at launch. Initialized statics must be
const. A static pointer to data or a function still contains an absolute address,
so the compiler rejects it: use constant integer offsets into an array instead.
The compiler checks the linked ELF relocations before caching the flat image.

## Commands and memory

A command has a 16-byte header: length, kernel address, argument length and worker
count. Arguments follow, and the command is padded to 64 bytes. Commands never
straddle the end of the host ring. A zero length marks unused space at the end.

`Runtime.send()` writes commands into the ring and updates the doorbell. A block
of recorded commands uses one doorbell write when it fits with any wrap padding.
Larger blocks are sent in pieces, split between commands. The dispatcher executes
launches in order and alternates argument buffers so it can prepare the next
launch while workers run. Work counts range from 1 through `2**32 - 1`; workers
receive contiguous partitions whose total is exactly the requested count.
Launch numbers wrap in the device status, while host launch numbers keep growing.

Each worker reports completion with a NoC atomic increment. The dispatcher
writes timing data before completion status, using ordered acknowledged writes.
`Runtime.wait()` polls that status in host memory. `Runtime.wait(launch=n)` waits
only until launch `n` is done.

The rest of the hugepage, `Runtime.back`, holds tensor data that crosses PCIe.
`tobytes` launches workers that write a tensor's tiles there. `frombytes` writes
small tensors into DRAM through the BAR, and from `tensor.DMA_UPLOAD` bytes on
it stages each bank's share there for `upload_kernel`, whose workers read it
through the PCIe tile. It stages in two halves, filling one while the chip reads
the other, and waits for the last chunk before it returns. A recording leaves
those launches out, since their staged data is gone by the time it replays.
`Tensor.zeros` sends nothing over PCIe: its workers write the 4 KiB tile of zeros
in their L1 over the tensor's storage.

Runtimes on one `Chip` share the free ranges of its eight DRAM banks. An
allocation uses the same address range in every bank, taking the first hole
that's big enough and rounding to 64 bytes. `rt.holes` lists chip-wide free
ranges; `rt.used` lists that runtime's live and deferred allocations. When you
free a tensor, queued kernels may still be using its memory. The runtime waits
for them to finish before reusing that space. A recording keeps its memory for
later calls.

## Recording repeated work

`tensor.jit(f)` runs the first call normally and records the second. Each input
is recorded where it is, and its address in the launches is a `runtime.Bound`
argument. Later calls resend the launches with the new inputs' addresses patched
in, so a replay copies nothing and reads and writes the inputs as a normal call
would, including inputs passed twice or written in place with `assign`, `put` or
`realize(into=...)`. A replay copies an input first only when it is laid out
otherwise than a new tensor of its shape, such as a slice or a repeat, or when
it is storage that the recording's own results overwrite; a written copy goes
back to the caller's tensor afterwards. A tensor by columns can't be copied, so
it must keep the recorded layout. If kernels have been evicted, the recording
reloads them and rebuilds its commands.

Keep the number of inputs, their runtime, shapes and dtypes fixed from the first
call. A failed call can be retried; a failed recording releases its abandoned
allocations. Deleting a JIT callable releases its recording's allocations once
queued work finishes. Live tensors and other recordings still keep their
storage. `Recording.close()` releases a manual recording explicitly; reusing its
context replaces the old launches and cached commands. `Recording.send(binding)`
gives Bound arguments other values for one replay. Recordings on the same
runtime cannot nest or replay inside another recording. The function must not
read tensors back to the host or depend on changing Python state. Results reuse
the same buffers, so a later call overwrites an earlier result. Copy a result if
you need to keep it.

Decode works this way because a token's position is tensor data. Moving to the
next position doesn't require a new shape or a new compiled program.

## Profiling

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

The script runs as it would under `python3`. If it creates a runtime, the wrapper
writes `viz.html` in the current directory when the script returns or raises an
exception, including Ctrl-C. The page includes each runtime's completed launches.
From Python, call `viz.save(path, rt, since=saved)`. Call `rt.wait()` first to
include all queued work. `save` doesn't wait, so you can also call it after a
launch times out.

Open the generated file in a browser with JavaScript enabled. File previews
that don't run scripts can't display the launch data. The source file
`baretile/viz.html` is a template and contains no launch data.

The timeline shows each launch from `go` to its last report, using the
dispatcher's clock. Scroll to zoom, drag to pan, and double-click to show the
whole timeline. The table sorts kernels by total cycles. Input shapes help
distinguish fused kernels that share a name such as `C matmul`.
Tab to a kernel button and press Enter or Space to select it. With the timeline
focused, Left and Right select launches, plus and minus zoom, and Home resets
the view. The page announces the selected launch.

Select a kernel to highlight its launches and see its details. For a fused
kernel, the page draws the compiled expression with inputs at the top. Each
distinct subexpression appears once in the graph. Reductions can still compute
that subexpression on several passes. Comparing graphs can reveal repeated
matmuls, but the page doesn't track shared tensors between kernels. A kernel
made by `Tensor.run` shows its cache key. A kernel absent from the tensor cache,
such as a custom C kernel, shows only its launch statistics.

The runtime keeps the last 1,024 launch records. The page includes the completed
launches still in that history. Launch arguments are plain numbers, so the
runtime can't tell which are the DRAM addresses a launch reads or writes.
