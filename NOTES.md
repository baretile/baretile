# Notes

How baretile drives a Blackhole, from the PCIe BARs up: the chip, the runtime, the compiler, and the models on top. Also
where the facts about the chip come from, how to bring up a card, and what we don't know yet. The facts the code uses
live in `baretile/bh.py`.

Sources are abbreviated: **ISA** is tt-isa-documentation/BlackholeA0, **sim** is ttsim's `src/`, **kmd** is tt-kmd, and
**umd** is tt-umd.

## The chip from the host

### PCIe function

The card is vendor 0x1E52, device 0xB140 (sim `pcie.cpp`, kmd). It has three 64-bit prefetchable BARs (ISA
`PCIExpressTile/README.md`):

| BAR | Size | Contents |
|---|---|---|
| BAR0 | 512 MiB | `0x0000_0000-0x193F_FFFF`: 202 TLB windows of 2 MiB (TLBs 0-201). TLBs 0-31 can do strided multicast, and 201 belongs to the kernel driver |
| | | `0x1FC0_0000`: TLB config, three u32s per TLB for TLBs 0-209, then `u32 strided[32]` at `0x1FC0_09D8` |
| | | `0x1FD0_4000` and `0x1FD1_4000`: the PCIe tile's NoC0 and NoC1 NIU registers |
| | | `0x1FE0_0000-0x1FFF_FFFF`: ARC on PCIe 1. It's reserved on PCIe 0, which is the one a p150 uses |
| BAR2 | 1 MiB | the PCIe controller (DBI): eDMA at `0x100-0x1A8`, iATU at `0x1000` (outbound region r at `0x1000 + 0x200 r`). The ISA doesn't cover it, so the offsets come from kmd `blackhole.c`, umd and sim |
| BAR4 | 32 GiB | 8 TLB windows of 4 GiB (TLBs 202-209) |

### NoC

The grid is 17 x 12. In raw NoC0 coordinates (ISA `Diagrams/Src/NoC.lua`, sim `tile.cpp`), DRAM is at x=0 (banks 0-3)
and x=9 (banks 4-7). ARC is at (8,0), PCIe 0 at (2,0) and PCIe 1 at (11,0). Ethernet is on y=1, and Tensix is at x=1..7
and 10..16, y=2..11. NoC1 numbers the grid from the opposite corner, so (x, y) on NoC0 is (16-x, 11-y) on NoC1.

At boot, firmware turns on **coordinate translation** (`NIU_CFG_0` bit 14) and programs the tiles' tables, which moves
the harvested Tensix columns to the highest x (ISA `NoC/Coordinates.md`). baretile uses translated coordinates
everywhere:

| Translated (x, y) | What |
|---|---|
| x 1..7 and 10..14, y 2..11 | Tensix. A p150 has 120 of them. From firmware 19.5 on, x 15 and 16 are its 2 harvested columns (umd). Before 19.5, all 140 run |
| x 17..18, y 12..23 | DRAM. Port p of bank b is at (17 + b//4, 12 + 3(b%4) + p), for p in 0..2. The three ports reach the same 4 GiB |
| (19, 24) | the PCIe tile that's connected to the host |
| x 20..31, y 25 | Ethernet |
| y 0 and 1 | x isn't translated here, so ARC stays at (8, 0) |

A NoC address is (x, y, a 64-bit address inside that tile).

You can read the translation tables. The PCIe tile's NIU has them at BAR0 `0x1FD04118` (x) and `0x1FD04130` (y): 32
entries of 5 bits, six to a word. Each entry is the raw NoC0 coordinate that a translated coordinate maps to (ISA
`NoC/Coordinates.md`). `Chip` reads them at startup, so `chip.raw` knows where the workers and DRAM ports sit on the raw
grid. `test_raw_places` checks the Tensix tiles against their own NIUs' `NOC_NODE_ID` at `0xFFB2_0044`, which holds raw
x | y << 6, followed by the grid's size, 17 and 12.

The good columns get translated x values in their raw order, and the harvested columns get the highest ones. ttsim's
p150 has raw columns 7 and 10 harvested, so translated x 7 is raw x 11. The three ports of bank b (umd's
`DRAM_CORES_NOC0`) sit in raw column 0 for banks 0-3 and column 9 for banks 4-7. Their rows, by b % 4, are 0, 1 and 11,
then 2, 10 and 3, then 9, 4 and 8, then 5, 7 and 6. Between them, the two columns have a bank port in every row.

Each NoC is a torus (ISA `NoC/README.md`, `RoutingPaths.md`). NoC0's links run toward higher x and y and wrap around,
and its packets travel along x first, then y. NoC1's links run the other way, and its packets go along y first. A
response comes back on the NoC that carried its request. A read's round trip therefore goes all the way around each axis
where the two tiles differ, however close they are. Distance decides how many of those links carry the data, and the
request takes the rest.

`bh.hops(a, b)` counts NoC0's links from raw position a to b, and NoC1's count is `hops(b, a)`. On NoC0, the DRAM port
whose data reaches a worker in the fewest links is in the DRAM column to its left, in its own row. On NoC1, it's the one
to its right.

### Inside a Tensix tile

This is the address map as the tile's own cores see it (ISA `TensixTile/BabyRISCV/README.md`):

| Address | What | Over NoC |
|---|---|---|
| `0x0000_0000-0x0017_FFFF` | L1, 1.5 MiB. Code can't run from anywhere else | yes |
| `0xFFB0_0000` | the core's private data RAM, which holds the stack: 8 KiB for BRISC and NCRISC, 4 KiB for a TRISC | no |
| `0xFFB1_2000-0xFFB1_2FFF` | tile control and debug registers, like soft reset and the reset PCs | yes |
| `0xFFB1_3138-0xFFB1_314B` | PC snapshots of the cores | yes |
| `0xFFB2_0000`, `0xFFB3_0000` | the NoC0 and NoC1 NIU registers | yes |
| `0xFFE0_0000`... | the Tensix coprocessor: GPRs, instruction push, mailboxes, config | no |

MMIO registers must be accessed 4 bytes at a time, from either side.

### DRAM and host memory

There are eight banks of 4 GiB. The address inside a DRAM tile is the byte offset in its bank (umd soc descriptor), and
ttsim models 4080 MiB per bank. GDDR has to be trained before it's used. Telemetry tag 22 has two bits per bank, trained
and error. umd waits up to 300 s for training and retrains with ARC message `0xB6`.

The device reaches host memory by sending NoC requests to the PCIe tile at (19, 24) (ISA `PCIExpressTile/README.md`). An
address below 2^58 goes to the host as a bus address, and bits 58 and 59 mean relaxed ordering and no-snoop. With bit 60
set, the request goes through the outbound iATU instead. `0xF800_0000_0000_0000` is the DBI. A host buffer at bus
address B is then NoC address (19, 24, B), and baretile doesn't touch the iATU. kmd's DMA-mask comment agrees.

kmd and umd send pinned memory through the bit-60 window instead, so our direct path depends on firmware having set up
window 0 after each boot. If that doesn't hold on a card, we fall back to what kmd does. Program outbound region r at
BAR2 + `0x1000 + 0x200 r`: the base at +`0x08`/`0x0C`, the limit at +`0x10`/`0x20`, the target at +`0x14`/`0x18`, CTRL_1
= `1 << 13` and CTRL_2 = `1 << 31` (kmd `blackhole.c` 86-120 and 878-911). Then set bit 60 on the bus address. ttsim
models that path too (`translate_pci_dma_addr`).

### Ethernet links

`Chip.links`, `Chip.send`, `Tensor.to` and `link_kernel` use the Ethernet links. `test_two_chips` tests them on ttsim's
two-chip build (`bh_x2`, the two chips of a P300). The sources are tt-isa-documentation's `BlackholeA0/EthernetTile/`
and its `ethdump` sample, tt-metal's `eth_fw_api.h`, `dev_mem_map.h` and `tunneling.h`, and tt-umd's topology discovery.

- There are 14 Ethernet tiles on NoC0 at y=1. Each has 512 KiB of L1, two RISC-V cores and a 400 GbE link. A p150 has 8
  with ports (telemetry tag 35), at translated x 24-31, y 25.
- E0 runs Tenstorrent's firmware, which trains and retrains the link. It uses the top 64 KiB of L1. Its boot results are
  at 0x7CC00, the port's status is at 0x7CC04 (1 when the link is up), and the peer's identity is at 0x7CFE0. Don't
  reset E0. E1 is free, and bit 0x1000 of `SOFT_RESET_0` holds it in reset.
- A transmit queue is a set of registers at 0xFFB90000 + 0x1000 n. You write the source, size and destination, all
  16-byte aligned, then `CMD` 2, which writes into the peer tile's L1. Packets arrive once and in order, so a flag sent
  after the data lands after it.
- The host can write those registers over the NoC, so `link_kernel` copies DRAM to DRAM without running any code on the
  Ethernet tiles. It uses a worker on each side, two slots in the Ethernet tile's L1, a flag per chunk, and a credit
  sent back.
- A link does about 50 GB/s each way, and a hop costs 0.5 us (tt-metal's numbers). umd calls the links flaky, so poll
  their status, and don't wait on them without a timeout.

### TLB windows

A TLB maps a range of BAR0 (or BAR4) onto a range of a tile's NoC address space. Its config is 96 bits at `BAR0 +
0x1FC0_0000 + 12 i`, written as three 32-bit uncached stores, since odd TLBs aren't 8-byte aligned. The window's address
is aligned to its size (ISA `PCIExpressTile/HostToDeviceTLBs.md`, sim, umd). For 2 MiB TLBs:

| Bits | Field |
|---|---|
| 0-42 | NoC address >> 21 |
| 43-48, 49-54 | x_end, y_end (the target, or the far corner of a multicast rectangle) |
| 55-60, 61-66 | x_start, y_start (multicast) |
| 67-68 | noc |
| 69 | multicast |
| 70-71 | ordering: 0 relaxed, 1 strict AXI, 2 posted writes, 3 counted writes |
| 72 | linked (never safe) |
| 73, 75, 76-77 | static VC enable, buddy, VC class |

For 4 GiB TLBs, address >> 32 is in bits 0-31, then x_end at bit 32, y_end at 38, x_start at 44, y_start at 50, noc at
56, mcast at 58, ordering at 59, linked at 61 and static VC at 62.

A read+write window on a static VC "crashes the host" (umd `architecture_tlbs.cpp`), so unicast uses a dynamic VC.
baretile keeps a shared window (TLB 0, strict ordering) and moves it when an access crosses into another 2 MiB block or
another tile. Each runtime's doorbell gets a TLB of its own. Multicast sets x_start and y_start, mcast and static VC
class 2, like umd does.

### Reset, and what the cores run

Each Tensix tile has `RISCV_DEBUG_REG_SOFT_RESET_0` at `0xFFB1_21B0` (ISA `TensixTile/SoftReset.md`). Use
read-modify-write on it. A set bit holds a core in reset: bit 11 is BRISC, bits 12-14 are TRISC0-2, and bit 18 is
NCRISC. At power-on it's `0x47800`, which holds all five (sim). umd releases with `0x80047000`, where bit 31 means a
staggered start and only matters for power.

When a core leaves reset, it invalidates its instruction cache, zeroes its GPRs and sets its PC. It also zeroes its data
RAM over up to 2048 cycles, unless `0xFFB1_2224` says not to. Load code while the core is still held.

There's no boot ROM. A released core fetches from L1 at its reset PC. For BRISC that's `0x0`. TRISC0, 1 and 2 start at
`0x6000`, `0xA000` and `0xE000`, or at `0xFFB1_2228`, `222C` and `2230` if bit n of `0xFFB1_2234` is set. NCRISC starts
at `0x12000`, or at `0xFFB1_2238` if bit 0 of `0xFFB1_223C` is set.

The cores are rv32im with Zicsr, Zaamo (no lr/sc), Zba, Zbb and part of F. There's no C extension, because `.ttinsn`
lives in its encoding space. baretile compiles with `clang --target=riscv32-unknown-elf -march=rv32im` and links a flat
binary with lld. The gotchas (ISA `BabyRISCV/`):

- `mhartid` reads 0 on all the cores, and ttsim kills the process if you read it. Tell cores apart by their entry point.
- `fence.i` doesn't flush the instruction cache, and ttsim rejects it. New code needs a release from reset, or a write
  to `RISCV_IC_INVALIDATE` in Tensix config.
- The L0 data cache (4 x 16 B) isn't coherent with NoC or host writes. When you poll L1, put a `fence` in the loop.
- Stores to one region (L1) land in order. Across regions a `fence` isn't enough: store, load the value back, use it,
  and then signal.
- Unsupported instructions are undefined behavior. They don't trap.

ARC runs the board firmware, and it's what set up translation and the PCIe windows. Ethernet core E0 runs Tenstorrent's
Ethernet firmware.

### Doing tt-kmd's job

| kmd | baretile (`pci.Card`, `Chip`) |
|---|---|
| enables the device, sets a 58-bit DMA mask and bus mastering, sets Max Read Request Size to 4096 | unbinds kmd, writes sysfs `enable`, sets memory and bus master in `PCI_COMMAND`, sets MRRS to 4096 |
| maps BAR0, BAR2 and BAR4, and reserves TLB 201 | mmaps sysfs `resource0` and reads and writes it a word at a time. libc's memcpy might go a byte at a time on uncached memory, so we don't use it. It zeroes the strided multicast registers. The TLBs other than 201 are ours |
| waits for ARC: `0x80030408` bit 0 (umd checks `& 7 == 5`, for up to 300 s after a reset) | waits for `& 7 == 5` |
| sends ARC `0xA0` (ASIC_STATE0). Power policy `0x21` turns everything off when no fd is open, which **clock-gates Tensix**, and sets `0x7FFE0F21` when one is opened | sends `0xA0`, then `0x7FFF0F21` (everything on, max AI clock), then `0x52` (AICLK go busy). Then it waits up to 0.2 s for telemetry to show the AI clock within 5% of 1,350 MHz, like umd does |
| pins host memory and programs an outbound iATU region | locks a 2 MiB hugepage, gets its bus address from `/proc/self/pagemap` (the IOMMU has to be off or in passthrough), and uses it as a NoC address |
| resets: sets `PCI_COMMAND` bit 6 as a marker, writes config `0x934 = 1` then `0x930 = 0x11`, waits for the device and firmware, and restores config space | `pci.reset` (`python3 -m baretile.chip reset`) does the same writes through sysfs `config`, then the kernel's remove and rescan. Untested |
| reads telemetry: tags at `*0x80030434`, values at `*0x80030430`, tag word `id \| index << 16` | `Chip.telemetry`. The Tensix columns come from tag 34, and GDDR training from tags 22 and 36 (it waits up to 300 s) |
| "NoC cleanup": puts cores back in reset when a process dies | a `Runtime` halts its cores at exit, and before it takes its hugepage |

**ARC messages** (kmd `msgqueue.c`, umd). `*0x8003042C` points at a control block. Word 0 is the base of 4 queues, and
word 1 is `entries | queues << 8`. Queue i (3 is the one for applications) starts with a 32-byte header: the request
wptr at +0, the response rptr at +4, the request rptr at +0x10 and the response wptr at +0x14. Pointers wrap at 2N.
After the header come N request slots and N response slots of 32 bytes each (`u32 header, u32 payload[7]`).

To send a message, write a slot, bump the request wptr, and poke ARC by writing `0x10000` to `0x80030100`. Then poll the
response wptr for up to 1 s. A status below 240 in the low byte means OK. ttsim answers kmd's and umd's messages and
kills the process on a code it doesn't know. It returns zeros, though, so `0x57`, the harvesting message, is useless
there.

## ttsim

ttsim runs Tensix instructions as soon as it can, in zero time. It ignores `STALLWAIT`'s busy conditions and answers NoC
and DRAM requests immediately. In a clock, each tile steps its RISC-V cores once and runs whatever Tensix work is ready.
A kernel's cycle count on ttsim is therefore the RISC-V instructions on its critical path. That's useful for comparing
versions of our code, and it tells you nothing about silicon. `pci.Sim.tick` runs 10 clocks per poll, and ttsim kills
the process on any contract violation.

You can see what a wait protects by breaking it on purpose. Break the pack's `SEMWAIT`, the math thread's wait for a Dst
half, or the SEM_L1 wait, and ttsim stops on a semaphore. Break BRISC's wait for `UNPACKED` or TRISC2's wait for `SENT`,
and the results come out wrong. ttsim can't check some waits: the `STALLWAIT` before a report, TRISC2's wait for
`READY`, the `tensix_idle` before it, and the accumulator semaphores. We'll find out on silicon whether those are right.

**ttsim-chaos** (`test/ttsim-chaos.patch`, against ttsim v1.11.5 at 464814f) adds the timing noise that ttsim leaves
out. The parts are off unless you set them:

- `TTSIM_CHAOS_NOC=lo,hi` delays NoC requests from Tensix tiles by `lo` to `hi` clocks. A delayed request counts as
  outstanding, and static-VC requests keep their order.
- `TTSIM_CHAOS_TENSIX=p` and `TTSIM_CHAOS_RISCV=p` make a pipe or a core skip a turn with probability p.
- `TTSIM_CHAOS_SEED=n` sets the seed.
- `TTSIM_CHAOS_UNHARVESTED=1` runs all 140 Tensix of a p150, the way firmware before 19.5 does.

The patch also models the NIU's read-data counter (counter 3, a flit for each 64 bytes a read brings in), and exports
`libttsim_chaos` so that `pci.Sim` knows the counter is there (`counts`). The suite passes with any of them set. That
checks the waits for a frame's reads and a matmul step's reads, and that TRISC0 doesn't push a 16th report before
relaying one.

## The runtime

This is `baretile/runtime.py`.

**Cores.** `Runtime(chip)` takes the chip's Tensix cores, or the rectangle from `start` to `end` in translated
coordinates. Its last core runs the **dispatcher**, and the rest are **workers**. The workers also form a grid of
complete NoC rows (`Runtime.grid`) for kernels that multicast along rows and columns. On a p150's 120 cores, the grid is
9 rows of 12.

A new Runtime that overlaps an older one stops it and halts its cores, including the ones it doesn't take. Cores idling
in firmware slow ttsim down, because it simulates every core that's running.

Grid workers get a DRAM bank and a port to read it through (`Runtime.readers`), for kernels where workers read from a
bank each. The runtime sorts the (worker, port) pairs by how many NoC0 links the data crosses on its way back, fewest
first. It takes a pair if the worker has no port yet, the bank has fewer readers than its share of the grid (14 or 13 of
108), and the port has fewer than a third of that (5). That spreads the reads out, so no row's links carry too many of
them. On ttsim's p150, a reader's data crosses 5.8 links on average, against 13.75 for a port picked at random.

**A worker's L1:**

| Address | What |
|---|---|
| `0x0` | firmware: a jump for each core, BRISC's at `0x0` and TRISC i's at `0x4 + 4i` |
| `0x1000` | the mailbox: `[go, kernel, arguments, workers] [done, the dispatcher's x \| y << 6, scratch, index]` |
| `0x1020` | the TRISCs' `[go, done 0, done 1, done 2]` |
| `0x1040` | `SYNC`: 64 bytes for a kernel's cores to talk through. The first 8 words are zeroed before each launch |
| `0x1080` | `FLAGS`: 32 words that workers use to signal each other over the NoC. Workers zero their own at the end |
| `0x1100` | two argument buffers. Launches alternate between them |
| `0x2000` | loaded kernels, 500 KiB. A kernel starts with four entry offsets, followed by a program for each core |
| `0x7F000` | a 4 KiB tile of zeros, written at boot. It's the compiler's slot 0 |
| `0x80000` | kernel data (`L1_DATA`) |

The dispatcher's tile has its firmware at `0x0` and its config at `0x1000`: `[doorbell, ring lo, hi, ring size, status
lo, hi, workers' rectangle, workers]`. Its outbox is at `0x1400`. The count of worker reports is at `0x80000`, with the
sum of the flits their reads brought in next to it. Commands are staged from `0x80040`.

**Boot** happens once, from the host. It multicasts reset and the worker firmware over the rectangle, then writes the
workers' mailboxes with their index and the dispatcher's coordinates. Then it writes the dispatcher's firmware and
config, points a TLB at the doorbell, sets the TRISC reset PCs, and multicasts the release of the BRISCs and TRISCs.

**Loading kernels** (`Runtime.load`). A kernel is multicast into the workers' L1 once and stays there, and launches name
its address. Images run wherever they land, using clang's medany model with linker relaxation and jump tables turned
off. When the 500 KiB fill up, the next load waits for the queue to drain and evicts everything. A kernel's `.bss` is
cleared at each launch.

**Commands.** A launch is a 16-byte header `[length, kernel address, argument bytes, workers]` followed by the
arguments, 64-byte aligned. Commands go into a ring in host memory, and then the host writes the new write pointer into
the doorbell. That doorbell write is the host's only PCIe access for a launch, and the same goes for a recorded block of
launches (`Runtime.block`). Commands don't wrap around the end of the ring. A length of 0 marks the tail as unused, and
pointers are free-running byte counts.

**The dispatcher** is BRISC firmware, about 1.3 KB of it. It loops:

1. Poll the doorbell. When it has run the commands it has, read what the host wrote since, up to `STAGE` (64 KiB), with
   a NoC read from the PCIe tile, and run those commands from L1.
2. Multicast the arguments to the workers and wait for their acknowledgements.
3. Fire the go: the same 16 bytes from its outbox (launch number, kernel, arguments, workers) into the mailboxes.
4. Poll the report count until the workers have all added to it.
5. Write the launch's times and bytes (`profile`), then `[read pointer, launches done]`, into host memory. These are
   posted writes on the same static VC, so they stay in order, and the host doesn't see a launch finish before its times
   arrive.

While a launch runs, the next command's arguments go into the other buffer.

**Multicast from a core** is a write with `NOC_CMD_BRCST_PACKET`, path reserve and static VC class 2. ttsim insists on
all three. The rectangle goes in `NOC_RET_ADDR_HI` as `end | start << 12`, with `NOC_CMD_BRCST_SRC_INCLUDE` off.
Recipients each send an acknowledgement, and that throws off `NIU_MST_REQS_OUTSTANDING_ID`, which goes up once and down
once per recipient. The sender counts `NIU_MST_WR_ACK_RECEIVED` up to the number of recipients instead, then clears the
outstanding counters (`0xFFFF` at `NIU + 0x60`) while nothing else is in flight.

**A worker** runs firmware for its four cores, about 700 bytes (800 where the NIU counts read data). At boot, BRISC
presets request initiators 1 and 2 (`noc_setup`). Then, for each launch, BRISC:

1. polls `go` with a `fence`, and sits out the launch if its index is past the launch's worker count,
2. takes its share of the units and zeroes `SYNC`'s counters,
3. invalidates the instruction caches (Tensix config 185, bits 0-3, then `l1_settle`),
4. writes the TRISCs' go and runs its own program,
5. waits for its transfers and for the TRISCs to finish,
6. marks the launch done in its mailbox and adds 1 to the dispatcher's count with a NoC atomic increment. The increment
   is posted, on a static VC, like tt-metal does it.

A TRISC polls its go, runs its program, waits until its Tensix thread has run what it pushed (`tensix_idle`), and writes
its done.

**Profiles** (`Runtime.profile`). The dispatcher reads its tile's wall clock (`RISCV_DEBUG_REG_WALL_CLOCK_L`, ISA
`DebugTimestamper.md`) at a launch's go and at its last report. It writes both into a ring of 1024 entries in host
memory, so launches are timed on the chip's own clock.

Where the NIU counts read data, we get bytes too. That's counter 3, `NIU_MST_RD_DATA_WORD_RECEIVED`, in 64-byte flits
(ISA `NoC/Counters.md`). It works on a card and on ttsim-chaos, but released ttsim stops if you read it. Workers read
the counter at go and at done and add the difference to a sum next to the dispatcher's count. They do that with a second
atomic increment on the same static VC, before their report, so the sum is complete once the count is. The dispatcher
records how much the sum grew before the next go. The result is the bytes each launch's reads brought in, from DRAM,
from other tiles' L1, or from copies inside a tile. Launches are named after their ops, and `profile(since)` sums them
by name.

**Replay** (`tensor.jit`). The first call runs normally. The second call is recorded: its launches go into
`Runtime.log`, and the DRAM it allocates stays allocated (`Runtime.pinned`). After that, a call copies its inputs into
the recorded call's inputs and sends the recorded launches as a block, with one doorbell write. Nothing gets built or
compiled.

For this to work, the function needs fixed input shapes and dtypes, can't depend on anything else that changes, and
can't read anything back. Its result and the tensors it keeps are the same tensors from call to call. A decode step
replays like this, because a token's position is data, not a shape. On the host, a step is the input copies and a write
to the ring.

**Waiting** (`Runtime.wait`) polls the status words in host memory rather than the chip. The timeout counts from the
last launch that finished. The host waits when it reads something back, and a readback is a launch like the others
(`readback_kernel`) that copies tiles into host memory. A tensor with fewer than 32 rows sends back its rows and not the
padding. On a timeout, the error names the workers that haven't finished. For the first four it also gives the cores,
their `SYNC` counters and `FLAGS`, their NoC requests in flight, and on a card their PCs (`Runtime.report`).

**DRAM** (`dram_alloc`, `dram_free`). A tensor takes the same address range in all 8 banks, with tile i in bank i % 8.
Allocation is first fit and 64-byte aligned. A `Tensor` frees its DRAM in `__del__`, but the memory comes back after the
launches queued before the free have finished.

**The device API** (`DEVICE`) is prepended to the device programs: `noc_me`, `noc_read`, `noc_write`, `noc_get`,
`noc_put`, `noc_wait`, `l1_settle`, `mcast_done` (waits for a multicast's acknowledgements), `tensix_idle` (TRISC only)
and `CLOCK` (the tile's wall clock). `noc_get` and `noc_put` use the preset initiators 1 and 2, and write the four
registers that change, plus the go. The rules come from ISA `NoC/MemoryMap.md` and `NoC/Counters.md`, and ttsim enforces
them:

- A request moves at most 16 KiB. Source and destination have to match in their low 4 bits for L1, and in their low 6
  bits when DRAM is involved.
- Reads must be acknowledged. Inline writes to L1 are broken on Blackhole.
- Wait for `NOC_CMD_CTRL` bit 0 before you touch an initiator, and read it back before you read counters.
- Before the NIU reads something a core just stored, do `fence`, load, `fence` (`l1_settle`). After data lands, `fence`
  before you read it. A `fence` is also a compiler barrier.

Not built: commands other than launches, NCRISC (it stays in reset), and a barrier between the host's ring writes and
the doorbell, which x86 keeps in order anyway.

## The compiler

`baretile/tensix.py` emits Tensix instructions and the four programs that make up a kernel. `baretile/kernels.py` has
the kernels written with them, and `baretile/tensor.py` builds the trees of ops that the kernels compute.

A few words we use a lot. A kernel works through *units*, where a unit is a result tile, a block of them, or a line of
tiles along an axis that's being reduced. A unit takes one or more *passes* over its tiles. For each tile, a *frame* of
L1 *slots* holds the inputs plus a slot for each op (its *intermediates*). Two frames alternate.

**Who drives the Tensix.** There are three threads, which are in-order instruction FIFOs. Any thread can issue to any
unit. TRISC t pushes to thread t. BRISC only moves data, and if BRISC and a TRISC push in the same cycle, the TRISC's
push gets dropped.

An instruction is `opcode << 24 | fields` (ISA `Diagrams/Src/Bits32.lua`, and ttsim's `tensix_isa.json` names the
fields). A TRISC pushes an instruction by executing it: `.ttinsn` is the word rotated left by 2, written into the code
(`tensix.tt`). If the FIFO is full, the TRISC stalls. A word that changes at run time, like a Dst half or a slot,
becomes a store of the word plus a C value (`field`). What a `REPLAY` records goes through stores too, because fused
`.ttinsn`s must not be recorded.

**Configuration.** There are three kinds, and you need all of them:

- Backend config is 32-bit words at `0xFFEF_0000`. The indices come from ttsim's `tensix_regs.json`, and we still have
  to check them on silicon.
- Thread config is 16-bit and set with `SETC16`, from its own thread. The reset values aren't documented, so we write
  all of them.
- The threads have their own address counters, set with `SETADC*` and `SETRWC`.

TRISC0 writes the backend config, reads it back between fences, sets up the semaphores, waits for its thread
(`tensix_idle`) and sets `READY`. The other TRISCs wait for that.

**Tiles.** A 32x32 tile is four 16x16 faces in row-major order, and each face is row-major too. Element (r, c) is at
`(r//16*2 + c//16)*256 + r%16*16 + c%16`, which is the layout `MVMUL` wants. A tensor's tiles are in row-major tile
order, with tile i in bank i % 8. Tile (i, j) of matrix b is tile `b // rep * bstride + i * stride + j + offset`, which
covers views and repeats.

A matmul's B can be laid out by columns instead (`Tensor.columns`, done by `columns_kernel`). The tiles of column j are
then a run down K in bank j % 8. A block float's run holds its tiles' exponents first, then their values. A bank's
columns follow each other, matrix after matrix. The next n tiles of a column are then one read, or two for a block
float, and they land the way the unpacker wants a region: exponents first, then values.

In that layout, tile (m, j) of matrix b is tile `b * kstride + m` of run `b * runs + j // 8` in its bank. `runs` is how
many runs a matrix has in each bank. `kstride` is for a K that's cut into a batch, with each part further along the same
runs. This layout is read as a matmul's B, in the paths a matmul takes (BRISC's `getc`, `getbc` and `stepc`), and by
`readback_kernel`. Anything else asserts.

**Precision.** Dst is fp32, and the packer rounds to bf16 half up. Four fidelity phases make a bf16 product exact.
`ELWADD` and `ELWSUB` align mantissas first, so about 0.4% of random sums come out a last-place step off. fp32 would
lose 13 bits in SrcA and SrcB, so fp32 tensors get unpacked into Dst and computed on the SFPU instead.

An fp32 tile unpacked into SrcA has to be marked TF32 (format 4), and ttsim drops its low 13 bits. This is how a partial
sum comes back into Dst. When a matmul's K is split into parts, the packer adds each part onto a sum it keeps in L1 in
fp32 (`tensix.accumulating`, which attention's output uses too). The sum is then reloaded into Dst by adding it to zeros
(`tensix.reload`, tested by `test_accumulate_parts`). Doing it without TF32's rounding would mean unpacking the sum into
Dst next to bf16 tiles. That needs a separate format for each unpacker context (`Ovrd_data_format`), which ttsim doesn't
model.

**Where the ISA and ttsim disagree.** `UNPACR` bit 7 is `MultiContextMode` in the ISA and `ovrd_thread_id` in ttsim, and
we set it to 1 either way. Bit 0 is ttsim's `last`, which must be 1. `SETDVALID` isn't supported, so we use
`UNPACR_NOP`. `addr_mode` is 3 bits on Blackhole according to ttsim, and the ISA's pages describe Wormhole's.

### The pipeline

The threads have separate jobs, so unpack, math and pack work on different tiles at the same time. The unpackers' Z
counter and the packer's channel-1 Y counter pick an L1 tile slot, so moving between slots doesn't touch config.

| Thread | For each tile (or matmul step) |
|---|---|
| 0, unpack | `SETADC` each unpacker's Z to its slot, then `UNPACR` into SrcA and SrcB. Report each frame or step (`SEMPOST` SEM_UNPACKED) |
| 1, math | wait for a free Dst half (`SEMWAIT`), aim at it, and zero it if the op accumulates. Run the FPU program, then `SEMPOST` SEM_DST |
| 2, pack | `SEMWAIT` SEM_DST, aim at the Dst half and the slot, do 16 `PACR`s, `SEMGET` SEM_DST, then `SEMPOST` SEM_L1 for an intermediate or SEM_PACKED for a result |

SrcA and SrcB have two banks each, and the hardware hands those over between threads. SEM_DST counts the results waiting
in Dst, up to its two halves. Before an op reads an intermediate, it takes a SEM_L1 for each op that's been packed since
it last looked (`drain`).

BRISC can't read semaphores, so TRISC0 and TRISC2 relay theirs into words of `SYNC`. Each word is written by one core
and read after a `fence`. A semaphore holds at most 15, and posting to a full one breaks the contract, so neither TRISC
pushes a 16th report before it relays one.

| Word | Written by | Means |
|---|---|---|
| `FETCHED` | BRISC | frames fetched |
| `ISSUED` | BRISC | matmul steps whose reads are out |
| `UNPACKED` | TRISC0 | reports relayed, so a slot can be refilled |
| `READY` | TRISC0 | config and semaphores are set up |
| `PACKED`, `RING` | TRISC2 | results packed, and where they are and where they go |
| `SENT` | BRISC | results sent home |

The four programs run the same loops over units, passes and tiles, and do their own part of each tile. A wait loop keeps
doing its core's other work while it waits: BRISC keeps sending, and TRISC0 and TRISC2 keep relaying. That way a core
can't end up waiting on a core that's waiting on it.

The threads have 32-entry replay buffers for the runs that repeat: a fidelity phase of 16 `MVMUL`s, an SFPU body, or 15
`PACR`s. An unpacker sees the low 8 bits of its Z (ISA `ADCs.md`), so what an unpacker reads has to sit below slot 256
of its base. A block-float B and a token's long row of A live past that (see Matmul), and the base moves to them.

### Matmul

`MVMUL` does `Dst[8x16] += SrcB[8x16] @ SrcA[16x16]`, so A goes to SrcB (unpacker 1) and B goes to SrcA (unpacker 0). A
pair of tiles takes 8 face products of 2 `MVMUL`s each, 16 per fidelity phase. Four address modes (slots 4-7) walk
through them, and their counters wrap at 64. A result tile is K steps into a Dst half, which sums in fp32.

**What stays in L1.** A's row stays in L1 as long as a worker's result tiles stay in that row. When the kernel has one
matmul and no passes, a unit is a block of up to 4 x 4 result tiles, with A's rows and B's current column kept in L1.

**Sharing tiles between workers** (`Runtime.grid`). A matmul with more than one tile row gives grid worker (gy, gx) a
block of the result. Workers in a grid row read their share of A's rows and multicast it along the row. Workers in a
grid column read their share of B's columns and multicast it down the column, into one of three buffers. A flag in
`FLAGS` follows each share.

Three buffers mean nobody has to send a "buffer free" message. A worker posts column c after its own unpacker is done
with column c - 2. By then it has all the shares of c - 1, and each of those was posted by a worker that was done with c
- 3, the column that buffer held.

When A's rows and B's buffers don't fit below slot 256, the kernel splits K into parts (`grid_parts`, which picks the
same split for any format of B). The parts' A rows rotate through three buffers of their own, by the same rule. After
each part but the last, the worker packs its result tiles onto fp32 accumulators in L1 (`tensix.accumulating`). The last
part starts each tile from its accumulator (`tensix.reload`, as TF32) and runs the rest of the expression on the
complete sum.

If a worker's accumulators don't fit in L1, its block goes a group of columns at a time, and each group goes through K's
parts. A's rows are sent again for each group, at the start of each part, by all the workers in the grid row, including
ones with no columns in that group. B's buffers and flags are numbered by a column's position in the block, so a block
that's cut short at the matrix's edge leaves no gaps. Input tiles are read from DRAM once, except A's, which are read
once per group. A B laid out by columns comes in as each share's runs, in reads of up to 16 KiB (`getc` and `getbc`,
which read a block float's exponents and values separately).

**A token's row** (an A that's one tile row). The grid workers take blocks of the result's columns. The first grid row
reads A in shares and multicasts it to the others, and B streams in. When there are too few columns to keep the workers
busy, K is cut into 2, 4 or 8 parts as a batch. The parts are rounded to bf16, and the kernel that reads the parts sums
them. That works for readers that take tile-sized slices of the sum too, like the two halves of a GLU.

Some rows are too long for the slots below 256, like a decode step's down projection. They stay past slot 256 while L1
has room, which is about 480 tiles of K. Unpacker 1's base moves from one region of 256 slots to the next within a
tile's steps.

Usually B's columns go by bank. That happens when B is one matrix and not a batch, its width is a multiple of 8 tiles,
and B isn't transposed or picked by index. Column j of B is in bank (j + its first tile) mod 8, as long as its row
stride is a multiple of 8. A weight's always is, because its width is a multiple of 256. If B is laid out by columns,
column j is in bank j % 8 for any width.

Grid workers take the columns in their reader's bank, in turns with that bank's other readers. The four programs share a
`reader` table for this: the port, the bank, the worker's position among the bank's readers, and how many readers there
are. A worker reads its bank through its reader's port. If a view's stride breaks the rule, the other reads go through
port 0. The point of all this is for a bank's reads to come from workers near it, over its three ports. Without it, the
banks get read from all over the grid through port 0, and two banks share a row's links. The probe measures grid rows
read both ways, interleaved and by columns from each worker's own bank, which will tell us what this is worth on a card.

A B laid out by columns streams in chunks of a column's tiles (`chunk_tiles`). A chunk is 8 tiles for bf16 and 16 for a
block float, so a ring of `CHUNKS` chunks is at most a 64-slot region. A chunk is a read of up to 16 KiB, or two for a
block float (exponents, then values). A chunk's reads share a transaction ID, and `CHUNKS` (4) are in flight, in a ring
of their own. For a bf16 B, that ring sits past the slots the kernel uses, and unpacker 0's base moves to it. BRISC
issues a chunk when the unpacker is done with the chunk that last used that part of the ring (`chunk_part`,
`chunk_issued`). Thread 0 waits for that ID's reads once per chunk (`landed_chunk`).

For a token's 1 x 1024 @ 1024 x 4096, that comes to 544 reads, where reading tile by tile would take 4,128 in bf16 and
8,224 in a block float. A worker has up to 64 KiB in flight in bf16, 68 KiB in bfp8 and 36 KiB in bfp4, compared with 8,
4.3 and 2.3 KiB tile by tile.

**Picking B's matrices on the chip** (`matmul(..., index=)`). Batch entry b is row b // k of A times the matrix of B
that its int32 index picks. BRISC fetches the indices first (up to 2400 tokens per call) and offsets B's tile addresses.
This is a mixture of experts' matmul. The chip chooses the experts, so a replayed step reads the experts that its tokens
chose. An index past the last matrix picks no matrix. The four cores read it and skip that entry's steps, so its result
is a zeroed Dst half, packed, and nothing gets read or multiplied.

**A's row shared by a batch.** Sometimes all the matrices in a batch read the same tile row of A, like a decode step's
tokens going through the experts. If the batch has more entries than the grid has rows, that row is read once, in shares
by the grid's first row, and multicast to the grid before the first tile. A row or column kept in L1 is tagged with its
matrix, so when a batch's entries share one (as a repeat's do), it isn't fetched again.

**Reads in flight** (`DEPTH`, 4). Step k's reads carry NoC transaction ID k % 4 + 1, and B streams through 4 slots.
Thread 0's TRISC waits for that ID's `NIU_MST_REQS_OUTSTANDING_ID` to reach 0 before it unpacks. BRISC reads initiator
1's `CMD` back and fences before it bumps `ISSUED`. The depth of 4 is a guess, and we'll tune it on the card.

**Matmul precision.** The FPU adds groups of 8 products after aligning them to 12 bits (ttsim's `mvmul`), so a matmul is
good to about 2^-11 of its larger terms. An fp32 matmul is bf16x3. The operands are split into hi + lo in bf16, where lo
is the operand minus itself rounded by `ftob`, and the matmul computes hi@hi + hi@lo + lo@hi in fp32.

**Fidelity phases** (`matmul(fidelity=)`, `tensor.FIDELITY`). Phase 0 multiplies the top 7 bits of A by the top 5 bits
of B. Phase 1 does A's top 7 by B's last 3. Phase 2 does A's last bit by B's top 5, and phase 3 does the last bit by the
last 3 (ISA `SrcASrcB.md`). With all four phases, the default, a product is exact. Two phases drop A's last bit, and one
phase keeps 5 bits of B. ttsim charges the same for any number of phases. On silicon, a phase costs an `MVMUL` per
block.

**Block floats** (`cast("bfp8")`, `cast("bfp4")`). A bfp8 tile is 1088 bytes: 64 exponents, one for each run of 16
values (a face's row), then 1024 bytes of values, each a sign and 7 bits of magnitude. A value is `magnitude *
2^(exponent - 133)`. bfp4 has 3 bits of magnitude and two values per byte, for 576 bytes per tile.

The packer casts fp32 to bfp8 (`Pack_out_format` 6, `Exp_section_size` 4) and rounds once, half up. The packer's bfp4
truncates, so bfp4 has a BRISC kernel of its own (`bfp4_kernel`) that rounds to the nearest 3 bits, like Tenstorrent's
host conversion.

The unpacker turns block-float values into the bf16 they stand for on the way into SrcA. A matmul by a block-float B
therefore gives the same bits as the bf16 matmul with the same values. The unpacker finds slot z's exponents at `base +
64 z`, and its values after the region's exponents, starting at `1024 z` (`512 z` for bfp4). A slot's first value has to
be below 65536, which is why a block-float B lives in regions past slot 255 (`B8`) and is read in two pieces per tile
(`getb`). Thread 0 switches unpacker 0 to it around the matmul's steps (`bfp_b`, with `RMWCIB` between `STALLWAIT`s).

`put` writes bf16 rows into a block-float tensor and rounds each run the way the cast does, which is how the KV cache
can be bfp8 or bfp4.

### Fusion and elementwise

**Fusion.** Ops build a tree, and `realize` compiles the tree into one kernel. For each output tile, a frame of slots
holds the inputs and then a slot for each op. Intermediates are packed into L1 and rounded to their dtype, so a fused
chain gives the same bits as launching the ops separately. The result is the only thing written to DRAM. Two frames
alternate, so tile n + 1 is fetched while tile n computes. A matmul's operands are realized first.

A kernel holds at most 15 ops, because SEM_L1 is 4 bits. When a tree is too big, `_fit` realizes part of it first,
preferring a value that two ops read if there's one that fits. A number is a 1 x 1 tensor in its tensor's dtype,
uploaded once per runtime.

**The SFPU** (`tensix.SFPU`) is a 32-lane fp32 vector unit that works on Dst. The math thread runs it after the FPU (ISA
`VectorUnit.md`, ttsim `tensix.cpp`).

- LRegs 0-7 are writable. 9, 10 and 11 are the constants 0.0, 1.0 and -1.0. Writing 12-15 configures load macros, and
  LReg 15 holds 2 x lane.
- An `SFPLOAD` or `SFPSTORE` moves 4 Dst rows x 8 columns, with even columns at row 4r and odd ones at 4r + 2, and 32 of
  them cover a tile. Load with address mode 2 and store with mode 3 (Dst + 2), and a body run 32 times walks the tile.
- `SFPMAD` is a fused multiply-add that rounds to nearest even, and denormals flush to zero. `SFPSWAP` with mod 1 leaves
  the min in vd and the max in vc. `SFPARECIP` is a reciprocal estimate, good to 0.6%. `SFPGT` with mod 8 gives an
  all-ones mask. Lane flags (`SFPENCC`, `SFPIADD` with an immediate, `SFPSETCC`) make instructions write the enabled
  lanes and leave the others alone.

The SFPU programs are tinygrad's primitives (exp2, log2, sin, sqrt, reciprocal, max, cmplt, cmpne and where), plus exp,
log and cos, with fp32 constants. The transcendentals reduce x in exact steps and finish with a fitted polynomial. That
gives fp32 tensors float32's accuracy: exp, log, sin and cos are within 2e-7, and sqrt is within 3 ulps.

Binary programs find their second operand 128 Dst rows further on. `where` is `keep(c, x) + drop(c, y)`. That's exact,
and with no `c * x` in it, an inf can't turn into a NaN. In a bf16 kernel a NaN is already inf by the time the SFPU sees
it, so fp32 kernels are the ones that check for NaN. A unary op on a result that nothing else reads runs in place in
Dst, on the unrounded result. Otherwise it runs on its operand plus the zero tile.

### Reductions and passes

`sum(axis)` and `max(axis)` work along axis 0 (the batch), axis 1 (the rows) or axis 2 (each row down to a column). The
tiles along the axis land in Dst next to the zero tile. The first goes into an accumulator (Dst row 256 or 320), and
later ones land 128 rows on, where an SFPU pass folds them in. A line's last tile, if it's cut short, is masked to the
identity value (`mask_columns`, `mask_rows`).

At the end, the tile is folded into itself so the result fills it and broadcasts. Along a row that's done by rotating
lane groups (`SFPSHFT2`), and down a column by `SFPTRANSP`. Accumulators have a semaphore each, and sums stay fp32 the
whole way.

If an expression reduces, a unit is a line of tiles along the axis, and the kernel goes over it in passes. A reduction
can be read in the pass after the one that swept it, and each pass recomputes what it needs. Softmax is three passes in
a launch: the max, the sum of e^x, then e^x times the reciprocal of the sum. An input that more than one pass reads
keeps its line in L1. A value that's one tile long along the axis gets its own slot and is worked out once per line. A
kernel reduces along one axis, and reductions along other axes get computed first.

### Attention

`Tensor.attend` (`kernels.attention_kernel`) computes softmax(q kᵀ + mask) v in one kernel and keeps the scores in L1. A
unit is up to eight tile rows of a head's queries.

BRISC starts by reading the unit's positions: each row's last key, and its first key with a sliding window. From those
it works out which tiles of keys the rows reach and which of those tiles some row sees only part of, and it tells the
TRISCs. The unit then sweeps those tiles twice, a frame at a time. A frame is a tile row of keys, plus the values in the
second sweep. Where a row sees part of a tile, BRISC writes the mask's tiles into the frame from the positions, with
-1e9 where a row mustn't look. The length of the keys is an argument, so a kernel works for any length and a server can
use the same replay for all of them.

- A tile of scores is built in Dst. The mask's tile (or the zero tile) goes in first, then `MVMUL` accumulates the
  queries times the keys onto it, reading the keys transposed.
- The first sweep folds the scores into each row's maximum. For a unit of up to two tile rows, the maximums live in
  Dst's two accumulators, and otherwise in L1 between key tiles. The second sweep subtracts the maximum, and the SFPU's
  exp turns the scores into p in place.
- p is packed as bf16, then multiplied by the values and by a tile of ones, which gives each row's sum of p. The
  products are packed onto their output tiles in L1 in fp32: the packer adds to what's there (`Pack_L1_Acc`, config word
  71 bit 19), and `repack` switches that on and off. A fused kernel then divides the output by the sums, so attention is
  two launches.
- Flash attention sweeps once, but that needs the output in L1 rescaled whenever a row's maximum grows, and the   packer
  can't do that. We sweep the keys twice and the values once instead.
- If a unit's keys fit in L1, they stay there from the first sweep to the second (`attention_store`), and each key is
  read once. A bf16 cache's keys go in free slots. A block float's keys go in regions of complete key rows, up to 64
  tiles each, below B8 and past the slots in use and the frames' regions. That fits 104 key tile rows of Qwen3.6's
  256-wide heads in bfp8. The unpacker switches to each row's region.

The results are within 1% of numpy.

A decode step has few units, one per key head per slot, so their keys are split among up to 8 **parts**: flash decoding.
It uses as many parts as the workers allow with at least 8 key tiles each. If a part's keys wouldn't fit in L1, it uses
enough parts to make them fit. The parts' p v, sums and maximums go back to DRAM, and a few fused kernels combine them,
weighting each part by exp(its maximum minus the overall maximum).

A part with no keys takes a tile that's masked out with -1e9. A row's maximum is held at -1e4 or more, so a row that
sees none of a part's keys gives that part weight 0. The parts' maximums are rounded to bf16 in Dst, so a part's p and
its weight agree. `llama.attention` uses this for heads 64 wide or wider, except gpt-oss's, because of its sinks.

### Shapes and movement

**Padding.** Tensors are padded to whole tiles with zeros, except for a dimension of 1, which fills its tile with
copies. A row vector's tile holds the row 32 times, so broadcasting can reuse the tile as it is. A result's padding is
zeroed in Dst before its last tiles are packed, so a matmul can sum over a padded K. `(R, 1) @ (1, C)` is broadcasting.

The number of rows in a tensor's last tile is a kernel argument, and the mask's words get pushed with it. A kernel then
works for any row count that fits in the same tiles, unless it reduces along rows, reads a transposed B, or picks
matrices by index. Attention takes its query rows the same way, so a prompt's last chunk doesn't compile anything new
beyond its tile rows, whatever its length. That matters because a kernel costs about 0.2 s of clang.

**Batches and views.** A batch is matrices stored back to back, and a batch of 1 repeats. Kernels take their inputs as
(address, batch stride, row stride, offset), so views go into kernels without a copy. That covers `x[b]`, slices on tile
boundaries, forward steps through the batch, and `split`. A result can be written into a view too (`realize(into)`,
`merge`). `repeat_interleave(n)` repeats matrices in place, which grouped-query attention's keys need.

**Transpose** (`.T`). A matmul reads a transposed B where it is. Unpacker 0 transposes faces on their way into SrcA
(`Haloize_mode`, config word 72 bit 8, switched by `haloize`), and the math walks the faces in transposed order (`mmT`,
five address modes). Anywhere else, `transpose_kernel` fetches tiles with their faces swapped.

**Movement** (`reshape`, `permute`, `expand`, `pad` and slices, done by `move_kernel`). They're maps from the result's
elements to the source's: an offset, three strides, bounds for each axis, and 0 outside them. A chain of them composes
into one launch, except across a reshape. The kernel runs on BRISC. It fetches source tiles into a cache in L1 (127
slots, indexed by tile number mod a prime). Where it can, it fetches a tile with one NoC read. Otherwise it copies runs,
four values at a time (two bf16 values per word).

**Rows by index.** `gather(ids)` copies rows out of a table, like the embedding. A result tile's 64 runs are read as
64-byte blocks, all in flight before the kernel waits. `put(ids, rows)` writes rows into a tensor in place, with a unit
per row. That's how a token's keys and values go into the KV cache at its position, where the position is data and not a
shape. A NoC read's latency, which ttsim hides, is then paid per tile instead of per run.

### Dtypes

**fp32** tiles are 4 KiB. The math thread unpacks the operands into Dst itself (`fill`, with unpacker 0's second
context), and the SFPU does the ops. Ops round once, so a fused fp32 tree gives numpy's float32 answer. A kernel
computes in one dtype, so a bf16 operand of an fp32 op gets cast first.

**int32** is two's complement through Dst and the LRegs. It uses `SFPSTORE`'s INT32 mode, because its FP32 mode flushes
small integers. Add and sub use `SFPIADD`, and comparisons go through sign-magnitude (`SFPCAST` mode 3).

**Casts** pack in another format, into two 4 KiB buffers at `OUTS`, with the packer switched by `repack`.

**Upload and download** convert between row-major and tile order with strided 8-byte copies, and write a bank's tiles in
one write.

### Autograd

It works like tinygrad's. Ops have rules (`_backward`) that run in reverse. The rules are lazy ops too, so the backward
pass fuses like the forward pass does. An fp32 matmul is recorded as one matmul. `test_autograd` checks the ops against
numpy's central differences.

## Models

`models/llama.py` is written once and runs on both baretile Tensors and numpy, with numpy as the reference at each step.
`models/qwen.py` adds what a mixture of experts, Gated DeltaNet and multi-token prediction need on top of a Llama block.

It loads Llama, Mistral, Qwen2 and Qwen3 checkpoints, the mixtures of experts (Mixtral, Qwen2-MoE, Qwen3-MoE, and
gpt-oss with its experts unpacked from MXFP4), and Qwen3.5 and Qwen3.6. The families other than Llama are also checked
against their transformers implementations (`test_hf_families`, `test_qwen3_moe`, `test_qwen3_5_moe`).

A config comes from config.json through `llama.config`, or you can build one by hand with `llama.model(...)`. What it
leaves out defaults to a plain Llama.

Weights are read when they're uploaded (`Weight`) and cast to their formats on the chip. `precision` picks the formats:

- `bf16`, `bfp8` or `bfp4`.
- `tt`, which is Tenstorrent's performance precision.
- `experts`, which puts the MLP matrices in bfp4 at one fidelity phase and the rest in bfp8 at two, with a bfp4 draft
  head and a bfp8 KV cache. That's what fits Qwen3.6 on one card.

The blocks' matrices and the output layer are read as a matmul's B, so they're laid out by columns (`Tensor.columns`). A
token's row and a grid's shares can then read them in long requests. `llama.arrays` gives numpy the same values.

A block's q, k and v are fused into one matrix, and so are gate and up (`fuse`). RoPE is one kernel for q, k and v (`x
cos + (x @ turn) sin`). q's tables are pre-multiplied by attention's 1/sqrt(head width), so attention gets q already
scaled. Each half of a block adds onto the residual stream in its last kernel, its output matmul or a mixture's sum. The
stream is built there, and no norm has to add up the blocks' outputs again. The tokenizer is tokenizer.json's byte-level
BPE in Python, and it gives the same tokens as tiktoken and Hugging Face on the same file.

- **A fixed-size KV cache.** A token's keys and values are `put` at its position, and attention's mask is built on the
  chip from the positions. The launches are the same from token to token, which is what lets a decode step replay.
  Prompts go in chunks.
- **Several sequences at once** (`forward`'s `slots`). Their caches sit side by side in a tensor. For a key head, a
  sequence's query heads form a batch entry, so the key head's rows are read once for all its query heads.
- **Mixture of experts** (`qwen.moe`). `topk` picks a token's k best experts on the chip, in a BRISC kernel that does a
  row per worker. Their SwiGLUs are matmuls that pick the experts' weights where they sit in DRAM, and the outputs are
  weighted and summed.
  - In a decode step, an expert that some token picked runs once, with the tokens as rows of its entry. `qwen.touched`
    lists the experts picked with a nonzero weight, in order, then none. It also gives each entry's weight for each
    token, 0 where the token picked a different expert. The weights are applied before the down projection, and the
    entries are summed. The touched experts' weights are read once, and so is the tokens' row.
  - A prompt's tokens are grouped by expert (`qwen.group`, a counting sort on BRISC), so an expert's weights are read
    once per 32 of its tokens. Unused tiles past the end pick no expert.
  - Up to 128 tokens (a decode or verify step), the first way wins, because its extra tile rows cost less than
    grouping's gathers of 64-byte runs. Past that, it takes whichever way makes fewer tile rows.
  - Nothing comes back to the host, so a decode step replays.
- **Gated DeltaNet** (`qwen.deltanet`, in Qwen3.5 and 3.6). A matmul makes q, k, v, z, b and a. Then comes a causal conv
  with 4 taps, which keeps its last 3 inputs, and a SiLU. A value head keeps a 128 x 128 state, updated as S = e^g S +
  beta k (v - e^g S^T k)^T, and its query reads the state. The output is RMS-normed and gated by SiLU(z), in fp32.
  - A decode step runs the recurrence in one kernel, with a unit per key head (`qwen.delta`, with `step` as the numpy
    version).
  - A prompt goes in pieces of 64 (`chunk`) using the UT transform, (I + L)^-1 = (I + P)(I + P^2)(I + P^4)... with P =
    -L, so there's no loop over tokens. The work that doesn't touch the state is done for all the pieces together, and
    then the state goes through the pieces in order.
  - The states are written back in place (`assign`), so a replay carries them forward.
- **Gated attention** (Qwen3.5 and 3.6). A head's output is multiplied by sigmoid(gate), RoPE covers a quarter of the
  head, q and k are RMS-normed per head, and the RMSNorms multiply by 1 + w.
- **Speculative decoding** (`qwen.draft`, and `forward`'s accept). The multi-token prediction layer is an extra block
  after the last one. From a position's hidden state and the next token, it drafts a token through the output layer. The
  one-card layout uses a bfp4 copy of the output layer for this (`precision`'s draft head, 0.29 GB), which reads half
  the bytes of the bfp8 one. A verify step takes a token and its draft and keeps two copies of the DeltaNet states,
  after both tokens and after the first. The next step's accept row says which to continue from, and the kernel reads
  that copy and writes both.
- **Token accuracy** (`llama.accuracy`) is measured the way Tenstorrent measures it. The prompt is the first 512 tokens
  of the reference file (tt_transformers' `tests/reference_outputs/<model>.refpt`), followed by 500 steps where the
  model is fed the file's next token. Top-1 counts the steps where our pick is the reference model's best token, and
  top-5 the ones where it's in the reference's top five. In numpy it runs as one long prefill, with the weights as the
  chip holds them, so it works for models of any size.

**Qwen3.6-35B-A3B** has 40 blocks, 2048 wide: 30 Gated DeltaNet blocks and 10 attention blocks. It has 256 experts of
width 512, picks 8 of them, and has a shared expert. Its vocabulary is 248,320 tokens. It's 71.0 GB in bf16 and 21.9 GB
in `experts`'s formats, which fits on one card.

### Gated DeltaNet's decode step

`qwen.delta` (`qwen.delta_kernel`) does a decode step's Gated DeltaNet in one kernel, from the token's projections to
its outputs. For a verify step it does the token and its draft (see Speculative decoding). It runs in fp32 on the SFPU,
and reads and writes each state once.

A unit is a key head of a sequence plus its value heads, so no other unit touches its channels of the conv's last
inputs. After BRISC reads them, it writes back the inputs that the token leaves, in place: they move up a position, and
the token's own input goes last. There are no shift launches, and the conv inputs don't travel through DRAM as tiles.
The math thread fills Dst from L1 itself and folds the conv, the SiLU, the lengths of q and k, and the gates into
broadcast tiles.

The recurrence splits by column. For a value head and a column tile j, an SFPU pass over the key tiles sums u = S^T k
and w = S^T q down the column. d = beta (v - e^g u) stays in Dst. The output's tile is e^g w + d (k . q), and the
column's new tiles are e^g S + k d. The SFPU programs are functions that the loops call, which keeps the loop bodies
from growing to thousands of inline Tensix words.

A dead sequence's units (a free slot, with live set to 0) read that flag and nothing more. Their math runs on whatever
the frame holds, and BRISC sends 0 for their outputs and leaves their states alone.

A decode step's Gated DeltaNet block comes to four kernels: the qkv matmul, this kernel, the output's norm and gate, and
the output matmul.

## Serving

`models/serve.py` is an OpenAI-compatible server.

**Slots.** The engine's KV cache has `slots` slots of `places` rows each, and each step is one replayed call for all the
slots. A free slot runs on its last row, which no sequence reaches (`llama.around`'s live). Its attention sees the
slot's first row, which costs a tile read. Its token gives the experts weight 0, so `touched` lists none for it, and the
delta kernel leaves its Gated DeltaNet states alone.

Attention in one kernel reads the rows that each sequence's position reaches, so the same replay works for any length.
Some models can't use that kernel: gpt-oss, because of its sinks, and models with heads narrower than 64. For them,
attention reads each slot's rows up to the power of 2 (at least 32) that holds the longest sequence. There's a recorded
replay for each of those lengths.

Prompts are prefilled a chunk between steps. A chunk is 2,048 tokens for a mixture of experts, since its chunks read
nearly all the experts anyway, and 256 tokens otherwise.

**Prefix cache.** A prompt goes to the free slot that holds the longest prefix of it, and the rest gets prefilled.

**Kept sessions.** Gated DeltaNet's states are valid for the exact tokens they've taken in. A session (its tokens, its
slot's cache rows and its states) is saved when its prompt is in and again when it ends. A prompt that starts with a
saved session picks up from there, in any slot, and an agent's next turn prefills its new tokens and no more.

Saving a session copies it on the chip with its tiles as they are. That's an `assign` of a tensor that's already
computed, done by `copy_kernel` on BRISC, which handles block floats too. The copy goes into DRAM the caches don't use
(`park` bytes, which defaults to what's free when the engine starts, minus 2 GiB). Saving a session and getting it back
cost a few launches and no PCIe traffic. Once that space fills up, the least recently used sessions go to the host
(`keep` bytes): they're read back, and later written over the BARs again. When the host's space is full too, they're
dropped.

**Speculative steps**, for a model with a prediction layer (see Speculative decoding). A call verifies the slots' tokens
and drafts, and another drafts the next tokens, picking each row's best token on the chip.

**Heads** (`heads`). The output layer writes its logits in fp32, and a call ends by computing the rows' heads on the
chip. A head is a row's 64 largest logits and their tokens, plus the log of the sum of exp(x t) over the row, where t =
1/temperature for that row's request.

- `topk` runs on parts of a row that fit in a worker's L1. Qwen3.6's 248,320 logits are 97 parts of 2,560. When no part
  size of up to 96 tiles divides the vocabulary, the parts are as big as `topk` can hold, so Llama 3's 128,256 become 24
  parts of 5,344. The parts' 64 go side by side, and another `topk` runs over those, with their tokens as labels.
- An fp32 kernel computes the parts' sums in a sweep that starts from the part's largest logit (the first of its 64),
  and another kernel adds up the parts.
- The logits are read once. `topk` reads the runs of up to 8 rows rather than their tiles. A row on its own is scanned
  by a unit per tile row, and that unit writes the 32 copies of its result.

The host reads back under 1 KB per row, not the row. Greedy sampling takes the first token. top_k up to 64 is exact from
the head. top_p and min_p are exact too when their cutoff falls inside the head, because the sum gives a token's share
of the row. If a draw without any of those lands past the head's probability mass, the host reads the row back and draws
from the rest of it. Penalties, constrained output and logprobs read the row too, and so does a vocabulary that no part
size divides, like Qwen2.5's and gpt-oss's. For those, the head is made on the host (`Head.of`: a pass finds the 64th
largest logit, and the logits at least that big get sorted).

**Requests.** The server supports presence, frequency and repetition penalties (applied the way vLLM applies them),
temperature, top_k, top_p, min_p, seed, stop strings and logprobs. It looks for stop strings where the newest token's
text could complete one. What a request leaves out comes from the model's `generation_config.json`, like vLLM does it
(for Qwen3.6, top_k 20 and top_p 0.95), and otherwise from OpenAI's defaults.

Chat templates are written out in Python. Qwen3.5's and 3.6's tool calls come back as OpenAI's `tool_calls`.
`tool_choice` and `response_format` (a JSON schema, parsed as a prefix) constrain what can be drawn. A drawn token is
kept if the text can still fit the shape, and otherwise the draw is repeated without that token. Reasoning comes back
apart from the answer (`reasoning_content`).

**HTTP.** It serves `/v1/models`, `/v1/completions` and `/v1/chat/completions`, whole or streamed. A request holds a
slot from when it arrives until it ends, and a request that finds no free slot gets a 429 immediately. OpenRouter then
tries the next provider, and it doesn't count 429s against uptime. Usage counts the prompt's cached tokens and comes at
the end of a stream. A request that produces no text for 600 s ends and gives up its slot, with a 504 unless it's
streaming.

## The cost model

`models/cost.py` estimates what a model costs to serve, from its config.

A card holds:

- the weights, in a mode's formats (`llama.precision`'s, and the KV cache's too unless `kv=` says otherwise),
- a GiB for kernels and activations,
- and the sessions: KV cache rows and Gated DeltaNet states.


A decode step for n users reads:

- the matrices other than the experts and the embedding,
- the experts its tokens pick, E (1 - (1 - k/E)^tokens) per block,
- the states, read and written once (a verify step reads one of its two and writes both),
- the rows that attention's kernel reaches, with the keys read once where a part keeps them in L1, and the values once,
- and the activations (`ACTIVATIONS`): 4 MB per block in 32-row tiles, the picked experts' outputs for their tokens'
  tile rows, and 45 MB of logits per tile row of tokens.

A step takes as long as DRAM or the matmuls, whichever is slower. Then add its launches (`LAUNCHES`) and the RISC-V work
its kernels do beyond their reads (`CONTROL`, which grows with the step's tokens). We counted the launches and that work
on ttsim at Qwen3.6's shapes, and the bytes with ttsim-chaos's read counter. A prompt's chunks read the experts their
tokens pick and attention's keys and values for every eight tile rows of a query head. They read their activations again
where a matmul's blocks go a group of columns at a time (`regroups`).

Until the probe measures a card (`probe=probe.json`), the cost model uses the spec sheet. That's 512 GB/s with 0.7 of it
reachable (a guess), 664 TFLOPS at one fidelity phase with bfp8 matmuls taking two, and a microsecond per launch.

A run takes keys from `CARD` and `USE`. `context=` is a session's length in positions, and `accept=` is the share of
drafts that get accepted. Qwen3.6 on one card is `python3 models/cost.py path/to/Qwen3.6-35B-A3B experts context=16384
accept=0.55`.

## tinygrad on baretile

`adapters/tinygrad_baretile.py` puts tinygrad's frontend on baretile's compiler and runtime. The BARETILE device takes
over at `create_linear_with_vars`, where tinygrad would schedule kernels. It runs the graph of tinygrad's primitives op
by op into baretile's lazy tensors and hands tinygrad back an empty schedule.

Movement stays a view (a small ShapeTracker) and is folded into baretile's three axes when an op needs the data. A
broadcast multiply followed by a sum becomes a baretile matmul. What baretile doesn't have runs lazily on tinygrad's CPU
device: 64-bit integers, bit operations, tinygrad's random numbers, symbolic shapes, and dtypes besides float32,
bfloat16, int32 and bool. `TinyJit` maps to baretile's replay. PyTorch runs through tinygrad's own frontend
(`extra/torch_backend`).

## Bring-up on a p150

Before you start, you need:

- 300 W on a 12V-2x6 cable.
- The card in `lspci -d 1e52:b140`, ideally at `LnkSta: Speed 32GT/s, Width x16`.
- The IOMMU off or in passthrough (`iommu=pt`), since `pci.Card` hands the device physical addresses.
- A few 2 MiB hugepages: `echo 4 > /proc/sys/vm/nr_hugepages`.
- Root.

`pci.Card` unbinds tt-kmd. Check the firmware version with tt-smi (which needs tt-kmd), and if it's old, flash the
latest bundle with tt-flash. The tests open the card when `TTSIM` isn't set.

Go in order. Each step needs the ones before it and no others.

| Step | Command | What it confirms, and what to suspect if it fails |
|---|---|---|
| 0 | `python3 -m baretile.chip` | ARC is up, Tensix is powered, the AI clock is at 1350 (800 means idle), the Tensix columns, DRAM training, and the telemetry tags. If the NoC isn't translating coordinates, the coordinates in `bh.py` are wrong for this card |
| 1 | `python3 -m pytest test -k "config_space or arc"` | the BARs, TLB config, ARC's status, the message queue and telemetry |
| 2 | `-k "dma or l1 or (dram and not freed)"` | the PCIe tile writing into host memory (bus addresses below 2^58), L1 writes, DRAM coordinates and training |
| 3 | `python3 -m baretile.hello` | reset bits and PCs, the firmware, the NIU's initiators, device writes into host memory, the dispatcher reading the ring, posted status writes, multicasts and the workers' atomic increments. A hang after the workers are done points at the count. A hang before any worker starts points at the multicast acknowledgements |
| 4 | `-k "every_core or relaunch or moves_data or queue or hung"` | the launch path, and instruction cache invalidation (a no-op on ttsim). A value that's wrong in `tolist` but right in `chip.read` points at the readback |
| 5 | `-k "tile_math or tensor_math"` | backend config indices (which came from ttsim, not the ISA), encodings, `.ttinsn`, semaphores, and bf16 unpack and pack |
| 6 | `-k "matmul or stream or bfp"` | `MVMUL`, address modes, fidelity, replay, grid multicasts, bfp8, and reads waited for by ID. If matmuls are wrong now and then, try `DEPTH = 1` |
| 7 | `-k "sfpu or reduc or softmax or any_shape or views or batch or movement"` | run-time slots, BRISC's 2-byte L1 accesses, SFPU encodings, lane flags, `SFPTRANSP` and `Haloize_mode` |
| 8 | the test suite, a few times | the waits that ttsim never exercises. A failure that comes and goes points here |
| 9 | `GPT2=... python3 -m pytest test -k gpt2`, then `models/gpt2.py` | everything together, and the first timings (`Runtime.profile`). If they're all 0, look at `bh.WALL_CLOCK` |
| 10 | `python3 -m baretile.probe`, `python3 -m baretile.bench` (and `1 768 3072`, `1 768 3072 4 bfp8`) | the numbers ttsim can't give: launch cost, DRAM GB/s by request size and reads in flight, read latency, NoC latency, MVMUL, unpack, pack and SFPU rates, BRISC's topk and PCIe, plus a matmul's TFLOPS and decode's GB/s. The DRAM probe reads block-float tiles the way matmuls do, from a near and a far DRAM port, and with the grid's workers reading chunks from their own banks. `probe 8 probe.json` feeds the cost model, which takes the slower of the bfp8 and bfp4 chunk bandwidths, since decode reads weights by columns |
| 11 | `python3 -m baretile.probe 256`, and a model of a few GB | uploads. If they're far under a GB/s, map the windows write-combined (`resource0_wc`) |
| 12 | `test/benchmark.py ttnn ...`, then `baretile ...` | baretile against TT-NN on the same card |

We've rehearsed this in order on ttsim-chaos. If something hangs, read `Runtime.wait`'s report next to "The pipeline"
above. TRISC0 waits for `FETCHED`, BRISC waits for `UNPACKED` and flags, and TRISC2 waits for `SENT`. A read ID that
stays in flight is waiting on the NoC. A hung kernel doesn't need a reset, because the next `Runtime` resets its cores.
A hung NoC or ARC does: `python3 -m baretile.chip reset`, which is untested.

## What we don't know yet

- `pci.Card` and `pci.reset` haven't run on a card. `test_card_sysfs` runs them against a fake sysfs and checks the
  files they write and the bits they set. We don't know yet:
  - whether ARC's power messages wake Tensix up to 1,350 MHz after kmd gated it,
  - whether window 0 reaches host memory without kmd's iATU setup,
  - whether device writes into host memory and the dispatcher's ordered posted writes behave like they do on ttsim.

- The ISA doesn't document BAR2, the DRAM tile's address map, ARC's protocol or soft reset's power-on value. ttsim
  models them, and silicon may differ. ttsim doesn't model `mhartid`, `fence.i`, the PC snapshots or request splitting,
  and its instruction cache invalidation does nothing.
- The card's rates: the DRAM bandwidth a step gets, launch cost, how fast the RISC-V work runs, a worker's read
  bandwidth, and the best `DEPTH`. The cost model guesses all of them.
- Qwen3.6's token accuracy with the matrices in bfp8 (which needs two cards), and with a bfp4 KV cache.
- Why ttsim decodes an `XMOV` inside a TRISC loop that's thousands of inline Tensix words long. To get around it, the
  delta kernel's SFPU programs are called as functions.
