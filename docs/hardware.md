# Blackhole hardware reference

This page lists the facts about Blackhole that baretile depends on, and where
each one comes from. The addresses and encodings in [bh.py](../baretile/bh.py) come from Tenstorrent's
[Blackhole ISA documentation](https://github.com/tenstorrent/tt-isa-documentation),
[ttsim](https://github.com/tenstorrent/ttsim),
[tt-kmd](https://github.com/tenstorrent/tt-kmd) and
[tt-umd](https://github.com/tenstorrent/tt-umd).
The Ethernet code also uses definitions from
[tt-metal](https://github.com/tenstorrent/tt-metal).

ISA paths below start at `BlackholeA0/`, and simulator paths start at `src/`.

## PCIe and TLB windows

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
4 KiB. `pci.release` binds tt-kmd's `tenstorrent` driver again afterwards, and
`pci.preflight` checks what `Card` needs without opening the card. It copies registers a word at a time. Using `memcpy` could split a register
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

## Coordinates and DRAM

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

## Host memory and ARC

The card backend uses locked 2 MiB hugepages for DMA. It reads physical addresses
from `/proc/self/pagemap` and passes them to the device. You'll need root for
this. The IOMMU must be disabled or in passthrough mode so the card can use
those addresses without translation.

Host memory is reached through the PCIe tile. Addresses below `2**58` use the
direct bus-address path. Bit 60 selects the outbound iATU path used by kmd and
UMD. baretile uses the direct path and doesn't program iATU windows. We still
need to check whether firmware sets that up after boot. The dispatcher's command
reads, readback and uploads all depend on it.

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

## Reset and RISC-V cores

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
these cores, and don't use `fence.i` to reload a kernel. Firmware invalidates
the instruction caches through Tensix configuration instead.

Unsupported instructions have undefined behavior instead of trapping. Use
instructions from the subset in the existing device programs.

The small RISC-V data cache isn't coherent with NoC writes. Poll shared L1 with
a fence. Before another unit reads data that a core has stored, `l1_settle`
uses a fence, a load and another fence. A fence alone doesn't settle stores
across memory regions.

## NoC transfers

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

## Ethernet links

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
