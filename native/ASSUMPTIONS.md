# Assumptions behind core optimisation

What this project assumes about the OCR pipeline, how each assumption was
established, and what an optimisation is allowed to change. Read it before
touching `ocr_core*.c` / `ocr_geometry.c`.

Tags: **[code]** read from the source, **[measured]** reproduced on this host by
the benchmark or the probes noted, **[policy]** a scope decision, not a fact of
the code — challenge it if it is wrong.

Sections A–D are the workload and the invariants. Section E is the cost model the
current numbers support. Section F lists the levers that follow from it.

## A. Workload and concurrency

- **A1. One request is in flight per daemon process. [code]**
  Both frontends are a single `accept` → read → `ocr_run` → `send` → `close` loop
  (`c_daemon.c`, `cpp_daemon.cpp`); there is no worker pool. So no buffer, session
  or scratch allocation can race, and anything process-wide can be reused across
  requests. Measured footprint: 17 threads per daemon (main + ONNX Runtime pools
  on this 12-core host), but only one of them runs our code at a time.
  *Breaks if:* the frontend ever gains concurrency — then reuse must move into
  per-thread state. The C core's single `g_error` buffer would start racing first.

- **A2. Latency is the metric, not throughput. [policy]** The benchmark measures
  one request at a time over localhost and reports medians and tails (`worst
  image`, and `heavy` = the slowest fifth of the corpus); under A5 it is that tail
  a user feels, not the corpus median. CPU time (2.52 s per request across
  runs) is reported but not optimised; it exists to show how wide the core runs
  (~6 threads since D3).

- **A3. The corpus is the target workload, but not the only input. [policy]**
  Production input is arbitrary screenshots: `resize_within_bounds` clamps the
  largest side to 2000 px and the smallest to 30 px, so the pipeline must stay
  correct for any aspect ratio. The corpus itself never hits the 2000 px clamp
  (its largest sources are 1100x159 and 768x360).

- **A4. The core is ~all of the latency. [measured]** A request the daemons reject
  without OCR work round-trips in 0.2 ms, against ~440 ms for a real one, so
  frontend and HTTP cost are out of scope. Optimising here means the detector, the
  recogniser and the shared geometry, nothing else.

- **A5. One user, one image at a time, at most ~1 screenshot per second. [policy]**
  The daemon is driven interactively, not as a batch service. Service time is
  ~0.39–0.45 s against a ~1 s arrival interval, so steady-state utilisation is
  ~40–45% and requests never queue for a free slot in the normal case. What
  follows:
  - Throughput is worthless as a target; only single-request latency counts, and
    there is no incentive to batch, pipeline or interleave requests.
  - The **budget is the arrival interval**. Before the detector budget, thread fix
    and per-crop recognition (B, D3, E) the worst images were `07.png` 0.89–1.05 s,
    `19.png` 0.89–1.15 s and `01.png` 0.78–0.89 s, and both candidates had two
    images over budget. Now the worst is `19.png` 0.67 s (67% of the interval) and
    no image is over, with ~30–39% of headroom left on the worst one. Any image that
    does go
    over arrives while another request is still in flight and waits in the kernel
    backlog (the frontends `listen(16)` and serve strictly serially), so latency
    doubles rather than degrades gracefully. That makes the tail the target, not
    the corpus median — which is why the harness reports `worst image` and `heavy`
    (slowest fifth) beside the median, and keeps `benchmarks/history.jsonl` so the
    tail can be followed across optimisations.
  - **The process is idle roughly half the time**, which makes memory and
    warm-up costs cheap (see F5) and means the machine is not thermally pinned the
    way a benchmark run pins it.
  - Measured: at a 1 s cadence, latency is the same as back-to-back — interleaved
    over four rounds per image, spaced − burst = −7, +15, −7, −4 ms on
    `01.png`, `07.png`, `19.png`, `11.png`, i.e. inside the ±50 ms run-to-run
    noise. So the benchmark's hotter back-to-back regime is a valid proxy for
    production, and the earlier apparent +9…+87 ms penalty was ordering bias.
  - Measured: idle is not free. While untouched for 1 s after a request the
    process burns ~0.10 s of CPU (five samples, all 0.1), i.e. ONNX Runtime's
    intra-op pool spin-waits. That costs nothing in the benchmark (it is
    back-to-back) but is real power draw for a resident daemon; it is a product
    question (cap spinning / fewer threads), not a latency lever.

## B. Pipeline invariants — the numbers that define the output

These are copied from the code and shared by both cores; each one changes the
recognised text if altered. **[code]**

| stage | rule |
| --- | --- |
| pre-clamp | longest side → 2000 px, then shortest side → 30 px, every dimension rounded to a multiple of 32 |
| vertical padding | if `h <= 30` or `w/h > 8`, pad top and bottom to `2 * max(int(w/8), 30)` |
| detector canvas | short side scaled up to **736** px (if smaller), dimensions rounded to 32, minimum 32; then capped at **3.0 Mpx** by shrinking both sides uniformly and flooring to the 32-px grid (E) |
| detection mask | probability `> 0.3`, then 2x2 dilation, anchor (0,0) |
| box filter | min-area-rect short side `>= 3`, mean probability `>= 0.5`, grow by `area * 1.6 / perimeter` per side, short side `>= 5` after growth, edges `> 3` px |
| crop | warp to integer `WxH` with bicubic + border replicate; rotate 90° CCW when `h/w >= 1.5` |
| recognition | 48 px high, **one crop per session run**, canvas width `max(320, 48 * aspect)` of that crop alone (no sibling padding), CTC greedy decode with repeat suppression |
| output | drop whitespace-only or mean-confidence `< 0.5` lines, join the rest with `\n` |

Coordinate maths is `double` with Python-style half-to-even rounding
(`ocr_py_round`) wherever it meets RapidOCR's own numbers; a `float` here flips a
crop by a pixel and changes the text.

## C. Equivalence and determinism — the oracle for any change

- **C1. Both cores return byte-identical text. [measured]** 20 of 20 corpus
  images, re-checked after every change in this document's history. Both call the
  same `ocr_geometry.c`, so a difference means a core diverged, not a crop.
- **C2. A process is deterministic. [measured]** Repeating every corpus request in
  the same process reproduces the bytes 20 of 20. So a change that keeps text
  identical on the corpus is text-neutral *for these inputs*, which is a cheap and
  strict regression gate.
- **C3. Byte-identity is the primary gate, CER only secondary. [policy]**
  Accuracy moved between builds of identical source once (25% / 1.38% → 30% /
  1.34%), and that was never explained. Treat sub-0.1% CER movement as noise and
  never accept text changes without knowing which stage caused them.
- **C4. Geometry changes affect both candidates. [code]** `ocr_geometry.c` is
  compiled into both archives. A change there must be re-measured for C *and*
  C++, and it invalidates earlier paired comparisons.

## D. Runtime and model

- **D1. Both cores use the same runtime and models. [code]** `native/lib/` ONNX
  Runtime 1.31.1, the two `models/` assets, `ORT_ENABLE_ALL`.
- **D2. The CPU memory arena is off by default, opt-in with `OCR_MEM_ARENA`. [code] [measured]**
  `DisableCpuMemArena()` in each core unless that variable is set, so a shipped
  `Run` allocates and frees through the default allocator. With one crop per run
  (B) the per-call allocations are small, but the detector `Run` still allocates
  tens of MB of activations, so the arena is not free to dismiss: re-measured
  interleaved on the current buffer-reuse build it is worth ~19 ms (C) and ~15 ms
  (C++) of corpus median and ~27–29 ms of the heavy-set median, while resident
  memory goes 208–315 MiB → ~1.16 GiB, because it caches a buffer for every
  canvas width it has seen and the corpus produces fifty-odd widths. It stays
  **off by default** — 1.16 GiB to save 15–19 ms is the wrong side of A5's trade —
  and `OCR_MEM_ARENA=1` turns it on for measurement or for a host with memory to
  spare.
- **D3. The intra-op pool is sized to the physical cores. [code] [measured]**
  Both cores call `SetIntraOpNumThreads(ocr_intra_op_threads())`, which reads
  `/sys/devices/system/cpu/cpu0/topology/thread_siblings_list` and divides the
  logical count by the siblings per core (12 → 6 here); on a host without SMT it
  returns 0 and ORT's own default stands. `OCR_THREADS` overrides it. The reason is
  measured, not theoretical — on this host the default put two pool threads on
  every physical core and cost ~20%:

  | intra-op threads | 19.png | 07.png | 01.png | 09.png | 18.png | 06.png | total | speedup | CPU/req |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | 1 | 1400 | 1743 | 1136 | 1096 | 1100 | 1246 | 7720 | 1.00x | 1372 ms |
  | 2 | 864 | 1012 | 715 | 693 | 712 | 739 | 4736 | 1.63x | 1685 ms |
  | 3 | 715 | 781 | 604 | 595 | 590 | 595 | 3880 | 1.99x | 2081 ms |
  | 4 | 662 | 672 | 560 | 569 | 559 | 523 | 3546 | 2.18x | 2556 ms |
  | 5 | 669 | 630 | 562 | 545 | 557 | 506 | 3468 | 2.23x | 3076 ms |
  | **6 (physical)** | **612** | **570** | **546** | **545** | **539** | **470** | **3282** | **2.35x** | 3523 ms |
  | 8 | 700 | 688 | 617 | 605 | 627 | 573 | 3808 | 2.03x | 5344 ms |
  | 12 (logical) | 771 | 809 | 675 | 655 | 693 | 664 | 4267 | 1.81x | 7863 ms |

  Best of three interleaved requests per image on the current per-crop build; CPU is
  utime+stime per request over the same six images. Multithreading is load-bearing
  rather than decorative: at one thread `07.png` (1743 ms) and `19.png` (1400 ms)
  both blow the 1 s budget, and six threads buy 2.35x wall for 2.6x the CPU. The
  efficiency is only 39% at six threads (4 threads reaches 55% and is 8% slower for
  27% less CPU), and above the physical count it inverts — 8 threads is 16% slower
  than 6, 12 is 30%, SMT siblings competing for the same execution units. Text is
  identical at every size (C1/C2 hold). Recognition is serial, one crop per `Run`,
  so the pool only helps inside a single `Run`; a process shows 17 threads.
- **D4. Nothing outside these files may be needed to build. [policy]** The C core
  must stay C11 + ONNX Runtime C API + libpng/libjpeg (OpenCV 5 has no C API, that
  is why the geometry exists at all); C++ may use OpenCV. No new runtime
  dependencies, no model swaps, no network access.

## E. Cost model (measured on the current build)

End-to-end per image, `cpp`, one request at a time (`OCR_DEBUG` set, which only
adds stderr prints): `04.png` ~0.19 s … `01.png` 0.51–0.53 s, `07.png` 0.52–0.55 s,
`19.png` 0.59–0.61 s. Absolute latency moves with the machine's state between runs
(~15%), so these are ranges over several probes; `benchmarks/history.jsonl` records
the tail of each run.

The stage split is what tells you which lever applies, and it differs by image
shape (`OCR_DEBUG`, medians of five requests, current build): **[measured]**

| image | detector | crops | recognition | wall | lines |
| --- | --- | --- | --- | --- | --- |
| 01.png | 459 | 4 | 21 | 486 | 2 |
| 18.png | 481 | 1 | 12 | 492 | 2 |
| 19.png | 462 | 2 | 87 | 552 | 3 |
| 06.png | 263 | 3 | 155 | 426 | 7 |
| 07.png | 157 | 6 | 304 | 469 | 12 |
| 13.png | 164 | 1 | 29 | 192 | 3 |

So **the strip screenshots are detector-bound and the multi-line ones are
recognition-bound**: `07.png` was 74% recogniser before B changed and is 65% now,
`19.png` 84% detector. Neither a canvas change nor a recognition change fixes both,
and the corpus-wide "detection 71% / recognition 29%" this section used to quote
(blank images at corpus dimensions) averaged those two regimes away.

Detector canvases were the reason the corpus is expensive. **[measured]** — sizes
reproduce exactly from the B rules, and they match the cores' own debug output:

| image | source | detector canvas | Mpx | before the cap |
| --- | --- | --- | --- | --- |
| 01.png | 652x86 | 4736x608 | 2.88 | 5568x736, 4.10 |
| 09.png | 217x33 | 4416x672 | 2.97 | 4832x736, 3.56 |
| 18.png | 300x46 | 4416x672 | 2.97 | 4800x736, 3.53 |
| 19.png | 1100x159 | 4544x640 | 2.91 | 5088x736, 3.74 |
| 10.png | 393x49 → padded 393x97 | 2976x736 | 2.19 | unchanged |
| 04.png | 275x277 | 736x736 | 0.54 | unchanged |

Corpus total 35.3 Mpx of detector canvas (mean 1.76, min 0.54), down from 38.5
Mpx; the cap binds on the four images that approached the budget, and leaves the
other sixteen alone. A thin strip is inflated hardest: `01.png`'s 86-px height
becomes 736, i.e. an 8.6x linear upscale of the image the model then has to
search.

Read the per-image numbers as what the user feels (A5): the ~361 ms corpus median
is an even blend of ten cheap images (0.18–0.45 s) and ten heavy ones, and the
heavy ones are where the 1 s budget was lost. After B, D3 and the buffer reuse in
E the tail is the **detector**: `19.png` 577 ms and `01.png` 512 ms are 81% and 94%
detector, while `07.png` — the image this document used to single out — sits at
546 ms.

Serial work, timed directly on the 4.10 Mpx canvas `01.png` used to take, against
`ocr_geometry.o`, with a synthetic probability field (20 groups, so treat the
component number as a lower bound for real text): **[measured]** — at today's
2.88 Mpx it scales with area to roughly 40 ms, ~8% of that request.

| serial step | ms | note |
| --- | --- | --- |
| resize to canvas | 9.0 | 652x86 → 5568x736 |
| NCHW normalise | 11.1 | writes 49 MB of floats |
| mask + 2x2 dilate | 4.3 | two 4.1 MB masks |
| connected components | 32.4 | both passes over 4.1 Mpx |
| **total** | **56.7** | ~11% of that image's request at the time |

Where the cycles actually go, per shape, sampled with `operf` as root (20
back-to-back requests each; the sampler throttled and lost samples, so read these
as shares, not exact percentages): **[measured]**

| image / symbol | 19.png (detector) | 01.png (strip) | 07.png (recogniser) |
| --- | --- | --- | --- |
| `libonnxruntime` (all model kernels) | 87.2% | 87.1% | 94.4% |
| kernel (`no-vmlinux`: syscalls, faults) | 7.9% | 7.6% | 1.8% |
| `libc` (memcpy/memset/malloc) | 3.5% | 3.8% | 2.3% |
| `ocr_group_pixels` (labelling + hull) | 0.60% | 0.75% | 0.27% |
| `ocr_nchw_normalize` | 0.24% | 0.19% | 0.12% |
| `ocr_resize_bilinear` | 0.22% | 0.21% | 0.21% |
| `ocr_detection_mask` | 0.07% | 0.08% | 0.03% |
| `ocr_warp_cubic` | 0.05% | 0.12% | 0.20% |
| `Engine::recognize` (crop assembly + decode) | 0.13% | 0.05% | 0.52% |

That table retires levers F2 and F3: the whole geometry is ~1.3% of a request, so
fusing the resize into the normalise or rewriting the labelling cannot pay for
itself. The synthetic serial table above (56.7 ms on a 4.1 Mpx canvas with 20
groups) is an upper bound that real images never approach — real strips have few
groups and the canvas has been capped since. ~90% of every request is ONNX
Runtime, which ships stripped, so `opreport` can only see it as one image; any
further work has to change *what* the runtime is given (canvas, model) or *which*
runtime it is. What was still ours to remove was page faults, and that is now gone
too (below).

### Allocation and page faults

Faults were the one place the `libc`/kernel share could still be ours, so they were
counted directly (`minflt` in `/proc/self/stat`), not inferred: one `19.png` request
took **153,636 minor faults**, ~600 MB of pages touched, and a single detector `Run`
accounted for ~114k of them. Splitting that between the runtime and our buffers
isolated the removable part — a freshly `malloc`ed 35 MB detector canvas costs
**28.4 ms** to touch against **7.4 ms** when the same buffer is reused, almost all
of it page faults. Both cores now keep the detector and recogniser
canvases and their scaled 8-bit copies on the engine for the life of the process,
grow them only when a shape needs more, and clear only the pad columns (the
normalise writes every other element). They also decode the runtime's output tensor
in place instead of copying it, removing an 11.6 MB detector copy and a 5.7 MB
logits copy per crop. Interleaved on the five heavy images: `19.png` −21 ms,
`18.png` −20, `09.png` −18, `07.png` −9, `01.png` −5; faults per `19.png` request
153,636 → 133,856, text byte-identical (C1/C2). The remaining faults are the
runtime's own per-`Run` activation allocations: nothing in our code allocates per
request any more except the crop list and the output string. The ORT arena would
remove more of them, but its price is 1.16 GiB (D2), so it stays opt-in.

Alternative runtimes were measured on the real canvases, same probe, best of N Runs
(**runtime** = which library, **EP** = execution provider inside it): **[measured]**

| runtime / EP | detector 3x640x4544 | recogniser 3x48x1680 |
| --- | --- | --- |
| **vendored 1.31.1, CPU (shipped)** | **415.8 ms** | **23.0 ms** |
| system `onnxruntime-cuda` 1.29, CPU | 631.1 ms | 30.6 ms |
| system 1.29, oneDNN EP | 702.7 ms | 104.6 ms |
| system 1.29, CUDA EP | does not load | does not load |

The shipped runtime is the fastest thing available here. The Arch package is 1.5x
slower on the detector and 1.3x on the recogniser with its own CPU kernels; its
oneDNN EP is worse still — 4.5x slower on the recogniser, because layout
conversions and per-op overhead dominate at a 48-px-high canvas; and its CUDA EP
cannot be created at all. `libonnxruntime_providers_cuda.so` was built against
cuDNN 7/8 while the system has cuDNN 9, so `cudnnGetConvolutionBackwardDataAlgorithm_v7`
is undefined. A GPU path therefore needs an ORT built against cuDNN 9, not a flag,
and XNNPACK needs a 1.31.1 build with `--use_xnnpack`; both are build projects.
The vendored library exports no provider factory functions at all (CPU only), and
`operf`'s provider-device list confirms it: only `CPUExecutionProvider`.


The 736 short side is load-bearing, not a dial that can be turned down for speed.
Detection is 81–94% of the request on the strips (`19.png` 486 of 598 ms, `01.png`
491 of 524 ms), so the short side was swept over the whole corpus with the 3.0 Mpx cap
and everything else unchanged: **[measured]**

| short side | mean detector | heavy mean wall | corpus mean | CER | images whose text moved |
| --- | --- | --- | --- | --- | --- |
| **736 (shipped)** | 325 ms | 562 ms | 381 ms | 1.34% | — |
| 640 | 271 ms | 573 ms | 339 ms | 1.31% | 6 |
| 600 | 228 ms | 499 ms | 296 ms | 1.34% | 5 |
| 560 | 196 ms | 439 ms | 266 ms | 1.48% | 7 |
| 512 | 156 ms | 375 ms | 233 ms | 1.31% | 7 |

Every step down moves text, and the movement is structural, not cosmetic: at 640 and
below `09.png` grows a spurious `2` line, `11.png` grows a one-glyph fragment line
(`L` at 640, `ぐ` at 512), `13.png` loses its emoji, and `05.png`/`14.png` merge two
lines into one. Some of those merges are *more* correct (`05.png`'s merged text then
matches the ground truth exactly), which is the point: the detector is one notch away
from re-segmenting lines, so these numbers cannot be read as free speed just because
the corpus CER happens to look flat. 736 stays. What is left at that size is the
model (~500 ms on a strip) plus ~40 ms of serial geometry (serial table above), and
neither shrinks without changing what is read.

Recognition: **101 crops over the 20 images** (1–12 per image; `07.png` has 12,
`09.png` and `17.png` have 1), **one session `Run` each**.

One crop per run is worth a lot and was counter-intuitive, so here is the evidence.
Timed in isolation against `PP-OCRv6_rec_small.onnx` on a 48x1680 canvas, 6 threads,
best of four: **24 ms per crop as a batch of one, 41 ms in a batch of six, 47 ms in
a batch of twelve.** The twelve crops of `07.png` are ~290 ms as twelve runs against
570 ms as one. Before the change that image spent 642 ms in the recogniser and
869 ms per request; it now spends **371 ms** and 582 ms. That is what moved the tail
from `07.png` to the detector-bound `19.png` (E table).

The cause is contention, not extra work, and it was measured rather than assumed.
Hardware counters (`perf_event_open`, user-space only because
`perf_event_paranoid` is 2, `inherit` so the pool threads are included; best Run of
each cell; no sampler can run here — `perf` is not installed, and `operf`
(oprofile 1.4.0) fails at `perf_event_open` with EPERM under `perf_event_paranoid`
2 while also wanting `kptr_restrict=0`, so symbol-level profiles need root):

| batch | 1 thread | 2 | 3 | 4 | 6 | scaling, 1 → 6 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 crop | 99.8 ms, IPC 3.15 | 54.0, 2.96 | 38.6, 2.87 | 30.3, 2.76 | **22.8, 2.62** | **4.4x** |
| 6 crops | 126.2 ms, IPC 2.83 | 69.7, 2.63 | 53.6, 2.38 | 45.5, 2.15 | **40.4, 1.85** | **3.1x** |
| 12 crops | 128.0, 2.76 | — | — | — | 44.2, 1.61 | 2.9x |

Per crop, wall clock and IPC (`instr / cycles`, cycles summed over the inherited
threads). Instructions per crop barely move (1.45–1.72 G): a batched `Run` does the
same arithmetic. What moves is cycles per instruction. Two separate effects, and
the earlier "the activations stop fitting in cache" note only described the first:

1. **Working set.** Even on one core a 6-crop Run costs 26% more per crop (IPC 3.15
   → 2.83) because six crops of activations overflow the 3 MiB per-core L2, where a
   single crop mostly fits.
2. **Shared bandwidth.** The 6-crop IPC keeps falling as the pool widens (2.83 at 1
   thread → 1.85 at 6) while the 1-crop IPC only drifts (3.15 → 2.62). Six cores
   streaming one 6x-larger tensor contend for L3/DRAM, which is why parallel
   scaling drops from 4.4x to 3.1x. Multi-core does not rescue batching here; it is
   what makes batching worst.

Neither counter set is complete: the generic `LLC` event reads 0 on this Zen 3 /
kernel, so last-level misses are inferred from IPC and scaling rather than counted,
and `cycles` includes pool spin-wait. Both would need raw events, i.e. root.

It also fixed accuracy rather than costing it. Padding every crop out to the widest
aspect in its group let the model read the zero columns: `13.png` gained a whole
phantom `5` line that no crop contained. Removing the padding drops that line and
corpus CER goes 1.38% → **1.34%**, i.e. back to the level the detector canvas budget
had cost (F1), with the 3.0 Mpx cap still in place.

Everything else tried on the recogniser was neutral or worse, and the knobs are
worth knowing before re-testing them: per-session thread counts (det/rec sized
independently) gained nothing over one shared size; running two rec batches
concurrently to overlap their `Run`s gained ~3% (inside noise) for extra threads;
batching at 12 was slower; and the vendored runtime has **no accelerated execution
provider compiled in** — `XNNPACK` and `OpenVINO` answer "not supported in this
build" — so `CPUExecutionProvider` (MLAS) is the only option.

## F. What follows — levers, and the assumption each depends on

1. **Detector canvas for strips. [done, B]** Detection cost scales with canvas
   area, and the 736 rule inflates the short side by 2.7x (`04.png`) to 22x
   (`09.png`). The images that approach the 1 s budget are all strip shapes with
   ≥3.5 Mpx canvases, so this is where the detector half of A5's goal is won. The
   implemented 3.0 Mpx cap shrank those four canvases by 20–30% (E) and bought
   `19.png` 155 ms and `01.png` 245 ms interleaved — but it also moves crops, so
   the budget was swept against the corpus' text:

   | budget | text vs uncapped | `19.png` wall |
   | --- | --- | --- |
   | 3.6 Mpx | all 20 images byte-identical | 979 ms |
   | **3.0 Mpx (shipped)** | 2 images, 1–2 chars each | **852 ms** |
   | 2.5 Mpx | a line splits on `19.png` (duplicated `雑`), spurious box on `09.png` | 838 ms |

   The cost at 3.0 is two tiny-glyph placements: `09.png`'s emoji is read as a
   different emoji (it was already wrong) and `18.png` moves a hyphen into the
   previous line, CER 1.34% → 1.38%. That is a text change, so it is recorded in
   history as one, not as a free win; 3.6 Mpx is the setting if text must stay
   byte-identical, and it gives back most of the saving. Below 3.0 the detector
   starts splitting lines, which is a quality cliff, not a trade. The cap is
   uniform shrink of both sides, so no glyph is distorted. Depends on A3, B, C3.
2. **Fuse resize + normalise. [measured out]** Tempting on paper — 74 MB of
   traffic per canvas — but the profile in E says `ocr_resize_bilinear` +
   `ocr_nchw_normalize` together are 0.3–0.5% of a request (~2–3 ms), so a fused
   kernel can return at most a few milliseconds. Not worth the byte-identity risk.
3. **Connected components. [measured out]** `ocr_group_pixels` is the largest piece
   of our own code and still only 0.27–0.75% of a request (2–4 ms): real images
   produce a handful of groups, not the twenty the synthetic microbenchmark used.
   A union-find rewrite would be invisible in the tail. Depends on B.
4. **Recognition batching. [done, B]** Every batch was padded to the widest crop's
   width and to at least 320 px, and the padding was real compute through a 48-row
   image — plus it invented a line on `13.png`. Recognition now runs one crop per
   `Run` at exactly its own aspect, which costs 24 ms per crop instead of 41 ms in a
   batch of six (E), took `07.png` from 642 to 371 ms of recognition, and improved
   CER 1.38% → 1.34% at the same time. Nothing here is a trade: it is faster and
   more accurate, and C1/C2 still hold because both cores share the change.
5. **Buffer reuse. [done, E]** Every request used to `malloc` its decoded image,
   canvas, masks, point lists, crops and output copies, and the detector canvas
   alone is 35 MB of floats. The canvases and the scaled copies now stay on the
   engine and the runtime's output tensor is decoded in place: faults per `19.png`
   request 153,636 → 133,856 and, interleaved over the five heavy images, −73 ms
   total (`19.png` −21, `18.png` −20, `09.png` −18, `07.png` −9, `01.png` −5), text
   unchanged. That was the last measurable cost in our own code — the profile's
   `libc` and kernel shares are now the runtime's allocations, not ours. ORT's CPU
   arena is a separate, larger retention trade (~15–19 ms median for ~1.16 GiB, D2)
   and is opt-in rather than the default. No output effect by construction, so C2
   verifies.
6. **Threads. [done, D3]** The old default sized the intra-op pool to the logical
   core count, which oversubscribed the six physical cores and cost ~20% on every
   image; the pool is now sized to the physical cores and is worth 220 ms on
   `07.png`, 220 ms on `19.png` and 265 ms on `01.png` interleaved (D3 table). This
   is the only lever measured so far that helps *every* image, including the
   recognition-bound one, which is why the sweep was worth running before more
   invasive changes. It is also free in accuracy terms (C1, C2 still hold).
   `OCR_THREADS` remains as an override for a shared or heterogeneous host.
   Remaining thread headroom under A5: only one user is waiting and the interval
   is ~1 s, so spending CPU to buy latency is acceptable here in a way it would not
   be for a batch service — but 7 and 8 threads were both *slower* than 6, so
   "more" is not the direction; the unexplored one is SPE (two requests' work never
   coexists under A1) and tiling one detector canvas across threads. Do not confuse
   any of this with the ~0.10 core-second/second the idle pool spin-waits (A5) —
   that is power, not latency.

## G. Protocol for claiming a win

```sh
make -C native
python3 benchmarks/check_cores.py  # both cores, all 20 images, byte compare + repeats
python3 benchmarks/run.py          # 5 trials: median, worst image, heavy, paired deltas,
                                   # budget verdict, and a record appended to history
```

`check_cores.py` exits non-zero on any difference, so it works as the build gate;
it takes about 40 s and needs no statistics. `run.py` is what tells you whether
the change actually helped. To find out *which stage* to aim at, run a daemon with
`OCR_DEBUG=1`: it prints one line per request with the ms spent in decode,
detector, crops and recogniser (`time det … rec …`), plus the detector
canvas size in Mpx — that is where the E table comes from. `OCR_THREADS` overrides
the intra-op pool size if you need to re-run the thread sweep on another host;
`OCR_MEM_ARENA=1` leaves ONNX Runtime's CPU arena enabled, which is how D2's trade
was measured.

Quote the paired delta (`cpp - c` / new build vs old) with its confidence
interval, the heavy-set delta and the per-image deltas for the shape you targeted,
and confirm exact match and CER are unchanged. A change that moves text is a change
in a different category and needs its own before/after CER and a look at
`mismatches`. For A5's budget the numbers to quote are `worst image` and `heavy`:
a change that shaves 10 ms off the corpus median but leaves `07.png` at 0.9 s has
not touched the problem. `benchmarks/history.jsonl` accumulates those two numbers
per run, and `python3 benchmarks/run.py --trend` prints the series, so "is the tail
closer to the 1 s budget than it was three runs ago" is answerable without keeping
old terminal output.

## H. Open questions

- Why did accuracy move between two builds of identical source (C3)? Until that is
  answered, some inference is not reproducible across builds, and only
  within-build byte comparison is a trustworthy oracle.
- What does detection actually cost per Mpx of canvas on this machine? The stage
  table in E gives 486–508 ms for 2.88–2.97 Mpx, i.e. ~170 ms/Mpx including the
  serial geometry, but that is four images; a proper sweep would separate the model
  from the fixed cost. This is now the tail, so it is the next thing worth knowing.
- ~~Why does one `Run` cost 1.7–2x more per crop when it carries six?~~ Answered
  in E with hardware counters: same instructions, falling IPC, +26% per crop even on
  one core (L2 overflow) and parallel scaling collapsing 4.4x → 3.1x as the shared
  memory subsystem saturates. It does not matter for latency as shipped (one crop
  per run is already the best measured arrangement); it would only matter to
  something that wanted batched recognition, and the fix there is not a flag — it is
  tiling the activations so a batch stays inside L3.
- **Nothing on the recogniser is left that is free.** `07.png` is at 546 ms and
  63% recogniser: 12 crops of aspect ~35 as 48x1680 canvases, one `Run` each, at
  ~24 ms per crop against a 48 px-high model. Splitting a long line changes what is
  read (a space is lost at each seam, CER 1.54%), a smaller-height or quantised
  model is a quality decision, and the runtime has no accelerated EP built in (E).
- **Quantisation is the remaining model-level lever.** Dynamic int8 on the detector
  and/or recogniser could plausibly halve the ~500 ms/~24 ms per canvas, but it
  changes the weights, so it needs the `onnxruntime.quantization` tooling and its
  own before/after CER on the corpus (C3) before it can be considered. Everything
  cheaper than that has now been measured and rejected (E).
- ~~Is the 736 short-side rule itself load-bearing, as distinct from the area cap?~~
  Answered in E: yes. All four candidates (640/600/560/512) re-segment lines on 5–7
  of 20 images — spurious fragment lines, a lost emoji, merged lines — so the detector
  floor on `19.png`/`01.png` is the model at 736 (~500 ms) plus ~40 ms of serial
  geometry, not a canvas dial.
- Should a resident daemon be quiet when idle? Today it spin-waits for ~0.10 s of
  CPU per idle second. Cheap to change, but it is a product decision, not a
  benchmark one, so it is deliberately not in F.

## Not covered yet

Assumptions about deployments, concurrency requirements above one, GPU/NPU use,
model replacement, and non-screenshot input (scans, photos with rotation) are out
of scope here and should be written down before any work assumes them.
