# OCR daemon benchmark

Two independent OCR daemons — one C11, one C++ — that answer `POST` with an
encoded screenshot and reply with the recognised text, plus the harness that
measures them on a fixed 20-image screenshots corpus.

Both link the same ONNX Runtime (the system one when it is 1.30 or newer,
otherwise the vendored 1.31.1) and the same
PP-OCRv6 models, so the benchmark compares two implementations and the two ways
they call that runtime, not two different OCR models.

## Contents

| path | what it is |
| --- | --- |
| `native/ocr_core_c.c` | C11 OCR core: ONNX Runtime **C** API, libpng/libjpeg, no OpenCV |
| `native/ocr_core.cpp` | C++ OCR core: ONNX Runtime C++ API, OpenCV 5 |
| `native/ocr_geometry.c` | crop geometry, detector-canvas rule and thread count shared by both cores (see below) |
| `native/c_daemon.c`, `native/cpp_daemon.cpp` | POSIX-socket HTTP frontends |
| `native/ocr_core.h` | the four-function C ABI both cores export |
| `native/ASSUMPTIONS.md` | the pipeline invariants and cost model that core optimisation rests on |
| `WRITEUP.md` | the optimisation record: what changed, what it bought, and what was rejected |
| `native/smoke.cpp` | asserts a white PNG comes back as empty text |
| `native/lib/` | vendored ONNX Runtime 1.31.1 (see `native/lib/NOTICE.md`, MIT) |
| `models/` | PP-OCRv6 det/rec models, with `manifest.json` hashes |
| `corpus/` | 20 screenshots and their ground truth (`manifest.json`) |
| `benchmarks/run.py` | the harness (Python standard library only) |
| `benchmarks/check_cores.py` | byte-compares both cores on the corpus, for gating core work |
| `benchmarks/candidates.json` | candidates, the run plan and the latency budget |
| `benchmarks/results.json` | output of the last run (raw samples included) |
| `benchmarks/history.jsonl` | one line per run: median, worst image, heavy, CER — the trend |
| `tools/model_manifest.py` | re-records `models/manifest.json` |

## Requirements

Arch Linux packages providing `opencv5` headers/libraries
(`/usr/include/opencv5`), `libpng`, `libjpeg`, a C11 compiler, `g++`, and
`python3`. ONNX Runtime *headers* come from the system package
(`/usr/include/onnxruntime`); the *library* is the system one when it is 1.30 or
newer, and the vendored 1.31.1 only when the system copy is older or missing (see
“Which ONNX Runtime”). The 1.29 headers compile against both because the ONNX
Runtime C API is backwards compatible.

## Build

```sh
make -C native                 # c-ocr-daemon, cpp-ocr-daemon
make -C native compile_commands.json   # what clangd reads
OCR_MODEL_DIR=models ./native/smoke && echo ok
OCR_DEBUG=1 OCR_MODEL_DIR=models ./native/cpp-ocr-daemon   # per-request stage timings on stderr
```

Two environment variables tune the runtime without a rebuild: `OCR_DEBUG` prints
the ms spent in decode, detector, crops and recognition per request
(with the detector canvas size and box coordinates), and
`OCR_THREADS` overrides the intra-op pool size that is otherwise derived from the
physical core count.

The Makefile asks `ldconfig` for the installed ONNX Runtime and links the system
library with `-lonnxruntime` when it is 1.30 or newer. Otherwise it links
`native/lib/libonnxruntime.so.1.31.1` with `-Wl,-rpath,'$ORIGIN/lib'`, falling
back to the system library if the vendored file is absent too, so a checkout
without `native/lib/` still builds. `ldd native/cpp-ocr-daemon | grep onnx`
shows which one a binary resolved.

## Install as a daemon

`make -C native install` copies a self-contained tree to
`$(PREFIX)` (default `~/.local/lib/ocr-daemon`):

```
ocr-daemon                     the binary
models/*.onnx                  det and rec models
lib/libonnxruntime.so.1.31.1   vendored runtime (only when the build fell back to it)
lib/libonnxruntime.so.1        symlink the binary resolves (same case)
```

When the build fell back to the vendored runtime, the binary's `$ORIGIN/lib`
RUNPATH makes it win over the system copy wherever the tree is moved, so the
service needs no `LD_LIBRARY_PATH`. When it linked the system library, the
service needs a system ONNX Runtime 1.30 or newer at run time. A user unit is
then:

```ini
[Service]
ExecStart=%h/.local/lib/ocr-daemon/ocr-daemon
Environment=OCR_MODEL_DIR=%h/.local/lib/ocr-daemon/models
Environment=OCR_PORT=9722
Restart=on-failure
```

Static linkage is not an option here: upstream ships only the shared
`libonnxruntime.so` (no `.a`), and static ORT would mean building 1.31.1 from
source. OpenCV's libraries are shared on this distribution too.

## Benchmark

```sh
python3 benchmarks/run.py                 # both candidates
python3 benchmarks/run.py --only c        # one candidate
```

It launches each candidate as a fresh process per trial (launch order rotating
per trial), warms it over the whole corpus in one interleaved pass, then issues
the requests **interleaved** across candidates with the order rotated per image
and per repetition. Every image is therefore served first by each candidate
equally often, so neither position nor thermal drift can systematically favour
one. The plan in `candidates.json` is `trials: 5, repetitions: 1, warmups: 1` —
100 timed requests per candidate in about 4 minutes, which is the setting to use
for core work. Raise `trials` for tighter tails.

It reports, per candidate: median with a bootstrap 95% confidence interval,
p90/p95/p99, startup, RSS, CPU time per request, the cost of a request with no
OCR work in it (`empty`), exact-match rate and CER. `benchmarks/results.json`
additionally holds every raw sample, per-trial and per-image medians, and each
mismatched transcription.

The table also carries the tail, because the corpus median hides it: `worst
image` is the slowest single image by median, and `heavy` is the median of that
candidate's slowest fifth of the corpus (four images). Both are per-image numbers,
not pooled ones, so one bad screenshot cannot be averaged away — which matches how
a user waits, one screenshot at a time.

Each run ends with the tail against the **latency budget** — one screenshot per
second, so `budget_ms` in `candidates.json` is 1000 — and with the change since
the previous recorded run:

```
tail budget: 1000 ms (one screenshot per second)
  c    worst 19.png 577 ms (58% of budget), 0/20 images over | previous 587 ms (-10), heavy 544 ms (-15), median 372 ms (-11)
  cpp  worst 19.png 579 ms (58% of budget), 0/20 images over | previous 606 ms (-26), heavy 539 ms (-14), median 381 ms (-20)
```

Once a run is recorded, each of those lines also carries the previous run's worst
latency and the deltas (`previous 1005 ms (+93), heavy …, median …`). Every run is
appended to `benchmarks/history.jsonl` — median, worst image and its latency,
heavy, p99, images over budget and CER per candidate — and `python3
benchmarks/run.py --trend` prints that series, so an optimisation is judged
against the budget and against what the number used to be rather than against one
run. Use `--no-history` for throwaway runs and `--budget-ms` to override the
budget for a single run.

Comparisons are **paired, not two medians**: every pair of candidates is compared
only on the requests both served (same image, same round), which cancels drift,
and the delta gets a 95% confidence interval plus a `SIGNIFICANT` /
`inside noise` verdict. On this machine per-trial medians move by up to ~50 ms, so
independent medians cannot resolve the ~2% differences that matter here. Each
comparison repeats its paired delta over the heavy set (the first candidate's
slowest fifth), and prints the largest **per-image** deltas, because the corpus is
uneven — a dense screenshot costs ~0.9 s, a thin strip 0.16 s — and a change that
helps one shape can hide inside the corpus-wide median.

The harness binds `127.0.0.1:9723` (one port per candidate above it), so it needs
a host that allows listening sockets; it leaves 9722 free.

`python3 benchmarks/check_cores.py` is the companion gate for core work: it posts
every corpus image to each candidate, compares the bytes across cores and across
repeated requests, and exits non-zero on any difference. Read
[`native/ASSUMPTIONS.md`](native/ASSUMPTIONS.md) before changing the cores — it
records which rules define the output and what a change is allowed to touch.

## Current results

Two candidates, vendored ONNX Runtime 1.31.1, 5 trials / 100 timed requests each,
exit 0, ~230 s (run of record: `benchmarks/history.jsonl`):

| candidate | median | 95% CI (median) | p90 | p95 | p99 | worst image | heavy | empty | cpu/req | startup | RSS | exact | CER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c | 341.7 ms | [237.2, 398.3] | 494.7 ms | 545.2 ms | 567.8 ms | 19.png 551 ms | 492.8 ms | 0.1 ms | 2070 ms | 394 ms | 237–332 MiB | 30% | 1.34% |
| cpp | 340.6 ms | [232.8, 400.8] | 494.9 ms | 559.7 ms | 605.1 ms | 19.png 561 ms | 493.5 ms | 0.2 ms | 2046 ms | 407 ms | 211 MiB | 30% | 1.34% |

The tail is the number that matters for this workload, and it now has room in it:
**0 of 20 images are above the 1 s arrival interval**, and the worst is at 55/56% of
it rather than 95%. Before the work described below the worst were `07.png`
949/979 ms (95%/98%) with two images over budget; the top of the corpus is now
`19.png` 551/561 ms, `18.png` 494/491 ms, `01.png` 491/496 ms, `09.png` 490/480 ms,
`07.png` 483/477 ms and `06.png` 416/420 ms. The last two rounds moved the corpus
median 372/382 → **361/362 → 342/341 ms**, the heavy-set median 544/539 →
**529/525 → 493/494 ms** and the worst image 587/606 → **577/579 → 551/561 ms**;
p90 is 495/495 ms against 553/573 before, so the whole distribution moved rather
than a single image.

The tail moved shape as well as down. `07.png` — twelve long lines and 65%
recogniser — is no longer the worst image, because recognition is now read one crop
per run at 24 ms per crop instead of 41 ms in a batch of six; the worst image is
`19.png`, which is 84% detector. That is where the next 100 ms would come from, and
`07.png` keeps a 45% margin instead of the 2–5% it had.

These numbers were collected with the machine at load ~6 (browser and agent), and
absolute latency drifts ±15% between runs, so read the ranking and the paired
deltas rather than the third digit. The recent changes are in our own code, not the
runtime: the cores keep their two NCHW canvases and the scaled 8-bit copies between
requests, decode the runtime's output tensor in place instead of copying it, and no
longer run the 180° classifier (`WRITEUP.md` §9). Buffer reuse drops minor faults
per `19.png` request from 153,636 to 133,856, and it is worth 21 ms on `19.png`,
20 on `18.png`, 18 on `09.png`, 9 on `07.png` and 5 on `01.png` in an interleaved
A/B — a freshly allocated 35 MB canvas costs 28.4 ms to touch against 7.4 ms reused,
almost all of it page faults. ONNX Runtime's own CPU arena is a larger trade and
stays off by default: re-measured on this build it is worth ~19 ms (C) and ~15 ms
(C++) of corpus median and ~27–29 ms of the heavy set, but it pins ~1.16 GiB of
cached activations (RSS 208–315 MiB → ~1.2 GiB). `OCR_MEM_ARENA=1` turns it on.

`cpp - c` = **−1.4 ms [−2.4, −0.0]** over 100 paired requests — technically outside
zero but a rounding error at this harness's resolution; earlier runs had it on both
sides of zero. C++ keeps ~25–120 MiB less resident (211 vs 237–332 MiB) because
`std::vector` grow-only buffers and `cv::Mat` allocation round differently from
`realloc`.

Absolute latency drifts between runs — the same binaries measured 383, 441, 442,
416 and 361 ms medians — so only within-run paired deltas are comparable, and the
worst image moves with it.

Accuracy is one number because both candidates return byte-identical text on all
20 screenshots. Per-image deltas are spread on both sides in the latest run
(`02.png` −17 ms for C, but `01.png` +18 ms and `06.png` +12 ms), which is why the
harness reports them per image instead of leaving it to the corpus median.

## Where the time goes

A request is the OCR core, essentially entirely:

| part | cost | share |
| --- | --- | --- |
| HTTP frontend (accept, read, answer, reject) | 0.1–0.2 ms | ~0.05% |
| OCR core (decode, detect, crop, recognise) | ~342 ms | ~99.9% |

CPU agrees from the other side: 2.07 s of CPU per request against 342 ms of wall
clock, i.e. the core runs ~6.1 threads wide — which is the intra-op pool described
below, so the CPU-time number moved with it rather than staying at the old ~7.

A root `operf` profile says where those cycles are: **87–94% inside ONNX Runtime**,
7–8% in the kernel and 2–4% in `libc` on the detector-bound shapes (syscalls and
page faults from activation allocation), and **~1.3% in our entire geometry** —
`ocr_group_pixels` 0.6–0.75%, the resize and normalise 0.2% each. So the code around
the models is not where the remaining time is; the runtime that executes them is.
The profile was sampled before the buffer reuse described above, so the kernel and
`libc` shares are now a little smaller and the runtime share a little larger.
Percentages only, because the sampler throttled and dropped samples.

Inside the core the split depends on the image's shape, and there are two regimes
rather than one ratio (`OCR_DEBUG=1` prints these stages per request):

| image | detector | crops | recognition | wall | lines |
| --- | --- | --- | --- | --- | --- |
| `01.png` | 459 ms | 4 ms | 21 ms | 486 ms | 2 |
| `18.png` | 481 ms | 1 ms | 12 ms | 492 ms | 2 |
| `19.png` | 462 ms | 2 ms | 87 ms | 552 ms | 3 |
| `06.png` | 263 ms | 3 ms | 155 ms | 426 ms | 7 |
| `07.png` | 157 ms | 6 ms | 304 ms | 469 ms | 12 |
| `13.png` | 164 ms | 1 ms | 29 ms | 192 ms | 3 |

**Strip screenshots are detector-bound; the multi-line ones are recognition-bound.**
Detection cost follows the detector canvas: the rule scales the short side to
736 px, so `01.png` (652x86) becomes a 2.9 Mpx canvas and a thin strip buys more
work than the extra scale is worth. `07.png` is the opposite case — twelve crops of
aspect ~35, each a 48x1680 recognition canvas, 63% of the request in the
recogniser even after it was made four times cheaper per canvas. A corpus-wide
"71% detection" average, as earlier versions of this README quoted, blends those
two regimes into a number that describes neither.

The detector's own canvas is not a dial either. Detection is 81–95% of a strip
request, so the 736-px short side was swept over the whole corpus (interleaved, cap
unchanged): 640 cuts mean detector time from 325 to 271 ms and 512 to 156 ms, but
every candidate below 736 re-segments lines somewhere — `09.png` grows a spurious `2`
line, `11.png` a one-glyph fragment line, `13.png` loses an emoji, `05.png` and
`14.png` merge two lines into one. So the strips stay detector-bound at ~500 ms of
inference plus ~40 ms of serial geometry, and further gains need a different detector
model, not a smaller canvas. Details in
[`native/ASSUMPTIONS.md`](native/ASSUMPTIONS.md) E.

## Getting under the 1 s budget

Three changes took the tail from 2 images over budget to none, and each acts on a
different stage:

**A detector canvas budget (3.0 Mpx).** `ocr_detection_canvas()` in the shared
`native/ocr_geometry.c` applies RapidOCR's 736-px short side and then caps the
result, shrinking both sides uniformly and flooring to the 32-px grid. Only the
four strip screenshots with ≥3.5 Mpx canvases are touched: `19.png` 5088x736 →
4544x640, `01.png` 5568x736 → 4736x608, and corpus canvas falls 38.5 → 35.3 Mpx.
Interleaved against the uncapped build it is worth 155 ms on `19.png` and 245 ms on
`01.png`.

The budget was chosen by sweeping it against the corpus' text, because a smaller
canvas changes the crops:

| budget | text | `19.png` wall |
| --- | --- | --- |
| none (736 rule) | reference | 1007 ms |
| 3.6 Mpx | all 20 images byte-identical | 979 ms |
| **3.0 Mpx (shipped)** | 2 images differ by 1–2 characters | 852 ms |
| 2.5 Mpx | a line splits on `19.png`, spurious box on `09.png` | 838 ms |

The `19.png` walls in that table were recorded before the thread and recognition
changes below, so only their ordering is meaningful; the same image is 577 ms
today.

So 3.0 Mpx is a deliberate, small accuracy trade rather than a free win: `09.png`'s
emoji is read as a different emoji (it was already wrong) and `18.png` moves a
hyphen into the previous line, worth +0.04 points of CER. 3.6 Mpx is the setting
for byte-identical text; below 3.0 the detector starts splitting lines.

**Intra-op threads sized to the physical cores.** ONNX Runtime defaults its pool to
the logical core count, but this host is 6 physical cores with SMT, so the default
put two pool threads on every core and cost ~20% on *every* image — including
`07.png`, which the canvas budget cannot help. `ocr_intra_op_threads()` reads
`/sys/devices/system/cpu/cpu0/topology/thread_siblings_list` and reports the
physical count (6 here); 7 and 8 threads were measurably worse than 6, and 12
worst. `OCR_THREADS` overrides it, and on a host without SMT nothing is set. Text
is unaffected. Re-swept on the current build over the six slowest images (best of
three, interleaved), six threads is **2.35x** the one-thread total — 7720 → 3282 ms —
and it is load-bearing: at one thread `07.png` (1743 ms) and `19.png` (1400 ms) are
both over budget. It is not free: CPU per request rises 1372 → 3523 ms, efficiency is
39%, 4 threads is 8% slower for 27% less CPU, and 8 and 12 threads are 16% and 30%
slower than 6 because SMT siblings compete for the same execution units.

**Recognition: one crop per Run.** The recogniser used to read up to six crops in
one Run, on a canvas padded out to the widest aspect in the group. Batching is the
wrong trade for this workload, and measurably so: timed in isolation on a 48x1680
canvas the model costs **24 ms per crop as a batch of one, 41 ms in a batch of six
and 47 ms in a batch of twelve** (best of four, 6 threads). Twelve single runs are
~290 ms where one batch of twelve is ~570 ms.

The reason is contention, not extra work, checked with hardware counters
(`perf_event_open`; `perf` itself is not installed here). Instructions per crop are
flat — 1.45 to 1.72 G whether one crop or six — while IPC falls 3.15 → 2.83 → 1.61 as
the batch grows, and for the wide batches it keeps falling as threads are added
(6 crops: IPC 2.83 on one thread, 1.85 on six; 1 crop: 3.15 on one, 2.62 on six).
Six crops of activations overflow the 3 MiB per-core L2, so a batched Run costs 26%
more per crop even single-threaded, and then six cores streaming one oversized
tensor saturate shared L3/DRAM, which drops parallel scaling from 4.4x to 3.1x. The
full table is in [`native/ASSUMPTIONS.md`](native/ASSUMPTIONS.md) E. `07.png` went
from 642 to 371 ms of recognition — 869 → 582 ms per request under `OCR_DEBUG` and
949 → 546 ms by the latest run — and `06.png` 664 → 506 ms; no image got
slower, and the corpus median fell 442 → 416 ms.

The padding was an accuracy bug as well. The zero columns to the right of a narrow
crop were read as content: `13.png` picked up a phantom `5` line that is in none of
its crops. Reading each crop on its own width deletes that line, so corpus CER
improves 1.38% → **1.34%** — back to where it was before the canvas-budget trade,
which stays in place. This change costs nothing and gives up nothing.

Two things were measured and left out of the default path, with numbers: ONNX
Runtime's CPU memory arena (~19 ms for C and ~15 ms for C++ of corpus median, and
~27–29 ms of the heavy set on the current build, but it pins ~1.16 GiB of cached
buffers for the corpus' fifty-odd canvas widths, so it is behind
`OCR_MEM_ARENA=1`) and every accelerated
execution provider — `XNNPACK` and `OpenVINO`
answer "not supported in this build" from the vendored 1.31.1 runtime, so the CPU
provider is the only one there is. Batch size 12 was slower than 6, per-session
thread counts gained nothing over one shared size, and running two recognition
batches concurrently was worth ~3%, inside noise.

## How the cores relate

The two cores decode, detect, crop, recognise and assemble text
independently, but share one implementation of the **crop geometry**
(`native/ocr_geometry.{c,h}`): OpenCV-compatible 11-bit fixed-point bilinear
resize, 2x2 dilation at anchor (0,0), 8-connected foreground and 4-connected hole
components, monotone-chain convex hull, rotating-calipers minimum-area rectangle,
perspective solve and bicubic warp with replicated borders, scanline box scoring,
reading-order sort. Those are the steps where a one-pixel difference changes
recognised text, so sharing them makes the benchmark compare runtimes and calling
code rather than OCR quality.

The C core needs them because **OpenCV 5 ships no C API** (zero `cv*` symbols in
`libopencv_core/imgproc/imgcodecs/geometry`). The only C++ the C daemon links is
`libstdc++`, because ONNX Runtime is a C++ library behind a C API.

The shared resize is not the naive form: the horizontal pass is computed once per
source row into a two-entry cache instead of once per output row, which halves
`07.png`'s detector-scale resize (7.2 → 4.07 ms). It is verified byte-identical
to a straightforward transcription of the documented algorithm, so the speedup
cannot move a pixel.

## No 180-degree classifier

The two PP-OCRv6 models read text as-is and do not classify orientation. Earlier
builds ran RapidOCR's separate 180° classifier on every crop and re-recognised the
ones it proposed flipping. Desktop screenshots are upright and the classifier
changed no corpus output, so it was removed: text stays byte-identical on all 20
images while the median drops 15–21 ms. The measurements and the cost on a
genuinely upside-down image are in
[`WRITEUP.md`](WRITEUP.md#9-dropping-the-180-classifier).

## Which ONNX Runtime

1.31.1 is vendored for systems that still ship 1.29 or older, which is ~20%
slower here (recognition at `6x3x48x1568`, 6 threads: 307 ms on 1.29 vs 229 ms on
1.31).
The gain arrived with **1.30**, not 1.31 — measured at core level with paired,
order-alternating runs:

| runtime | core-level paired delta | verdict |
| --- | --- | --- |
| 1.29 (system) | +82.3 ms, +20.5%, CI [+60.1, +102.3] | significant |
| 1.30.1 | +1.1 ms, CI [−8.1, +8.7] | inside noise |
| 1.31.1 (vendored fallback) | baseline | — |

1.30.1 was measured with its own official release build and then removed from the
workspace once it was shown equivalent; to repeat that, drop its library in a
subdirectory of `native/lib/` and link a variant binary with
`-l:libonnxruntime.so.1.30.1 -Wl,-rpath,'$ORIGIN/<that subdirectory>'` — the two
versions share the SONAME `libonnxruntime.so.1`, so they cannot sit in the same
directory.

Vendoring a prebuilt shared library means ONNX Runtime is not visible to LTO, so
`-march=native -flto` only speeds up our own preprocessing (detector `nchw`
3.6 → 2.3 ms, `rec.pre` 4.6 → 3.4 ms, ~2.5 ms of a ~950 ms request). It is
deliberately not enabled.

Alternatives were measured on the real canvases and are all worse or unavailable
here: the Arch `onnxruntime-cuda` 1.29 package is 1.5x slower on the detector with
its own CPU kernels, 1.7x slower with its oneDNN EP (4.5x on the recogniser), and
its CUDA EP fails to load because it was built against cuDNN 7/8 while the system
has cuDNN 9. The vendored 1.31.1 CPU build stays, and its provider list is CPU-only.
Numbers in [`native/ASSUMPTIONS.md`](native/ASSUMPTIONS.md) E.

## Interpreting the results

- The native detector grows a **minimum-area rectangle** instead of applying
  RapidOCR's Shapely/Clipper polygon offset. That approximation is shared by both
  cores, so it is a property of this pipeline, not a difference between
  candidates; inspect `mismatches` in `results.json` before trusting it.
- The corpus accuracy is a single number because both cores agree: 30% exact
  match, 1.34% CER. Two changes are attributable inside that number: the 3.0 Mpx
  detector canvas moved `09.png`'s emoji and `18.png`'s hyphen placement (the trade
  the budget sweep table records, +0.04 points), and reading one crop per run
  removed the phantom `5` line that batch padding had invented on `13.png` (−0.04
  points), so the two cancel and the figure is back to the pre-canvas-budget level.
  Exact
  match is harsh here — a newline or trailing space where the ground truth
  differs fails the whole image — and it drifts by a character between builds of
  identical source (25% / 1.38% was measured before), so treat sub-0.1%-CER
  differences that are *not* attributable to a stage as noise. When the RapidOCR
  Python pipeline was still part of the harness the two sides agreed on 15 of 20
  images, with the differences split between them (a stray quote, an emoji, one
  CJK glyph).
- Per-trial medians move by up to ~50 ms across a five-trial run, so never
  compare independently collected medians at this scale; use the paired deltas,
  which cancel drift.
- p99 rests on 100 samples and is a coarse tail.
- This measures localhost HTTP latency only. It runs requests back-to-back; the
expected use is one user sending at most about one screenshot per second.
Measured at that cadence, latency is indistinguishable from the back-to-back
numbers here (within run-to-run noise), so these transfer — but a user feels the
per-image tail, not the median, and the slow ones are the dense strips and the
multi-line screenshot (`07.png`), for two different reasons. See
[`native/ASSUMPTIONS.md`](native/ASSUMPTIONS.md) A5 and E.

## Provenance

`native/lib/` holds the CPU `libonnxruntime.so.1.31.1` from the
`onnxruntime==1.31.1` wheel (MIT, see `native/lib/NOTICE.md`). `models/` holds the two
RapidOCR 3.10.0 assets with their SHA-256 in `models/manifest.json`; run
`python3 tools/model_manifest.py` after replacing one. `corpus/` documents how its
screenshots were transcribed. No Python daemon is part of the benchmark any more:
C and C++ are the candidates.
