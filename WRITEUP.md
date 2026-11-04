# Optimising the OCR daemons — what was done and what it bought

This is the record of the optimisation campaign on the two native OCR daemons
(C11 and C++), measured on the 20-screenshot corpus with the harness in
`benchmarks/`. It covers what changed, the evidence for each change, the
progression of the recorded runs, and the levers that were measured and left
alone.

Scope throughout: the models (`PP-OCRv6_det_small` / `PP-OCRv6_rec_small`), the
vendored ONNX Runtime **1.31.1**, and the pipeline
rules in `native/ASSUMPTIONS.md` B were fixed. Optimisation changed *how* the
work is arranged, not what is read.

## Result in one table

First and last recorded runs of the campaign (`benchmarks/history.jsonl`), same
corpus, same models, same runtime, 5 trials / 100 timed requests per candidate:

| | median | heavy (slowest fifth) | worst image | over 1 s | exact | CER |
| --- | --- | --- | --- | --- | --- | --- |
| first run, C | 441.3 ms | 945.3 ms | 19.png 1005 ms | 2/20 | 30% | 1.34% |
| first run, C++ | 446.1 ms | 950.4 ms | 07.png 1045 ms | 2/20 | 30% | 1.34% |
| **final, C** | **341.7 ms** | **492.8 ms** | **19.png 551 ms** | **0/20** | 30% | 1.34% |
| **final, C++** | **340.6 ms** | **493.5 ms** | **19.png 561 ms** | **0/20** | 30% | 1.34% |

The tail moved further than the median: the worst image went from over the 1 s
arrival interval to 55% of it, and the heavy-set median fell 48% while the corpus
median fell 23%. Accuracy did not move in the net: the two changes that altered
text cancelled (a +0.04-point canvas trade and a −0.04-point padding bug fix), and
removing the direction classifier (§9) left every output byte-identical. Both
cores still return byte-identical text on all 20 images.

Absolute latency drifts ±15% between runs on this machine (load, background
agents), so the first-vs-final comparison above is directional. Every individual
lever below was judged with an *interleaved, paired* A/B and its confidence
interval, not by comparing two separate runs.

## How the work was measured

The measurement harness is as much a part of this as the code, because the early
version could not resolve the differences being chased.

- **Interleaved request order.** Candidates are warmed in one interleaved pass,
  then every request is issued across candidates with the order rotated per image
  and per repetition. Every image is served first by each candidate equally often,
  so thermal drift and ordering cannot systematically favour one side.
- **Paired comparisons, not two medians.** Each candidate pair is compared only on
  the requests both served (same image, same round), with a bootstrap 95%
  confidence interval and a `SIGNIFICANT` / `inside noise` verdict. Per-trial
  medians move by up to ~50 ms here, so independent medians cannot resolve the
  ~2% differences worth having.
- **Tail metrics.** A user sends at most about one screenshot per second and
  waits for each one, so the harness reports the slowest image by median and the
  median of the slowest fifth of the corpus (`heavy`) next to the corpus median.
  A run ends with both against the **1000 ms budget**.
- **History.** Every run appends median, worst image and its latency, heavy, p99,
  over-budget count and CER to `benchmarks/history.jsonl`;
  `python3 benchmarks/run.py --trend` prints the series. That is how a change is
  judged against the budget *and* against what the number used to be.
- **A cheap correctness gate.** `benchmarks/check_cores.py` posts all 20 images to
  both cores and byte-compares the results across cores and across repeated
  requests. It exits non-zero on any difference and takes ~40 s, so it gates every
  core change before the timed run.
- **Practical run time.** 5 trials × 20 images × 2 candidates is 200 timed
  requests plus 40 warmups, about 230 s — under the five minutes that makes it
  usable during iteration. That plan (`trials: 5, repetitions: 1, warmups: 1`) is
  the recorded setting.

## The optimisations

### 1. One geometry for both cores

The C and C++ cores originally used their own crop code: OpenCV's
`getPerspectiveTransform` + `warpPerspective` on one side, a transcription on the
other. They disagreed on crops by a pixel, which changed recognised text, so the
benchmark was partly comparing geometry rather than runtimes.

All crop geometry now lives in `native/ocr_geometry.c` and is compiled into both
cores: OpenCV-compatible 11-bit fixed-point bilinear resize, 2×2 dilation at
anchor (0,0), 8-connected foreground and 4-connected hole components,
monotone-chain convex hull, rotating-calipers minimum-area rectangle, perspective
solve, bicubic warp with replicated borders, scanline box scoring and reading-order
sort. The C core needs this because **OpenCV 5 ships no C API**; the C++ core calls
the same code so the two cannot drift again.

The shared resize is not the naive form: the horizontal pass is cached per source
row (two-entry cache) instead of recomputed per output row, which halved the
detector-scale resize on `07.png` (7.2 → 4.07 ms). It is verified byte-identical
to a straightforward transcription of the documented algorithm, so the speedup
cannot move a pixel.

### 2. A detector canvas budget for the strips

Detection cost scales with canvas area, and the 736-px short side inflates a thin
strip by up to 22×. Four strip screenshots landed at 3.5–4.1 Mpx canvases, and
they were the images approaching the budget. `ocr_detection_canvas()` now applies
the 736 rule and then caps the result at **3.0 Mpx**, shrinking both sides
uniformly and flooring to the 32-px grid: `19.png` 5088×736 → 4544×640, `01.png`
5568×736 → 4736×608. Corpus detector canvas fell 38.5 → 35.3 Mpx.

Interleaved, the cap is worth **155 ms on `19.png` and 245 ms on `01.png`**. It
was swept against the corpus' text rather than picked for speed:

| budget | text vs uncapped | `19.png` wall at the time |
| --- | --- | --- |
| 3.6 Mpx | all 20 images byte-identical | 979 ms |
| **3.0 Mpx (shipped)** | 2 images differ by 1–2 characters | 852 ms |
| 2.5 Mpx | a line splits on `19.png`, spurious box on `09.png` | 838 ms |

So 3.0 Mpx is a deliberate small accuracy trade: `09.png`'s emoji is read as a
different emoji (it was already wrong) and `18.png` moves a hyphen into the
previous line, worth +0.04 points of CER. 3.6 Mpx is the setting for byte-identical
text; below 3.0 the detector starts splitting lines, which is a quality cliff.

### 3. Intra-op threads sized to the physical cores

ONNX Runtime defaults its intra-op pool to the logical core count. This host is 6
physical cores with SMT (12 logical), so the default ran two pool threads per core
and cost ~20% on *every* image. Both cores now size the pool from
`/sys/devices/system/cpu/cpu0/topology/thread_siblings_list`
(`ocr_intra_op_threads()`); `OCR_THREADS` overrides it.

Re-swept on the per-crop build over the six slowest images, best of three,
interleaved:

| threads | total wall | vs 1 thread | CPU per request |
| --- | --- | --- | --- |
| 1 | 7720 ms | 1.00× | 1372 ms |
| 2 | 4736 ms | 1.63× | 1685 ms |
| 3 | 3880 ms | 1.99× | 2081 ms |
| 4 | 3546 ms | 2.18× | 2556 ms |
| 5 | 3468 ms | 2.23× | 3076 ms |
| **6 (physical)** | **3282 ms** | **2.35×** | 3523 ms |
| 8 | 3808 ms | 2.03× | 5344 ms |
| 12 (logical) | 4267 ms | 1.81× | 7863 ms |

Multithreading is load-bearing, not decorative: at one thread `07.png` (1743 ms)
and `19.png` (1400 ms) both blow the budget. It is not free either — six threads
buy 2.35× wall for 2.6× the CPU (39% efficiency); 4 threads is 8% slower for 27%
less CPU, and above the physical count it inverts because SMT siblings compete for
the same execution units. Text is identical at every size.

### 4. Recognition: one crop per `Run`

Recognition used to read up to six crops in one `Run`, on a canvas padded to the
widest aspect in the group. Timed in isolation on a 48×1680 canvas, 6 threads,
best of four:

| arrangement | per crop | for `07.png`'s 12 crops |
| --- | --- | --- |
| 1 crop per `Run` | **24 ms** | ~290 ms |
| batch of 6 | 41 ms | ~570 ms |
| batch of 12 | 47 ms | — |

The reason is contention, not extra arithmetic, checked with hardware counters:
instructions per crop are flat (1.45–1.72 G) while IPC falls 3.15 → 2.83 → 1.61 as
the batch grows, and for wide batches it keeps falling as threads are added (6
crops: IPC 2.83 on one thread, 1.85 on six; 1 crop: 3.15 on one, 2.62 on six). Six
crops of activations overflow the 3 MiB per-core L2, and then six cores streaming
one oversized tensor saturate shared L3/DRAM, which drops parallel scaling from
4.4× to 3.1×.

Every crop is now read on a canvas of exactly its own aspect, one `Run` each.
`07.png` went from 642 to 371 ms of recognition, `06.png` 664 → 506 ms, and the
corpus median fell with it.

### 5. The padding bug that the batching hid

The zero columns added to pad a narrow crop to the group's width were read as
content: `13.png` picked up a phantom `5` line that is in none of its crops.
Reading each crop on its own width removed it, improving corpus CER 1.38% →
**1.34%** — back to the level the canvas budget had cost. This one is unusual in
being both faster and more accurate.

### 6. The 180° classifier was a proposal, not a verdict

Obeying the classifier on every crop was worse than not running it: on `07.png` it
scores a thin Dutch line 180° at 0.99 confidence, and the recogniser then averages
the rotated and upright boxes into `we slak`. That single line is the difference
between ~1.3% and ~3.6% CER.

Both cores therefore treated the verdict as a proposal that the recogniser
arbitrates:

1. propose a rotation only when the softmax is at least 0.9 for upside down;
2. when it does, recognise the crop as-is *and* rotated, and keep the rotation
   only if that reading has strictly higher mean confidence.

The recogniser reads glyphs; the classifier reads a 48×192 thumbnail, so the tie
break goes to the recogniser. It cost one extra recognition on proposed lines only
and helped on genuinely upside-down input, though it did not rescue a whole
rotated page (§9). It was removed once the
workload turned out to be upright desktop screenshots; §9 has that measurement.

### 7. Stop re-faulting the canvases and copying the outputs

With the runtime itself ~90% of a request and the geometry ~1.3%, the only
remaining cost in our own code was memory handling, so it was counted rather than
guessed. A `19.png` request took **153,636 minor faults** — ~600 MB of pages
touched — and one detector `Run` accounted for ~114k of them. Isolating our share
was decisive: a freshly `malloc`ed 35 MB detector canvas costs **28.4 ms** to touch
against **7.4 ms** when the same buffer is reused, almost entirely page faults.

Changes, in both cores:

- the detector and recogniser NCHW canvases stay on the engine for the process
  lifetime;
- their scaled 8-bit copies do too (the detector one is ~10 MB per request);
- buffers grow only when a shape needs more, and only the right-hand pad columns
  are cleared, because the normalise writes every other element;
- the runtime's output tensor is decoded **in place** instead of copied — that
  removes an 11.6 MB detector map copy and a 5.7 MB logits copy per crop.

Interleaved A/B on the five heavy images on the C++ build — the C core got the
same change once it was validated — is `19.png` −21 ms, `18.png` −20, `09.png`
−18, `07.png` −9, `01.png` −5, **−73 ms** total, with faults per `19.png` request
down to **133,856** and text byte-identical. The full-corpus harness re-run of the
same change is noisier and diluted by the 15 light images (Appendix B): C −2.0 ms
[−5.7, +0.8], heavy −5.2 ms; C++ −2.4 ms [−4.4, −0.3], heavy −13.7 ms.

The remaining faults are the runtime's own per-`Run` activation allocations.
Nothing in our code allocates per request any more except the crop list and the
output string. ONNX Runtime's CPU arena would remove more of them, and was
re-measured on this build: ~19 ms (C) and ~15 ms (C++) of corpus median and
~27–29 ms of the heavy set, but it pins ~1.16 GiB of cached buffers (RSS 208–315
MiB → ~1.2 GiB). That is the wrong side of a one-user, one-second workload, so it
stays **off by default** behind `OCR_MEM_ARENA=1`.

### 8. Runtime and model provenance

- ONNX Runtime **1.31.1** is vendored in `native/lib/`, with a Makefile fallback
  to the system library. The distribution's 1.29 is +82.3 ms (+20.5%, CI
  [+60.1, +102.3]) slower at core level; 1.30.1 is inside noise (+1.1 ms,
  [−8.1, +8.7]), so 1.31.1 is the version that matters and it stays.
- System alternatives were measured on the real canvases and are worse or broken:
  the Arch `onnxruntime-cuda` 1.29 package is 1.5× slower on the detector with its
  own CPU kernels (631 vs 416 ms) and 1.3× on the recogniser; its oneDNN EP is
  4.5× slower on the recogniser; its CUDA EP cannot load at all because
  `libonnxruntime_providers_cuda.so` was built against cuDNN 7/8 and the system has
  cuDNN 9. The vendored library is CPU-only and has no provider factories.
- LTO was tried and deliberately not enabled: ONNX Runtime is a prebuilt shared
  library, so `-march=native -flto` only speeds up our own preprocessing (~2.5 ms
  of a ~950 ms request).

### 9. Dropping the 180° classifier

The workload is upright desktop screenshots, and PP-OCRv6 det/rec carry no
orientation stage of their own. The classifier had one job that never paid for
itself on this corpus: it ran on all 101 crops, proposed a flip once (`07.png`),
and the recogniser then rejected the rotated reading after the extra recognition
had already been paid for.

A build with the classifier removed — model not loaded, no proposal, no second
recognition — was compared against the previous binaries, interleaved over 60
paired requests per core:

| core | median delta | 95% CI | heavy delta | text |
| --- | --- | --- | --- | --- |
| C | −15.2 ms | [−21.4, −10.5] | −38.7 ms | byte-identical |
| C++ | −21.2 ms | [−25.3, −13.5] | −25.1 ms | byte-identical |

The recorded 5-trial runs (08:41 → 06:56, the latter with the classifier gone)
move C 360.9 → 341.7 ms median and 529.0 → 492.8 heavy; C++ 361.5 → 340.6 and
525.4 → 493.5. Some of that is the usual drift; the paired A/B above is the
attribution. Removing the model also takes ~0.5 MiB RSS and ~60 ms of C++ startup.

The cost is confined to upside-down input. Feeding all 20 screenshots rotated
180°: the classifier build scored 72.84% CER (1 exact), the no-classifier build
82.95% (0 exact). That is a real regression on a case this workload never sees;
upright text is identical.

## The recorded progression

`benchmarks/history.jsonl`, in order. Median, heavy median and worst image per
candidate:

| timestamp | C median | C heavy | C worst | C++ median | C++ heavy | C++ worst | over |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 05:17 | 441.3 | 945.3 | 19.png 1005 | 446.1 | 950.4 | 07.png 1045 | 2/20 |
| 06:10 | 441.9 | 723.5 | 07.png 949 | 442.2 | 760.5 | 07.png 979 | 0/20 |
| 06:48 | 394.4 | 584.5 | 19.png 621 | 400.3 | 589.5 | 19.png 636 | 0/20 |
| 06:55 | 415.7 | 595.8 | 19.png 665 | 420.0 | 600.4 | 19.png 668 | 0/20 |
| 08:05 | 371.8 | 543.9 | 19.png 587 | 381.5 | 538.7 | 19.png 606 | 0/20 |
| 08:41 | 360.9 | 529.0 | 19.png 577 | 361.5 | 525.4 | 19.png 579 | 0/20 |
| 06:56 | 341.7 | 492.8 | 19.png 551 | 340.6 | 493.5 | 19.png 561 | 0/20 |

Two caveats for reading the series:

- The 06:48 → 06:55 pair is the same per-crop build with ONNX Runtime's arena on
  and off; disabling it cost ~0 within noise and returned RSS from 1.15 GiB to
  ~200 MiB. It is not a code improvement.
- Run-to-run drift on this machine is ±15%, so small steps in the sequence are not
  attributable to individual changes. The reliable per-change numbers are the
  paired A/Bs quoted above.

The 08:05 → 08:41 pair brackets the memory work: 08:05 is the recorded baseline
and 08:41 is after it. Those two runs moved the worst image 587/606 → **577/579 ms**,
the heavy median 544/539 → **529/525 ms** and the corpus median 372/382 →
**361/362 ms**; CPU is 2.07 s per request on the final build. The last row (06:56)
adds the classifier removal (§9): worst 577/579 → **551/561 ms**, heavy 529/525 →
**493/494 ms**, median 361/362 → **342/341 ms**. Part of that is drift — each
change's own paired measurement is in Appendix B and §9.

## Measured and rejected

Every one of these was tried or measured, not assumed away:

| lever | result |
| --- | --- |
| Detector short side below 736 | 640/600/560/512 cut detector time (325 → 271/228/196/156 ms mean) but re-segment lines on 5–7 of 20 images: spurious fragment lines, a lost emoji, merged lines. 736 is the model's floor. |
| Recognition batching | 41 ms per crop at batch 6, 47 at batch 12, against 24 as a single `Run`; and padding invented a line. |
| More intra-op threads | 7/8/12 threads are 16–30% slower than 6 on this SMT host. |
| Per-session thread counts | No gain over one shared size. |
| Two recognition batches concurrently | ~3%, inside noise, for extra threads. |
| Fusing resize into NCHW normalise | Geometry is ~1.3% of a request; the whole fused kernel can return ~2–3 ms. |
| Rewriting connected components | `ocr_group_pixels` is our largest function at 0.27–0.75% of a request. |
| ORT CPU arena by default | ~15–19 ms median / ~27–29 ms heavy for ~1.16 GiB resident; opt-in instead. |
| Accelerated EPs (`XNNPACK`, `OpenVINO`, oneDNN, CUDA) | Not compiled into the vendored build, or slower, or ABI-broken against the system's cuDNN. |
| System ONNX Runtime 1.29 | +20.5% at core level; 1.30.1 equivalent to 1.31.1. |
| LTO / `-march=native` | Only reaches our preprocessing (~2.5 ms of a ~950 ms request), not the prebuilt runtime. |

## Where the time goes now

A root `operf` profile (20 requests per shape; the sampler throttled, so read
shares not exact percentages) puts **87–94% of a request inside ONNX Runtime**,
7–8% in the kernel and 2–4% in `libc` on the detector-bound shapes, and ~1.3% in
our entire geometry. That is the basis for the ordering of this document: the
runtime is the model execution, and everything else is a rounding error except the
memory handling, which was removed last.

The shape split still matters for any future work:

| image | detector | crops | recognition | wall | lines |
| --- | --- | --- | --- | --- | --- |
| `01.png` | 459 ms | 4 ms | 21 ms | 486 ms | 2 |
| `18.png` | 481 ms | 1 ms | 12 ms | 492 ms | 2 |
| `19.png` | 462 ms | 2 ms | 87 ms | 552 ms | 3 |
| `06.png` | 263 ms | 3 ms | 155 ms | 426 ms | 7 |
| `07.png` | 157 ms | 6 ms | 304 ms | 469 ms | 12 |
| `13.png` | 164 ms | 1 ms | 29 ms | 192 ms | 3 |

Strip screenshots are detector-bound; the multi-line ones are recognition-bound.
`19.png` is the worst remaining image at 84% detector, and its cost is the model on
a 2.9 Mpx canvas, not overhead around it.

## What the campaign taught

- **Measure the tail, not the average.** The corpus median hid a 1 s image behind
  a 0.2 s one; the heavy-set and worst-image columns are what the budget is about.
- **Paired and interleaved, or don't bother.** Drift on this host is larger than
  most of the wins; a before/after pair of independent runs cannot see them.
- **Byte identity is a cheap, strict oracle.** Text is deterministic per process
  and identical across cores, so `check_cores.py` catches a geometry regression in
  seconds, before spending four minutes on timings. CER is the secondary check for
  changes that deliberately move text.
- **A batch is not automatically faster.** Six crops per `Run` cost 26% more per
  crop even single-threaded because the activations leave L2, and the padding it
  required invented a line of text.
- **Check the profile before optimising the code you can see.** The geometry is
  the readable part of this project and it is 1.3% of the time; page faults in
  buffer setup were worth 73 ms and were invisible in the source.
- **Measure the rejected levers too.** The arena, oneDNN, CUDA, LTO and smaller
  canvases all sound like wins and none of them is, here. Writing down why keeps
  them from being re-tried.

## Reproducing

```sh
make -C native
python3 benchmarks/check_cores.py          # 20/20 byte-identical, ~40 s
python3 benchmarks/run.py                  # 5 trials, ~230 s, appends to history
python3 benchmarks/run.py --trend          # the recorded series
OCR_DEBUG=1 OCR_MODEL_DIR=models ./native/cpp-ocr-daemon   # per-stage timings
OCR_MEM_ARENA=1 python3 benchmarks/run.py --only cpp       # the arena trade
```

The pipeline invariants and the per-stage cost model behind these numbers are in
[`native/ASSUMPTIONS.md`](native/ASSUMPTIONS.md); the current results and the
harness' own documentation are in [`README.md`](README.md).

---

## Appendix A — raw `operf` profile

`operf` (oprofile 1.4.0) attached to a running `cpp-ocr-daemon`, 20 back-to-back
requests per shape; event `CPU_CLK_UNHALTED`, count 100000, no callgraph (the
vendored runtime is stripped). The sampler throttled and lost samples, so these
are shares of the surviving samples, not absolute costs. Reports were taken before
the buffer-reuse change in §7, so the kernel and `libc` shares still contain the
per-request canvas faults that change removed.

### A.1 By module

| module | 19.png detector | 01.png strip | 07.png recogniser |
| --- | --- | --- | --- |
| total samples | 1,787,286 | 1,526,692 | 1,788,249 |
| `libonnxruntime.so.1.31.1` | 1,558,131 (87.18%) | 1,329,826 (87.11%) | 1,688,281 (94.41%) |
| kernel (`no-vmlinux`) | 140,982 (7.89%) | 115,472 (7.56%) | 32,929 (1.84%) |
| `libc.so.6` | 63,150 (3.53%) | 58,710 (3.85%) | 40,799 (2.28%) |
| `cpp-ocr-daemon` | 23,898 (1.34%) | 22,022 (1.44%) | 24,716 (1.38%) |
| `libpng16` | 626 (0.04%) | 302 (0.02%) | 793 (0.04%) |
| other libraries | 499 (0.03%) | 360 (0.02%) | 731 (0.04%) |

The `cpp-ocr-daemon` row is all of our code, including the geometry, the HTTP
frontend and OpenCV's entry points: 1.3–1.4% of the samples.

### A.2 By symbol, own code only

| symbol | 19.png detector | 01.png strip | 07.png recogniser |
| --- | --- | --- | --- |
| `ocr_group_pixels` | 10,764 (0.60%) | 11,491 (0.75%) | 4,740 (0.27%) |
| `Engine::recognize` | 2,295 (0.13%) | 728 (0.05%) | 9,379 (0.52%) |
| `ocr_nchw_normalize` | 4,352 (0.24%) | 2,925 (0.19%) | 2,222 (0.12%) |
| `ocr_resize_bilinear` | 3,971 (0.22%) | 3,211 (0.21%) | 3,802 (0.21%) |
| `ocr_warp_cubic` | 960 (0.05%) | 1,812 (0.12%) | 3,562 (0.20%) |
| `ocr_detection_mask` | 1,225 (0.07%) | 1,212 (0.08%) | 494 (0.03%) |
| `ocr_box_score` | 94 (0.01%) | 393 (0.03%) | 91 (0.01%) |
| `compare_by_x` | 93 (0.01%) | 118 (0.01%) | 155 (0.01%) |
| `ocr_min_area_rect` | 66 (0.00%) | 84 (0.01%) | 125 (0.01%) |
| `ocr_py_round` | 54 (0.00%) | 28 (0.00%) | 111 (0.01%) |

Everything else in the daemon image (`Session::run`, `add_nchw`, `read_request`,
`main`, `ocr_perspective_transform`, `ocr_rect_points`) is below 0.01% in every
shape. The two functions the profile retired as optimisation targets, resize and
normalise, are 0.4–0.5% together on the detector shapes and 0.3% on the
recogniser.

---

## Appendix B — per-image A/B for the memory work

### B.1 Full-corpus paired A/B

5 trials, 20 images, interleaved; `c-base` / `cpp-base` are the binaries built
before the change. Medians are per-image median request times (ms); `Δ` is the
**paired** delta (median of per-request differences over the same image and
round), which is what the harness tests — the difference of two medians is a
different, noisier number and is not used here.

| image | c-base | c | Δ c | cpp-base | cpp | Δ cpp |
| --- | --- | --- | --- | --- | --- | --- |
| `01.png` | 527 | 501 | −21 | 514 | 512 | −2 |
| `02.png` | 283 | 285 | +2 | 282 | 271 | −14 |
| `03.png` | 408 | 390 | −15 | 387 | 394 | +5 |
| `04.png` | 165 | 160 | −5 | 164 | 170 | +1 |
| `05.png` | 345 | 353 | −1 | 356 | 351 | −6 |
| `06.png` | 463 | 446 | −12 | 457 | 462 | +4 |
| `07.png` | 561 | 540 | −25 | 543 | 536 | −7 |
| `08.png` | 243 | 250 | +10 | 244 | 233 | −7 |
| `09.png` | 498 | 490 | −6 | 511 | 519 | +11 |
| `10.png` | 372 | 387 | +14 | 374 | 367 | −7 |
| `11.png` | 170 | 174 | −1 | 170 | 165 | −3 |
| `12.png` | 383 | 383 | +15 | 395 | 388 | −7 |
| `13.png` | 218 | 220 | +5 | 215 | 213 | −1 |
| `14.png` | 375 | 357 | −9 | 363 | 351 | +3 |
| `15.png` | 280 | 284 | 0 | 289 | 288 | +3 |
| `16.png` | 430 | 412 | −25 | 416 | 420 | +5 |
| `17.png` | 164 | 163 | +2 | 166 | 169 | −5 |
| `18.png` | 515 | 522 | +21 | 524 | 505 | −16 |
| `19.png` | 578 | 600 | +5 | 627 | 567 | −44 |
| `20.png` | 208 | 207 | 0 | 212 | 211 | +3 |

Paired over 100 requests: C **−2.0 ms [−5.7, +0.8]** (inside noise), heavy set
**−5.2 ms**; C++ **−2.4 ms [−4.4, −0.3]** (significant), heavy set **−13.7 ms**.
All four candidates returned 30% exact and 1.34% CER — no text moved.

The corpus median understates the change because 15 of the 20 images are light and
have nothing to reclaim; the heavy set and the targeted probe below are the useful
numbers.

### B.2 Targeted five-image probe

The headline in §7 is from a targeted probe on the C++ build, alternating the old
and new binaries over the five heavy images:

| image | paired Δ |
| --- | --- |
| `19.png` | −21 ms |
| `18.png` | −20 ms |
| `09.png` | −18 ms |
| `07.png` | −9 ms |
| `01.png` | −5 ms |
| **total** | **−73 ms** |

The C core received the same code afterwards; its full-corpus paired numbers are
in B.1, and `check_cores.py` confirms both cores return identical text.

### B.3 Fault probe

Counted from `/proc/self/stat` (`minflt`), user space, one request at a time:

| probe | value |
| --- | --- |
| minor faults per `19.png` request, before | 153,636 |
| minor faults per `19.png` request, after | 133,856 |
| of which one detector `Run`, before | ~114,000 |
| 35 MB detector canvas, freshly allocated | 28.4 ms |
| same buffer reused | 7.4 ms |

The canvas figure is the cost of touching the whole buffer once, which is why
retaining it removes most of our own fault cost; the remaining ~134k are the
runtime's per-`Run` activation allocations.

---

## Appendix C — per-image A/B for the ORT CPU arena

Same harness, 100 paired requests per comparison, `OCR_MEM_ARENA=1` against the
shipped default (off); `Δ` is the paired delta, negative meaning the arena is
faster.

| image | c | c + arena | Δ c | cpp | cpp + arena | Δ cpp |
| --- | --- | --- | --- | --- | --- | --- |
| `01.png` | 500 | 483 | −25 | 502 | 478 | −36 |
| `02.png` | 276 | 275 | +4 | 274 | 266 | −6 |
| `03.png` | 391 | 358 | −33 | 391 | 355 | −36 |
| `04.png` | 166 | 170 | +4 | 173 | 168 | −5 |
| `05.png` | 341 | 314 | −24 | 343 | 314 | −27 |
| `06.png` | 446 | 425 | −27 | 454 | 418 | −31 |
| `07.png` | 545 | 534 | −9 | 524 | 533 | 0 |
| `08.png` | 229 | 228 | +2 | 236 | 228 | −8 |
| `09.png` | 495 | 452 | −44 | 503 | 455 | −53 |
| `10.png` | 387 | 359 | −29 | 378 | 352 | −34 |
| `11.png` | 168 | 158 | −10 | 159 | 159 | −1 |
| `12.png` | 372 | 350 | −22 | 390 | 360 | −24 |
| `13.png` | 204 | 209 | +4 | 215 | 211 | −2 |
| `14.png` | 352 | 327 | −29 | 342 | 332 | −15 |
| `15.png` | 275 | 258 | −17 | 283 | 262 | −14 |
| `16.png` | 413 | 385 | −22 | 398 | 386 | −21 |
| `17.png` | 169 | 162 | −1 | 165 | 163 | +2 |
| `18.png` | 501 | 472 | −26 | 514 | 483 | −31 |
| `19.png` | 568 | 526 | −40 | 572 | 536 | −37 |
| `20.png` | 209 | 201 | −8 | 205 | 200 | −10 |

C: **−19.0 ms [−23.2, −12.8]**, heavy set **−27.0 ms**; C++: **−15.3 ms
[−20.8, −9.7]**, heavy set **−28.6 ms**. All significant. Resident memory goes
208–315 MiB → ~1.16 GiB. Almost every image gains, which is why the arena looks
attractive until the RSS number is read.
