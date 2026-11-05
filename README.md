# OCR daemon benchmark

Two OCR daemons, C11 and C++20, that answer `POST` with an encoded 
screenshot and reply with the recognised text, plus benchmark for
20-image screenshots corpus.

Both link the same ONNX Runtime (the system one when it is 1.30 or newer,
otherwise the vendored 1.31.1) and the same PP-OCRv6-small model.

## Contents

| path | what it is |
| --- | --- |
| `native/ocr_core_c.c` | C11 OCR core: ONNX Runtime **C** API, libpng/libjpeg, no OpenCV |
| `native/ocr_core.cpp` | C++20 OCR core: ONNX Runtime C++ API, OpenCV 5 |
| `native/ocr_geometry.c` | crop geometry, detector-canvas rule and thread count shared by both cores (see below) |
| `native/c_daemon.c`, `native/cpp_daemon.cpp` | POSIX-socket HTTP frontends |
| `native/ASSUMPTIONS.md` | the pipeline invariants and cost model that core optimisation rests on |
| `WRITEUP.md` | the optimisation record: what changed, what it bought, and what was rejected |
| `native/smoke.cpp` | asserts a white PNG comes back as empty text |
| `native/lib/` | vendored ONNX Runtime 1.31.1 (see `native/lib/NOTICE.md`, MIT) |
| `models/` | PP-OCRv6 det/rec models, with `manifest.json` hashes |
| `corpus/` | 20 screenshots and their ground truth (`manifest.json`) |
| `benchmarks/` | benchmark and results history |

## Requirements

`opencv5`, `libpng`, `libjpeg`, a C11 compiler, `g++`, and `python3`.
Needs ONNX Runtime *headers* from the system package, uses vendored runtime if 
sys is < v1.30.

## Build

```sh
make -C native                 # c-ocr-daemon, cpp-ocr-daemon
make -C native compile_commands.json   # what clangd reads
OCR_MODEL_DIR=models ./native/smoke && echo ok
OCR_DEBUG=1 OCR_MODEL_DIR=models ./native/cpp-ocr-daemon   # per-request stage timings on stderr
```

`OCR_DEBUG` prints the ms spent in decode, detector, crops and recognition per request
and `OCR_THREADS` overrides the intra-op pool size that is otherwise derived from the
physical core count.

The Makefile links the system library with `-lonnxruntime` when it is 1.30 or newer.
Otherwise it links `native/lib/libonnxruntime.so.1.31.1` with `-Wl,-rpath,'$ORIGIN/lib'`,
falling back to the system library if the vendored file is absent too, so a checkout
without `native/lib/` still builds. `ldd native/cpp-ocr-daemon | grep onnx`
shows which one a binary resolved.

## Install as a daemon

`make -C native install` copies a self-contained tree to
`$(PREFIX)` (default `~/.local/lib/ocr-daemon`):

```
ocr-daemon                     the binary
models/*.onnx                  det and rec models
lib/libonnxruntime.so.1.31.1   vendored runtime (only when fallback)
lib/libonnxruntime.so.1        symlink the binary resolves (only when fallback)
```

A user unit is then:

```ini
[Service]
ExecStart=%h/.local/lib/ocr-daemon/ocr-daemon
Environment=OCR_MODEL_DIR=%h/.local/lib/ocr-daemon/models
Environment=OCR_PORT=9722
Restart=on-failure
```

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
one.

It reports, per candidate: median with a bootstrap 95% confidence interval,
p90/p95/p99, startup, RSS, CPU time per request, the cost of a request with no
OCR work in it (`empty`), exact-match rate and CER. `benchmarks/results.json`
additionally holds every raw sample, per-trial and per-image medians, and each
mismatched transcription. `worst image` is the slowest single image by median,
and `heavy` is the median of that candidate's slowest fifth of the corpus.

Each run ends with the tail against the **latency budget**, `budget_ms` in 
`candidates.json`, default is 1 000.

```
tail budget: 1000 ms (one screenshot per second)
  c    worst 19.png 577 ms (58% of budget), 0/20 images over | previous 587 ms (-10), heavy 544 ms (-15), median 372 ms (-11)
  cpp  worst 19.png 579 ms (58% of budget), 0/20 images over | previous 606 ms (-26), heavy 539 ms (-14), median 381 ms (-20)
```

Each run shows the previous run and the delta. (`previous 1005 ms (+93), heavy …, median …`).
Every run is stored in `benchmarks/history.jsonl`, printable with 
`python3 benchmarks/run.py --trend` Use `--budget-ms` to override the budget.

The harness binds `127.0.0.1:9723`, and ports above per candidate.

## Current results

Two candidates, vendored ONNX Runtime 1.31.1, 5 trials / 100 timed requests each,
exit 0, ~230 s (run of record: `benchmarks/history.jsonl`):

| candidate | median | 95% CI (median) | p90 | p95 | p99 | worst image | heavy | empty | cpu/req | startup | RSS | exact | CER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c | 341.7 ms | [237.2, 398.3] | 494.7 ms | 545.2 ms | 567.8 ms | 19.png 551 ms | 492.8 ms | 0.1 ms | 2070 ms | 394 ms | 237–332 MiB | 30% | 1.34% |
| cpp | 340.6 ms | [232.8, 400.8] | 494.9 ms | 559.7 ms | 605.1 ms | 19.png 561 ms | 493.5 ms | 0.2 ms | 2046 ms | 407 ms | 211 MiB | 30% | 1.34% |

Worst is the number that matters for snappy UX. All images ≤ 1s.
Before the work described below the worst was `07.png` 949/979 ms (95%/98%) with
two images over budget. Now `19.png` 551/561 ms, `18.png` 494/491 ms. Corpus median
went 372/382 → **361/362 → 342/341 ms**, the heavy-set median 544/539 → **529/525 →
493/494 ms** and the worst image 587/606 → **577/579 → 551/561 ms**. p90 is 495/495 ms
against 553/573 before.

These numbers were collected on a Ryzen 5600X 6-core.

Both candidates return byte-identical text on all 20 screenshots.
Per-image deltas are spread on both sides in the latest run
(`02.png` −17 ms for C, but `01.png` +18 ms and `06.png` +12 ms).

## Where the time goes

A request is the OCR core, essentially entirely:

| part | cost | share |
| --- | --- | --- |
| HTTP frontend (accept, read, answer, reject) | 0.1–0.2 ms | ~0.05% |
| OCR core (decode, detect, crop, recognise) | ~342 ms | ~99.9% |

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
