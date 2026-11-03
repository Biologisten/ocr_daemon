# Vendored ONNX Runtime

`libonnxruntime.so.1.31.1` is the CPU build shipped in the PyPI wheel
`onnxruntime==1.31.1` (`onnxruntime/capi/`), which is the runtime RapidOCR
itself uses.

- Upstream: https://github.com/microsoft/onnxruntime (tag `v1.31.1`)
- License: MIT
- It is vendored because the distribution packages only 1.29, which measures
  ~20% slower on this corpus (see “Which ONNX Runtime” in the top-level README).
- The `libonnxruntime.so` and `libonnxruntime.so.1` symlinks exist only so the
  linker and the dynamic loader can resolve the library; the real file is
  `libonnxruntime.so.1.31.1`. The Makefile links it by exact name with
  `-Wl,-rpath,'$ORIGIN/lib'`, so both daemons resolve this copy at run time.

The ONNX Runtime headers in `/usr/include/onnxruntime` (1.29) are used at compile
time; the C API is backwards compatible, and both daemons produce byte-identical
OCR text against every runtime version that has been tried here.

To compare another runtime, put it in a subdirectory (builds share the SONAME
`libonnxruntime.so.1`, so two cannot sit in this directory) and link a variant
binary with `-l:libonnxruntime.so.<version> -Wl,-rpath,'$ORIGIN/<subdirectory>'`.
