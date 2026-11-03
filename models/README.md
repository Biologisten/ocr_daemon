# Pinned OCR assets

The benchmark uses the three RapidOCR 3.10.0 assets checked in here:

- `PP-OCRv6_det_small.onnx` — text detection
- `PP-OCRv6_rec_small.onnx` — text recognition
- `ch_ppocr_mobile_v2.0_cls_mobile.onnx` — 180-degree classifier

`manifest.json` records their size and SHA-256. Re-record it with
`python3 tools/model_manifest.py` after replacing an asset, and pass this
directory to the daemons as `OCR_MODEL_DIR` (the benchmark does that for you).
