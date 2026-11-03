#!/usr/bin/env python3
"""Record hashes for the RapidOCR models used by every benchmark candidate."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
NAMES = (
    "PP-OCRv6_det_small.onnx",
    "PP-OCRv6_rec_small.onnx",
    "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
)


def main() -> None:
    assets = []
    for name in NAMES:
        path = MODEL_DIR / name
        if not path.is_file():
            raise SystemExit(f"missing model: {path}")
        with path.open("rb") as model:
            digest = hashlib.file_digest(model, "sha256").hexdigest()
        assets.append({"file": name, "bytes": path.stat().st_size, "sha256": digest})
    (ROOT / "models/manifest.json").write_text(
        json.dumps({"rapidocr": "3.10.0", "assets": assets}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
