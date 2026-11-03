#!/usr/bin/env python3
"""Byte-compare the candidates on every corpus image, and re-check determinism.

This is the fast gate for core work: both cores share the crop geometry, so
identical text means the change did not alter what the models were asked to read.
Repeating every request inside the same process checks that a process is
deterministic, which is what makes the comparison meaningful.

Exits non-zero on any difference, so it can gate a build:

    python3 benchmarks/check_cores.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run import ROOT, launch, load_corpus, post, stop, wait_until_ready  # noqa: E402


def main() -> None:
    config = json.loads((ROOT / "benchmarks/candidates.json").read_text())
    candidates = config["candidates"]
    if len(candidates) < 2:
        raise SystemExit("need at least two candidates to compare")
    corpus = load_corpus(ROOT / config["corpus"])
    base_port = int(config.get("port", 9723))
    running: dict[str, object] = {}
    texts: dict[str, dict[str, str]] = {}
    failures: list[str] = []
    try:
        for index, candidate in enumerate(candidates):
            name = candidate["name"]
            process = launch(candidate, base_port + index)
            running[name] = process
            wait_until_ready(process, name, base_port + index)
            texts[name] = {}
        for sample in corpus:
            payload = Path(sample["path"]).read_bytes()
            for index, candidate in enumerate(candidates):
                name = candidate["name"]
                status, body, _ = post(base_port + index, payload)
                if status != 200:
                    failures.append(f"{name} answered HTTP {status} for {sample['file']}")
                    continue
                texts[name][sample["file"]] = body.decode("utf-8")
        reference = candidates[0]["name"]
        for candidate in candidates[1:]:
            name = candidate["name"]
            for sample in corpus:
                first, second = texts[reference].get(sample["file"]), texts[name].get(sample["file"])
                if first is not None and second is not None and first != second:
                    failures.append(f"{sample['file']}: {reference} != {name}\n"
                                    f"  {reference}: {first!r}\n  {name}: {second!r}")
        # Same process, same request: must reproduce the same bytes.
        for index, candidate in enumerate(candidates):
            name = candidate["name"]
            for sample in corpus:
                status, body, _ = post(base_port + index, Path(sample["path"]).read_bytes())
                if status == 200 and body.decode("utf-8") != texts[name].get(sample["file"]):
                    failures.append(f"{sample['file']}: {name} is not deterministic across repeats")
    finally:
        for process in running.values():
            stop(process)
    for message in failures:
        print(message)
    print(f"{len(corpus)} images, {len(candidates)} candidates, "
          f"{len(failures)} failure(s) -> {'FAIL' if failures else 'identical'}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
