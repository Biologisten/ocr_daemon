#!/usr/bin/env python3
"""Reproducible localhost OCR latency benchmark without third-party packages.

Every trial starts all candidates as fresh processes, warms them over the whole
corpus in one interleaved pass, then issues the timed requests interleaved across
candidates with the rotation advancing per image, so machine drift and request
order cannot favour one candidate.

Latency numbers are reported with the spread, not just a central value: each
candidate gets a bootstrap confidence interval for its median, and the other
candidates are compared against the first one **pairwise** on the requests they
both served, which is far less noisy than comparing two independently collected
medians.

The corpus is deliberately uneven (a dense screenshot costs ~1 s, a thin strip
~0.2 s), and a user waits per image, so the tail is reported next to the median:
the slowest image (`worst image`) and the median of the slowest fifth of the
corpus (`heavy`). Comparisons repeat their paired delta over that heavy set, so a
change that only helps the expensive screenshots does not hide inside the
corpus-wide median.
"""

from __future__ import annotations

import argparse
import base64
import http.client
from datetime import datetime, timezone
import json
import os
import platform
import random
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST, PORT = "127.0.0.1", 9722
# A valid 96x32 white PNG; every candidate must answer it with HTTP 200. Its
# round trip is recorded too: it is a request with almost no OCR work in it, so
# it measures the fixed per-request cost of the frontend.
READY_PROBE = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAGAAAAAgCAIAAABiouoDAAAAZklEQVRoBe3BAQEAAAgCIP0/mhb4"
    "ICjytiJvK/K2Im8r8rYibyvytiJvK/K2Im8r8rYibyvytiJvK/K2Im8r8rYibyvytiJvK/K2Im8r"
    "8rYibyvytiJvK/K2Im8r8rYibyvytiJvO1U5X8GMXDILAAAAAElFTkSuQmCC"
)
BOOTSTRAP_ITERATIONS = 4000
BOOTSTRAP_SEED = 20261102


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low, high = int(position), min(int(position) + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def bootstrap_median_ci(values: list[float], iterations: int = BOOTSTRAP_ITERATIONS) -> tuple[float, float]:
    """Percentile bootstrap over the median, resampling the given units."""
    if len(values) < 2:
        return (values[0], values[0]) if values else (0.0, 0.0)
    rng = random.Random(BOOTSTRAP_SEED)
    medians = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(iterations))
    return medians[int(0.025 * iterations)], medians[int(0.975 * iterations)]


def edit_distance(expected: str, actual: str) -> int:
    previous = list(range(len(actual) + 1))
    for row, expected_char in enumerate(expected, 1):
        current = [row]
        for column, actual_char in enumerate(actual, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (expected_char != actual_char),
            ))
        previous = current
    return previous[-1]


def post(port: int, image: bytes) -> tuple[int, bytes, float]:
    started = time.perf_counter_ns()
    connection = http.client.HTTPConnection(HOST, port, timeout=60)
    try:
        connection.request("POST", "/", body=image, headers={"Content-Type": "image/png"})
        response = connection.getresponse()
        body = response.read()
        return response.status, body, (time.perf_counter_ns() - started) / 1_000_000
    finally:
        connection.close()


def wait_until_ready(process: subprocess.Popen[bytes], name: str, port: int) -> tuple[float, float]:
    """Returns (startup_ms, probe_ms); probe_ms is the readiness round trip."""
    started = time.perf_counter()
    while time.perf_counter() - started < 90:
        if process.poll() is not None:
            stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
            detail = f"\n{stderr}" if stderr else ""
            raise RuntimeError(f"{name} exited with {process.returncode}{detail}")
        try:
            status, _, elapsed = post(port, READY_PROBE)
            if status == 200:
                return (time.perf_counter() - started) * 1_000, elapsed
        except (ConnectionRefusedError, TimeoutError, OSError, http.client.HTTPException):
            pass
        time.sleep(0.02)
    raise TimeoutError(f"{name} did not answer the readiness probe on port {port} within 90 seconds")


def rss_kib(pid: int) -> int | None:
    status = Path(f"/proc/{pid}/status")
    if not status.exists():
        return None
    for line in status.read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return None


def cpu_ms(pid: int) -> float | None:
    """utime + stime in milliseconds, for the daemon itself and its threads."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
    except OSError:
        return None
    ticks = os.sysconf("SC_CLK_TCK")
    return (int(fields[13]) + int(fields[14])) * 1000.0 / ticks


def stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def load_corpus(path: Path) -> list[dict[str, str]]:
    corpus = json.loads(path.read_text())["images"]
    if len(corpus) != 20:
        raise SystemExit(f"{path} must contain exactly 20 images (found {len(corpus)})")
    for sample in corpus:
        image = path.parent / sample["file"]
        if not image.is_file() or "text" not in sample:
            raise SystemExit(f"invalid corpus entry: {sample}")
        sample["path"] = str(image)
    return corpus


def launch(candidate: dict, port: int) -> subprocess.Popen[bytes]:
    environment = os.environ | {
        key: str(ROOT / value) if key == "OCR_MODEL_DIR" else value
        for key, value in candidate.get("env", {}).items()
    }
    environment["OCR_PORT"] = str(port)
    return subprocess.Popen(
        candidate["command"], cwd=ROOT, env=environment,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )


def state() -> dict:
    return {"startup_ms": [], "probe_ms": [], "reject_ms": [], "rss_kib": [], "cpu_ms": [],
            "requests_per_trial": 0, "samples": [], "matches": [], "character_errors": 0,
            "character_total": 0, "mismatches": {}}


def summarise(candidate: dict, data: dict, config: dict) -> dict:
    latencies = [item["latency_ms"] for item in data["samples"]]
    per_image: dict[str, list[float]] = {}
    per_trial: dict[int, list[float]] = {}
    for item in data["samples"]:
        per_image.setdefault(item["image"], []).append(item["latency_ms"])
        per_trial.setdefault(item["trial"], []).append(item["latency_ms"])
    image_medians = {name: statistics.median(values) for name, values in per_image.items()}
    median_lo, median_hi = bootstrap_median_ci(list(image_medians.values()))
    # The corpus is uneven and a user waits per image, so the tail is reported
    # next to the median: the slowest image by median, and the slowest fifth of
    # the corpus (four images) as a single "heavy" number that does not hinge on
    # one screenshot.
    slowest = sorted(image_medians.items(), key=lambda item: (-item[1], item[0]))
    heavy = slowest[:max(1, len(slowest) // 5)]
    budget = float(config.get("budget_ms", 1000))
    return {
        "name": candidate["name"],
        "startup_ms": {"median": statistics.median(data["startup_ms"]), "samples": data["startup_ms"]},
        "probe_ms": {"median": statistics.median(data["probe_ms"]), "samples": data["probe_ms"]},
        "reject_ms": {"median": statistics.median(data["reject_ms"]), "samples": data["reject_ms"]},
        "rss_kib": data["rss_kib"],
        # CPU seconds the daemon burned per request, summed over its worker
        # threads: higher than the wall-clock median whenever inference is
        # parallel, which it is.
        "cpu_ms_per_request": (
            statistics.median(data["cpu_ms"]) / data["requests_per_trial"]
            if data["cpu_ms"] and data["requests_per_trial"] else None
        ),
        "requests": len(latencies),
        "latency_ms": {
            "median": statistics.median(latencies),
            "median_ci": [median_lo, median_hi],
            "mean": statistics.fmean(latencies),
            "p90": percentile(latencies, 0.90),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "min": min(latencies),
            "max": max(latencies),
        },
        "median_by_trial": {str(trial): statistics.median(values) for trial, values in sorted(per_trial.items())},
        "median_by_image": image_medians,
        "worst_image": {"name": slowest[0][0], "median_ms": slowest[0][1]},
        "heavy_images": [name for name, _ in heavy],
        "heavy_median_ms": statistics.median([value for _, value in heavy]),
        # The budget is the arrival interval (one screenshot per second), so this
        # counts the images whose typical latency would already exceed it.
        "budget_ms": budget,
        "images_over_budget": sorted(name for name, value in image_medians.items() if value > budget),
        "exact_match_rate": sum(data["matches"]) / len(data["matches"]),
        "character_error_rate": data["character_errors"] / max(data["character_total"], 1),
        "mismatches": data["mismatches"],
        "samples": data["samples"],
    }


def compare(base: dict, other: dict) -> dict:
    """Pairwise comparison over the requests both candidates served.

    Every (trial, repetition, image) key is served by every candidate inside the
    same round, so the differences are paired and machine drift cancels out
    instead of showing up as a difference between two independent medians.

    The delta is `other - base`, so a negative median means `other` is faster.
    """
    base_by_key = {(s["trial"], s["repetition"], s["image"]): s["latency_ms"] for s in base["samples"]}
    paired = [(s["image"], s["latency_ms"] - base_by_key[(s["trial"], s["repetition"], s["image"])])
              for s in other["samples"] if (s["trial"], s["repetition"], s["image"]) in base_by_key]
    if not paired:
        return {}
    deltas = [delta for _, delta in paired]
    # Per-image paired delta: the corpus is uneven (a screenshot is ~2.5 s of
    # detector work, a strip is ~450 ms), so a core change that helps one shape
    # and hurts another can hide inside the corpus-wide median. Breaking the
    # delta down by image shows which shapes moved.
    per_image: dict[str, list[float]] = {}
    for image, delta in paired:
        per_image.setdefault(image, []).append(delta)
    # The same comparison restricted to the base candidate's slowest fifth, which
    # is the set an interactive user actually notices.
    heavy = {name for name, _ in sorted(base["median_by_image"].items(),
                                       key=lambda item: (-item[1], item[0]))
             [:max(1, len(base["median_by_image"]) // 5)]}
    heavy_deltas = [delta for image, delta in paired if image in heavy]
    lo, hi = bootstrap_median_ci(deltas)
    return {
        "requests": len(deltas),
        "median_delta_ms": statistics.median(deltas),
        "median_delta_ci": [lo, hi],
        "other_faster": sum(1 for delta in deltas if delta < 0),
        "base_faster": sum(1 for delta in deltas if delta > 0),
        "ties": sum(1 for delta in deltas if delta == 0),
        "significant": lo > 0 or hi < 0,
        "median_delta_by_image": {name: statistics.median(values) for name, values in per_image.items()},
        "heavy_images": sorted(heavy),
        "heavy_median_delta_ms": statistics.median(heavy_deltas) if heavy_deltas else None,
    }


def run_candidates(candidates: list[dict], corpus: list[dict[str, str]], config: dict) -> list[dict]:
    for candidate in candidates:
        command = candidate["command"]
        if not shutil.which(command[0]) and not (ROOT / command[0]).exists():
            raise SystemExit(f"candidate executable is unavailable: {command[0]}")
    states = {candidate["name"]: state() for candidate in candidates}
    # Every candidate stays resident during a trial, so each needs its own port.
    ports = {candidate["name"]: PORT + index for index, candidate in enumerate(candidates)}
    payloads = {sample["file"]: Path(sample["path"]).read_bytes() for sample in corpus}
    for trial in range(config["trials"]):
        running = {}
        try:
            # Sequential launches keep startup comparable. Trial-to-trial drift
            # (thermal, allocator, ordering) moves a candidate's median by up to
            # ~30 ms here, which is more than the gap between these candidates,
            # so trials alternate the launch order to keep any one of them from
            # always sitting in the same slot.
            shift = trial % len(candidates)
            launch_order = candidates[shift:] + candidates[:shift]
            for candidate in launch_order:
                name = candidate["name"]
                process = launch(candidate, ports[name])
                running[name] = process
                startup, probe = wait_until_ready(process, name, ports[name])
                states[name]["startup_ms"].append(startup)
                states[name]["probe_ms"].append(probe)
            # A rejected body exercises the frontend without any OCR work, so it
            # measures the fixed per-request cost: accept, parse, answer.
            for candidate in candidates:
                name = candidate["name"]
                rejected = []
                for _ in range(3):
                    status, _, elapsed = post(ports[name], b"")
                    if status != 400:
                        raise RuntimeError(f"{name} answered HTTP {status} for an empty body, expected 400")
                    rejected.append(elapsed)
                states[name]["reject_ms"].append(statistics.median(rejected))
            # Warm every candidate over the whole corpus first, interleaved like
            # the measurements, so the first timed pass is not a cold run and
            # warmup work is not repeated before every single request.
            for _ in range(config["warmups"]):
                for sample in corpus:
                    data = payloads[sample["file"]]
                    for candidate in candidates:
                        post(ports[candidate["name"]], data)
            # CPU is sampled around the measured work only: startup, the probe
            # and warmup are excluded on both sides.
            cpu_started = {candidate["name"]: cpu_ms(running[candidate["name"]].pid)
                           for candidate in candidates}
            for candidate in candidates:
                states[candidate["name"]]["requests_per_trial"] = config["repetitions"] * len(corpus)
            for repetition in range(config["repetitions"]):
                for index, sample in enumerate(corpus):
                    data = payloads[sample["file"]]
                    # Each image starts with a different candidate, and the
                    # sequence advances every trial, so no candidate always runs
                    # first (or last) on the same work. The trial term must not
                    # be a multiple of the candidate count: `trial * 20 % 2` is
                    # always 0, which pinned every image to the same first
                    # candidate for the whole run.
                    offset = (trial + index + repetition) % len(candidates)
                    order = candidates[offset:] + candidates[:offset]
                    for position, candidate in enumerate(order):
                        name = candidate["name"]
                        status, body, elapsed = post(ports[name], data)
                        if status != 200:
                            raise RuntimeError(f"{name} returned HTTP {status} for {sample['file']}")
                        actual = body.decode("utf-8")
                        expected = sample["text"]
                        entry = states[name]
                        entry["samples"].append({
                            "image": sample["file"], "trial": trial, "repetition": repetition,
                            "position": position, "latency_ms": elapsed,
                        })
                        entry["matches"].append(actual == expected)
                        entry["character_errors"] += edit_distance(expected, actual)
                        entry["character_total"] += len(expected)
                        if actual != expected:
                            entry["mismatches"][sample["file"]] = {"expected": expected, "actual": actual}
            for candidate in candidates:
                name = candidate["name"]
                rss = rss_kib(running[name].pid)
                if rss is not None:
                    states[name]["rss_kib"].append(rss)
                cpu = cpu_ms(running[name].pid)
                if cpu is not None and cpu_started[name] is not None:
                    states[name]["cpu_ms"].append(cpu - cpu_started[name])
        finally:
            for process in running.values():
                stop(process)
    return [summarise(candidate, states[candidate["name"]], config) for candidate in candidates]


def all_comparisons(results: list[dict]) -> list[dict]:
    comparisons = []
    for index, base in enumerate(results):
        for other in results[index + 1:]:
            comparison = {"base": base["name"], "other": other["name"]}
            comparison |= compare(base, other)
            comparisons.append(comparison)
    return comparisons


def write_results(output: Path, config: dict, results: list[dict], comparisons: list[dict]) -> None:
    output.write_text(json.dumps({
        "machine": {
            "platform": platform.platform(), "python": sys.version,
            "cpu_count": os.cpu_count(),
        },
        "config": config,
        "results": results,
        "comparisons": comparisons,
    }, indent=2) + "\n")


def read_history(path: Path) -> list[dict]:
    """Append-only records of past runs; a truncated line is skipped, not fatal."""
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def record(entries: list[dict], path: Path, config: dict, results: list[dict]) -> dict:
    """Append one compact record, enough to follow the tail across optimisations."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": {"platform": platform.platform(), "cpu_count": os.cpu_count()},
        "config": {key: config.get(key) for key in ("trials", "repetitions", "warmups", "budget_ms")},
        "candidates": [
            {
                "name": result["name"],
                "median_ms": result["latency_ms"]["median"],
                "p99_ms": result["latency_ms"]["p99"],
                "worst_image": result["worst_image"]["name"],
                "worst_ms": result["worst_image"]["median_ms"],
                "heavy_ms": result["heavy_median_ms"],
                "images_over_budget": len(result["images_over_budget"]),
                "exact_match_rate": result["exact_match_rate"],
                "character_error_rate": result["character_error_rate"],
            }
            for result in results
        ],
    }
    with path.open("a") as handle:
        handle.write(json.dumps(entry) + "\n")
    return entry


def previous_for(entries: list[dict], name: str) -> dict | None:
    """Most recent recorded numbers for one candidate, or None."""
    for entry in reversed(entries):
        for candidate in entry.get("candidates", []):
            if candidate.get("name") == name:
                return candidate | {"timestamp": entry.get("timestamp")}
    return None


def report_budget(results: list[dict], entries: list[dict]) -> None:
    """The tail against the arrival interval, plus the change since last run.

    A user sends at most about one screenshot per second, so an image whose
    typical latency is at or above the budget is the thing optimisation is for.
    """
    if not results:
        return
    budget = results[0]["budget_ms"]
    total = len(results[0]["median_by_image"])
    print(f"\ntail budget: {budget:.0f} ms (one screenshot per second)")
    for result in results:
        worst = result["worst_image"]
        over = result["images_over_budget"]
        line = (f"  {result['name']:<4} worst {worst['name']} {worst['median_ms']:.0f} ms "
                f"({worst['median_ms'] / budget:.0%} of budget), {len(over)}/{total} images over")
        before = previous_for(entries, result["name"])
        if before is None:
            line += " | no previous run recorded"
        else:
            line += (f" | previous {before['worst_ms']:.0f} ms ({worst['median_ms'] - before['worst_ms']:+.0f}), "
                     f"heavy {before['heavy_ms']:.0f} ms ({result['heavy_median_ms'] - before['heavy_ms']:+.0f}), "
                     f"median {before['median_ms']:.0f} ms ({result['latency_ms']['median'] - before['median_ms']:+.0f})")
        print(line)


def print_history(entries: list[dict], limit: int = 20) -> None:
    if not entries:
        print("no runs recorded yet")
        return
    print(f"{'timestamp':<21} {'trials':>6} {'candidate':<10} {'median':>8} {'worst':>9} {'ms':>5} "
          f"{'budget':>7} {'heavy':>8} {'over':>5} {'CER':>7}")
    for entry in entries[-limit:]:
        config = entry.get("config", {})
        budget = float(config.get("budget_ms") or 1000)
        trials = f"{config.get('trials')}x{config.get('repetitions')}"
        for candidate in entry.get("candidates", []):
            print(f"{entry.get('timestamp', '?'):<21} {trials:>6} {candidate['name']:<10} "
                  f"{candidate['median_ms']:8.1f} {candidate['worst_image']:>9} {candidate['worst_ms']:5.0f} "
                  f"{candidate['worst_ms'] / budget:6.0%} {candidate['heavy_ms']:8.1f} "
                  f"{candidate['images_over_budget']:5} {candidate['character_error_rate']:6.2%}")


def report(results: list[dict], comparisons: list[dict]) -> None:
    print(f"{'candidate':<12} {'median':>8} {'95% CI (median)':>19} {'p90':>7} {'p95':>7} {'p99':>7} "
          f"{'worst image':>15} {'heavy':>7} {'empty':>6} {'cpu/req':>7} {'startup':>8} "
          f"{'exact':>6} {'CER':>6}")
    for result in results:
        latency = result["latency_ms"]
        lo, hi = latency["median_ci"]
        cpu = result["cpu_ms_per_request"]
        worst = result["worst_image"]
        worst_text = f"{worst['name']} {worst['median_ms']:.0f}"
        print(f"{result['name']:<12} {latency['median']:8.1f} {f'[{lo:.1f}, {hi:.1f}]':>19} "
              f"{latency['p90']:7.1f} {latency['p95']:7.1f} {latency['p99']:7.1f} "
              f"{worst_text:>15} {result['heavy_median_ms']:7.1f} "
              f"{result['reject_ms']['median']:6.1f} {cpu:7.0f} {result['startup_ms']['median']:8.1f} "
              f"{result['exact_match_rate']:6.1%} {result['character_error_rate']:6.2%}")
    print()
    for comparison in comparisons:
        lo, hi = comparison["median_delta_ci"]
        verdict = "SIGNIFICANT" if comparison["significant"] else "inside noise"
        print(f"{comparison['other']} - {comparison['base']}: {comparison['median_delta_ms']:+.1f} ms "
              f"[{lo:+.1f}, {hi:+.1f}] over {comparison['requests']} paired requests "
              f"({comparison['other']} faster in {comparison['other_faster']}, "
              f"{comparison['base']} faster in {comparison['base_faster']}) -> {verdict}")
        heavy_delta = comparison.get("heavy_median_delta_ms")
        if heavy_delta is not None:
            print(f"    heavy images ({', '.join(name for name in comparison['heavy_images'])}): "
                  f"{heavy_delta:+.1f} ms paired median")
        movers = sorted(comparison["median_delta_by_image"].items(), key=lambda item: abs(item[1]), reverse=True)
        print("    largest per-image deltas: " + ", ".join(
            f"{name} {delta:+.0f} ms" for name, delta in movers[:6]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "benchmarks/candidates.json")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results.json")
    parser.add_argument("--history", type=Path, default=ROOT / "benchmarks/history.jsonl")
    parser.add_argument("--no-history", action="store_true", help="do not append a run record")
    parser.add_argument("--budget-ms", type=float, default=None,
                        help="override the per-image latency budget (default 1000, the arrival interval)")
    parser.add_argument("--trend", action="store_true", help="print recorded runs and exit")
    args = parser.parse_args()
    entries = read_history(args.history)
    if args.trend:
        print_history(entries)
        return
    config = json.loads(args.config.read_text())
    if args.budget_ms is not None:
        config["budget_ms"] = args.budget_ms
    global PORT
    PORT = int(config.get("port", PORT))
    corpus = load_corpus(ROOT / config["corpus"])
    candidates = [c for c in config["candidates"] if not args.only or c["name"] in args.only]
    if not candidates:
        raise SystemExit("no matching candidates")
    results = run_candidates(candidates, corpus, config)
    comparisons = all_comparisons(results)
    write_results(args.output, config, results, comparisons)
    report(results, comparisons)
    report_budget(results, entries)
    if not args.no_history:
        record(entries, args.history, config, results)
        print(f"\nrecorded in {args.history} ({len(entries) + 1} run(s))")
        print("see the trend with: python3 benchmarks/run.py --trend")


if __name__ == "__main__":
    main()
