#!/usr/bin/env python3
"""Benchmark screen capture latency and process CPU usage.

Usage:
    PYTHONPATH=src ./.venv/bin/python scripts/benchmark_capture.py --duration 8
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass

from chesslab.capture.screen_capture import (
    CaptureConfig,
    MonitorSelectionError,
    ScreenCapture,
    ScreenCaptureError,
)


@dataclass(frozen=True)
class CaptureMetrics:
    frames: int
    mean_capture_ms: float
    p95_capture_ms: float
    process_cpu_pct: float
    missed_intervals: int


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = min(len(sorted_values) - 1, int(len(sorted_values) * percentile))
    return sorted_values[idx]


def benchmark(
    *,
    duration_s: float,
    interval_ms: int,
    monitor_index: int | None,
) -> CaptureMetrics:
    capture = ScreenCapture(
        CaptureConfig(interval_ms=interval_ms, monitor_index=monitor_index)
    )
    if capture.config.monitor_index is None:
        monitors = capture.list_monitors()
        if not monitors:
            raise MonitorSelectionError("No physical monitor detected.")
        if len(monitors) == 1:
            capture.select_monitor(monitors[0].index)
        else:
            indices = [m.index for m in monitors]
            raise MonitorSelectionError(
                "Multiple monitors detected. Provide --monitor. "
                f"Available monitor indexes: {indices}"
            )

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    deadline = wall_start + duration_s
    interval_s = interval_ms / 1000.0
    durations_ms: list[float] = []
    missed_intervals = 0

    while time.perf_counter() < deadline:
        started = time.perf_counter()
        cap_start = time.perf_counter()
        capture.capture_once()
        cap_elapsed_ms = (time.perf_counter() - cap_start) * 1000.0
        durations_ms.append(cap_elapsed_ms)

        loop_elapsed = time.perf_counter() - started
        if loop_elapsed > interval_s:
            missed_intervals += 1
            continue
        time.sleep(interval_s - loop_elapsed)

    cpu_end = time.process_time()
    wall_end = time.perf_counter()

    wall_elapsed = max(1e-6, wall_end - wall_start)
    cpu_pct = ((cpu_end - cpu_start) / wall_elapsed) * 100.0
    frames = len(durations_ms)
    mean_ms = statistics.fmean(durations_ms) if durations_ms else 0.0
    p95_ms = _percentile(durations_ms, 0.95)

    return CaptureMetrics(
        frames=frames,
        mean_capture_ms=mean_ms,
        p95_capture_ms=p95_ms,
        process_cpu_pct=cpu_pct,
        missed_intervals=missed_intervals,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10.0, help="Duration in seconds.")
    parser.add_argument("--interval-ms", type=int, default=500, help="Capture interval in ms.")
    parser.add_argument(
        "--monitor",
        type=int,
        default=None,
        help="1-based monitor index. If omitted, auto-select only when one monitor exists.",
    )
    args = parser.parse_args()

    try:
        metrics = benchmark(
            duration_s=args.duration,
            interval_ms=args.interval_ms,
            monitor_index=args.monitor,
        )
    except (MonitorSelectionError, ScreenCaptureError) as exc:
        print(f"benchmark_error={exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"frames={metrics.frames}")
    print(f"mean_capture_ms={metrics.mean_capture_ms:.3f}")
    print(f"p95_capture_ms={metrics.p95_capture_ms:.3f}")
    print(f"process_cpu_pct={metrics.process_cpu_pct:.2f}")
    print(f"missed_intervals={metrics.missed_intervals}")


if __name__ == "__main__":
    main()
