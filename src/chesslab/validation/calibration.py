"""Pre-session calibration checks on real screenshots."""

from __future__ import annotations

import math
from glob import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
from numpy.typing import NDArray

from chesslab.detection.board_detector import BoardDetector
from chesslab.live import LiveFrameStatus, LivePositionProcessor, LiveProcessorConfig
from chesslab.recognition.theme_detector import ThemeDetector
from chesslab.splitting.board_splitter import BoardSplitter
from chesslab.validation.catalog import SetupSupportProfile, SupportLevel


@dataclass(frozen=True)
class CalibrationMetrics:
    """Aggregated metrics collected during preflight."""

    frame_count: int
    detection_rate: float
    theme_match_rate: float
    ready_rate: float
    uncertain_rate: float
    invalid_rate: float
    p95_total_latency_ms: float
    p95_detection_latency_ms: float
    p95_split_latency_ms: float
    p95_recognition_latency_ms: float
    status_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_count": self.frame_count,
            "detection_rate": round(self.detection_rate, 5),
            "theme_match_rate": round(self.theme_match_rate, 5),
            "ready_rate": round(self.ready_rate, 5),
            "uncertain_rate": round(self.uncertain_rate, 5),
            "invalid_rate": round(self.invalid_rate, 5),
            "p95_total_latency_ms": round(self.p95_total_latency_ms, 3),
            "p95_detection_latency_ms": round(self.p95_detection_latency_ms, 3),
            "p95_split_latency_ms": round(self.p95_split_latency_ms, 3),
            "p95_recognition_latency_ms": round(self.p95_recognition_latency_ms, 3),
            "status_counts": dict(self.status_counts),
        }


@dataclass(frozen=True)
class CalibrationReport:
    """Final compatibility report for one setup."""

    setup_id: str
    support_level: str
    template_theme: str
    expected_theme: str
    passed: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    metrics: CalibrationMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "setup_id": self.setup_id,
            "support_level": self.support_level,
            "template_theme": self.template_theme,
            "expected_theme": self.expected_theme,
            "passed": self.passed,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "metrics": self.metrics.to_dict(),
        }


def run_preflight_calibration(
    image_paths: Sequence[Path],
    profile: SetupSupportProfile,
    *,
    confidence_threshold: float = 0.60,
) -> CalibrationReport:
    """Run compatibility checks for one setup on captured screenshots."""
    if not image_paths:
        raise ValueError("Calibration requires at least one image.")

    detector = BoardDetector()
    splitter = BoardSplitter()
    theme_detector = ThemeDetector()
    processor = LivePositionProcessor(
        LiveProcessorConfig(
            theme=profile.template_theme,
            white_perspective=profile.perspective == "white",
            confidence_threshold=confidence_threshold,
        )
    )

    expected_theme = profile.expected_theme_id

    detections = 0
    theme_matches = 0
    ready = 0
    uncertain = 0
    invalid = 0

    status_counts: dict[str, int] = {}
    total_latencies: list[float] = []
    detection_latencies: list[float] = []
    split_latencies: list[float] = []
    recognition_latencies: list[float] = []

    for image_path in image_paths:
        frame = _read_image(image_path)

        live_result = processor.process_frame(frame)
        status = live_result.status.value
        status_counts[status] = status_counts.get(status, 0) + 1

        total_latencies.append(live_result.total_elapsed_ms)
        detection_latencies.append(live_result.detection_elapsed_ms)
        if live_result.split_elapsed_ms is not None:
            split_latencies.append(live_result.split_elapsed_ms)
        if live_result.recognition_elapsed_ms is not None:
            recognition_latencies.append(live_result.recognition_elapsed_ms)

        if live_result.status is LiveFrameStatus.READY:
            ready += 1
        elif live_result.status is LiveFrameStatus.POSITION_UNCERTAIN:
            uncertain += 1
        elif live_result.status is LiveFrameStatus.INVALID_POSITION:
            invalid += 1

        detection = detector.detect(frame)
        if not detection.detected or detection.board is None:
            continue

        detections += 1
        split = splitter.split(
            frame,
            detection.board,
            white_perspective=profile.perspective == "white",
        )
        detected_theme = theme_detector.detect(split.squares)
        if (
            detected_theme.matched
            and detected_theme.theme is not None
            and detected_theme.theme.qualified_name == expected_theme
        ):
            theme_matches += 1

    total = len(image_paths)
    detection_rate = detections / total
    theme_match_rate = theme_matches / detections if detections else 0.0
    ready_rate = ready / total
    uncertain_rate = uncertain / total
    invalid_rate = invalid / total

    metrics = CalibrationMetrics(
        frame_count=total,
        detection_rate=detection_rate,
        theme_match_rate=theme_match_rate,
        ready_rate=ready_rate,
        uncertain_rate=uncertain_rate,
        invalid_rate=invalid_rate,
        p95_total_latency_ms=_p95(total_latencies),
        p95_detection_latency_ms=_p95(detection_latencies),
        p95_split_latency_ms=_p95(split_latencies),
        p95_recognition_latency_ms=_p95(recognition_latencies),
        status_counts=status_counts,
    )

    blockers: list[str] = []
    thresholds = profile.preflight_thresholds

    if detection_rate < thresholds.min_detection_rate:
        blockers.append(
            "Detection rate too low "
            f"({detection_rate:.1%} < {thresholds.min_detection_rate:.1%})."
        )
    if theme_match_rate < thresholds.min_theme_match_rate:
        blockers.append(
            "Theme match rate too low "
            f"({theme_match_rate:.1%} < {thresholds.min_theme_match_rate:.1%})."
        )
    if ready_rate < thresholds.min_ready_rate:
        blockers.append(
            "Ready frame rate too low "
            f"({ready_rate:.1%} < {thresholds.min_ready_rate:.1%})."
        )
    if uncertain_rate > thresholds.max_uncertain_rate:
        blockers.append(
            "Uncertain frame rate too high "
            f"({uncertain_rate:.1%} > {thresholds.max_uncertain_rate:.1%})."
        )
    if metrics.p95_total_latency_ms > thresholds.max_p95_total_latency_ms:
        blockers.append(
            "Total latency too high "
            f"(p95 {metrics.p95_total_latency_ms:.1f} ms > "
            f"{thresholds.max_p95_total_latency_ms:.1f} ms)."
        )

    warnings: list[str] = []
    if profile.support_level is not SupportLevel.OFFICIAL:
        warnings.append(
            f"Setup support level is '{profile.support_level.value}', not 'official'."
        )
    if profile.template_quality != "real":
        warnings.append(
            f"Template quality is '{profile.template_quality}'. "
            "Synthetic templates are not production-grade."
        )

    return CalibrationReport(
        setup_id=profile.id,
        support_level=profile.support_level.value,
        template_theme=profile.template_theme,
        expected_theme=expected_theme,
        passed=len(blockers) == 0,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        metrics=metrics,
    )


def resolve_image_paths(inputs: Sequence[str]) -> list[Path]:
    """Expand explicit files and glob patterns into sorted unique paths."""
    resolved: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.exists():
            if path.is_file():
                resolved.append(path.resolve())
            elif path.is_dir():
                for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
                    for candidate in sorted(path.glob(ext)):
                        if candidate.is_file():
                            resolved.append(candidate.resolve())
            continue

        for candidate in sorted(glob(raw)):
            candidate_path = Path(candidate)
            if candidate_path.is_file():
                resolved.append(candidate_path.resolve())

    unique_sorted = sorted(set(resolved))
    return unique_sorted


def _read_image(path: Path) -> NDArray:
    frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise ValueError(f"Unable to read image '{path}'.")
    return frame


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]
