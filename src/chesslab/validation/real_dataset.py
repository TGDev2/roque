"""Validation on annotated real-world screenshot datasets."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2

from chesslab.detection.board_detector import BoardDetector, BoardRegion
from chesslab.live import LiveFrameStatus, LivePositionProcessor, LiveProcessorConfig
from chesslab.recognition.theme_detector import ThemeDetector
from chesslab.splitting.board_splitter import BoardSplitter
from chesslab.validation.catalog import SetupSupportProfile, SupportCatalog

_LATENCY_JITTER_MS = 5.0


@dataclass(frozen=True)
class AnnotatedSample:
    """One annotated frame from a real dataset."""

    setup_id: str
    image_path: Path
    board: BoardRegion
    expected_piece_placement: str | None = None


@dataclass(frozen=True)
class DatasetMetrics:
    """Aggregated E2E metrics for a setup dataset."""

    sample_count: int
    detection_rate: float
    mean_iou: float
    theme_match_rate: float
    ready_rate: float
    uncertain_rate: float
    invalid_rate: float
    p95_total_latency_ms: float
    p95_detection_latency_ms: float
    p95_split_latency_ms: float
    p95_recognition_latency_ms: float
    fen_match_rate: float | None
    status_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "detection_rate": round(self.detection_rate, 5),
            "mean_iou": round(self.mean_iou, 5),
            "theme_match_rate": round(self.theme_match_rate, 5),
            "ready_rate": round(self.ready_rate, 5),
            "uncertain_rate": round(self.uncertain_rate, 5),
            "invalid_rate": round(self.invalid_rate, 5),
            "p95_total_latency_ms": round(self.p95_total_latency_ms, 3),
            "p95_detection_latency_ms": round(self.p95_detection_latency_ms, 3),
            "p95_split_latency_ms": round(self.p95_split_latency_ms, 3),
            "p95_recognition_latency_ms": round(self.p95_recognition_latency_ms, 3),
            "fen_match_rate": (
                None if self.fen_match_rate is None else round(self.fen_match_rate, 5)
            ),
            "status_counts": dict(self.status_counts),
        }


@dataclass(frozen=True)
class DatasetValidationResult:
    """Validation result for one setup against real annotated captures."""

    setup_id: str
    support_level: str
    passed: bool
    failures: tuple[str, ...]
    metrics: DatasetMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "setup_id": self.setup_id,
            "support_level": self.support_level,
            "passed": self.passed,
            "failures": list(self.failures),
            "metrics": self.metrics.to_dict(),
        }


def load_real_dataset_manifest(path: Path) -> dict[str, list[AnnotatedSample]]:
    """Load setup-keyed samples from a manifest file.

    Supported schema:
    {
      "setups": [
        {
          "setup_id": "chesscom_green_white",
          "samples": [{ "image": "...", "board": {...} }]
        }
      ]
    }
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    grouped: dict[str, list[AnnotatedSample]] = {}

    setups = payload.get("setups")
    if not isinstance(setups, list):
        raise ValueError(
            f"Invalid manifest {path}: expected a top-level 'setups' list."
        )

    for idx, setup_entry in enumerate(setups):
        if not isinstance(setup_entry, dict):
            raise ValueError(f"Invalid setup entry at index {idx} in {path}.")
        setup_id = setup_entry.get("setup_id")
        if not isinstance(setup_id, str) or not setup_id.strip():
            raise ValueError(f"Missing setup_id at index {idx} in {path}.")

        samples_raw = setup_entry.get("samples")
        if not isinstance(samples_raw, list) or not samples_raw:
            raise ValueError(
                f"Setup '{setup_id}' must define a non-empty 'samples' list."
            )

        samples: list[AnnotatedSample] = []
        for s_idx, row in enumerate(samples_raw):
            if not isinstance(row, dict):
                raise ValueError(
                    f"Invalid sample at index {s_idx} in setup '{setup_id}'."
                )
            image = row.get("image")
            board_raw = row.get("board")
            if not isinstance(image, str) or not image.strip():
                raise ValueError(
                    f"Invalid image path in setup '{setup_id}' sample {s_idx}."
                )
            if not isinstance(board_raw, dict):
                raise ValueError(
                    f"Missing board annotation in setup '{setup_id}' sample {s_idx}."
                )

            sample = AnnotatedSample(
                setup_id=setup_id,
                image_path=(base / image).resolve(),
                board=BoardRegion(
                    x=int(board_raw["x"]),
                    y=int(board_raw["y"]),
                    width=int(board_raw["width"]),
                    height=int(board_raw["height"]),
                    confidence=1.0,
                ),
                expected_piece_placement=(
                    str(row["expected_piece_placement"])
                    if "expected_piece_placement" in row
                    else None
                ),
            )
            samples.append(sample)

        grouped[setup_id] = samples

    return grouped


def validate_real_dataset(
    *,
    manifest_path: Path,
    catalog: SupportCatalog,
    setup_ids: Sequence[str] | None = None,
    confidence_threshold: float = 0.60,
) -> list[DatasetValidationResult]:
    """Validate real captures setup by setup against the support catalog."""
    samples_by_setup = load_real_dataset_manifest(manifest_path)

    if setup_ids is None:
        target_setup_ids = sorted(samples_by_setup.keys())
    else:
        target_setup_ids = sorted(dict.fromkeys(setup_ids))

    results: list[DatasetValidationResult] = []

    for setup_id in target_setup_ids:
        profile = catalog.get_profile(setup_id)
        if profile is None:
            raise ValueError(f"Setup '{setup_id}' is not declared in support catalog.")
        samples = samples_by_setup.get(setup_id, [])
        if not samples:
            raise ValueError(
                f"Setup '{setup_id}' has no samples in manifest {manifest_path}."
            )
        results.append(
            _validate_profile_samples(
                profile=profile,
                samples=samples,
                confidence_threshold=confidence_threshold,
            )
        )

    return results


def _validate_profile_samples(
    *,
    profile: SetupSupportProfile,
    samples: Sequence[AnnotatedSample],
    confidence_threshold: float,
) -> DatasetValidationResult:
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

    warmup_frame = cv2.imread(str(samples[0].image_path), cv2.IMREAD_UNCHANGED)
    if warmup_frame is None:
        raise ValueError(f"Unable to read sample image: {samples[0].image_path}")
    processor.process_frame(warmup_frame)

    detections = 0
    ious: list[float] = []
    theme_matches = 0
    ready = 0
    uncertain = 0
    invalid = 0
    fen_checks = 0
    fen_matches = 0

    status_counts: dict[str, int] = {}
    total_latencies: list[float] = []
    detection_latencies: list[float] = []
    split_latencies: list[float] = []
    recognition_latencies: list[float] = []

    for sample in samples:
        frame = cv2.imread(str(sample.image_path), cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise ValueError(f"Unable to read sample image: {sample.image_path}")

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

        if sample.expected_piece_placement is not None:
            fen_checks += 1
            if live_result.piece_placement == sample.expected_piece_placement:
                fen_matches += 1

        detection = detector.detect(frame)
        if not detection.detected or detection.board is None:
            continue

        detections += 1
        ious.append(_iou(detection.board, sample.board))

        split = splitter.split(
            frame,
            detection.board,
            white_perspective=profile.perspective == "white",
        )
        detected_theme = theme_detector.detect(split.squares)
        if (
            detected_theme.matched
            and detected_theme.theme is not None
            and detected_theme.theme.qualified_name == profile.expected_theme_id
        ):
            theme_matches += 1

    total = len(samples)
    detection_rate = detections / total
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    theme_match_rate = theme_matches / detections if detections else 0.0
    ready_rate = ready / total
    uncertain_rate = uncertain / total
    invalid_rate = invalid / total
    fen_match_rate = fen_matches / fen_checks if fen_checks else None

    metrics = DatasetMetrics(
        sample_count=total,
        detection_rate=detection_rate,
        mean_iou=mean_iou,
        theme_match_rate=theme_match_rate,
        ready_rate=ready_rate,
        uncertain_rate=uncertain_rate,
        invalid_rate=invalid_rate,
        p95_total_latency_ms=_p95(total_latencies),
        p95_detection_latency_ms=_p95(detection_latencies),
        p95_split_latency_ms=_p95(split_latencies),
        p95_recognition_latency_ms=_p95(recognition_latencies),
        fen_match_rate=fen_match_rate,
        status_counts=status_counts,
    )

    failures: list[str] = []
    thresholds = profile.dataset_thresholds
    if thresholds is not None:
        if detection_rate < thresholds.min_detection_rate:
            failures.append(
                "Detection rate too low "
                f"({detection_rate:.1%} < {thresholds.min_detection_rate:.1%})."
            )
        if mean_iou < thresholds.min_mean_iou:
            failures.append(
                f"Mean IoU too low ({mean_iou:.3f} < {thresholds.min_mean_iou:.3f})."
            )
        if theme_match_rate < thresholds.min_theme_match_rate:
            failures.append(
                "Theme match rate too low "
                f"({theme_match_rate:.1%} < {thresholds.min_theme_match_rate:.1%})."
            )
        if ready_rate < thresholds.min_ready_rate:
            failures.append(
                "Ready frame rate too low "
                f"({ready_rate:.1%} < {thresholds.min_ready_rate:.1%})."
            )
        if uncertain_rate > thresholds.max_uncertain_rate:
            failures.append(
                "Uncertain frame rate too high "
                f"({uncertain_rate:.1%} > {thresholds.max_uncertain_rate:.1%})."
            )
        if invalid_rate > thresholds.max_invalid_rate:
            failures.append(
                "Invalid frame rate too high "
                f"({invalid_rate:.1%} > {thresholds.max_invalid_rate:.1%})."
            )
        if _latency_exceeds(
            metrics.p95_total_latency_ms,
            thresholds.max_p95_total_latency_ms,
        ):
            failures.append(
                "Total latency too high "
                f"(p95 {metrics.p95_total_latency_ms:.1f} ms > "
                f"{thresholds.max_p95_total_latency_ms:.1f} ms)."
            )
        if _latency_exceeds(
            metrics.p95_detection_latency_ms,
            thresholds.max_p95_detection_latency_ms,
        ):
            failures.append(
                "Detection latency too high "
                f"(p95 {metrics.p95_detection_latency_ms:.1f} ms > "
                f"{thresholds.max_p95_detection_latency_ms:.1f} ms)."
            )
        if _latency_exceeds(
            metrics.p95_split_latency_ms,
            thresholds.max_p95_split_latency_ms,
        ):
            failures.append(
                "Split latency too high "
                f"(p95 {metrics.p95_split_latency_ms:.1f} ms > "
                f"{thresholds.max_p95_split_latency_ms:.1f} ms)."
            )
        if _latency_exceeds(
            metrics.p95_recognition_latency_ms,
            thresholds.max_p95_recognition_latency_ms,
        ):
            failures.append(
                "Recognition latency too high "
                f"(p95 {metrics.p95_recognition_latency_ms:.1f} ms > "
                f"{thresholds.max_p95_recognition_latency_ms:.1f} ms)."
            )
    else:
        failures.append(
            "No dataset thresholds configured for this setup in support catalog."
        )

    return DatasetValidationResult(
        setup_id=profile.id,
        support_level=profile.support_level.value,
        passed=len(failures) == 0,
        failures=tuple(failures),
        metrics=metrics,
    )


def _iou(detected: BoardRegion, expected: BoardRegion) -> float:
    x1 = max(detected.x, expected.x)
    y1 = max(detected.y, expected.y)
    x2 = min(detected.x + detected.width, expected.x + expected.width)
    y2 = min(detected.y + detected.height, expected.y + expected.height)

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_det = detected.width * detected.height
    area_gt = expected.width * expected.height
    union = area_det + area_gt - inter
    if union <= 0:
        return 0.0
    return inter / union


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    idx = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[idx]


def _latency_exceeds(value_ms: float, threshold_ms: float) -> bool:
    return value_ms > threshold_ms + _LATENCY_JITTER_MS
