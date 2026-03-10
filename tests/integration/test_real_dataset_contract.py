"""Integration contract tests on a real annotated screenshot dataset.

Enable with:
    CHESSLAB_DATASET_MANIFEST=/abs/path/to/manifest.json ./.venv/bin/pytest tests/integration -q
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import pytest

from chesslab.detection.board_detector import BoardDetector, BoardRegion
from chesslab.splitting.board_splitter import BoardSplitter

DEFAULT_MANIFEST = Path("tests/integration/fixtures/manifest.json")
DATASET_MANIFEST = Path(os.environ.get("CHESSLAB_DATASET_MANIFEST", DEFAULT_MANIFEST))

if not DATASET_MANIFEST.exists():
    pytestmark = pytest.mark.skip(
        reason=(
            "Real dataset manifest not found. "
            "Set CHESSLAB_DATASET_MANIFEST to enable integration checks."
        )
    )


@dataclass(frozen=True)
class Sample:
    image_path: Path
    x: int
    y: int
    width: int
    height: int


def _iou(region: BoardRegion, sample: Sample) -> float:
    x1 = max(region.x, sample.x)
    y1 = max(region.y, sample.y)
    x2 = min(region.x + region.width, sample.x + sample.width)
    y2 = min(region.y + region.height, sample.y + sample.height)

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_det = region.width * region.height
    area_gt = sample.width * sample.height
    union = area_det + area_gt - inter
    return inter / union if union > 0 else 0.0


def _load_samples(manifest_path: Path) -> list[Sample]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    samples: list[Sample] = []
    for row in data:
        board = row["board"]
        samples.append(
            Sample(
                image_path=(base / row["image"]).resolve(),
                x=int(board["x"]),
                y=int(board["y"]),
                width=int(board["width"]),
                height=int(board["height"]),
            )
        )
    return samples


def test_real_dataset_detection_and_split_contract() -> None:
    samples = _load_samples(DATASET_MANIFEST)
    if not samples:
        pytest.skip("Dataset manifest is present but empty.")

    detector = BoardDetector()
    splitter = BoardSplitter()

    detections = 0
    ious: list[float] = []

    for sample in samples:
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_UNCHANGED)
        assert image is not None, f"Failed to read image: {sample.image_path}"

        result = detector.detect(image)
        if not result.detected or result.board is None:
            continue

        detections += 1
        iou = _iou(result.board, sample)
        ious.append(iou)
        assert result.elapsed_ms < 300, (
            f"{sample.image_path.name}: detection latency {result.elapsed_ms:.1f} ms >= 300"
        )

        split = splitter.split(image, result.board)
        assert len(split.squares) == 64
        assert split.elapsed_ms < 50, (
            f"{sample.image_path.name}: split latency {split.elapsed_ms:.1f} ms >= 50"
        )

    detection_rate = detections / len(samples)
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    assert detection_rate >= 0.90, (
        f"Detection rate {detection_rate:.2%} < 90% on real dataset"
    )
    assert mean_iou >= 0.90, f"Mean IoU {mean_iou:.3f} < 0.90 on real dataset"
