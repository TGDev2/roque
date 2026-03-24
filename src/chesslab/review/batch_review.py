"""Batch orchestration for post-game review persistence.

Turns a list of saved PGNs into persisted review summaries while surfacing
explicit statuses for already reviewed games, duplicate selections, failures,
and cancellations.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence

from chesslab.persistence.game_storage import GameStorage, PersistenceError
from chesslab.persistence.review_history import ReviewHistoryStorage, ReviewSummary
from chesslab.review.game_reviewer import (
    GameReview,
    GameReviewer,
    ReviewCancelledError,
    ReviewConfig,
    ReviewError,
)


@dataclass(frozen=True)
class BatchReviewTarget:
    """Saved PGN selected for batch review."""

    file_path: Path
    label: str | None = None

    @property
    def display_label(self) -> str:
        if self.label:
            return self.label
        return self.file_path.name


class BatchReviewItemStatus(str, Enum):
    """Outcome for one target inside a batch review run."""

    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED_ALREADY_REVIEWED = "skipped_already_reviewed"
    SKIPPED_DUPLICATE_SELECTION = "skipped_duplicate_selection"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class BatchReviewItemResult:
    """Result for one reviewed or skipped PGN."""

    target: BatchReviewTarget
    status: BatchReviewItemStatus
    message: str
    review: GameReview | None = None
    summary: ReviewSummary | None = None


@dataclass(frozen=True)
class BatchReviewResult:
    """Aggregate outcome of a batch review run."""

    items: tuple[BatchReviewItemResult, ...]
    cancelled: bool = False

    @property
    def completed_count(self) -> int:
        return sum(
            item.status is BatchReviewItemStatus.COMPLETED
            for item in self.items
        )

    @property
    def failed_count(self) -> int:
        return sum(
            item.status is BatchReviewItemStatus.FAILED
            for item in self.items
        )

    @property
    def skipped_reviewed_count(self) -> int:
        return sum(
            item.status is BatchReviewItemStatus.SKIPPED_ALREADY_REVIEWED
            for item in self.items
        )

    @property
    def skipped_duplicate_count(self) -> int:
        return sum(
            item.status is BatchReviewItemStatus.SKIPPED_DUPLICATE_SELECTION
            for item in self.items
        )

    @property
    def cancelled_count(self) -> int:
        return sum(
            item.status is BatchReviewItemStatus.CANCELLED
            for item in self.items
        )

    @property
    def processed_count(self) -> int:
        return len(self.items)


BatchReviewItemStartedCallback = Callable[[int, BatchReviewTarget], None]
BatchReviewItemProgressCallback = Callable[[int, BatchReviewTarget, int, int], None]
BatchReviewItemFinishedCallback = Callable[[int, BatchReviewItemResult], None]


class BatchReviewService:
    """Review saved PGNs sequentially and persist each successful result."""

    def __init__(
        self,
        *,
        storage: GameStorage | None = None,
        review_history: ReviewHistoryStorage | None = None,
        review_config: ReviewConfig | None = None,
    ) -> None:
        self._storage = storage or GameStorage()
        self._review_history = review_history or ReviewHistoryStorage()
        self._review_config = review_config or ReviewConfig()

    def run(
        self,
        targets: Sequence[BatchReviewTarget],
        *,
        on_item_started: BatchReviewItemStartedCallback | None = None,
        on_item_progress: BatchReviewItemProgressCallback | None = None,
        on_item_finished: BatchReviewItemFinishedCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> BatchReviewResult:
        """Review each selected PGN and persist completed summaries."""
        results: list[BatchReviewItemResult] = []
        seen_paths: set[str] = set()
        reviewed_paths = {
            summary.source_path
            for summary in self._review_history.list_recent_reviews(limit=None)
            if summary.source_path
        }
        cancelled = False

        for index, target in enumerate(targets):
            resolved_path = str(target.file_path.resolve())
            if resolved_path in seen_paths:
                result = BatchReviewItemResult(
                    target=target,
                    status=BatchReviewItemStatus.SKIPPED_DUPLICATE_SELECTION,
                    message="Selected more than once in this batch.",
                )
                results.append(result)
                if on_item_finished is not None:
                    on_item_finished(index, result)
                continue

            seen_paths.add(resolved_path)

            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break

            if resolved_path in reviewed_paths:
                result = BatchReviewItemResult(
                    target=target,
                    status=BatchReviewItemStatus.SKIPPED_ALREADY_REVIEWED,
                    message="Already present in review history.",
                )
                results.append(result)
                if on_item_finished is not None:
                    on_item_finished(index, result)
                continue

            try:
                record = self._storage.load_game(target.file_path)
            except PersistenceError as exc:
                result = BatchReviewItemResult(
                    target=target,
                    status=BatchReviewItemStatus.FAILED,
                    message=str(exc),
                )
                results.append(result)
                if on_item_finished is not None:
                    on_item_finished(index, result)
                continue

            if not record.board.move_stack:
                result = BatchReviewItemResult(
                    target=target,
                    status=BatchReviewItemStatus.FAILED,
                    message="No moves to review.",
                )
                results.append(result)
                if on_item_finished is not None:
                    on_item_finished(index, result)
                continue

            if on_item_started is not None:
                on_item_started(index, target)

            reviewer = GameReviewer(self._review_config)
            try:
                review = reviewer.review(
                    record.board.copy(),
                    progress_callback=(
                        None
                        if on_item_progress is None
                        else lambda current, total, *, _index=index, _target=target: (
                            on_item_progress(_index, _target, current, total)
                        )
                    ),
                    cancel_event=cancel_event,
                )
            except ReviewCancelledError:
                result = BatchReviewItemResult(
                    target=target,
                    status=BatchReviewItemStatus.CANCELLED,
                    message="Cancelled during analysis.",
                )
                results.append(result)
                if on_item_finished is not None:
                    on_item_finished(index, result)
                cancelled = True
                break
            except ReviewError as exc:
                result = BatchReviewItemResult(
                    target=target,
                    status=BatchReviewItemStatus.FAILED,
                    message=str(exc),
                )
                results.append(result)
                if on_item_finished is not None:
                    on_item_finished(index, result)
                continue

            source_path = record.file_path or target.file_path
            try:
                summary = self._review_history.record_review(
                    review,
                    source_path=source_path,
                )
            except PersistenceError as exc:
                result = BatchReviewItemResult(
                    target=target,
                    status=BatchReviewItemStatus.FAILED,
                    message=f"Review could not be saved: {exc}",
                    review=review,
                )
                results.append(result)
                if on_item_finished is not None:
                    on_item_finished(index, result)
                continue

            reviewed_paths.add(str(source_path.resolve()))
            result = BatchReviewItemResult(
                target=target,
                status=BatchReviewItemStatus.COMPLETED,
                message=(
                    f"Reviewed {len(review.moves)} move"
                    f"{'' if len(review.moves) == 1 else 's'}."
                ),
                review=review,
                summary=summary,
            )
            results.append(result)
            if on_item_finished is not None:
                on_item_finished(index, result)

        return BatchReviewResult(items=tuple(results), cancelled=cancelled)
