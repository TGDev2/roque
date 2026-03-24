"""Game persistence and PGN import/export."""

from __future__ import annotations

from chesslab.persistence.game_storage import (
    GameRecord,
    GameStorage,
    GameSummary,
    ImportedGameSaveResult,
    PersistenceError,
    build_headers,
)
from chesslab.persistence.remote_import import (
    RemoteGameCandidate,
    RemoteGameClient,
    RemoteGameSource,
    RemoteImportBatchResult,
    RemoteImportError,
    RemoteImportFailure,
    RemoteImportNetworkError,
    RemoteImportService,
    RemoteImportUserNotFoundError,
)
from chesslab.persistence.review_history import (
    ReviewHistoryStorage,
    ReviewSummary,
    TrainingPositionSummary,
    build_review_summary,
)
from chesslab.persistence.training_history import (
    TrainingAttempt,
    TrainingAttemptOutcome,
    TrainingHistoryStorage,
)

__all__ = [
    "GameRecord",
    "GameStorage",
    "GameSummary",
    "ImportedGameSaveResult",
    "PersistenceError",
    "RemoteGameCandidate",
    "RemoteGameClient",
    "RemoteGameSource",
    "RemoteImportBatchResult",
    "RemoteImportError",
    "RemoteImportFailure",
    "RemoteImportNetworkError",
    "RemoteImportService",
    "RemoteImportUserNotFoundError",
    "ReviewHistoryStorage",
    "ReviewSummary",
    "TrainingAttempt",
    "TrainingAttemptOutcome",
    "TrainingHistoryStorage",
    "TrainingPositionSummary",
    "build_headers",
    "build_review_summary",
]
