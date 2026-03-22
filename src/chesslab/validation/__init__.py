"""Calibration and real-dataset validation helpers."""

from chesslab.validation.calibration import (
    CalibrationMetrics,
    CalibrationReport,
    resolve_image_paths,
    run_preflight_calibration,
    run_preflight_calibration_on_frames,
)
from chesslab.validation.catalog import (
    DatasetThresholds,
    PreflightThresholds,
    SetupSupportProfile,
    SupportCatalog,
    SupportCatalogError,
    SupportLevel,
    load_support_catalog,
)
from chesslab.validation.real_dataset import (
    AnnotatedSample,
    DatasetMetrics,
    DatasetValidationResult,
    load_real_dataset_manifest,
    validate_real_dataset,
)

__all__ = [
    "AnnotatedSample",
    "CalibrationMetrics",
    "CalibrationReport",
    "DatasetMetrics",
    "DatasetThresholds",
    "DatasetValidationResult",
    "PreflightThresholds",
    "SetupSupportProfile",
    "SupportCatalog",
    "SupportCatalogError",
    "SupportLevel",
    "load_real_dataset_manifest",
    "load_support_catalog",
    "resolve_image_paths",
    "run_preflight_calibration",
    "run_preflight_calibration_on_frames",
    "validate_real_dataset",
]
