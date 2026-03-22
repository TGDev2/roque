"""CLI for support perimeter listing, calibration, and real dataset validation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from chesslab.validation import CalibrationReport, DatasetValidationResult
from chesslab.validation.calibration import resolve_image_paths, run_preflight_calibration
from chesslab.validation.catalog import SetupSupportProfile, load_support_catalog
from chesslab.validation.real_dataset import validate_real_dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m chesslab.validation",
        description=(
            "Inspect setup support levels, run preflight calibration, "
            "and validate real annotated datasets."
        ),
    )
    parser.add_argument(
        "--catalog",
        default=None,
        help="Optional support catalog path (default: bundled JSON).",
    )
    parser.add_argument(
        "--log-level",
        default="ERROR",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity for underlying pipeline modules (default: ERROR).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    list_cmd = sub.add_parser("list", help="List supported setup profiles.")
    list_cmd.add_argument(
        "--official-only",
        action="store_true",
        help="Show only officially supported setups.",
    )
    list_cmd.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output instead of table text.",
    )

    preflight = sub.add_parser(
        "preflight",
        help="Run pre-session calibration on captured screenshots.",
    )
    preflight.add_argument(
        "--setup-id",
        required=True,
        help="Setup ID from the support catalog.",
    )
    preflight.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.60,
        help="Recognition confidence threshold (default: 0.60).",
    )
    preflight.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output.",
    )
    preflight.add_argument(
        "images",
        nargs="+",
        help="One or more image paths or glob patterns.",
    )

    validate_cmd = sub.add_parser(
        "validate-real",
        help="Run dataset validation on annotated real captures.",
    )
    validate_cmd.add_argument(
        "--manifest",
        required=True,
        help="Path to the setup-grouped dataset manifest JSON.",
    )
    validate_cmd.add_argument(
        "--setup-id",
        action="append",
        default=[],
        help="Restrict validation to one or more setup IDs.",
    )
    validate_cmd.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.60,
        help="Recognition confidence threshold (default: 0.60).",
    )
    validate_cmd.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    catalog_path = Path(args.catalog) if args.catalog else None
    catalog = load_support_catalog(catalog_path)

    if args.command == "list":
        return _cmd_list(catalog.profiles, official_only=args.official_only, as_json=args.json)

    if args.command == "preflight":
        profile = catalog.get_profile(args.setup_id)
        if profile is None:
            print(
                f"Unknown setup_id '{args.setup_id}'.",
                file=sys.stderr,
            )
            return 2
        paths = resolve_image_paths(args.images)
        if not paths:
            print("No image found from provided paths/patterns.", file=sys.stderr)
            return 2

        report = run_preflight_calibration(
            paths,
            profile,
            confidence_threshold=args.confidence_threshold,
        )
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            _print_preflight(report, profile)
        return 0 if report.passed else 1

    if args.command == "validate-real":
        results = validate_real_dataset(
            manifest_path=Path(args.manifest),
            catalog=catalog,
            setup_ids=args.setup_id or None,
            confidence_threshold=args.confidence_threshold,
        )
        if args.json:
            print(json.dumps([result.to_dict() for result in results], indent=2))
        else:
            _print_dataset_results(results)
        return 0 if all(result.passed for result in results) else 1

    parser.error(f"Unknown command: {args.command}")
    return 2


def _cmd_list(
    profiles: tuple[SetupSupportProfile, ...],
    *,
    official_only: bool,
    as_json: bool,
) -> int:
    rows = [
        profile
        for profile in profiles
        if not official_only or profile.is_official
    ]
    if as_json:
        payload = [
            {
                "id": row.id,
                "platform": row.platform,
                "site_theme": row.site_theme,
                "perspective": row.perspective,
                "template_theme": row.template_theme,
                "support_level": row.support_level.value,
                "template_quality": row.template_quality,
                "notes": row.notes,
            }
            for row in rows
        ]
        print(json.dumps(payload, indent=2))
        return 0

    print(
        f"{'Setup ID':<26} {'Support':<13} {'Template':<18} {'Perspective':<12} {'Quality':<10}"
    )
    print("-" * 86)
    if not rows:
        print("(no setup matches the current filters)")
        return 0
    for row in rows:
        print(
            f"{row.id:<26} "
            f"{row.support_level.value:<13} "
            f"{row.template_theme:<18} "
            f"{row.perspective:<12} "
            f"{row.template_quality:<10}"
        )
    return 0


def _print_preflight(report: CalibrationReport, profile: SetupSupportProfile) -> None:
    data = report.to_dict()
    print(f"Setup: {profile.id}")
    print(f"Support level: {profile.support_level.value}")
    print(f"Expected theme: {profile.expected_theme_id}")
    print(f"Template theme: {profile.template_theme}")
    print(f"Result: {'PASS' if data['passed'] else 'FAIL'}")

    metrics = data["metrics"]
    assert isinstance(metrics, dict)
    print(
        "Metrics: "
        f"detection={metrics['detection_rate']:.1%} "
        f"theme_match={metrics['theme_match_rate']:.1%} "
        f"ready={metrics['ready_rate']:.1%} "
        f"uncertain={metrics['uncertain_rate']:.1%} "
        f"p95_total={metrics['p95_total_latency_ms']:.1f}ms"
    )

    blockers = data["blockers"]
    warnings = data["warnings"]
    if blockers:
        print("Blockers:")
        for line in blockers:
            print(f"- {line}")
    if warnings:
        print("Warnings:")
        for line in warnings:
            print(f"- {line}")


def _print_dataset_results(results: list[DatasetValidationResult]) -> None:
    for result in results:
        data = result.to_dict()
        metrics = data["metrics"]
        assert isinstance(metrics, dict)
        print(
            f"{data['setup_id']}: {'PASS' if data['passed'] else 'FAIL'} "
            f"(support={data['support_level']}, "
            f"detection={metrics['detection_rate']:.1%}, "
            f"iou={metrics['mean_iou']:.3f}, "
            f"theme_match={metrics['theme_match_rate']:.1%}, "
            f"ready={metrics['ready_rate']:.1%}, "
            f"uncertain={metrics['uncertain_rate']:.1%}, "
            f"invalid={metrics['invalid_rate']:.1%})"
        )
        failures = data["failures"]
        if failures:
            for line in failures:
                print(f"- {line}")


if __name__ == "__main__":
    raise SystemExit(main())
