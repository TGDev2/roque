#!/usr/bin/env python3
"""Promote a setup from candidate to official in the support catalog.

Usage:
    python scripts/promote_setup.py chesscom_green_black [--dry-run]

Steps executed:
  1. Load the support catalog and verify the setup exists as a candidate.
  2. Run ``validate-real`` against the dataset manifest and verify all
     thresholds pass.
  3. Update ``supported_setups.json`` to set ``support_level`` to
     ``"official"`` and refresh the notes field.
  4. Re-run validation to confirm the promoted setup passes as official.
  5. Print a summary with the metrics.

Use ``--dry-run`` to run steps 1–2 without modifying the catalog.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chesslab.validation import load_support_catalog, validate_real_dataset  # noqa: E402
from chesslab.validation.catalog import SupportLevel  # noqa: E402


def _resolve_manifest(profile) -> Path | None:  # noqa: ANN001
    if profile.dataset_manifest is None:
        return None
    return (PROJECT_ROOT / profile.dataset_manifest).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("setup_id", help="ID of the setup to promote (e.g. chesscom_green_black)")
    parser.add_argument("--dry-run", action="store_true", help="Validate only; do not modify the catalog")
    args = parser.parse_args()

    catalog_path = PROJECT_ROOT / "src" / "chesslab" / "validation" / "supported_setups.json"
    catalog = load_support_catalog(catalog_path)

    # --- Step 1: verify setup exists and is promotable -----------------------
    profile = catalog.get_profile(args.setup_id)
    if profile is None:
        print(f"ERROR: setup '{args.setup_id}' not found in catalog.")
        sys.exit(1)

    if profile.support_level is SupportLevel.OFFICIAL:
        print(f"Setup '{args.setup_id}' is already official. Nothing to do.")
        sys.exit(0)

    if profile.support_level is not SupportLevel.CANDIDATE:
        print(
            f"ERROR: setup '{args.setup_id}' has support level "
            f"'{profile.support_level.value}'. Only candidate setups can be promoted."
        )
        sys.exit(1)

    if not profile.uses_real_templates:
        print(f"ERROR: setup '{args.setup_id}' uses synthetic templates. Cannot promote.")
        sys.exit(1)

    manifest_path = _resolve_manifest(profile)
    if manifest_path is None or not manifest_path.is_file():
        print(f"ERROR: no dataset manifest found for '{args.setup_id}'.")
        sys.exit(1)

    # --- Step 2: validate against real dataset --------------------------------
    print(f"Validating '{args.setup_id}' against {manifest_path.name} ...")
    results = validate_real_dataset(
        manifest_path=manifest_path,
        catalog=catalog,
        setup_ids=[args.setup_id],
    )

    result = results[0]
    print(f"  Samples:         {result.metrics.sample_count}")
    print(f"  Detection rate:  {result.metrics.detection_rate:.1%}")
    print(f"  Mean IoU:        {result.metrics.mean_iou:.3f}")
    print(f"  Theme match:     {result.metrics.theme_match_rate:.1%}")
    print(f"  Ready rate:      {result.metrics.ready_rate:.1%}")
    print(f"  Uncertain rate:  {result.metrics.uncertain_rate:.1%}")
    print(f"  Invalid rate:    {result.metrics.invalid_rate:.1%}")
    print(f"  Pass:            {result.passed}")

    if not result.passed:
        print("\nVALIDATION FAILED — cannot promote. Failures:")
        for f in result.failures:
            print(f"  - {f}")
        sys.exit(1)

    if args.dry_run:
        print("\n--dry-run: validation passed. Catalog not modified.")
        sys.exit(0)

    # --- Step 3: update the catalog ------------------------------------------
    with catalog_path.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    for entry in raw["setups"]:
        if entry["id"] == args.setup_id:
            entry["support_level"] = "official"
            entry["notes"] = (
                f"Official {profile.perspective}-perspective setup for Observe Live "
                f"on {profile.platform} {profile.site_theme} theme. "
                "Run preflight calibration before starting a session."
            )
            break

    with catalog_path.open("w", encoding="utf-8") as fh:
        json.dump(raw, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"\nCatalog updated: '{args.setup_id}' is now official.")

    # --- Step 4: re-validate after promotion ----------------------------------
    catalog2 = load_support_catalog(catalog_path)
    results2 = validate_real_dataset(
        manifest_path=manifest_path,
        catalog=catalog2,
        setup_ids=[args.setup_id],
    )
    if results2[0].passed:
        print("Post-promotion validation: PASS")
    else:
        print("WARNING: post-promotion validation FAILED. Review the catalog change.")
        sys.exit(1)

    # --- Step 5: summary ------------------------------------------------------
    print(f"\nPromotion complete for '{args.setup_id}'.")
    print("Next steps:")
    print("  1. Update tests that referenced the old candidate status.")
    print("  2. Run the full test suite: pytest tests/ -q")
    print("  3. Commit the catalog and test changes.")


if __name__ == "__main__":
    main()
