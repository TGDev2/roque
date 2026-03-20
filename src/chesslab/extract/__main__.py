"""CLI entry point for template extraction.

Usage::

    python -m chesslab.extract <screenshot> [options]
    python -m chesslab.extract --list-themes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from chesslab.extract.extractor import ExtractionError, TemplateExtractor

_THEMES_JSON = Path(__file__).resolve().parents[1] / "recognition" / "themes.json"
_TEMPLATES_ROOT = Path(__file__).resolve().parents[1] / "recognition" / "templates"


def _qualified_dir_name(source: str, name: str) -> str:
    return f"{source}-{name}"


def _load_signatures() -> list[dict[str, object]]:
    with _THEMES_JSON.open() as fh:
        return json.load(fh)["themes"]


def _list_themes() -> None:
    """Print all known theme signatures."""
    sigs = _load_signatures()
    print(f"{'Theme':<25} {'Light BGR':<20} {'Dark BGR':<20}")
    print("-" * 65)
    for sig in sigs:
        name = _qualified_dir_name(str(sig["source"]), str(sig["name"]))
        light = sig["light_bgr"]
        dark = sig["dark_bgr"]
        has_templates = (_TEMPLATES_ROOT / name / "theme.json").exists()
        marker = " [templates]" if has_templates else ""
        print(f"{name:<25} {str(light):<20} {str(dark):<20}{marker}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m chesslab.extract",
        description=(
            "Extract piece templates from a starting-position screenshot. "
            "The screenshot must show a standard chess start position."
        ),
    )
    parser.add_argument(
        "screenshot",
        nargs="?",
        help="Path to a starting-position screenshot.",
    )
    parser.add_argument(
        "--theme",
        default=None,
        help=(
            "Theme qualified name (e.g. 'chess.com/brown' or 'chess.com-brown'). "
            "If omitted, auto-detection is attempted."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: auto into recognition/templates/).",
    )
    parser.add_argument(
        "--list-themes",
        action="store_true",
        help="List known theme signatures and exit.",
    )
    parser.add_argument(
        "--white-perspective",
        action="store_true",
        default=True,
        help="Board is from white's perspective (default: True).",
    )
    parser.add_argument(
        "--black-perspective",
        action="store_true",
        default=False,
        help="Board is from black's perspective.",
    )
    return parser.parse_args()


def _normalize_theme_name(theme: str) -> str:
    """Normalise user input like 'chess.com/brown' to 'chess.com-brown'."""
    return theme.replace("/", "-").strip()


def main() -> None:
    args = _parse_args()

    if args.list_themes:
        _list_themes()
        return

    if not args.screenshot:
        print("Error: screenshot argument is required.", file=sys.stderr)
        print("Usage: python -m chesslab.extract <screenshot> [--theme NAME]", file=sys.stderr)
        print("       python -m chesslab.extract --list-themes", file=sys.stderr)
        sys.exit(1)

    screenshot = Path(args.screenshot)
    if not screenshot.is_file():
        print(f"Error: file not found: {screenshot}", file=sys.stderr)
        sys.exit(1)

    white_perspective = not args.black_perspective

    try:
        extractor = TemplateExtractor(screenshot, white_perspective=white_perspective)
    except ExtractionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Determine theme name.
    if args.theme:
        theme_name = _normalize_theme_name(args.theme)
    else:
        # Auto-detect from the screenshot.
        print("Detecting theme...")
        detection = extractor.detect_theme()
        if detection.matched and detection.theme is not None:
            theme_name = _qualified_dir_name(
                detection.theme.source, detection.theme.name,
            )
            print(f"Detected theme: {theme_name} (distance: {detection.distance:.1f})")
            print(f"  observed light BGR: {detection.observed_light_bgr}")
            print(f"  observed dark BGR:  {detection.observed_dark_bgr}")

            response = input(f"Use '{theme_name}' as theme name? [Y/n] ").strip().lower()
            if response and response not in ("y", "yes"):
                theme_name = input("Enter theme name (e.g. 'chess.com-custom'): ").strip()
                if not theme_name:
                    print("Error: theme name cannot be empty.", file=sys.stderr)
                    sys.exit(1)
        else:
            print("Could not auto-detect theme.")
            print(f"  observed light BGR: {detection.observed_light_bgr}")
            print(f"  observed dark BGR:  {detection.observed_dark_bgr}")
            print(f"  closest match distance: {detection.distance:.1f} (threshold: 30.0)")
            theme_name = input("Enter theme name (e.g. 'chess.com-custom'): ").strip()
            if not theme_name:
                print("Error: theme name cannot be empty.", file=sys.stderr)
                sys.exit(1)

    theme_name = _normalize_theme_name(theme_name)
    output_dir = Path(args.output_dir) if args.output_dir else None

    print(f"Extracting templates for '{theme_name}'...")
    try:
        result = extractor.extract(theme_name, output_dir)
    except ExtractionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracted {result.template_count} templates to {result.output_dir}")
    print(f"  synthesised: {result.synthesised_count}")
    print(f"  light BGR: {result.light_bgr}")
    print(f"  dark BGR:  {result.dark_bgr}")
    print("Done.")


if __name__ == "__main__":
    main()
