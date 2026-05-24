"""Command-line entry point.

Phase 0 stub: parses arguments and reports that inference is not implemented yet.
The real pipeline (board localization -> piece detection -> square assignment -> FEN)
is built out over the phases in plan.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    from chessvision import __version__

    parser = argparse.ArgumentParser(
        prog="chessvision",
        description="Read a chess position from a photo and output FEN.",
    )
    parser.add_argument("image", type=Path, help="path to the input photo")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.image.exists():
        parser.error(f"image not found: {args.image}")

    print(
        f"chessvision: inference is not implemented yet (Phase 0 stub).\n"
        f"  would process: {args.image}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
