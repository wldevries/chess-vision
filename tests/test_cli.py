"""Smoke tests for the Phase 0 scaffold: confirms the package imports and the
CLI is wired up. Replaced/expanded as real functionality lands."""

import chessvision
from chessvision.cli import build_parser


def test_version_is_exposed():
    assert chessvision.__version__


def test_parser_requires_image():
    parser = build_parser()
    args = parser.parse_args(["board.jpg"])
    assert str(args.image) == "board.jpg"
