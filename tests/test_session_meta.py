"""Session metadata -> per-piece box sizing (offline, in-memory; no files/images)."""

from __future__ import annotations

from chessvision.data.session_meta import SessionMetadata, resolve_box_sizes


def test_resolve_box_sizes_from_mm():
    set_def = {
        "k": {"height_mm": 95.0, "base_mm": 38.0},
        "q": {"height_mm": 80.0},  # no base -> default radius
        "p": {"height_mm": None, "base_mm": 20.0},  # unmeasured height -> omitted
    }
    sizes = resolve_box_sizes(set_def, square_mm=50.0, default_radius_squares=0.3)
    assert sizes["k"] == (95.0 / 50.0, (38.0 / 2) / 50.0)  # (1.9, 0.38)
    assert sizes["q"] == (80.0 / 50.0, 0.3)  # falls back to default radius
    assert "p" not in sizes  # null height -> caller uses PIECE_HEIGHT_SCALE


def test_resolve_box_sizes_no_square():
    assert resolve_box_sizes({"k": {"height_mm": 95.0}}, square_mm=None) == {}
    assert resolve_box_sizes({"k": {"height_mm": 95.0}}, square_mm=0) == {}


def test_session_metadata_resolution_and_fallbacks():
    meta = SessionMetadata(
        sets={"big": {"k": {"height_mm": 100.0, "base_mm": 40.0}}},
        boards={"small": {"square_mm": 40.0}, "unmeasured": {"square_mm": None}},
        sessions={
            "s1": {"set": "big", "board": "small"},
            "s2": {"set": "big", "board": "unmeasured"},
            "s3": {"set": "missing", "board": "small"},
        },
    )
    assert meta.piece_box_sizes("s1")["k"] == (100.0 / 40.0, (40.0 / 2) / 40.0)
    assert meta.piece_box_sizes("s2") == {}  # board square_mm unmeasured
    assert meta.piece_box_sizes("s3") == {}  # unknown set
    assert meta.piece_box_sizes("unknown") == {}  # unknown session
    assert meta.info("s1")["board"] == "small"


def test_load_returns_none_when_absent(tmp_path):
    assert SessionMetadata.load(tmp_path) is None
