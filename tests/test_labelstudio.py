"""Tests for building Label Studio pre-annotation tasks from capture records."""

from __future__ import annotations

from chessvision.data import labelstudio as ls

# A square footprint roughly 100px on a side near the top-left of a 1000x800 image.
PIECE = {
    "square": "e2",
    "piece": "q",
    "color": "w",
    "base": [250.0, 400.0],
    "quad": [[200.0, 350.0], [300.0, 350.0], [300.0, 450.0], [200.0, 450.0]],
}

RECORD = {
    "filename": "euwe-0000_ply000.jpg",
    "corners": {
        "top_left": [100.0, 50.0],
        "top_right": [900.0, 60.0],
        "bottom_right": [880.0, 760.0],
        "bottom_left": [120.0, 740.0],
    },
    "pieces": [PIECE, {**PIECE, "piece": "p", "color": "b", "square": "e7"}],
}

SIZE = (1000, 800)


def test_piece_label_is_verbose_color_plus_name() -> None:
    assert ls.piece_label({"piece": "Q", "color": "w"}) == "WhiteQueen"
    assert ls.piece_label({"piece": "n", "color": "b"}) == "BlackKnight"


def test_build_task_is_points_only_by_default() -> None:
    task = ls.build_task(RECORD, SIZE, "s3://chess/captures/x.jpg")
    assert task["data"]["image"] == "s3://chess/captures/x.jpg"
    assert len(task["predictions"]) == 1
    result = task["predictions"][0]["result"]
    # 4 corner keypoints + 1 base keypoint per piece, no boxes.
    assert len(result) == 4 + len(RECORD["pieces"])
    assert sum(r["from_name"] == "corners" for r in result) == 4
    assert sum(r["from_name"] == "pieces" for r in result) == 2
    assert all(r["type"] == "keypointlabels" for r in result)
    corner_labels = {r["value"]["keypointlabels"][0] for r in result if r["from_name"] == "corners"}
    assert corner_labels == {"TopLeft", "TopRight", "BottomRight", "BottomLeft"}


def test_include_boxes_adds_rectangles_on_boxes_control() -> None:
    result = ls.build_task(RECORD, SIZE, "s3://x", include_boxes=True)["predictions"][0]["result"]
    # corners + (box + point) per piece.
    assert len(result) == 4 + 2 * len(RECORD["pieces"])
    boxes = [r for r in result if r["from_name"] == "boxes"]
    assert len(boxes) == 2
    assert all(r["type"] == "rectanglelabels" for r in boxes)


def test_omits_board_when_corners_absent() -> None:
    result = ls.build_task({"pieces": []}, SIZE, "s3://x")["predictions"][0]["result"]
    assert result == []


def test_all_coordinates_are_percentages_in_range() -> None:
    result = ls.build_task(RECORD, SIZE, "s3://x")["predictions"][0]["result"]
    for region in result:
        v = region["value"]
        for key in ("x", "y", "width", "height"):
            if key in v:
                assert 0.0 <= v[key] <= 100.0
        for px, py in v.get("points", []):
            assert 0.0 <= px <= 100.0 and 0.0 <= py <= 100.0


def test_rectangle_is_centred_on_base_and_rises_from_footprint() -> None:
    result = ls.build_task(RECORD, SIZE, "s3://x", include_boxes=True)["predictions"][0]["result"]
    rect = next(r for r in result if r["type"] == "rectanglelabels")
    v = rect["value"]
    # base x = 250/1000 = 25%; box centred on it.
    center_x = v["x"] + v["width"] / 2
    assert abs(center_x - 25.0) < 1e-6
    # Bottom at the footprint's front edge (y=450 -> 56.25%).
    bottom = v["y"] + v["height"]
    assert abs(bottom - 56.25) < 1e-6


def test_piece_keypoint_sits_at_the_base_point() -> None:
    result = ls.build_task(RECORD, SIZE, "s3://x")["predictions"][0]["result"]
    kp = next(r for r in result if r["from_name"] == "pieces")
    assert kp["value"]["x"] == 25.0  # 250/1000
    assert kp["value"]["y"] == 50.0  # 400/800
    assert kp["value"]["keypointlabels"] == ["WhiteQueen"]


def test_labeling_config_lists_all_twelve_piece_labels() -> None:
    for color in ("White", "Black"):
        for name in ("Pawn", "Knight", "Bishop", "Rook", "Queen", "King"):
            assert f'value="{color}{name}"' in ls.LABELING_CONFIG
