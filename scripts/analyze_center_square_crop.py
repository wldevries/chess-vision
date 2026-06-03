"""What would a centered-square crop do to the current corner labels?

For each labeled corner-dataset image, take the largest centered square
(side = min(w,h)) and check whether all 4 labelled corners still fall inside it.
Reports, overall and per board: how many images keep all corners, how many lose
>=1, and how far outside the worst corner lands (as a fraction of the crop side).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

LABELS = Path("data/labels.jsonl")


def square_bounds(w: int, h: int):
    side = min(w, h)
    if w >= h:  # landscape: crop left/right
        x_lo, x_hi, y_lo, y_hi = (w - side) / 2, (w + side) / 2, 0, h
    else:  # portrait: crop top/bottom
        x_lo, x_hi, y_lo, y_hi = 0, w, (h - side) / 2, (h + side) / 2
    return side, x_lo, x_hi, y_lo, y_hi


def main() -> None:
    by_board = defaultdict(lambda: {"n": 0, "all_in": 0, "lost": 0, "max_over": 0.0, "overs": []})
    for line in LABELS.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        w, h = r["width"], r["height"]
        side, x_lo, x_hi, y_lo, y_hi = square_bounds(w, h)
        worst_over = 0.0  # max distance outside the box, frac of side
        for x, y in r["corners"].values():
            ox = max(x_lo - x, x - x_hi, 0.0)
            oy = max(y_lo - y, y - y_hi, 0.0)
            worst_over = max(worst_over, ox / side, oy / side)
        b = by_board[r.get("board") or "(untagged)"]
        b["n"] += 1
        if worst_over <= 0:
            b["all_in"] += 1
        else:
            b["lost"] += 1
            b["max_over"] = max(b["max_over"], worst_over)
            b["overs"].append(worst_over)

    tot = defaultdict(float)
    print(f"{'board':16s} {'n':>4} {'all-in':>7} {'lose>=1':>8} {'lost%':>6} {'worst-over':>11}")
    for board in sorted(by_board):
        b = by_board[board]
        pct = 100 * b["lost"] / b["n"]
        print(
            f"{board:16s} {b['n']:>4} {b['all_in']:>7} {b['lost']:>8} {pct:>5.0f}% "
            f"{b['max_over']*100:>9.1f}% of side"
        )
        tot["n"] += b["n"]
        tot["all_in"] += b["all_in"]
        tot["lost"] += b["lost"]
    print(
        f"{'ALL':16s} {int(tot['n']):>4} {int(tot['all_in']):>7} {int(tot['lost']):>8} "
        f"{100*tot['lost']/tot['n']:>5.0f}%"
    )
    # how severe are the losses? buckets of worst-overshoot
    all_overs = sorted(o for b in by_board.values() for o in b["overs"])
    if all_overs:
        print("\nworst-corner overshoot among the clipped images (frac of crop side):")
        for q, label in [(0.5, "median"), (0.9, "p90"), (1.0, "max")]:
            v = all_overs[min(len(all_overs) - 1, int(q * (len(all_overs) - 1)))]
            print(f"  {label:6s}: {v*100:.1f}%")


if __name__ == "__main__":
    main()
