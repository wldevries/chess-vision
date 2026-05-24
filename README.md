# ChessVision

Read a chess position from a photo and output [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation),
designed to generalize across many boards, piece sets, and environments. See [`plan.md`](plan.md)
for the full design.

## Approach (image → FEN)

1. **Board localization** — predict the 4 board corners → a homography (the 9×9 lattice / 64
   squares are exact given the corners).
2. **Piece localization + classification** in the natural, un-warped photo — the model predicts
   each piece's class **and its board-contact point** (the base, where it meets the board).
3. **Square assignment** — map each contact point through the homography to a square, emit FEN.

> **Design rule (don't relitigate):** the board-contact point is **predicted directly**, never
> read off a detection box's bottom-center. Due to perspective and occlusion the box bottom is a
> biased proxy for the base; the homography→square step is exact given a *correct* contact point,
> so that's where accuracy is won. See `plan.md` §4 and `CLAUDE.md` anti-patterns.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned via `.python-version`.

```bash
uv sync              # create .venv and install deps from uv.lock
```

## Usage

```bash
uv run chessvision path/to/photo.jpg     # via console script
uv run python predict.py path/to/photo.jpg   # equivalent
```

## Development

```bash
uv run pytest        # run tests
uv run ruff check .  # lint
uv run ruff format . # format
```

> End-to-end inference is not wired up yet — the CLI is still a stub. Built so far: the homography
> utility (Phase 1) and a ChessReD-trained piece detector (Phase 2, in progress). See `plan.md` §6
> for the phase order.
