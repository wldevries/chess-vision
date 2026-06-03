# ChessVision

Read a chess position from a photo and output [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation),
designed to generalize across many boards, piece sets, and environments. See `CLAUDE.md`
for the design notes and current state.

## Approach (image → FEN)

1. **Board localization** — predict the 4 board corners → a homography (the 9×9 lattice / 64
   squares are exact given the corners).
2. **Piece localization + classification** in the natural, un-warped photo — the model predicts
   each piece's class **and its board-contact point** (the base, where it meets the board).
3. **Square assignment** — map each contact point through the homography to a square, emit FEN.

More detail per stage: [`docs/piece-detection.md`](docs/piece-detection.md) (stage 2) and
[`docs/corner-capture-mode.md`](docs/corner-capture-mode.md) (collecting corner data for stage 1).

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned via `.python-version`.

```bash
uv sync              # create .venv and install deps from uv.lock
```

## Dataset (ChessReD)

Training and the geometry self-check use **ChessReD** (the Chess Recognition Dataset). It is
**not** committed — `data/` is gitignored — so you download it yourself from the official repo:

> https://github.com/tmasouris/end-to-end-chess-recognition

Follow that repo's download link (the images are a large archive), then **extract it under
`data/othersets/ChessReD/`** so the loader finds `annotations.json` and the images at the paths
it expects:

```
data/othersets/ChessReD/
├── annotations.json          # COCO-style labels (full set + the chessred2k detection subset)
├── chessred/images/          # the full ~10,800-image set (loader's default images_root)
└── chessred2k/               # the ~2,000-image subset that carries piece bounding boxes
```

`chessvision.data.chessred.ChessReD.load` reads `<data-root>/annotations.json` and defaults its
images to `<data-root>/chessred/images`, so the training/eval scripts take
`--data-root data/othersets/ChessReD`. Only **chessred2k** has piece boxes (Phase 2 detection);
the full set is position-only (square + class). Check the upstream repo for ChessReD's license
before redistributing.

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
> utility (Phase 1) and a ChessReD-trained piece detector (Phase 2, in progress). See `CLAUDE.md`
> for the phase order and current state.
