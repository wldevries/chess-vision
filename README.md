# ChessVision

Read a chess position from a photo and output [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation),
designed to generalize across many boards, piece sets, and environments. See [`plan.md`](plan.md)
for the full design.

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

> Inference is not implemented yet — the CLI is a Phase 0 stub. See `plan.md` §6 for the phase order.
