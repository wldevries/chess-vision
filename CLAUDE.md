# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

**Phase 0 (scaffold) is done.** Managed by `uv`, Python 3.12 (pinned in `.python-version`). Layout: `chessvision/` package (`cli.py` is a stub entry point), `predict.py` shim, `tests/`, `data/` (sample `IMG_*.jpg` photos), `models/`, `notebooks/`. `plan.md` is the authoritative design doc — read it before non-trivial work. The CLI does not do inference yet; everything from Phase 1 onward is unbuilt.

## Commands

```bash
uv sync                  # install deps from uv.lock into .venv
uv run pytest            # all tests
uv run pytest tests/test_cli.py::test_version_is_exposed   # single test
uv run ruff check .      # lint
uv run ruff format .     # format
uv run chessvision <img> # CLI (also: uv run python predict.py <img>)

uv run python scripts/sync_captures.py up    # push data/captures/ to MinIO
uv run python scripts/sync_captures.py down  # pull it back (size-based skip)
```

The captured dataset lives in a MinIO bucket on the local network (S3-compatible).
Config is in `.env` (gitignored; template in `.env.example`): `MINIO_ENDPOINT_URL`
(the API port `:9000`, not the console `:9001`), `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`,
`MINIO_BUCKET`. Helper module: `chessvision/data/storage.py` (boto3 + python-dotenv).
MinIO Community Edition no longer creates access keys in the web console — use
`mc admin accesskey create <alias> --access-key chess-app`.

Add a dependency with `uv add <pkg>` (or `uv add --dev <pkg>`); keep upper bounds on volatile deps (see Reproducibility). `uv.lock` is committed — never edit it by hand.

## Goal

Read a chess position from a single photo and output **FEN**. The hard requirement is **generalization across many boards, piece sets, and environments** — not making one sample setup work. Overfitting to a single domain is the explicit failure mode being designed against (a prior project failed exactly this way).

> Licensing constraint: this is a clean reimplementation. Do **not** copy from the prior (non-MIT) project's source. Verify the license of every dataset and reference system before use (`chesscog`, ChessReD, LiveChess2FEN, Roboflow sets).

## Architecture (planned pipeline: image → FEN)

Three stages, with a deliberate split between **learned localization** and **deterministic geometry**:

1. **Board localization** — a model predicts board geometry. Start with 4-corner regression; upgrade to an 81-point heatmap later only if occlusion hurts corner accuracy.
2. **Piece detection in the *original*, un-warped image** (Approach A — see below).
3. **Square assignment** — map each detection's ground-contact point to a square via the homography, then emit FEN.

### Geometry: 9×9 points ⇒ 8×8 squares
The board has 8×8 squares bounded by 9 lines per axis, crossing at an **81-point lattice**. Homography operates on *points*: define 81 canonical corners (e.g. `(i/8, j/8)` for `i,j in 0..8`), compute `H` from canonical corners → 4 detected image corners, project the 81 points, and the 64 squares are the gaps between adjacent points. **Build and unit-test this homography + "base-point → square" utility as a standalone module before any ML** (Phase 1).

### Approach A vs B (decided: A)
Warping flattens the board *plane*, but pieces are 3D objects standing above it — only the base touches the plane, so a warped piece's body shears toward the image edges.
- **Approach A (use this):** detect pieces in the natural un-warped image (they look normal; the model learns real appearance), then map the **bottom-center of each box** through the homography to find its square. Warping is used *only* for square assignment, never on piece images.
- **Approach B (fallback for dense/occluded positions only):** warp to top-down, slice 64 square crops, classify into 13 classes. Requires training on sheared warped crops with vertical headroom — a different data distribution.

Do not mix the two: training data must match the approach (natural images for A, warped crops for B).

## Hard constraints / anti-patterns

These approaches are **banned** — they assume the environment and do not generalize (all caused concrete failures in the prior project):
- **No color thresholds** (e.g. "warm board on cool background").
- **No Hough-line + intersection clustering** (brittle; blew up to TiB-scale memory / infinite loops on cluttered photos).
- **No hand-built lattice indexing** (desyncs the moment points aren't a perfect grid).

Generalization must come from *learned* localization + *deterministic* geometry, not tuned heuristics. Normalize (remove scale/perspective variance) before classifying. Remember pieces are tall and lean **away from the camera**, into the squares behind them — any square-based reasoning must account for this.

## Reproducibility (required from day one)

- Use `uv` for env/deps. Pin exact versions **with upper bounds** on volatile deps. Two specific landmines: **`numpy<2`** (NumPy 2.x breaks OpenCV) and **`torch.load(weights_only=...)`** (the default flipped) — always set it explicitly.
- Record dataset versions/hashes and train/val/test splits; keep a diverse **real held-out test set** for honest generalization numbers.
- Track experiments (CSV or W&B) so numbers are comparable across runs.
- Report both **per-square accuracy** and **whole-board (all-64-correct) accuracy**.

## Phase order

0 Scaffold → 1 Homography utility (unit-tested, manual corners) → 2 Piece detector baseline on natural images (get a real mAP) → 3 Board localizer (4-corner regression, replaces manual corners) → 4 Glue to FEN + eval → 5 Synthetic / domain randomization → 6 Harden & package. Each phase should produce a measured number before moving on.
