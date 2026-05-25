# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

**Phases 0–1 done; Phase 2 in progress.** Managed by `uv`, Python 3.12 (pinned in `.python-version`). Layout: `chessvision/` package (`cli.py` is a stub entry point), `predict.py` shim, `tests/`, `data/`, `models/`, `notebooks/`. `plan.md` is the authoritative design doc — read it before non-trivial work.

- **Phase 1 (homography)** is built and validated: `chessvision/geometry.py` + `selfcheck.py`, self-check at 99.96% **on ground-truth contact points** (this validates the geometry, *not* any box→point heuristic — see the contact-point anti-pattern).
- **Phase 2 (piece detector)** — box-detector baseline **trained**: Faster R-CNN ResNet50-FPN v2 on the official chessred2k split, **best val mAP 0.864, mAP@50 0.999** (`chessvision/detector.py`, `chessvision/data/detection.py`, `scripts/train_detector.py`; weights in `runs/detector/best.pt`). This is a *box* detector, so its box→contact step is the **known weak link** — the next step is to **transplant a base-keypoint head** onto this trunk (see "Contact points" below).
- **Contact points** — `chessvision/data/contact.py` generates the doctrine-pure base point (each piece's square center projected through the homography). **Visually validated** on the most-occluded boards (`scripts/build_occlusion_tasks.py --overlay-dir`): the point lands at the base even when the base is hidden. Key consequence: contact-point labels are **auto-generated geometric truth**, so the keypoint head needs **no manual labelling** — the S3/Label-Studio review loop is *optional QA*, not a training prerequisite.
- **Phase 3 (corner regression)** not built — corners are still manual.
- CUDA torch is pinned via the `cu128` index in `pyproject.toml` (`[tool.uv.sources]`); don't let a bare `uv pip install torch` drift it back to CPU. Keep Label Studio in a *separate* venv (it drags `opencv-python-headless`, which collides with our `opencv-python`).

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
uv run python scripts/sync_captures.py tasks # build Label Studio pre-annotations -> bucket tasks/

# Read-position (live FEN, Phase 4) mode in the web app: camera -> mark corners -> predicted FEN.
# OFF unless a keypoint checkpoint is passed; toggle "Read position" in the header.
uv run python -m chessvision.capture --keypoint-ckpt runs/keypoint_captures/best.pt
```

Read-position mode glue is `chessvision/inference.py` (`LivePredictor`/`build_prediction`)
behind `POST /api/live/predict`. It detects pieces, maps each contact keypoint through the
homography to a square, and emits one board FEN per orientation (R0..R270) in a single pass —
the same detection relabelled four ways. **Board orientation is a deliberate manual toggle**
(which physical corner is a8 is NOT recoverable from geometry; all four rotations are valid),
so the UI suggests one and the user rotates to the reading that matches reality. Clicked
corners are sorted into TL/TR/BR/BL by `geometry.order_corners` (click order doesn't matter).

`tasks` reads local captures (marked corners + FEN-projected piece estimates) and
writes one Label Studio task-JSON per frame to the same bucket under `tasks/` (not
stored locally). Tasks are points-only: four labelled corner keypoints (`corners`
control) plus one labelled keypoint per piece at its base (`pieces` control), verbose
labels (`WhiteRook`, `TopLeft`, ...). Boxes are deferred — `--with-boxes` additionally
emits approximate piece bounding boxes on a `boxes` control for a later detection pass.
Point a Label Studio source storage
at `s3://<bucket>/tasks/` with "Treat every bucket object as a source file" OFF, and
paste the labelling config printed by the command into the project. Builder lives in
`chessvision/data/labelstudio.py`.

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
2. **Piece localization + classification in the *original*, un-warped image** (Approach A — see below). The model predicts, per piece, its **class** and its **board-contact point** (the base, where it meets the board).
3. **Square assignment** — map each piece's predicted contact point to a square via the homography, then emit FEN. The contact point is predicted directly, **never derived from a bounding box** (see anti-patterns).

### Geometry: 9×9 points ⇒ 8×8 squares
The board has 8×8 squares bounded by 9 lines per axis, crossing at an **81-point lattice**. Homography operates on *points*: define 81 canonical corners (e.g. `(i/8, j/8)` for `i,j in 0..8`), compute `H` from canonical corners → 4 detected image corners, project the 81 points, and the 64 squares are the gaps between adjacent points. **Build and unit-test this homography + "base-point → square" utility as a standalone module before any ML** (Phase 1).

### Approach A vs B (decided: A)
Warping flattens the board *plane*, but pieces are 3D objects standing above it — only the base touches the plane, so a warped piece's body shears toward the image edges.
- **Approach A (use this):** localize + classify pieces in the natural un-warped image (they look normal; the model learns real appearance) **and predict each piece's board-contact point directly** (a keypoint at the base). Map that contact point through the homography to find its square. Warping is used *only* for square assignment, never on piece images. **Do not read the contact point off a detection box's bottom-center** — see the contact-point anti-pattern below.
- **Approach B (fallback for dense/occluded positions only):** warp to top-down, slice 64 square crops, classify into 13 classes. Requires training on sheared warped crops with vertical headroom — a different data distribution.

Do not mix the two: training data must match the approach (natural images for A, warped crops for B).

## Hard constraints / anti-patterns

These approaches are **banned** — they assume the environment and do not generalize (all caused concrete failures in the prior project):
- **No color thresholds** (e.g. "warm board on cool background").
- **No Hough-line + intersection clustering** (brittle; blew up to TiB-scale memory / infinite loops on cluttered photos). *Corollary, measured 2026-05: refining the learned corner regressor's output by snapping the projected board grid onto a Scharr edge field (both full-grid and boundary-only variants) **regressed** corners — ChessReD val 0.0075→0.0079, captures 0.0225→0.0572 — because maximizing edge-overlap ≠ corner-correctness when pieces, wood grain, and multi-edge frames are strong false edges. The learned global estimate beats local edge-fitting; don't retry grid/edge corner refinement.*
- **No hand-built lattice indexing** (desyncs the moment points aren't a perfect grid).
- **No bounding-box bottom-center as the board-contact point.** (We keep relitigating this — don't.) A tight box's bottom edge is the *front rim* of the base, not its center, and the error is not a fixable constant: it varies with piece height, board foreshortening (far ranks compress), and camera azimuth. Worse, in dense or low-angle views a nearer piece occludes the base, so the lowest *visible* pixel jumps up to the body and the derived point lands a whole square too far back. The homography→square step is exact **given a correct contact point** (self-check: 99.96% on ground-truth points), so accuracy is bottlenecked by the contact point — spend model capacity predicting it **directly** (a base keypoint), not deriving it from a box. Note: ChessReD gives the true square + corners per piece, so a *correct* contact-point training label is the square-center projected through the homography — independent of any box.

Generalization must come from *learned* localization + *deterministic* geometry, not tuned heuristics. Normalize (remove scale/perspective variance) before classifying. Remember pieces are tall and lean **away from the camera**, into the squares behind them — any square-based reasoning must account for this.

## Reproducibility (required from day one)

- Use `uv` for env/deps. Pin exact versions **with upper bounds** on volatile deps. Two specific landmines: **`numpy<2`** (NumPy 2.x breaks OpenCV) and **`torch.load(weights_only=...)`** (the default flipped) — always set it explicitly.
- Record dataset versions/hashes and train/val/test splits; keep a diverse **real held-out test set** for honest generalization numbers.
- Track experiments (CSV or W&B) so numbers are comparable across runs.
- Report both **per-square accuracy** and **whole-board (all-64-correct) accuracy**.

## Phase order

0 Scaffold → 1 Homography utility (unit-tested, manual corners) → 2 Piece detector baseline on natural images (get a real mAP) → 3 Board localizer (4-corner regression, replaces manual corners) → 4 Glue to FEN + eval → 5 Synthetic / domain randomization → 6 Harden & package. Each phase should produce a measured number before moving on.
