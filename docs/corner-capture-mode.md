# Corner-capture mode

A lightweight flow for collecting **board-corner training data** for the Phase-3 corner
localizer: dump phone photos in an inbox, mark the four corners in the web app, and the
labels feed straight into corner training. It is deliberately separate from the piece
(FEN) capture loop. See `CLAUDE.md` for where this sits in the wider pipeline.

## Why it exists — corners need *viewpoints*, not positions

The corner localizer is a global 4-corner regressor, and its binding constraint is the
number of **distinct camera viewpoints** it has seen — not architecture, augmentation, or
normalization. Within a single capture session the camera and board are fixed, so hundreds
of frames collapse to a handful of distinct geometries; multi-seed augmentation and
normalization sweeps both came back inside the noise of a held-out eval starved of
viewpoints.

The key insight: the corner detector **does not care what is on the board or whether the
position is legal** — it only consumes the four corner marks. So corner data can drop the
entire FEN / legality / moves workflow that makes piece capture expensive:

> Scatter pieces arbitrarily → sweep the camera through many poses → label corners only
> (a "Predict" prefill does most of the work) → done. No FEN, no moves.

This decouples cheap corner-pose collection from expensive piece-label collection. The
piece detector still needs positions with a known FEN; the corner detector just needs
diverse angles, heights, rolls, distances, and board rotations.

## What the data needs (priority order)

1. **Viewpoint diversity — the #1 lever.** Vary camera height, yaw, **roll** (in-plane
   tilt — the deployment variation), distance, and board rotation on the table. Aim for
   many distinct poses per board, only 1–3 frames per pose (extra frames of a *fixed* pose
   add nothing — corners only care about the viewpoint).
2. **Board appearance** — shoot the physical boards under varied lighting; add new boards
   when possible. This is also how an unseen board (e.g. `dennis`) becomes a trained one.
3. **Piece presence / occlusion — secondary, arbitrary content.** Inference always has
   pieces on the board, so don't shoot only-empty boards (distribution shift). But the
   arrangement can be anything. Worth deliberately staging a few **hard cases**: a tall
   piece on a corner square, or captured pieces lined up beside the board with their tops
   crossing the board-boundary line in the image — those silhouettes sit exactly where the
   detector localizes the edge.

Don't over-invest in pieces: they are second-order to pose.

## Workflow: shoot freely, label later

Shooting and labelling are **separate steps** — marking corners live while also moving the
camera every shot is awkward.

1. **Shoot** (phone, no app in hand). Scatter pieces (no legality), re-scatter a few times
   for occlusion variety, and sweep poses: each camera × a few heights × yaws × rolls ×
   near/far × a couple of board rotations. Tens of distinct poses per board in minutes.
2. **Stage** the photos in the inbox: drop them under `data/source/inbox/` (subfolders are
   fine, e.g. one per date). The inbox is local-only (`data/` is gitignored). HEIC/HEIF is
   not decoded — convert to JPEG first.
3. **Label** in the web app's corner-label mode (below).
4. **Sync + train** (below).

## Labelling in the app

Run the capture app with the unified store as `--corners-root` (default `data`); the
`--corner-ckpt` enables the "Predict" prefill:

```bash
uv run python -m chessvision.capture --corners-root data --corner-ckpt runs/corners/best.pt
```

Toggle **"Label corners"** in the header. The mode is a date-grouped photo browser:
clicking a photo opens the corner-marking modal — the perspective 9×9 grid overlay with
draggable handles, seeded by **Predict**. A **sticky Board dropdown** (`boards.json`,
remembered between photos) tags each frame. Save writes the label and auto-advances to the
next unlabelled photo.

**The grid overlay is the accuracy mechanism, not a convenience.** Placing the four
corners projects the full 9×9 lattice over the photo; you align the lattice to the board's
inner cell lines and drop each handle where the cell edges converge. In particular this
fixes the **staunton rim-offset** error: that board has an upstanding rim that hides the
outer playing-cell edge, so by eye the corner gets marked on the *rim* instead of the true
grid corner. With the grid projected, mark the **playing-grid corner, not the rim.**

EXIF orientation is normalized **on label**: the rotation is baked into the stored pixels
once, and the corners are recorded in that frame — so the browser overlay, the stored JPEG,
and the trainer all share one pixel frame. Publish-safe EXIF (capture time, lens-aware
device slug) is whitelisted into the label row; GPS and serial/owner tags are never read.

## Storage layout (flat, unified `data/`)

Corner labels live in the **same flat `data/` tree as the captures**, not a separate
`data/corners/` dataset (an earlier design that has since been merged). `CornerStore`
(`chessvision/data/corner_capture.py`) owns it:

```
data/
  source/
    inbox/<relpath>       raw phone dumps for labelling (local-only)
    <session>/<file>.jpg  capture originals (migrated in)
  store/<relpath>         EXIF-normalized JPEG, written when a photo is labelled
  labels.jsonl            one row per labelled photo (the trainable artifact)
```

The record id, `src`, and `image` are all the **same source-relative path** — an inbox
photo at `source/inbox/<x>.jpg` has id `inbox/<x>.jpg` and its normalized JPEG mirrors it
at `store/inbox/<x>.jpg`. So provenance is a pure identity. `labels.jsonl` is whole-file,
written atomically (temp file + rename); a corner row carries the four ordered corners
(TL/TR/BR/BL via `order_corners`) plus board/set/device/surface tags. Rows may *also* carry
piece keypoints + FEN when labelled with the in-app position tool — that turns a corner
photo into a keypoint-head training sample — but corner-capture itself is corner-only.

## Training and the split

Corner training picks these up automatically:

```bash
uv run python scripts/train_corner_regressor.py --corners-root data --no-captures
```

Confirm the log prints a `corner-ds: +N` line — pointing `--corners-root` at a stale path
silently trains ChessReD-only.

The held-out eval is derived **per board, by pose** (`select_corner_dataset_poses`): every
labelled frame is clustered into a distinct corner pose by geometry (`dedup_thr`), the
pose-cluster is the atomic split unit (so a recurring viewpoint lands wholly on one side),
and a deterministic share of each board's poses is held out as the `cds_*` eval and
checkpoint-selection metric. A final anti-leak pass drops any train pose within `dedup_thr`
of a held-out one. `test_boards` (e.g. `dennis`) are dropped from **both** train and
held-out so they stay an honest unseen-board probe. There is **no manual "held-out" tag** —
tagging a session's board is what makes it appear in eval.

## Sync (back up labels + images)

```bash
uv run python scripts/sync_captures.py up --local data/store --prefix corners
```

`labels.jsonl` and the normalized images travel; derived top-level files are excluded.
Retag past sessions with the **Edit sessions** modal in the app — the per-board split keys
on the `board` tag.

## Non-goals / constraints

- **Not a piece-data source.** Corner-only frames have no FEN; the piece/keypoint stack
  filters them out (`has_pieces` requires placed pieces).
- **Don't shoot only-empty boards** — distribution shift from real inference.
- **Don't retune `dedup_thr`** — within-pose jitter (≤~0.006) and distinct setups (≥~0.025)
  are cleanly separated; the fix for a noisy eval is more *viewpoints*, not a threshold.
- **Orientation stays a manual toggle** downstream. Which physical corner is a8 is not
  geometry-recoverable (all four rotations are valid), so corner-capture does not change the
  a8 ambiguity — see the live read-position mode.
