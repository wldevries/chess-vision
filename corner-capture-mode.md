# Plan: a dedicated corner-capture mode

Status: **built (2026-05-27)** as a *standalone* dataset, revising the original proposal
(2026-05-25). Companion to `plan.md` (Phase 3 board localizer) and the capture app
(`chessvision/capture`). Motivated by a measured bottleneck, not a hunch.

## 0. What was built (supersedes §5–§7 below)

The labelling tool was built as a **separate corner dataset under `data/corners/`**, not
folded into the capture store / Label Studio (the §5–§6 approach). Decision: corner data
is corner-only and needs no FEN/piece/LS machinery, and routing it through the captures
bucket + `label-studio.json` would couple labelling to MinIO and the piece-label shape for
no benefit. So:

- **Dataset & store** — `chessvision/data/corner_capture.py`. Layout `data/corners/`:
  `inbox/` (raw phone dumps, subfolders OK, local-only — `/data/` is gitignored) and
  `store/` (`images/<id>.jpg` EXIF-normalized JPEGs + `labels.jsonl`, the trainable
  artifact). `CornerStore` lists the inbox (date-ordered, labelled-state), normalizes EXIF
  **on label** (one pixel frame for display + storage + training), and upserts label rows.
- **Web app — corner-label mode** (`--corners-root data/corners`). A third mode beside
  capture/read: a date-grouped photo browser; clicking a photo opens the **existing**
  corner modal (perspective 9×9 grid, draggable handles, "Predict" corner-assist) seeded
  from the normalized still; a **sticky Board dropdown** (boards.json, remembered) tags
  each frame; Save writes to the store and auto-advances to the next unlabelled photo.
  Endpoints: `/api/corners-label/{available,inbox,image,save}` in `chessvision/capture/app.py`.
- **Training** — `train_corner_regressor.py --corners-root data/corners` (on by default;
  `--no-corner-ds` to disable). `select_corner_dataset_poses` reuses the per-board,
  by-pose split (anti-leak). Its held-out poses are the new `cds_*` eval and the
  checkpoint-selection metric (preferred over `cap_*`).
- **Sync** — no new code: `uv run python scripts/sync_captures.py up --local data/corners/store --prefix corners` (labels.jsonl isn't a derived sidecar, so it travels).

**Follow-on (not built):** an offline position-capture flow (shoot per-ply, assign a game
from the PGN catalog, project pieces from the FEN) using a *separate* inbox, never
auto-promoted to `data/captures`. See the memory `decoupled-capture-from-inbox`.

---

## Original proposal (2026-05-25)


## 1. Problem

The corner localizer's "works on your boards" eval is starved of **distinct viewpoints**,
and that — not model architecture, augmentation, or normalization — is the binding
constraint. Measured on the current capture set:

| board | photos | distinct corner **poses** | held-out poses |
|---|---|---|---|
| staunton-56mm | 333 | 12 | ~3 |
| cheap-30mm | 30 | 8 | ~2 |
| rimless-45mm | 46 | 17 | ~3–4 |
| **total** | **409** | **37** | **~8 (16 frames)** |

409 photos collapse to 37 distinct camera geometries because within a session the camera
and board are fixed — many frames are the same shot repeated (one staunton pose has **108**
frames). The held-out eval is ~8 independent viewpoints, so its `cap_mean`/`cap_worst`
have a large standard error (~0.007 seed swing). Effects below ~0.01 are unmeasurable; two
multi-seed sweeps (augmentation, normalization) both came back inside the noise. See memory:
*capture-eval-too-noisy-to-tune*, *geom-aug-needs-varied-pose-testset*.

The dedup threshold is **not** the culprit: within-pose jitter is ≤0.006 and distinct
setups are ≥0.025 apart, so `dedup_thr=0.02` cleanly separates real moves from frame
jitter. The low pose count is genuine — too few distinct setups, especially staunton.

## 2. Key insight — corners need viewpoints, not positions

The piece detector needs **positions**: legal-ish, varied, with a known FEN for labels.
That is what makes capture a hassle (set up a puzzle, play the moves). The corner detector
needs **viewpoints** and *does not care* what is on the board or whether the position is
legal — it only consumes the 4 corner marks. So corner data can drop the entire
FEN/legality/moves workflow:

> Pick an arbitrary piece scatter → sweep the camera through many poses → label corners
> only (corner-assist pre-fills) → done. No FEN, no moves.

This decouples cheap corner-pose collection from expensive piece-label collection.

## 3. What the data needs (priority order)

1. **Viewpoint diversity — the #1 lever.** The corner net is a global regressor weak on
   pose. Vary: camera height (plyo box, ~3 heights × 2 cameras), yaw, **roll** (in-plane
   tilt — the deployment variation we cannot currently measure, see *geom-aug-needs-…*),
   distance (closer/further), and board rotation/azimuth on the table. Target **30–50
   distinct poses per board**, ~1–3 frames per pose (more frames per fixed pose add
   nothing — corners only care about viewpoint).
2. **Board appearance** — already covered by shooting the 3 physical boards under varied
   lighting; keep doing it, add boards when possible.
3. **Piece presence / occlusion — secondary realism, arbitrary content.** Inference always
   has pieces, so do *not* shoot only-empty boards (distribution shift). But the arrangement
   can be anything:
   - vary **density** across shots (sparse / full / a couple empty for range);
   - deliberately include a few **hard cases**: a tall piece sitting on a corner square,
     and captured pieces lined up beside the board with their tops crossing the
     board-boundary line in the image. These silhouettes cross exactly where the corner
     detector localizes the edge — the realistic failure mode worth hardening against.

Do not over-invest here: pieces are second-order to pose.

## 4. Capture protocol — shoot freely, label later (decoupled)

Shooting and labelling are **separate steps**. Marking corners live while also repositioning
the camera every shot is awkward; instead just *shoot*, then label at a desk (§5/§7).

**Shoot (phone, no app in hand):**
- Scatter pieces on the board (no legality). Re-scatter a few times for occlusion variety,
  and stage 1–2 "hard" edge/poking-out arrangements (§3).
- For each arrangement, sweep poses: each camera × each height × a few yaws × a few rolls ×
  near/far × a couple of board rotations. 1–3 frames per pose — tens of distinct poses per
  board in minutes, no FEN entry.
- **Drop the photos in a staging folder** (e.g. `data/corners/<board>/`) for processing.
  Any phone/camera works. Watch EXIF orientation — normalize rotation on import so the
  grid overlay and stored corners use the same pixel frame.

## 5. Corner-labelling tool (`chessvision/capture`)

Not a live-capture mode — an **import-and-label** flow over the staged photos (§4).
**Corners are marked in-app, not in Label Studio** — the live grid overlay is what makes the
labels accurate (see §7), so this tool owns the labelling.

- **Import from the staging folder.** Point the app at `data/corners/<board>/`; it lists the
  photos and steps through them. Normalize EXIF orientation on load.
- **Mark corners with the live 9×9 grid overlay.** Placing the 4 corners projects the lattice
  over the photo; the user adjusts handles until the grid lines align with the board cells,
  then saves. This visual feedback is the whole point of doing it here.
- **Corner-assist prefill.** The "Predict" button pre-fills the 4 handles per photo (runs the
  current regressor on the still image); the user nudges against the overlay. Keep the
  existing `order_corners` TL/TR/BR/BL sorting.
- **No FEN/position editor.** Corner-only; hide piece-position UI.
- **On save, write into the capture store** as a `kind: "corner"` session (§6) with the
  board tag — so all existing tooling sees it. The staging folder is just an inbox.

## 6. Pipeline integration & tagging

- **Corner training already picks these up for free.** `select_capture_corner_poses` keys on
  `sample.has_all_corners`, not on FEN, so corner-only frames flow straight into the corner
  train/held-out split and the per-board pose clustering.
- **The piece/keypoint stack must skip them.** Corner-only frames have no FEN/piece labels.
  - Verify (do not assume) that `finetune_keypoint_captures.py`, `eval_detector_on_captures.py`,
    `save_capture_positions.py`, and `CaptureDataset` consumers filter out FEN-less frames.
  - If they do not, gate on the `kind: "corner"` tag (or "frame has no FEN") to exclude them.
- **Staging vs store.** `data/corners/<board>/` is a raw phone-photo **inbox** — not loaded
  by `CaptureDataset` and not synced. The labelling tool (§5) consumes it and writes the
  labelled result into the capture store (`data/captures/<corner-session>/`); only the store
  is synced and trained on. Decide whether the inbox is gitignored (likely yes) or discarded
  after import.
- **Sync:** the labelled corner sessions travel through MinIO like any capture
  (`sync_captures.py up/down`); they are not derived files. No change to the derived-file
  exclusions.

## 7. Labelling — in-app, with the grid overlay

Corners only, marked **in the capture app** using the live 9×9 grid overlay (§5) — not the
S3/Label-Studio loop. The overlay is not a convenience, it is the accuracy mechanism:

- **It fixes the staunton rim-offset error.** The staunton board has an upstanding rim that
  obscures the outer playing-cell edge, so by eye the corner gets marked on the *rim* rather
  than the true grid corner (a known systematic mislabel — see memory
  *staunton-corner-labels-rim-offset*). With the grid projected, the user aligns the lattice
  to the inner cell lines and places the handle where the cell edges converge, ignoring the
  rim. Mark the **playing-grid corner, not the rim.**
- Label Studio stays available as *optional QA* only; it is not in the corner-capture loop.

## 8. Success criteria

- Distinct corner poses go from **37 → 100+** total; held-out (25% per board) reaches
  ~25–30 independent viewpoints.
- The held-out `cap_*` standard error shrinks enough that a multi-seed sweep can resolve
  sub-0.01 effects — i.e. normalization / colour-aug become *measurable* instead of noise.
- A **roll-varied** subset exists in held-out, reopening the geometric-aug question on an
  eval that can actually credit it.

## 9. Non-goals / risks

- Not a piece-data source — no FEN, do not feed the piece/keypoint models from corner-only
  frames (the skip in §6 is the safeguard).
- Do not shoot only-empty boards (distribution shift from real inference).
- Do not retune `dedup_thr` — it is already well-calibrated (§1); the fix is data.
- Keep orientation a manual toggle downstream as today — corner-only capture does not change
  the a8-corner ambiguity (it is not geometry-recoverable; see `plan.md` §4 / live mode).

## 10. Concrete next steps

1. Build the import-and-label tool in the capture app (§5): import from `data/corners/<board>/`,
   EXIF-normalize, grid-overlay marking + corner-assist prefill, save as a `kind: "corner"`
   session into the capture store.
2. Verify/medium-task the piece-pipeline skip for FEN-less frames (§6).
3. Shoot freely with the phone per the §4 protocol (roll + a few hard cases) and stage the
   photos in `data/corners/<board>/`; then label them with the tool.
4. `sync_captures.py up`; re-run the corner train/eval; confirm pose counts and the eval's
   reduced noise (the log now prints distinct-pose counts).
5. With a less-noisy eval, re-run the normalization and colour-aug sweeps — now decisive.
