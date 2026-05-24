# ChessVision — Starter Plan

A from-scratch project to read a chess position from a photo and output FEN, designed to
**generalize across many boards, piece sets, and environments** — not a single sample setup.

> License note: this plan is standard technique + public datasets, written fresh. It does
> **not** derive from the prior (non-MIT) project's source code. Build clean.

---

## 1. Learnings carried over from the prior project

- **Keep the homography idea.** Once you know the board corners, projecting the grid into the
  image is exact and perspective-correct. Make it a small, well-tested utility.
- **Drop everything that assumes the environment.** No color thresholds (warm-board-on-cool-bg
  only worked on one sample), no Hough-line + intersection clustering (brittle, and blew up to
  3.25 TiB / infinite loops on cluttered photos), no hand-built lattice indexing (desyncs the
  moment points aren't a perfect grid). Generalization comes from *learned* localization plus
  *deterministic* geometry — not tuned heuristics.
- **Normalize before you classify.** The old detector ran on a heavily downscaled full photo and
  missed ~half the pieces. Removing scale/perspective variance is the biggest robustness win.
- **Pieces are tall and lean.** Tall pieces extend into the squares *behind* them (away from the
  camera). Any square-based reasoning must account for this (see §4).
- **The board-contact point must be predicted, not inferred from a box.** A bounding box's
  bottom-center is a perspective-biased, occlusion-fragile proxy for where a piece meets the board
  (full reasoning in §4). The homography is exact given a correct contact point, so put the model
  effort *there*. Don't relitigate this.
- **Reproducible env from day one.** Pin deps with upper bounds. This week alone, two breakages
  came from unpinned transitive deps: NumPy 2.x vs OpenCV, and `torch.load` flipping its
  `weights_only` default. Don't inherit that pain.

---

## 2. Geometry clarification: 9×9 points ⇒ 8×8 squares

Chess has **8×8 squares**, but they're bounded by **9 lines** in each direction. Those lines
cross at a **9×9 = 81-point lattice**. Homography works on *points*, so:

- Define the **81 corner points** of a canonical board (e.g. a 1.0×1.0 square, corners at
  `(i/8, j/8)` for `i, j in 0..8`).
- Compute the homography `H` from those canonical corners to the 4 detected image corners.
- Project the 81 points through `H`. The **64 squares** are the cells between adjacent points.

So "9×9 grid" = the corner lattice; the 64 playable squares are the gaps. Build and unit-test
this as a standalone utility before any ML.

---

## 3. Recommended pipeline (single image → FEN)

Three stages:

1. **Board localization** — a model predicts board geometry. Start with **4-corner regression**
   (simplest) → homography. Upgrade later to predicting the full 81-point heatmap if occlusion
   hurts corner accuracy.
2. **Piece localization + classification in the *original* image** — pieces look natural (see §4);
   the model predicts each piece's **class** and its **board-contact point** (the base).
3. **Square assignment via the contact point** — map each piece's predicted contact point to a
   square using the homography, then emit FEN. The contact point is **predicted directly, never
   read off a bounding box** (see §4).

---

## 4. Warping & pieces — the key design decision

**What warping does:** a homography flattens the **board plane** to a clean top-down square. But a
chess piece is a 3D object standing *above* that plane — only its base touches it. So after
warping, a piece's base lands on the right square, but its **body shears/smears toward the image
edges**, overlapping the squares behind it. This distortion is unavoidable when warping pieces.

This forces a choice, and the choice decides **what data you train on**:

### Approach A — Detect in the original photo, assign by contact point  ✅ recommended
- Run the model on the **natural, un-warped** image. Pieces look normal; the model learns real
  appearance.
- For each piece, **predict its board-contact point directly** (a base keypoint, where the piece
  meets the board). That point lies on the board plane, so the homography maps it to the correct
  square even though the piece's body sticks up.
- **Warping is used only for square assignment — never on piece images.**
- **Training implication:** training on pieces in normal perspective images is exactly correct.
  Your worry ("is it a problem to train on pieces not in a warped/top-down image?") only applies
  if you later feed warped images. In Approach A you never do, so it's a non-issue.

> ⚠️ **Do not derive the contact point from a bounding box's bottom-center.** (This keeps coming
> up — settle it here.) A tight box's bottom edge is the *front rim* of the base, not its center,
> and the offset is not a fixable constant: it changes with piece height, board foreshortening
> (far ranks compress), and camera azimuth. In dense or low-angle scenes a nearer piece occludes
> the base, so the lowest *visible* pixel jumps to the body and the point lands a whole square too
> far back. The homography→square math is exact **given a correct contact point** (the §1 self-check
> hit 99.96% on ground-truth points), so the contact point is the bottleneck — predict it directly.
> A box detector can still *find/classify* pieces, but a box is the wrong place to read the base
> from. **Good contact-point labels are free:** ChessReD gives each piece's true square + the board
> corners, so the contact-point target = that square's center projected through the homography,
> independent of any box — and **validated**: overlaying these points on the most-occluded boards,
> each lands at the piece's base even when the base is hidden behind a nearer piece (so no manual
> labelling is needed to train a base-keypoint head; a Label Studio review is optional QA only).
> (The first Faster R-CNN baseline is a box detector — best chessred2k val mAP ≈ 0.86, mAP@50 ≈ 0.999;
> its box→contact conversion is the expected weak link, and the planned successor is a base-keypoint
> head transplanted onto that trunk.)

### Approach B — Warp the board, classify per-square crops
- Warp to canonical top-down, slice into 64 squares, classify each crop into 13 classes
  (empty + 6 white + 6 black).
- Crops **must include vertical headroom** toward the camera-far direction to capture the leaning
  body of the piece on that square.
- **Training implication:** input is sheared top-down crops, so you must train on warped crops.
  Training on normal piece images here would be a distribution mismatch and would hurt accuracy.

**Recommendation:** start with **Approach A**. It matches your instinct (train on natural images)
and sidesteps the shear problem. The square assignment is robust *as long as the contact point is
predicted directly* — not read off a box bottom (see the warning above). Keep Approach B as a
comparison if localization-in-image struggles with dense/occluded positions.

---

## 5. Data strategy — where generalization actually comes from

The old model failed because it overfit one domain. Fix this at the data level.

### On chesscog's rendered dataset
Worth using as a **bootstrap, not as the final training set.**
- ✅ Good for: teaching piece/board shape, getting a working pipeline fast, validating code.
- ❌ Not enough alone: the renders are "too clean" — wooden table, gray void, no reflections or
  clutter. That is *precisely* the domain gap that breaks on real photos (the prior model's exact
  failure). Train only on that and you'll repeat the mistake.

### How to close the gap
1. **Bootstrap** on clean renders (chesscog or your own) to get shapes learned.
2. **Composite augmentation:** use the board mask to paste random real backgrounds behind the
   board; add lighting jitter, blur, noise, synthetic reflections/shadows. Cheap variety.
3. **Domain randomization (own renders):** render in Blender headless with randomized piece
   styles, board materials, lighting, camera angles, backgrounds, and distractor objects. Labels
   (corners + per-square contents) come for free. Variety — not per-scene realism — is what
   closes the sim-to-real gap.
4. **Fine-tune on real data** (e.g. **ChessReD**, a large real-photo chess recognition dataset)
   and keep a real, diverse **held-out test set** to measure honest generalization.

> Always check each dataset's license before training.

---

## 6. Phased steps

- **Phase 0 — Scaffold.** `uv init`; pin torch / torchvision / numpy with upper bounds; repo
  layout (`data/`, `models/`, `chessvision/`, `notebooks/`, `tests/`); a `predict.py` CLI stub.
- **Phase 1 — Homography utility.** Build + unit-test "4 corners → 81 points → 64 square polygons"
  and the "base-point → square" lookup, using manually clicked corners. No ML yet.
- **Phase 2 — Piece detector baseline.** Train/fine-tune an object detector on pieces in natural
  images (start from a public dataset). Get a real mAP / per-class number.
- **Phase 3 — Board localizer.** Train 4-corner regression; replace the manual corners.
- **Phase 4 — Glue → FEN + eval.** Assemble end-to-end. Report **per-square accuracy** and
  **whole-board (all-64-correct) accuracy** on a diverse held-out set.
- **Phase 5 — Synthetic + domain randomization.** Add your renderer / compositing; retrain;
  measure the generalization lift on real photos.
- **Phase 6 — Harden / package.** Confidence handling, side-to-move & castling heuristics,
  packaging, reproducible inference.

---

## 7. First three concrete actions

1. `uv init chessvision` + pin deps (with `numpy<2`-style upper bounds); commit the skeleton.
2. Write and unit-test the homography + base-point square-assignment utility against a couple of
   hand-clicked images.
3. Pull one public dataset and train a baseline **piece detector on natural images** — get a real
   accuracy number to anchor everything else.

---

## 8. Reproducibility checklist

- `pyproject.toml` with exact pins **and** upper bounds on volatile deps (numpy, torch, opencv).
- Record dataset versions/hashes and train/val/test splits.
- Be explicit about `torch.load(weights_only=...)` and model-format assumptions.
- Track experiments (even a simple CSV/W&B) so generalization numbers are comparable across runs.

---

## 9. Reference systems to study (verify licenses before reuse)

- **chesscog** (Wölflein & Arandjelović, *Determining Chess Game State From an Image*) — the
  canonical multi-stage approach + a rendered dataset.
- **ChessReD** — large real-photo chess recognition dataset; good for fine-tuning and honest eval.
- **LiveChess2FEN** — real-time oriented; useful for the detection→square mapping ideas.
- Roboflow chess-piece datasets — quick detector baselines.
