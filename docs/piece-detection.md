# Piece detection (image → pieces on squares)

The piece model is the part of the pipeline that reads **what is on the board**: for each
piece it predicts a **class** (one of 12: white/black × pawn/knight/bishop/rook/queen/king)
and its **board-contact point** — the base, where the piece meets the board. Downstream
(`chessvision/inference.py`) each contact point is mapped through the board homography to a
square, and the 64 squares become a FEN. The model runs on the **natural, un-warped photo**
(Approach A); warping is used only for square assignment, never on the piece pixels. See
`CLAUDE.md` for where this sits in the wider pipeline and `docs/corner-capture-mode.md` for
the corner localizer that supplies the homography.

## The core rule: predict the contact point, never derive it

A piece is a 3D object standing above the board plane; only its base touches the plane. The
square is decided by that base. The model predicts the base **directly as a keypoint** — it
is **never** read off a detection box's bottom-center. A tight box's bottom edge is the
*front rim* of the base, not its center, and the error varies with piece height, board
foreshortening, and camera azimuth; worse, in dense/low-angle views a nearer piece occludes
the base and the lowest visible pixel jumps up to the body, landing the derived point a
whole square too far back. The homography→square step is exact given a *correct* contact
point (geometry self-check 99.96%), so model capacity is spent predicting the base, not a
box bottom. This is settled — don't relitigate it (see `CLAUDE.md` anti-patterns).

## The contact-point label is geometric truth (no manual labelling)

ChessReD gives each piece's true square plus the four board corners, so the contact-point
**training label** is just the square's center projected through the homography
(`chessvision/data/contact.py`) — computed without any bounding box. That means contact
labels are **auto-generated geometric truth**: the keypoint head needs *no* manual point
labelling on ChessReD. It has been visually validated on the most-occluded boards (the point
lands at the base even when the base is hidden). On the capture store the contact points are
hand-tagged once in the app and reused.

`contact.py` also yields an **occlusion score** (a piece is base-occluded when its contact
point falls inside a nearer piece's box) for curating hard images.

## Models and training recipes

Several models share the same `(image, target)` shape (labels `1..12`, one keypoint
`(N,1,3)` per piece) so datasets concatenate with no remapping. The lineage:

### 1. Box-detector baseline — `chessvision/detector.py`
Faster R-CNN ResNet50-FPN v2, COCO-pretrained, head resized to 12 classes + background.
Trained on the official **chessred2k** split (`scripts/train_detector.py`,
`chessvision/data/detection.py`; weights `runs/detector/best.pt`). Best val **mAP 0.864,
mAP@50 0.999**. This is a *box* detector only — it has no contact point — so it is scored on
ChessReD only and serves as the trunk for the keypoint head. (A YOLO box baseline,
`scripts/train_yolo_detector.py`, ties it at ~30× smaller — see below.)

### 2. Keypoint R-CNN — `chessvision/keypoint_detector.py`
Grafts a standard torchvision 1-keypoint head onto the Faster R-CNN v2 trunk to predict the
single contact point per piece — the FEN-relevant model that actually runs on captures.
Several ways to train it, in order of how they turned out:

- **Head-only pretrain** (`scripts/train_keypoint_head.py`) — freeze the trunk from
  `runs/detector/best.pt`, train only the keypoint branch on ChessReD contact targets. The
  old sequential first step.
- **Full multi-domain pretrain** (`scripts/pretrain_keypoint.py`) — from COCO source, unlock
  everything, train on ChessReD + external synthetic sets. *Synthetic pretrain was measured
  to **hurt** every board (render bias) and was dropped.*
- **Capture fine-tune** (`scripts/finetune_keypoint_captures.py`) — domain step on the
  user's own boards. **Scope gotcha:** the deployed checkpoint that works used `--unfreeze
  backbone`; freezing the trunk and training only the heads degrades class accuracy to
  ~0.70. Always retrain with `--unfreeze backbone`.
- **Joint training — the winner** (`scripts/train_keypoint_joint.py`). One stage from COCO
  source on a balanced **~50/50 ChessReD + capture-store** mix (a `WeightedRandomSampler`
  rebalances each epoch; `--mix` is the lever that keeps the model from drifting back to the
  single ChessReD appearance domain). Beats pretrain→finetune on trained boards. Adding
  **appearance aug + board-crop** (`--aug-color`/`--aug-noise`/`--board-crop`) is the current
  **champion** (measured: staunton 0.973 / cheap 0.983 / rimless 0.965 localization; unseen
  dennis 0.916 ceiling). Board-crop = crop to the detected board so pieces get more pixels;
  it helps *with* a matching crop fine-tune but **hurts as inference-only** (scale shift).

### 3. YOLO-pose — `scripts/train_yolo_pose.py`, `chessvision/yolo_keypoint.py`
The browser-deployable counterpart: a single-stage Ultralytics YOLO-pose (box + contact
keypoint), trained on the **same mix** as the champion joint trainer. `yolo11n/s-pose`
**match** the 234 MB Keypoint R-CNN on trained boards at ~40× smaller **and beat it on
unseen dennis** (s-pose 0.962 vs 0.916 localization). `YoloKeypointDetector` wraps it behind
the exact torchvision `model([img]) -> [{scores,labels,keypoints}]` interface so it is scored
by the *same* metric path — no metric fork. This greenlights the all-JS web app
(`web/`, onnxruntime-web). Run YOLO under its own dependency group: `uv sync --group yolo`.

## Data sources

- **ChessReD (chessred2k)** — `chessvision/data/detection.py`. The only ChessReD images with
  piece boxes; contact targets are the projected square centers (`contact.py`). Official
  train/val/test split → mAP comparable to the literature, no game-level leakage. 3072² images
  are downscaled to `max_size`.
- **Capture store** — `chessvision/data/capture_detection.py`. The hand-labelled capture
  photos (four corners + one contact keypoint per piece, **no boxes**) reshaped to the same
  target. Boxes are **synthesized** as a vertical-cylinder RoI around the contact point
  (`geometry.project_piece_box`) — a region hint only, never a contact source. Off-board
  pieces (resting beside the board) are dropped. **Split by session/board, never random** —
  a session is one board/set/room/pose, so a random split leaks near-duplicate frames; unseen
  boards (e.g. `dennis`) are held out of both train and selection as an honest probe.
- **Position-label tool** — `chessvision/data/positions.py` + the in-app tool. Project a known
  FEN onto a corner photo and nudge the bases; trains the keypoint head on new boards with
  **no Label Studio**. This is the cheap way to add appearance/lighting/angle diversity.

## Evaluation and the current bottleneck

Eval is **per board** (`scripts/eval_per_board_captures.py`, `eval_yolo_pose_vs_keypoint.py`,
`eval_keypoint_head.py`) and reports both **per-piece** and **whole-board** numbers:

- **localization** — fraction of pieces whose contact point lands on the right square.
- **class_acc** — fraction of (correctly localized) pieces classified correctly.
- **board_exact** — all ~32 pieces correct on one board.

Measured finding (held-out capture sessions): **end-to-end == the GT-corner ceiling exactly**
on both boards, i.e. **the corner model is *not* the bottleneck** — per-piece **class
accuracy** is. staunton localization 0.962 / **class_acc 0.905**; cheap 0.973 / **0.809**.
`board_exact` is low (~14% / 0%) purely from compounding ~30 pieces × ~94% conditional, so
the lever for whole-board accuracy is class accuracy, not geometry.

Per-class confusion (`--confusion`): the gap is almost all **dark/black pieces** (missed more
*and* misclassified more) plus **tall-piece K/Q/B shape confusion**; white pawn/rook/knight
are ~solved. Counts are color-balanced, so it is an **appearance/legibility** problem, not
class imbalance → the cheap high-leverage fix is **lighting + angle diversity** on the dark
pieces (via the position-label tool), favouring diverse real captures over synthetic renders.

> Honest-eval caveat: the deployed model scores 0.81–0.91 on the capture sessions but drops
> to **0.48–0.72** on the diverse `data/` position set — that diverse set is the real
> generalization test, and adding those positions to training measurably lifts unseen boards.

## Known limitations

- **Base-occlusion (Keypoint R-CNN):** torchvision's keypoint loss only supervises keypoints
  *inside* the proposal box, so a piece whose true base sits below its visible box can't be
  learned/predicted by the R-CNN head (measured v1 limitation; YOLO-pose is not bound by it
  the same way).
- **Inference-only board-crop hurts** (scale shift) — only use it with a matching crop
  fine-tune.
- **Synthetic pretrain hurts** (render bias) — dropped; real capture diversity is the lever.

## Square assignment (the deterministic handoff)

`chessvision/inference.py` maps each predicted contact point through the homography to a
square and emits one FEN **per orientation** (R0..R270) in a single pass — the same detection
relabelled four ways. Which physical corner is a8 is **not** geometry-recoverable (all four
rotations are valid), so orientation stays a manual toggle in the UI. The square comes from
the predicted keypoint, never from a box bottom.
