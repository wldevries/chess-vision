#!/usr/bin/env bash
# Lattice (81-point) vs 4-corner, matched resolution, on the FROZEN snapshot. Trains the
# lattice at 384 and 512 and cell-evals each (confidence-weighted H-fit). Compare board-ok
# to the corner baselines in runs/corners_res/ (same snapshot, same seed, same held-out).
set -euo pipefail

DATA_ROOT="data/Chess Recognition Dataset (ChessReD)_2_all"
SNAP="data/corners_snap"
COMMON=(--data-root "$DATA_ROOT" --epochs 40 --batch-size 16 --device cuda --amp \
        --seed 0 --corners-root "$SNAP" --lattice)
OUT="runs/corners_lattice"

for size in 384 512; do
  echo "############ LATTICE image-size $size ############"
  uv run python scripts/train_corner_regressor.py "${COMMON[@]}" \
    --image-size "$size" --out-dir "$OUT/l$size"
  echo "==== CELL ASSIGNMENT lattice $size (confidence-weighted) ===="
  uv run python scripts/eval_cell_assignment.py \
    --ckpt "$OUT/l$size/best.pt" --corners-root "$SNAP" --lattice
  echo "==== CELL ASSIGNMENT lattice $size (UNweighted, --no-conf) ===="
  uv run python scripts/eval_cell_assignment.py \
    --ckpt "$OUT/l$size/best.pt" --corners-root "$SNAP" --lattice --no-conf
done

echo "############ lattice sweep done ############"
