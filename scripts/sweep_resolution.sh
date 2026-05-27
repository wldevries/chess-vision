#!/usr/bin/env bash
# Resolution A/B for corner precision, measured by CELL assignment (board-ok), on a
# FROZEN snapshot so the only variable is --image-size. Same seed, same data, same
# held-out split. Compares 384 (control) vs 512 (treatment).
set -euo pipefail

DATA_ROOT="data/Chess Recognition Dataset (ChessReD)_2_all"
SNAP="data/corners_snap"
COMMON=(--data-root "$DATA_ROOT" --epochs 40 --batch-size 16 --device cuda --amp \
        --seed 0 --corners-root "$SNAP")
OUT="runs/corners_res"

for size in 384 512; do
  echo "############ image-size $size ############"
  uv run python scripts/train_corner_regressor.py "${COMMON[@]}" \
    --image-size "$size" --out-dir "$OUT/r$size"
  echo "==== CELL ASSIGNMENT (image-size $size) ===="
  uv run python scripts/eval_cell_assignment.py \
    --ckpt "$OUT/r$size/best.pt" --corners-root "$SNAP"
done

echo "############ resolution sweep done ############"
