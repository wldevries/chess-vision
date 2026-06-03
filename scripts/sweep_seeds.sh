#!/usr/bin/env bash
# Measure the corner-eval NOISE FLOOR: train the identical config across N seeds
# (only the seed varies), so the spread across runs is pure training stochasticity.
# Aggregate with scripts/aggregate_seeds.py to get per-metric mean +/- std = the
# error bar that tells us which differences are real.
set -euo pipefail

DATA_ROOT="data/othersets/ChessReD"
COMMON=(--data-root "$DATA_ROOT" --epochs 40 --batch-size 16 --device cuda --amp)
SEEDS=(0 1 2 3 4)
OUT="runs/corners_seeds"

for s in "${SEEDS[@]}"; do
  echo "=== seed $s ==="
  uv run python scripts/train_corner_regressor.py "${COMMON[@]}" \
    --seed "$s" --out-dir "$OUT/s$s"
done

echo "=== sweep done; aggregating ==="
uv run python scripts/aggregate_seeds.py --root "$OUT" --seeds "${SEEDS[@]}"
