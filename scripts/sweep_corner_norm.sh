#!/usr/bin/env bash
# Multi-seed sweep: does in-model ImageNet normalization help the corner localizer?
# {normalize, no-normalize} x {seed 0,1,2} = 6 runs, augmentation off (defaults).
# Multi-seed is the point: the ~16-pose held-out eval has a ~0.007 seed swing, so a
# single run per config cannot separate a small normalization effect from noise.
# Compare distributions, e.g.:
#   duckdb -c "SELECT regexp_extract(filename,'norm_sweep/([a-z]+)_',1) v,
#     min(cap_mean_corner_err) FROM read_json_auto('runs/corners_norm_sweep/*/history.jsonl',
#     filename=true) WHERE epoch=40 GROUP BY v"
set -euo pipefail

DATA_ROOT="data/Chess Recognition Dataset (ChessReD)_2_all"
COMMON=(--data-root "$DATA_ROOT" --epochs 30 --batch-size 16 --device cuda --amp)

for seed in 0 1 2; do
  for variant in norm plain; do
    flag=$([ "$variant" = norm ] && echo --normalize || echo --no-normalize)
    echo "=== variant: ${variant} seed ${seed} ==="
    uv run python scripts/train_corner_regressor.py "${COMMON[@]}" \
      --seed "$seed" "$flag" \
      --out-dir "runs/corners_norm_sweep/${variant}_s${seed}"
  done
done

echo "=== norm sweep done ==="
