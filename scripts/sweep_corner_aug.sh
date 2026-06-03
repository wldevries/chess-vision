#!/usr/bin/env bash
# Ablation sweep: which augmentation family helps the corner localizer?
# Four variants on top of the established baseline (hflip 0.5 + jitter 0.1),
# fixed seed, each to its own out-dir. Compare cap_mean / cap_worst across them.
set -euo pipefail

DATA_ROOT="data/othersets/ChessReD"
COMMON=(--data-root "$DATA_ROOT" --epochs 40 --batch-size 16 --device cuda --amp --seed 0)
OFF_COLOR=(--hue 0 --saturation 0 --grayscale-prob 0)
OFF_GEOM=(--rotate 0 --scale 0 --perspective 0)
ON_COLOR=(--hue 0.05 --saturation 0.3 --grayscale-prob 0.2)
ON_GEOM=(--rotate 5 --scale 0.1 --perspective 0.04)

run() {  # name, extra-flags...
  local name="$1"; shift
  echo "=== variant: $name ==="
  uv run python scripts/train_corner_regressor.py "${COMMON[@]}" \
    --out-dir "runs/corners_sweep/$name" "$@"
}

run baseline "${OFF_COLOR[@]}" "${OFF_GEOM[@]}"
run color    "${ON_COLOR[@]}"  "${OFF_GEOM[@]}"
run geom     "${OFF_COLOR[@]}" "${ON_GEOM[@]}"
run all       "${ON_COLOR[@]}"  "${ON_GEOM[@]}"

echo "=== sweep done ==="
