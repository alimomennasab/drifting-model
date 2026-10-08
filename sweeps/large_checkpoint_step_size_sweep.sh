#!/usr/bin/env bash
export INCEPTION_WEIGHTS=/data/ali/weights/weights-inception-2015-12-05-6726825d.pth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/.."
OUT_DIR=/data/ali/gmd_gens/large_checkpoint_step_size_sweeps
mkdir -p "${OUT_DIR}"

for steps in 1 5; do
  for ss in 0.05 0.1; do
    echo "=== drift-steps=${steps} step-size=${ss} ==="
    CUDA_VISIBLE_DEVICES=6 python -u drift_full.py \
      --drift-steps "${steps}" \
      --step-size "${ss}" \
      --num-y 50 \
      --num-y-pos 50 \
      --gen-batch-size 128 \
      --decode-batch-size 4 \
      --out-dir "${OUT_DIR}" \
      --fid \
      > "${OUT_DIR}/drift_steps${steps}_ss${ss}.log" 2>&1
  done
done
