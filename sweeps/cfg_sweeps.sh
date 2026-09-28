#!/usr/bin/env bash
export INCEPTION_WEIGHTS=/data/ali/weights/weights-inception-2015-12-05-6726825d.pth
OUT_DIR=/data/ali/gmd_gens/cfg_experiments
GPU=${GPU:-7}
mkdir -p "${OUT_DIR}"

for gen_steps in 1 2; do
  for omega in 1 2 4 8 48; do
    echo "=== omega=${omega} gen_steps=${gen_steps} ==="
      CUDA_VISIBLE_DEVICES=${GPU} python -u drift.py \
      --data-root "/data/ali/imf_latents/train" \
      --drift-steps 10 \
      --tau-start 10 \
      --tau-end 10 \
      --cfg-omega "${omega}" \
      --gen-steps "${gen_steps}" \
      --train-batch "/data/ali/imf_latents/train_overfit30_5classes.pt" \
      --checkpoint-path "/data/ali/imf_runs/overfit_dde_x_pred_lpips_ploss_muon_20000steps_30samples5classes.pt" \
      --num-y 1000 \
      --num-y-pos 50 \
      --skip-first 11 \
      --gen-batch-size 16 \
      --decode-batch-size 2 \
      --out-dir "${OUT_DIR}" \
      --fid
  done
done
