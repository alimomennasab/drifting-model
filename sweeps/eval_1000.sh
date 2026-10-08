#!/usr/bin/env bash
export INCEPTION_WEIGHTS=/data/ali/weights/weights-inception-2015-12-05-6726825d.pth
OUT_DIR=/data/ali/gmd_gens/backbone1000
GPU=${GPU:-6}
mkdir -p "${OUT_DIR}"

echo "=== 1000img/10class backbone, omega=2, 1-step, tau=10 ==="
CUDA_VISIBLE_DEVICES=${GPU} python -u drift.py \
  --data-root "/data/ali/imf_latents/train" \
  --drift-steps 10 \
  --tau-start 10 \
  --tau-end 10 \
  --cfg-omega 2 \
  --gen-steps 1 \
  --train-batch "/data/ali/imf_latents/train_overfit1000_10classes.pt" \
  --checkpoint-path "/data/ali/imf_runs/overfit_dde_x_pred_lpips_ploss_muon_30000steps_1000samples10classes.pt" \
  --num-y 1000 \
  --num-y-pos 50 \
  --skip-first 100 \
  --gen-batch-size 16 \
  --decode-batch-size 4 \
  --out-dir "${OUT_DIR}" \
  --fid
