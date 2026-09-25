#!/usr/bin/env bash
export INCEPTION_WEIGHTS=/data/ali/weights/weights-inception-2015-12-05-6726825d.pth

for steps in 2 5 10 15 20; do
  echo "=== drift-steps=${steps} ==="
    CUDA_VISIBLE_DEVICES=6 python drift.py \
    --data-root "/data/ali/imf_latents/train" \
    --drift-steps "${steps}" \
    --train-batch "/data/ali/imf_latents/train_overfit30_10classes.pt" \
    --checkpoint-path "/data/ali/imf_runs/overfit_dde_x_pred_lpips_ploss_muon_20000steps_30samples10classes.pt" \
    --num-y 1000 \
    --num-y-pos 50 \
    --fid
done
