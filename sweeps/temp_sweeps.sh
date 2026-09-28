#!/usr/bin/env bash
export INCEPTION_WEIGHTS=/data/ali/weights/weights-inception-2015-12-05-6726825d.pth
OUT_DIR=/data/ali/gmd_gens/temp_experiments
mkdir -p "${OUT_DIR}"

for taus in "5 10" "10 15" "5 5" "10 10" "15 15" "5 20"; do
  read -r tau_start tau_end <<< "${taus}"
  echo "=== tau=${tau_start}-${tau_end} ==="
    CUDA_VISIBLE_DEVICES=7 python -u drift.py \
    --data-root "/data/ali/imf_latents/train" \
    --drift-steps 10 \
    --tau-start "${tau_start}" \
    --tau-end "${tau_end}" \
    --train-batch "/data/ali/imf_latents/train_overfit30_5classes.pt" \
    --checkpoint-path "/data/ali/imf_runs/overfit_dde_x_pred_lpips_ploss_muon_20000steps_30samples5classes.pt" \
    --num-y 1000 \
    --num-y-pos 50 \
    --skip-first 11 \
    --out-dir "${OUT_DIR}" \
    --fid
done
