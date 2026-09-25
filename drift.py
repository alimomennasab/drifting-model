""""
export INCEPTION_WEIGHTS=/data/ali/weights/weights-inception-2015-12-05-6726825d.pth

    CUDA_VISIBLE_DEVICES=6 python drift.py \
    --data-root "/data/ali/imf_latents/train" \
    --drift-steps 10 \
    --train-batch "/data/ali/imf_latents/train_overfit30_10classes.pt" \
    --checkpoint-path "/data/ali/imf_runs/overfit_dde_x_pred_lpips_ploss_muon_20000steps_30samples10classes.pt" \
    --num-y 1000 \
    --num-y-pos 50 \
    --fid

"""


import argparse
import os

import torch
import utils.torch_util as tu
from imf import iMeanFlow
from utils.vae_util import VAEWrapper
from utils.drift_util import compute_sharpener_drift, create_positive_bank
from utils.plot import plot_class_comparison
from utils.fid import (
    get_inception_model,
    extract_inception_features,
    compute_fid_between_features,
)


def decode_latents(vae, latents, batch_size=8):
    """Decode latents w/ the VAE in small batches for better memory usage"""
    chunks = []
    with torch.no_grad():
        for start in range(0, len(latents), batch_size):
            batch = latents[start : start + batch_size]
            chunks.append(vae.decode(batch).float().cpu())
            torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drift-steps", type=int, default=10)
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--train-batch", type=str, required=True)
    p.add_argument("--checkpoint-path", type=str, required=True)
    p.add_argument("--num-y", type=int, required=True, help="Amount of generations produced **PER CLASS**")
    p.add_argument("--num-y-pos", type=int, required=True, help="Amount of real images in positive image bank **PER CLASS**")
    p.add_argument("--out-dir", type=str, default="/data/ali/gmd_gens/")
    p.add_argument("--decode-batch-size", type=int, default=8)
    p.add_argument("--plot-max", type=int, default=8, help="Max images per row in class PNGs")
    p.add_argument("--fid", action='store_true')
    args = p.parse_args()

    # load meanflow model alongside VAE decoder & feature extractor 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mf_chkpt = torch.load(args.checkpoint_path)
    mf = iMeanFlow(model_str="imfDiT_B_2", parametrization="xpred", derivative="dde")
    mf.load_state_dict(mf_chkpt['model'], strict=False)
    mf.to(device)
    mf.eval()

    vae = VAEWrapper(decode_batch_size=16)

    # load data
    ds = torch.load(args.train_batch, map_location="cpu")
    print(ds.keys())
    x_batch = ds["x_batch"] # images
    y_batch = ds["y_batch"] # labels
    print(x_batch.shape)
    print(y_batch.shape)


    # config
    k = args.num_y  # gens per class
    k_pos = args.num_y_pos # reals per class
    steps = int(args.drift_steps)
    temperatures = torch.linspace(0.3, 0.08, steps, device=device)
    step_size = 0.2
    lambda_rep = 0.1
    sigma_r = 1.5
    unique_labels = y_batch.unique(sorted=True)  # [10]
    cfg_omega = 48.0
    interval_min = 0.4
    interval_max = 0.65
    n_samples = len(unique_labels) * k 
    skip_first = len(y_batch) // len(unique_labels)  # 30 // 10 = 3

    # create seeds and labels
    seeds = torch.arange(k).repeat(len(unique_labels)) # [0,1,2,3,4, 0,1,2,3,4, ...]
    gen_labels = unique_labels.repeat_interleave(k) # [c0,c0,c0,c0,c0, c1,c1,..., c10,...]

    # load positive image bank for drift computation
    y_pos_img_bank_dict = create_positive_bank(args.data_root, unique_labels.tolist(), skip_first)
    print(y_pos_img_bank_dict.keys())
    print(y_pos_img_bank_dict['class_0000'].shape)
    # only compute drift with the first k_pos images per class
    # the remaining images in the bank are used later for fid computation
    y_pos_for_drift = {key: latents[:k_pos] for key, latents in y_pos_img_bank_dict.items()}
    print(y_pos_for_drift.keys())
    print(y_pos_for_drift['class_0000'].shape)

    # generate images in chunks for better memory usage
    gen_batch_size = 64
    generated_chunks = []
    with torch.no_grad():
        for start in range(0, n_samples, gen_batch_size):
            end = min(start + gen_batch_size, n_samples)
            chunk = mf.generate(
                n_sample=end - start,
                rng=tu.BatchGenerator(device=device, seeds=seeds[start:end]),
                num_steps=1,
                omega=cfg_omega,
                t_min=interval_min,
                t_max=interval_max,
                labels=gen_labels[start:end].to(device),
            ).cpu()
            generated_chunks.append(chunk)
            torch.cuda.empty_cache()
        generated_latents = torch.cat(generated_chunks, dim=0).to(device)
        print("generated latents shape: ", generated_latents.shape)


    # inference-time GMD sharpening in MeanFlow's latent space
    sharpened_latents = generated_latents.clone()
    print(f"PERFORMING {steps} DRIFT STEPS")
    print(f"TEMPERATURES: {temperatures}")
    with torch.no_grad():
        # for current class: avoid computing drift with own samples -> create mask
        for class_id in unique_labels:
            key = f"class_{class_id:04d}"
            mask = (gen_labels == class_id)

            y = generated_latents[mask].flatten(1) # [k, D] generations w/ current label
            y_pos = y_pos_for_drift[key].to(device).flatten(1) # [k_pos, D] held-out positive samples w/ current label

            for step, tau in enumerate(temperatures, start=1):
                drift = compute_sharpener_drift(
                    y,
                    y_pos,
                    tau,
                    lambda_rep=lambda_rep,
                    sigma_r=sigma_r,
                )
                y = y + step_size * drift
                print(
                    f"Sharpener step {step:>2d}/{steps}: "
                    f"tau={tau.item():.3f} "
                    f"mean_drift_norm={drift.norm(dim=1).mean().item():.4f}"
                )

            sharpened_latents[mask] = y.reshape(generated_latents[mask].shape)

    # Free backbone memory
    del mf
    torch.cuda.empty_cache()

    generated_images = decode_latents(
        vae, generated_latents, batch_size=args.decode_batch_size
    )
    sharpened_images = decode_latents(
        vae, sharpened_latents, batch_size=args.decode_batch_size
    )
    del generated_latents, sharpened_latents
    torch.cuda.empty_cache()

    run_dir = os.path.join(
        args.out_dir,
        f"gmd_gens_{len(x_batch)}samples_{y_batch.unique().numel()}classes_{k}gens_{steps}steps",
    )
    os.makedirs(run_dir, exist_ok=True)
    print(f"Writing class plots to {run_dir}")

    plot_n = min(args.plot_max, k)
    for class_id in unique_labels.tolist():
        key = f"class_{class_id:04d}"
        mask = gen_labels == class_id

        positive_images = decode_latents(
            vae,
            y_pos_for_drift[key][:plot_n].to(device),
            batch_size=args.decode_batch_size,
        )
        plot_path = plot_class_comparison(
            steps,
            k_pos,
            positive_images[:plot_n],
            generated_images[mask][:plot_n],
            sharpened_images[mask][:plot_n],
            os.path.join(run_dir, f"{key}.png"),
            class_id=class_id,
        )
        print(f"Saved {plot_path}")

    # compute FID for gens & sharpened vs remaining unused positive bank samples
    if args.fid:
        fid_latents = torch.cat(
            [y_pos_img_bank_dict[f"class_{c:04d}"][k_pos:] for c in unique_labels.tolist()],
            dim=0,
        )
        if fid_latents.shape[0] == 0:
            raise ValueError("No leftover bank images for FID; need bank size > k_pos per class")

        real_images = decode_latents(
            vae, fid_latents.to(device), batch_size=args.decode_batch_size
        )

        inception_model = get_inception_model(device)
        # normalize & clamp: fid expects [0, 1] values
        real = ((real_images + 1) / 2).clamp(0, 1)
        gen = ((generated_images + 1) / 2).clamp(0, 1)
        sharp = ((sharpened_images + 1) / 2).clamp(0, 1)

        feat_real, _ = extract_inception_features(real, inception_model)
        feat_gen, _ = extract_inception_features(gen, inception_model)
        feat_sharp, _ = extract_inception_features(sharp, inception_model)

        fid_gen = compute_fid_between_features(feat_gen, feat_real)
        fid_sharp = compute_fid_between_features(feat_sharp, feat_real)
        print(f"FID GENERATED: {fid_gen:.4f}")
        print(f"FID SHARPENED: {fid_sharp:.4f}")


if __name__ == "__main__":
    main()
