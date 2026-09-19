""""
    CUDA_VISIBLE_DEVICES=7 python drift.py \
    --drift-steps 10 \
    --dataset-dir "/data/ali/imf_latents/train_overfit30_10classes.pt" \
    --checkpoint-path "/data/ali/imf_runs/overfit_dde_x_pred_lpips_ploss_muon_20000steps_30samples10classes.pt" \
    --pos-img-bank "/data/ali/imf_latents/positive_bank_30samples_10classes.pt"

"""

import argparse
import os

import torch
import utils.torch_util as tu
from imf import iMeanFlow
from utils.vae_util import VAEWrapper
from utils.drift_util import compute_sharpener_drift
from utils.plot import plot_class_comparison


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drift-steps", default=10)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--out-dir", default="/data/ali/gmd_gens/")
    p.add_argument("--pos-img-bank", required=True, help="All images available for positive image bank creation, stored in latent space as shards")
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
    ds = torch.load(args.dataset_dir, map_location="cpu")
    print(ds.keys())
    x_batch = ds["x_batch"] # images
    y_batch = ds["y_batch"] # labels
    print(x_batch.shape)
    print(y_batch.shape)


    # config
    k = 5  # gens per class
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

    # create seeds and labels
    seeds = torch.arange(k).repeat(len(unique_labels)) # [0,1,2,3,4, 0,1,2,3,4, ...]
    gen_labels = unique_labels.repeat_interleave(k) # [c0,c0,c0,c0,c0, c1,c1,..., c10,...]

    # load positive image bank for drift computation
    pos_img_bank_dict = torch.load(args.pos_img_bank, map_location="cpu")
    print(pos_img_bank_dict.keys())
    print(pos_img_bank_dict['class_0000'].shape)
    # keep only the k first images per class
    pos_img_bank_dict = {key: latents[:k] for key, latents in pos_img_bank_dict.items()}
    print(pos_img_bank_dict.keys())
    print(pos_img_bank_dict['class_0000'].shape)

    # decode latents to image for feature extraction
    with torch.no_grad():
        generated_latents = mf.generate(
            n_sample=n_samples,
            rng=tu.BatchGenerator(device=device, seeds=seeds),
            num_steps=1,
            omega=cfg_omega,
            t_min=interval_min,
            t_max=interval_max,
            labels=gen_labels.to(device),
        ).to(device)
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
            y_pos = pos_img_bank_dict[key].to(device).flatten(1) # [k_pos, D] held-out positive samples w/ current label

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

        generated_images = vae.decode(generated_latents).float()
        sharpened_images = vae.decode(sharpened_latents).float()

    run_dir = os.path.join(
        args.out_dir,
        f"gmd_gens_{len(x_batch)}samples_{y_batch.unique().numel()}classes_{steps}steps",
    )
    os.makedirs(run_dir, exist_ok=True)
    print(f"Writing class plots to {run_dir}")

    with torch.no_grad():
        for class_id in unique_labels.tolist():
            key = f"class_{class_id:04d}"
            mask = gen_labels == class_id

            positive_images = vae.decode(
                pos_img_bank_dict[key].to(device)
            ).float()
            plot_path = plot_class_comparison(
                steps,
                positive_images,
                generated_images[mask],
                sharpened_images[mask],
                os.path.join(run_dir, f"{key}.png"),
                class_id=class_id,
            )
            print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
