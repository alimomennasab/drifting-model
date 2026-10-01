""""
Experiment: does drift sharpen images in-between meanflow steps?
Idea: perform one meanflow step to halfway-denoised state, perform drift to guide noised
image towards image bank (noised the same way), perform one more meanflow step to fully
denoise.


export INCEPTION_WEIGHTS=/data/ali/weights/weights-inception-2015-12-05-6726825d.pth

    CUDA_VISIBLE_DEVICES=6 python -u drift_multistep.py \
    --data-root "/data/ali/imf_latents/train" \
    --drift-steps 2 \
    --train-batch "/data/ali/imf_latents/train_overfit1000_10classes.pt" \
    --checkpoint-path "/data/ali/imf_runs/overfit_dde_x_pred_lpips_ploss_muon_30000steps_1000samples10classes.pt" \
    --num-y 1000 \
    --num-y-pos 50 \
    --skip-first 100 \
    --mid-t 0.5 \
    > drift_multistep2.log 2>&1

"""


import argparse
import json
import os

import torch
import utils.torch_util as tu
from imf import iMeanFlow
from utils.vae_util import VAEWrapper
from utils.drift_util import compute_sharpener_drift, create_positive_bank
from utils.plot import plot_class_rows
from utils.fid import (
    get_inception_model,
    extract_inception_features,
    compute_fid_between_features,
)


def decode_latents(vae, latents, device, batch_size=8):
    """Decode latents w/ the VAE in small batches for better memory usage"""
    chunks = []
    with torch.no_grad():
        for start in range(0, len(latents), batch_size):
            batch = latents[start : start + batch_size].to(device)
            chunks.append(vae.decode(batch).float().cpu())
            torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0)


def mf_step_chunks(mf, z, t, r, omega, t_min, t_max, labels, batch_size, device):
    """Run mf_step over z in chunks"""
    chunks = []
    with torch.no_grad():
        for start in range(0, len(z), batch_size):
            end = min(start + batch_size, len(z))
            chunk = mf.mf_step(
                z[start:end].to(device),
                t=t,
                r=r,
                omega=omega,
                t_min=t_min,
                t_max=t_max,
                labels=labels[start:end].to(device),
            ).cpu()
            chunks.append(chunk)
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
    p.add_argument("--gen-batch-size", type=int, default=64)
    p.add_argument("--plot-max", type=int, default=8, help="Max images per row in class PNGs")
    p.add_argument("--skip-first", type=int, default=5, help="My train images are the first few images of the dataset, so skip these to ensure no leakage into the drift/fid bank")
    p.add_argument("--tau-start", type=float, default=5.0)
    p.add_argument("--tau-end", type=float, default=15.0)
    p.add_argument("--step-size", type=float, default=0.2)
    p.add_argument("--lambda-rep", type=float, default=0.1)
    p.add_argument("--sigma-r", type=float, default=1.5)
    p.add_argument("--cfg-omega", type=float, default=2.0, help="MeanFlow was trained with omega in [1, 8]")
    p.add_argument("--t-min", type=float, default=0.4, help="CFG interval start")
    p.add_argument("--t-max", type=float, default=0.65, help="CFG interval end")
    p.add_argument("--mid-t", type=float, default=0.5, help="Time to pause for drift (1=noise, 0=clean)")
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
    temperatures = torch.linspace(args.tau_start, args.tau_end, steps, device=device)
    step_size = args.step_size
    lambda_rep = args.lambda_rep
    sigma_r = args.sigma_r
    unique_labels = y_batch.unique(sorted=True)  # [10]
    cfg_omega = args.cfg_omega
    interval_min = args.t_min
    interval_max = args.t_max
    mid_t = args.mid_t
    n_samples = len(unique_labels) * k
    skip_first = args.skip_first
    gen_batch_size = args.gen_batch_size

    # create seeds and labels
    seeds = torch.arange(k).repeat(len(unique_labels)) # [0,1,2,3,4, 0,1,2,3,4, ...]
    gen_labels = unique_labels.repeat_interleave(k) # [c0,c0,c0,c0,c0, c1,c1,..., c10,...]

    # load positive image bank for drift computation
    y_pos_img_bank_dict = create_positive_bank(args.data_root, unique_labels.tolist(), skip_first)
    print(y_pos_img_bank_dict.keys())
    print(next(iter(y_pos_img_bank_dict.values())).shape)
    # only compute drift with the first k_pos images per class
    # the remaining images in the bank are used later for fid computation
    y_pos_for_drift = {key: latents[:k_pos] for key, latents in y_pos_img_bank_dict.items()}
    print(y_pos_for_drift.keys())
    print(next(iter(y_pos_for_drift.values())).shape)

    # -----------------------------------------------------------------------
    # FIRST GENERATION STEP: MEANFLOW FROM 1 (noise) TO mid_t
    # We also perform a one-step generation on that same noise for comparison
    # -----------------------------------------------------------------------
    print("PERFORMING FIRST GENERATION STEP")
    mid_chunks = []
    onestep_chunks = []
    with torch.no_grad():
        for start in range(0, n_samples, gen_batch_size):
            end = min(start + gen_batch_size, n_samples)
            rng = tu.BatchGenerator(device=device, seeds=seeds[start:end])
            z = rng.randn((end - start, 4, 32, 32)).to(torch.float32)
            labels_chunk = gen_labels[start:end].to(device)
            mid_chunks.append(
                mf.mf_step(
                    z,
                    t=1.0,
                    r=mid_t,
                    omega=cfg_omega,
                    t_min=interval_min,
                    t_max=interval_max,
                    labels=labels_chunk,
                ).cpu()
            )
            onestep_chunks.append(
                mf.mf_step(
                    z,
                    t=1.0,
                    r=0.0,
                    omega=cfg_omega,
                    t_min=interval_min,
                    t_max=interval_max,
                    labels=labels_chunk,
                ).cpu()
            )
            torch.cuda.empty_cache()
    mid_latents = torch.cat(mid_chunks, dim=0)
    onestep_latents = torch.cat(onestep_chunks, dim=0)
    print("mid latents shape: ", mid_latents.shape)
    print("1-step latents shape: ", onestep_latents.shape)

    # 2-step no-drift control uses the same mid-state
    generated_latents = mid_latents.clone()
    sharpened_latents = mid_latents.clone()

    # -----------------------------------------------------------------------
    # DRIFT AT mid_t TOWARD A TEMPORARILY NOISED BANK
    # -----------------------------------------------------------------------
    print(f"PERFORMING {steps} DRIFT STEPS AT t={mid_t}")
    print(f"TEMPERATURES: {temperatures}")
    with torch.no_grad():
        for class_id in unique_labels:
            key = f"class_{class_id:04d}"
            mask = (gen_labels == class_id)

            y = sharpened_latents[mask].to(device).flatten(1)
            y_pos = y_pos_for_drift[key].to(device).flatten(1)
            y_pos_noised = (1 - mid_t) * y_pos + mid_t * torch.randn_like(y_pos)

            for step, tau in enumerate(temperatures, start=1):
                drift = compute_sharpener_drift(
                    y,
                    y_pos_noised,
                    tau,
                    lambda_rep=lambda_rep,
                    sigma_r=sigma_r,
                )
                y = y + step_size * drift
                print(
                    f"{key} sharpener step {step:>2d}/{steps}: "
                    f"tau={tau.item():.3f} "
                    f"mean_drift_norm={drift.norm(dim=1).mean().item():.4f}"
                )

            sharpened_latents[mask] = y.reshape(sharpened_latents[mask].shape).cpu()

    # -----------------------------------------------------------------------
    # SECOND GENERATION STEP: MEANFLOW FROM mid_t TO 0 (clean)
    # -----------------------------------------------------------------------
    print("PERFORMING SECOND GENERATION STEP")

    # control: second step on non-drifted latents
    generated_latents = mf_step_chunks(
        mf, generated_latents, mid_t, 0.0,
        cfg_omega, interval_min, interval_max, gen_labels, gen_batch_size, device,
    )

    # second step on drifted-latents
    sharpened_latents = mf_step_chunks(
        mf, sharpened_latents, mid_t, 0.0,
        cfg_omega, interval_min, interval_max, gen_labels, gen_batch_size, device,
    )
    print("generated latents shape: ", generated_latents.shape)
    print("sharpened latents shape: ", sharpened_latents.shape)

    # Free backbone memory
    del mf
    torch.cuda.empty_cache()

    print("Decoding one-step images")
    onestep_images = decode_latents(
        vae, onestep_latents, device, batch_size=args.decode_batch_size
    )

    print("Decoding two-step images")
    generated_images = decode_latents(
        vae, generated_latents, device, batch_size=args.decode_batch_size
    )

    print("Decoding drift-sharpend two-step images")
    sharpened_images = decode_latents(
        vae, sharpened_latents, device, batch_size=args.decode_batch_size
    )
    del onestep_latents, generated_latents, sharpened_latents
    torch.cuda.empty_cache()

    run_dir = os.path.join(
        args.out_dir,
        f"gmd_mid_{len(x_batch)}samples_{y_batch.unique().numel()}classes_{k}gens_{steps}steps_tau{args.tau_start:g}-{args.tau_end:g}"
        f"_omega{cfg_omega:g}_midt{mid_t:g}",
    )
    os.makedirs(run_dir, exist_ok=True)
    print(f"Writing class plots to {run_dir}")

    plot_n = min(args.plot_max, k)
    for class_id in unique_labels.tolist():
        key = f"class_{class_id:04d}"
        mask = gen_labels == class_id

        mid_row = decode_latents(
            vae,
            mid_latents[mask][:plot_n],
            device,
            batch_size=args.decode_batch_size,
        )
        plot_path = plot_class_rows(
            [
                ("1-step generated", onestep_images[mask][:plot_n]),
                (f"After first step (t={mid_t:g})", mid_row),
                ("2-step sharpened", sharpened_images[mask][:plot_n]),
                ("2-step unsharpened", generated_images[mask][:plot_n]),
            ],
            os.path.join(run_dir, f"{key}.png"),
            class_id=class_id,
            subtitle=f"{steps} drift steps | t_mid={mid_t:g}",
        )
        print(f"Saved {plot_path}")

    results = {"args": vars(args)}
    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # compute FID for 2-step gens & mid-run-sharpened vs leftover bank
    if args.fid:
        fid_latents = torch.cat(
            [y_pos_img_bank_dict[f"class_{c:04d}"][k_pos:] for c in unique_labels.tolist()],
            dim=0,
        )
        if fid_latents.shape[0] == 0:
            raise ValueError("No leftover bank images for FID; need bank size > k_pos per class")

        real_images = decode_latents(
            vae, fid_latents, device, batch_size=args.decode_batch_size
        )

        inception_model = get_inception_model(device)
        # normalize & clamp: fid expects [0, 1] values
        real = ((real_images + 1) / 2).clamp(0, 1)
        onestep = ((onestep_images + 1) / 2).clamp(0, 1)
        gen = ((generated_images + 1) / 2).clamp(0, 1)
        sharp = ((sharpened_images + 1) / 2).clamp(0, 1)

        feat_real, _ = extract_inception_features(real, inception_model, batch_size=16)
        feat_onestep, _ = extract_inception_features(onestep, inception_model, batch_size=16)
        feat_gen, _ = extract_inception_features(gen, inception_model, batch_size=16)
        feat_sharp, _ = extract_inception_features(sharp, inception_model, batch_size=16)

        fid_onestep = compute_fid_between_features(feat_onestep, feat_real)
        fid_gen = compute_fid_between_features(feat_gen, feat_real)
        fid_sharp = compute_fid_between_features(feat_sharp, feat_real)
        print(f"FID 1-STEP GENERATED: {fid_onestep:.4f}")
        print(f"FID 2-STEP UNSHARPENED: {fid_gen:.4f}")
        print(f"FID 2-STEP SHARPENED: {fid_sharp:.4f}")
        results["fid"] = {
            "onestep": fid_onestep,
            "generated": fid_gen,
            "sharpened": fid_sharp,
        }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {results_path}")


if __name__ == "__main__":
    main()
