""""
    CUDA_VISIBLE_DEVICES=6 python drift.py
"""

import torch
import utils.torch_util as tu
from imf import iMeanFlow
from utils.vae_util import VAEWrapper
from utils.drift_util import compute_sharpener_drift

# todo
# - final display: 3 columns (original image, MF generation, GMD sharpened image)

# load data
overfit_dataset_path = "/data/ali/imf_latents/train_overfit10.pt" # 10 samples
ds = torch.load(overfit_dataset_path, map_location="cpu")
print(ds.keys())
x_batch = ds["x_batch"] # images
y_batch = ds["y_batch"] # labels
print(x_batch.shape)
print(y_batch.shape)
num_images = len(x_batch)

# load meanflow model alongside VAE decoder & feature extractor 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mf_chkpt_path = "/data/ali/imf_runs/overfit_dde_x_pred_lpips_ploss_muon_20000steps_10samples_model.pt"
mf_chkpt = torch.load(mf_chkpt_path)
mf = iMeanFlow(model_str="imfDiT_B_2", parametrization="xpred", derivative="dde")
mf.load_state_dict(mf_chkpt['model'], strict=False)
mf.to(device)
mf.eval()

vae = VAEWrapper(decode_batch_size=num_images)


# config
steps = 10
temperatures = torch.linspace(0.3, 0.08, steps, device=device)
step_size = 0.2
lambda_rep = 0.1
sigma_r = 1.5
sample_seed = 0 # repeating the seed gives every class exactly the same initial noise.
seeds = torch.full((num_images,), sample_seed, dtype=torch.long)
cfg_omega = 48.0
interval_min = 0.4
interval_max = 0.65


# decode latents to image for feature extraction
with torch.no_grad():
    generated_latents = mf.generate(
        n_sample=num_images,
        rng=tu.BatchGenerator(device=device, seeds=seeds),
        num_steps=1,
        omega=cfg_omega,
        t_min=interval_min,
        t_max=interval_max,
        labels=y_batch.to(device),
    ).to(device)


# inference-time GMD sharpening in MeanFlow's latent space
with torch.no_grad():
    latent_shape = generated_latents.shape
    y = generated_latents.flatten(1)
    y_pos = x_batch.to(device).flatten(1)

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

    sharpened_latents = y.reshape(latent_shape)
    generated_images = vae.decode(generated_latents).float()
    sharpened_images = vae.decode(sharpened_latents).float()

print(f"Real images: {tuple(y.shape)}")
print(f"Generated images: {tuple(generated_images.shape)}")
print(f"Sharpened images: {tuple(sharpened_images.shape)}")
    
