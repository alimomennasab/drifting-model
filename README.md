# One-Step MeanFlow Generation with GMD Sharpening

This project combines a pretrained
MeanFlow (https://arxiv.org/pdf/2505.13447, https://arxiv.org/pdf/2601.22158) with an inference-time [Generative Model Drifting (GMD)](https://arxiv.org/pdf/2602.04770) sharpener.

The original hope was that MeanFlow would produce a slightly blurry image latent in one pass, and GMD would apply a few nonparametric updates toward a batch of real reference latents to sharpen it.

## MeanFlow model

The backbone is trained with `train_imf.py` (`imfDiT_B_2`, `xpred`, DDE). Instead of generating images in pixel space, we generate in latent space. 

The single-step generation flow is as follows:

```text
noise + class labels -> pretrained MeanFlow -> generated latents
```


The data used in this experiment is a small subset of ImageNet, encoded to latents and split per class into (1) MeanFlow training images, (2) a drift / positive bank of real references, and (3) leftover held-out reals set for FID computation. The three splits are disjoint so the bank and FID set never contain training images.

## GMD sharpener

Instead of training a GMD model, we used its drift objective as an inference-time update on frozen MeanFlow latents.

For a generated latent $y$ and real reference latents
$\mathcal{R}=\{R_j\}$ (the **positive bank**: held-out reals, typically 50/class), each step is:

**1. Attraction toward nearby real samples**

$$
a(y)=\sum_j \mathrm{softmax}_j\!\left(-\frac{\|y-R_j\|_2^2}{2\tau_k^2}\right)R_j
$$

**2. Repulsion between generated samples**

$$
\rho(y_i)=\frac{\sum_{j\ne i}w_{ij}(y_i-y_j)}{\sum_{j\ne i}w_{ij}},
\qquad
w_{ij}=\exp\!\left(-\frac{\|y_i-y_j\|_2^2}{2\sigma_r^2}\right),
\qquad \sigma_r=1.5
$$

**3. Combined drift V**

$$
V(y)=(a(y)-y)-\lambda_{\mathrm{rep}}\rho(y),
\qquad \lambda_{\mathrm{rep}}=0.1
$$

**4. Inference update**

$$
y\leftarrow y+\eta V(y),
\qquad \eta=0.2
$$

The sharpener operates directly in MeanFlow's latent space. Each latent is
flattened from `[B, 4, 32, 32]` to `[B, 4096]` while distances and updates are
computed, then reshaped before VAE decoding.

We also tried the same drift **mid-run**: MeanFlow `1 → t_mid`, pull the noisy latent toward a bank noised to the same `t`, then MeanFlow `t_mid → 0` (`drift_multistep.py`).

## Findings

The “sharpened” images were clearly just snapping onto photos from the positive bank, not cleaning up the MeanFlow sample. Higher τ only mixes several bank photos into a blur. Mid-run drift does the same thing at a noisier time, so MeanFlow finishes a different image of that class.

## Conclusion

Drifting is **not** an ideal inference-time sharpening method here: as implemented, it snaps generations onto images from the positive bank (or a blurry mix of them). That is retrieval, not a fix for MeanFlow mode-covering.

A tiny step (or a hard cap on how far the latent may move) can avoid a full snap, but then there is little evidence of toy-style sharpening either. The kernel only has retrieve-or-blur behaviour in this space.

## Next steps

- Put GMD in **training** (`L_MF + λ · drift`), so sampling stays ordinary MeanFlow and the model cannot copy a 50-image bank.
- Optional: feature-space (DINOv2) energy with a trust region, evaluated with Inception so the metric is not the same as the kernel.
- Do not spend more sweeps on unconstrained latent-bank attraction.

## Project layout

```text
drift.py                  Generate, then drift toward the bank (post-hoc)
drift_multistep.py        MeanFlow 1→t_mid, drift, t_mid→0
train_imf.py              Train the MeanFlow backbone
imf.py                    MeanFlow model, generate, mf_step
utils/drift_util.py       Bank construction, attraction / repulsion / V
utils/vae_util.py         VAE decoding
sweeps/                   FID / τ / CFG eval scripts
plan.md                   Longer experimental log
```

## Setup and usage

```bash
pip install -r requirements.txt
export INCEPTION_WEIGHTS=/data/ali/weights/weights-inception-2015-12-05-6726825d.pth
```

Train a backbone if needed:

```bash
python train_imf.py --derivative dde --experiment-name ... --batch-file ...
```

Post-hoc drift (keep train / bank / FID splits disjoint):

```bash
python drift.py \
  --data-root /path/to/latents \
  --train-batch /path/to/train_split.pt \
  --num-y 1000 --num-y-pos 50 \
  --fid
```
