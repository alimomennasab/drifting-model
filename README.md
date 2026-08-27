# One-Step MeanFlow Generation with GMD Sharpening

This project combines a pretrained
MeanFlow (https://arxiv.org/pdf/2505.13447, https://arxiv.org/pdf/2601.22158) with an inference-time [Generative Model Drifting (GMD)](https://arxiv.org/pdf/2602.04770) sharpener.

MeanFlow produces an image latent in one inference pass, and GMD applies several small nonparametric updates using a batch of real reference latents.

## MeanFlow model

We use a pretrained MeanFlow model as the generation backbone. The current script loads a trained
`imfDiT_B_2` checkpoint using the `xpred` parameterization and the DDE
derivative estimator. Instead of generating images in pixel space, we generate in latent space. The single-step generation flow is as follows:

```text
noise + class labels -> pretrained MeanFlow -> generated latents
```

The model checkpoint and dataset batch paths are configured directly in
`drift.py`. The dataset batch contains a small set of 10 paired latent images and class labels.

## GMD sharpener
Instead of training a GMD model, we utilize its drift objective as an inference-time image sharpener.

For a generated latent \(y\) and real reference latents
\(\mathcal{R}=\{R_j\}\), the sharpener performs the following operations.

**1. Attraction toward nearby real samples**

$$
a(y)=\sum_j \operatorname{softmax}_j\!\left(-\frac{\|y-R_j\|_2^2}{2\tau_k^2}\right)R_j
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

For each generated latent `y`, GMD computes:

- **Attraction:** a Gaussian-kernel weighted pull toward real
  latents.
- **Repulsion:** a Gaussian-kernel direction based on the other generated
  samples in the batch. Note: self-comparisons are excluded.
- **Drift:** the attraction update combined with a small weighted repulsion
  term.

The current configuration applies 10 gentle drift steps. The temperature is decreased from `0.3` to `0.08` across these 10 steps, with:

```text
step size = 0.2
repulsion weight = 0.1
repulsion bandwidth = 1.5
```

After sharpening, the real reference latents, original generated latents, and
sharpened latents are decoded with the VAE and plotted side by side.

## Inference-time flow

To reiterate, this implementation does **not** train MeanFlow, GMD, the VAE, or any other network. It loads a frozen VAE and frozen MeanFlow, and performs direct tensor updates on its generated latent samples with `torch.no_grad()`.


```text
one MeanFlow evaluation
        -> 10 kernel drift updates
        -> VAE decoding
        -> comparison plot
```


## Project layout

```text
drift.py                  End-to-end generation and sharpening script
imf.py                    MeanFlow model and sampling logic
models/                   MeanFlow neural-network components
utils/drift_util.py       Attraction, repulsion, and drift computations
utils/vae_util.py         VAE decoding utilities
utils/plot.py             Real/generated/sharpened comparison plots
```

## Setup and usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Update the checkpoint, reference-data, and output paths in `drift.py`, then
run:

```bash
python drift.py
```