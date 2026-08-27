"""
Improved MeanFlow in PyTorch with DDE (Differential Derivation Equation) approximation.

The DDE replaces JVP with a central finite-difference approximation of d f / d t under
stop-gradient parameters, following Wang et al. 2509.04394 (Transition Models).

  d f_theta-_,t,r / dt  ~=  ( f_theta-(x_{t+eps}, t+eps, r) - f_theta-(x_{t-eps}, t-eps, r) ) / (2 eps)

This is plugged into the iMF training objective in place of jax.jvp on the u-head.
"""

import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from models import imfDiT
from utils.vae_util import VAEWrapper
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
from modules.autoencoding_utils.lpips import LPIPS, VGG16FeatureExtractor


class iMeanFlow(nn.Module):
    """improved MeanFlow"""

    def __init__(
        self,
        model_str: str,
        dtype: torch.dtype = torch.float32,
        img_size: int = 32,
        img_channels: int = 4,
        num_classes: int = 1000,
        # Noise distribution
        P_mean: float = -0.4,
        P_std: float = 1.0,
        # Loss
        data_proportion: float = 0.5,
        cfg_beta: float = 1.0,
        class_dropout_prob: float = 0.1,
        # Training dynamics
        norm_p: float = 1.0,
        norm_eps: float = 0.01,
        # DDE
        dde_eps: float = 1e-3,
        # Derivative estimator for du/dt: "dde" (finite-difference) or
        # "jvp" (torch.func.jvp, exact forward-mode derivative; matches JAX iMF).
        derivative: str = "dde",
        # Parametrization for the u-field:
        #   "velocity" -> net output IS the (mean) velocity u   [iMF, bounded du/dt]
        #   "xpred"    -> net output is x_pred (clean sample); u = (z_t - x_pred)/
        #                 clip(t, t_clip_min, 1)  [UNITE/pMF; 1/t^2 du/dt blow-up].
        # Exp 1a isolates whether x-pred breaks meanflow on a SMOOTH (VAE) latent.
        parametrization: str = "velocity",
        t_clip_min: float = 0.05,
        # pMF does NOT normalize the x_pred output (it RMSNorms the pre-projection
        # hidden state inside the net, and zero-inits the head). An extra LayerNorm
        # on x_pred is a DEVIATION that destabilized training -> keep it opt-in/off.
        xpred_layernorm: bool = False,
        # Eval-only or training mode
        eval_mode: bool = False,
        # Add perceptual loss (LPIPS) to the current loss term loss_u + loss_v
        p_loss: bool = False,
        perceptual_metric: str = "lpips" # lpips, euclidean
    ):
        super().__init__()
        self.model_str = model_str
        self.dtype = dtype
        self.img_size = img_size
        self.img_channels = img_channels
        self.num_classes = num_classes

        self.P_mean = P_mean
        self.P_std = P_std
        self.data_proportion = data_proportion
        self.cfg_beta = cfg_beta
        self.class_dropout_prob = class_dropout_prob
        self.norm_p = norm_p
        self.norm_eps = norm_eps
        self.dde_eps = dde_eps
        assert derivative in {"dde", "jvp"}, derivative
        self.derivative = derivative
        assert parametrization in {"velocity", "xpred"}, parametrization
        self.parametrization = parametrization
        print("Parametrization: ", self.parametrization)
        self.t_clip_min = t_clip_min
        self.eval_mode = eval_mode

        self.p_loss = p_loss
        self.perceptual_metric = perceptual_metric
        if self.p_loss: 
            assert perceptual_metric in ("lpips", "eucl")
            print("using p_loss with metric ", self.perceptual_metric)

        # Faithful pMF port: real pMF applies `x_pred = encoder_layer_norm(net_out)`
        # — a per-token LayerNorm over the feature dim with learnable affine — so
        # the predicted clean sample is unit-scaled regardless of the (adaLN-zero)
        # net output. Our latent is (B, C, H, W); the analog is a LayerNorm over
        # the channel dim per spatial location. Only built for xpred.
        self.xpred_layernorm = xpred_layernorm
        if parametrization == "xpred" and xpred_layernorm:
            self.xpred_norm = nn.LayerNorm(img_channels, elementwise_affine=True)

        net_fn = getattr(imfDiT, self.model_str)
        self.net: imfDiT.imfDiT = net_fn(
            input_size=self.img_size,
            in_channels=self.img_channels,
            num_classes=self.num_classes,
            eval_mode=eval_mode,
        )

        # initialize VAE decoder & perceptual loss model if using perceptual loss
        if self.p_loss:
            print("USING PERCEPTUAL LOSS")
            self.vae = VAEWrapper(decode_batch_size=64)
            if self.perceptual_metric == "lpips":
                self.perceptual_net = LPIPS().eval()
            elif self.perceptual_metric == "eucl":
                self.perceptual_net = VGG16FeatureExtractor(pretrained=True, requires_grad=False)
        else:
            print("NOT USING PERCEPTUAL LOSS")

    # -----------------------------------------------------------------------
    # Schedules
    # -----------------------------------------------------------------------
    def logit_normal_dist(self, bz, device):
        rnd_normal = torch.randn(bz, 1, 1, 1, dtype=self.dtype, device=device)
        return torch.sigmoid(rnd_normal * self.P_std + self.P_mean)

    def sample_tr(self, bz, device):
        t = self.logit_normal_dist(bz, device)
        r = self.logit_normal_dist(bz, device)
        t, r = torch.maximum(t, r), torch.minimum(t, r)

        data_size = int(bz * self.data_proportion)
        fm_mask = (torch.arange(bz, device=device) < data_size).reshape(bz, 1, 1, 1)
        r = torch.where(fm_mask, t, r)
        return t, r, fm_mask

    def sample_cfg_scale(self, bz, device, s_max=7.0):
        u = torch.rand(bz, 1, 1, 1, dtype=torch.float32, device=device)
        if self.cfg_beta == 1.0:
            s = torch.exp(u * torch.log1p(torch.tensor(s_max, dtype=torch.float32, device=device)))
        else:
            smax = torch.tensor(s_max, dtype=torch.float32, device=device)
            b = torch.tensor(self.cfg_beta, dtype=torch.float32, device=device)
            log_base = (1.0 - b) * torch.log1p(smax)
            log_inner = torch.log1p(u * torch.expm1(log_base))
            s = torch.exp(log_inner / (1.0 - b))
        return s

    def sample_cfg_interval(self, bz, device, fm_mask=None):
        t_min = torch.rand(bz, 1, 1, 1, dtype=self.dtype, device=device) * 0.5
        t_max = 0.5 + torch.rand(bz, 1, 1, 1, dtype=self.dtype, device=device) * 0.5
        if fm_mask is not None:
            t_min = torch.where(fm_mask, torch.zeros_like(t_min), t_min)
            t_max = torch.where(fm_mask, torch.ones_like(t_max), t_max)
        return t_min, t_max

    # -----------------------------------------------------------------------
    # u/v functions
    # -----------------------------------------------------------------------
    def u_fn(self, x, t, h, omega, t_min, t_max, y): # return (u head, v head) after running network once
        bz = x.shape[0]
        out_u, out_v = self.net(
            x,
            t.reshape(bz),
            h.reshape(bz),
            omega.reshape(bz),
            t_min.reshape(bz),
            t_max.reshape(bz),
            y,
        )
        if self.parametrization == "xpred":
            # UNITE/pMF parametrization: the net predicts the CLEAN sample x_pred,
            # and the (mean) velocity is recovered as u = (z_t - x_pred)/clip(t).
            # Doing the /t INSIDE u_fn means the JVP / DDE du/dt differentiates
            # through it, capturing the 1/t^2 term that we suspect breaks meanflow.
            # z_t = (1-t) x + t e  =>  e - x = (z_t - x)/t, so x_pred ~ x recovers v.
            # FAITHFUL pMF: x_pred is the RAW (zero-init) head output; the ONLY
            # taming is clip(t,0.05,1) in the denominator + stop_grad(du_dt) in the
            # target + the adaptive loss weight. No output LayerNorm (that diverged).
            if self.xpred_layernorm:
                def _norm(o):
                    return self.xpred_norm(o.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                out_u, out_v = _norm(out_u), _norm(out_v)
            t_safe = t.reshape(bz, 1, 1, 1).clamp(self.t_clip_min, 1.0)
            u = (x - out_u) / t_safe
            v = (x - out_v) / t_safe
            return u, v
        return out_u, out_v

    def v_cond_fn(self, x, t, omega, y): # return instantaneous velocity v (only v head)
        h = torch.zeros_like(t)
        t_min = torch.zeros_like(t)
        t_max = torch.ones_like(t)
        v = self.u_fn(x, t, h, omega, t_min, t_max, y=y)[1]
        return v

    def v_fn(self, x, t, omega, y): 
        # combine the conditioned & unconditioned input into one tensor
        # run the model on the doubled tensor
        # split the double tensor into both the conditioned & unconditioned prediction
        bz = x.shape[0]
        x_d = torch.cat([x, x], dim=0) # duplicate all noisy inputs
        y_null = torch.full((bz,), self.num_classes, dtype=y.dtype, device=y.device)
        y_d = torch.cat([y, y_null], dim=0) # combine condition (y) with no condition (y_null)
        t_d = torch.cat([t, t], dim=0) # concat the inputs & conditions/no cond
        w_d = torch.cat([omega, torch.ones_like(omega)], dim=0)

        out = self.v_cond_fn(x_d, t_d, w_d, y_d)
        v_c, v_u = torch.chunk(out, 2, dim=0)
        return v_c, v_u

    def cond_drop(self, v_t, v_g, labels):
        bz = v_t.shape[0]
        rand_mask = torch.rand(bz, device=labels.device) < self.class_dropout_prob
        num_drop = int(rand_mask.sum().item())
        drop_mask = (torch.arange(bz, device=labels.device) < num_drop).view(bz, 1, 1, 1)

        labels = torch.where(
            drop_mask.view(bz),
            torch.full_like(labels, self.num_classes),
            labels,
        )
        v_g = torch.where(drop_mask, v_t, v_g)
        return labels, v_g

    def guidance_fn(self, v_t, z_t, t, r, y, fm_mask, w, t_min, t_max):
        # take both conditional & unconditional velocity predictions
        v_c, v_u = self.v_fn(z_t, t, w, y=y)
        v_g_fm = v_t + (1 - 1 / w) * (v_c - v_u) # cfg performed here

        w_int = torch.where((t >= t_min) & (t <= t_max), w, torch.ones_like(w))
        v_c_int = self.v_cond_fn(z_t, t, w_int, y=y)
        v_g = v_t + (1 - 1 / w_int) * (v_c_int - v_u)

        v_g = torch.where(fm_mask, v_g_fm, v_g)
        return v_g, v_c_int

    # -----------------------------------------------------------------------
    # Forward / loss
    # -----------------------------------------------------------------------
    def forward(self, images, labels):
        """images: (B, C, H, W); labels: (B,) int64.

        Important ordering note: the live (grad-tracked) ``u_fn`` call must
        run before any ``torch.no_grad`` forward pass through the same network
        when running under bf16/fp16 autocast.  Otherwise the autocast weight
        cache stores a grad-less bf16 copy of the weights and the gradient
        chain through the live call is broken.
        """
        device = images.device
        x = images.to(self.dtype)
        bz = x.shape[0]

        # sample t, r
        t, r, fm_mask = self.sample_tr(bz, device)

        # interpolant + instantaneous velocity v_t = e - x
        e = torch.randn_like(x)
        z_t = (1 - t) * x + t * e
        v_t = e - x

        # CFG schedule
        t_min, t_max = self.sample_cfg_interval(bz, device, fm_mask)
        omega = self.sample_cfg_scale(bz, device) # omega: 

        # Class dropout: precompute drop mask (independent of any forward).
        rand_mask = torch.rand(bz, device=device) < self.class_dropout_prob # drop % of condition labels for CFG
        num_drop = int(rand_mask.sum().item())
        drop_mask = (torch.arange(bz, device=device) < num_drop).view(bz, 1, 1, 1)
        labels_dropped = torch.where(
            drop_mask.view(bz),
            torch.full_like(labels, self.num_classes),
            labels,
        )

        # ---- Live u (with gradients) computed FIRST so autocast caches the
        # grad-tracked bf16 weights. ----
        u_live, v_live = self.u_fn(
            z_t, t, t - r, omega, t_min, t_max, y=labels_dropped
        )

        # ---- Build the v_g target and v_c (no grad). ----
        with torch.no_grad():
            v_g, v_c = self.guidance_fn(
                v_t, z_t, t, r, labels, fm_mask, omega, t_min, t_max
            ) 
            # v_c: instantaneous velocity predicted from model's v head
            # v_g: 
            v_c = v_c.detach()
            # Apply class-dropout flow-matching override: when label was
            # dropped, target v_g should fall back to v_t.
            v_g = torch.where(drop_mask, v_t, v_g).detach()

        # ---- du/dt estimator: JVP or DDE -----------------------------------
        if self.derivative == "jvp":
            # Forward-mode AD via the classical fwAD API. Together with the
            # `sdpa_jvp_patch` (Triton flash-attention+JVP kernel), this lets
            # us run JVP at flash-attention speed instead of falling back to
            # the math SDPA backend.
            import torch.autograd.forward_ad as fwAD
            with torch.no_grad(), fwAD.dual_level():
                dual_z = fwAD.make_dual(z_t, v_c)
                dual_t = fwAD.make_dual(t, torch.ones_like(t))
                dual_r = fwAD.make_dual(r, torch.zeros_like(r))
                u_dual, _ = self.u_fn(
                    dual_z, dual_t, dual_t - dual_r,
                    omega, t_min, t_max, y=labels_dropped,
                )
                _, du_dt_t = fwAD.unpack_dual(u_dual)
                du_dt = (du_dt_t if du_dt_t is not None
                         else torch.zeros_like(z_t)).detach()
        else:
            # DDE central-difference approximation along v_c trajectory.
            eps = self.dde_eps
            with torch.no_grad():
                t_plus = (t + eps).clamp(max=1.0)
                t_minus = (t - eps).clamp(min=0.0)
                z_plus = z_t + eps * v_c
                z_minus = z_t - eps * v_c
                h_plus = t_plus - r
                h_minus = t_minus - r
                u_plus, _ = self.u_fn(z_plus, t_plus, h_plus, omega, t_min, t_max,
                                      y=labels_dropped)
                u_minus, _ = self.u_fn(z_minus, t_minus, h_minus, omega, t_min, t_max,
                                       y=labels_dropped)
                du_dt = ((u_plus.float() - u_minus.float()) / (2.0 * eps)).detach()

        # Compound objective V = u + (t-r) * du/dt
        V = u_live + (t - r) * du_dt

        # adaptive weighting
        def adp_wt(loss):
            adp = (loss + self.norm_eps) ** self.norm_p
            return loss / adp.detach()

        loss_u = ((V - v_g) ** 2).flatten(1).sum(dim=1)
        loss_u_w = adp_wt(loss_u)
        loss_v = ((v_live - v_g) ** 2).flatten(1).sum(dim=1)
        loss_v_w = adp_wt(loss_v)
        loss_perceptual = 0
        if self.p_loss:
            t_safe = t.clamp(self.t_clip_min, 1.0)
            # u = (z_t - x_pred)/clip(t)

            # u = (x - out_u) / t_safe
            # out_u = x - t_safe * u
            # out_u is x_pred, and x is a latent z_t here, so
            # x_pred = z_t - t_safe * u

            x_pred = z_t - t_safe * u_live
            decoded_x_pred = self.vae.decode(x_pred)
            decoded_x = self.vae.decode(x)

            if self.perceptual_metric == "lpips":
                loss_perceptual = self.perceptual_net(decoded_x_pred, decoded_x)
            elif self.perceptual_metric == "eucl":
                decoded_x_pred_feats = self.perceptual_net(decoded_x_pred)
                decoded_x_feats = self.perceptual_net(decoded_x)
                # each VGG output is a set of 5 tensors, each of size torch.Size([10, 64, 256, 256])
                # each tensor is the output from a VGG layer. Earlier layers -> coarse details, later layers -> fine details
                # we take the mean of the perceptual loss of each layer's outputs (of x & x_pred)
                layer_losses = [
                    ((d_x.float() - d_x_pred.float()).pow(2)
                    .flatten(1)
                    .mean(dim=1)).sqrt()
                    for d_x, d_x_pred in zip(decoded_x_feats, decoded_x_pred_feats)
                ]
                # each layer loss value is of shape [B]
                loss_perceptual = torch.stack(layer_losses).mean(dim=0)

        loss = (loss_u_w + loss_v_w + loss_perceptual).mean() # avg across each sample in batch

        with torch.no_grad():
            loss_u_raw = ((V - v_g) ** 2).mean()
            loss_v_raw = ((v_live - v_g) ** 2).mean()
            # Health diagnostics — for xpred we expect du_dt / corr_over_u to
            # blow up at small t (the 1/t^2 term); for velocity they stay bounded.
            u_norm = u_live.detach().float().pow(2).mean().sqrt()
            dudt_norm = du_dt.detach().float().pow(2).mean().sqrt()
            corr_norm = ((t - r) * du_dt).detach().float().pow(2).mean().sqrt()
            corr_over_u = corr_norm / (u_norm + 1e-6)

        return loss, {
            "loss": loss.detach(),
            "loss_u": loss_u_raw,
            "loss_v": loss_v_raw,
            "loss_perceptual": loss_perceptual,
            "u_norm": u_norm,
            "du_dt_norm": dudt_norm,
            "corr_over_u": corr_over_u,
        }

    # -----------------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, n_sample, rng, num_steps, omega, t_min, t_max, labels=None):
        device = next(self.parameters()).device
        x_shape = (n_sample, self.img_channels, self.img_size, self.img_size)
        z_t = rng.randn(x_shape).to(self.dtype)

        if labels is not None:
            y = labels.to(z_t.device)
        else:
            y = rng.randint(0, self.num_classes, size=(n_sample,), dtype=torch.int32).to(z_t.device)

        t_steps = torch.linspace(1.0, 0.0, num_steps + 1).to(self.dtype).to(z_t.device)

        omega = torch.as_tensor(omega, dtype=self.dtype, device=z_t.device)
        t_min = torch.as_tensor(t_min, dtype=self.dtype, device=z_t.device)
        t_max = torch.as_tensor(t_max, dtype=self.dtype, device=z_t.device)

        for i in range(num_steps):
            t = t_steps[i]
            r = t_steps[i + 1]
            bsz = z_t.shape[0]
            t_b = t.expand(bsz)
            r_b = r.expand(bsz)
            omega_b = omega.expand(bsz)
            t_min_b = t_min.expand(bsz)
            t_max_b = t_max.expand(bsz)

            u = self.u_fn(z_t, t_b, t_b - r_b, omega_b, t_min_b, t_max_b, y=y)[0]
            z_t = z_t - (t_b - r_b)[:, None, None, None] * u

        return z_t
