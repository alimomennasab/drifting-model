"""Optimizer construction for the PyTorch iMF trainer.

AdamW remains the default so existing commands and checkpoints keep working.
When ``--optimizer muon`` is selected, eligible Transformer matrices use the
standalone Muon implementation while boundary parameters continue to use
AdamW.

The parameter split mirrors the repository's main UNITE trainer:

* Muon: matrix-shaped hidden/conditioning weights.
* AdamW: input patch projection, output projections, learned tokens, biases,
  normalization/gating vectors, and every other non-matrix parameter.

The standalone ``muon`` package already applies the fan-in/fan-out scaling
used by UNITE's ``match_rms_adamw`` setting.
"""

from __future__ import annotations

from typing import Any

import torch


OPTIMIZER_STATE_FORMAT = "imf_split_optimizer_v1"
MUON_EXCLUDED_NAME_PARTS = (
    "x_embedder",
    "final_layer",
    "_tokens",
)


def is_muon_parameter(name: str, parameter: torch.nn.Parameter) -> bool:
    """Return whether a parameter should receive Muon updates."""
    return (
        parameter.requires_grad
        and parameter.ndim >= 2
        and not any(part in name for part in MUON_EXCLUDED_NAME_PARTS)
    )


def split_muon_parameters(
    model: torch.nn.Module,
) -> tuple[
    list[torch.nn.Parameter],
    list[torch.nn.Parameter],
    list[str],
    list[str],
]:
    """Split trainable parameters into disjoint Muon and AdamW groups."""
    muon_parameters: list[torch.nn.Parameter] = []
    adamw_parameters: list[torch.nn.Parameter] = []
    muon_names: list[str] = []
    adamw_names: list[str] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if is_muon_parameter(name, parameter):
            muon_parameters.append(parameter)
            muon_names.append(name)
        else:
            adamw_parameters.append(parameter)
            adamw_names.append(name)

    all_trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    selected_ids = [id(parameter) for parameter in muon_parameters + adamw_parameters]
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("Muon/AdamW parameter groups overlap.")
    if set(selected_ids) != {id(parameter) for parameter in all_trainable}:
        raise RuntimeError("Muon/AdamW parameter groups do not cover the model.")
    if not muon_parameters:
        raise RuntimeError("Muon mode selected, but no eligible matrix parameters were found.")
    if not adamw_parameters:
        raise RuntimeError("Muon mode selected, but no AdamW remainder parameters were found.")

    return muon_parameters, adamw_parameters, muon_names, adamw_names


def _standalone_muon_class():
    """Build the repository-compatible adapter around ``SingleDeviceMuon``."""
    try:
        import muon as muon_package
    except ImportError as error:
        raise ImportError(
            "Muon mode requires the standalone optimizer package. Install it with "
            "`python -m pip install muon-optimizer==0.1.0`."
        ) from error

    class RepositoryMuon(muon_package.SingleDeviceMuon):
        """Standalone Muon with safe handling of unused DDP parameters."""

        def __init__(
            self,
            params,
            lr: float = 0.02,
            momentum: float = 0.95,
            weight_decay: float = 0.0,
        ):
            super().__init__(
                params,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )

        @torch.no_grad()
        def step(self, closure=None):
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            else:
                loss = None

            for group in self.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    state = self.state[parameter]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(parameter)
                    update = muon_package.muon_update(
                        parameter.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                    )
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                    parameter.add_(update, alpha=-group["lr"])
            return loss

    return RepositoryMuon


def _build_adamw(parameters, args):
    return torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        betas=(args.adam_b1, args.adam_b2),
        weight_decay=args.weight_decay,
        eps=args.adam_eps,
    )


def build_optimizers(
    model: torch.nn.Module,
    args,
) -> tuple[dict[str, torch.optim.Optimizer], dict[str, Any]]:
    """Build the selected optimizer configuration and a serializable summary."""
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    total_parameter_count = sum(parameter.numel() for parameter in trainable_parameters)

    if args.optimizer == "adamw":
        optimizers = {"adamw": _build_adamw(trainable_parameters, args)}
        summary = {
            "mode": "adamw",
            "total_parameters": total_parameter_count,
            "adamw_tensors": len(trainable_parameters),
            "adamw_parameters": total_parameter_count,
            "adamw_learning_rate": args.learning_rate,
        }
        return optimizers, summary

    if args.optimizer != "muon":
        raise ValueError(f"Unsupported optimizer mode: {args.optimizer!r}")

    (
        muon_parameters,
        adamw_parameters,
        muon_names,
        adamw_names,
    ) = split_muon_parameters(model)
    RepositoryMuon = _standalone_muon_class()

    optimizers = {
        "muon": RepositoryMuon(
            muon_parameters,
            lr=args.muon_learning_rate,
            momentum=args.muon_momentum,
            weight_decay=args.muon_weight_decay,
        ),
        "adamw": _build_adamw(adamw_parameters, args),
    }
    summary = {
        "mode": "split_muon_adamw",
        "total_parameters": total_parameter_count,
        "muon_tensors": len(muon_parameters),
        "muon_parameters": sum(parameter.numel() for parameter in muon_parameters),
        "adamw_tensors": len(adamw_parameters),
        "adamw_parameters": sum(parameter.numel() for parameter in adamw_parameters),
        "muon_learning_rate": args.muon_learning_rate,
        "adamw_learning_rate": args.learning_rate,
        "muon_momentum": args.muon_momentum,
        "muon_weight_decay": args.muon_weight_decay,
        "muon_adjust_lr": "match_rms_adamw (built into standalone muon_update)",
        "muon_first_names": muon_names[:8],
        "adamw_first_names": adamw_names[:8],
    }
    return optimizers, summary


def set_optimizer_lrs(
    optimizers: dict[str, torch.optim.Optimizer],
    *,
    adamw_lr: float,
    muon_lr: float,
) -> None:
    """Set each optimizer's LR without collapsing the two LR scales."""
    for group in optimizers["adamw"].param_groups:
        group["lr"] = adamw_lr
    if "muon" in optimizers:
        for group in optimizers["muon"].param_groups:
            group["lr"] = muon_lr


def zero_grad_optimizers(
    optimizers: dict[str, torch.optim.Optimizer],
) -> None:
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)


def unscale_optimizers(
    optimizers: dict[str, torch.optim.Optimizer],
    scaler: torch.cuda.amp.GradScaler,
) -> None:
    for optimizer in optimizers.values():
        scaler.unscale_(optimizer)


def step_optimizers(
    optimizers: dict[str, torch.optim.Optimizer],
    scaler: torch.cuda.amp.GradScaler | None,
) -> None:
    if scaler is None:
        for optimizer in optimizers.values():
            optimizer.step()
        return

    for optimizer in optimizers.values():
        scaler.step(optimizer)
    scaler.update()


def optimizer_state_dict(
    optimizers: dict[str, torch.optim.Optimizer],
) -> dict[str, Any]:
    """Serialize optimizers while preserving the old AdamW-only format."""
    if set(optimizers) == {"adamw"}:
        return optimizers["adamw"].state_dict()
    return {
        "format": OPTIMIZER_STATE_FORMAT,
        "adamw": optimizers["adamw"].state_dict(),
        "muon": optimizers["muon"].state_dict(),
    }


def load_optimizer_state_dict(
    optimizers: dict[str, torch.optim.Optimizer],
    state: dict[str, Any],
) -> None:
    """Load either an original AdamW checkpoint or a split-Muon checkpoint."""
    if state.get("format") == OPTIMIZER_STATE_FORMAT:
        if set(optimizers) != {"adamw", "muon"}:
            raise ValueError(
                "The checkpoint contains split Muon/AdamW state, but the current "
                "command did not select `--optimizer muon`."
            )
        optimizers["adamw"].load_state_dict(state["adamw"])
        optimizers["muon"].load_state_dict(state["muon"])
        return

    if set(optimizers) != {"adamw"}:
        raise ValueError(
            "The checkpoint contains the original AdamW-only optimizer state. "
            "Resume it with `--optimizer adamw`; start Muon as a fresh run."
        )
    optimizers["adamw"].load_state_dict(state)
