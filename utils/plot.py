from pathlib import Path

import matplotlib.pyplot as plt
import torch


def _prepare_images(images: torch.Tensor) -> torch.Tensor:
    """Convert a BCHW image tensor into display-ready BHWC images."""
    images = images.detach().float().cpu()
    if images.min() < 0:
        images = (images + 1) / 2
    images = images.clamp(0, 1)

    if images.shape[1] == 1:
        return images[:, 0]
    if images.shape[1] == 3:
        return images.permute(0, 2, 3, 1)
    raise ValueError(f"Expected 1 or 3 image channels, got {images.shape[1]}.")


def plot_image_triplets(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    sharpened_images: torch.Tensor,
    output_path: str | Path,
) -> Path:
    """Plot corresponding real, generated, and sharpened images side by side."""
    batches = (real_images, generated_images, sharpened_images)
    if any(images.ndim != 4 for images in batches):
        raise ValueError("All image tensors must have shape [B, C, H, W].")

    batch_size = real_images.shape[0]
    if any(images.shape[0] != batch_size for images in batches[1:]):
        raise ValueError("All image tensors must have the same batch size.")

    display_batches = [_prepare_images(images) for images in batches]
    fig, axes = plt.subplots(
        batch_size,
        3,
        figsize=(9, 3 * batch_size),
        squeeze=False,
    )
    titles = ("Real", "Generated", "Sharpened")

    for row in range(batch_size):
        for column, (title, images) in enumerate(zip(titles, display_batches)):
            cmap = "gray" if images[row].ndim == 2 else None
            axes[row, column].imshow(images[row], cmap=cmap)
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(title)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
