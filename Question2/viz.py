"""Shared plotting helpers: frame grids, animated GIFs and digit-overlap detection."""
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

FIGURES = Path("figures")
GIFS = Path("gifs")


def frame_grid(rows: list[tuple[str, list[Tensor | None]]], out: Path, title: str | None = None,
               column_labels: list[str] | None = None) -> None:
    """Draw labelled rows of 64x64 frames (None leaves a blank cell) and save to `out`."""
    n_rows, n_cols = len(rows), max(len(frames) for _, frames in rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.15 * n_cols + 1.6, 1.15 * n_rows + 0.4), squeeze=False)
    for r, (label, frames) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.set_xticks([]), ax.set_yticks([])
            frame = frames[c] if c < len(frames) else None
            if frame is None:
                ax.axis("off")
                continue
            ax.imshow(frame.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
            if r == 0 and column_labels:
                ax.set_title(column_labels[c], fontsize=8)
        axes[r, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def save_gif(rows: list[tuple[Tensor, Tensor]], out: Path, seconds_per_frame: float = 0.6) -> None:
    """GIF stepping through the 5 forecast horizons.

    Each row is (5 input frames, 5 future frames); every GIF frame shows all rows as
    [the 5 inputs | the future frame at horizon h], so the row layout of the figure is kept.
    """
    images = []
    for h in range(5):
        lines = []
        for inputs, future in rows:
            cells = [inputs[i].squeeze().numpy() for i in range(5)] + [future[h].squeeze().numpy()]
            cells = [np.pad(c, 2, constant_values=0.5) for c in cells]
            lines.append(np.concatenate(cells, axis=1))
        image = np.concatenate(lines, axis=0)
        images.append((np.clip(image, 0, 1) * 255).astype(np.uint8))
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, images, duration=seconds_per_frame, loop=0)


def blob_count(frame: Tensor, threshold: float = 0.2) -> int:
    """Number of bright connected regions in a frame (two separate digits give 2)."""
    binary = (frame.squeeze().numpy() > threshold).astype(np.uint8)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return int((stats[1:, cv2.CC_STAT_AREA] >= 15).sum())     # ignore specks


def digits_overlap(frames: Tensor) -> bool:
    """True if the two digits touch or overlap in any of the given frames."""
    return any(blob_count(f) < 2 for f in frames)
