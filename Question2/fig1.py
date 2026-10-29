from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import load_splits
from flow import flow_directions_and_speeds

NUM_VIDEOS = 50
OUT = Path("figures/fig1_flow.png")


def plot_figure1(directions: np.ndarray, speeds: np.ndarray, out: Path = OUT) -> None:
    kept = len(directions)
    weights = np.full(kept, 1.0 / kept)     # fraction of kept pixels, not counts
    median_speed = float(np.median(speeds))

    fig, (ax_dir, ax_speed) = plt.subplots(1, 2, figsize=(11, 4))

    heights, _, _ = ax_dir.hist(directions, bins=np.arange(0, 361, 20), weights=weights, edgecolor="black")
    ax_dir.set_ylim(0, 1.3 * max(heights.max(), 1 / 18))   # headroom for the legend
    ax_dir.axhline(1 / 18, color="red", linestyle="--", label="uniform (1/18)")
    ax_dir.set(xlabel="flow direction θ (degrees; 0 = right, 90 = up)",
               ylabel="fraction of kept pixels", title="(a) direction", xticks=range(0, 361, 60))
    ax_dir.legend()

    ax_speed.hist(speeds, bins=40, weights=weights, edgecolor="black")
    ax_speed.axvline(median_speed, color="red", linestyle="--", label=f"median = {median_speed:.3g} px/frame")
    ax_speed.set(xlabel="flow speed (pixels per frame)", ylabel="fraction of kept pixels", title="(b) speed")
    ax_speed.legend()

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"saved {out}: {kept} pixels kept, median speed {median_speed:.3g} px/frame")


if __name__ == "__main__":
    train = load_splits()["train"]
    directions, speeds = flow_directions_and_speeds(train[:NUM_VIDEOS])
    plot_figure1(directions, speeds)