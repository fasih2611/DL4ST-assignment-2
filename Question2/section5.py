"""Section 5: symmetrised B and D (Table 6) and the final model's Figure 6 (Figures 3 and 5 repeated)."""
import json

import torch

from data import load_splits
from metrics import fmt, test_windows
from model import MODEL_D, count_parameters
from section2 import RESULTS, load_model
from section3 import figure3_and_4, figure5
from section4 import COLUMNS, evaluate, mean_seeds
from transforms import SymmetrisedForecaster

FINAL = "D"   # final model: symmetrised D (the optional D+B variant was not trained)


def symmetrised(name: str, seed: int) -> SymmetrisedForecaster:
    return SymmetrisedForecaster(load_model(name, seed)).eval()


def table6(names: tuple[str, ...] = ("B", "D")) -> None:
    test_videos = load_splits()["test"]
    inputs, targets, _ = test_windows(test_videos)
    header = ("| Model | E(5) 1x | C_M | E_M | C_R | E_R | E(2) 1x | E(2) 2x | E(2) 3x "
              "| Params | Train s/epoch (underlying model) | Inference s (1,000 videos) |")
    lines, rows = [header, "|---" * 12 + "|"], {}
    for name in names:
        per_seed = [evaluate(symmetrised(name, s), test_videos, inputs, targets) for s in (0, 1)]
        train_s = [json.load(open(f"checkpoints/{name}_seed{s}.json"))["mean_epoch_seconds"] for s in (0, 1)]
        rows[name] = {k: [p[k] for p in per_seed] for k in COLUMNS + ["inference_s"]}
        rows[name]["train_s_per_epoch"] = train_s
        cells = []
        for k in COLUMNS:
            v = rows[name][k]
            if k in ("CM", "CR"):    # consistency gaps of wrapped models in scientific notation
                cells.append(f"{sum(v) / 2:.2e} ({v[0]:.2e}, {v[1]:.2e})")
            else:
                cells.append(mean_seeds(v))
        label = f"Symmetrised {name}" + (f" ({MODEL_D}, final model)" if name == FINAL else "")
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {count_parameters(symmetrised(name, 0)):,} | "
                     f"{mean_seeds(train_s, '.2f')} | {mean_seeds(rows[name]['inference_s'], '.3f')} |")
    lines.append("\nMean (seed 0, seed 1); test split, t = 1. Inference time (median of 5 passes) includes all "
                 "8 transformations, forward passes, inverse transformations and the averaging.")
    (RESULTS / "table6.md").write_text("\n".join(lines) + "\n")
    (RESULTS / "table6.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines))


def figure6() -> None:
    test_videos = load_splits()["test"]
    inputs, targets, _ = test_windows(test_videos)
    final = symmetrised(FINAL, 0)
    # (a) Figure 3 on exactly the same videos as model A's Figure 3
    model_a_panels = json.loads((RESULTS / "figure3_model_A.json").read_text())
    picks = {"M": [], "R": []}
    for p in model_a_panels:
        picks[p["T"]].append(p["video"] - 9000)
    figure3_and_4(final, inputs, targets, tag="final_model", picks=picks, histograms=False)
    # (b) Figure 5 on exactly the same video
    model_a_fig5 = sorted(RESULTS.parent.glob("figures/fig5_model_A_speeds_video*.png"))[0]
    video = int(model_a_fig5.stem.rsplit("video", 1)[1])
    figure5(final, test_videos, tag="final_model", video=video)


if __name__ == "__main__":
    with torch.no_grad():
        table6()
        figure6()
