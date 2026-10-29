"""Section 2 results: Table 2 (model A vs Blank and Copy-last) and Figure 2 (example forecasts)."""
import json
from pathlib import Path

import torch

from data import load_splits
from metrics import blank_predictor, copy_last_predictor, fmt, per_video_mse, predict, summarise, test_windows
from model import ConvLSTMForecaster, build_model, count_parameters
from viz import FIGURES, digits_overlap, frame_grid

RESULTS = Path("results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(name: str, seed: int) -> torch.nn.Module:
    model = build_model(name)
    model.load_state_dict(torch.load(f"checkpoints/{name}_seed{seed}.pt", map_location="cpu"))
    return model.to(DEVICE).eval()


def mean_and_seeds(values: list[float]) -> str:
    return f"{fmt(sum(values) / len(values))} ({fmt(values[0])}, {fmt(values[1])})"


def table2(inputs, targets) -> dict:
    keys = ["MSE_1", "MSE_2", "MSE_3", "MSE_4", "MSE_5", "E5"]
    per_seed = [summarise(per_video_mse(predict(load_model("A", s), inputs, DEVICE), targets)) for s in (0, 1)]
    rows = {"Model A": {k: [r[k] for r in per_seed] for k in keys}}
    for name, fn in [("Blank", blank_predictor), ("Copy-last", copy_last_predictor)]:
        r = summarise(per_video_mse(predict(fn, inputs, DEVICE), targets))
        rows[name] = {k: [r[k]] for k in keys}

    lines = ["| Model | " + " | ".join(k.replace("_", "") for k in keys) + " | Parameters |",
             "|---" * (len(keys) + 2) + "|"]
    params = {"Model A": f"{count_parameters(ConvLSTMForecaster()):,}", "Blank": "0", "Copy-last": "0"}
    for name, vals in rows.items():
        cells = [mean_and_seeds(v) if len(v) == 2 else fmt(v[0]) for v in vals.values()]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {params[name]} |")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "table2.md").write_text("\n".join(lines) + "\n")
    (RESULTS / "table2.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines))
    return rows


def pick_videos(inputs, targets) -> list[int]:
    """Two test videos whose digits stay apart, then one whose digits overlap in the true future."""
    apart, overlapping = [], []
    for i in range(len(targets)):
        frames = torch.cat([inputs[i], targets[i]])
        if digits_overlap(targets[i]) and not digits_overlap(inputs[i]):
            overlapping.append(i)
        elif not digits_overlap(frames):
            apart.append(i)
    return apart[:2] + overlapping[:1]


def figure2(inputs, targets) -> None:
    predictions = predict(load_model("A", 0), inputs, DEVICE)
    chosen = pick_videos(inputs, targets)
    rows = []
    for i in chosen:
        tag = f"test video {9000 + i}"
        rows += [(f"{tag}\ninput (1-5)", list(inputs[i])),
                 ("truth (6-10)", list(targets[i])),
                 ("model A (6-10)", list(predictions[i]))]
    frame_grid(rows, FIGURES / "fig2_model_A_examples.png",
               title="Figure 2: model A (seed 0) forecasts; last video has overlapping digits",
               column_labels=[f"col {c}: frame {c} / {c + 5} (h={c})" for c in range(1, 6)])
    print("Figure 2 videos (dataset index):", [9000 + i for i in chosen], "- last one overlaps")


if __name__ == "__main__":
    test_inputs, test_targets, _ = test_windows(load_splits()["test"])
    table2(test_inputs, test_targets)
    figure2(test_inputs, test_targets)
