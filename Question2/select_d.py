"""Choose model D among the candidates using the validation split only.

Rule (fixed before training): lowest mean validation E(2) over speeds 1x, 2x, 3x (t = 1),
averaged over seeds 0 and 1. The winner's checkpoints are copied to checkpoints/D_seed{0,1}.
"""
import shutil

import torch

from data import load_splits
from metrics import fmt, per_video_mse, predict, test_windows
from model import ARCHITECTURES, MODEL_D, count_parameters, build_model
from section2 import DEVICE, RESULTS, load_model

CANDIDATES = [name for name in ARCHITECTURES if name.startswith("D")]


def validation_e2(name: str, seed: int, val_videos) -> dict[int, float]:
    model = load_model(name, seed)
    out = {}
    for speed in (1, 2, 3):
        X, Y, _ = test_windows(val_videos, speed=speed)
        out[speed] = per_video_mse(predict(model, X, DEVICE), Y)[:, :2].mean().item()
    return out


if __name__ == "__main__":
    val = load_splits()["val"]
    lines = ["| Candidate | Params | val E(2) 1x | val E(2) 2x | val E(2) 3x | mean over speeds |",
             "|---|---|---|---|---|---|"]
    scores = {}
    for name in CANDIDATES:
        per_seed = [validation_e2(name, s, val) for s in (0, 1)]
        by_speed = {k: sum(p[k] for p in per_seed) / 2 for k in (1, 2, 3)}
        scores[name] = sum(by_speed.values()) / 3
        lines.append(f"| {name} | {count_parameters(build_model(name)):,} | "
                     + " | ".join(fmt(by_speed[k]) for k in (1, 2, 3)) + f" | {fmt(scores[name])} |")
    winner = min(scores, key=scores.get)
    lines.append(f"\nSelected model D: **{winner}** (validation split only; test split not used).")
    (RESULTS / "model_D_selection.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    for seed in (0, 1):
        for ext in ("pt", "json"):
            shutil.copyfile(f"checkpoints/{winner}_seed{seed}.{ext}", f"checkpoints/D_seed{seed}.{ext}")
    if winner != MODEL_D:
        print(f"\n!! Set MODEL_D = \"{winner}\" in model.py so build_model('D') matches the copied weights.")
