"""Section 3: mirror/rotation consistency (Table 3, Figures 3-4) and faster motion (Table 4, Figure 5)."""
import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from data import load_splits
from metrics import copy_last_predictor, blank_predictor, fmt, per_video_mse, predict, test_windows
from section2 import DEVICE, RESULTS, load_model
from transforms import mirror, rotate
from viz import FIGURES, GIFS, digits_overlap, frame_grid, save_gif

Forecaster = Callable[[Tensor], Tensor] | torch.nn.Module
TRANSFORMS = {"M": mirror, "R": rotate}


def consistency(model: Forecaster, inputs: Tensor, targets: Tensor) -> dict[str, Tensor]:
    """Per-video E(5), and C_T, E_T for T in {M, R} (each averaged over pixels and 5 horizons)."""
    plain = predict(model, inputs, DEVICE)
    out = {"E5": per_video_mse(plain, targets).mean(dim=1)}
    for name, T in TRANSFORMS.items():
        on_transformed = predict(model, T(inputs), DEVICE)          # f(T(X))
        out[f"C{name}"] = per_video_mse(on_transformed, T(plain)).mean(dim=1)     # vs T(f(X))
        out[f"E{name}"] = per_video_mse(on_transformed, T(targets)).mean(dim=1)   # vs T(Y)
    return out


def table3(models: dict[str, list[Forecaster]], inputs: Tensor, targets: Tensor) -> dict:
    keys = ["E5", "CM", "EM", "CR", "ER"]
    rows, lines = {}, ["| Model | E(5) | C_M | E_M | C_R | E_R |", "|---|---|---|---|---|---|"]
    for name, fs in models.items():
        per = [{k: v.mean().item() for k, v in consistency(f, inputs, targets).items()} for f in fs]
        rows[name] = {k: [p[k] for p in per] for k in keys}
        cells = [f"{fmt(sum(v) / len(v))} ({fmt(v[0])}, {fmt(v[1])})" if len(v) == 2 else fmt(v[0])
                 for v in rows[name].values()]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    (RESULTS / "table3.md").write_text("\n".join(lines) + "\n")
    (RESULTS / "table3.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines))
    return rows


def figure3_and_4(model: torch.nn.Module, inputs: Tensor, targets: Tensor, tag: str = "model_A",
                  picks: dict[str, list[int]] | None = None, histograms: bool = True) -> dict:
    """Top-2 per-video C_M and C_R videos (4 rows x 10 frames each, plus GIFs) and their histograms.

    Pass `picks` (test-split positions per transformation) to redraw the same videos for another model.
    """
    per_video = consistency(model, inputs, targets)
    if picks is None:
        picks = {name: per_video[f"C{name}"].topk(2).indices.tolist() for name in TRANSFORMS}
    plain = predict(model, inputs, DEVICE)

    panels = []
    for name, T in TRANSFORMS.items():
        for rank, i in enumerate(picks[name], 1):
            X, Y, fX = inputs[i], targets[i], plain[i]
            fTX = predict(model, T(X)[None], DEVICE)[0]
            rows = [(X, fX), (T(X), T(Y)), (T(X), fTX), (T(X), T(fX))]
            labels = ["(1) X | f(X)", f"(2) {name}(X) | {name}(Y)", f"(3) {name}(X) | f({name}(X))",
                      f"(4) {name}(X) | {name}(f(X))"]
            video = 9000 + i
            stem = f"fig3_{tag}_{name}_top{rank}_video{video}"
            frame_grid([(lab, list(a) + list(b)) for lab, (a, b) in zip(labels, rows)], FIGURES / f"{stem}.png",
                       title=f"T = {name}, test video {video}: per-video C_{name} = {fmt(per_video[f'C{name}'][i].item())}",
                       column_labels=[f"in {c}" for c in range(1, 6)] + [f"h={h}" for h in range(1, 6)])
            save_gif(rows, GIFS / f"{stem}.gif")
            panels.append({"T": name, "rank": rank, "video": video, "C": per_video[f"C{name}"][i].item(),
                           "E": per_video[f"E{name}"][i].item(), "png": f"figures/{stem}.png", "gif": f"gifs/{stem}.gif"})

    (RESULTS / f"figure3_{tag}.json").write_text(json.dumps(panels, indent=2))
    for p in panels:
        print(f"Figure 3 [{p['T']} #{p['rank']}] video {p['video']}: C={fmt(p['C'])}, E={fmt(p['E'])} -> {p['png']}, {p['gif']}")
    if not histograms:
        return {"picks": picks, "panels": panels}

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, name in zip(axes, TRANSFORMS):
        values = per_video[f"C{name}"].numpy()
        ax.hist(values, bins=40, edgecolor="black")
        for rank, i in enumerate(picks[name], 1):
            ax.axvline(values[i], color="red", linestyle="--")
            ax.annotate(f"video {9000 + i}", (values[i], ax.get_ylim()[1] * (0.9 - 0.12 * rank)),
                        rotation=90, ha="right", va="top", fontsize=8, color="red")
        ax.set(xlabel=f"per-video C_{name}", ylabel="number of test videos",
               title=f"({'a' if name == 'M' else 'b'}) C_{name}, {tag.replace('_', ' ')} seed 0")
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig4_{tag}_consistency_hist.png", dpi=200)
    plt.close(fig)
    return {"picks": picks, "panels": panels}


def speed_errors(model: Forecaster, test_videos: Tensor, speed: int) -> dict[str, float]:
    X, Y, mask = test_windows(test_videos, speed=speed)
    mse_h = per_video_mse(predict(model, X, DEVICE), Y)
    out = {"E2": mse_h[:, :2].mean().item()}
    if mask.all():
        out["E5"] = mse_h.mean().item()
    return out


def table4(models: dict[str, list[Forecaster]], test_videos: Tensor) -> dict:
    cols = [("E2", 1), ("E2", 2), ("E2", 3), ("E5", 1), ("E5", 2)]
    rows, lines = {}, ["| Model | E(2) 1x | E(2) 2x | E(2) 3x | E(5) 1x | E(5) 2x |", "|---|---|---|---|---|---|"]
    for name, fs in models.items():
        per = [{k: speed_errors(f, test_videos, k) for k in (1, 2, 3)} for f in fs]
        rows[name] = {f"{m}@{k}x": [p[k][m] for p in per] for m, k in cols}
        cells = [f"{fmt(sum(v) / len(v))} ({fmt(v[0])}, {fmt(v[1])})" if len(v) == 2 else fmt(v[0])
                 for v in rows[name].values()]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    (RESULTS / "table4.md").write_text("\n".join(lines) + "\n")
    (RESULTS / "table4.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines))
    return rows


def pick_speed_video(test_videos: Tensor) -> int:
    """First test video whose digits are apart in every displayed frame at 1x, 2x and 3x."""
    for i in range(len(test_videos)):
        shown = []
        for k in (1, 2, 3):
            X, Y, _ = test_windows(test_videos[i:i + 1], speed=k)
            shown += list(X[0]) + list(Y[0, :2])
        if not digits_overlap(torch.stack(shown)):
            return i
    raise RuntimeError("no suitable video")


def figure5(model: torch.nn.Module, test_videos: Tensor, tag: str = "model_A", video: int | None = None) -> int:
    i = pick_speed_video(test_videos) if video is None else video - 9000
    rows = []
    for k in (1, 2, 3):
        X, Y, _ = test_windows(test_videos[i:i + 1], speed=k)
        P = predict(model, X, DEVICE)
        rows += [(f"{k}x input", list(X[0])),
                 (f"{k}x truth h=1,2", list(Y[0, :2])),
                 (f"{k}x prediction h=1,2", list(P[0, :2]))]
    frame_grid(rows, FIGURES / f"fig5_{tag}_speeds_video{9000 + i}.png",
               title=f"Figure 5: test video {9000 + i} at 1x, 2x, 3x ({tag.replace('_', ' ')} seed 0)")
    print(f"Figure 5 video {9000 + i} -> figures/fig5_{tag}_speeds_video{9000 + i}.png")
    return 9000 + i


if __name__ == "__main__":
    test = load_splits()["test"]
    X, Y, _ = test_windows(test)
    models = {"Model A": [load_model("A", 0), load_model("A", 1)],
              "Blank": [blank_predictor], "Copy-last": [copy_last_predictor]}
    table3(models, X, Y)
    figure3_and_4(models["Model A"][0], X, Y)
    table4({"Model A": models["Model A"], "Copy-last": [copy_last_predictor]}, test)
    figure5(models["Model A"][0], test)
