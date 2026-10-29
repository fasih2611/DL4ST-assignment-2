"""Section 4: Table 5 for models A-D (errors, consistency, speeds, parameters, timings)."""
import json
import time

import torch

from data import load_splits
from metrics import fmt, test_windows
from model import count_parameters
from section2 import DEVICE, RESULTS, load_model
from section3 import consistency, speed_errors

MODELS = ["A", "B", "C", "D"]
COLUMNS = ["E5", "CM", "EM", "CR", "ER", "E2@1x", "E2@2x", "E2@3x"]


@torch.no_grad()
def inference_seconds(forecaster, inputs: torch.Tensor, batch_size: int = 32, repeats: int = 5) -> float:
    """Median over `repeats` of the wall-clock time to predict all inputs at batch 32.

    Timing starts after each batch is on the device, as the handout specifies.
    """
    return sorted(_one_timing(forecaster, inputs, batch_size) for _ in range(repeats))[repeats // 2]


@torch.no_grad()
def _one_timing(forecaster, inputs: torch.Tensor, batch_size: int) -> float:
    if isinstance(forecaster, torch.nn.Module):
        forecaster.eval()
    batches = [b.to(DEVICE) for b in inputs.split(batch_size)]
    for b in batches[:2]:                      # warm-up so one-off kernel selection is not timed
        forecaster(b)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    tic = time.perf_counter()
    for b in batches:
        forecaster(b)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - tic


def evaluate(forecaster, test_videos, inputs, targets) -> dict[str, float]:
    row = {k: v.mean().item() for k, v in consistency(forecaster, inputs, targets).items()}
    for k in (1, 2, 3):
        row[f"E2@{k}x"] = speed_errors(forecaster, test_videos, k)["E2"]
    row["inference_s"] = inference_seconds(forecaster, inputs)
    return row


def mean_seeds(values: list[float], digits: str | None = None) -> str:
    f = (lambda v: format(v, digits)) if digits else fmt
    return f"{f(sum(values) / len(values))} ({f(values[0])}, {f(values[1])})"


def table5(names: list[str] = MODELS, labels: dict[str, str] | None = None, out_name: str = "table5") -> dict:
    labels = labels or {}
    test_videos = load_splits()["test"]
    inputs, targets, _ = test_windows(test_videos)
    header = ("| Model | E(5) 1x | C_M | E_M | C_R | E_R | E(2) 1x | E(2) 2x | E(2) 3x "
              "| Params | Train s/epoch | Inference s (1,000 videos) |")
    lines = [header, "|---" * 12 + "|"]
    rows = {}
    for name in names:
        per_seed = [evaluate(load_model(name, s), test_videos, inputs, targets) for s in (0, 1)]
        train_s = [json.load(open(f"checkpoints/{name}_seed{s}.json"))["mean_epoch_seconds"] for s in (0, 1)]
        rows[name] = {k: [p[k] for p in per_seed] for k in COLUMNS + ["inference_s"]}
        rows[name]["train_s_per_epoch"] = train_s
        params = count_parameters(load_model(name, 0))
        cells = [mean_seeds(rows[name][k]) for k in COLUMNS]
        lines.append(f"| {labels.get(name, name)} | " + " | ".join(cells) + f" | {params:,} | "
                     f"{mean_seeds(train_s, '.2f')} | {mean_seeds(rows[name]['inference_s'], '.3f')} |")
    device = torch.cuda.get_device_name() if DEVICE.type == "cuda" else "CPU"
    lines.append(f"\nHardware: {device}. Mean (seed 0, seed 1); test split, t = 1. Inference time is the median of 5 timed passes.")
    (RESULTS / f"{out_name}.md").write_text("\n".join(lines) + "\n")
    (RESULTS / f"{out_name}.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines))
    return rows


if __name__ == "__main__":
    from model import MODEL_D
    table5(labels={"D": f"D ({MODEL_D})"})
