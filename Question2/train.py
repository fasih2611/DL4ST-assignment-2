"""Train one model (A, B, C or D) with one seed under the shared protocol."""
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from data import HORIZON, INPUT_LEN, allowed_starts, load_splits, to_float
from metrics import per_video_mse, predict, test_windows
from model import ARCHITECTURES, build_model, count_parameters
from transforms import G, apply_symmetry

EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
CHECKPOINTS = Path("checkpoints")


@dataclass(frozen=True)
class TrainingRule:
    speeds: tuple[int, ...]       # k drawn uniformly from these
    speed_augmented: bool         # which start-frame rule applies
    symmetry_augmented: bool      # draw g uniformly from G


RULES = {
    "A": TrainingRule(speeds=(1,), speed_augmented=False, symmetry_augmented=False),
    "B": TrainingRule(speeds=(1,), speed_augmented=False, symmetry_augmented=True),
    "C": TrainingRule(speeds=(1, 2, 3), speed_augmented=True, symmetry_augmented=False),
}
# Every model D candidate is trained exactly like model C; only the architecture differs.
for _name in ARCHITECTURES:
    if _name.startswith("D"):
        RULES[_name] = RULES["C"]


def sample_batch(videos: Tensor, rule: TrainingRule, gen: torch.Generator) -> tuple[Tensor, Tensor, Tensor]:
    """One random window per video. videos: uint8 (B, 20, 1, 64, 64) on the training device.

    Returns float inputs (B, 5, 1, 64, 64), targets (B, 5, 1, 64, 64) and mask (B, 5).
    """
    batch = videos.shape[0]
    speeds = torch.tensor(rule.speeds)[torch.randint(len(rule.speeds), (batch,), generator=gen)]
    starts = torch.tensor([
        allowed_starts(int(k), rule.speed_augmented)[
            torch.randint(len(allowed_starts(int(k), rule.speed_augmented)), (1,), generator=gen).item()]
        for k in speeds
    ])
    offsets = torch.arange(INPUT_LEN + HORIZON)
    frames = starts[:, None] + speeds[:, None] * offsets      # (B, 10), 1-based frame numbers
    mask = frames[:, INPUT_LEN:] <= 20                        # which targets exist
    indices = (frames.clamp(max=20) - 1).to(videos.device)    # clamp only to keep gather in range

    clip = to_float(torch.take_along_dim(videos, indices[:, :, None, None, None], dim=1))
    inputs, targets = clip[:, :INPUT_LEN], clip[:, INPUT_LEN:].clone()
    targets[~mask.to(videos.device)] = 0.0                    # missing targets: zeros, masked out

    if rule.symmetry_augmented:
        choice = torch.randint(len(G), (batch,), generator=gen)
        for index, g in enumerate(G):
            rows = (choice == index).nonzero().squeeze(1).to(videos.device)
            if len(rows):
                inputs[rows] = apply_symmetry(inputs[rows], g)     # same g for input and target
                targets[rows] = apply_symmetry(targets[rows], g)
    return inputs, targets, mask.to(videos.device)


def masked_loss(predictions: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    """Squared error averaged over pixels and available target frames per video, then over videos."""
    per_frame = ((predictions - targets) ** 2).mean(dim=(-3, -2, -1))     # (B, 5)
    per_video = (per_frame * mask).sum(dim=1) / mask.sum(dim=1)
    return per_video.mean()


def train(name: str, seed: int, epochs: int = EPOCHS) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)    # shuffling, window, speed and symmetry sampling

    splits = load_splits()
    train_videos = splits["train"].to(device)
    val_inputs, val_targets, _ = test_windows(splits["val"])
    rule = RULES[name]

    model = build_model(name).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    log = {"model": name, "seed": seed, "parameters": count_parameters(model),
           "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu", "epochs": []}

    for epoch in range(1, epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize()
        tic = time.perf_counter()
        order = torch.randperm(len(train_videos), generator=gen)
        total = 0.0
        for batch_ids in order.split(BATCH_SIZE):
            inputs, targets, mask = sample_batch(train_videos[batch_ids.to(device)], rule, gen)
            loss = masked_loss(model(inputs), targets, mask)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(batch_ids)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - tic

        val_e5 = per_video_mse(predict(model, val_inputs, device), val_targets).mean().item()
        log["epochs"].append({"epoch": epoch, "train_loss": total / len(train_videos),
                              "val_E5": val_e5, "seconds": seconds})
        print(f"[{name} seed {seed}] epoch {epoch:2d}  train {total / len(train_videos):.4g}  "
              f"val E5 {val_e5:.4g}  {seconds:.1f}s")

    log["mean_epoch_seconds"] = sum(e["seconds"] for e in log["epochs"]) / epochs
    CHECKPOINTS.mkdir(exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINTS / f"{name}_seed{seed}.pt")    # final-epoch weights
    (CHECKPOINTS / f"{name}_seed{seed}.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(RULES))
    parser.add_argument("--seed", type=int, required=True, choices=[0, 1])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    train(args.model, args.seed, args.epochs)