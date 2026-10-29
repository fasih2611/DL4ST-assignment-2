from typing import NamedTuple

import torch
from torch import Tensor
from torchvision.datasets import MovingMNIST

NUM_FRAMES = 20
INPUT_LEN = 5
HORIZON = 5

# Videos are split by dataset index: [start, stop)
SPLIT_RANGES: dict[str, tuple[int, int]] = {
    "train": (0, 8000),
    "val": (8000, 9000),
    "test": (9000, 10000),
}


class Window(NamedTuple):
    inputs: Tensor        # (..., 5, 1, 64, 64)
    targets: Tensor       # (..., 5, 1, 64, 64), zeros where the frame does not exist
    target_mask: Tensor   # (5,) bool, True where the target frame exists


def load_splits(root: str = "data") -> dict[str, Tensor]:
    """Return {"train", "val", "test"} -> uint8 videos of shape (N, 20, 1, 64, 64)."""
    videos = MovingMNIST(root=root, split=None, download=True).data
    return {name: videos[start:stop] for name, (start, stop) in SPLIT_RANGES.items()}


def to_float(videos: Tensor) -> Tensor:
    """uint8 frames -> float32 intensities in [0, 1]."""
    return videos.float() / 255.0


def allowed_starts(speed: int, speed_augmented: bool) -> list[int]:
    """Allowed 1-based start frames t for a window at the given speed."""
    if not speed_augmented:
        assert speed == 1, "normal-speed training (models A, B) uses 1x only"
        return list(range(1, 12))  # t = 1..11: all five target frames exist
    # Models C, D: all five inputs and at least two targets exist, i.e. t + 6k <= 20
    last_start = NUM_FRAMES - 6 * speed
    return list(range(1, last_start + 1))


def window(video: Tensor, start: int = 1, speed: int = 1) -> Window:
    """Cut the window starting at frame `start` (1-based) at the given speed.

    `video` is (20, 1, 64, 64) or batched (B, 20, 1, 64, 64).
    Inputs are frames start, start+k, ..., start+4k; targets are start+5k, ..., start+9k.
    """
    input_frames = [start + speed * j for j in range(INPUT_LEN)]
    target_frames = [start + speed * j for j in range(INPUT_LEN, INPUT_LEN + HORIZON)]
    existing = [frame <= NUM_FRAMES for frame in target_frames]

    def frames_at(numbers: list[int]) -> Tensor:
        indices = [n - 1 for n in numbers]  # 1-based frame numbers -> 0-based indices
        return video[..., indices, :, :, :]

    inputs = frames_at(input_frames)
    targets = torch.zeros_like(inputs)
    kept_slots = [h for h in range(HORIZON) if existing[h]]
    targets[..., kept_slots, :, :, :] = frames_at([target_frames[h] for h in kept_slots])
    return Window(inputs, targets, torch.tensor(existing))