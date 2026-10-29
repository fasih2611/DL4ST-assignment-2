import torch
from torch import Tensor, nn
from data import to_float, window

BATCH_SIZE = 32


def blank_predictor(inputs: Tensor) -> Tensor:
    """All-zero frames."""
    return torch.zeros_like(inputs)


def copy_last_predictor(inputs: Tensor) -> Tensor:
    """Repeat the last input frame for all five horizons."""
    return inputs[:, -1:].expand_as(inputs).clone()


@torch.no_grad()
def predict(model: nn.Module, inputs: Tensor, device: torch.device) -> Tensor:
    """Run `model` (or a predictor function) over float inputs (N, 5, 1, 64, 64) in batches of 32."""
    if isinstance(model, nn.Module):
        model.eval()
    outputs = []
    for batch in inputs.split(BATCH_SIZE):
        outputs.append(model(batch.to(device)).cpu())
    return torch.cat(outputs)


def per_video_mse(predictions: Tensor, targets: Tensor) -> Tensor:
    """(N, 5, 1, 64, 64) x2 -> (N, 5): MSE_h of each video, averaged over the 64 x 64 pixels."""
    return ((predictions - targets) ** 2).mean(dim=(-3, -2, -1))


def test_windows(videos_uint8: Tensor, speed: int = 1, start: int = 1):
    """Float inputs, targets and target mask for evaluation windows (default 1x, t = 1)."""
    w = window(to_float(videos_uint8), start=start, speed=speed)
    return w.inputs, w.targets, w.target_mask


def summarise(mse_h: Tensor) -> dict[str, float]:
    """Per-horizon MSE_h, E(5) and E(2), each averaged over the videos."""
    per_horizon = mse_h.mean(dim=0)
    result = {f"MSE_{h + 1}": per_horizon[h].item() for h in range(mse_h.shape[1])}
    result["E5"] = mse_h.mean(dim=1).mean().item()
    result["E2"] = mse_h[:, :2].mean(dim=1).mean().item()
    return result

def fmt(value: float) -> str:
    """Three significant figures; scientific notation below 1e-3 (handout reporting rule)."""
    if value == 0:
        return "0"
    if abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:#.3g}"
