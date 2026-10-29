"""Empirical one-step reach: how far (in folded cells and original pixels) can a change travel in one step?"""
import torch
import torch.nn.functional as F

from model import ConvLSTMForecaster


def one_step_reach(model: ConvLSTMForecaster, patch: int, grid: int = 16) -> dict[str, int]:
    """Gradient of the centre output cell w.r.t. the new frame and each previous state.

    Returns the largest Chebyshev distance (in folded cells) with a non-zero gradient, per source.
    """
    torch.manual_seed(0)
    hidden = model.layer1.hidden_channels
    frame = torch.rand(1, 1, grid * patch, grid * patch, requires_grad=True)
    states = [(torch.randn(1, hidden, grid, grid, requires_grad=True),
               torch.randn(1, hidden, grid, grid, requires_grad=True)) for _ in range(2)]
    prediction, _ = model.step(frame, states)
    centre = grid // 2
    folded_out = F.pixel_unshuffle(prediction, patch)          # back to the folded grid
    folded_out[0, :, centre, centre].sum().backward()

    def extent(grad_folded):
        rows, cols = (grad_folded.abs().sum(0) > 0).nonzero(as_tuple=True)
        return int(torch.maximum((rows - centre).abs(), (cols - centre).abs()).max())

    sources = {"new frame u": F.pixel_unshuffle(frame.grad, patch)[0],
               "h1(prev)": states[0][0].grad[0], "c1(prev)": states[0][1].grad[0],
               "h2(prev)": states[1][0].grad[0], "c2(prev)": states[1][1].grad[0]}
    return {name: extent(g) for name, g in sources.items()}


if __name__ == "__main__":
    model = ConvLSTMForecaster().double().float()
    cells = one_step_reach(model, patch=4)
    for name, d in cells.items():
        print(f"{name:12s} {d} cells = {4 * d} px")
    print("reach:", max(cells.values()), "cells =", 4 * max(cells.values()), "original pixels per step")
