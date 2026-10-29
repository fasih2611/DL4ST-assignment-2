from typing import NamedTuple
import torch
from torch import Tensor

class Symmetry(NamedTuple):
    """One element of G: rotate by `quarter_turns` anticlockwise, then mirror if `mirrored`."""
    quarter_turns: int
    mirrored: bool


# G = { R^a, M R^a : a = 0, 1, 2, 3 }
G: list[Symmetry] = [Symmetry(a, m) for m in (False, True) for a in range(4)]


def mirror(x: Tensor) -> Tensor:
    return x.flip(-1)


def rotate(x: Tensor, quarter_turns: int = 1) -> Tensor:
    """R^a: rotate by `quarter_turns` x 90 degrees anticlockwise (negative = clockwise)."""
    return torch.rot90(x, k=quarter_turns, dims=(-2, -1))


def apply_symmetry(x: Tensor, g: Symmetry) -> Tensor:
    """g(x): rotate first, then mirror."""
    rotated = rotate(x, g.quarter_turns)
    return mirror(rotated) if g.mirrored else rotated


def invert_symmetry(x: Tensor, g: Symmetry) -> Tensor:
    """g^-1(x): R^a is undone by R^-a, and every M R^a is its own inverse."""
    if g.mirrored:
        return apply_symmetry(x, g)
    return rotate(x, -g.quarter_turns)

class SymmetrisedForecaster(torch.nn.Module):
    """f-bar(X) = 1/8 sum over g in G of g^-1( f( g(X) ) ): exactly consistent under mirror and rotation."""

    def __init__(self, base: torch.nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, inputs: Tensor) -> Tensor:
        total = torch.zeros_like(inputs)
        for g in G:
            total += invert_symmetry(self.base(apply_symmetry(inputs, g)), g)
        return total / len(G)
