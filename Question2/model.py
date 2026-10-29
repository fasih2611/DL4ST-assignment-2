import torch
import torch.nn.functional as F
from torch import Tensor, nn

INPUT_LEN = 5
HORIZON = 5


class ConvLSTMCell(nn.Module):
    """One ConvLSTM layer: a single convolution produces all four gates [i, f, o, g]."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(in_channels + hidden_channels, 4 * hidden_channels, kernel_size,
                               padding=dilation * (kernel_size // 2), dilation=dilation, bias=True)

    def forward(self, u: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        h_prev, c_prev = state
        i, f, o, g = self.gates(torch.cat([u, h_prev], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c_prev + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTMForecaster(nn.Module):
    """Fold 4x4 patches -> two ConvLSTM layers -> 1x1 conv -> unfold + sigmoid."""

    def __init__(self, patch: int = 4, hidden_channels: int = 64, kernel_size: int = 3,
                 dilations: tuple[int, int] = (1, 1)) -> None:
        super().__init__()
        self.patch = patch
        folded = patch * patch                    # 1 x 64 x 64  ->  16 x 16 x 16
        self.layer1 = ConvLSTMCell(folded, hidden_channels, kernel_size, dilations[0])
        self.layer2 = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size, dilations[1])
        self.readout = nn.Conv2d(hidden_channels, folded, kernel_size=1, bias=True)

    def zero_state(self, batch: int, height: int, width: int, like: Tensor) -> tuple[Tensor, Tensor]:
        shape = (batch, self.layer1.hidden_channels, height, width)
        return like.new_zeros(shape), like.new_zeros(shape)

    def step(self, frame: Tensor, states: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, list]:
        """One time step: frame (B, 1, 64, 64) -> prediction of the next frame (B, 1, 64, 64)."""
        u = F.pixel_unshuffle(frame, self.patch)
        h1, c1 = self.layer1(u, states[0])
        h2, c2 = self.layer2(h1, states[1])
        prediction = torch.sigmoid(F.pixel_shuffle(self.readout(h2), self.patch))
        return prediction, [(h1, c1), (h2, c2)]

    def forward(self, inputs: Tensor) -> Tensor:
        """inputs (B, 5, 1, 64, 64) -> predictions (B, 5, 1, 64, 64).

        Runs 9 steps without teacher forcing: steps 1-5 read the input frames,
        steps 6-9 read the model's own previous prediction. Outputs of steps 5-9 are returned.
        """
        batch, _, _, height, width = inputs.shape
        grid = (height // self.patch, width // self.patch)
        states = [self.zero_state(batch, *grid, like=inputs) for _ in range(2)]

        outputs = []
        frame = inputs[:, 0]
        for step in range(1, INPUT_LEN + HORIZON):          # steps 1..9
            prediction, states = self.step(frame, states)
            if step >= INPUT_LEN:                           # outputs of steps 5..9 form Y-hat
                outputs.append(prediction)
            frame = inputs[:, step] if step < INPUT_LEN else prediction
        return torch.stack(outputs, dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Architectures by model name. A, B and C share the baseline; the D candidates only enlarge reach.
ARCHITECTURES: dict[str, dict] = {
    "A": {}, "B": {}, "C": {},
    "D-dilation": {"dilations": (2, 2)},   # 3x3 kernels with dilation 2: reach 16 px, same parameters as A
    "D-patch8": {"patch": 8},              # 8x8 patches, one cell = 8 px: reach 16 px
}
MODEL_D = "D-dilation"   # set from the validation comparison (results/model_D_selection.md)


def build_model(name: str) -> ConvLSTMForecaster:
    """Model by name; "D" resolves to the selected candidate."""
    return ConvLSTMForecaster(**ARCHITECTURES[MODEL_D if name == "D" else name])
