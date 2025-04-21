import torch
import torch.nn as nn

class LiquidCell(nn.Module):
    """Liquid neural network cell for sequence modeling."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.i2h = nn.Linear(input_dim, hidden_dim)
        self.h2h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.activation = nn.SiLU()
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.activation(self.i2h(x) + self.h2h(h))

class LNNFeatureExtractor(nn.Module):
    """LNN feature extractor for RL agents."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.cell = LiquidCell(input_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, output_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq, feat]
        h = torch.zeros(x.size(0), self.cell.h2h.out_features, device=x.device)
        for t in range(x.size(1)):
            h = self.cell(x[:, t, :], h)
        return self.proj(h) 