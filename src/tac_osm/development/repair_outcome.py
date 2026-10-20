"""Small factual post-repair median forecaster; no repair-selection claims."""
import torch
from torch import nn


class RepairOutcomeForecaster(nn.Module):
    def __init__(self, features, center, scale, median, seed=0, width=16):
        super().__init__(); torch.manual_seed(seed)
        self.network = nn.Sequential(nn.Linear(features, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.network[-1].weight); nn.init.zeros_(self.network[-1].bias)
        self.register_buffer('center', torch.as_tensor(center).double())
        self.register_buffer('scale', torch.as_tensor(scale).double())
        self.register_buffer('log_median', torch.as_tensor(median, dtype=torch.float64).log())
        self.double()

    def forward(self, features):
        # Start at the train-only median; positive outputs in SECONDS.
        delta = self.network((features-self.center)/self.scale).squeeze(-1)
        return torch.exp(self.log_median+delta)
