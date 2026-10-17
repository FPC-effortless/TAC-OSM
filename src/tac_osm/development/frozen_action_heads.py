"""DEV-045B public-feature action heads; simulator outcomes enter losses only."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FrozenActionHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 32), nn.Tanh(), nn.Linear(32, 2))

    def forward(self, public_features):
        return self.net(public_features)


def action_head_loss(logits, outcomes, objective):
    if logits.shape != outcomes.shape or logits.shape[-1] != 2:
        raise ValueError("matching WAIT/SERVICE outcomes required")
    p = logits.sigmoid()
    offset = p.new_tensor([0., 1.])
    predicted = 4 * p + offset
    truth = 4 * outcomes + offset
    regret = (F.softmax(-predicted / .25, -1) *
              (truth - truth.min(-1, keepdim=True).values)).sum(-1).mean()
    best = truth.argmin(-1, keepdim=True)
    alternative = 1 - best
    ranking = F.relu(.2 + predicted.gather(-1, best) -
                     predicted.gather(-1, alternative)).mean()
    future = F.binary_cross_entropy_with_logits(logits, outcomes)
    losses = {"forecast": future, "decision": regret, "contrastive": ranking,
              "joint": future + regret + .2 * ranking,
              "probability_joint": future + regret + .2 * ranking}
    if objective not in losses:
        raise ValueError("unregistered objective")
    return losses[objective]


@torch.no_grad()
def frozen_features(model, events, queries):
    """Capture actual public pre-head reads without exposing evaluator labels."""
    captured = []
    hook = model.head.register_forward_pre_hook(lambda _module, args: captured.append(args[0]))
    try:
        original = model.decode(model.encode(*events), queries).softmax(-1)[..., 1]
    finally:
        hook.remove()
    features = captured[0].reshape(original.shape[0], 8, 3, 44)
    return features[:, :, :2].flatten(-2).clone(), original[:, :, :2].clone()


def predictive_metrics(probabilities, outcomes):
    p = torch.as_tensor(probabilities, dtype=torch.float64)
    y = torch.as_tensor(outcomes, dtype=torch.float64)
    if p.shape != y.shape or p.shape[-1] != 2 or not torch.isfinite(p).all():
        raise ValueError("finite matching two-action forecasts required")
    ece = 0.
    for i in range(10):
        mask = (p >= i / 10) & (p <= 1 if i == 9 else p < (i + 1) / 10)
        if mask.any():
            ece += float(mask.sum() / p.numel() * (p[mask].mean() - y[mask].mean()).abs())
    return {"brier": float((p-y).square().mean()), "ece_10": ece}
