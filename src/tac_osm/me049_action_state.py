"""ME049 action-dependent native recurrence on authentic randomized outcome records.

No temporal user trajectory exists in Criteo: 12 coordinates are not time.
Changing a hypothetical action changes a candidate native *feature state*, not
an observed evolving physical world; causal effects require additional proof.
"""
import torch
from torch import nn
from torch.nn import functional as F
from tac_osm.me048_real_action_outcomes import ActionOutcomeModel

NEW_ARM = "action_state_tac"
ARMS = ("action_context", "action_gru", "action_tac", NEW_ARM)


class ActionStateTAC(ActionOutcomeModel):
    """Real candidate action is injected BEFORE fast/slow associative dynamics."""

    def __init__(self):
        super().__init__("action_tac")
        self.action_token = nn.Embedding(2, 16)
        nn.init.normal_(self.action_token.weight, std=0.01)
        self.arm = NEW_ARM

    def logits_for_logged_action(self, x, goal, action):
        if x.ndim != 2 or x.shape[1] != 12:
            raise ValueError("12 authentic covariates required")
        if goal.shape != (len(x),) or action.shape != (len(x),):
            raise ValueError("one action and goal per observed record")
        if not torch.isin(action, torch.tensor([0, 1], device=action.device)).all():
            raise ValueError("observed action must have a supported binary label")
        if not torch.isin(goal, torch.tensor([0, 1], device=goal.device)).all():
            raise ValueError("goal must be visit or conversion")
        v = ((x - self.trained_mean) / self.trained_std).clamp(-15, 15)
        tokens = self.coordinate(v.unsqueeze(-1))
        conditioned_tokens = tokens + self.action_token(action)[:, None, :]
        state = self.state(conditioned_tokens)
        combined = torch.cat((state, self.goal(goal)), dim=-1)
        logits = self.action_head(combined)
        return logits.gather(1, action[:, None])[:, 0]

    def forward(self, x, goal):
        """Both proposed actions receive different native recurrent states."""
        zero = torch.zeros(len(x), dtype=torch.long, device=x.device)
        one = torch.ones(len(x), dtype=torch.long, device=x.device)
        return torch.stack((self.logits_for_logged_action(x, goal, zero),
                            self.logits_for_logged_action(x, goal, one)), dim=-1)


def authentic_logged_action_loss(net, x, action, visit, conversion):
    """Supervise factual labels only; no unobserved potential outcomes."""
    if isinstance(net, ActionStateTAC):
        g_visit = torch.zeros(len(x), dtype=torch.long, device=x.device)
        g_conv = torch.ones(len(x), dtype=torch.long, device=x.device)
        pred_v = net.logits_for_logged_action(x, g_visit, action)
        pred_c = net.logits_for_logged_action(x, g_conv, action)
        return F.binary_cross_entropy_with_logits(pred_v, visit) + (
            0.3 * F.binary_cross_entropy_with_logits(pred_c, conversion))
    from tac_osm.me048_real_action_outcomes import true_observed_action_loss
    return true_observed_action_loss(net, x, action, visit, conversion)


def make_model(name):
    if name == NEW_ARM:
        return ActionStateTAC()
    if name not in ARMS:
        raise ValueError("unregistered experiment architecture")
    return ActionOutcomeModel(name)
