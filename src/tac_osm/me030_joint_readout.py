"""Query-conditioned native-state readout; ME029 recurrence is unchanged."""
import torch
from torch import nn
from tac_osm.me028_nonlinear_readout import NativeNonlinearObservableReadout
from tac_osm.me029_order_state import OrderFreeNonlinearControl


class JointNativeReadout(NativeNonlinearObservableReadout):
    def __init__(self, mean: float, scale: float, popularity: torch.Tensor):
        super().__init__(mean, scale, popularity)
        # Every coefficient is scored; retain the direct state path as well.
        self.observable_hidden = nn.Linear(90, 12)

    @torch.jit.export
    def predict_features(self, latent: torch.Tensor, common: torch.Tensor,
                         prior: torch.Tensor) -> torch.Tensor:
        return (self.head(latent) + self.observable_output(torch.tanh(
            self.observable_hidden(torch.cat((latent, common), -1)))) + torch.logit(prior))


class JointPooledReadout(OrderFreeNonlinearControl):
    def __init__(self, mean: float, scale: float, popularity: torch.Tensor):
        super().__init__(mean, scale, popularity)
        self.observable_hidden = nn.Linear(90, 12)

    @torch.jit.export
    def predict_features(self, latent: torch.Tensor, common: torch.Tensor,
                         prior: torch.Tensor) -> torch.Tensor:
        return (self.head(latent) + self.observable_output(torch.tanh(
            self.observable_hidden(torch.cat((latent, common), -1)))) + torch.logit(prior))


def make_model(arm, mean, scale, pop):
    if arm == 'additive_native':
        return NativeNonlinearObservableReadout(mean, scale, pop)
    if arm == 'joint_native':
        return JointNativeReadout(mean, scale, pop)
    if arm == 'joint_pool':
        return JointPooledReadout(mean, scale, pop)
    raise ValueError(arm)


COUNTS = {'additive_native': 135377, 'joint_native': 136145, 'joint_pool': 136681}
