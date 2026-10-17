"""DEV-036 candidate: bounded conditional corrective feedback in local MTSK."""
import torch
from .local_mtsk import BankedMTSK, LocalMTSKPredictor


class BoundedBankedMTSK(BankedMTSK):
    """Preserve state, parameters, addressing and readout; alter only the writer."""

    def transition(self, packed, x):
        c = self.config
        j, d = c.experts, c.rank
        memory, previous = self.unpack(packed)
        raw_key = self.key(x)
        key = raw_key / torch.linalg.vector_norm(raw_key, dim=-1, keepdim=True).clamp_min(1.)
        value = self.value(x)
        prediction = torch.einsum('...jde,...e->...jd', memory, key)
        delta = value.unsqueeze(-2) - prediction
        prior = self.key_prior(x).reshape(*x.shape[:-1], j, d).softmax(-2)
        posterior = (prior.clamp_min(1e-8).log() - .5 * delta.square()).softmax(-2)
        responsibility = (1 - c.temporal_prior) * posterior + c.temporal_prior * previous
        responsibility = responsibility / responsibility.sum(-2, keepdim=True).clamp_min(1e-8)
        base = self.decay_logits.sigmoid().reshape(-1, 1, 1)
        conditioned = self.input_decay(x).reshape(*x.shape[:-1], j if self.multiple else 1, d, 1).sigmoid()
        alpha = (.5 * base + .5 * conditioned).clamp(1e-4, .9999)
        # 0 < beta <= alpha. With ||key|| <= 1 and 0 <= r <= 1,
        # 0 <= alpha - beta*r*||key||^2 <= alpha < 1 per value-coordinate row.
        beta = alpha * self.beta_logits.sigmoid().reshape(-1, 1, 1)
        updated = alpha * memory + beta * responsibility.unsqueeze(-1) * delta.unsqueeze(-1) * key.unsqueeze(-2).unsqueeze(-2)
        next_state = torch.cat((updated.flatten(-3), responsibility.flatten(-2)), -1)
        return next_state, alpha, beta, key, responsibility

    def proposal(self, packed, x):
        return self.transition(packed, x)[0]


def build_bounded(seed=34311):
    """Match every trainable initial tensor of the unbounded local-multi arm."""
    model = LocalMTSKPredictor('mtsk_local_multi', seed)
    bounded = BoundedBankedMTSK(local=True, multiple=True)
    bounded.load_state_dict(model.memory.state_dict(), strict=True)
    model.memory = bounded
    return model
