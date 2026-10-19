"""DEV065 semantic-gradient experiment. No teacher or analytical controller.

The runtime path accepts public events/choices only. Training may explicitly
provide a posterior target to ``training_forward``; deployment never does.
"""
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from .belief_agent import cosine
from .belief_bottleneck import BeliefBottleneck, relative_events


class SemanticStateAgent(BeliefBottleneck):
    def encode(self, events, present, keys):
        events = relative_events(events, present, keys, self.clock_mode, float(self.clock_unit))
        subject = cosine(events[:, :, 6:22], keys)
        other = cosine(events[:, :, 22:38], keys)
        x = torch.cat((events[:, :, :6], events[:, :, 39:42], subject, other,
                       events[:, :, 38:39], events[:, :, 42:43] / 32,
                       events[:, :, 43:46]), -1)
        lengths = present.sum(-1).clamp_min(1).cpu()
        if self.arm in ('tac', 'no_casm'):
            local = []
            for slot in range(2):
                order = torch.tensor([slot, 1-slot], device=x.device)
                xx = torch.cat((x[:, :, :9], subject[:, :, order], other[:, :, order], x[:, :, 13:]), -1)
                _, h = self.local(pack_padded_sequence(xx, lengths, batch_first=True, enforce_sorted=False))
                local.append(h[0])
            bank = torch.stack(local, 1)
            query = self.query(torch.cat(local, -1))
            weights = ((self.key(bank) * query[:, None]).sum(-1) / self.width**.5).softmax(-1)
            latent = query + (weights[:, :, None] * self.value(bank)).sum(1)
            if self.arm == 'tac':
                selector = self.selector(latent).softmax(-1)
                latent = latent + sum(selector[:, i, None] * op(latent) for i, op in enumerate(self.ops))
            else:
                latent = latent + self.refine(latent)
        elif self.arm == 'gru':
            _, h = self.sequence(pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False))
            latent = self.query(h[0])
        else:
            padded = x.new_zeros(len(x), 64, 18)
            padded[:, :x.shape[1]] = x * present[:, :, None]
            latent = self.sequence(padded.flatten(1))
        return latent, self.belief(latent).softmax(-1)

    def act_from_belief(self, posterior, keys, candidates, prices):
        """Declared semantic interface; no latent or hidden-world input."""
        binding = cosine(candidates[:, :, 3:], keys)
        prefix = posterior.new_zeros(len(posterior), 5, self.width)
        features = torch.cat((prefix, posterior[:, None].expand(-1, 5, -1),
                              candidates[:, :, :3], binding, prices[:, None].expand(-1, 5, -1)), -1)
        actor = self.actor(features)
        future = self.future(actor)
        cost = self.cost(actor).squeeze(-1)
        logits = self.policy(torch.cat((actor, future.softmax(-1), cost[:, :, None]), -1)).squeeze(-1)
        return {'future': future, 'cost': cost, 'logits': logits}

    def training_forward(self, events, present, keys, candidates, prices,
                         isolate=False, semantic_target=None, exact_mask=None):
        latent, posterior = self.encode(events, present, keys)
        policy_belief = posterior.detach() if isolate else posterior
        if semantic_target is not None:
            if exact_mask is None or semantic_target.shape != posterior.shape:
                raise ValueError('Explicit row mask and matching semantic targets required')
            policy_belief = torch.where(exact_mask[:, None], semantic_target.detach(), policy_belief)
        elif exact_mask is not None:
            raise ValueError('A mask alone cannot supply training semantics')
        output = self.act_from_belief(policy_belief, keys, candidates, prices)
        joint = posterior.reshape(-1, 4, 3)
        return dict(output, posterior=posterior, latent=latent,
                    health_posterior=joint.sum(-1), relation_posterior=joint.sum(-2))

    def forward(self, events, present, keys, candidates, prices):
        return self.training_forward(events, present, keys, candidates, prices)


def configurations():
    from .belief_bottleneck import configurations as previous
    prior = previous()
    cells = {
        'tac_coupled_predicted': ('tac_rel_belief', False, False),
        'tac_coupled_mixed': ('tac_rel_belief', False, True),
        'tac_isolated_predicted': ('tac_rel_belief', True, False),
        'tac_isolated_mixed': ('tac_rel_belief', True, True),
        'gru_isolated_mixed': ('gru_rel_belief', True, True),
        'no_casm_isolated_mixed': ('no_casm_rel_belief', True, True),
    }
    return {name: dict(prior[base], isolate=isolate, mixed=mixed)
            for name, (base, isolate, mixed) in cells.items()}
