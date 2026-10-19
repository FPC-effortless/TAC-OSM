"""DEV068 public-measurement state, factual dynamics and learned cost policy.

No simulator, Bayesian filter, action teacher or search is imported here.
The 12 state classes name paired inspection outcomes and public graph classes;
the 21 future classes additionally name terminal verification outcomes.
"""
import torch
from .semantic_state import SemanticStateAgent
from .belief_agent import cosine, mlp


class PublicOutcomeAgent(SemanticStateAgent):
    def __init__(self, arm='tac', width=32, aux=32, seed=0):
        super().__init__(arm, width, aux, seed, clock_mode='event_gap', bypass=False)
        self.actor = mlp(20, width, width)
        self.future = mlp(width, width, 21)
        self.cost = mlp(width, width, 1)
        self.policy = torch.nn.Sequential(torch.nn.Linear(30, 64), torch.nn.GELU(),
            torch.nn.Linear(64, 64), torch.nn.GELU(), torch.nn.Linear(64, 1))

    def world_from_state(self, state, keys, candidates, prices, remaining):
        if remaining.shape != (len(state), 1):
            raise ValueError('Explicit normalized remaining public action budget required')
        binding = cosine(candidates[:, :, 3:], keys)
        public = torch.cat((candidates[:, :, :3], binding,
                            prices[:, None].expand(-1, 5, -1),
                            remaining[:, None].expand(-1, 5, -1)), -1)
        actor = self.actor(torch.cat((state[:, None].expand(-1, 5, -1), public), -1))
        future = self.future(actor)
        cost = self.cost(actor).squeeze(-1)
        # Only supervised predictions + public choices enter the value policy.
        # In particular: no encoder latent, current state distribution or actor
        # vector can bypass the factual prediction interface.
        features = torch.cat((future.softmax(-1), cost[:, :, None], public), -1)
        return dict(future=future, cost=cost, features=features)

    def forward(self, events, present, keys, candidates, prices, remaining):
        _, state = self.encode(events, present, keys)
        out = self.world_from_state(state, keys, candidates, prices, remaining)
        q = self.policy(out['features']).squeeze(-1)
        return dict(out, posterior=state, q=q, logits=-q)

    def decide(self, state, candidates, prices, remaining):
        return self(state['events'], state['present'], state['keys'], candidates, prices, remaining)


def configurations():
    target = sum(p.numel() for p in PublicOutcomeAgent().parameters())
    result = {}
    for name, backend, teacher in [('tac_outcome', 'tac', False),
            ('gru_outcome', 'gru', False), ('no_casm_outcome', 'no_casm', False),
            ('tac_teacher_belief', 'tac', True)]:
        aux = 32 if backend == 'tac' else min(range(4, 192), key=lambda a:
            abs(sum(p.numel() for p in PublicOutcomeAgent(backend, 32, a).parameters())-target))
        count = sum(p.numel() for p in PublicOutcomeAgent(backend, 32, aux).parameters())
        assert abs(count/target-1) <= .05
        result[name] = dict(backend=backend, width=32, aux=aux, teacher=teacher, parameters=count)
    return result
