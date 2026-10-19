"""Learned actor with generic probability/log-probability input coordinates.

No health-specific branch, stopping threshold, posterior filter or physical law.
"""
from .semantic_state import SemanticStateAgent


class StableSemanticAgent(SemanticStateAgent):
    def __init__(self, *args, coordinate='probability', **kwargs):
        if coordinate not in ('probability', 'log_probability'):
            raise ValueError(coordinate)
        super().__init__(*args, **kwargs)
        self.coordinate = coordinate

    def act_from_belief(self, posterior, keys, candidates, prices):
        features = posterior if self.coordinate == 'probability' else posterior.clamp_min(1e-12).log()/16
        return super().act_from_belief(features, keys, candidates, prices)


def configurations():
    from .belief_bottleneck import configurations as previous
    prior = previous()
    cells = {
        'tac_moving_probability': ('tac_rel_belief', 'moving', 'probability'),
        'tac_moving_log': ('tac_rel_belief', 'moving', 'log_probability'),
        'tac_frozen_probability': ('tac_rel_belief', 'frozen', 'probability'),
        'tac_frozen_log': ('tac_rel_belief', 'frozen', 'log_probability'),
        'gru_frozen_log': ('gru_rel_belief', 'frozen', 'log_probability'),
        'no_casm_frozen_log': ('no_casm_rel_belief', 'frozen', 'log_probability'),
    }
    return {name: dict(prior[base], schedule=schedule, coordinate=coordinate)
            for name, (base, schedule, coordinate) in cells.items()}
