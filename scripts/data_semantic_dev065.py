"""New public-history corpus; no DEV063 test namespaces or deployed oracle."""
import numpy as np
from data_belief_dev063 import BeliefWorld, snapshot, batch, relation


def corpus(seed, count):
    records = []
    for index in range(count):
        # Both public sensor laws are represented in training, identically for all arms.
        w = BeliefWorld(seed, index, (32, 128)[index % 2],
                        error=(.1, .2)[(index // 2) % 2],
                        inspection=(.05, .1, .2)[index % 3])
        belief = w.reference()
        rng = np.random.default_rng(np.random.SeedSequence([seed, index, 665]))
        for _ in range(int(rng.integers(7))):
            action = int(rng.integers(1, 5)) if rng.random() < .4 else int(rng.choice(5, p=belief.policy()))
            if action == 0:
                break
            belief.update(action, w.act(action))
        records.append(snapshot(w, belief))
    return records
