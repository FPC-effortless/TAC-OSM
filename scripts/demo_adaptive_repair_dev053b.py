"""Load actual MTSK/GRU weights; update only from the chosen public prefix."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from tac_osm.development.adaptive_repair import AdaptiveModel, NEURAL, LEVELS, rollout_probe
from run_learned_belief_dev052b import digest, write


def run(folder, arm, seed, family, index, level, output=None):
    torch.set_num_threads(1)
    folder = Path(folder)
    streams = json.loads((folder/'streams.json').read_text())[family]
    row = streams[index]
    common = json.loads((folder/'common.json').read_text())
    report = json.loads((folder/f'seed{seed}/report.json').read_text())
    cell = report['arms'][arm]
    checkpoint = folder/f'seed{seed}/{arm}.pt'
    assert digest(checkpoint) == cell['checkpoint_sha256']
    model = AdaptiveModel(seed, arm)
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu')); model.eval()
    with torch.no_grad():
        state = model.initial(torch.zeros(1, 5))
        for t, token in enumerate(row['tokens'][:level]):
            state = model.observe(state, torch.tensor([token], dtype=torch.float32))
            if t in (0, 3, 7, 15, 31):
                q = torch.sigmoid(model.predict(state)/cell['temperature'])[0].tolist()
                print(f'After {t+1} public attempts/successes: next-repair forecasts {np.round(q,4).tolist()}')
        predictions = torch.sigmoid(model.predict(state)/cell['temperature'])[0].tolist()
    result = rollout_probe(row, level, predictions, arm, common['parameters'])
    print(json.dumps({'arm': arm, 'seed': seed, 'family': family, 'public_episodes_observed': level,
                      'retained_state_bytes': model.state_scalars*4,
                      'forecast_repair_success': predictions, 'native_probe_cost_per_entity': result['cost']/2,
                      'total_cost_per_entity': result['total_cost_including_observations']/2,
                      'residual_faults': result['failures'], 'blocked': result['blocked']}, indent=2))
    if output: write(output, result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', required=True); p.add_argument('--arm', choices=NEURAL, default='mtsk')
    p.add_argument('--seed', type=int, default=7301); p.add_argument('--family', choices=['markov','drift'], default='drift')
    p.add_argument('--index', type=int, default=0); p.add_argument('--level', type=int, choices=LEVELS, default=32)
    p.add_argument('--output')
    a = p.parse_args(); run(a.evidence, a.arm, a.seed, a.family, a.index, a.level, a.output)
