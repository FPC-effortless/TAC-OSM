"""Public-history-only preparation, with verified legacy TRAINING visits."""
from pathlib import Path
import json
import numpy as np
import torch
from data_belief_dev063 import BeliefWorld, snapshot, batch
from run_learned_stack_dev055 import digest


def corpus(seed, count):
    records = []
    for index in range(count):
        world = BeliefWorld(seed, index, (32, 128)[index % 2],
                            error=(.1, .2)[(index//2) % 2], inspection=(.05, .1, .2)[index % 3])
        belief = world.reference()
        rng = np.random.default_rng(np.random.SeedSequence([seed, index, 667]))
        for _ in range(int(rng.integers(10))):
            action = int(rng.integers(1, 5)) if rng.random() < .4 else int(rng.choice(5, p=belief.policy()))
            if action == 0:
                break
            belief.update(action, world.act(action))
        row = snapshot(world, belief)
        graph = 1 if belief.parents[1] == 0 else 2 if belief.parents[0] == 1 else 0
        target = torch.zeros(4, 3, dtype=torch.float64)
        target[:, graph] = torch.tensor(belief.p, dtype=torch.float64)
        row['posterior'] = target.flatten()
        row['prices'] = torch.tensor([world.error, world.inspection], dtype=torch.float64)
        records.append(row)
    return records


def legacy_training(directory, count, seed):
    root = Path(directory)
    assert json.loads((root/'summary.json').read_text())['status'] == 'COMPLETE'
    manifest = json.loads((root/'manifest.json').read_text())
    pool, refs, hashes = [], [], {}
    for initialization in range(96501, 96521):
        filename = f'group-{initialization}/dagger.pt'
        assert digest(root/filename) == manifest[filename]
        hashes[filename] = manifest[filename]
        rows = torch.load(root/filename, weights_only=True)
        allowed = {6652000+initialization*4+stage for stage in range(3)}
        for index, row in enumerate(rows):
            assert row['world_seed'] in allowed, 'Only prior TRAINING DAgger namespaces allowed'
            assert row['world_seed'] not in range(6651001, 6651007)
            pool.append(row); refs.append([filename, index])
    selected = np.random.default_rng(seed).choice(len(pool), count, replace=False)
    result = []
    for index in selected:
        row = dict(pool[int(index)])
        target = row['posterior'].double(); row['posterior'] = target/target.sum()
        result.append(row)
    return result, dict(files_sha256=hashes, available=len(pool),
                        selected_refs=[refs[int(i)] for i in selected],
                        scope='Prior TRAINING visits only; no old test/fixed/intervention histories loaded')


def prepare(records, gate):
    chunks = [batch(records, range(start, min(start+128, len(records))), gate)
              for start in range(0, len(records), 128)]
    length = max(c['events'].shape[1] for c in chunks)
    result = {}
    for name in chunks[0]:
        values = []
        for chunk in chunks:
            value = chunk[name]
            if name in ('events', 'present'):
                padded = value.new_zeros((len(value), length)+value.shape[2:])
                padded[:, :value.shape[1]] = value; value = padded
            values.append(value)
        result[name] = torch.cat(values)
    return result


def take(data, indices, dtype):
    result = {name: value[indices] for name, value in data.items()}
    length = max(1, int(result['present'].sum(-1).max()))
    result['events'] = result['events'][:, :length]
    result['present'] = result['present'][:, :length]
    return {name: value.to(torch.float64 if name == 'posterior' else dtype)
            if value.is_floating_point() else value for name, value in result.items()}
