"""Separately registered FP64 diagnostic supplement; preserves FP32 exceptions."""
import argparse
import hashlib
import itertools
import json

import numpy as np
import torch

from scripts import diaggrad001_loss_flow as original
from scripts import me106_priority_panel as panel

PATH = original.OUT/'FP64_DIAGNOSTIC.json'
REG = original.DOC/'FP64_PROTOCOL.json'


def strict_compare(actual, expected, path='result'):
    if isinstance(actual, dict) and isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for k in actual:
            strict_compare(actual[k], expected[k], path+'.'+k)
    elif isinstance(actual, list) and isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, (a, b) in enumerate(zip(actual, expected)):
            strict_compare(a, b, f'{path}[{i}]')
    elif isinstance(actual, float) and isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10, err_msg=path)
    else:
        assert actual == expected, path


def run(replay):
    panel._frozen([REG, original.ROOT/'scripts/diaggrad001_precision.py'])
    if PATH.exists() and not replay:
        raise RuntimeError('FP64 original exists; replay only')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    d = panel.fit_data()
    rawscale = torch.tensor(np.nanstd(d['raw'][..., :6].astype(np.float64), axis=(0, 1)).clip(1e-4), dtype=torch.float64)
    index = json.loads((panel.OUT/'FIT_INDEX.json').read_text())
    rows = []
    for row in index['models']:
        recipe, seed = row['architecture'], int(row['seed'])
        path = panel.OUT/recipe/f'seed-{seed}'/'final.pt'
        assert panel.sha(path) == row['checkpoint_sha256']
        cp = torch.load(path, map_location='cpu', weights_only=False)
        model = panel.initialize(d, seed)
        model.load_state_dict(cp['model'], strict=True)
        model = model.double()
        versions = model.versions()
        for length, population, repeat in itertools.product(original.CONTEXTS, ('general', 'gap'), range(2)):
            rng = np.random.default_rng(107000+length+1000*repeat+(10000 if population=='gap' else 0))
            origins = np.asarray(d[population])[rng.integers(len(d[population]), size=16)]
            batch = panel.tensor_batch(d, origins, length, rng)
            input_sha = hashlib.sha256(b''.join(t.numpy().tobytes() for t in batch)).hexdigest()
            batch = [x.double() if x.is_floating_point() else x for x in batch]
            rows.append(dict(recipe=recipe, seed=seed, history=length, origin_population=population,
                repeat=repeat, input_sha256=input_sha,
                measurement=original.probe(model, cp, batch, rawscale, recipe)))
        assert model.versions() == versions and panel.sha(path) == row['checkpoint_sha256']
        print(f'FP64 {recipe}/{seed} complete', flush=True)
    result = dict(no_new_model_fits=True, precision='FP64', rows=rows,
                  protocol_sha256=panel.sha(REG), source_sha256=panel.sha(original.ROOT/'scripts/diaggrad001_precision.py'),
                  summary=original.summarize(rows))
    if replay:
        strict_compare(result, json.loads(PATH.read_text()))
        print('DIAGGRAD001_FP64_REPLAY_PASS')
    else:
        panel.write(PATH, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('run', 'replay'))
    run(parser.parse_args().mode == 'replay')
