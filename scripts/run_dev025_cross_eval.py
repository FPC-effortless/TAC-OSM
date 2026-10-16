"""DEV-025 frozen cross-evaluation; zero training; separate-process replay."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import torch  # noqa: E402
from tac_osm.development.entity_aging import EntityAgingControl  # noqa: E402
from tac_osm.development.noisy_memory import parameter_hash  # noqa: E402
from run_memory_controls_dev import evaluate  # noqa: E402
from audit_dev025_cross_eval import config, inputs, source_hashes, metrics, spec_and_sources, digest  # noqa: E402


def run(cohort, regime, parents, output, replay=False, smoke=False):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    c = config()
    spec = spec_and_sources()
    spec.update(h_levels=[512], eval_steps=4)
    for seed in (c['seeds'][:1] if smoke else c['seeds']):
        arm = "masked_local" if cohort == "DEV023" else "joint"
        source = next(parents.glob(f"*-{regime}-{seed}-{arm}-NOT-CONFIRMATORY"))
        cell = output / f"{cohort}-{regime}-{seed}"
        original = cell / 'original'
        original.mkdir(parents=True, exist_ok=True)
        if not replay:
            for name in ('report.json','model.pt'):
                shutil.copyfile(source/name, original/name)
        parent, frozen = inputs(original, cell.name)
        model = EntityAgingControl('masked_local', seed)
        model.load_state_dict(torch.load(original/'model.pt', map_location='cpu', weights_only=True))
        assert parameter_hash(model) == frozen['parameter_hash']
        sources = source_hashes(parent)
        for grid in c['cohorts']:
            folder = cell / ('grid-'+grid)
            folder.mkdir(exist_ok=True)
            rows, timing = evaluate(model, seed+c['evaluation_offsets'][grid], spec, False)
            assert parameter_hash(model) == frozen['parameter_hash']
            if grid == cohort:
                assert rows == [r for r in parent['rows'] if r['condition']['history']==512], 'diagonal replay differs'
            summary = metrics(rows)
            path = folder/'report.json'
            record = dict(study='DEV-025',cohort=cohort,regime=regime,seed=seed,grid=grid,evaluation_offset=c['evaluation_offsets'][grid],training_steps=0,frozen_input=frozen,source_sha256=sources,rows=rows,summary=summary)
            if replay:
                assert record == json.loads(path.read_text())
                (folder/'replay.json').write_text(json.dumps(dict(verified=True,report_sha256=digest(path),checkpoint_sha256=frozen['checkpoint_sha256'],source_sha256=sources))+'\n')
            else:
                path.write_text(json.dumps(record)+'\n')
                (folder/'timing.json').write_text(json.dumps(timing)+'\n')
            print('DEV025_REPLAY' if replay else 'DEV025_EVALUATED', cohort,regime,seed,grid,flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cohort',choices=('DEV023','DEV024'),required=True)
    p.add_argument('--regime',choices=('short','long'),required=True)
    p.add_argument('--parents',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replay',action='store_true')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    run(a.cohort,a.regime,a.parents,a.output,a.replay,a.smoke)
