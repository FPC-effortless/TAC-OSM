"""Preserve the complete matrix and audits, including failed qualifications."""
import gzip
import json
from pathlib import Path
import subprocess
from package_research_dev062 import package
from run_learned_stack_dev055 import digest, write


def run():
    root = Path('research/dev065-semantic-stability')
    statistics = json.loads((root/'statistics.json').read_text())
    audit = json.loads((root/'audit.json').read_text())
    interventions = json.loads((root/'interventions.json').read_text())
    assert statistics['status'] == audit['status'] == interventions['status'] == 'COMPLETE'
    assert audit['checkpoints'] == 132 and interventions['checkpoints'] == 120
    assert audit['clock_invariance_cases'] == 264 and audit['semantic_interface_cases'] == 132
    assert audit['batched_state_parity_cases'] == 132
    assert audit['portable_batched_decisions'] == audit['portable_decisions']
    files = (
        'src/tac_osm/development/belief_agent.py',
        'src/tac_osm/development/belief_bottleneck.py',
        'src/tac_osm/development/semantic_state.py',
        'src/tac_osm/development/uncertain_world.py',
        'scripts/data_belief_dev062.py', 'scripts/data_belief_dev063.py',
        'scripts/data_semantic_dev065.py', 'scripts/run_semantic_dev065.py',
        'research/dev065-semantic-stability/PROTOCOL.md',
        'research/dev065-semantic-stability/protocol.json')
    for filename in files:
        assert subprocess.check_output(['git', 'show', '1510f7a:'+filename]) == Path(filename).read_bytes()
    profile = json.loads((root/'runtime-profile.json').read_text())
    for filename, expected in profile['source_sha256'].items():
        assert digest(filename) == expected
        assert subprocess.check_output(['git','show',profile['source_commit']+':'+filename]) == Path(filename).read_bytes()
    write(root/'source-provenance.json', dict(runtime_profiling_commit=profile['source_commit'],
        runtime_source_sha256=profile['source_sha256'], frozen_fit_commit=subprocess.check_output(
        ['git', 'rev-parse', '1510f7a'], text=True).strip(),
        files_sha256={name: digest(name) for name in files}))
    package('dev065-semantic-stability', 'results/dev065-main', 'results/dev065-agents', 'results/dev065-smoke')
    reference = root/'public-reference.json'
    with gzip.open(root/'public-reference.json.gz', 'wb') as stream:
        stream.write(reference.read_bytes())
    reference.unlink()
    index = json.loads((root/'evidence-index.json').read_text())
    passed = all(v is True for v in statistics['gates'].values() if isinstance(v, bool))
    index['decision'] = 'LOCAL_SCIENTIFIC_GATES_PASS_REMOTE_PENDING' if passed else 'NO_PROMOTION_REGISTERED_GATES_FAILED'
    index['commercial_qualification'] = 'UNMET_REAL_FORECAST_AND_REPAIR_OUTCOME_GATES'
    index['derived_evidence_sha256'] = {name: digest(root/name) for name in (
        'statistics.json', 'statistics.csv', 'audit.json', 'interventions.json',
        'intervention-trajectories.jsonl.gz', 'public-reference.json.gz',
        'hardware.json', 'runtime-profile.json', 'diagnostics.json', 'learning-curves.png', 'per-seed.png',
        'local-verification.json', 'source-provenance.json')}
    assert digest(root/'intervention-trajectories.jsonl.gz') == interventions['details_sha256']
    write(root/'evidence-index.json', index)


if __name__ == '__main__':
    run()
