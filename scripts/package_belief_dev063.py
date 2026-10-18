"""Package every fit plus separately hashed derived integrity evidence."""
import gzip
import json
from pathlib import Path

from package_research_dev062 import package
from run_learned_stack_dev055 import digest, write


def run():
    root = Path('research/dev063-belief-integrity')
    statistics = json.loads((root / 'statistics.json').read_text())
    audit = json.loads((root / 'audit.json').read_text())
    interventions = json.loads((root / 'interventions.json').read_text())
    assert statistics['status'] == audit['status'] == interventions['status'] == 'COMPLETE'
    assert audit['checkpoints'] == 154 and interventions['checkpoints'] == 140
    package('dev063-belief-integrity', 'results/dev063-main',
            'results/dev063-agents', 'results/dev063-smoke-final')
    reference = root / 'public-reference.json'
    with gzip.open(root / 'public-reference.json.gz', 'wb') as stream:
        stream.write(reference.read_bytes())
    reference.unlink()
    index = json.loads((root / 'evidence-index.json').read_text())
    index['decision'] = 'NO_PROMOTION_CALIBRATION_AND_COMPETITIVE_GATES_FAILED'
    index['derived_evidence_sha256'] = {
        filename: digest(root / filename) for filename in [
            'statistics.json', 'statistics.csv', 'audit.json', 'interventions.json',
            'intervention-queries.jsonl.gz', 'public-reference.json.gz',
            'hardware.json', 'diagnostics.json', 'learning-curves.png', 'per-seed.png',
            'local-verification.json',
        ]}
    assert digest(root / 'intervention-queries.jsonl.gz') == interventions['details_sha256']
    write(root / 'evidence-index.json', index)


if __name__ == '__main__':
    run()
