"""Preserve rejected factual-outcome pilots and the unopened-test decision."""
import gzip
import json
from pathlib import Path
import subprocess
import tarfile
from run_learned_stack_dev055 import write, digest


def run():
    destination = Path('research/real-maint-004'); root = Path('results/real004-main')
    summary = json.loads((root/'summary.json').read_text())
    statistics = json.loads((destination/'statistics.json').read_text())
    audit = json.loads((destination/'audit.json').read_text())
    assert summary['status'] == 'VALIDATION_REJECTED' and not summary['test_opened']
    assert audit['status'] == 'COMPLETE' and audit['checkpoints'] == 24
    assert audit['portable_predictions'] == 10 and audit['telemetry']['max_abs'] <= 1e-8
    assert statistics['decision'] == 'NO_PROMOTION_VALIDATION_REJECTED'
    assert not (root/'test-labels.json').exists()
    for name, sha in json.loads((root/'manifest.json').read_text()).items(): assert digest(root/name) == sha
    frozen = summary['source']
    files = ['src/tac_osm/development/repair_outcome.py', 'scripts/data_real_maintenance_004.py',
        'scripts/run_real_maintenance_004.py', 'research/real-maint-004/PROTOCOL.md', 'research/real-maint-004/protocol.json']
    for name in files: assert subprocess.check_output(['git', 'show', frozen+':'+name]) == Path(name).read_bytes()
    write(destination/'source-provenance.json', dict(frozen_fit_commit=frozen,
        fitting_files_sha256={name: digest(name) for name in files},
        audit_runtime_metadata_change='Export metadata adds source/recipe; fitting files and all weights unchanged'))
    write(destination/'qualification.json', dict(status='COMPLETE',
        gates=dict(validation_entry=False, checkpoint_replay=True, standalone_replay=True, test_seal_retained=True),
        numerical_qualified=False, commercial_qualified=False, decision='NO_PROMOTION',
        remote='VERIFY_EXACT_FINAL_HEAD_SEPARATELY'))
    archives = {}
    for name, folder in [('evidence.tar.gz', root),
            ('diagnostic-forecasters.tar.gz', Path('results/real004-forecasters-with-recipe')),
            ('initial-export-metadata.tar.gz', Path('results/real004-forecasters'))]:
        with tarfile.open(destination/name, 'w:gz', compresslevel=6) as archive: archive.add(folder, arcname=folder.name)
        archives[name] = digest(destination/name)
    for log in ('real004-main', 'real004-audit'):
        source = Path('/workspace/scratch')/(log+'.log')
        if source.exists():
            with gzip.open(destination/(log+'.log.gz'), 'wb') as out: out.write(source.read_bytes())
    write(destination/'evidence-index.json', dict(study='REAL-MAINT-004', archives=archives,
        decision='NO_PROMOTION_VALIDATION_REJECTED_TEST_UNOPENED', run=summary,
        source_data_sha256=summary['source_sha256'], raw_and_EMA_checkpoints=48,
        diagnostic_export_scope='Two first-registered recipe0/seed97400 pilots, not best seeds or promoted candidates',
        commercial_qualified=False,
        derived_sha256={name: digest(destination/name) for name in ('statistics.json', 'audit.json',
            'qualification.json', 'source-provenance.json')}))


if __name__ == '__main__': run()
