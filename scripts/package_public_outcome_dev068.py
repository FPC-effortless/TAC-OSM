"""Archive the entire failed matrix, paid acquisition and interruption record."""
import gzip
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
from run_learned_stack_dev055 import write, digest


def run():
    root = Path('results/dev068-main'); destination = Path('research/dev068-public-outcome')
    summary = json.loads((root/'summary.json').read_text())
    audit = json.loads((destination/'audit.json').read_text())
    assert summary['status'] == 'VALIDATION_REJECTED' and summary['fits'] == 88 and summary['complete_fresh_groups'] == 20
    assert not summary['tests_opened'] and not (root/'TEST-OPENED.json').exists()
    assert summary['charged_total_cpu_upper'] <= 24000
    assert audit['status'] == 'COMPLETE' and audit['checkpoints'] == 88 and audit['reconstructed_datasets'] == 2
    assert audit['portable_decisions'] == 29 and audit['native_actions'] == 4114 and audit['telemetry']['max_abs'] <= 1e-8
    for name, sha in json.loads((root/'manifest.json').read_text()).items(): assert digest(root/name) == sha
    assert digest('scripts/resume_public_outcome_dev068.py') == summary['recovery_wrapper_sha256']
    frozen = summary['source']; sources = audit['source_files_sha256']
    for name, sha in sources.items():
        assert digest(name) == sha and subprocess.check_output(['git','show',frozen+':'+name]) == Path(name).read_bytes()
    write(destination/'source-provenance.json', dict(frozen_fit_commit=frozen, frozen_files_sha256=sources,
        recovery_wrapper_sha256=summary['recovery_wrapper_sha256'],
        recovery='Executor-only restart of same two seeds; all partial progress and conservative CPU charges retained',
        optimizer_limit='DEV068 final EMA checkpoints only; raw/intermediate optimizer states were not serialized. DEV069 adds atomic continuation states.'))
    write(destination/'qualification.json', dict(status='COMPLETE', decision='NO_PROMOTION_VALIDATION_REJECTED',
        gates=dict(complete_registered_matrix=True, seed_validation_entry=False, checkpoints_replayed=True,
                   paid_public_data_reconstructed=True, native_replayed=True, portable_replayed=True, tests_remain_closed=True),
        qualified=False, commercial_qualified=False, cross_host='UNKNOWN until new remote head replay'))
    archives = {}; parts = []
    with tempfile.TemporaryDirectory(prefix='dev068-evidence-') as temporary:
        full = Path(temporary)/'evidence.tar.gz'
        with tarfile.open(full,'w:gz',compresslevel=6) as archive: archive.add(root,arcname=root.name)
        with full.open('rb') as stream:
            index = 0
            while chunk := stream.read(64*1024*1024):
                name=f'evidence.tar.gz.part{index:03d}'; (destination/name).write_bytes(chunk)
                archives[name]=digest(destination/name); parts.append(name); index += 1
    with tarfile.open(destination/'diagnostic-agents.tar.gz','w:gz',compresslevel=6) as archive:
        archive.add('results/dev068-agents',arcname='dev068-agents')
    archives['diagnostic-agents.tar.gz']=digest(destination/'diagnostic-agents.tar.gz')
    with tarfile.open(destination/'source-snapshot.tar.gz','w:gz',compresslevel=6) as archive:
        for name in sources:
            payload=subprocess.check_output(['git','show',frozen+':'+name]); info=tarfile.TarInfo(name); info.size=len(payload)
            archive.addfile(info,io.BytesIO(payload))
        archive.add('scripts/resume_public_outcome_dev068.py',arcname='scripts/resume_public_outcome_dev068.py')
    archives['source-snapshot.tar.gz']=digest(destination/'source-snapshot.tar.gz')
    for source,name in [('/workspace/scratch/dev068-main.log','initial-run.log.gz'),
                        ('/workspace/scratch/dev068-resume.log','resumed-run.log.gz'),
                        ('/workspace/scratch/dev068-audit.log','checkpoint-audit.log.gz')]:
        with gzip.open(destination/name,'wb') as out: out.write(Path(source).read_bytes())
    write(destination/'evidence-index.json',dict(study='DEV068-PUBLIC-OUTCOME',decision='NO_PROMOTION',
        archives=archives,evidence_parts=parts,run=summary,
        scope='All88final EMA checkpoints, factual public receipts/acquisition fees, separate teacher-control targets, full pilot/fresh curves and interrupted-fit logs. No sealed tests opened. No raw optimizer state existed in DEV068.',
        derived_sha256={name:digest(destination/name) for name in ('audit.json','statistics.json','qualification.json',
            'source-provenance.json','interim-validation-interventions.json','interim-world-interventions.json',
            'per-seed-validation.png','learning-curves.png')},commercial_qualified=False))


if __name__ == '__main__': run()
