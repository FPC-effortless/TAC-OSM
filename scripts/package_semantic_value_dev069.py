"""Preserve policy-stopped diagnostic evidence; never resume or promote it."""
import gzip
import io
import json
from pathlib import Path
import subprocess
import tarfile
from run_learned_stack_dev055 import write,digest


def run():
    root=Path('results/dev069-main');out=Path('research/dev069-semantic-value')
    summary=json.loads((root/'summary.json').read_text());audit=json.loads((out/'audit.json').read_text())
    partial=json.loads((out/'audit-with-partials.json').read_text())
    assert summary['status']=='ADMINISTRATIVELY_STOPPED' and summary['fits']==24 and summary['complete_fresh_groups']==2
    assert not summary['tests_opened'] and not (root/'TEST-OPENED.json').exists()
    assert audit['status']=='COMPLETE' and audit['checkpoints']==24 and audit['raw_checkpoints']==24
    assert audit['portable_decisions']==50 and audit['native_actions']==5655 and audit['telemetry']['max_abs']==0
    assert partial['partial_training_snapshots']==2 and partial['partial_frozen_body_cases']==24
    for name,sha in json.loads((root/'manifest.json').read_text()).items():assert digest(root/name)==sha,name
    for name in audit['source_files_sha256']:
        assert subprocess.check_output(['git','show',summary['source']+':'+name])==Path(name).read_bytes()
    archives={}
    for name,folder in [('evidence.tar.gz',root),('unqualified-agents.tar.gz',Path('results/dev069-agents'))]:
        with tarfile.open(out/name,'w:gz',compresslevel=6) as archive:archive.add(folder,arcname=folder.name)
        archives[name]=digest(out/name)
    with tarfile.open(out/'source-snapshot.tar.gz','w:gz',compresslevel=6) as archive:
        for name in audit['source_files_sha256']:
            payload=subprocess.check_output(['git','show',summary['source']+':'+name]);info=tarfile.TarInfo(name);info.size=len(payload)
            archive.addfile(info,io.BytesIO(payload))
    archives['source-snapshot.tar.gz']=digest(out/'source-snapshot.tar.gz')
    for source,name in [('/workspace/scratch/dev069-main.log','stopped-run.log.gz'),
                        ('/workspace/scratch/dev069-audit.log','audit.log.gz'),
                        ('/workspace/scratch/dev069-audit-with-partials.log','partial-audit.log.gz')]:
        with gzip.open(out/name,'wb') as stream:stream.write(Path(source).read_bytes())
    write(out/'evidence-index.json',dict(study='DEV069-FROZEN-WORLD-SEMANTIC-VALUE',decision='NO_PROMOTION_INCOMPLETE_POLICY_STOP',
        source=summary['source'],archives=archives,
        reports_sha256={name:digest(out/name) for name in ('audit.json','audit-with-partials.json','statistics.json','interventions.json')},
        dependency='DEV068 evidence-index/archive with same parent file hashes; parent train/encoders NOT duplicated here',
        scope='24completeheadfits(12pilot+12fresh);48finalraw/EMAstates;2partial6-headraw/EMA/optimizer/RNGsnapshots;allremaining18freshseedgroupsuncompleted;testneveropened',
        stop='Administrative application of standing real-world policy after full missed-research reconciliation, not performance-based stopping or executor interruption',
        whole_world_model_qualified=False,component_qualified=False,commercial_qualified=False))


if __name__=='__main__':run()
