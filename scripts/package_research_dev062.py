"""Preserve complete study evidence in Git-sized, SHA256-indexed archive parts."""
import argparse
import json
from pathlib import Path
import tarfile
import tempfile
from run_learned_stack_dev055 import digest,write


def package(study,run,exports,smoke):
    destination=Path('research')/study;destination.mkdir(exist_ok=True);archives={};parts=[]
    with tempfile.TemporaryDirectory(prefix=study+'-archive-') as tmp:
        archive=Path(tmp)/'evidence.tar.gz'
        with tarfile.open(archive,'w:gz',compresslevel=6) as stream:stream.add(run,arcname=Path(run).name)
        with archive.open('rb') as stream:
            i=0
            while chunk:=stream.read(64*1024*1024):
                name=f'evidence.tar.gz.part{i:03d}';(destination/name).write_bytes(chunk);archives[name]=digest(destination/name);parts.append(name);i+=1
    for filename,source in [('learned-agent.tar.gz',exports),('smoke-only.tar.gz',smoke)]:
        with tarfile.open(destination/filename,'w:gz',compresslevel=6) as stream:stream.add(source,arcname=Path(source).name)
        archives[filename]=digest(destination/filename)
    run_summary=json.loads((Path(run)/'summary.json').read_text());write(destination/'evidence-index.json',{'study':study,'decision':'NO_PROMOTION_PENDING_ALL_GATES','archives':archives,'evidence_parts':parts,'run':run_summary,'scope':'All fitted checkpoints, raw/EMA deployment weights, shared data, collection trajectories, curves, failed outcomes and test trajectories preserved. Smoke excluded. See local/remote audits separately.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--run',required=True);p.add_argument('--exports',required=True);p.add_argument('--smoke',required=True);a=p.parse_args();package(a.study,a.run,a.exports,a.smoke)
