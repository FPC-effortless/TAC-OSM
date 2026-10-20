"""Restore pinned public-outcome evidence without fitting."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def run(destination):
    source=Path(__file__).resolve().parents[1]/'research/dev068-public-outcome'
    index=json.loads((source/'evidence-index.json').read_text()); destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    for name,sha in index['archives'].items():
        with (source/name).open('rb') as stream: assert hashlib.file_digest(stream,'sha256').hexdigest()==sha,name
    # Parts are bounded; combined archive is held on disk rather than duplicating
    # the complete training corpus in memory.
    import tempfile
    with tempfile.TemporaryFile() as combined:
        for name in index['evidence_parts']:
            with (source/name).open('rb') as stream:
                while block:=stream.read(1024*1024):combined.write(block)
        combined.seek(0)
        with tarfile.open(fileobj=combined,mode='r:gz') as archive:archive.extractall(destination,filter='data')
    with tarfile.open(source/'diagnostic-agents.tar.gz','r:gz') as archive:archive.extractall(destination,filter='data')
    print('RESTORED',len(index['evidence_parts']),'evidence parts and diagnostic agents',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--destination',default='results')
    run(parser.parse_args().destination)
