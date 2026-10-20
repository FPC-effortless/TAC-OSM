"""Recorded-only restoration; frozen parent evidence is a separate dependency."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def run(destination):
    source=Path(__file__).resolve().parents[1]/'research/dev069-semantic-value'
    index=json.loads((source/'evidence-index.json').read_text());destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    for name,sha in index['archives'].items():
        with (source/name).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==sha,name
    for name in ('evidence.tar.gz','unqualified-agents.tar.gz'):
        with tarfile.open(source/name,'r:gz') as archive:archive.extractall(destination,filter='data')
    print('Restored stopped evidence and unqualified agents; NO fitting, NO test opening',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--destination',default='results')
    run(p.parse_args().destination)
