"""Restore one SHA256-indexed recorded study; never retrain during CI."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile


def restore(study,destination='results'):
    root=Path('research')/study;index=json.loads((root/'evidence-index.json').read_text())
    for name,expected in index['archives'].items():
        with (root/name).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==expected,name
    with tempfile.TemporaryFile() as merged:
        for name in index['evidence_parts']:
            with (root/name).open('rb') as stream:shutil.copyfileobj(stream,merged)
        merged.seek(0)
        with tarfile.open(fileobj=merged) as archive:archive.extractall(destination,filter='data')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',choices=['dev0615','dev062','dev062-r1','real-maint-001'],required=True);p.add_argument('--destination',default='results');a=p.parse_args();restore(a.study,a.destination)
