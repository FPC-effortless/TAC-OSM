"""Restore SHA256-indexed actual-outcome research evidence; no fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def run(destination):
    source = Path(__file__).resolve().parents[1]/'research/real-maint-004'
    index = json.loads((source/'evidence-index.json').read_text())
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    for name, sha in index['archives'].items():
        path = source/name
        with path.open('rb') as stream: assert hashlib.file_digest(stream, 'sha256').hexdigest() == sha
        with tarfile.open(path, 'r:gz') as archive: archive.extractall(destination, filter='data')
    print('RESTORED', len(index['archives']), 'verified archives', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--destination', default='results')
    run(parser.parse_args().destination)
