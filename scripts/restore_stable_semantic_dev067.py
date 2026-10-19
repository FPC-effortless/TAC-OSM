"""Restore immutable DEV067 evidence without refitting or new evaluation."""
import hashlib
import json
from pathlib import Path
from restore_research_evidence_dev062 import restore


if __name__ == '__main__':
    root = Path('research/dev067-stable-interface')
    index = json.loads((root/'evidence-index.json').read_text())
    for filename, expected in index['derived_evidence_sha256'].items():
        with (root/filename).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, filename
    restore('dev067-stable-interface')
