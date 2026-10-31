"""Validate immutable peer equations and original-source identities; never run downloads."""
import hashlib
import json
from pathlib import Path
import re
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def verify():
    pin = json.loads((HERE/'PNDS_PIN.json').read_text())
    if pin['repository'] != 'FPC-effortless/pnds-lean' or not re.fullmatch('[0-9a-f]{40}', pin['commit']):
        raise ValueError('immutable PNDS source identity required')
    base = 'https://raw.githubusercontent.com/FPC-effortless/pnds-lean/'+pin['commit']+'/'
    def fetch(path):
        if not isinstance(path, str) or path.startswith('/') or '..' in path.split('/'):
            raise ValueError('repository-relative reviewed paths only')
        with urllib.request.urlopen(base+path, timeout=30) as response:
            raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError('oversized peer record')
        return raw
    raw = fetch(pin['source_map_path'])
    if digest(raw) != pin['source_map_sha256'] or raw != (HERE/'MATH_SOURCE_MAP.json').read_bytes():
        raise ValueError('peer mathematical record differs from pinned local record')
    data = json.loads(raw)
    if not re.fullmatch('[0-9a-f]{40}', data['tac_commit']):
        raise ValueError('original implementation commit missing')
    for path, expected in data['tac_source_sha256'].items():
        if digest((ROOT/path).read_bytes()) != expected:
            raise ValueError('actual implementation changed from mathematical source pin: '+path)
    for path, expected in data['mathematical_files_sha256'].items():
        peer = fetch(path)
        if digest(peer) != expected:
            raise ValueError('independent mathematical file identity mismatch: '+path)
        if path == data['python_reference'] and peer != (HERE/'math_reference.py').read_bytes():
            raise ValueError('local mathematical reference differs from authority')
    local = json.loads((ROOT/'research/pnds-variant-math-links.json').read_text())
    remote = json.loads(fetch('foundation/v1/variant-math-links.json'))
    candidate_id = data['variant_id']
    a = [x for x in local['prospective_candidates'] if x['id'] == candidate_id]
    b = [x for x in remote['prospective_candidates'] if x['id'] == candidate_id]
    if len(a) != 1 or a != b or a[0]['tac_source_commit'] != data['tac_commit']:
        raise ValueError('prospective source/math registry row mismatch')
    print(json.dumps({'pnds_commit': pin['commit'], 'source_commit': data['tac_commit'],
        'mathematical_source_bytes_match': True, 'candidate_row_equal': True,
        'legacy_whole_registry_equality_claimed': False,
        'whole_model_proved': False, 'training_authorized': False}))


if __name__ == '__main__':
    verify()
