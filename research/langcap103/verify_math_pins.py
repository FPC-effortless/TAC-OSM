"""Compare fixed peer source bytes, never execute a downloaded module."""
import hashlib
import json
from pathlib import Path
import re
import urllib.request

HERE=Path(__file__).resolve().parent


def verify():
    pin=json.loads((HERE/'PNDS_PIN.json').read_text())
    if pin['repository']!='FPC-effortless/pnds-lean' or not re.fullmatch('[0-9a-f]{40}',pin['commit']):
        raise ValueError('Expected immutable PNDS repository/commit')
    base='https://raw.githubusercontent.com/FPC-effortless/pnds-lean/'+pin['commit']+'/'
    def fetch(path):
        with urllib.request.urlopen(base+path,timeout=30) as r: return r.read(2_000_000)
    raw=fetch(pin['source_map_path'])
    if hashlib.sha256(raw).hexdigest()!=pin['source_map_sha256'] or raw!=(HERE/'MATH_SOURCE_MAP.json').read_bytes():
        raise ValueError('Pinned peer mathematical record mismatch')
    data=json.loads(raw)
    for path,expected in data['mathematical_files_sha256'].items():
        remote=fetch(path)
        if hashlib.sha256(remote).hexdigest()!=expected:raise ValueError('Peer equation bytes mismatch')
        if path.endswith('reference.py') and remote!=(HERE/'math_reference.py').read_bytes():
            raise ValueError('Independent Python equation copy differs from mathematical authority')
    print(json.dumps({'pnds_commit':pin['commit'],'pnds_pr':pin['pull_request'],
        'source_equation_bytes_match':True,'whole_graph_proved':False,'training_authorized':False}))


if __name__=='__main__': verify()
