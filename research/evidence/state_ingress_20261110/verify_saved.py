"""Read saved original test evidence only; no Torch, model, data or Lean calls."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

ROOT=Path(__file__).resolve().parent
receipt=json.loads((ROOT/'RECEIPT.json').read_text())
assert hashlib.sha256((ROOT/'HOSTED.zip').read_bytes()).hexdigest()==receipt['zip_sha256']
with zipfile.ZipFile(ROOT/'HOSTED.zip') as z:
    suites=list(ET.fromstring(z.read('TESTS.xml')).iter('testsuite'))
    assert sum(int(s.attrib['tests']) for s in suites)==106
    assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
    protocol=json.loads(z.read('PROTOCOL.json'))
    repo=ROOT.parents[2]
    assert all(hashlib.sha256((repo/f).read_bytes()).hexdigest()==h for f,h in protocol['source_pins'].items())
    for line in z.read('SOURCE_SHA256.txt').decode().splitlines():
        sha,path=line.split(None,1)
        assert hashlib.sha256((repo/path.strip()).read_bytes()).hexdigest()==sha
print('PASS saved original106test/source/plan evidence; engineering only, not independent qualification')
