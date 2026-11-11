"""Read original source/test/failure evidence; no Torch/model/corpus calls."""
import base64
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

ROOT=Path(__file__).resolve().parent
repo=ROOT.parents[2]
receipt=json.loads((ROOT/'RECEIPT.json').read_text())
assert hashlib.sha256((ROOT/'HOSTED.zip').read_bytes()).hexdigest()==receipt['zip_sha256']
with zipfile.ZipFile(ROOT/'HOSTED.zip') as z:
    suites=list(ET.fromstring(z.read('TESTS.xml')).iter('testsuite'))
    assert sum(int(s.attrib['tests']) for s in suites)==161
    assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
    cases=list(ET.fromstring(z.read('TESTS.xml')).iter('testcase'))
    assert any(c.attrib.get('name')=='test_named_model_and_valid_state_mutations_cannot_restore[buffer]' for c in cases)
    protocol=json.loads(z.read('PROTOCOL.json'))
    assert all(hashlib.sha256((repo/f).read_bytes()).hexdigest()==h for f,h in protocol['source_pins'].items())
    for line in z.read('SOURCE_SHA256.txt').decode().splitlines():
        sha,path=line.split(None,1)
        assert hashlib.sha256((repo/path.strip()).read_bytes()).hexdigest()==sha
folder=repo/'research/stream_binding_20261111'
initial=json.loads((folder/'INITIAL_FAILURE.json').read_text())
raw=base64.b64decode((folder/'INITIAL_BUFFER_FAILURE.log.b64').read_text())
assert hashlib.sha256(raw).hexdigest()==initial['log_sha256']
assert initial['tests_failed']==1 and initial['tests_passed']==53
print('PASS saved161tests/source pins and preserved original1failure/53pass log; no model execution or qualification')
