"""Read-only archive/coverage checker; no Torch, corpus or model calls."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

ROOT=Path(__file__).resolve().parent
receipt=json.loads((ROOT/'RECEIPT.json').read_text())
for name,pin in [('INITIAL_HOSTED.zip',receipt['initial_zip_sha256']),
                 ('REFERENCE_HOSTED.zip',receipt['reference_zip_sha256'])]:
    path=ROOT/name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==pin
    with zipfile.ZipFile(path) as z:
        x=json.loads(z.read('RESULTS.json'))
        assert x['status']=='PASS' and x['untrained_parameters']==123822574
        assert {(r['depth'],r['prefix_tokens']) for r in x['rows']}=={(d,t) for d in (2,8,20) for t in (64,128)}
        assert len(x['rows'])==6
        assert x['weights_before_sha256']==x['weights_after_sha256']==receipt['weights_before_after_sha256']
        assert x['optimizer_steps']==x['GPU_minutes']==0 and not x['corpus_or_TEST_access']
        for row in x['rows']:
            assert row['status']=='PASS' and all(row['checks'].values())
            assert row['compact']['retained_storage_bytes']==row['compact']['logical_bytes']
            assert row['compact']['retained_storage_bytes']<row['before']['retained_storage_bytes']
            assert len(row['copy_samples_ms'])==7
        suites=list(ET.fromstring(z.read('TESTS.xml')).iter('testsuite'))
        assert sum(int(s.attrib['tests']) for s in suites)==93
        assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
print('PASS: both original hosted ZIPs, six-row coverage, all93checks and unchanged weights; no model execution')
