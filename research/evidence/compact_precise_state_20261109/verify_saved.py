"""Stdlib-only original archive check, not new model or checkpoint execution."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

ROOT=Path(__file__).resolve().parent
receipt=json.loads((ROOT/'RECEIPT.json').read_text())
assert hashlib.sha256((ROOT/'HOSTED.zip').read_bytes()).hexdigest()==receipt['zip_sha256']
with zipfile.ZipFile(ROOT/'HOSTED.zip') as z:
    x=json.loads(z.read('RESULTS.json'))
    assert z.read('RESULTS.json')==(ROOT/'HOSTED_RESULTS.json').read_bytes()
    assert x['status']=='PASS' and x['untrained_parameters']==123873263
    assert x['weights_before_sha256']==x['weights_after_sha256']==receipt['weights_before_after_sha256']
    assert x['optimizer_steps']==x['GPU_minutes']==0 and not x['corpus_or_TEST_access']
    assert len(x['rows'])==6
    assert {(r['depth'],r['prefix_tokens']) for r in x['rows']}=={(d,t) for d in (2,8,20) for t in (64,128)}
    for row in x['rows']:
        assert row['status']=='PASS' and all(v is True for v in row['checks'].values())
        assert row['compact']['retained_storage_bytes']==row['compact']['logical_bytes']
        assert row['before']['retained_storage_bytes']>row['compact']['retained_storage_bytes']
        assert len(row['copy_samples_ms'])==7
    suites=list(ET.fromstring(z.read('TESTS.xml')).iter('testsuite'))
    assert sum(int(s.attrib['tests']) for s in suites)==73
    assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
local=json.loads((ROOT.parent.parent/'compact_precise_state_20261109/LOCAL_RESULTS.json').read_text())
assert local['weights_before_sha256']==local['weights_after_sha256']
assert (local['weights_after_sha256']==x['weights_after_sha256']) is receipt['hosted_and_local_weight_hashes_match']
assert receipt['hosted_and_local_weight_hashes_match'] is False  # preserve the actual identity mismatch
print('PASS saved archive/73tests/six scoped rows and within-host identity; cross-host initialization hashes DIFFER, not exact checkpoint replay')
