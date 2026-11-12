"""Stdlib-only byte/receipt check; does not import Torch or run any model."""
import hashlib
import json
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'research/evidence/bound_snapshot_20261112'
PINS = {
    'ORIGINAL_CAPTURE.zip': '9a2c33960b7f87ca120961d7fcb5d521050c8f3b7f1a0e1c090e85bfb3589118',
    'COLD_REPLAY.zip': '5a2e2d960fd2940549c2ee5f69b268a9b0ab0508613a30789eea1096759093c3',
    'CHECKS.zip': '1bb0e7f8baea65c53ab90c2012690e9024e5c2941d3c3342f76448e25bb35f71',
}

def check(condition, message):
    if not condition:
        raise ValueError(message)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def main():
    for name, pin in PINS.items():
        check(sha((EVIDENCE/name).read_bytes()) == pin, name+' digest')
    source = json.loads((ROOT/'research/bound_snapshot_20261112/REPLAY_SOURCE.json').read_text())
    with ZipFile(EVIDENCE/'ORIGINAL_CAPTURE.zip') as capture, ZipFile(EVIDENCE/'COLD_REPLAY.zip') as cold, ZipFile(EVIDENCE/'CHECKS.zip') as controls:
        raw_index = capture.read('snapshots/INDEX.json')
        check(sha(raw_index) == source['index_sha256'], 'index pin')
        index = json.loads(raw_index)
        for path, pin in index['source_sha'].items():
            check(sha((ROOT/path).read_bytes()) == pin, 'frozen source '+path)
        counts = {}
        for case, case_index in index['cases'].items():
            prefix = 'snapshots/'+case+'/'
            raw = capture.read(prefix+'manifest.json')
            check(sha(raw) == case_index['manifest_sha256'], 'raw manifest pin')
            manifest = json.loads(raw)
            check(manifest['binding'] == case_index['original_binding'], 'original binding')
            payload = capture.read(prefix+'tensors.safetensors')
            check(sha(payload) == manifest['tensor_sha256'] and len(payload) == manifest['tensor_bytes'], 'payload pin')
            header = json.loads(payload[8:8+struct.unpack('<Q',payload[:8])[0]])
            counts[case] = sum(math.prod(v['shape']) for k,v in header.items() if k.startswith('parameter.'))
            check(sha(capture.read(prefix+'expected.safetensors')) == case_index['expected_sha256'], 'expected output pin')
        check(counts == source['untrained_parameter_counts_from_saved_headers'], 'untrained parameter counts')
        for z in (capture, controls):
            suites = ET.fromstring(z.read('TESTS.xml')).findall('testsuite')
            check(sum(int(s.attrib['tests']) for s in suites) == 219, 'test count')
            check(all(int(s.attrib[k]) == 0 for s in suites for k in ('failures','errors','skipped')), 'test outcomes')
            check(json.loads(z.read('CPU_PLAN.json')) == json.loads((ROOT/'research/bound_snapshot_20261112/COMPUTE_DECISION.json').read_text()), 'CPU plan')
        replay = json.loads(cold.read('REPLAY.json'))
        check(replay['index_sha256'] == source['index_sha256'] and replay['status'] == 'PASS', 'replay source/status')
        check(replay['scope'] == 'UNTRAINED_ENGINEERING_FIXTURE_ONLY' and not replay['trained'] and not replay['independently_qualified'], 'scope')
        check(replay['optimizer_steps'] == replay['GPU_minutes'] == 0 and not replay['TEST_opened'], 'no training/GPU/TEST')
        check(len(replay['rows']) == 2 and {r['case'] for r in replay['rows']} == {'fp32','fp64'}, 'cases')
        for row in replay['rows']:
            check(row['status'] == 'PASS' and row['loaded_binding_byte_exact'] and row['prototype_unchanged'] and row['initial_model_identity_different'], 'binding/prototype')
            check(len(row['all_output_checks']) == 40 and set(row['all_output_checks']) == set(row['output_byte_exact']) == set(row['max_abs_errors']), 'output coverage')
            check(all(row['all_output_checks'].values()) and all(row['output_byte_exact'].values()) and all(v == 0 for v in row['max_abs_errors'].values()), 'recorded output equality')
        check('ModuleNotFoundError' in (EVIDENCE/'INITIAL_COLD_READER_ERROR.log').read_text(), 'first failure preserved')
    print('PASS: saved ZIPs, frozen source/model/state/output pins, 219 controls, 2 untrained cold replay receipts; no model executed')

if __name__ == '__main__':
    main()
