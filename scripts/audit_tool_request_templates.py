#!/usr/bin/env python3
"""Pinned publisher TRAIN + original split ledger; no model/tokenizer/effects."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tac_osm.csl125_training_data import SOURCE_HASHES, SOURCE_REVISION
from tac_osm.jsonl_byte_lineage import read_source_jsonl
from tac_osm.request_template_audit import audit_requests


def run(publisher, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError('preserve prior template audit')
    protocol = json.loads((ROOT/'research/tool_template_20261113/PROTOCOL.json').read_text())
    if hashlib.sha256((ROOT/'src/tac_osm/csl125_training_data.py').read_bytes()).hexdigest() != protocol['source_adapter_sha256']:
        raise ValueError('frozen adapter changed')
    for name, expected in SOURCE_HASHES.items():
        path = publisher/name  # Explicit original3file allowlist, never enumerate TEST.
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('publisher pin mismatch: '+name)
    if not re.search(r'license:\s*apache-2\.0',(publisher/'README.md').read_text(),re.I):
        raise ValueError('publisher license declaration changed')
    records, receipt = read_source_jsonl(publisher/'DroidCall_train.jsonl',SOURCE_HASHES['DroidCall_train.jsonl'])
    old = ROOT/'research/evidence/tool_lineage_20261107/HOSTED_AUDIT.zip'
    if hashlib.sha256(old.read_bytes()).hexdigest() != protocol['frozen_lineage_zip_sha256']:
        raise ValueError('original lineage archive changed')
    with ZipFile(old) as z:
        raw = z.read(protocol['frozen_lineage_member'])
    if hashlib.sha256(raw).hexdigest() != protocol['frozen_lineage_sha256']:
        raise ValueError('original lineage pin mismatch')
    lineage = [json.loads(line) for line in raw.splitlines()]
    by_index = {r.record_index:r for r in records}
    for row in lineage:
        r = by_index.get(row['record_index'])
        if r is None or row['raw_line_sha256'] != r.raw_line_sha256 or (row['byte_begin'],row['byte_end']) != (r.byte_begin,r.byte_end):
            raise ValueError('original source/lineage mismatch')
    # No answers/tools/catalogue are passed to the grouping function.
    methods, ledger = audit_requests([(r.record_index,r.value.get('query')) for r in records],lineage)
    for row in ledger:
        row['raw_line_sha256'] = by_index[row['record_index']].raw_line_sha256
    forbidden = ('torch','tokenizers','xgrammar')
    if any(n in sys.modules for n in forbidden):
        raise RuntimeError('source-only audit imported neural/tokenizer runtime')
    result = {
        'status':'COMPLETE_LEXICAL_AUDIT_NOT_GENERALIZATION_CLEARANCE',
        'study_id':protocol['study_id'],'execution_level':'internal_research',
        'publisher':'mllmTeam/DroidCall','source_revision':SOURCE_REVISION,
        'license':'Apache-2.0 publisher declaration','source_is_synthetic':True,
        'source_receipt':receipt,'original_lineage_sha256':protocol['frozen_lineage_sha256'],
        'source_rows':len(records),'frozen_splits_unchanged':True,'methods':methods,
        'request_grouping_uses_labels':False,
        'cohort_selection_caveat':'Existing accepted/quarantine membership depended on labels/schema; fixed prior cohort only.',
        'limits':'Lexical redactions have false positives/negatives; not semantic equivalence, exhaustive paraphrase detection, actual trained-model contamination, no-call competence or authentic operational outcomes.',
        'independently_qualified':False,'training_authorized':False,'promotion_authorized':False,
        'protected_TEST_opened':False,'optimizer_updates':0,'model_executions':0,'tokenizer_fit_steps':0,'GPU_minutes':0,
        'PNDS':'No neural graph change; full source IEEE/AD/provenance OPEN; fresh Lean NOT_RUN',
        'device':{k:'NOT_RUN' for k in ('target','operators','numerical_agreement','p50_p95','RAM','energy','airplane_mode')},
        'source_code_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
            'src/tac_osm/request_template_audit.py','scripts/audit_tool_request_templates.py','research/tool_template_20261113/PROTOCOL.json')},
    }
    output.mkdir(parents=True)
    (output/'RESULTS.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    (output/'REQUEST_KEYS.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in ledger))
    print(json.dumps({k:{a:b for a,b in v.items() if a != 'cross_partition_groups'} for k,v in methods.items()},sort_keys=True))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publisher-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.publisher_dir,args.output)
