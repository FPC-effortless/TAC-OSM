#!/usr/bin/env python3
"""Source/offset/carry-sequencing audit, not neural evaluation or training."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tac_osm.causal_byte_corpus import (
    load_byte_corpus,causal_byte_chunks,model_input_values,
    incoming_byte_state,outgoing_byte_state,
)


def audit(output):
    if output.exists() or output.is_symlink():
        raise FileExistsError('preserve prior corpus audit')
    study=ROOT/'research/causal_byte_corpus_20261114'
    protocol=json.loads((study/'PROTOCOL.json').read_text())
    docs=load_byte_corpus(ROOT/'research/source-bundles/LANGUAGE-SOURCE023',study/'CORPUS.json',
                         expected_manifest_sha256=protocol['corpus_manifest_sha256'])
    counts=Counter(d.source_role for d in docs)
    if counts!={'FIT':13,'FIT_SUPPLEMENT':1,'DEV':4}:
        raise ValueError('original source cohort changed')
    by_path={d.path:d for d in docs}
    rows=[];cases=[]
    for length in protocol['frozen_design']['chunk_lengths']:
        for role in ('FIT','DEV'):
            chunks=list(causal_byte_chunks(docs,role=role,chunk_length=length))
            carry=None;state_sentinel=object();targets=0;resets=0
            # Only a unique opaque object is used, never a neural model/state.
            for c in chunks:
                incoming=incoming_byte_state(c,carry)
                if incoming is not (None if c.reset_state else state_sentinel):
                    raise ValueError('document state reset/identity contract')
                carry=outgoing_byte_state(c,state_sentinel)
                if set(model_input_values(c))!={'tokens','stream_ids'}:
                    raise ValueError('input projection contains target metadata')
                d=by_path[c.document];a=c.input_offset;b=c.next_input_offset
                if bytes(c.inputs)!=d.raw[a:b] or bytes(c.targets)!=d.raw[a+1:b+1]:
                    raise ValueError('original next-byte source offsets changed')
                targets+=len(c.targets);resets+=c.reset_state
                rows.append({'role':role,'chunk_length':length,'document':c.document,
                    'source_sha256':d.sha256,'input_begin':a,'input_end':b,
                    'target_begin':a+1,'target_end':b+1,'reset_state':c.reset_state,
                    'input_sha256':hashlib.sha256(bytes(c.inputs)).hexdigest(),
                    'target_sha256':hashlib.sha256(bytes(c.targets)).hexdigest()})
            selected=[d for d in docs if d.role==role]
            for d in selected:
                part=[c for c in chunks if c.document==d.path]
                if (bytes(v for c in part for v in c.inputs)!=d.raw[:-1] or
                    bytes(v for c in part for v in c.targets)!=d.raw[1:] or
                    [c.input_offset for c in part]!=list(range(0,len(d.raw)-1,length))):
                    raise ValueError('missing/repeated/cross-document target')
            if targets!=sum(len(d.raw)-1 for d in selected) or resets!=len(selected):
                raise ValueError('pair or reset denominator')
            cases.append({'role':role,'chunk_length':length,'documents':len(selected),
                'chunks':len(chunks),'source_bytes':sum(len(d.raw) for d in selected),
                'within_document_next_byte_pairs':targets,'nonascii_targets':sum(v>=128 for d in selected for v in d.raw[1:]),
                'document_state_resets':resets,'status':'PASS_SOURCE_OFFSET_AND_OPAQUE_CARRY'})
    if any(n in sys.modules for n in ('torch','tokenizers','xgrammar')):
        raise RuntimeError('source-only audit imported neural/tokenizer runtime')
    ledger=''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows).encode()
    result={'study_id':protocol['study_id'],'execution_level':'internal_research',
        'status':'PASS_AUTHENTIC_SOURCE_ADAPTER_ONLY','publisher':'Python Software Foundation / CPython',
        'publisher_commit':'60403a5409ff2c3f3b07dd2ca91a7a3e096839c7',
        'corpus_manifest_sha256':protocol['corpus_manifest_sha256'],
        'source_role_counts':dict(counts),'cases':cases,'chunk_ledger_sha256':hashlib.sha256(ledger).hexdigest(),
        'source_code_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
            'src/tac_osm/causal_byte_corpus.py','scripts/audit_causal_byte_corpus.py','research/causal_byte_corpus_20261114/PROTOCOL.json')},
        'DEV_exposure':'PREVIOUSLY_EXPOSED_NOT_TEST','source_synthetic':False,
        'independently_qualified':False,'model_executions':0,'optimizer_updates':0,'tokenizer_fit_steps':0,'GPU_minutes':0,
        'protected_TEST_opened':False,'training_authorized':False,'promotion_authorized':False,
        'PNDS':'Partial inherited scalar Lean/Python only; full model/source IEEE/AD/caller integration OPEN; fresh Lean NOT_RUN',
        'physical_device':{k:'NOT_RUN' for k in ('target','operators','numerical_agreement','p50_p95','RAM','energy','airplane_mode')},
        'limits':'Original source and integer offsets/opaque carry only. No actual neural forward/state-gradient/endtoend noninterference, predictive score, original trained-checkpoint replay, retrieval or efficiency result. Subsequent fitting must meet100k/matched/data/budget/replay/math gates.'}
    output.mkdir(parents=True)
    (output/'RESULTS.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    (output/'CHUNKS.jsonl').write_bytes(ledger)
    print(json.dumps(cases,sort_keys=True))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    audit(args.output)
