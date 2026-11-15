#!/usr/bin/env python3
"""Untrained original-weight FIT causal/gradient integration; zero optimizer steps."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
import torch
from safetensors.torch import save,load
from tac_osm.byte100k_001 import ByteLanguage100K
from tac_osm.causal_byte_corpus import load_byte_corpus,causal_byte_chunks
from tac_osm.byte100k_001 import predict_byte_chunk
from tac_osm.bound_snapshot import save_bound_snapshot,restore_bound_snapshot
from scripts.validate_research_parity import check_model_parameter_target
from research.byte100k_001.checks import check_case,identity,compare_tensor

SOURCES=('src/tac_osm/byte100k_001.py','research/byte100k_001/checks.py',
         'research/byte100k_001/math_reference.py','scripts/check_byte100k_integration.py',
         'research/byte100k_001/PROTOCOL.json','research/byte100k_001/AMENDMENT_OBJECTIVE_FP64.json')


def sha(raw):return hashlib.sha256(raw).hexdigest()

def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def run(output,original=None,expected_index=None):
    if output.exists() or output.is_symlink():raise FileExistsError('preserve original diagnostics')
    p=json.loads((ROOT/'research/byte100k_001/PROTOCOL.json').read_text())
    if not (ROOT/'research/byte100k_001/AMENDMENT_OBJECTIVE_FP64.json').is_file():
        raise ValueError('registered objective precision amendment required')
    source_sha={path:sha((ROOT/path).read_bytes()) for path in SOURCES}
    old=None
    if original is not None:
        if expected_index is None:raise ValueError('external original index pin required')
        raw=(original/'INDEX.json').read_bytes()
        if sha(raw)!=expected_index:raise ValueError('original index hash mismatch')
        old=json.loads(raw)
        if old['source_sha256']!=source_sha or old['variant_id']!='BYTE100K-001':raise ValueError('original code/variant changed')
    docs=load_byte_corpus(ROOT/'research/source-bundles/LANGUAGE-SOURCE023',
        ROOT/p['source']['manifest'],expected_manifest_sha256=p['source']['manifest_sha256'])
    selected={d.path:d for d in docs if d.role=='FIT'}
    chosen=[selected[n] for n in p['source']['real_FIT_documents']]
    torch.set_num_threads(1)
    output.mkdir(parents=True);(output/'models').mkdir();(output/'observed').mkdir()
    index={'variant_id':'BYTE100K-001','trained':False,'source_sha256':source_sha,'models':{},'observed':{}}
    rows=[];models=[];replay_rows=[]
    for seed in p['seeds']:
        torch.manual_seed(seed+999 if old else seed)
        model=ByteLanguage100K().eval()
        counts=model.validate_live_count()
        check_model_parameter_target(counts['trainable'],counts['deployed'],label='BYTE100K-001')
        initial=identity(model)
        context={'depth':4,'stream_ids':('untrained-seed-'+str(seed),),'observed_tokens':0}
        if old:
            model.core,_=restore_bound_snapshot(original/'models'/str(seed),model.core,
                expected_manifest_sha256=old['models'][str(seed)]['manifest_sha256'],**context)
            if initial==old['models'][str(seed)]['model_identity'] or identity(model)!=old['models'][str(seed)]['model_identity']:
                raise ValueError('original weights not restored into distinct prototype')
        before=identity(model)
        zero=model.core.initial_state(1,torch.device('cpu'),torch.float32,depth=4,stream_ids=context['stream_ids'])
        manifest_pin=save_bound_snapshot(output/'models'/str(seed),model.core,zero,**context)
        index['models'][str(seed)]={'manifest_sha256':manifest_pin,'model_identity':before,'config':asdict(model.core.config)}
        previous=None
        for i,doc in enumerate(chosen):
            tensors={};row,carry=check_case(model,doc,tensors)
            # At actual document boundary compare against fresh prediction.
            first=next(causal_byte_chunks((doc,),role='FIT',chunk_length=128))
            with torch.no_grad():
                reset,_=predict_byte_chunk(model,first,previous);fresh,_=predict_byte_chunk(model,first,None)
            row['checks']['document_boundary_reset']=torch.equal(reset,fresh)
            row['seed']=seed;row['status']='PASS' if all(row['checks'].values()) else 'FAIL';rows.append(row)
            previous=carry
            name=str(seed)+'-'+str(i)+'.safetensors';raw=save(tensors)
            (output/'observed'/name).write_bytes(raw);index['observed'][name]=sha(raw)
            if old:
                # Genuine predictions/gradients computed BEFORE expected tensors open.
                expected=(original/'observed'/name).read_bytes()
                if sha(expected)!=old['observed'][name]:raise ValueError('original expected tensor pin')
                baseline=load(expected)
                if set(baseline)!=set(tensors):raise ValueError('original diagnostic tensor coverage')
                comparisons=[compare_tensor(tensors[k],baseline[k]) for k in tensors]
                replay_rows.append({'seed':seed,'document':doc.path,'tensors':len(tensors),
                    'status':'PASS' if all(g for g,_ in comparisons) else 'FAIL',
                    'maximum_absolute_error':max(e for _,e in comparisons)})
        models.append({'seed':seed,**counts,'model_identity_before':before,'model_identity_after':identity(model),
                       'unchanged':identity(model)==before,'different_prototype_restored':bool(old)})
    status='PASS' if all(r['status']=='PASS' for r in rows+replay_rows) and all(m['unchanged'] for m in models) else 'FAIL'
    result={'variant_id':'BYTE100K-001','scope':'UNTRAINED_GENUINE_FIT_INTEGRATION_NOT_CAPABILITY',
        'status':status,'execution_level':'internal_research','trained':False,'independently_qualified':False,
        'torch':torch.__version__,'models':models,'cases':rows,'original_tensor_replay':replay_rows,
        'source_manifest_sha256':p['source']['manifest_sha256'],'code_sha256':source_sha,
        'optimizer_updates':0,'GPU_minutes':0,'DEV_model_scoring':False,'protected_TEST_opened':False,
        'training_authorized':False,'promotion_authorized':False,'physical_device':'NOT_RUN',
        'historical_FP32_11of84':'PRESERVED_UNCHANGED','initial_FP32_CE_fixture_failure':'PRESERVED_UNCHANGED',
        'PNDS':'PARTIAL_ONLY; source graph/IEEE/AD/provenance refinement OPEN',
        'limits':'No learned NLL/accuracy/recall/efficiency gain. No optimizer/RNG training resume. Original saved models are untrained inference snapshots only; local and hosted initializations are separate. Clean-runner replay requires actual original index/files.'}
    write(output/'INDEX.json',index);write(output/'RESULTS.json',result)
    print(json.dumps({'status':status,'untrained_models':len(models),'FIT_cases':len(rows),'replay_cases':len(replay_rows),
        'all_active_deployed_parameters':counts,'index_sha256':sha((output/'INDEX.json').read_bytes())},sort_keys=True))
    return status

if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--original',type=Path)
    cli.add_argument('--expected-index-sha')
    args=cli.parse_args()
    raise SystemExit(0 if run(args.output,args.original,args.expected_index_sha)=='PASS' else 2)
