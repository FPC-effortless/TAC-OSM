#!/usr/bin/env python3
"""Pinned full-formulation/shape receipt. Does not evaluate predictions or train."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'research/mathematical-engineering/context028'


def lean_shapes(text):
    block=text.split('structure Weights where\n',1)[1].split('\nstructure State where',1)[0]
    fields={}
    for line in block.splitlines():
        if not line.strip(): continue
        name,kind=line.strip().split(' : ',1)
        if re.fullmatch(r'Mat \d+ \d+',kind): shape=[int(n) for n in kind.split()[1:]]
        elif re.fullmatch(r'Vec \d+',kind): shape=[int(kind.split()[1])]
        elif re.fullmatch(r'Fin \d+ → Mat \d+ \d+',kind): shape=[int(kind.split()[i]) for i in (1,4,5)]
        else: raise ValueError('unknown Lean learned tensor type '+kind)
        if name in fields: raise ValueError('duplicate Lean weight '+name)
        fields[name]=shape
    return fields


def check_shapes(declared,mapping,actual):
    if set(declared)!=set(mapping) or set(mapping.values())!=set(actual) or len(mapping)!=len(set(mapping.values())):
        raise ValueError('weight mapping coverage differs')
    for name,source in mapping.items():
        if declared[name]!=actual[source]: raise ValueError(f'weight shape differs: {name}/{source}')


def audit():
    link=json.loads((STUDY/'PNDS_GRAPH_LINK.json').read_text())
    source=ROOT/link['tac_source_path']
    if hashlib.sha256(source.read_bytes()).hexdigest()!=link['tac_source_sha256']: raise ValueError('candidate source hash differs')
    base=f"https://raw.githubusercontent.com/FPC-effortless/pnds-lean/{link['pnds_commit']}/"
    blobs={}
    for path,key in [(link['lean_path'],'lean_sha256'),(link['python_path'],'python_sha256'),(link['record_path'],'record_sha256')]:
        with urllib.request.urlopen(base+path,timeout=30) as response: raw=response.read()
        if hashlib.sha256(raw).hexdigest()!=link[key]: raise ValueError('PNDS pinned file differs '+path)
        blobs[path]=raw
    record=json.loads(blobs[link['record_path']])
    if record['tac_commit']!=link['tac_commit'] or record['tac_source_sha256']!=link['tac_source_sha256']:
        raise ValueError('reciprocal source identity differs')
    sys.path.insert(0,str(source.parent))
    from context_transition import ContextTransition026
    model=ContextTransition026()
    actual={name:list(p.shape) for name,p in model.named_parameters()}
    fields=lean_shapes(blobs[link['lean_path']].decode())
    check_shapes(fields,link['weight_map'],actual)
    count=sum(p.numel() for p in model.parameters())
    if count!=500122 or model.parameter_count()!=count: raise ValueError('actual untrained model count differs')
    return {'study':'CONTEXT028','status':'PASS_PINNED_FORMULATION_AND_ALL_WEIGHT_SHAPES_ONLY','learned_tensors':len(actual),
        'learned_parameters':count,'weights':actual,'pnds_commit':link['pnds_commit'],'candidate_source_sha256':link['tac_source_sha256'],
        'model_forward_called':False,'trained':False,'DEV_NLL':'NOT_RUN','IEEE_AD':'OPEN',
        'Lean_to_Python_execution_equivalence':'OPEN','independently_qualified':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try: result=audit()
    except Exception as error:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps({'status':'BLOCKED_ERROR','error':str(error)},indent=2)+'\n')
        raise
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({key:result[key] for key in ('status','learned_tensors','learned_parameters','model_forward_called')}))

if __name__=='__main__': main()
