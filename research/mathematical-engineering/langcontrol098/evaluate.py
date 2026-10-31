"""Source-bound acquisition/query actor; no judge answers or scorer arguments.

Acquisition and query run in separate fresh processes. Query consumes only its
observed bytes, saved native state and checkpoint. It never teacher-forces gold.
"""
from __future__ import annotations
import time
PROCESS_START=time.perf_counter()
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

_PATH=Path(__file__).with_name('resources.py')
_SPEC=importlib.util.spec_from_file_location('langcontrol098_resources',_PATH)
R=importlib.util.module_from_spec(_SPEC);sys.modules[_SPEC.name]=R;_SPEC.loader.exec_module(R)
M=R.load_local('models')
CONDITIONS=(*R.CONDITIONS,'binding_shuffle')


def load_actor(path,mode):
    rows=R.read_json(path)
    if not isinstance(rows,list) or not 1<=len(rows)<=128:raise ValueError('bounded actor-only case list required')
    keys={'id','domain','acquisition_bytes'} if mode=='acquire' else {'id','domain','query_bytes','generation_limit'}
    seen=set()
    for row in rows:
        if not isinstance(row,dict) or set(row)!=keys:raise ValueError('actor schema has missing or forbidden fields')
        if not isinstance(row['id'],str) or not row['id'] or row['id'] in seen or row['domain'] not in ('lexical','arc'):
            raise ValueError('invalid or duplicate actor case')
        seen.add(row['id'])
        R.bytes_list(row['acquisition_bytes' if mode=='acquire' else 'query_bytes'],'observations',maximum=272)
        if mode=='query' and row['generation_limit']!={'lexical':4,'arc':40}[row['domain']]:
            raise ValueError('generation budget must be fixed independently of target length')
    return rows


def load_checkpoint(path,expected,config,source_commit):
    if R.sha256(path)!=expected:raise ValueError('checkpoint differs from sealed exact bytes')
    register=R.load_register(config);source=R.verify_source(source_commit,config)
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(payload,dict) or set(payload)!={'schema','metadata','model','optimizer','rng'} or payload['schema']!='langcontrol098.checkpoint.v1':
        raise ValueError('sealed checkpoint schema required')
    metadata=payload['metadata']
    if (metadata.get('study_id')!='LANGCONTROL098'
            or metadata.get('family') not in R.FAMILIES or metadata.get('objective') not in R.OBJECTIVES
            or metadata.get('seed') not in R.SEEDS or metadata.get('status')!='SEALED_BEFORE_DEV'
            or metadata.get('optimizer_steps')!=register['training']['steps']
            or metadata.get('source_commit')!=source_commit or metadata.get('source_hashes')!=source
            or metadata.get('register_sha256')!=R.sha256(config)
            or metadata.get('data_manifest_sha256')!=register['data']['manifest_sha256']):
        raise ValueError('checkpoint provenance or complete fixed fit differs')
    if metadata.get('objective_control')!=R.load_local('objective').objective_mapping(metadata['objective']):
        raise ValueError('checkpoint coefficient-control mapping differs')
    model=M.make_model(metadata['family'],metadata['seed']).eval();model.load_state_dict(payload['model'],strict=True)
    counts=R.assert_counts(model,M.PARAMETERS[metadata['family']])
    if counts!=metadata['counts'] or R.model_digest(model)!=metadata['final_model_sha256']:
        raise ValueError('loaded exact model count/content differs')
    return model,metadata


def pack(state):
    return {'arm':state.arm,'owners':state.owners,'clock':state.clock,'tensors':R.cpu_tree(state.tensors)}


def unpack(value):
    if not isinstance(value,dict) or set(value)!={'arm','owners','clock','tensors'}:raise ValueError('state schema')
    if value['arm'] not in R.FAMILIES or type(value['clock']) is not int or not 1<=value['clock']<=192:
        raise ValueError('legitimate acquisition clock required')
    if not isinstance(value['tensors'],dict) or not value['tensors']:raise ValueError('complete state required')
    for tensor in value['tensors'].values():
        if not isinstance(tensor,torch.Tensor) or tensor.dtype!=torch.float32 or tensor.ndim<1 or tensor.shape[0]!=1 or not bool(torch.isfinite(tensor).all()):
            raise ValueError('finite native state tensors required')
    return M.State098(value['arm'],tuple(value['owners']),value['clock'],value['tensors'])


def combine(states):
    first=states[0]
    if any(s.arm!=first.arm or s.clock!=first.clock or set(s.tensors)!=set(first.tensors) for s in states):raise ValueError('combine requires identical native geometry and clock')
    return M.State098(first.arm,tuple(s.owners[0] for s in states),first.clock,
        {key:torch.cat([s.tensors[key] for s in states],dim=0) for key in first.tensors})


def split(state):
    return [M.State098(state.arm,(owner,),state.clock,{k:t[i:i+1].clone() for k,t in state.tensors.items()}) for i,owner in enumerate(state.owners)]


def acquire(model,rows,controls=None):
    states=[None]*len(rows);aux=[None]*len(rows);elapsed=aux_elapsed=0.
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[len(row['acquisition_bytes'])].append(i)
    for length,indices in groups.items():
        for start in range(0,len(indices),32):
            group=indices[start:start+32];owners=tuple(f'case-{i}' for i in group)
            tokens=torch.tensor([rows[i]['acquisition_bytes'] for i in group],dtype=torch.long)
            tic=time.perf_counter();_,state=model(tokens,owners=owners);elapsed+=time.perf_counter()-tic
            for i,s in zip(group,split(state)):states[i]=s
    if controls is not None:
        if len(controls)!=len(rows):raise ValueError('binding-control cohort differs')
        groups=defaultdict(list)
        for i,(row,c) in enumerate(zip(rows,controls)):
            if (set(c)!={'id','acquisition_bytes','available','reason'} or c['id']!=row['id'] or type(c['available']) is not bool or not isinstance(c['reason'],str)):
                raise ValueError('past-only binding control schema')
            R.bytes_list(c['acquisition_bytes'],'past-only binding observations',maximum=192)
            if len(c['acquisition_bytes'])!=len(row['acquisition_bytes']):raise ValueError('binding control changed acquisition clock')
            if c['available']:
                if row['domain']!='arc':raise ValueError('binding-control scope is ARC only')
                groups[len(c['acquisition_bytes'])].append(i)
        for length,indices in groups.items():
            for start in range(0,len(indices),32):
                group=indices[start:start+32];owners=tuple(f'case-{i}' for i in group)
                tokens=torch.tensor([controls[i]['acquisition_bytes'] for i in group],dtype=torch.long)
                tic=time.perf_counter();_,state=model(tokens,owners=owners);aux_elapsed+=time.perf_counter()-tic
                for i,s in zip(group,split(state)):aux[i]=s
    return states,aux,{'acquisition_seconds':elapsed,'binding_acquisition_seconds':aux_elapsed}


def save_states(path,states,aux,rows,metadata,cost,acquisition_sha,control_sha):
    info={'schema':'langcontrol098.acquisition-state.v1','checkpoint':metadata,
          'case_ids':[r['id'] for r in rows],'domains':[r['domain'] for r in rows],
          'acquisition_sha256':acquisition_sha,'control_sha256':control_sha,
          'state_digest':R.state_digest(states),'aux_state_digest':R.state_digest([s for s in aux if s is not None]),
          'aux_available':[s is not None for s in aux],
          'state_payload_bytes':sum(s.tensor_bytes() for s in states),
          'auxiliary_state_payload_bytes':sum(s.tensor_bytes() for s in aux if s is not None),'cost':cost}
    encoded=json.dumps(info,sort_keys=True,allow_nan=False)
    with Path(path).open('xb') as handle:
        torch.save({'metadata':encoded,'states':[pack(s) for s in states],
                    'auxiliary_states':[pack(s) if s is not None else None for s in aux]},handle)
    return info


def load_states(path,checkpoint,rows):
    if Path(path).stat().st_size>16*1024*1024:raise ValueError('oversized acquired state archive')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if set(saved)!={'metadata','states','auxiliary_states'}:raise ValueError('state archive schema')
    info=json.loads(saved['metadata'])
    if (info.get('schema')!='langcontrol098.acquisition-state.v1' or info['checkpoint']!=checkpoint
            or info['case_ids']!=[r['id'] for r in rows] or info['domains']!=[r['domain'] for r in rows]
            or len(saved['states'])!=len(rows) or len(saved['auxiliary_states'])!=len(rows)):
        raise ValueError('state archive source/case binding differs')
    states=[unpack(s) for s in saved['states']]
    aux=[unpack(s) if s is not None else None for s in saved['auxiliary_states']]
    for i,state in enumerate(states):
        if state.arm!=checkpoint['family'] or state.owners!=(f'case-{i}',):raise ValueError('state owner or model family differs')
        if aux[i] is not None and (aux[i].owners!=state.owners or aux[i].arm!=state.arm or aux[i].clock!=state.clock):
            raise ValueError('binding control changed state ownership/clock')
    if (R.state_digest(states)!=info['state_digest'] or R.state_digest([s for s in aux if s is not None])!=info['aux_state_digest']
            or [s is not None for s in aux]!=info['aux_available']):raise ValueError('saved state digest differs')
    return states,aux,info


def donor_map(path,rows,states):
    donor_rows=R.read_json(path)
    if not isinstance(donor_rows,list) or len(donor_rows)!=len(rows):raise ValueError('fixed donor cohort required')
    index={r['id']:i for i,r in enumerate(rows)};result=[]
    for i,(row,spec) in enumerate(zip(rows,donor_rows)):
        if set(spec)!={'id','donor_id','available','reason'} or spec['id']!=row['id'] or type(spec['available']) is not bool:
            raise ValueError('fixed pre-scoring donor schema')
        if not spec['available']:
            if spec['donor_id'] is not None:raise ValueError('unavailable donor must remain absent')
            result.append(None);continue
        j=index.get(spec['donor_id'])
        if j is None or j==i or rows[j]['domain']!=row['domain'] or states[j].clock!=states[i].clock:
            raise ValueError('donor must be another same-clock observed episode')
        if any(states[j].tensors[k].shape!=states[i].tensors[k].shape for k in states[i].tensors):
            raise ValueError('donor native state geometry differs')
        result.append(j)
    return result


def predict(model,states,aux,rows,donors):
    n=len(rows);first_lp=np.full((5,n,256),-np.log(256.),dtype=np.float32)
    generated=np.zeros((5,n,40),dtype=np.uint8);available=np.zeros((5,n),dtype=bool)
    costs={c:{'intervention_seconds':0.,'query_seconds':0.,'generation_seconds':0.} for c in CONDITIONS}
    before=R.state_digest(states);aux_before=R.state_digest([s for s in aux if s is not None])
    for ci,condition in enumerate(CONDITIONS):
        currents=[]
        tic=time.perf_counter()
        for i,state in enumerate(states):
            if condition=='binding_shuffle':current=aux[i]
            elif condition=='donor':
                j=donors[i]
                current=None if j is None else M.State098(state.arm,state.owners,state.clock,{k:t.clone() for k,t in states[j].tensors.items()})
            else:current=model.intervene(state,condition)
            currents.append(current);available[ci,i]=current is not None
        costs[condition]['intervention_seconds']=time.perf_counter()-tic
        groups=defaultdict(list)
        for i,current in enumerate(currents):
            if current is not None:groups[(current.clock,len(rows[i]['query_bytes']),rows[i]['generation_limit'])].append(i)
        for (_,_,limit),indices in groups.items():
            for start in range(0,len(indices),32):
                group=indices[start:start+32];state=combine([currents[i] for i in group]);owners=state.owners
                tokens=torch.tensor([rows[i]['query_bytes'] for i in group],dtype=torch.long)
                tic=time.perf_counter();logits,state=model(tokens,state=state,owners=owners)
                lp=logits[:,-1].log_softmax(-1);costs[condition]['query_seconds']+=time.perf_counter()-tic
                first_lp[ci,group]=lp.cpu().numpy()
                tic=time.perf_counter();token=logits[:,-1].argmax(-1)
                for j in range(limit):
                    generated[ci,group,j]=token.cpu().numpy().astype(np.uint8)
                    if j+1<limit:
                        logits,state=model(token[:,None],state=state,owners=owners);token=logits[:,-1].argmax(-1)
                costs[condition]['generation_seconds']+=time.perf_counter()-tic
    if R.state_digest(states)!=before or R.state_digest([s for s in aux if s is not None])!=aux_before:
        raise ValueError('query mutated sealed acquired state')
    return {'log_probs':first_lp,'generated_bytes':generated,'available':available,
            'lengths':np.array([r['generation_limit'] for r in rows],dtype=np.int64)},costs


def run(args):
    if args.output.exists() or (args.mode=='acquire' and args.state.exists()):raise FileExistsError('never overwrite actor evidence')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    model,checkpoint=load_checkpoint(args.checkpoint,args.sha256,args.config,args.source_commit)
    before_weights=R.model_digest(model)
    with torch.inference_mode():
        if args.mode=='acquire':
            if args.acquisition is None or args.queries is not None or args.donors is not None:raise ValueError('acquisition accepts only observed A and past-only control A')
            rows=load_actor(args.acquisition,'acquire');controls=R.read_json(args.controls) if args.controls is not None else None
            setup=time.perf_counter()-PROCESS_START
            states,aux,cost=acquire(model,rows,controls)
            cost.update(setup_seconds=setup,model_bytes=sum(t.numel()*t.element_size() for t in model.state_dict().values()),peak_rss_bytes=R.peak_rss_bytes())
            info=save_states(args.state,states,aux,rows,checkpoint,cost,R.sha256(args.acquisition),R.sha256(args.controls) if args.controls else None)
            R.write_json(args.output,{**info,'state_file_sha256':R.sha256(args.state),'optimizer_steps':0})
            return
        if args.queries is None or args.acquisition is not None or args.controls is not None or args.donors is None:
            raise ValueError('query accepts Q, acquired state, checkpoint and fixed donor IDs only')
        rows=load_actor(args.queries,'query');states,aux,info=load_states(args.state,checkpoint,rows)
        donors=donor_map(args.donors,rows,states);setup=time.perf_counter()-PROCESS_START
        arrays,cost=predict(model,states,aux,rows,donors)
        if R.model_digest(model)!=before_weights:raise ValueError('query changed frozen model weights')
        metadata={'schema':'langcontrol098.predictions.v1','checkpoint':checkpoint,'checkpoint_sha256':args.sha256,
            'conditions':CONDITIONS,'case_ids':[r['id'] for r in rows],'domains':[r['domain'] for r in rows],
            'acquisition_sha256':info['acquisition_sha256'],'query_sha256':R.sha256(args.queries),'donor_sha256':R.sha256(args.donors),
            'state_file_sha256':R.sha256(args.state),'optimizer_steps':0,'information_boundary':'Observed Q and complete acquired state only; no acquisition file, targets or judge inputs',
            'cost':{'acquisition':info['cost'],'query_setup_seconds':setup,'per_condition_costs':cost,
                    'state_payload_bytes':info['state_payload_bytes'],'auxiliary_state_payload_bytes':info['auxiliary_state_payload_bytes'],
                    'state_metadata_bytes':len(json.dumps(info,sort_keys=True).encode()),'peak_rss_bytes':R.peak_rss_bytes()}}
        with args.output.open('xb') as handle:np.savez_compressed(handle,**arrays,metadata=np.array(json.dumps(metadata,sort_keys=True,allow_nan=False)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('acquire','query'),required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--config',type=Path,default=R.HERE/'REGISTER.json');p.add_argument('--source-commit',required=True)
    p.add_argument('--acquisition',type=Path);p.add_argument('--controls',type=Path);p.add_argument('--queries',type=Path);p.add_argument('--donors',type=Path)
    p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())

if __name__=='__main__':main()
