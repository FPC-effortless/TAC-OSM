"""Separate source-pinned acquisition/query actor; no judge/targets inputs."""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
_spec=importlib.util.spec_from_file_location('langtrain099_common',Path(__file__).with_name('common.py'))
C=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=C;_spec.loader.exec_module(C)
M=C.load('model')
CONDITIONS=(*C.CONDITIONS,'binding_shuffle')

def load_checkpoint(path,expected,source_commit):
    if C.sha(path)!=expected:raise ValueError('original checkpoint archive differs')
    source=C.source_hashes(source_commit);saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('schema')!='langtrain099.checkpoint.v1':raise ValueError('actual original099 archive required')
    meta=saved['metadata'];reg=C.read(C.HERE/'REGISTER.json')
    if (meta.get('source_commit')!=source_commit or meta.get('source_hashes')!=source
            or meta.get('seed') not in C.SEEDS or (meta.get('status'),meta.get('optimizer_steps')) not in
                (('INITIAL_BEFORE_DEV',0),('SEALED_BEFORE_DEV',reg['training']['steps']))):
        raise ValueError('source/seed/complete-training binding differs')
    model=C.make(meta['seed']).eval();model.load_state_dict(saved['model'],strict=True)
    if model.counts()!=meta['counts'] or C.state_digest(model)!=meta['model_sha256']:
        raise ValueError('actual10M source weights/counts differ')
    return model,meta

def load_actor(path,mode):
    rows=C.read_json(path)
    if not isinstance(rows,list) or not 1<=len(rows)<=128:raise ValueError('bounded actor-only case list required')
    keys={'id','domain','acquisition_bytes'} if mode=='acquire' else {'id','domain','query_bytes','generation_limit'}
    seen=set()
    for row in rows:
        if not isinstance(row,dict) or set(row)!=keys:raise ValueError('actor schema has missing or forbidden fields')
        if not isinstance(row['id'],str) or not row['id'] or row['id'] in seen or row['domain'] not in ('lexical','arc'):
            raise ValueError('invalid or duplicate actor case')
        seen.add(row['id'])
        C.bytes_list(row['acquisition_bytes' if mode=='acquire' else 'query_bytes'],'observations',maximum=272)
        if mode=='query' and row['generation_limit']!={'lexical':4,'arc':40}[row['domain']]:
            raise ValueError('generation budget must be fixed independently of target length')
    return rows


def pack(state):
    return {'arm':state.arm,'owners':state.owners,'clock':state.clock,'tensors':C.cpu_tree(state.tensors)}


def unpack(value):
    if not isinstance(value,dict) or set(value)!={'arm','owners','clock','tensors'}:raise ValueError('state schema')
    if value['arm'] not in C.FAMILIES or type(value['clock']) is not int or not 1<=value['clock']<=192:
        raise ValueError('legitimate acquisition clock required')
    if not isinstance(value['tensors'],dict) or not value['tensors']:raise ValueError('complete state required')
    for tensor in value['tensors'].values():
        if not isinstance(tensor,torch.Tensor) or tensor.dtype!=torch.float32 or tensor.ndim<1 or tensor.shape[0]!=1 or not bool(torch.isfinite(tensor).all()):
            raise ValueError('finite native state tensors required')
    return M.State099(tuple(value['owners']),value['clock'],value['tensors'])


def combine(states):
    first=states[0]
    if any(s.arm!=first.arm or s.clock!=first.clock or set(s.tensors)!=set(first.tensors) for s in states):raise ValueError('combine requires identical native geometry and clock')
    return M.State099(tuple(s.owners[0] for s in states),first.clock,
        {key:torch.cat([s.tensors[key] for s in states],dim=0) for key in first.tensors})


def split(state):
    return [M.State099((owner,),state.clock,{k:t[i:i+1].clone() for k,t in state.tensors.items()}) for i,owner in enumerate(state.owners)]


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
            C.bytes_list(c['acquisition_bytes'],'past-only binding observations',maximum=192)
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
    info={'schema':'langtrain099.acquisition-state.v1','checkpoint':metadata,
          'case_ids':[r['id'] for r in rows],'domains':[r['domain'] for r in rows],
          'acquisition_sha256':acquisition_sha,'control_sha256':control_sha,
          'state_digest':C.native_digest(states),'aux_state_digest':C.native_digest([s for s in aux if s is not None]),
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
    if (info.get('schema')!='langtrain099.acquisition-state.v1' or info['checkpoint']!=checkpoint
            or info['case_ids']!=[r['id'] for r in rows] or info['domains']!=[r['domain'] for r in rows]
            or len(saved['states'])!=len(rows) or len(saved['auxiliary_states'])!=len(rows)):
        raise ValueError('state archive source/case binding differs')
    states=[unpack(s) for s in saved['states']]
    aux=[unpack(s) if s is not None else None for s in saved['auxiliary_states']]
    for i,state in enumerate(states):
        if state.arm!=checkpoint['family'] or state.owners!=(f'case-{i}',):raise ValueError('state owner or model family differs')
        if aux[i] is not None and (aux[i].owners!=state.owners or aux[i].arm!=state.arm or aux[i].clock!=state.clock):
            raise ValueError('binding control changed state ownership/clock')
    if (C.native_digest(states)!=info['state_digest'] or C.native_digest([s for s in aux if s is not None])!=info['aux_state_digest']
            or [s is not None for s in aux]!=info['aux_available']):raise ValueError('saved state digest differs')
    return states,aux,info


def donor_map(path,rows,states):
    donor_rows=C.read_json(path)
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
    before=C.native_digest(states);aux_before=C.native_digest([s for s in aux if s is not None])
    for ci,condition in enumerate(CONDITIONS):
        currents=[]
        tic=time.perf_counter()
        for i,state in enumerate(states):
            if condition=='binding_shuffle':current=aux[i]
            elif condition=='donor':
                j=donors[i]
                current=None if j is None else M.State099(state.owners,state.clock,{k:t.clone() for k,t in states[j].tensors.items()})
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
    if C.native_digest(states)!=before or C.native_digest([s for s in aux if s is not None])!=aux_before:
        raise ValueError('query mutated sealed acquired state')
    return {'log_probs':first_lp,'generated_bytes':generated,'available':available,
            'lengths':np.array([r['generation_limit'] for r in rows],dtype=np.int64)},costs

def run(args):
    C.setup(9901)
    if args.output.exists():raise FileExistsError('retain original predictions')
    model,checkpoint=load_checkpoint(args.checkpoint,args.sha256,args.source_commit)
    before=C.state_digest(model)
    with torch.inference_mode():
        if args.mode=='acquire':
            if args.acquisition is None or args.queries is not None or args.donors is not None:raise ValueError('only past acquisition observations accepted')
            if args.state.exists():raise FileExistsError('retain original acquired state')
            rows=load_actor(args.acquisition,'acquire');controls=C.read(args.controls)
            states,aux,cost=acquire(model,rows,controls)
            info=save_states(args.state,states,aux,rows,checkpoint,cost,C.sha(args.acquisition),C.sha(args.controls))
            C.save(args.output,info)
        else:
            if args.queries is None or args.acquisition is not None or args.controls is not None or args.donors is None:
                raise ValueError('queries accept saved state and observed query only')
            rows=load_actor(args.queries,'query');states,aux,info=load_states(args.state,checkpoint,rows)
            donors=donor_map(args.donors,rows,states);arrays,cost=predict(model,states,aux,rows,donors)
            metadata={'schema':'langtrain099.predictions.v1','checkpoint':checkpoint,'checkpoint_sha256':args.sha256,
                      'conditions':CONDITIONS,'case_ids':[r['id'] for r in rows],'domains':[r['domain'] for r in rows],
                      'acquisition_sha256':info['acquisition_sha256'],'query_sha256':C.sha(args.queries),
                      'state_file_sha256':C.sha(args.state),'source_commit':args.source_commit,
                      'state_payload_bytes':info['state_payload_bytes'],'cost':cost,'optimizer_steps':0}
            with args.output.open('xb') as f:np.savez_compressed(f,**arrays,metadata=np.array(json.dumps(metadata,sort_keys=True,allow_nan=False)))
    if C.state_digest(model)!=before:raise ValueError('evaluation mutated original weights')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('acquire','query'),required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--source-commit',required=True)
    p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--acquisition',type=Path);p.add_argument('--queries',type=Path);p.add_argument('--controls',type=Path);p.add_argument('--donors',type=Path)
    run(p.parse_args())

