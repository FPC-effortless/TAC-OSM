"""Separated general-development acquisition/query worker; no gold/test input.

All outputs are fixed-budget greedy bytes from unchanged frozen models. These
tasks test selected reasoning/transfer/execution capabilities, not certified
general intelligence. Query processes never receive acquisition text.
"""
import time
PROCESS_START = time.perf_counter()
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys

import numpy as np
import torch


def _adapters():
    name = 'langpersist096_general_adapters'
    path = Path(__file__).with_name('adapters.py').resolve()
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path:
            raise ValueError('adapter source collision')
        return module
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name,None)
        raise
    return module


ADAPTERS = _adapters()
CONDITIONS = ('carry','reset')
DOMAINS = ('arc','gsm8k','humaneval')


def load_cases(path, mode):
    if Path(path).stat().st_size > 2*1024*1024:
        raise ValueError('oversized actor-only file')
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows,list) or not 1 <= len(rows) <= 128:
        raise ValueError('bounded nonempty actor task list required')
    required = {'task_id','domain', 'acquisition_bytes'} if mode == 'acquire' else {'task_id','domain','query_bytes','generation_limit'}
    seen=set()
    for row in rows:
        if not isinstance(row,dict) or set(row) != required:
            raise ValueError('actor schema includes denied or missing fields')
        if row['domain'] not in DOMAINS or not isinstance(row['task_id'],str) or not row['task_id'] or len(row['task_id'].encode()) > 128 or row['task_id'] in seen:
            raise ValueError('invalid or duplicate actor task')
        seen.add(row['task_id'])
        values=row['acquisition_bytes' if mode == 'acquire' else 'query_bytes']
        if not isinstance(values,list) or not 1 <= len(values) <= 1024 or any(type(x) is not int or not 0 <= x <= 255 for x in values):
            raise ValueError('only bounded observed byte lists allowed')
        if mode == 'query' and (type(row['generation_limit']) is not int or not 1 <= row['generation_limit'] <= 128):
            raise ValueError('generation budget must be fixed before scoring')
    return rows


def _digest(states):
    digest=hashlib.sha256()
    for state in states:
        digest.update(json.dumps({'arm':state.arm,'owners':state.owners,'clock':state.clock,
            'checkpoint':state.checkpoint_sha256,'implementation':state.implementation_sha256},sort_keys=True).encode())
        digest.update(ADAPTERS._tensor_digest(state.tensors).encode())
    return digest.hexdigest()


def acquire_cases(adapter, rows):
    states=[]
    for i,row in enumerate(rows):
        observed=torch.tensor([row['acquisition_bytes']],dtype=torch.long)
        states.append(adapter.observe(adapter.start((f'general-{i}',)),observed)[1])
    return states


def query_cases(adapter, states, rows):
    if len(states) != len(rows):
        raise ValueError('query/state case mismatch')
    generated=np.zeros((2,len(rows),128),dtype=np.uint8)
    query_seconds=generation_seconds=0.
    per_condition={c:{'query_seconds':0.,'generation_seconds':0.} for c in CONDITIONS}
    before=_digest(states)
    for i,(state,row) in enumerate(zip(states,rows)):
        adapter.validate(state)
        for ci,condition in enumerate(CONDITIONS):
            started=time.perf_counter()
            current=ADAPTERS.intervene(state,condition)
            logits,current=adapter.observe(current,torch.tensor([row['query_bytes']],dtype=torch.long))
            elapsed=time.perf_counter()-started
            query_seconds += elapsed; per_condition[condition]['query_seconds'] += elapsed
            started=time.perf_counter()
            token=logits.argmax(-1)
            generated[ci,i,0]=int(token[0])
            for j in range(1,row['generation_limit']):
                logits,current=adapter.step(current,token)
                token=logits.argmax(-1)
                generated[ci,i,j]=int(token[0])
            elapsed=time.perf_counter()-started
            generation_seconds += elapsed; per_condition[condition]['generation_seconds'] += elapsed
    if _digest(states) != before:
        raise ValueError('query mutated committed acquired states')
    return generated,{'query_seconds':query_seconds,'generation_seconds':generation_seconds,
                      'per_condition_costs':per_condition}


def save_states(path, adapter, states, rows, cost, acquisition_sha):
    metadata={'schema':'langpersist096.general-state.v1','arm':adapter.arm,
        'checkpoint_sha256':adapter.checkpoint,'implementation_sha256':adapter.implementation,
        'acquisition_sha256':acquisition_sha,'task_ids':[r['task_id'] for r in rows],
        'domains':[r['domain'] for r in rows],'clocks':[s.clock for s in states],
        'owners':[list(s.owners) for s in states], 'state_digest':_digest(states),'cost':dict(cost)}
    metadata['cost']['state_payload_bytes']=sum(s.tensor_bytes() for s in states)
    metadata['cost']['serialized_metadata_bytes']=0
    for _ in range(8):
        encoded=json.dumps(metadata,sort_keys=True,separators=(',',':'),allow_nan=False)
        size=len(encoded.encode())
        if size == metadata['cost']['serialized_metadata_bytes']:
            break
        metadata['cost']['serialized_metadata_bytes']=size
    else:
        raise ValueError('state accounting failed to converge')
    with Path(path).open('xb') as handle:
        torch.save({'metadata':encoded,'tensors':[s.tensors for s in states]},handle)
    return metadata


def load_states(path, adapter, query_rows):
    if Path(path).stat().st_size > 4*1024*1024:
        raise ValueError('oversized general state')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(saved,dict) or set(saved) != {'metadata','tensors'}:
        raise ValueError('general state schema')
    metadata=json.loads(saved['metadata'])
    if (metadata['schema'],metadata['arm'],metadata['checkpoint_sha256'],metadata['implementation_sha256']) != ('langpersist096.general-state.v1',adapter.arm,adapter.checkpoint,adapter.implementation):
        raise ValueError('general state source binding')
    if metadata['task_ids'] != [r['task_id'] for r in query_rows] or metadata['domains'] != [r['domain'] for r in query_rows] or len(saved['tensors']) != len(query_rows):
        raise ValueError('query task order/binding differs')
    if len(metadata['clocks']) != len(query_rows) or len(metadata['owners']) != len(query_rows):
        raise ValueError('state metadata length')
    states=[]
    for i,tensors in enumerate(saved['tensors']):
        if metadata['owners'][i] != [f'general-{i}'] or type(metadata['clocks'][i]) is not int or not 1 <= metadata['clocks'][i] <= 1024:
            raise ValueError('general state owner/clock')
        state=ADAPTERS.State096(adapter.arm,tuple(metadata['owners'][i]),metadata['clocks'][i],adapter.checkpoint,adapter.implementation,tensors)
        adapter.validate(state); states.append(state)
    if _digest(states) != metadata['state_digest'] or sum(s.tensor_bytes() for s in states) != metadata['cost']['state_payload_bytes'] or len(saved['metadata'].encode()) != metadata['cost']['serialized_metadata_bytes']:
        raise ValueError('state digest/accounting differs')
    return states,metadata


def peak_rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024


def run(args):
    torch.set_num_threads(1)
    model=ADAPTERS.make_model(args.arm,args.checkpoint,args.sha256)
    adapter=ADAPTERS.FrozenAdapter096(model,args.arm)
    model_bytes=sum(t.numel()*t.element_size() for t in model.state_dict().values())
    if args.mode == 'acquire':
        if args.acquisition is None or args.queries is not None:
            raise ValueError('acquisition phase receives only acquisition actor file')
        rows=load_cases(args.acquisition,'acquire')
        setup=time.perf_counter()-PROCESS_START
        started=time.perf_counter(); states=acquire_cases(adapter,rows)
        costs={'setup_seconds':setup,'acquisition_seconds':time.perf_counter()-started,
               'query_seconds':0.,'generation_seconds':0.,'model_bytes':model_bytes,
               'peak_rss_bytes':peak_rss()}
        meta=save_states(args.state,adapter,states,rows,costs,ADAPTERS.sha256(args.acquisition))
        with Path(args.output).open('x') as handle:
            json.dump({**meta,'state_file_sha256':ADAPTERS.sha256(args.state),'optimizer_steps':0},handle,sort_keys=True,allow_nan=False)
        return
    if args.queries is None or args.acquisition is not None:
        raise ValueError('query phase cannot receive acquisition input')
    rows=load_cases(args.queries,'query')
    states,saved=load_states(args.state,adapter,rows)
    cost=dict(saved['cost']); cost['setup_seconds'] += time.perf_counter()-PROCESS_START
    generated,measured=query_cases(adapter,states,rows)
    per_condition=measured.pop('per_condition_costs')
    cost.update(measured); cost['peak_rss_bytes']=max(cost['peak_rss_bytes'],peak_rss())
    per_condition={name:{**cost,**times} for name,times in per_condition.items()}
    metadata={'schema':'langpersist096.general-predictions.v1','arm':args.arm,'conditions':CONDITIONS,
        'checkpoint_sha256':args.sha256,'implementation_sha256':adapter.implementation,
        'acquisition_sha256':saved['acquisition_sha256'],'query_sha256':ADAPTERS.sha256(args.queries),
        'task_ids':saved['task_ids'],'domains':saved['domains'],'cost':cost,
        'per_condition_costs':per_condition,
        'cost_basis':'flat totals measure both interventions; each policy ledger charges common setup/acquisition once and is not additive',
        'optimizer_steps':0,'information_boundary':'queries and source-bound acquired states only; no gold or raw acquisition'}
    with Path(args.output).open('xb') as handle:
        np.savez_compressed(handle,generated_bytes=generated,
            lengths=np.array([r['generation_limit'] for r in rows],dtype=np.int64),
            metadata=np.array(json.dumps(metadata,sort_keys=True,allow_nan=False)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('acquire','query'),required=True)
    parser.add_argument('--arm',choices=('tac','gru','ssm'),required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--acquisition',type=Path)
    parser.add_argument('--queries',type=Path)
    parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or (args.mode=='acquire' and args.state.exists()):
        raise FileExistsError('do not overwrite research artifacts')
    run(args)


if __name__=='__main__':
    main()
