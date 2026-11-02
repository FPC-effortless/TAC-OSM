"""Frozen, zero-shot delayed retrieval and original-text carry diagnostics.

Structured episodes are generated evaluation fixtures, never training data.
No grammar parser performs writes and no correct answer enters a query input.
"""
import dataclasses,hashlib,json,random
import torch
from torch.nn import functional as F
from .common import verify_data,require

CANDIDATES=b'abcdefgh?'

def shuffle_state(state):
    if isinstance(state,torch.Tensor):return state.roll(1,0)
    if dataclasses.is_dataclass(state):
        return dataclasses.replace(state,**{f.name:shuffle_state(getattr(state,f.name)) for f in dataclasses.fields(state)})
    if isinstance(state,tuple):return tuple(shuffle_state(x) for x in state)
    return state

def state_hash(state):
    from collect011.model import named_state_tensors
    h=hashlib.sha256()
    for n,t in sorted(named_state_tensors(state).items()):h.update(n.encode());h.update(t.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()

def episode_spec():
    rng=random.Random(140001);owners=[]
    for owner in range(12):
        keys=[]
        while len(keys)<6:
            key=''.join(rng.choices('ijklmnopqrstuvwxyz',k=4))
            if key not in keys:keys.append(key)
        initial=[(owner*4+i)%8 for i in range(4)];latest=initial.copy()
        latest[0]=(latest[0]+3)%8;latest[1]=(latest[1]+5)%8
        records=[(keys[i],initial[i]) for i in range(4)]+[(keys[i],latest[i]) for i in range(2)]
        prefix='Use ? if absent.\n'+''.join(f'The mark for {key} is {chr(97+value)}.\n' for key,value in records)
        owners.append(dict(keys=keys,initial=initial,latest=latest,prefix=prefix))
    return owners

def retrieval(model):
    data,_=verify_data();episodes=episode_spec();owners=tuple(f'assay-{i}' for i in range(12));rows=[]
    for delay in (64,256,1024):
        raw=[e['prefix'].encode()+data['dev'][i*2048:i*2048+delay]+b'\n' for i,e in enumerate(episodes)]
        require(len({len(x) for x in raw})==1,'unregistered padding would change delay')
        x=torch.tensor([list(v) for v in raw]);_,state=model(x,owners=owners)
        before=state_hash(state)
        for mode in ('normal','wrong_history','reset'):
            for query in range(6):
                q=[f'The mark for {e["keys"][query]} is '.encode() for e in episodes]
                tokens=torch.tensor([list(v) for v in q])
                use=None if mode=='reset' else shuffle_state(state) if mode=='wrong_history' else state
                logits,_=model(tokens,state=use,owners=owners);last=logits[:,-1]
                scores=last[:,list(CANDIDATES)];pred=scores.argmax(-1);full=last.argmax(-1)
                target=[e['latest'][query] if query<4 else 8 for e in episodes]
                for owner in range(12):
                    y=target[owner];rows.append(dict(delay=delay,owner=owner,query=query,mode=mode,
                        kind='absent' if query>=4 else 'overwrite' if query<2 else 'unchanged',
                        target=chr(CANDIDATES[y]),prediction=chr(CANDIDATES[int(pred[owner])]),
                        full_vocab_prediction=int(full[owner]),target_log_probability=float(last[owner].log_softmax(-1)[CANDIDATES[y]]),
                        candidate_logits=scores[owner].tolist()))
                require(state_hash(state)==before,'query branch mutated shared prefill state')
    summaries=[]
    for delay in (64,256,1024):
        for mode in ('normal','wrong_history','reset'):
            subset=[r for r in rows if r['delay']==delay and r['mode']==mode]
            present=[r for r in subset if r['kind']!='absent'];absent=[r for r in subset if r['kind']=='absent'];over=[r for r in subset if r['kind']=='overwrite']
            metric=lambda rr:sum(r['prediction']==r['target'] for r in rr)/len(rr)
            summaries.append(dict(delay=delay,mode=mode,present_accuracy=metric(present),absent_marker_accuracy=metric(absent),
                overwrite_accuracy=metric(over),complete_present_map=sum(all(r['prediction']==r['target'] for r in present if r['owner']==o) for o in range(12))/12,
                absent_false_accept=sum(r['prediction']!='?' for r in absent)/len(absent),present_examples=48,absent_examples=24))
    return dict(status='ZERO_SHOT_GENERATED_DIAGNOSTIC_NOT_TRAINED_RETRIEVAL_TASK',episodes=episodes,rows=rows,summaries=summaries,
        caution='12 shared histories, reused across delays/models/seeds; nine-way closed-set decisions, no tuned rejection threshold. Wrong-history permutes all carried tensors; reset removes all history. These models were not trained on this instruction grammar. No production absent-key or exact-memory guarantee.')

def continuation(model):
    from collect012.train import batch
    data,_=verify_data();g=torch.Generator().manual_seed(140099);inputs=[]
    for _ in range(2):inputs.append(batch(data['dev'],g,4,512,'cpu')[:2])
    rows=[]
    for mode in ('carry','reset','wrong_history'):
        sums=[0.,0.,0.,0.]
        for x,y in inputs:
            state=None;owners=tuple(f'carry-{i}' for i in range(4))
            for j in range(4):
                old=None if mode=='reset' else shuffle_state(state) if mode=='wrong_history' and state is not None else state
                logits,state=model(x[:,j*128:(j+1)*128],state=old,owners=owners)
                sums[j]+=float(F.cross_entropy(logits.flatten(0,1),y[:,j*128:(j+1)*128].flatten(),reduction='sum'))
        rows.append(dict(mode=mode,chunk_nll=[v/1024 for v in sums],targets_per_chunk=1024))
    return dict(status='ORIGINAL_EXPOSED_DEV_CONTINUATION_NOT_FACT_RETRIEVAL',windows=8,window_length=512,rows=rows)
