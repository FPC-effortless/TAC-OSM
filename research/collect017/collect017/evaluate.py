"""Frozen-checkpoint ordinary text and source-grounded held-out copy assays."""
import argparse,hashlib,json,random
from dataclasses import replace
from pathlib import Path
import torch
from torch.nn import functional as F
from .common import ROOT,build,digest,require,verify_data,fingerprint
from collect016.data import pools,history,query,LETTERS
from .model import run_chunks
from collect011.model import named_state_tensors,CollectState
from collect014.train import evaluate as short_dev,batch
def state_hash(state):
    h=hashlib.sha256()
    for n,v in sorted(named_state_tensors(state).items()):h.update(n.encode());h.update(v.detach().contiguous().numpy().tobytes())
    return h.hexdigest()
def phase_reset(m,s):
    return CollectState(s.native,tuple(replace(b.phase_memory.start(s.owners),clock=s.clock) for b in m.core.blocks),s.signature)
def decode(m,state,owners,text):
    x=torch.tensor([list(t) for t in text],dtype=torch.long)
    logits,branch=m(x,state=state,owners=owners);first=logits[:,-1].clone();out=[]
    for i in range(4):
        token=logits[:,-1].argmax(-1);out.append(token)
        if i<3:logits,branch=m(token[:,None],state=branch,owners=owners)
    return first,torch.stack(out,1)
def retrieval(m,data,protocol):
    pool=pools(data);rows=[];cohorts=[];oracle=[];before=fingerprint(m)
    for n in protocol['key_counts']:
      for gap in protocol['gaps']:
       for lo in range(0,protocol['histories'],4):
        hs=[history(data,pool,random.Random(160002+i),n=n,partition='dev',identity=i,gap=gap,overwrite=True) for i in range(lo,lo+4)]
        ws=[history(data,pool,random.Random(160002+i),n=n,partition='dev',identity=i,gap=gap,overwrite=True,class_shift=1) for i in range(lo,lo+4)]
        owners=tuple(f'dev-{i}' for i in range(lo,lo+4));x=torch.tensor([list(h['prefix']) for h in hs]);wx=torch.tensor([list(h['prefix']) for h in ws])
        _,normal,_,_=run_chunks(m,x,owners=owners,collect_balance=False)
        _,wrong,_,_=run_chunks(m,wx,owners=owners,collect_balance=False)
        original=state_hash(normal)
        states={'normal':normal,'wrong_history':wrong,'reset_all':m.start(owners),'query_swap':normal}
        for j in range(n+2):
            keys=[(h['keys']+h['absent'])[j] for h in hs]
            answers=[h['latest'].get(k,b'????') for h,k in zip(hs,keys)]
            for condition,s in states.items():
                qs=[query(h['keys'][(j+1)%n]) if condition=='query_swap' and j<n else query(k) for h,k in zip(hs,keys)]
                first,predicted=decode(m,s,owners,qs)
                closed=first[:,list(LETTERS)+[63]].argmax(-1)
                for b,(h,k,answer) in enumerate(zip(hs,keys,answers)):
                    pred=bytes(predicted[b].tolist());r=dict(history=lo+b,keys=n,gap=gap,condition=condition,key=k,absent=j>=n,
                        answer_hex=answer.hex(),prediction_hex=pred.hex(),first_top1=int(first[b].argmax()),
                        closed_first=int((list(LETTERS)+[63])[int(closed[b])]),first_correct=int(pred[0]==answer[0]),
                        closed_first_correct=int((list(LETTERS)+[63])[int(closed[b])]==answer[0]),exact4=int(pred==answer),
                        target_log_probability=float(first[b].log_softmax(-1)[answer[0]]))
                    if n==4 and gap==256 and condition in ('normal','wrong_history','reset_all'):r['first_logits']=first[b].tolist()
                    rows.append(r)
            require(state_hash(normal)==original,'query branch mutated shared prefix state')
        for i,h in enumerate(hs):
            for fact in h['facts']:
                off=fact['source_offset'];require(data[off:off+8].decode('ascii')==fact['quote'],'source quotation differs')
            cohorts.append(dict(history=lo+i,keys=n,gap=gap,facts=h['facts'],filler_offset=h['filler_offset'],prefix_sha256=hashlib.sha256(h['prefix']).hexdigest(),held_out_key_family=True))
            oracle.append(dict(history=lo+i,keys=n,gap=gap,present_exact4=1.,absent_exact4=1.))
    summary=[]
    for n in protocol['key_counts']:
      for gap in protocol['gaps']:
       for c in ('normal','wrong_history','reset_all','query_swap'):
        cell=[r for r in rows if r['keys']==n and r['gap']==gap and r['condition']==c]
        p=[r for r in cell if not r['absent']];a=[r for r in cell if r['absent']]
        mean=lambda z,k:sum(r[k] for r in z)/len(z)
        maps=sum(all(r['exact4'] for r in p if r['history']==i) for i in range(protocol['histories']))/protocol['histories']
        firstmaps=sum(all(r['first_correct'] for r in p if r['history']==i) for i in range(protocol['histories']))/protocol['histories']
        summary.append(dict(keys=n,gap=gap,condition=c,present_queries=len(p),absent_queries=len(a),first_accuracy=mean(p,'first_correct'),
            closed_first_accuracy=mean(p,'closed_first_correct'),exact4=mean(p,'exact4'),complete_maps=maps,first_complete_maps=firstmaps,
            absent_first=mean(a,'first_correct'),absent_exact4=mean(a,'exact4'),present_mean_log_probability=mean(p,'target_log_probability')))
    require(fingerprint(m)==before,'evaluation changed weights')
    return dict(summary=summary,rows=rows,cohorts=cohorts,oracle_unmatched_parser=oracle,weights_unchanged=True,
        unique_histories=protocol['histories']*len(protocol['key_counts'])*len(protocol['gaps']),
        caution='Generated source-copy interface; ordinary-text loss reported separately. Histories reused across models/conditions, not independent repetitions.')
def continuation(m,data):
    g=torch.Generator().manual_seed(160099);out=[]
    for _ in range(4):
        x,y,indices=batch(data,g,4,512,'cpu');owners=tuple(f'text-{i}' for i in range(4))
        for mode in ('carry','reset_128'):
            state=None;loss=0.
            for lo in range(0,512,128):
                if mode=='reset_128':state=None
                z,state=m(x[:,lo:lo+128],state=state,owners=owners)
                if lo:loss+=float(F.cross_entropy(z.flatten(0,1),y[:,lo:lo+128].flatten(),reduction='sum'))
            out.append(dict(mode=mode,indices=indices,nll=loss/(4*384),targets=1536))
    return out
def evaluate(folder,out):
    torch.set_num_threads(1);r=json.loads((folder/'FIT_RECEIPT.json').read_text());require(r['status']=='FULL_FIT_COMPLETE_INTERNAL_UNQUALIFIED','not a completed full fit')
    ck=folder/r['checkpoint'];require(digest(ck)==r['checkpoint_sha256'],'checkpoint changed')
    saved=torch.load(ck,weights_only=True,map_location='cpu');m=build(r['arm'],r['seed']);m.load_state_dict(saved['model']);m.eval();data,_=verify_data()
    replay=torch.load(folder/'REPLAY.pt',weights_only=True,map_location='cpu')
    with torch.inference_mode():
        z,_,_,_=run_chunks(m,replay['tokens']);error=float((z-replay['logits']).abs().max());require(error<=2e-5,'checkpoint replay exceeds declared cross-runtime tolerance')
        ordinary=short_dev(m,data['dev'],'cpu',64);cont=continuation(m,data['dev'])
        protocol=json.loads((ROOT/'EVALUATION_PROTOCOL.json').read_text());probe=retrieval(m,data['dev'],protocol)
    out.mkdir(parents=True,exist_ok=True);(out/'RETRIEVAL.json').write_text(json.dumps(probe,indent=2))
    result=dict(arm=r['arm'],seed=r['seed'],checkpoint_sha256=digest(ck),torch=torch.__version__,replay_max_abs_error=error,
        ordinary_dev=ordinary,continuation=cont,retrieval_summary=probe['summary'],optimizer_updates=0,weights_unchanged=True)
    (out/'EVALUATION.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='retrieval_summary'},indent=2))
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();evaluate(a.run,a.output)
if __name__=='__main__':main()
