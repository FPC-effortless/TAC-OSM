"""Source-informed phase query sensitivity and admission audit, no optimization.

The first layer receives normalized byte embeddings before its own recurrent
outputs are mixed. Its final-space address is therefore context-free. Higher
layers can contextualize addresses, and query writes can also change the state;
first-layer invariance does not imply the full model ignores query identities.
"""
import argparse,json,math,random
from pathlib import Path
import torch
from torch.nn import functional as F
from .common import ROOT,build,digest,verify_data,require,fingerprint
from .data import pools,history,query
from .model import run_chunks
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--step',type=int,default=512);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    r=json.loads((a.run/'FIT_RECEIPT.json').read_text());ck=a.run/f'checkpoint_{a.step:04}.pt';m=build(r['arm'],r['seed']);m.load_state_dict(torch.load(ck,weights_only=True,map_location='cpu')['model']);m.eval();before=fingerprint(m)
    if r['arm']=='current_only':
        a.output.write_text(json.dumps(dict(arm=r['arm'],seed=r['seed'],status='NOT_APPLICABLE_NO_PHASE_ADDRESSES',optimizer_updates=0),indent=2));return
    data,_=verify_data();d=data['dev'];pool=pools(d);captured={i:[] for i in range(4)};handles=[];rows=[]
    def make_hook(layer):
        def capture(module,args):
            x=args[0];shape=(*x.shape[:2],module.heads,module.key_dim//2)
            content=F.linear(x,module.content.weight).reshape(shape).tanh();role=F.linear(x,module.role.weight).reshape(shape).tanh().roll(1,-1)
            theta=math.pi*(content+role);key=torch.stack((theta.cos(),theta.sin()),-1).flatten(-2)/math.sqrt(module.key_dim//2)
            router=F.linear(x,module.router.weight,module.router.bias);route=F.one_hot(router.argmax(-1),module.heads).to(x.dtype)
            admission=F.linear(x,module.admission.weight,module.admission.bias).sigmoid();selected=(admission*route).sum(-1)
            captured[layer].append(dict(key=(key*route[...,None]).flatten(-2).detach(),admission=selected.detach()))
        return capture
    for i,b in enumerate(m.core.blocks):handles.append(b.phase_memory.register_forward_pre_hook(make_hook(i)))
    with torch.inference_mode():
      for lo in range(0,24,4):
        hs=[history(d,pool,random.Random(160002+i),n=4,partition='dev',identity=i,gap=256,overwrite=True) for i in range(lo,lo+4)]
        owners=tuple(f'address-{i}' for i in range(lo,lo+4));x=torch.tensor([list(h['prefix']) for h in hs])
        for i in range(4):captured[i]=[]
        _,state,_,_=run_chunks(m,x,owners=owners,collect_balance=False)
        admission={i:torch.cat([z['admission'] for z in captured[i]],1) for i in range(4)}
        queries={i:[] for i in range(4)}
        for j in range(4):
            for i in range(4):captured[i]=[]
            m(torch.tensor([list(query(h['keys'][j])) for h in hs]),state=state,owners=owners)
            for i in range(4):queries[i].append(captured[i][-1]['key'][:,-1])
        for i in range(4):
            keys=torch.stack(queries[i],1)
            for b,h in enumerate(hs):
                q=keys[b];cos=q@q.T;pair=[float(cos[u,v]) for u in range(4) for v in range(u+1,4)]
                differences=[float((q[u]-q[v]).abs().max()) for u in range(4) for v in range(u+1,4)]
                quote_positions=[j*14+k for j in range(len(h['facts'])) for k in range(5,13)]
                filler_start=len(h['facts'])*14+1
                rows.append(dict(history=lo+b,layer=i+1,query_pair_cosines=pair,query_pair_max_abs_differences=differences,
                    query_vectors=q.tolist(),mean_quote_admission=float(admission[i][b,quote_positions].mean()),
                    mean_filler_admission=float(admission[i][b,filler_start:].mean())))
    for h in handles:h.remove()
    require(before==fingerprint(m),'address audit changed weights')
    summary=[]
    for i in range(1,5):
        z=[q for q in rows if q['layer']==i];pairs=[v for q in z for v in q['query_pair_cosines']];diff=[v for q in z for v in q['query_pair_max_abs_differences']]
        summary.append(dict(layer=i,mean_query_pair_cosine=sum(pairs)/len(pairs),max_query_pair_difference=max(diff),
            equal_query_pair_fraction=sum(v==0. for v in diff)/len(diff),mean_quote_admission=sum(q['mean_quote_admission'] for q in z)/len(z),
            mean_filler_admission=sum(q['mean_filler_admission'] for q in z)/len(z)))
    a.output.write_text(json.dumps(dict(arm=r['arm'],seed=r['seed'],checkpoint_step=a.step,checkpoint_sha256=digest(ck),summary=summary,rows=rows,
        optimizer_updates=0,weights_unchanged=True,scope='Phase key sensitivity and selected gate only; not a causal source-to-fact retrieval proof or all-model query independence'),indent=2))
    print(json.dumps(dict(arm=r['arm'],seed=r['seed'],summary=summary),indent=2))
if __name__=='__main__':main()
