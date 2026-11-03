"""Fixed-final full-vocabulary autoregressive evaluation, never teacher-forced answers."""
import argparse,hashlib,json,random,statistics
from pathlib import Path
import torch
from torch.nn import functional as F
from .common import ROOT,build,verify_data,digest,require,fingerprint
from .data import pools,history,query
from .model import run_chunks,tensors
def state_hash(s):
    h=hashlib.sha256()
    for n,t in sorted(tensors(s).items()):h.update(n.encode());h.update(t.detach().contiguous().numpy().tobytes())
    return h.hexdigest()
def episode(data,pool,identity,n,gap,family='dev',shift=0):
    rng=random.Random(180002+identity*1000003+n*10007+gap)
    h=history(data,pool,rng,n=n,partition=family,identity=identity,gap=0,overwrite=True,class_shift=shift)
    desired=112 if gap<=16 else 384
    lead=desired+1-9-len(h['prefix'])-gap;require(lead>=0,'evaluation prefix does not fit planned first position')
    # Source filler offsets are independent of source quotation RNG consumption.
    rr=random.Random(180992+identity*100003+n*1009+gap)
    left=rr.randrange(len(data)-max(lead,1));middle=rr.randrange(len(data)-max(gap,1))
    prefix=data[left:left+lead]+h['prefix']+data[middle:middle+gap]
    return dict(h,observed_prefix=prefix,leading_offset=left,leading_bytes=lead,filler_offset=middle,filler_bytes=gap,first_position=desired)
def decode(m,state,key):
    owner=('eval',);tokens=torch.tensor([list(query(key))]);z,new=m(tokens,state=state,owners=owner)
    first=z[0,-1].clone();answer=[]
    for j in range(4):
        token=int(z[0,-1].argmax());answer.append(token)
        if j<3:z,new=m(torch.tensor([[token]]),state=new,owners=owner)
    return bytes(answer),first
def retrieval(m,data,spec):
    pool=pools(data);rows=[];cohorts=[];conditions=spec['conditions'];counts=spec['histories']
    with torch.inference_mode():
      for cell in spec['cells']:
        n,gap=cell['keys'],cell['gap']
        for identity in range(counts):
            h=episode(data,pool,identity,n,gap);wrong=episode(data,pool,identity,n,gap,shift=1)
            require(h['keys']==wrong['keys'] and h['leading_offset']==wrong['leading_offset'] and h['filler_offset']==wrong['filler_offset'],'controlled counterfactual differs outside quotations')
            _,normal,_,_=run_chunks(m,torch.tensor([list(h['observed_prefix'])]),chunk=32,owners=('eval',),collect_balance=False)
            _,false,_,_=run_chunks(m,torch.tensor([list(wrong['observed_prefix'])]),chunk=32,owners=('eval',),collect_balance=False)
            originals=(state_hash(normal),state_hash(false))
            cohorts.append(dict(keys=n,gap=gap,identity=identity,facts=h['facts'],wrong_facts=wrong['facts'],source_prefix_sha256=hashlib.sha256(h['observed_prefix']).hexdigest(),leading_offset=h['leading_offset'],leading_bytes=h['leading_bytes'],filler_offset=h['filler_offset'],first_position=h['first_position']))
            for absent,keys in ((False,h['keys']),(True,h['absent'])):
              for j,key in enumerate(keys):
                target=b'????' if absent else h['latest'][key]
                for condition in conditions:
                    st=normal if condition in ('normal','query_swap') else false if condition=='wrong_history' else None
                    actual_key=h['keys'][(j+1)%n] if condition=='query_swap' and not absent and n>1 else key
                    pred,logits=decode(m,st,actual_key);candidates=list(b'abcdefgh')+[63];closed=candidates[int(logits[candidates].argmax())]
                    record=dict(keys=n,gap=gap,history=identity,condition=condition,key=key,queried_key=actual_key,absent=absent,answer_hex=target.hex(),prediction_hex=pred.hex(),first_top1=pred[0],closed_first=closed,
                        first_correct=int(pred[0]==target[0]),closed_first_correct=int(closed==target[0]),exact4=int(pred==target),target_log_probability=float(logits.log_softmax(-1)[target[0]]),
                        target_alphabet_mass=float(logits.softmax(-1)[candidates].sum()),first_logits=logits.tolist())
                    rows.append(record)
            require(originals==(state_hash(normal),state_hash(false)),'query mutated shared prefix')
    summary=[]
    for cell in spec['cells']:
      for condition in conditions:
        q=[r for r in rows if r['keys']==cell['keys'] and r['gap']==cell['gap'] and r['condition']==condition];p=[r for r in q if not r['absent']];a=[r for r in q if r['absent']]
        avg=lambda xs,k:statistics.mean(r[k] for r in xs)
        summary.append(dict(**cell,condition=condition,present_queries=len(p),absent_queries=len(a),first_accuracy=avg(p,'first_correct'),closed_first_accuracy=avg(p,'closed_first_correct'),exact4=avg(p,'exact4'),
            complete_maps=sum(all(r['exact4'] for r in p if r['history']==i) for i in range(counts))/counts,absent_first=avg(a,'first_correct'),absent_exact4=avg(a,'exact4'),
            present_mean_log_probability=avg(p,'target_log_probability'),target_alphabet_mass=avg(p,'target_alphabet_mass')))
    return dict(summary=summary,rows=rows,cohorts=cohorts)
def natural_dev(m,data):
    rng=random.Random(140001);indices=[rng.randrange(len(data)-129) for _ in range(64)];total=0.
    with torch.inference_mode():
        for lo in range(0,64,4):
            x=torch.tensor([list(data[i:i+129]) for i in indices[lo:lo+4]]);z,_,_,_=run_chunks(m,x[:,:-1],chunk=32,collect_balance=False)
            total+=float(F.cross_entropy(z.flatten(0,1),x[:,1:].flatten(),reduction='sum'))
    return dict(nll=total/8192,targets=8192,offsets=indices)
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    r=json.loads((a.run/'FIT_RECEIPT.json').read_text());require(r['status']=='FULL_FIT_COMPLETE_INTERNAL_UNQUALIFIED','not a complete original');ck=a.run/r['checkpoint'];require(digest(ck)==r['checkpoint_sha256'],'checkpoint differs')
    m=build(r['arm'],r['seed']);m.load_state_dict(torch.load(ck,weights_only=True,map_location='cpu')['model']);m.eval();before=fingerprint(m);data,_=verify_data();spec=json.loads((ROOT/'EVALUATION_PROTOCOL.json').read_text())
    probe=retrieval(m,data['dev'],spec);ordinary=natural_dev(m,data['dev']);require(before==fingerprint(m),'evaluation changed model')
    stages=[]
    for step in spec['stage_diagnostics']['checkpoints']:
        sm=build(r['arm'],r['seed']);sm.load_state_dict(torch.load(a.run/f'checkpoint_{step:04}.pt',weights_only=True,map_location='cpu')['model']);sm.eval()
        sub=dict(spec);sub['cells']=[dict(keys=1,gap=0)];sub['conditions']=['normal','wrong_history','reset_all']
        q=retrieval(sm,data['dev'],sub);(a.run/f'STAGE_{step:04}.json').write_text(json.dumps(q));stages.append(dict(step=step,summary=q['summary'],selection=False))
    samples=[]
    for temperature in spec['generation']['temperatures']:
        random_stream=torch.Generator().manual_seed(180100);generated=[];payload=[]
        with torch.inference_mode():
            z,st=m(torch.tensor([list(data['train'][20000:20064])]),owners=('sample',))
            for j in range(512):
                require(bool(torch.isfinite(z).all()),'nonfinite generation logits');token=torch.multinomial((z[0,-1]/temperature).softmax(-1),1,generator=random_stream);generated.append(int(token))
                z,st=m(token.reshape(1,1),state=st,owners=('sample',))
                if j+1 in (256,512):payload.append(dict(generated_bytes=j+1,state_tensor_bytes=st.state_bytes()))
        raw=bytes(generated);samples.append(dict(temperature=temperature,status='COMPLETE',generated_bytes=len(raw),hex=raw.hex(),text=raw.decode('utf8','replace'),payload=payload))
    require(before==fingerprint(m),'stage/generation changed final model')
    (a.run/'GENERATION.json').write_text(json.dumps(dict(rows=samples,checkpoint_sha256=digest(ck),weights_unchanged=True,optimizer_updates=0)))
    (a.run/'RETRIEVAL.json').write_text(json.dumps(probe))
    result=dict(arm=r['arm'],seed=r['seed'],checkpoint_sha256=digest(ck),retrieval_summary=probe['summary'],ordinary_dev=ordinary,optimizer_updates=0,weights_unchanged=True,independently_qualified=False,protected_test_read=False,torch=torch.__version__,stage_diagnostics=stages,generation_statuses=[q['status'] for q in samples])
    (a.run/'EVALUATION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
