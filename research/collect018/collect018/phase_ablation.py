"""Declared supplemental dependence checks; no refitting and no primary replacement."""
import argparse,dataclasses,json,statistics
from pathlib import Path
import torch
from .common import ROOT,build,verify_data,digest,fingerprint,require
from .data import pools
from .model import run_chunks
from .evaluate import episode,decode,state_hash

def intervene(m,state,condition):
    if condition=='normal':return state
    if condition=='phase_values_zero':return dataclasses.replace(state,phase=tuple(dataclasses.replace(p,memory=torch.zeros_like(p.memory)) for p in state.phase))
    fresh=m.start(state.owners)
    if condition=='phase_all_fresh':return dataclasses.replace(state,phase=tuple(dataclasses.replace(p,clock=state.clock) for p in fresh.phase))
    native=dataclasses.replace(fresh.native,native=dataclasses.replace(fresh.native.native,clock=state.clock),ssa=dataclasses.replace(fresh.native.ssa,clock=state.clock))
    return dataclasses.replace(state,native=native)
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    r=json.loads((a.run/'FIT_RECEIPT.json').read_text());ck=a.run/r['checkpoint'];require(digest(ck)==r['checkpoint_sha256'],'checkpoint differs')
    if r['arm'].startswith('gru_'):
        a.output.write_text(json.dumps(dict(status='INCONCLUSIVE_NOT_APPLICABLE_NO_PHASE_STATE',arm=r['arm'],seed=r['seed'],optimizer_updates=0)));return
    m=build(r['arm'],r['seed']);m.load_state_dict(torch.load(ck,weights_only=True,map_location='cpu')['model']);m.eval();before=fingerprint(m);data,_=verify_data();pool=pools(data['dev']);rows=[]
    with torch.inference_mode():
      for n,gap in ((1,0),(4,0),(4,256)):
       for identity in range(24):
        h=episode(data['dev'],pool,identity,n,gap);_,state,_,_=run_chunks(m,torch.tensor([list(h['observed_prefix'])]),chunk=32,owners=('eval',),collect_balance=False);original=state_hash(state)
        for condition in ('normal','phase_values_zero','phase_all_fresh','native_all_fresh'):
         st=intervene(m,state,condition)
         for absent,keys in ((False,h['keys']),(True,h['absent'])):
          for key in keys:
           target=b'????' if absent else h['latest'][key];answer,logits=decode(m,st,key)
           rows.append(dict(keys=n,gap=gap,history=identity,condition=condition,key=key,absent=absent,target_hex=target.hex(),prediction_hex=answer.hex(),first_correct=int(answer[0]==target[0]),exact4=int(answer==target),first_logits=logits.tolist()))
        require(state_hash(state)==original,'intervention changed caller-owned prefix')
    summary=[]
    for n,gap in ((1,0),(4,0),(4,256)):
      for condition in ('normal','phase_values_zero','phase_all_fresh','native_all_fresh'):
        q=[x for x in rows if (x['keys'],x['gap'],x['condition'])==(n,gap,condition)];pr=[x for x in q if not x['absent']];ab=[x for x in q if x['absent']]
        summary.append(dict(keys=n,gap=gap,condition=condition,present_first=statistics.mean(x['first_correct'] for x in pr),present_exact4=statistics.mean(x['exact4'] for x in pr),absent_first=statistics.mean(x['first_correct'] for x in ab)))
    require(before==fingerprint(m),'ablation changed weights');a.output.write_text(json.dumps(dict(arm=r['arm'],seed=r['seed'],summary=summary,rows=rows,checkpoint_sha256=digest(ck),weights_unchanged=True,optimizer_updates=0,primary_replacement=False)))
if __name__=='__main__':main()
