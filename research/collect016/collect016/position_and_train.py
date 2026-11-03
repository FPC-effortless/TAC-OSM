"""Post-null task-position and recent-TRAIN fit checks; never replace primary."""
import argparse,importlib,json,sys
from pathlib import Path
from unittest.mock import patch
import torch
from .common import ROOT,verify_data,digest,require,fingerprint
from .data import pools,training_batch
from .model import run_chunks
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--attention',action='store_true');a=p.parse_args();torch.set_num_threads(1)
    if a.attention:
        for q in (ROOT.parent/'collect017',ROOT.parent/'collect017_work'):
            if (q/'collect017/common.py').is_file():sys.path.insert(0,str(q));break
        c=importlib.import_module('collect017.common');ev=importlib.import_module('collect017.evaluate')
    else:c=importlib.import_module('collect016.common');ev=importlib.import_module('collect016.evaluate')
    r=json.loads((a.run/'FIT_RECEIPT.json').read_text());ck=a.run/r['checkpoint'];require(digest(ck)==r['checkpoint_sha256'],'checkpoint differs')
    m=c.build(r['arm'],r['seed']);m.load_state_dict(torch.load(ck,weights_only=True,map_location='cpu')['model']);m.eval();before=fingerprint(m);data,_=verify_data();pool=pools(data['train']);recent=[]
    with torch.inference_mode():
      for step in range(497,513):
        x,y,mask,meta=training_batch(data['train'],pool,seed=r['seed'],step=step)
        z,_,_,_=run_chunks(m,x,collect_balance=False);at=int(mask[1].nonzero()[0]);first=z[1,at];target=int(y[1,at]);candidates=list(b'abcdefgh')+[63]
        recent.append(dict(step=step,key=meta['key'],absent=meta['absent'],prediction_index=at,target=target,prediction=int(first.argmax()),
            closed_prediction=candidates[int(first[candidates].argmax())],first_correct=int(first.argmax())==target,
            first_nll=float(-first.log_softmax(-1)[target]),remaining_answer_nll=float(torch.nn.functional.cross_entropy(z[1,at+1:at+4],y[1,at+1:at+4]))))
      position=[];original=ev.history
      for key_family in ('train','dev'):
        def generate(*args,**kwargs):kwargs['partition']=key_family;return original(*args,**kwargs)
        with patch.object(ev,'history',generate):q=ev.retrieval(m,data['dev'],dict(histories=24,key_counts=[4],gaps=[287]))
        for h in q['cohorts']:h.update(held_out_key_family=key_family=='dev',quote_source='DEV',key_family=key_family,first_answer_position=380)
        position.append(dict(key_family=key_family,**q))
    require(before==fingerprint(m),'post-hoc check changed weights')
    out=dict(status='POST_NULL_DIAGNOSTIC_COMPLETE',arm=r['arm'],seed=r['seed'],checkpoint_sha256=digest(ck),torch=torch.__version__,
        recent_train=recent,position_matched=position,optimizer_updates=0,weights_unchanged=True,
        caution='Declared after C16 nulls; exact primary remains unchanged. Fixed final checkpoint only. Recent TRAIN is not generalization; position/delay are not independently randomized.')
    a.output.write_text(json.dumps(out,indent=2));print(json.dumps(dict(arm=r['arm'],seed=r['seed'],recent_train_first_accuracy=sum(v['first_correct'] for v in recent)/16,
        matched=[dict(key_family=v['key_family'],normal=next(z for z in v['summary'] if z['condition']=='normal')) for v in position]),indent=2))
if __name__=='__main__':main()
