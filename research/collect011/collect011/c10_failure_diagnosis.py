"""Ground-truth-assisted component diagnosis, NEVER a deployment algorithm.

Determines whether an observed error was already in the last written value or
introduced by address/memory/readout interference. No tuning or model updates.
"""
import argparse,json,sys
from pathlib import Path
import torch
from .memory import write_step

def main():
    p=argparse.ArgumentParser();p.add_argument("--prior-root",type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.prior_root.resolve()))
    from models import Model
    from calibration_probe import dataset
    torch.set_num_threads(1)
    groups=dataset(256,32,"family",11_000_000_000);x,y,rows=groups["present"]
    with torch.inference_mode():
        ck=torch.load(a.prior_root/"checkpoints/phase_wide_hardnce_1.pt",weights_only=True,map_location="cpu")
        m=Model("phase_wide_hardnce").double();m.load_state_dict(ck["state_dict"]);m.eval()
        prefix=x[:16,:-5];k,v,c,r,_=m.encode(prefix)
        s=torch.zeros(16,1,17,64,dtype=torch.float64);pp=torch.eye(64,dtype=torch.float64).expand(16,1,64,64).clone()/.1
        for kk,vv,cc,rr in zip(k.unbind(1),v.unbind(1),c.unbind(1),r.unbind(1)):
            s,pp=write_step(s,pp,kk,vv[:,None,:],cc[:,None]*rr,floor=1e-9)
        errors=[];last_value_correct=0
        for lo in range(0,len(x),16):
            suffix=torch.cat((prefix[:,-5:],x[lo:lo+16,-5:]),1)
            kk,vv,cc,rr,_=m.encode(suffix)
            ss,pr=s,pp
            for t in range(5,10):ss,pr=write_step(ss,pr,kk[:,t],vv[:,t,None,:],cc[:,t,None]*rr[:,t],floor=1e-9)
            q=kk[:,-1];read=(ss@q[...,None]).squeeze(-1).squeeze(1)
            predictions=m.decode(read).argmax(-1)
            for h in range(16):
                _,target,records,key,owner=rows[lo+h]
                index=max(i for i,(key0,_) in enumerate(records) if key0==key)
                pos=index*6+5;latest=v[h,pos]
                echo=int(m.decode(latest[None]).argmax(-1));last_value_correct+=echo==target
                if int(predictions[h])!=target:
                    novelty=float(.1*(q[h]*(pr[h]@q[h,:, :,None]).squeeze(-1)).sum())
                    other=[(key0,i) for i,(key0,_) in enumerate(records) if key0!=key]
                    other_last={key0:i for key0,i in other}
                    competitors=sorted([(float((q[h,0]*k[h,i*6+5,0]).sum()),key0) for key0,i in other_last.items()],reverse=True)[:4]
                    earlier=[(i,val) for i,(key0,val) in enumerate(records[:index]) if key0==key and float(c[h,i*6+5])>0]
                    previous=earlier[-1] if earlier else None
                    features=m.ev(__import__('models').windows(prefix[h:h+1])).flatten(-2)
                    commit_probability=float(m.commit(features)[0,pos,0].sigmoid())
                    errors.append(dict(history=h,key=key,target=target,prediction=int(predictions[h]),
                        last_record=index,last_written_value_decode=echo,last_write_commit=float(c[h,pos]),
                        last_write_commit_probability=commit_probability,last_admitted_earlier_record=previous,
                        query_last_write_cosine=float((q[h,0]*k[h,pos,0]).sum()),novelty=novelty,
                        last_value_read_l2=float((latest-read[h]).norm()),nearest_other_addresses=competitors))
        result=dict(status="ORACLE_ASSISTED_DIAGNOSIS_ONLY",seed=1,namespace=11_000_000_000,
            queries=len(y),last_written_value_decode_correct=last_value_correct,errors=errors,
            qualification="uses record identity/ground truth to locate last write; not an inference path or capability score")
        root=Path(__file__).resolve().parents[1]
        (root/"results/C10_FAILURE_DIAGNOSIS.json").write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))

if __name__=="__main__":main()
