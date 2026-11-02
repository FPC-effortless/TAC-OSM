"""Frozen COLLECT010 replay: new histories; no parameter updates or threshold fit.

Requires --prior-root pointing to the verified original COLLECT010 bundle.
Original phase_wide_hardnce seeds; FP32/FP64; original .1 ridge/.25 novelty.
New 11B namespace is exploratory held-out history evidence, NOT a sealed TEST.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
from .memory import write_step

def main():
    p=argparse.ArgumentParser();p.add_argument("--prior-root",type=Path,required=True);args=p.parse_args()
    sys.path.insert(0,str(args.prior_root.resolve()))
    from models import Model
    from calibration_probe import dataset
    torch.set_num_threads(1)
    groups=dataset(256,32,"family",11_000_000_000)
    results=[];predictions=[]
    for seed in (1,2,3):
        path=args.prior_root/"checkpoints"/f"phase_wide_hardnce_{seed}.pt"
        ck=torch.load(path,weights_only=True,map_location="cpu")
        for dtype in (torch.float32,torch.float64):
            m=Model("phase_wide_hardnce").to(dtype);m.load_state_dict(ck["state_dict"]);m.eval()
            with torch.inference_mode():
                prefix=groups["present"][0][:16,:-5]
                k,v,c,r,_=m.encode(prefix);batch,_,heads,dim=k.shape
                s=k.new_zeros(batch,heads,17,dim);pp=torch.eye(dim,dtype=dtype).expand(batch,heads,dim,dim).clone()/.1
                for kk,vv,cc,rr in zip(k.unbind(1),v.unbind(1),c.unbind(1),r.unbind(1)):
                    s,pp=write_step(s,pp,kk,vv[:,None,:].expand(-1,heads,-1),cc[:,None]*rr,floor=1e-9)
                metrics={};entries={};small_margin=0;bad_raw=0;false_reject=0
                for label,(x,y,rows) in groups.items():
                    preds=[];raws=[];novelties=[];margins=[]
                    for lo in range(0,len(x),16):
                        context=torch.cat((prefix[:,-5:],x[lo:lo+16,-5:]),1)
                        kk,vv,cc,rr,_=m.encode(context)
                        ss,pr=s,pp
                        for j in range(5,10):
                            ss,pr=write_step(ss,pr,kk[:,j],vv[:,j,None,:].expand(-1,heads,-1),cc[:,j,None]*rr[:,j],floor=1e-9)
                        q=kk[:,-1];route=rr[:,-1]
                        read=((ss@q[...,None]).squeeze(-1)*route[...,None]).sum(1)
                        logits=m.decode(read);raw=logits.argmax(-1)
                        novelty=.1*((q*(pr@q[...,None]).squeeze(-1)).sum(-1)*route).sum(-1)
                        pred=torch.where(novelty>.25,16,raw)
                        top=logits.topk(2,-1).values
                        preds.extend(pred.tolist());raws.extend(raw.tolist());novelties.extend(novelty.tolist());margins.extend((top[:,0]-top[:,1]).tolist())
                    ok=torch.tensor(preds)==y;hist={}
                    for i,row in enumerate(rows):hist.setdefault(row[4],[]).append(bool(ok[i]))
                    permap=[sum(a)/len(a) for a in hist.values()]
                    metrics[label]=dict(n=len(y),guarded_accuracy=float(ok.double().mean()),raw_accuracy=float((torch.tensor(raws)==y).double().mean()),
                        complete_map=sum(all(a) for a in hist.values())/len(hist),map_accuracy=permap,
                        false_rejections=sum(raw==int(t) and pred!=int(t) for raw,pred,t in zip(raws,preds,y)),
                        min_margin=min(margins))
                    entries[label]=dict(predictions=preds,raw_predictions=raws,targets=y.tolist(),novelty=novelties,margins=margins)
                result=dict(seed=seed,dtype=str(dtype),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    metrics=metrics,minimum_precision_eigenvalue=float(torch.linalg.eigvalsh(pp).min()))
                results.append(result);predictions.append(dict(seed=seed,dtype=str(dtype),entries=entries))
                print(json.dumps(result),flush=True)
                root=Path(__file__).resolve().parents[1]/"results"
                (root/"C10_PRECISION_REPLAY.json").write_text(json.dumps(dict(namespace=11_000_000_000,ridge=.1,threshold=.25,N=256,K=32,maps=16,
                    optimizer_updates=0,status="FROZEN_COMPONENT_REPLAY_NOT_LM_TRAINING",rows=results),indent=2))
                (root/"C10_PRECISION_PREDICTIONS.json").write_text(json.dumps(predictions))

if __name__=="__main__":main()
