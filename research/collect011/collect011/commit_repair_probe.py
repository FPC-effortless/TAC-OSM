"""Frozen structured-record repair, not a natural-language write oracle.

The validated event gate accepts exactly [A-H][A-H][A-H]:[a-p]; ending NOW.
It sees only raw observed characters, with no ID table, target or future token.
Compare against the same trained frontend's original learned hard commit.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
from .memory import write_step
from .support import SupportLane

def syntax_commit(x,enc):
    allowed_key=torch.tensor([enc[c] for c in 'ABCDEFGH'],device=x.device)
    allowed_value=torch.tensor([enc[c] for c in 'abcdefghijklmnop'],device=x.device)
    out=torch.zeros_like(x,dtype=torch.bool)
    if x.shape[1]>=6:
        w=x.unfold(1,6,1)
        valid=(w[...,:3,None]==allowed_key).any(-1).all(-1)
        valid &= w[...,3]==enc[':']
        valid &= (w[...,4,None]==allowed_value).any(-1)
        valid &= w[...,5]==enc[';']
        out[:,5:]=valid
    return out

def run():
    p=argparse.ArgumentParser();p.add_argument('--prior-root',type=Path,required=True)
    p.add_argument('--protocol',default='COMMIT_REPAIR_PROTOCOL.json');p.add_argument('--output',default='COMMIT_REPAIR');a=p.parse_args()
    sys.path.insert(0,str(a.prior_root.resolve()))
    from models import Model,ENC,CHARS
    from calibration_probe import dataset
    torch.set_num_threads(1);root=Path(__file__).resolve().parents[1]
    protocol=json.loads((root/a.protocol).read_text());results=[];allpred=[]
    for seed in protocol['seeds']:
        checkpoint=a.prior_root/'checkpoints'/f'phase_wide_hardnce_{seed}.pt'
        ck=torch.load(checkpoint,weights_only=True,map_location='cpu')
        m=Model('phase_wide_hardnce');m.load_state_dict(ck['state_dict']);m.eval()
        for n,numkeys in protocol['loads']:
            groups=dataset(n,numkeys,'family',protocol['namespace'])
            with torch.inference_mode():
                prefix=groups['present'][0][:16,:-5]
                k,v,c,r,_=m.encode(prefix);gate=syntax_commit(prefix,ENC).to(k.dtype)
                missed=int(((gate==1)&(c==0)).sum());false=int(((gate==0)&(c==1)).sum())
                # Membership is accumulated from raw observations at validated
                # commits, never from evaluation rows/labels/target dictionaries.
                support=[SupportLane() for _ in range(16)]
                for h in range(16):
                    for pos in torch.nonzero(gate[h]).flatten().tolist():
                        keybytes=''.join(CHARS[int(i)] for i in prefix[h,pos-5:pos-2]).encode()
                        support[h].add(keybytes)
                for mode in protocol['modes']:
                    gain=c if mode=='learned_hard' else gate
                    s=k.new_zeros(16,1,17,64);pp=torch.eye(64).expand(16,1,64,64).clone()/.1
                    for kk,vv,gg,rr in zip(k.unbind(1),v.unbind(1),gain.unbind(1),r.unbind(1)):
                        s,pp=write_step(s,pp,kk,vv[:,None,:],gg[:,None]*rr,floor=1e-9)
                    metrics={};entries={}
                    for label,(xx,yy,rows) in groups.items():
                        preds=[];raws=[];novelties=[]
                        for lo in range(0,len(xx),16):
                            observed=torch.cat((prefix[:,-5:],xx[lo:lo+16,-5:]),1)
                            kk,vv,cc,rr,_=m.encode(observed)
                            gg=cc if mode=='learned_hard' else syntax_commit(observed,ENC).to(kk.dtype)
                            ss,pr=s,pp
                            for t in range(5,10):ss,pr=write_step(ss,pr,kk[:,t],vv[:,t,None,:],gg[:,t,None]*rr[:,t],floor=1e-9)
                            q=kk[:,-1];rw=rr[:,-1];read=((ss@q[...,None]).squeeze(-1)*rw[...,None]).sum(1)
                            raw=m.decode(read).argmax(-1);novelty=.1*((q*(pr@q[...,None]).squeeze(-1)).sum(-1)*rw).sum(-1)
                            if mode=='validated_event_support':
                                present=torch.tensor([support[h].contains(''.join(CHARS[int(i)] for i in xx[lo+h,-4:-1]).encode()) for h in range(16)])
                                pred=torch.where(present,raw,16)
                            else:pred=torch.where(novelty>.25,16,raw)
                            raws.extend(raw.tolist());preds.extend(pred.tolist());novelties.extend(novelty.tolist())
                        ok=torch.tensor(preds)==yy;hist={}
                        for i,row in enumerate(rows):hist.setdefault(row[4],[]).append(bool(ok[i]))
                        metrics[label]=dict(n=len(yy),accuracy=float(ok.double().mean()),raw_accuracy=float((torch.tensor(raws)==yy).double().mean()),
                            complete_map=sum(all(v) for v in hist.values())/len(hist),
                            false_rejections=sum(raw==int(t) and pred!=int(t) for raw,pred,t in zip(raws,preds,yy)))
                        entries[label]=dict(predictions=preds,raw_predictions=raws,targets=yy.tolist(),novelty=novelties)
                    row=dict(seed=seed,N=n,K=numkeys,mode=mode,metrics=metrics,learned_missed_writes=missed,
                        learned_false_writes=false,support_lane_bytes=512 if mode=='validated_event_support' else 0,
                        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
                    results.append(row);allpred.append(dict(seed=seed,N=n,K=numkeys,mode=mode,entries=entries))
                    print(json.dumps(row),flush=True)
                    (root/f'results/{a.output}.json').write_text(json.dumps(dict(protocol=protocol,optimizer_updates=0,rows=results),indent=2))
                    (root/f'results/{a.output}_PREDICTIONS.json').write_text(json.dumps(allpred))

if __name__=='__main__':run()
