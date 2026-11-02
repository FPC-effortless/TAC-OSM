"""Frozen structured-memory engineering: full/block/diagonal curvature.

No neural fitting, threshold tuning or LM quality claim. Uses the already
validated causal commit and support lane to isolate curvature interference.
"""
import argparse,json,sys,time
from pathlib import Path
import torch
from .common import ROOT

def update(memory,geometry,key,value,gain,mode):
    if mode=='full':
        z=(geometry@key[...,None]).squeeze(-1);s=(key*z).sum(-1)
        candidate=geometry-gain[...,None,None]*z[...,None]*z[...,None,:]/(1+gain*s)[...,None,None]
        geometry=(candidate+candidate.transpose(-1,-2))/2
    elif mode=='diagonal':
        z=key/geometry;s=(key*z).sum(-1)
        geometry=geometry+gain[...,None]*key.square()
    else:
        kb=key.reshape(*key.shape[:-1],4,16)
        zb=(geometry@kb[...,None]).squeeze(-1);sb=(kb*zb).sum(-1)
        z=zb.flatten(-2);s=sb.sum(-1)
        candidate=geometry-gain[...,None,None,None]*zb[...,None]*zb[...,None,:]/(1+gain[...,None]*sb)[...,None,None]
        geometry=(candidate+candidate.transpose(-1,-2))/2
    error=value-(memory@key[...,None]).squeeze(-1)
    memory=memory+gain[...,None,None]*error[...,None]*(z/s.clamp_min(1e-9)[...,None])[...,None,:]
    return memory,geometry

def main():
    p=argparse.ArgumentParser();p.add_argument('--prior-root',type=Path,required=True);p.add_argument('--repair-root',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.prior_root.resolve()));sys.path.insert(0,str(a.repair_root.resolve()))
    from models import Model,ENC,CHARS
    from calibration_probe import dataset
    from collect011.commit_repair_probe import syntax_commit
    from collect011.support import SupportLane
    torch.set_num_threads(1);rows=[]
    protocol=json.loads((ROOT/'CURVATURE_DIAGNOSTIC_PROTOCOL.json').read_text())
    with torch.inference_mode():
        for seed in protocol['seeds']:
            m=Model('phase_wide_hardnce');m.load_state_dict(torch.load(a.prior_root/'checkpoints'/f'phase_wide_hardnce_{seed}.pt',weights_only=True,map_location='cpu')['state_dict']);m.eval()
            for n,knum in protocol['loads']:
                groups=dataset(n,knum,'family',protocol['namespace']);prefix=groups['present'][0][:16,:-5]
                k,v,c,r,_=m.encode(prefix);commits=syntax_commit(prefix,ENC)
                support=[SupportLane() for _ in range(16)]
                for owner in range(16):
                    for pos in torch.nonzero(commits[owner]).flatten().tolist():
                        support[owner].add(''.join(CHARS[int(i)] for i in prefix[owner,pos-5:pos-2]).encode())
                # Query encodings are identical across all operator controls.
                encoded={}
                for label,(x,y,rr) in groups.items():
                    items=[]
                    for lo in range(0,len(x),16):
                        context=torch.cat((prefix[:,-5:],x[lo:lo+16,-5:]),1)
                        kk,_,_,rw,_=m.encode(context)
                        items.append((kk[:,-1],rw[:,-1],x[lo:lo+16,-4:-1]))
                    encoded[label]=items
                for mode in protocol['modes']:
                    memory=torch.zeros(16,1,17,64)
                    geometry=(torch.eye(64).expand(16,1,64,64).clone()/.1 if mode=='full' else
                        torch.full((16,1,64),.1) if mode=='diagonal' else torch.eye(16).expand(16,1,4,16,16).clone()/.1)
                    tic=time.perf_counter();events=0
                    for t in torch.nonzero(commits.any(0)).flatten().tolist():
                        g=commits[:,t,None].to(k.dtype)*r[:,t]
                        memory,geometry=update(memory,geometry,k[:,t],v[:,t,None,:],g,mode);events+=int(commits[:,t].sum())
                    seconds=time.perf_counter()-tic;metrics={}
                    for label,(x,y,rr) in groups.items():
                        predictions=[]
                        for q,rw,keychars in encoded[label]:
                            read=((memory@q[...,None]).squeeze(-1)*rw[...,None]).sum(1)
                            raw=m.decode(read).argmax(-1)
                            present=torch.tensor([support[h].contains(''.join(CHARS[int(i)] for i in keychars[h]).encode()) for h in range(16)])
                            predictions.extend(torch.where(present,raw,16).tolist())
                        ok=(torch.tensor(predictions)==y).reshape(-1,16)
                        metrics[label]=dict(accuracy=float(ok.double().mean()),complete_map=float(ok.all(0).double().mean()),
                            predictions=predictions,targets=y.tolist())
                    row=dict(seed=seed,N=n,K=knum,mode=mode,metrics=metrics,events=events,
                        operator_seconds=seconds,curvature_state_bytes_per_owner=geometry.numel()*4//16,
                        memory_state_bytes_per_owner=memory.numel()*4//16,support_bytes_per_owner=512,
                        finite=bool(torch.isfinite(memory).all() and torch.isfinite(geometry).all()),optimizer_updates=0)
                    rows.append(row);print(json.dumps({key:value for key,value in row.items() if key!='metrics'}),flush=True)
                    (ROOT/'results/CURVATURE_DIAGNOSTIC.json').write_text(json.dumps(dict(protocol=protocol,rows=rows),indent=2))

if __name__=='__main__':main()
