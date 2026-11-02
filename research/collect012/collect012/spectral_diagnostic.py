"""Fresh-history no-fit Woodbury spectral/diagonal state-cost diagnostic."""
import argparse,json,sys,time
from pathlib import Path
import torch
from .common import ROOT
from .curvature_diagnostic import update as full_update

def step(memory,sketch,diagonal,key,value,gain,ridge,floor):
    # B is a square-root Gram sketch, not the inverse-curvature matrix.
    residual_diagonal=(ridge+diagonal-sketch.square().sum(-2)).clamp_min(floor)
    invd=1/residual_diagonal
    direct=invd*key
    bd=sketch*invd[...,None,:]
    middle=torch.eye(sketch.shape[-2])+bd@sketch.transpose(-1,-2)
    rhs=(sketch@direct[...,None])
    solved=torch.linalg.solve(middle,rhs)
    z=direct-(bd.transpose(-1,-2)@solved).squeeze(-1)
    leverage=(key*z).sum(-1)
    if not bool((leverage>0).all()):raise ValueError('sketch inverse direction lost positive leverage')
    error=value-(memory@key[...,None]).squeeze(-1)
    memory=memory+gain[...,None,None]*error[...,None]*(z/leverage[...,None])[...,None,:]
    joined=torch.cat((sketch,(gain.sqrt()[...,None]*key)[...,None,:]),-2)
    _,singular,right=torch.linalg.svd(joined,full_matrices=False)
    candidate=singular[...,:sketch.shape[-2],None]*right[...,:sketch.shape[-2],:]
    sketch=torch.where((gain!=0)[...,None,None],candidate,sketch)
    diagonal=diagonal+gain[...,None]*key.square()
    return memory,sketch,diagonal,float(residual_diagonal.min()),float(leverage.min())

def main():
    p=argparse.ArgumentParser();p.add_argument('--prior-root',type=Path,required=True);p.add_argument('--repair-root',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.prior_root.resolve()));sys.path.insert(0,str(a.repair_root.resolve()))
    from models import Model,ENC,CHARS
    from calibration_probe import dataset
    from collect011.commit_repair_probe import syntax_commit
    from collect011.support import SupportLane
    protocol=json.loads((ROOT/'SPECTRAL_DIAGNOSTIC_PROTOCOL.json').read_text());torch.set_num_threads(1);rows=[]
    with torch.inference_mode():
        for seed in protocol['seeds']:
            m=Model('phase_wide_hardnce');m.load_state_dict(torch.load(a.prior_root/'checkpoints'/f'phase_wide_hardnce_{seed}.pt',weights_only=True,map_location='cpu')['state_dict']);m.eval()
            for n,knum in protocol['loads']:
                groups=dataset(n,knum,'family',protocol['namespace']);prefix=groups['present'][0][:16,:-5]
                k,v,c,r,_=m.encode(prefix);commits=syntax_commit(prefix,ENC);support=[SupportLane() for _ in range(16)]
                for owner in range(16):
                    for pos in torch.nonzero(commits[owner]).flatten().tolist():support[owner].add(''.join(CHARS[int(i)] for i in prefix[owner,pos-5:pos-2]).encode())
                encoded={}
                for label,(x,y,rr) in groups.items():
                    items=[]
                    for lo in range(0,len(x),16):
                        context=torch.cat((prefix[:,-5:],x[lo:lo+16,-5:]),1);kk,_,_,rw,_=m.encode(context)
                        items.append((kk[:,-1],rw[:,-1],x[lo:lo+16,-4:-1]))
                    encoded[label]=items
                for mode in protocol['modes']:
                    memory=torch.zeros(16,1,17,64);full=torch.eye(64).expand(16,1,64,64).clone()/.1
                    rank=0 if mode=='full' else int(mode.removeprefix('spectral'))
                    sketch=torch.zeros(16,1,rank,64);diagonal=torch.zeros(16,1,64)
                    tic=time.perf_counter();minimum_d=float('inf');minimum_leverage=float('inf');events=0
                    for t in torch.nonzero(commits.any(0)).flatten().tolist():
                        g=commits[:,t,None].to(k.dtype)*r[:,t];events+=int(commits[:,t].sum())
                        if mode=='full':memory,full=full_update(memory,full,k[:,t],v[:,t,None,:],g,'full')
                        else:
                            memory,sketch,diagonal,md,ml=step(memory,sketch,diagonal,k[:,t],v[:,t,None,:],g,protocol['ridge'],protocol['diagonal_floor'])
                            minimum_d=min(minimum_d,md);minimum_leverage=min(minimum_leverage,ml)
                    seconds=time.perf_counter()-tic;metrics={}
                    for label,(x,y,rr) in groups.items():
                        predictions=[]
                        for q,rw,keychars in encoded[label]:
                            read=((memory@q[...,None]).squeeze(-1)*rw[...,None]).sum(1);raw=m.decode(read).argmax(-1)
                            present=torch.tensor([support[h].contains(''.join(CHARS[int(i)] for i in keychars[h]).encode()) for h in range(16)])
                            predictions.extend(torch.where(present,raw,16).tolist())
                        flags=(torch.tensor(predictions)==y).reshape(-1,16)
                        metrics[label]=dict(accuracy=float(flags.double().mean()),complete_map=float(flags.all(0).double().mean()),predictions=predictions,targets=y.tolist())
                    row=dict(mode=mode,seed=seed,N=n,K=knum,rank=rank,metrics=metrics,events=events,optimizer_updates=0,
                        curvature_state_bytes_per_owner=(4096 if mode=='full' else rank*64+64)*4,
                        other_state_bytes_per_owner=17*64*4+512,operator_seconds=seconds,
                        finite=bool(torch.isfinite(memory).all() and torch.isfinite(full if mode=='full' else sketch).all() and torch.isfinite(diagonal).all()),
                        minimum_residual_diagonal=None if mode=='full' else minimum_d,minimum_leverage=None if mode=='full' else minimum_leverage)
                    rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='metrics'}),flush=True)
                    (ROOT/'results/SPECTRAL_DIAGNOSTIC.json').write_text(json.dumps(dict(protocol=protocol,rows=rows),indent=2,allow_nan=False))

if __name__=='__main__':main()
