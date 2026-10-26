"""No-fit per-coordinate task interference on original final ME110 models."""
import json
import math
import numpy as np
import torch
from scripts import me110_typed_training as run
from scripts.me110_objective import losses
from tac_osm.typed_role_learning import uncertainty_masks


def execute():
    out=run.ROOT/'research/mathematical-engineering/diagbelief001/ROUTED_GRADIENTS.json'
    if out.exists():raise RuntimeError('original diagnostic exists')
    torch.set_num_threads(1);d=run.fit_data();monitor=np.load(run.DOC/'FIT_PLANS.npz')['monitor'];rows=[]
    for seed in run.SEEDS:
        path=run.OUT/run.ARCH/f'seed-{seed}'/'final.pt';sha=run.sha(path)
        m=run.initialize(d,seed);m.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);m.eval()
        x=run.tensor_batch(d,monitor[:8].tolist()+monitor[64:72].tolist(),256,np.random.default_rng(111900))
        f,_,tr=m(*x[:6],return_trace=True);point,aux,parts=losses(m,f,tr,x,run.raw_scale(d))
        params=tuple(m.parameters());masks=uncertainty_masks(m);vectors={}
        for name,loss in {'point':point,'nll':parts['nll'],'interval':.25*parts['interval'],'reconstruction':.3*parts['reconstruction_nll']}.items():
            gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
            gs=[torch.zeros_like(p) if g is None else g for p,g in zip(params,gs)]
            vectors[name]={role:torch.cat([g[mask if role=='U' else ~mask].double() for g,mask in zip(gs,masks)]) for role in ('P','U')}
        pairs=[]
        names=list(vectors)
        for i,a in enumerate(names):
            for b in names[i+1:]:
                row=dict(left=a,right=b)
                for role in ('P','U'):
                    aa,bb=vectors[a][role],vectors[b][role];na,nb=float(aa.norm()),float(bb.norm())
                    row[role]=dict(left_norm=na,right_norm=nb,cosine=float(aa@bb)/(na*nb) if na*nb else None)
                pairs.append(row)
        assert sha==run.sha(path);rows.append(dict(seed=seed,checkpoint_sha256=sha,pairs=pairs))
    run.write(out,dict(scope='Fixed16 augmented FIT origins, one gradient batch per seed; no training or displacement',models=rows,qualification='local_gradient_geometry_not_causal_training_outcome'))
    for row in rows:
        print(row['seed'],[(p['left'],p['right'],p['U']['cosine']) for p in row['pairs']])
if __name__=='__main__':execute()
