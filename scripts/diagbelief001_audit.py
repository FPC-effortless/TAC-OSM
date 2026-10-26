"""Read-only original ME110 FIT audit; never constructs an optimizer."""
import csv
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from scripts import me110_typed_training as run
from scripts.me110_objective import losses
from tac_osm.typed_role_learning import uncertainty_masks
from tac_osm.research_observatory import gradient_conflict_atlas

OUT=run.ROOT/'research/mathematical-engineering/diagbelief001/RESULTS.json'


def gradient_stats(model, loss):
    params=tuple(model.parameters()); masks=uncertainty_masks(model)
    gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    p2=u2=0.
    for g,m in zip(gs,masks):
        if g is not None:
            p2+=float(g[~m].double().square().sum());u2+=float(g[m].double().square().sum())
    return dict(point_coordinate_l2=math.sqrt(p2),uncertainty_coordinate_l2=math.sqrt(u2))


def fit_loader_audit():
    """Instrument real CSV row numeric accesses, then independently refit stats."""
    original=csv.DictReader; counts={'numeric_fields':0,'cutoff_year_rows':0}
    class Row(dict):
        def __getitem__(self,key):
            if key in run.data.NUMERIC or key=='wd':
                assert int(dict.__getitem__(self,'year'))<2015
                counts['numeric_fields']+=1
            if key=='year' and int(dict.__getitem__(self,'year'))>=2015:
                counts['cutoff_year_rows']+=1
            return dict.__getitem__(self,key)
    def reader(*args,**kwargs):
        for row in original(*args,**kwargs):yield Row(row)
    with patch.object(csv,'DictReader',reader): d=run.data.load('fit')
    z=d['raw'].astype(np.float64).copy();z[...,:6]=np.log1p(z[...,:6])
    # Source does float32 log1p before a float64 aggregate. Match that contract
    # and also report the error from the independent all-float64 calculation.
    z32=d['raw'].copy();z32[...,:6]=np.log1p(z32[...,:6]);z32=z32.astype(np.float64)
    mean=np.nanmean(z32,axis=(0,1)).astype(np.float32)
    std=np.nanstd(z32,axis=(0,1)).clip(.0001).astype(np.float32)
    assert np.array_equal(mean,d['mean']) and np.array_equal(std,d['std'])
    assert d['raw'].shape[1]==run.data.FIT_STOP and counts['cutoff_year_rows']==12
    return d,dict(**counts,fit_hours=d['raw'].shape[1],source_sha256=run.sha(run.data.ARCHIVE),
        actual_fit_transform_bit_equal=True,protected_numeric_accesses=0,
        float64_reference_mean_max_abs=float(np.max(np.abs(np.nanmean(z,axis=(0,1))-d['mean']))),
        scope='actual FIT loader and transform only; not full REPORT/cache/ancestor provenance')


def execute():
    if OUT.exists():raise RuntimeError('audit exists; do not overwrite')
    torch.set_num_threads(1);d,loader=fit_loader_audit()
    monitor=np.load(run.DOC/'FIT_PLANS.npz')['monitor'];rows=[]
    for seed in run.SEEDS:
        path=run.OUT/run.ARCH/f'seed-{seed}'/'final.pt';before=run.sha(path)
        cp=torch.load(path,map_location='cpu',weights_only=False)
        model=run.initialize(d,seed);model.load_state_dict(cp['model']);model.eval()
        sums={k:np.zeros(6,dtype=np.float64) for k in ('missing','floor_dominates_total','left_floor_active','right_floor_active','width','minimum_width','mean_nominal_variance','mean_actual_latent_variance')}
        with torch.no_grad():
            for i in range(0,len(monitor),32):
                x=[torch.from_numpy(a) for a in run.data.batch(d,monitor[i:i+32],256)]
                f,_,tr=model(*x[:6],return_trace=True);missing=~x[1][:,-1,:6]
                sums['missing']+=missing.sum(0).numpy()
                for key,tensor in (
                    ('floor_dominates_total',tr['floor_variance']>tr['modeled_variance']),
                    ('left_floor_active',tr['base_log_scale']*tr['width_factors'][:,:6]<tr['protected_floor_scale']),
                    ('right_floor_active',tr['base_log_scale']*tr['width_factors'][:,6:]<tr['protected_floor_scale'])):
                    sums[key]+=(tensor & missing).sum(0).numpy()
                q=model.raw_quantiles(f);sums['width']+=torch.where(missing,q[...,2]-q[...,0],0).sum(0).numpy()
                ff=type(f)(f.log_location,tr['protected_floor_scale'].clamp_min(1e-12),tr['protected_floor_scale'].clamp_min(1e-12))
                fq=model.raw_quantiles(ff);sums['minimum_width']+=torch.where(missing,fq[...,2]-fq[...,0],0).sum(0).numpy()
                # Actual pre-censor standardized-log variance of equal-mass
                # half-normal law; not its raw-unit variance or base variance.
                v=.5*(f.left_scale.square()+f.right_scale.square())-(f.right_scale-f.left_scale).square()/(2*math.pi)
                sums['mean_nominal_variance']+=tr['total_variance'].sum(0).numpy()
                sums['mean_actual_latent_variance']+=v.sum(0).numpy()
        summaries={k:np.divide(v,sums['missing'],out=np.zeros(6),where=sums['missing']>0).tolist()
                   for k,v in sums.items() if k not in ('missing','mean_nominal_variance','mean_actual_latent_variance')}
        summaries['missing_channel_counts']=sums['missing'].astype(int).tolist()
        summaries['actual_to_nominal_latent_variance_ratio']=(sums['mean_actual_latent_variance']/sums['mean_nominal_variance']).tolist()
        # Fixed augmented, genuine FIT diagnostic batch, same origins/RNG all
        # seeds. Reuses existing Observatory; no parameter displacement.
        x=run.tensor_batch(d,monitor[:8].tolist()+monitor[64:72].tolist(),256,np.random.default_rng(111900))
        f,_,tr=model(*x[:6],return_trace=True);p,u,parts=losses(model,f,tr,x,run.raw_scale(d))
        mean_grad=torch.autograd.grad(parts['reconstruction_nll'],tr['current_mean'],retain_graph=True)[0]
        atlas=gradient_conflict_atlas(model,{'point':p,'nll':parts['nll'],'interval':parts['interval'],'reconstruction':parts['reconstruction_nll']})
        stats={k:gradient_stats(model,parts[k]) for k in ('nll','interval','reconstruction_nll')}
        assert before==run.sha(path)
        rows.append(dict(seed=seed,checkpoint_sha256=before,floors=summaries,
            reconstruction_current_mean_gradient_norm=float(mean_grad.norm()),objective_gradient_roles=stats,
            gradient_atlas=atlas,hidden_reconstruction_labels=int(x[9].sum()),no_optimizer_created=True))
    result=dict(study='DIAGBELIEF001',scope='FIT_ONLY_NO_UPDATES',loader=loader,models=rows,
        integrity_clearance='INCONCLUSIVE',report_scored=False,protected_numeric_access=False,
        note='Local gradients and floor activity do not prove causal explanation of qualification failures')
    run.write(OUT,result);print(json.dumps({k:v for k,v in result.items() if k!='models'},indent=2))
    for r in rows: print(r['seed'],json.dumps(r['floors']),r['objective_gradient_roles'])

if __name__=='__main__':execute()
