"""Post-hoc descriptive FIT reconstruction capability check; no fit or gate."""
import hashlib
import numpy as np
import torch
from scripts import me111_conditional_training as run


def execute():
    path=run.OUT/'CURRENT_STATE_DIAGNOSTIC.json'
    if path.exists():raise RuntimeError('original current-state diagnostic exists')
    torch.set_num_threads(1);run.budget_audit();d=run.fit_data()
    monitor=np.load(run.DOC/'FIT_PLANS.npz')['monitor'];rng=np.random.default_rng(111700)
    batches=[run.tensor_batch(d,monitor[i:i+32],256,rng) for i in range(0,len(monitor),32)]
    digest=hashlib.sha256()
    for x in batches:
        for value in x:digest.update(value.numpy().tobytes())
    counts=sum((x[9].sum(0).numpy() for x in batches),np.zeros(6,dtype=np.int64));rows=[]
    for seed in run.SEEDS:
        _,initial,_=run.parent_artifacts(seed)
        finalpath=run.OUT/run.ARCH/f'seed-{seed}'/'final.pt';identity=run.sha(finalpath)
        final=torch.load(finalpath,map_location='cpu',weights_only=False)
        stages={}
        for label,cp in [('initial',initial),('final',final)]:
            m=run.initialize(d,seed);m.load_state_dict(cp['model']);m.eval();versions=m.versions()
            absolute=np.zeros(6);squared=np.zeros(6);nll=0.;n=0
            with torch.no_grad():
                for x in batches:
                    _,_,tr=m(*x[:6],return_trace=True);err=tr['current_mean']-x[8];mask=x[9]
                    absolute+=torch.where(mask,err.abs(),0).sum(0).numpy()
                    squared+=torch.where(mask,err.square(),0).sum(0).numpy()
                    count=int(mask.sum());nll+=float(m.reconstruction_loss(tr,x[8],mask))*count;n+=count
            assert m.versions()==versions
            stages[label]=dict(standardized_log_mae_by_channel=(absolute/counts).tolist(),
                standardized_log_mse_by_channel=(squared/counts).tolist(),current_gaussian_nll=nll/n)
        assert identity==run.sha(finalpath);rows.append(dict(seed=seed,checkpoint_sha256=identity,stages=stages))
    run.write(path,dict(scope='POST_HOC_DESCRIPTIVE_EXPOSED_FIT_ONLY; not acceptance or independent qualification',
        registered_after_original_REPORT=True,monitor_origins=320,hidden_recorded_current_labels_by_channel=counts.tolist(),
        actual_augmented_batch_sha256=digest.hexdigest(),models=rows,no_new_fits=True,no_optimizer_created=True,
        no_report_or_protected_scoring=True,quantities='errors in standardized-log input units, not physical-unit errors'))
    print('CURRENT_STATE_DIAGNOSTIC_COMPLETE',counts.tolist())
if __name__=='__main__':execute()
