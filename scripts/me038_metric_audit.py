"""Independent NumPy final-metric reductions, without selection or refitting."""
import json,time
import numpy as np
import torch
import scripts.me038_research as b
from tac_osm.me038_innovation_forecast import build_model


def run():
    torch.set_num_threads(1);frozen=b.verify();result=json.loads((b.OUT/'results.json').read_text())
    fits={r['key']:r for r in frozen['fits']};receipts=[];values=0
    for r in result['models']+result['persistence']:
        if r['dataset']!='tetouan':continue
        start=time.process_time();z,meta=b.data('tetouan');_,va,end=meta['split_edges'];h=r['horizon']
        if 'arm' in r:
            model=build_model(r['arm'],h,r['seed']);model.load_state_dict(torch.load(b.ROOT/fits[r['key']]['selected']['checkpoint'],weights_only=False)['state_dict'])
        else:model=b.Persistence(h)
        model.eval();ss=np.zeros(z.size(1));sa=np.zeros_like(ss);n=0
        ix=np.arange(va,end-h+1)
        with torch.no_grad():
            for i in range(0,len(ix),16):
                x,y=b.batch(z,ix[i:i+16],h);error=(model(x)-y).numpy().astype(np.float64)
                ss+=np.sum(error*error,axis=(0,1),dtype=np.float64);sa+=np.sum(np.abs(error),axis=(0,1),dtype=np.float64)
                n+=error.shape[0]*h
        got={'MSE':float((ss/n).mean()),'MAE':float((sa/n).mean()),'channel_MSE':(ss/n).tolist(),'channel_MAE':(sa/n).tolist(),
             'power_MSE':float((ss/n)[5:].mean()),'power_MAE':float((sa/n)[5:].mean())}
        u=2.**-53;gamma=(n+8)*u/(1-(n+8)*u);diffs=[];bounds=[]
        for k,v in got.items():
            ref=np.asarray(r['test'][k]);error=np.abs(np.asarray(v)-ref);bound=4*gamma*np.maximum(1,np.abs(ref))
            assert (error<=bound).all(),(r.get('key'),k,error,bound)
            diffs.append(float(error.max()));bounds.append(float(bound.max()))
        values+=n*z.size(1);receipts.append({'key':r.get('key',f'persistence-tetouan-h{h}'),'scalar_forecasts':n*z.size(1),
            'maximum_metric_absolute_difference':max(diffs),'maximum_declared_bound':max(bounds),'CPU_seconds':time.process_time()-start,'status':'PASS'})
    receipt={'status':'PASS','models_and_references':len(receipts),'scalar_test_forecasts_recomputed':values,
        'method':'Independent NumPy float64 reductions; fixed batch16 and float32 subtraction.',
        'original_checkpoints_no_refit_or_reselection':True,'scope':'All Tetouan selected models and persistence; not legacy test matrix.',
        'tolerance':'4*gamma_(n+8)*max(1,abs(metric)); u=2^-53; preregistered before test scoring.',
        'addendum_sha256':b.sha(b.REPORT/'AUDIT-ADDENDUM.md'),'rows':receipts}
    b.dump(b.OUT/'metric-audit.json',receipt);print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':run()
