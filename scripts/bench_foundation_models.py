#!/usr/bin/env python3
"""BENCH-FM-001: frozen official TAC vs TimesFM3/TabFM1.1, real MetroPT pilot.
Never fabricate a model result; each inference arm runs independent CLI entry.
"""
from __future__ import annotations
import argparse,hashlib,json,os,pathlib,time,traceback
import numpy as np
SENSORS=12;TP3_INDEX=1;N_EVAL=24;N_TAB_CONTEXT=64
TIMESFM_ID='google/timesfm-3.0-pytorch'
TABFM_ID='google/tabfm-1.0.0-pytorch'

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    return h.hexdigest()

def write_json(path,obj):
    pathlib.Path(path).parent.mkdir(parents=True,exist_ok=True)
    pathlib.Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')

def metrics(yhat,true):
    a=np.asarray(yhat,dtype=np.float64);b=np.asarray(true,dtype=np.float64)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f'Prediction shape/finite mismatch: {a.shape} {b.shape}')
    loss=np.abs(a-b)
    return dict(n=len(a),tp3_h12_mae=float(loss[:,0].mean()),
                tp3_h60_mae=float(loss[:,1].mean()),tp3_macro_mae=float(loss.mean()))

def pack(original,root):
    from scripts.me004_native_real_forecast import prepare,make_model
    from scripts.me004_future_week import source,prep
    import torch
    torch.set_num_threads(1)
    original=pathlib.Path(original);root=pathlib.Path(root);root.mkdir(parents=True,exist_ok=True)
    report=json.loads((original/'results.json').read_text())
    if report.get('study')!='ME004' or len(report.get('fits',[]))!=6:
        raise ValueError('Original ME004 original six-fit research manifest missing')
    for item in report['fits']:
        if digest(original/item['checkpoint'])!=item['checkpoint_sha256']:
            raise ValueError('Immutable trained checkpoint hash mismatch: '+item['checkpoint'])
    physical,times,src=source(root)
    _training,_validation,_threshold,computed_cohort=prepare(physical,times)
    for k in ('mean_train_only','std_train_only','horizons_in_samples','train_windows'):
        if k in ('mean_train_only','std_train_only'):
            if not np.allclose(computed_cohort[k],report['cohort'][k],atol=5e-6,rtol=1e-6):
                raise ValueError('Training-only normalization differs from original')
        elif computed_cohort[k]!=report['cohort'][k]:
            raise ValueError('ME004 training example selection drift: '+k)
    x,y,thr,future_info=prep(physical,times,report['cohort'])
    xx=x.numpy();yy=y.numpy()
    valid=np.flatnonzero(np.all(xx[:,:,SENSORS:]==1.0,axis=(1,2)))
    if len(valid)<N_EVAL:raise ValueError('Too few fully measured future histories')
    chosen=valid[np.linspace(0,len(valid)-1,N_EVAL,dtype=np.int64)]
    assert len(np.unique(chosen))==N_EVAL
    fit_x,fit_y=_training
    fit_x=fit_x.numpy();fit_y=fit_y.numpy()
    valid_train=np.flatnonzero(np.all(fit_x[:,:,SENSORS:]==1.0,axis=(1,2)))
    if len(valid_train)<N_TAB_CONTEXT:raise ValueError('Too few fully measured training histories')
    train_chosen=valid_train[np.linspace(0,len(valid_train)-1,N_TAB_CONTEXT,dtype=np.int64)]
    assert len(np.unique(train_chosen))==N_TAB_CONTEXT
    X=xx[chosen];Y=yy[chosen,:,TP3_INDEX]
    # Real training windows, chosen without looking at validation outcomes.
    np.savez_compressed(root/'fixed-inputs.npz',eval_x=X,eval_y=Y,
        train_x=fit_x[train_chosen],train_y=fit_y[train_chosen,:,TP3_INDEX],
        future_origin_indexes=chosen,original_fit_indexes=train_chosen)
    result={'study':'BENCH-FM-001','checkpoint_family':'original_me004',
            'dataset':'genuine_UCI_MetroPT3','source_sha256':src['source_sha256'],
            'sample':'24 evenly spaced complete-input origins from ME004F previously opened future-week',
            'n_orig_future_windows':len(xx),'n_fully_observed_origins':len(valid),
            'n_scored':len(chosen),'tabfm_train_example_count':N_TAB_CONTEXT,
            'target':'TP3 normalized physical residuals at +12 and +60 observed row steps',
            'no_original_checkpoint_training':True,'not_new_scientific_test':True,
            'immutable_original_checkpoints_verified':6,'models':{},
            'source_span':future_info}
    for kind in ['tac_two_timescale','gru']:
        for item in report['fits']:
            if item['model']!=kind:continue
            ck=torch.load(original/item['checkpoint'],weights_only=True,map_location='cpu')
            model=make_model(kind).eval();model.load_state_dict(ck['state_dict'],strict=True)
            t0=time.perf_counter()
            with torch.inference_mode(): pred=model(torch.from_numpy(X)).numpy()[:,:,TP3_INDEX]
            result['models'][kind+'-'+str(item['seed'])]={
                **metrics(pred,Y),'inference_cpu_seconds':round(time.perf_counter()-t0,6),
                'checkpoint_sha256':item['checkpoint_sha256'],'parameters':item['parameter_count']}
    result['models']['persistence']=metrics(np.zeros_like(Y),Y)
    write_json(root/'tac-control-results.json',result)
    print(json.dumps({'status':'TAC_BASELINE_PASS','models':result['models'],
                      'n_orig_future_windows':len(xx),'n_scored':len(chosen)},sort_keys=True),flush=True)

def fm3(root):
    import torch
    from timesfm3 import TimesFM3Evaluator,ModelConfig
    torch.set_num_threads(2)
    p=pathlib.Path(root);d=np.load(p/'fixed-inputs.npz')
    x=d['eval_x'];y=d['eval_y']
    contexts=[np.ascontiguousarray(s[:,:SENSORS].T,dtype=np.float32) for s in x]
    forecaster=TimesFM3Evaluator(ModelConfig(checkpoint_path=TIMESFM_ID,per_core_batch_size=2,device='cpu'))
    start=time.perf_counter()
    outputs=list(forecaster.predict_batch(contexts=contexts,horizon=60,return_quantiles=True,use_symmetric_averaging=False))
    if len(outputs)!=len(x):raise ValueError('Unmatched TimesFM sample count')
    preds=[]
    for i,o in enumerate(outputs):
        a=np.asarray(o.forecast,dtype=np.float64)
        if a.shape!=(SENSORS,60):raise ValueError('Wrong official TimesFM forecast shape '+str(a.shape))
        preds.append([a[TP3_INDEX,11]-x[i,-1,TP3_INDEX],a[TP3_INDEX,59]-x[i,-1,TP3_INDEX]])
    result={'model_id':TIMESFM_ID,'status':'MEASURED_OFFICIAL_PRETRAINED',
            **metrics(preds,y),'inference_cpu_seconds':round(time.perf_counter()-start,3),
            'horizon_mapping':{'12':11,'60':59},'context_shape':[12,32]}
    write_json(p/'timesfm3-results.json',result);print(json.dumps(result),flush=True)

def tabfm(root):
    import pandas as pd
    from tabfm import TabFMRegressor,tabfm_v1_0_0_pytorch
    p=pathlib.Path(root);d=np.load(p/'fixed-inputs.npz')
    Xtr=d['train_x'][:,:,:SENSORS];Xte=d['eval_x'][:,:,:SENSORS]
    Ytr=d['train_y'];Yte=d['eval_y']
    # All models consume 12 sensor streams; TabFM gets 384 flattened values.
    train=pd.DataFrame(Xtr.reshape(len(Xtr),-1),dtype=np.float32)
    test=pd.DataFrame(Xte.reshape(len(Xte),-1),columns=train.columns,dtype=np.float32)
    import torch
    torch.set_num_threads(2)
    model=tabfm_v1_0_0_pytorch.load(model_type='regression')
    pred=[]
    t0=time.perf_counter()
    for j in range(2):
        reg=TabFMRegressor(model=model)
        targets=Ytr[:,j]+Xtr[:,-1,TP3_INDEX]
        reg.fit(train,targets)
        level=np.asarray(reg.predict(test),dtype=np.float64).reshape(-1)
        pred.append(level-Xte[:,-1,TP3_INDEX])
    preds=np.stack(pred,axis=1)
    result={'model_id':TABFM_ID,'status':'MEASURED_OFFICIAL_PRETRAINED',
        **metrics(preds,Yte),'inference_cpu_seconds':round(time.perf_counter()-t0,3),
        'training_context_rows':len(Xtr),'flattened_sensor_features':Xtr.shape[1]*Xtr.shape[2],
        'per_horizon_refit_of_context_encoder':True}
    write_json(p/'tabfm10-results.json',result);print(json.dumps(result),flush=True)

def main():
    q=argparse.ArgumentParser();q.add_argument('--stage',choices=('pack','timesfm3','tabfm10'),required=True)
    q.add_argument('--original',default='.');q.add_argument('--out',default='results/bench-fm-001');a=q.parse_args()
    if a.stage=='pack':pack(a.original,a.out)
    elif a.stage=='timesfm3':fm3(a.out)
    else:tabfm(a.out)

if __name__=='__main__':main()
