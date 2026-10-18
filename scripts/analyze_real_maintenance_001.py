"""Single-machine real forecasting comparisons; four faults are not twenty fleets."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from run_real_maintenance_001 import seconds,FEATURES
from run_learned_stack_dev055 import write
from summarize_memory_dev061 import interval,signflip,holm
from ridge_real_maintenance_001 import fit_ridge,predict_ridge
import time
FAILURES=[('2020-04-18','2020-04-18 23:59:00'),('2020-05-29 23:30:00','2020-05-30 06:00:00'),('2020-06-05 10:00:00','2020-06-07 14:30:00'),('2020-07-15 14:30:00','2020-07-15 19:00:00')]


def anomalies(report,test):
    t=np.array(test['time']);flags=np.array(test['squared_residual'])>report['anomaly_threshold'];known=np.zeros(len(t),bool);events=[]
    for start,end in FAILURES:
        a,b=seconds(start),seconds(end);inside=(t>=a)&(t<=b);before=(t>=a-86400)&(t<a);known|=inside;events.append({'start':start,'end':end,'covered_windows':int(inside.sum()),'observed_detection':bool(flags[inside].any()),'pre24h_alert':bool(flags[before].any())})
    return {'events':events,'outside_reported_windows_alert_fraction':float(flags[~known].mean()),'outside_reported_windows_health':'UNKNOWN, not verified negatives','threshold':report['anomaly_threshold']}


def run(directory,output):
    torch.set_num_threads(1);root=Path(directory);summary=json.loads((root/'summary.json').read_text());assert summary['status']=='COMPLETE' and summary['test_opened'];train=torch.load(root/'_datasets/train.pt',weights_only=True);test=torch.load(root/'_datasets/test.pt',weights_only=True);mean=train['x'].mean((0,1)).double();scale=train['x'].std((0,1)).clamp_min(.01).double();y=(test['y'].double()-mean)/scale;persistence=(test['x'][:,-1].double()-mean)/scale;baseline={'persistence_normalized_MSE':float(((y-persistence)**2).mean()),'train_mean_normalized_MSE':float((y**2).mean())};tables={a:{} for a in ('tac','no_casm','gru','mlp')}
    validation=torch.load(root/'_datasets/validation.pt',weights_only=True);started=time.process_time();ridge,candidates=fit_ridge(train,validation);ridge_cpu=time.process_time()-started;torch.save(ridge,Path(output).with_name('ridge.pt'));ridge_mse=float(((predict_ridge(ridge,test['x'])-y)**2).mean());baseline['ridge']={'normalized_MSE':ridge_mse,'selected_lags':ridge['lags'],'selected_penalty':ridge['penalty'],'validation_candidates':candidates,'fitting_cpu_seconds':ridge_cpu,'coefficients':ridge['weights'].numel()}
    for a in tables:
        for seed in range(92701,92721):
            folder=root/f'group-{seed}'/a;r=json.loads((folder/'report.json').read_text());t=json.loads((folder/'test.json').read_text());tables[a][seed]={'normalized_MSE':t['normalized_MSE'],'Gaussian_NLL':t['Gaussian_NLL'],'per_feature_normalized_MSE':dict(zip(FEATURES,t['per_feature_normalized_MSE'],strict=True)),'anomalies':anomalies(r,t)}
    comparisons={}
    for a in ('no_casm','gru','mlp'):
        delta=[tables['tac'][s]['normalized_MSE']-tables[a][s]['normalized_MSE'] for s in tables[a]];comparisons[a]={'MSE':interval(delta),'p':signflip(delta)}
    adjusted=holm({a:v['p'] for a,v in comparisons.items()})
    for a,v in comparisons.items():v['holm_p']=adjusted[a]
    vs_persistence=interval([t['normalized_MSE']-baseline['persistence_normalized_MSE'] for t in tables['tac'].values()]);ridge_delta=interval([t['normalized_MSE']-ridge_mse for t in tables['tac'].values()]);gates={'no_worse_than_ridge':ridge_delta['paired_seed_bootstrap95'][1]<=0,'better_than_persistence':vs_persistence['paired_seed_bootstrap95'][1]<0,'better_than_GRU_MLP':all(comparisons[a]['MSE']['paired_seed_bootstrap95'][1]<0 and comparisons[a]['holm_p']<.05 for a in ('gru','mlp')),'physical_repair_validation':'NOT_AVAILABLE','commercial_cost_validation':'NOT_AVAILABLE'}
    aggregate={a:{'mean_normalized_MSE':float(np.mean([m['normalized_MSE'] for m in v.values()])),'mean_Gaussian_NLL':float(np.mean([m['Gaussian_NLL'] for m in v.values()])),'per_seed':v} for a,v in tables.items()};write(output,{'status':'COMPLETE','run':summary,'total_fitting_cpu_including_ridge':summary['fitting_cpu_seconds']+ridge_cpu,'source':json.loads((root/'_datasets/provenance.json').read_text()),'baselines':baseline,'arms':aggregate,'comparisons':comparisons,'versus_ridge':ridge_delta,'versus_persistence':vs_persistence,'gates':gates,'promotion':'NOT_ESTABLISHED','commercial':'NOT_ESTABLISHED','scope':'One real machine, four reported fault intervals, twenty training seeds conditional on the same time series. No repair policy, causal effect or actual intervention costs. Published May maintenance timestamp inconsistent; not corrected.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.directory,a.output)
