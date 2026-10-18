"""Retrospective only: explain opened REAL001 errors; never qualify a new fit."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from tac_osm.development.maintenance_forecast import MaintenanceForecast
from run_real_maintenance_001 import FEATURES
from data_real_maintenance_002 import write


@torch.no_grad()
def run(directory,output):
    torch.set_num_threads(1);root=Path(directory);train=torch.load(root/'_datasets/train.pt',weights_only=True);test=torch.load(root/'_datasets/test.pt',weights_only=True);std=train['x'].std((0,1)).clamp_min(.01).double();naive=(test['y'].double()-test['x'][:,-1].double()).abs().mean(0);scale=(train['y'].double()-train['x'][:,-1].double()).abs().mean(0);low=train['x'].double().amin((0,1));high=train['x'].double().amax((0,1));arms={}
    for arm in ('tac','no_casm','gru','mlp'):
        rows=[]
        for path in sorted(root.glob(f'group-*/{arm}/report.json')):
            r=json.loads(path.read_text());m=MaintenanceForecast(arm,r['spec']['width'],r['spec']['aux'],r['seed']).double().eval();m.load_state_dict(torch.load(path.parent/'agent.pt',weights_only=True));error=[];squared=[]
            for i in range(0,len(test['x']),256):
                pred=m(test['x'][i:i+256].double())['mean'];diff=pred-test['y'][i:i+256].double();error.append(diff.abs());squared.append((diff/std)**2)
            mae=torch.cat(error).mean(0);mse=torch.cat(squared).mean(0);rows.append({'seed':r['seed'],'raw_MAE':mae.tolist(),'context_normalized_MSE':mse.tolist()})
        mean_mae=torch.tensor([r['raw_MAE'] for r in rows],dtype=torch.float64).mean(0);mean_mse=torch.tensor([r['context_normalized_MSE'] for r in rows],dtype=torch.float64).mean(0);arms[arm]={'per_seed':rows,'mean_original_NMSE':float(mean_mse.mean()),'channels':[{'name':name,'analog':i<7,'model_MAE':float(mean_mae[i]),'persistence_MAE':float(naive[i]),'relative_MAE':float(mean_mae[i]/naive[i]) if naive[i]>0 else None,'MASE':float(mean_mae[i]/scale[i]) if scale[i]>1e-6 else None,'NMSE_share':float(mean_mse[i]/mean_mse.sum()),'context_normalized_MSE':float(mean_mse[i])} for i,name in enumerate(FEATURES)]}
    result={'status':'RETROSPECTIVE_OPENED_TEST_ONLY','source_run':str(root),'warning':'Changing metrics does not repair the old model. No new holdout or promotion claim.','train_context_scale':std.tolist(),'train_persistence_MAE':scale.tolist(),'test_target_outside_train_context_range_fraction':((test['y'].double()<low)|(test['y'].double()>high)).double().mean(0).tolist(),'arms':arms};write(output,result);print(json.dumps({'TAC_NMSE':arms['tac']['mean_original_NMSE'],'analog_relative_MAE':{r['name']:r['relative_MAE'] for r in arms['tac']['channels'] if r['analog']},'largest_NMSE_shares':sorted([(r['name'],r['NMSE_share']) for r in arms['tac']['channels']],key=lambda v:-v[1])[:4]}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',default='results/real-maint-001-main');p.add_argument('--output',required=True);a=p.parse_args();run(a.directory,a.output)
