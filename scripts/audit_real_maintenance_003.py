"""Restore every recorded model/reference without refitting or new test access."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch
from run_real_maintenance_003 import ROOT,PROTOCOL,make,evaluate,SimpleReference,UnivariateReference
from data_real_maintenance_003 import write,digest,ROOMS,EXOGENOUS
from audit_identity_dev0615 import compare


def compare_metrics(actual,expected,telemetry):
    compare(json.loads(json.dumps(actual)),expected,telemetry)


@torch.no_grad()
def standalone(folder,report,model,data,destination):
    destination.mkdir(parents=True,exist_ok=True)
    shutil.copy(ROOT/'src/tac_osm/development/thermal_forecast.py',destination/'model.py')
    shutil.copy(ROOT/'scripts/runtime_thermal_forecast_003.py',destination/'runtime.py')
    shutil.copy(folder/'agent.pt',destination/'model.pt')
    write(destination/'metadata.json',dict(spec=report['spec'],seed=report['seed'],
        channels=ROOMS,input_channels=ROOMS+EXOGENOUS,input_units=['degF']*11+['W'],
        temperature_unit='degF',horizon_minutes=[30,60],
        checkpoint_sha256=report['checkpoint_sha256'],fitting_source=report['source'],
        scope='Research room-temperature forecasts, one building; no calibrated probabilities, repair policy or commercial qualification.'))
    (destination/'LICENSE-NOTICE.txt').write_text('Project metadata declaresApache-2.0. Source data: Kim,Leach,Im, Curated Test Fault Data Set,NREL2019, DOI10.25984/1862682, CC-BY4.0. Demo packets contain their public measurements.\n')
    (destination/'README.md').write_text('Requires Python and CPU PyTorch only. Run `python runtime.py < demo-input.jsonl`. Input channel order/units are in metadata.json: ten room temperatures and outside temperature in degreesF, Net_Total_Power inW. Supply8past15-minute readings and corresponding logger minute-of-day. Output is two point-forecast vectors in degreesF, explicitly namedtemperature_F. The frozen protocol mistakenly saidC; training used rawF throughout. Convert a temperature toC with(F-32)*5/9; convert an MAE with5/9only. No model weights or scores were changed by this label correction. This is neither a failure probability nor repair authorization. First predeclared initialization, not best seed.\n')
    requests=[];expected=[]
    for i in range(min(2,len(data['x']))):
        requests.append(dict(values=data['x'][i].tolist(),minute_of_day=data['minutes'][i].tolist()))
        expected.append(dict(channels=ROOMS,horizon_minutes=[30,60],
            temperature_F=model(data['x'][i:i+1],data['minutes'][i:i+1])[0].tolist()))
    text=''.join(json.dumps(row)+'\n' for row in requests)
    (destination/'demo-input.jsonl').write_text(text)
    (destination/'demo-output.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in expected))
    guard="""import sys,runpy,builtins
sys.path.insert(0,'.')
import torch
original=builtins.__import__
def checked(name,globals=None,locals=None,fromlist=(),level=0):
    if level==0 and name.split('.')[0] in ('numpy','tac_osm','investigation_world'):raise ImportError('research import in portable forecast')
    return original(name,globals,locals,fromlist,level)
builtins.__import__=checked
runpy.run_path('runtime.py',run_name='__main__')
"""
    response=subprocess.run([sys.executable,'-I','-B','-c',guard],cwd=destination,input=text,
        text=True,capture_output=True);assert response.returncode==0,response.stderr
    telemetry={'values':0,'max_abs':0.}
    compare([json.loads(line) for line in response.stdout.splitlines()],expected,telemetry)
    assert telemetry['max_abs']<=1e-8
    return len(expected)


@torch.no_grad()
def restore_reference(report,root,normalization):
    name=report['name']
    if name in ('persistence','drift'):return SimpleReference(name)
    path=root/'references'/name/'agent.pt';assert digest(path)==report['checkpoint_sha256']
    model=UnivariateReference() if report['kind']=='univariate' else make(dict(kind='linear',revin=False),0,normalization).double()
    model.load_state_dict(torch.load(path,weights_only=True));return model.eval()


@torch.no_grad()
def run(directory,output,exports=None,smoke=False):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    root=Path(directory);summary=json.loads((root/'summary.json').read_text())
    assert summary['status'] in (('SMOKE_ONLY',) if smoke else ('COMPLETE','VALIDATION_REJECTED'))
    assert summary['protocol_sha256']==digest(PROTOCOL)
    for name,sha in json.loads((root/'manifest.json').read_text()).items():assert digest(root/name)==sha,name
    if summary['status']=='VALIDATION_REJECTED':
        assert not summary['test_opened'] and not (root/'TEST-EXPOSURE.json').exists() and not (root/'_datasets/test.pt').exists()
    train=torch.load(root/'_datasets/train.pt',weights_only=True)
    validation=torch.load(root/'_datasets/validation.pt',weights_only=True)
    normalization=torch.load(root/'_datasets/normalization.pt',weights_only=True)
    scale=normalization['naive_scale'];telemetry={'values':0,'max_abs':0.}
    compare(normalization['location'].tolist(),train['x'].mean((0,1)).tolist(),telemetry)
    compare(normalization['scale'].tolist(),train['x'].std((0,1),unbiased=False).tolist(),telemetry)
    compare(scale.tolist(),(train['y']-train['x'][:,-1,None,:10]).abs().mean(0).tolist(),telemetry)
    test=torch.load(root/'_datasets/test.pt',weights_only=True) if summary['test_opened'] else None
    result=dict(status='RUNNING',checkpoints=0,references=0,portable_predictions=0,telemetry=telemetry,
        test_opened=summary['test_opened'],scope='All recorded models/references restored, no refitting or new2019 parsing. Physical sensing, not repair execution.')
    start=time.process_time();source_files=('src/tac_osm/development/thermal_forecast.py',
        'scripts/data_real_maintenance_003.py','scripts/run_real_maintenance_003.py',
        'scripts/analyze_real_maintenance_003.py','research/real-maint-003/PROTOCOL.md',
        'research/real-maint-003/protocol.json')
    for path in sorted(root.glob('*/**/report.json')):
        # Reference reports have no arm; they are checked below without SGD refits.
        report=json.loads(path.read_text())
        if 'arm' not in report:continue
        folder=path.parent;assert report['status']=='COMPLETE' and report['updates']==(8 if smoke else 1024)
        assert digest(folder/'agent.pt')==report['checkpoint_sha256']
        if not smoke:
            for name in source_files:
                assert subprocess.check_output(['git','show',report['source']+':'+name],cwd=ROOT)==(ROOT/name).read_bytes(),name
        model=make(report['spec'],report['seed'],normalization).double().eval()
        model.load_state_dict(torch.load(folder/'agent.pt',weights_only=True))
        compare_metrics(evaluate(model,validation,scale,True),report['validation'],telemetry)
        if (folder/'test.json').exists():
            assert test is not None
            compare_metrics(evaluate(model,test,scale,True),json.loads((folder/'test.json').read_text()),telemetry)
        if exports and report['seed']==97301+10000000*int(smoke):
            result['portable_predictions']+=standalone(folder,report,model,validation,Path(exports)/report['arm'])
        result['checkpoints']+=1;write(output,result)
    expected=8 if smoke else summary['fits']+summary['pilot_fits']
    assert result['checkpoints']==expected
    reference_selection=json.loads((root/'reference-selection.json').read_text())
    for report in reference_selection['candidates']:
        model=restore_reference(report,root,normalization)
        compare_metrics(evaluate(model,validation,scale),report['validation'],telemetry)
        if test is not None:
            compare_metrics(evaluate(model,test,scale,True),json.loads((root/'references'/report['name']/'test.json').read_text()),telemetry)
        result['references']+=1
    assert result['references']==8 and telemetry['max_abs']<=1e-8
    result.update(status='COMPLETE',cpu_seconds=time.process_time()-start,
        host=dict(python=sys.version,torch=str(torch.__version__),cpu_models=sorted({line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')})))
    write(output,result);print('FORECAST_AUDIT',json.dumps({key:result[key] for key in ('status','checkpoints','references','portable_predictions','telemetry')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--exports');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    try:run(args.directory,args.output,args.exports,args.smoke)
    except Exception:
        from audit_failure_dev062 import preserve
        preserve(args.output);raise
