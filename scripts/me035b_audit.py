"""Replay the learned AR/native refinement and export raw-unit models."""
import argparse,json,tempfile,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from tac_osm.me035_ar_backbone import initialize
from tac_osm.me035_forecast_runtime import PhysicalForecast
from scripts.me035b_backbone_research import ROOT,PROTOCOL,evaluate
from scripts.me035_architecture_research import source,metrics,predict,file_sha,write_json


def verify(out):
    r=json.loads((out/'results.json').read_text())
    assert r['complete'] and r['all_models_frozen_before_scoring'] and r['protocol_sha256']==file_sha(PROTOCOL)
    for n,h in json.loads((out/'frozen-models.json').read_text()).items():
        if file_sha(out/n)!=h:raise ValueError('artifact hash mismatch: '+n)
    index=out/'original-payloads.json'
    if not index.exists():write_json(index,{p.name:file_sha(p) for p in out.iterdir() if p.is_file()})
    for n,h in json.loads(index.read_text()).items():
        if file_sha(out/n)!=h:raise ValueError('artifact hash mismatch: '+n)
    ex=out/'raw-runtime-exports.json'
    if ex.exists():
        for m in json.loads(ex.read_text())['models']:
            if file_sha(out/m['name'])!=m['sha256']:raise ValueError('runtime artifact hash mismatch')
    return r


def audit(out,quick=False):
    torch.set_num_threads(1);r=verify(out);parent=ROOT/'results/me035'
    assert file_sha(parent/'results.json')==r['parent_results_sha256']
    assert file_sha(parent/'ridge-coefficients.npy')==r['ridge_coefficients_sha256']
    train,dev,src=source(parent);assert src==r['source']
    coef=np.load(parent/'ridge-coefficients.npy');count=exact=sexact=0;receipts=0;fresh=[];rawmodels=[];initial_checks=[]
    raw=dev['x']*torch.tensor(src['scale'])+torch.tensor(src['mean'])
    for fit in r['fits']:
        old=parent/fit['parent_checkpoint'];assert file_sha(old)==fit['parent_sha256']
        initial=torch.load(old,weights_only=True)['state_dict']
        for ck in fit['checkpoints']:
            c=torch.load(out/(ck['name']+'.pt'),weights_only=True);assert c['step']==ck['step'] and c['seed']==fit['seed']
            assert {float(v['step']) for v in c['optimizer']['state'].values()}==({float(ck['step'])} if ck['step'] else set())
            g=torch.Generator().manual_seed(fit['seed']+350)
            for _ in range(ck['step']):torch.randint(len(train['x']),(64,),generator=g)
            assert torch.equal(g.get_state(),c['sampler_rng'])
            net=initialize(initial,coef,fit['seed']);net.load_state_dict(c['state_dict'],strict=True)
            n=torch.jit.script(net);expected=np.load(out/(ck['name']+'-predictions.npy'))
            a=predict(n,dev['x']);b=predict(torch.jit.load(str(out/(ck['name']+'.jit.pt'))),dev['x'])
            np.testing.assert_allclose(a,expected,rtol=0,atol=1e-6);np.testing.assert_allclose(b,expected,rtol=0,atol=1e-6)
            count+=a.size;exact+=int(np.array_equal(a,expected))*a.size;sexact+=int(np.array_equal(b,expected))*b.size;receipts+=1
            if ck['step']==0:
                error=np.abs(a-np.load(parent/'ridge-predictions.npy'))
                initial_checks.append({'seed':fit['seed'],'registered_atol':1e-6,'max_abs_error':float(error.max()),
                    'values_exceeding_registered_tolerance':int((error>1e-6).sum()),'PASS':bool((error<=1e-6).all())})
            if ck['step']==r['evaluation']['selected_step']:
                name=f'raw-ar-native-s{fit["seed"]}.jit.pt'
                pipe=torch.jit.script(PhysicalForecast(net,torch.tensor(src['mean']),torch.tensor(src['scale']))).eval()
                if not (out/name).exists():pipe.save(str(out/name))
                pipeline=torch.jit.load(str(out/name)).eval()
                rawpred=predict(pipeline,raw);reference=expected*np.array(src['scale'],np.float32)+np.array(src['mean'],np.float32)
                np.testing.assert_allclose(rawpred,reference,rtol=0,atol=1e-4)
                receipt={'name':name,'seed':fit['seed'],'sha256':file_sha(out/name),'physical_forecast_values':a.size,
                    'max_physical_unit_error':float(np.max(abs(rawpred-reference))),'active_trained_parameters':32154}
                if not quick:
                    with tempfile.TemporaryDirectory() as folder:
                        tmp=Path(folder);torch.save(raw[:4],tmp/'x.pt')
                        code='import sys,torch,numpy as np\nassert not any(n.startswith(("tac_osm","scripts")) for n in sys.modules)\ntorch.set_num_threads(1)\nm=torch.jit.load(sys.argv[1]).eval();x=torch.load(sys.argv[2],weights_only=True)\nwith torch.no_grad():y=m(x)\nnp.save(sys.argv[3],y.numpy())\n'
                        subprocess.run([sys.executable,'-c',code,str((out/name).resolve()),str(tmp/'x.pt'),str(tmp/'p.npy')],cwd=tmp,check=True,capture_output=True)
                        np.testing.assert_allclose(np.load(tmp/'p.npy'),reference[:4],rtol=0,atol=1e-4)
                        fresh.append({'seed':fit['seed'],'physical_values':84})
                rawmodels.append(receipt)
    with tempfile.TemporaryDirectory() as folder:
        work=Path(folder)
        for p in out.glob('*.jit.pt'):(work/p.name).symlink_to(p.resolve())
        assert evaluate(work,dev,r)==r['evaluation']
        corrupt=work/'corrupt';corrupt.mkdir()
        for p in out.iterdir():
            if p.is_file():(corrupt/p.name).symlink_to(p.resolve())
        bad=corrupt/(r['fits'][0]['checkpoints'][0]['name']+'.pt');bad.unlink();bad.write_bytes(b'corrupt')
        p=subprocess.run([sys.executable,'-m','scripts.me035b_audit','--out',str(corrupt),'--verify-only'],cwd=ROOT,capture_output=True,text=True)
        assert p.returncode and 'artifact hash mismatch' in p.stderr
    exports={'status':'PASS','input':'float32 Bx96x7 publisher-unit history','output':'Bx3x7 publisher-unit forecast, horizons1/6/24h','models':rawmodels}
    write_json(out/'raw-runtime-exports.json',exports)
    initial_pass=all(c['PASS'] for c in initial_checks)
    result={'status':'PASS' if initial_pass else 'REPLAY_PASS_INITIAL_EQUIVALENCE_TOLERANCE_FAIL',
        'forecast_replay_status':'PASS','original_checkpoints_optimizer_RNG_receipts':receipts,'scalar_forecasts':count,
        'bit_exact_checkpoint_forecasts':exact,'bit_exact_scripted_forecasts':sexact,
        'initial_ridge_equivalence':initial_pass,'initial_ridge_checks':initial_checks,'fresh_raw_Torch_only_checks':fresh,
        'raw_runtime_forecast_values':sum(m['physical_forecast_values'] for m in rawmodels),
        'raw_runtime_max_physical_error':max(m['max_physical_unit_error'] for m in rawmodels),
        'selection_and_metrics_reproduced':True,'corruption_rejected_before_source':True}
    write_json(out/('verification.json' if quick else 'replay.json'),result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('results/me035b'));p.add_argument('--quick',action='store_true');p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    if a.verify_only:verify(a.out)
    else:audit(a.out,a.quick)
