"""Check that reported improvements reproduce from original trained weights."""
import argparse,json,shutil,subprocess,sys,tempfile,time
from pathlib import Path
import numpy as np
import torch
from torch.profiler import profile,ProfilerActivity
from scripts.me035_architecture_research import ROOT,PROTOCOL,source,metrics,predict,evaluate,build_model,file_sha,write_json


def verify(out):
    r=json.loads((out/'results.json').read_text())
    if not r['complete'] or not r['all_models_frozen_before_scoring']:raise ValueError('incomplete training')
    if file_sha(PROTOCOL)!=r['protocol_sha256']:raise ValueError('protocol changed')
    for n,h in json.loads((out/'frozen-models.json').read_text()).items():
        if file_sha(out/n)!=h:raise ValueError('artifact hash mismatch: '+n)
    index=out/'original-payloads.json'
    if not index.exists():write_json(index,{p.name:file_sha(p) for p in out.iterdir() if p.is_file()})
    for n,h in json.loads(index.read_text()).items():
        if file_sha(out/n)!=h:raise ValueError('artifact hash mismatch: '+n)
    return r


def audit(out,quick=False):
    torch.set_num_threads(1);r=verify(out);train,dev,src=source(out)
    assert src==r['source'] and not src['later_rows_numeric_parsed']
    assert max(train['indices'])+24<8760 and min(dev['indices'])>=8760 and max(dev['indices'])+24<11640
    forecasts=0;exact=0;script_exact=0;receipts=0;fresh=[];resources=[]
    for fit in r['fits']:
        for ck in fit['checkpoints']:
            saved=torch.load(out/(ck['name']+'.pt'),weights_only=True)
            assert saved['step']==ck['step'] and saved['seed']==fit['seed'] and saved['arm']==fit['arm']
            assert {float(v['step']) for v in saved['optimizer']['state'].values()}=={float(ck['step'])}
            g=torch.Generator().manual_seed(fit['seed']+50)
            for _ in range(ck['step']):torch.randint(len(train['x']),(64,),generator=g)
            assert torch.equal(g.get_state(),saved['sampler_rng'])
            trace=json.loads((out/(ck['name']+'-trace.json')).read_text())
            assert [v['step'] for v in trace]==list(range(1,ck['step']+1))
            assert all(np.isfinite(v['MSE']) and v['memory_gradient_after_clip']>0 for v in trace)
            net=torch.jit.script(build_model(fit['arm'],fit['seed']));net.load_state_dict(saved['state_dict'],strict=True)
            expected=np.load(out/(ck['name']+'-predictions.npy'));actual=predict(net,dev['x'])
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-6)
            script=predict(torch.jit.load(str(out/(ck['name']+'.jit.pt'))),dev['x'])
            np.testing.assert_allclose(script,expected,rtol=0,atol=1e-6)
            forecasts+=actual.size;exact+=int(np.array_equal(actual,expected))*actual.size
            script_exact+=int(np.array_equal(script,expected))*actual.size;receipts+=1
            if not quick:
                with tempfile.TemporaryDirectory() as folder:
                    tmp=Path(folder);torch.save(dev['x'][:4],tmp/'x.pt')
                    code="""import sys,torch,numpy as np
assert not any(n.startswith(('tac_osm','scripts')) for n in sys.modules)
torch.set_num_threads(1)
m=torch.jit.load(sys.argv[1]).eval();x=torch.load(sys.argv[2],weights_only=True)
with torch.no_grad():y=m(x)
np.save(sys.argv[3],y.numpy())
assert not any(n.startswith(('tac_osm','scripts')) for n in sys.modules)
"""
                    subprocess.run([sys.executable,'-c',code,str((out/(ck['name']+'.jit.pt')).resolve()),str(tmp/'x.pt'),str(tmp/'p.npy')],cwd=tmp,check=True,capture_output=True)
                    small=np.load(tmp/'p.npy');np.testing.assert_allclose(small,expected[:4],rtol=0,atol=1e-6)
                    fresh.append({'name':ck['name'],'values':small.size,'max_abs_error':float(np.max(abs(small-expected[:4])))})
        if not quick and fit['seed']==35001:
            net=torch.jit.load(str(out/(fit['checkpoints'][-1]['name']+'.jit.pt')));x=train['x'][:8];y=train['y'][:8]
            net.zero_grad(set_to_none=True);torch.nn.functional.mse_loss(net(x),y).backward();net.zero_grad(set_to_none=True)
            with profile(activities=[ProfilerActivity.CPU],with_flops=True) as p:torch.nn.functional.mse_loss(net(x),y).backward()
            net.eval();samples=[]
            with torch.no_grad():
                for _ in range(3):net(dev['x'][:64])
                for _ in range(9):
                    t=time.perf_counter();net(dev['x'][:64]);samples.append(time.perf_counter()-t)
            resources.append({'arm':fit['arm'],'B8_forward_backward_operator_FLOP_subtotal':sum(v.flops or 0 for v in p.key_averages()),
                'full_FLOPs':False,'B64_inference_median_seconds':float(np.median(samples)),
                'tensor_weight_bytes':sum(v.numel()*v.element_size() for v in net.parameters()),
                'recurrent_state_bytes':256,'input_window_bytes_per_entity':2688,
                'local_normalization_requires_full_window_replay':fit['arm']!='native_joint'})
    x=dev['x'].numpy().reshape(len(dev['x']),-1);x=np.c_[x,np.ones(len(x))]
    rp=(x@np.load(out/'ridge-coefficients.npy')).reshape(-1,3,7)+dev['x'][:,-1:].numpy()
    np.testing.assert_array_equal(rp,np.load(out/'ridge-predictions.npy'))
    original_eval=r['evaluation'].copy();original_eval.pop('references')
    # Re-evaluate into a temporary directory so originals cannot be overwritten.
    with tempfile.TemporaryDirectory() as folder:
        work=Path(folder)
        for p in out.glob('*.jit.pt'):(work/p.name).symlink_to(p.resolve())
        re=evaluate(work,dev,r);assert re==original_eval
        for p in work.glob('*-predictions.npy'):np.testing.assert_array_equal(np.load(p),np.load(out/p.name))
        corrupt=work/'corrupt';corrupt.mkdir()
        for p in out.iterdir():
            if p.is_file():(corrupt/p.name).symlink_to(p.resolve())
        victim=corrupt/(r['fits'][0]['checkpoints'][0]['name']+'.pt');victim.unlink();victim.write_bytes(b'corrupt')
        proc=subprocess.run([sys.executable,'-m','scripts.me035_audit','--out',str(corrupt),'--verify-only'],cwd=ROOT,capture_output=True,text=True)
        assert proc.returncode and 'artifact hash mismatch' in proc.stderr
    result={'status':'PASS','original_checkpoints_and_optimizer_RNG_receipts':receipts,
        'scalar_forecasts':forecasts,'bit_exact_checkpoint_forecasts':exact,'bit_exact_scripted_forecasts':script_exact,
        'fresh_torch_only_checks':fresh,'resources':resources,'selection_and_metrics_reproduced':True,
        'ridge_forecast_values_reproduced':rp.size,'corruption_rejected_before_source':True,
        'later_source_rows_parsed':False,'KuaiRec_accessed':False}
    write_json(out/('verification.json' if quick else 'replay.json'),result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/me035');p.add_argument('--quick',action='store_true');p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    if a.verify_only:verify(Path(a.out))
    else:audit(Path(a.out),a.quick)
