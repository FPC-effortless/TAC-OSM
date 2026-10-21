"""Replay complete ME031 source/roles/features, original checkpoints and exports."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity
from tac_osm.me031_crossfit_state import CrossfitNativeEncoder, ResidualProbe
from scripts.me031_crossfit_research import (
    ROOT, PROTOCOL, ARMS, load_source, subset, roles_manifest, targets, extract,
    evaluate_frozen, file_sha, write_json, head_logits, fit_only_popularity
)


def verify_payloads(out):
    r=json.loads((out/'results.json').read_text())
    assert r['complete'] and len(r['cells'])==9 and r['all_models_frozen_before_outer_scoring']
    assert not r['historical_DEV_scored'] and not r['reserved_cohorts_opened']
    assert r['protocol_sha256']==file_sha(PROTOCOL)
    for name,digest in json.loads((out/'frozen-payloads.json').read_text()).items():
        if file_sha(out/name)!=digest:raise ValueError('frozen artifact hash mismatch: '+name)
    index_path=out/'payload-index.json'
    if not index_path.exists():
        write_json(index_path,{p.name:file_sha(p) for p in out.iterdir()
                   if p.is_file() and p.name not in ('replay.json','audit.log')})
    for name,digest in json.loads(index_path.read_text()).items():
        if file_sha(out/name)!=digest:raise ValueError('final artifact hash mismatch: '+name)
    return r


def restore_probe(out,name):
    ck=torch.load(out/(name+'.pt'),weights_only=True)
    assert ck['step']==500
    net=torch.jit.script(ResidualProbe(ck['with_state']))
    net.load_state_dict(ck['state_dict'],strict=True);net.eval()
    return net,ck


def check_optimizer(out,name,ck,rows,steps,seed):
    assert ck['step']==steps
    assert {float(v['step']) for v in ck['optimizer']['state'].values()}=={float(steps)}
    sampler=torch.Generator().manual_seed(seed+300)
    for _ in range(steps):torch.randint(rows,(128,),generator=sampler)
    assert torch.equal(sampler.get_state(),ck['sampler_rng'])
    trace=json.loads((out/(name+'-trace.json')).read_text())
    assert [t['step'] for t in trace]==list(range(1,steps+1))
    assert all(t['cpu_seconds']>=0 and np.isfinite(t['loss']) for t in trace)


def audit(out,quick=False):
    out=out.resolve();r=verify_payloads(out)
    torch.set_num_threads(1)
    data=load_source(ROOT/'results/me022')['train']
    roles=json.loads((out/'roles.json').read_text())
    assert roles==roles_manifest(data)
    paths=0;exact=0;features_checked=0;receipts=0;fresh=[];resources=[]
    with tempfile.TemporaryDirectory() as td:
        tmp=Path(td)
        # Evaluation function writes predictions into a new directory; originals immutable.
        work=tmp/'replay';work.mkdir()
        for p in out.iterdir():
            if p.suffix=='.pt':(work/p.name).symlink_to(p)
        for cell in r['cells']:
            tag=cell['tag'];role=roles[cell['rotation']]
            rep,read,test=[subset(data,torch.tensor(role[k]['indices'])) for k in ('representation','readout','evaluation')]
            ec=cell['encoder'];ck=torch.load(out/(ec['name']+'.pt'),weights_only=True)
            mean=float(rep['duration'].mean());scale=max(float(rep['duration'].std(unbiased=False)),1e-5)
            pop=fit_only_popularity(rep)
            assert ck['mean']==mean and ck['scale']==scale and torch.equal(ck['popularity'],pop)
            encoder=torch.jit.script(CrossfitNativeEncoder(mean,scale,pop))
            encoder.load_state_dict(ck['state_dict'],strict=True);encoder.eval()
            exported=torch.jit.load(str(out/(ec['name']+'.jit.pt')))
            check_optimizer(out,ec['name'],ck,len(rep['y']),300,cell['seed']);receipts+=1
            cache=torch.load(out/(tag+'-features.pt'),weights_only=True)
            rebuilt={}
            for name,part in (('readout',read),('evaluation',test)):
                f=extract(encoder,part)
                if name=='readout':
                    mu=f[...,:90].reshape(-1,90).mean(0)
                    sc=f[...,:90].reshape(-1,90).std(0,unbiased=False).clamp_min(.05)
                    torch.testing.assert_close(mu,cache['mean'],rtol=0,atol=1e-7)
                    torch.testing.assert_close(sc,cache['scale'],rtol=0,atol=1e-7)
                f[...,:90]=(f[...,:90]-mu)/sc
                torch.testing.assert_close(f,cache[name],rtol=0,atol=1e-6)
                rebuilt[name]=f;features_checked+=f.numel()
            sample_args=tuple(t[:4] for t in (test['items'],test['metadata'],test['action'],test['duration']))
            with torch.no_grad():
                torch.testing.assert_close(encoder.prefix_features(*sample_args),exported.prefix_features(*sample_args),rtol=0,atol=1e-7)
            base,bck=restore_probe(out,cell['base']['name'])
            check_optimizer(out,cell['base']['name'],bck,len(read['y']),500,cell['base']['seed']);receipts+=1
            f=cache['evaluation'];offset=head_logits(base,f,f[...,90])
            for pr in cell['probes']:
                net,pck=restore_probe(out,pr['name'])
                check_optimizer(out,pr['name'],pck,len(read['y']),500,pr['seed']);receipts+=1
                actual=torch.sigmoid(head_logits(net,f,offset)[:,-1]).numpy().astype(np.float64)
                expected=np.load(out/(pr['arm']+'-'+tag+'-predictions.npy'))
                np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-7)
                paths+=len(actual);exact+=int(np.array_equal(actual,expected))
                if cell['seed']==14001 and cell['rotation']==0 and not quick:
                    probe=ResidualProbe(pr['with_state']);probe.load_state_dict(pck['state_dict']);probe.train()
                    x=cache['readout'][:128];off=torch.zeros((128,4));y=targets(read)[:128]
                    probe(x,off).sum().backward();probe.zero_grad(set_to_none=True)
                    with profile(activities=[ProfilerActivity.CPU],with_flops=True) as prof:
                        loss=torch.nn.functional.binary_cross_entropy_with_logits(probe(x,off),y)
                        loss.backward()
                    net.eval();tms=[]
                    with torch.no_grad():
                        for _ in range(3):net(x,off)
                        for _ in range(9):
                            t=time.perf_counter();net(x,off);tms.append(time.perf_counter()-t)
                    # Full feature extraction + base + residual; common path deliberately
                    # uses the same encoder implementation rather than skipping recurrence.
                    inp=tuple(v[:128] for v in (test['items'],test['metadata'],test['action'],test['duration']))
                    def pipeline():
                        ff=encoder.prefix_features(*inp)
                        ff[...,:90]=(ff[...,:90]-cache['mean'])/cache['scale']
                        return net(ff,base(ff,ff[...,90]))[:,-1]
                    full=[]
                    with torch.no_grad():
                        for _ in range(3):pipeline()
                        for _ in range(9):
                            tt=time.perf_counter();pipeline();full.append(time.perf_counter()-tt)
                    resources.append({'arm':pr['arm'],
                        'shared_encoder_pipeline_B128_median_seconds':float(np.median(full)),
                        'shared_encoder_pipeline_B128_samples_seconds':full,
                        'shared_encoder_pipeline_note':'Both paths execute native prefix features; no optimized common-only feature extractor claimed.','residual_forward_backward_FLOP_subtotal_B128_four_targets':sum(e.flops or 0 for e in prof.key_averages()),
                        'complete_pipeline_FLOPs':False,'optimizer_included':False,
                        'probe_only_inference_cpu_median_seconds':float(np.median(tms)),
                        'probe_only_inference_samples_seconds':tms,
                        'limitation':'Fixed-feature residual probe only; ignores encoder/base and unsupported nonlinear/elementwise operator work.'})
            if not quick:
                inputs=tmp/'inputs.pt';torch.save({'args':sample_args,'mean':cache['mean'],'scale':cache['scale']},inputs)
                common=next(p for p in cell['probes'] if p['arm']=='common_residual')
                state=next(p for p in cell['probes'] if p['arm']=='state_residual')
                code="""import sys,torch,numpy as np
assert not any(n.startswith(('tac_osm','scripts')) for n in sys.modules)
torch.set_num_threads(1)
z=torch.load(sys.argv[1],weights_only=True)
e,b,c,s=[torch.jit.load(p).eval() for p in sys.argv[2:6]]
with torch.no_grad():
 f=e.prefix_features(*z['args']);f[...,:90]=(f[...,:90]-z['mean'])/z['scale']
 off=b(f,f[...,90]);p=torch.stack((torch.sigmoid(c(f,off)[:,-1]),torch.sigmoid(s(f,off)[:,-1])))
 np.save(sys.argv[6],p.numpy())
assert not any(n.startswith(('tac_osm','scripts')) for n in sys.modules)
"""
                output=tmp/'actual.npy'
                args=[sys.executable,'-c',code,str(inputs)]+[str(out/(n+'.jit.pt')) for n in (ec['name'],cell['base']['name'],common['name'],state['name'])]+[str(output)]
                subprocess.run(args,check=True,cwd=tmp,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                expected=np.stack([np.load(out/(a+'-'+tag+'-predictions.npy'))[:4] for a in ARMS])
                actual=np.load(output);np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-6)
                fresh.append({'tag':tag,'forecasts':8,'max_abs_error':float(np.max(np.abs(actual-expected)))})
        actual_eval=evaluate_frozen(work,data,roles,r)
        assert actual_eval==r['evaluation']
        export_forecasts=0;exact_exports=0
        for p in work.glob('*-predictions.npy'):
            actual=np.load(p);expected=np.load(out/p.name)
            np.testing.assert_array_equal(actual,expected)
            export_forecasts+=len(actual);exact_exports+=1
        # A changed byte must fail before source validation or feature extraction.
        corrupt=tmp/'corrupt';corrupt.mkdir()
        for p in out.iterdir():
            if p.is_file():(corrupt/p.name).symlink_to(p)
        cp=corrupt/(r['cells'][0]['encoder']['name']+'.pt');cp.unlink();cp.write_bytes(b'corrupt')
        proc=subprocess.run([sys.executable,'-m','scripts.me031_audit','--out',str(corrupt),'--verify-only'],capture_output=True,text=True)
        assert proc.returncode and 'artifact hash mismatch' in proc.stderr
    result={'status':'PASS','features_rebuilt_and_checked':features_checked,'optimizer_RNG_clock_receipts':receipts,
        'checkpoint_residual_forecasts':paths,'bit_exact_checkpoint_paths':exact,
        'export_and_reference_forecasts':export_forecasts,'bit_exact_export_reference_paths':exact_exports,
        'fresh_torch_only_pipeline_checks':fresh,'resource_subtotals':resources,
        'corruption_rejected_before_source':True,'historical_DEV_scored':False,'reserved_cohorts_opened':False}
    write_json(out/('verification.json' if quick else 'replay.json'),result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/me031');p.add_argument('--quick',action='store_true');p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    if a.verify_only:verify_payloads(Path(a.out))
    else:audit(Path(a.out),a.quick)
