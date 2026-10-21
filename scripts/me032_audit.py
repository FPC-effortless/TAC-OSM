"""ME032 immutable replay and source/gradient-allocation verification."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np
import torch
from torch.profiler import profile,ProfilerActivity
from tac_osm.me032_conditional_memory import FrozenContext,CommonHead,COUNTS
from scripts.me032_conditional_research import (
    ROOT,PROTOCOL,ARMS,context_features,build_student,evaluate,predict_student,
    subset,targets,roles_manifest,load_source,file_sha,write_json,verify_parent
)
from scripts.me025_real_transformer_parity import fit_only_popularity
from scripts.me014b_exact_action_memory import batch


def verify(out):
    record=json.loads((out/'results.json').read_text())
    assert record['complete'] and len(record['cells'])==9
    assert record['all_models_frozen_before_evaluation']
    assert not record['historical_DEV_scored'] and not record['reserved_cohorts_opened']
    assert record['protocol_sha256']==file_sha(PROTOCOL)
    for name,digest in json.loads((out/'frozen-payloads.json').read_text()).items():
        if file_sha(out/name)!=digest:raise ValueError('frozen artifact hash mismatch: '+name)
    index=out/'payload-index.json'
    if not index.exists():write_json(index,{p.name:file_sha(p) for p in out.iterdir() if p.is_file() and p.name!='replay.json'})
    for name,digest in json.loads(index.read_text()).items():
        if file_sha(out/name)!=digest:raise ValueError('final artifact hash mismatch: '+name)
    return record


def optimizer_receipt(out,name,ck,rows,steps,seed):
    assert ck['step']==steps
    assert {float(x['step']) for x in ck['optimizer']['state'].values()}=={float(steps)}
    gen=torch.Generator().manual_seed(seed+300)
    for _ in range(steps):torch.randint(rows,(128,),generator=gen)
    assert torch.equal(ck['sampler_rng'],gen.get_state())
    trace=json.loads((out/(name+'-trace.json')).read_text())
    assert [t['step'] for t in trace]==list(range(1,steps+1))
    assert all(t['cpu_seconds']>=0 and np.isfinite(t['loss']) for t in trace)
    return trace


def audit(out,quick=False):
    out=out.resolve();record=verify(out);torch.set_num_threads(1)
    parent_root=ROOT/'results/me031';parent=verify_parent(parent_root)
    assert record['parent_results_sha256']==file_sha(parent_root/'results.json')
    data=load_source(ROOT/'results/me022')['train'];roles=roles_manifest(data)
    assert roles==json.loads((out/'roles.json').read_text())
    features=0;receipts=0;paths=0;exact=0;teacher_forecasts=0;teacher_exact=0;fresh=[];resources=[]
    with tempfile.TemporaryDirectory() as td:
        tmp=Path(td);work=tmp/'replay';work.mkdir()
        for p in out.iterdir():
            if p.suffix=='.pt':(work/p.name).symlink_to(p)
        for cell in record['cells']:
            tag=cell['tag'];role=roles[cell['rotation']]
            r,s,t=[subset(data,torch.tensor(role[k]['indices'])) for k in ('representation','readout','evaluation')]
            pc=parent_root/(cell['parent_encoder_name']+'.pt')
            assert file_sha(pc)==cell['parent_encoder_sha256']
            pck=torch.load(pc,weights_only=True)
            assert float(r['duration'].mean())==pck['mean']
            assert max(float(r['duration'].std(unbiased=False)),1e-5)==pck['scale']
            assert torch.equal(fit_only_popularity(r),pck['popularity'])
            context=FrozenContext(pck['state_dict']['emb.weight'],pck['mean'],pck['scale'],pck['popularity'],torch.zeros(26),torch.ones(26))
            raw=context_features(torch.jit.script(context),r)
            context.feature_mean.copy_(raw[...,:26].reshape(-1,26).mean(0))
            context.feature_scale.copy_(raw[...,:26].reshape(-1,26).std(0,unbiased=False).clamp_min(.05))
            context=torch.jit.script(context).eval()
            cache=torch.load(out/(tag+'-context.pt'),weights_only=True)
            for name,v in context.state_dict().items():assert torch.equal(v,cache['state_dict'][name])
            for name,part in (('R',r),('S',s),('T',t)):
                rebuilt=context_features(context,part)
                torch.testing.assert_close(rebuilt,cache[name],rtol=0,atol=1e-7)
                features+=rebuilt.numel()
            exported_context=torch.jit.load(str(out/('context-'+tag+'.jit.pt')))
            sample=batch(t,slice(0,4))
            with torch.no_grad():torch.testing.assert_close(context(*sample),exported_context(*sample),rtol=0,atol=1e-7)
            teacher=CommonHead();tc=torch.load(out/(cell['teacher']['name']+'.pt'),weights_only=True)
            teacher.load_state_dict(tc['state_dict'],strict=True);teacher.eval()
            optimizer_receipt(out,cell['teacher']['name'],tc,len(r['y']),500,cell['teacher']['seed']);receipts+=1
            with torch.no_grad():
                tp=(cache['T'][:,-1,26]+teacher(cache['T'][:,-1,:26])).sigmoid().numpy().astype(np.float64)
            original_tp=np.load(out/('teacher-'+tag+'-predictions.npy'))
            np.testing.assert_allclose(tp,original_tp,rtol=0,atol=1e-7)
            teacher_forecasts+=len(tp);teacher_exact+=int(np.array_equal(tp,original_tp))
            for student in cell['students']:
                ck=torch.load(out/(student['name']+'.pt'),weights_only=True)
                net=torch.jit.script(build_student(student['arm'],tc['state_dict'],student['seed']))
                net.load_state_dict(ck['state_dict'],strict=True);net.eval()
                assert sum(p.numel() for p in net.parameters() if p.requires_grad)==COUNTS[student['arm']]
                trace=optimizer_receipt(out,student['name'],ck,len(s['y']),300,student['seed']);receipts+=1
                count=sum(tr['memory_gradient_norm_after_clip']>0 for tr in trace)
                assert count==student['positive_memory_gradient_steps'] and count>0
                changed=any(not torch.equal(v,net.common.state_dict()[k]) for k,v in tc['state_dict'].items())
                assert changed==(student['arm']=='joint_native')
                with torch.no_grad():assert torch.equal(net.correction(torch.zeros(4,4,64),cache['T'][:4,:,:26]),torch.zeros(4,4))
                actual=predict_student(net,t,cache['T']);expected=np.load(out/(student['arm']+'-'+tag+'-predictions.npy'))
                np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-7)
                exact+=int(np.array_equal(actual,expected));paths+=len(actual)
                if not quick and cell['rotation']==0 and cell['seed']==14001:
                    # Operator subtotal of the actual S-step graph, including memory.
                    xi=s['items'][:8];xm=s['metadata'][:8];xc=cache['S'][:8];y=targets(s)[:8]
                    net.train();net.zero_grad(set_to_none=True);net(xi,xm,xc).sum().backward();net.zero_grad(set_to_none=True)
                    with profile(activities=[ProfilerActivity.CPU],with_flops=True) as prof:
                        loss=torch.nn.functional.binary_cross_entropy_with_logits(net(xi,xm,xc),y)
                        loss.backward()
                    net.eval();inp=batch(t,slice(0,128));samples=[]
                    with torch.no_grad():
                        for _ in range(3):net(inp[0],inp[1],context(*inp))
                        for _ in range(9):
                            start=time.perf_counter();net(inp[0],inp[1],context(*inp));samples.append(time.perf_counter()-start)
                    resources.append({'arm':student['arm'],'S_step_forward_backward_FLOP_subtotal_B8_four_targets':sum(e.flops or 0 for e in prof.key_averages()),
                        'full_FLOPs':False,'optimizer_counted':False,'frozen_context_precomputation_counted':False,
                        'pipeline_B128_latency_median_seconds':float(np.median(samples)),
                        'pipeline_B128_latency_samples_seconds':samples,
                        'limitation':'Operator coverage omits unsupported/fused/nonlinear/elementwise work; not a full compute-parity audit.'})
                if not quick:
                    inp=tmp/'inputs.pt';torch.save(batch(t,slice(0,4)),inp);output=tmp/'actual.npy'
                    code="""import sys,torch,numpy as np
assert not any(n.startswith(('tac_osm','scripts')) for n in sys.modules)
torch.set_num_threads(1)
x=torch.load(sys.argv[1],weights_only=True)
c=torch.jit.load(sys.argv[2]).eval();m=torch.jit.load(sys.argv[3]).eval()
with torch.no_grad():np.save(sys.argv[4],m(x[0],x[1],c(*x))[:,-1].sigmoid().numpy())
assert not any(n.startswith(('tac_osm','scripts')) for n in sys.modules)
"""
                    subprocess.run([sys.executable,'-c',code,str(inp),str(out/('context-'+tag+'.jit.pt')),str(out/(student['name']+'.jit.pt')),str(output)],
                        check=True,cwd=tmp,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    pp=np.load(output);np.testing.assert_allclose(pp,expected[:4],rtol=0,atol=1e-6)
                    fresh.append({'name':student['name'],'forecasts':4,'max_abs_error':float(np.max(np.abs(pp-expected[:4])))})
        rebuilt=evaluate(work,data,roles,record);assert rebuilt==record['evaluation']
        exported_paths=0;exported_forecasts=0
        for p in work.glob('*-predictions.npy'):
            np.testing.assert_array_equal(np.load(p),np.load(out/p.name));exported_paths+=1;exported_forecasts+=len(np.load(p))
        corrupt=tmp/'corrupt';corrupt.mkdir()
        for p in out.iterdir():
            if p.is_file():(corrupt/p.name).symlink_to(p)
        bad=corrupt/(record['cells'][0]['students'][0]['name']+'.pt');bad.unlink();bad.write_bytes(b'corrupt')
        proc=subprocess.run([sys.executable,'-m','scripts.me032_audit','--out',str(corrupt),'--verify-only'],capture_output=True,text=True)
        assert proc.returncode and 'artifact hash mismatch' in proc.stderr
    result={'status':'PASS','rebuilt_context_values':features,'optimizer_RNG_clock_receipts':receipts,
        'checkpoint_student_forecasts':paths,'bit_exact_checkpoint_paths':exact,
        'checkpoint_teacher_forecasts':teacher_forecasts,'bit_exact_teacher_checkpoint_paths':teacher_exact,
        'export_reference_forecasts':exported_forecasts,'bit_exact_export_reference_paths':exported_paths,
        'fresh_Torch_only_pipeline_checks':fresh,'resource_subtotals':resources,
        'zero_state_anchor_verified':True,'frozen_joint_common_behavior_verified':True,
        'positive_memory_gradient_verified':True,'corruption_before_source':True,
        'historical_DEV_scored':False,'reserved_cohorts_opened':False}
    write_json(out/('verification.json' if quick else 'replay.json'),result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/me032');p.add_argument('--quick',action='store_true');p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    if a.verify_only:verify(Path(a.out))
    else:audit(Path(a.out),a.quick)
