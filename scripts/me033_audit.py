"""Immutable factorial replay, feature provenance and moment-state audit."""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from torch.profiler import profile,ProfilerActivity
from tac_osm.me033_semantic_moments import COUNTS,build_model
from tac_osm.me032_conditional_memory import FrozenContext,split_rms
from scripts.me033_semantic_research import (
    ROOT,PROTOCOL,SEEDS,ARMS,offered_items,predict,evaluate,verify_parent,
    roles_manifest,subset,targets,load_source,file_sha,write_json
)
from scripts.me032_conditional_research import context_features
from scripts.me025_real_transformer_parity import fit_only_popularity
from scripts.me014b_exact_action_memory import batch


def verify(out):
    r=json.loads((out/'results.json').read_text())
    assert r['complete'] and len(r['cells'])==9 and r['all_models_frozen_before_scoring']
    assert not r['historical_DEV_scored'] and not r['reserved_cohorts_opened']
    assert r['protocol_sha256']==file_sha(PROTOCOL)
    for name,digest in json.loads((out/'frozen-payloads.json').read_text()).items():
        if file_sha(out/name)!=digest:raise ValueError('frozen artifact hash mismatch: '+name)
    index=out/'payload-index.json'
    if not index.exists():write_json(index,{p.name:file_sha(p) for p in out.iterdir() if p.is_file() and p.name!='replay.json'})
    for name,digest in json.loads(index.read_text()).items():
        if file_sha(out/name)!=digest:raise ValueError('final artifact hash mismatch: '+name)
    exports=out/'inherited-context-exports.json'
    if exports.exists():
        for name,digest in json.loads(exports.read_text()).items():
            if file_sha(out/name)!=digest:raise ValueError('context artifact hash mismatch: '+name)
    return r


def receipt(out,st,ck,rows):
    assert ck['step']==300 and ck['arm']==st['arm'] and ck['seed']==st['seed']
    assert {float(v['step']) for v in ck['optimizer']['state'].values()}=={300.}
    gen=torch.Generator().manual_seed(st['seed']+300)
    for _ in range(300):torch.randint(rows,(128,),generator=gen)
    assert torch.equal(gen.get_state(),ck['sampler_rng'])
    trace=json.loads((out/(st['name']+'-trace.json')).read_text())
    assert [t['step'] for t in trace]==list(range(1,301))
    assert all(np.isfinite(t['real_loss']) and t['cpu_seconds']>=0 for t in trace)
    assert sum(t['embedding_grad_after_clip']>0 for t in trace)==st['nonzero_embedding_gradient_steps']
    return trace


def audit(out,quick=False):
    out=out.resolve();r=verify(out);torch.set_num_threads(1)
    parent_root=ROOT/'results/me032';parent=verify_parent(parent_root)
    assert file_sha(parent_root/'results.json')==r['parent_results_sha256']
    data=load_source(ROOT/'results/me022')['train'];roles=roles_manifest(data)
    assert roles==json.loads((out/'roles.json').read_text())
    feature_count=0;checkpoint_forecasts=0;checkpoint_exact=0;receipts=0;fresh=[];resources=[]
    stream_checks=[];redundancy=[];exports={}
    with tempfile.TemporaryDirectory() as td:
        tmp=Path(td);work=tmp/'replay';work.mkdir()
        for p in out.iterdir():
            if p.suffix=='.pt':(work/p.name).symlink_to(p)
        for cell in r['cells']:
            tag=cell['tag'];role=roles[cell['rotation']]
            rr,ss,tt=[subset(data,torch.tensor(role[k]['indices'])) for k in ('representation','readout','evaluation')]
            inherited=torch.load(out/(tag+'-inherited.pt'),weights_only=True)
            assert file_sha(parent_root/cell['parent_context_name'])==inherited['parent_context_sha256']
            assert file_sha(parent_root/cell['parent_teacher_name'])==inherited['parent_teacher_sha256']
            original=torch.load(parent_root/cell['parent_context_name'],weights_only=True)
            tc=torch.load(parent_root/cell['parent_teacher_name'],weights_only=True)
            assert all(torch.equal(v,tc['state_dict'][k]) for k,v in inherited['common_state'].items())
            cs=inherited['context_state'];embedding=inherited['embedding']
            assert torch.equal(embedding,cs['emb.weight'])
            assert all(torch.equal(v,original['state_dict'][k]) for k,v in cs.items())
            assert float(rr['duration'].mean())==float(cs['duration_mean'])
            assert max(float(rr['duration'].std(unbiased=False)),1e-5)==float(cs['duration_scale'])
            assert torch.equal(fit_only_popularity(rr),cs['item_popularity'])
            context=FrozenContext(embedding,float(cs['duration_mean']),float(cs['duration_scale']),cs['item_popularity'],torch.zeros(26),torch.ones(26))
            raw=context_features(torch.jit.script(context),rr)
            torch.testing.assert_close(raw[...,:26].reshape(-1,26).mean(0),cs['feature_mean'],rtol=0,atol=1e-7)
            torch.testing.assert_close(raw[...,:26].reshape(-1,26).std(0,unbiased=False).clamp_min(.05),cs['feature_scale'],rtol=0,atol=1e-7)
            context.load_state_dict(cs,strict=True);context=torch.jit.script(context).eval()
            for label,part in (('S',ss),('T',tt)):
                actual=context_features(context,part)
                torch.testing.assert_close(actual,inherited['context_'+label],rtol=0,atol=1e-7)
                feature_count+=actual.numel()
            # Inherited source bytes copied after fitting, not a new fitted/exported context.
            context_name='context-'+tag+'.jit.pt';source=parent_root/context_name
            if not (out/context_name).exists():shutil.copyfile(source,out/context_name)
            assert file_sha(out/context_name)==file_sha(source);exports[context_name]=file_sha(source)
            ct=inherited['context_T'];offered=offered_items(tt)
            for st in cell['students']:
                ck=torch.load(out/(st['name']+'.pt'),weights_only=True)
                net=torch.jit.script(build_model(st['arm'],inherited['common_state'],embedding,st['seed']))
                net.load_state_dict(ck['state_dict'],strict=True);net.eval()
                assert sum(p.numel() for p in net.parameters() if p.requires_grad)==COUNTS[st['arm']]
                receipt(out,st,ck,len(ss['y']));receipts+=1
                changed=any(not torch.equal(v,net.common.state_dict()[k]) for k,v in inherited['common_state'].items())
                assert changed==st['common_head_changed']==(st['arm']=='common_adapt')
                if st['arm']!='common_adapt':
                    name='memory.core.emb.weight' if st['arm']=='native_adapt' else 'memory.emb.weight'
                    echanged=not torch.equal(net.state_dict()[name],embedding)
                    assert echanged==st['embedding_changed']==('frozen' not in st['arm'])
                expected=np.load(out/(st['arm']+'-'+tag+'-predictions.npy'));actual=predict(net,tt,ct)
                np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-7)
                checkpoint_forecasts+=len(actual);checkpoint_exact+=int(np.array_equal(actual,expected))
                if st['arm'] in ('history_adapt','query_adapt','history_frozen','query_frozen'):
                    count=16;items=tt['items'][:count];meta=tt['metadata'][:count];offers=offered[:count]
                    m=net.memory
                    with torch.no_grad():
                        bulk=m(items,meta,offers);state=m.empty(meta);left=m.empty(meta);right=m.empty(meta);latents=[]
                        for j in range(128):
                            state=m.observe(state,items[:,j],meta[:,j])
                            if j<63:left=m.observe(left,items[:,j],meta[:,j])
                            else:right=m.observe(right,items[:,j],meta[:,j])
                            if (j+1)%32==0:latents.append(m.project(state,offers[:,(j+1)//32-1]))
                        stream=torch.stack(latents,1)
                        torch.testing.assert_close(stream,bulk,rtol=0,atol=1e-5)
                        torch.testing.assert_close(m.project(m.merge(left,right),offers[:,-1]),bulk[:,-1],rtol=0,atol=1e-5)
                        x=ct[:count,:,:26];offset=ct[:count,:,26]+net.common(x)
                        stream_p=(offset+net.correction(stream,x)).sigmoid()
                        bulk_p=(offset+net.correction(bulk,x)).sigmoid()
                        torch.testing.assert_close(stream_p,bulk_p,rtol=0,atol=1e-6)
                        stream_checks.append({'name':st['name'],'real_histories':count,'prefixes':4,
                            'component_float32_bytes':16 if st['arm'].startswith('query') else 48,
                            'max_latent_abs_error':float((stream-bulk).abs().max()),
                            'max_prediction_abs_error':float((stream_p-bulk_p).abs().max())})
                        if 'frozen' in st['arm']:
                            rawc=x*cs['feature_scale']+cs['feature_mean']
                            u=torch.cat((rawc[...,:8],rawc[...,23:26]),-1) if st['arm'].startswith('query') else rawc[...,15:26]
                            recovered=split_rms(m.pool(u))
                            torch.testing.assert_close(recovered,bulk,rtol=0,atol=1e-5)
                            redundancy.append({'name':st['name'],'real_histories':count,
                                'max_latent_reconstruction_error':float((recovered-bulk).abs().max())})
                if not quick and cell['rotation']==0 and cell['seed']==14001:
                    xi=ss['items'][:8];xm=ss['metadata'][:8];xo=offered_items(ss)[:8];xc=inherited['context_S'][:8];y=targets(ss)[:8]
                    net.train();net.zero_grad(set_to_none=True);net(xi,xm,xo,xc).sum().backward();net.zero_grad(set_to_none=True)
                    with profile(activities=[ProfilerActivity.CPU],with_flops=True) as prof:
                        torch.nn.functional.binary_cross_entropy_with_logits(net(xi,xm,xo,xc),y).backward()
                    net.eval();inp=batch(tt,slice(0,128));offers=offered[:128];samples=[]
                    with torch.no_grad():
                        for _ in range(3):net(inp[0],inp[1],offers,context(*inp))
                        for _ in range(9):
                            start=time.perf_counter();net(inp[0],inp[1],offers,context(*inp));samples.append(time.perf_counter()-start)
                    resources.append({'arm':st['arm'],'S_step_forward_backward_FLOP_subtotal_B8_four_targets':sum(e.flops or 0 for e in prof.key_averages()),
                        'full_FLOPs':False,'optimizer_and_context_precomputation_counted':False,
                        'pipeline_B128_latency_median_seconds':float(np.median(samples)),
                        'pipeline_B128_latency_samples_seconds':samples,
                        'limitation':'Operator-coverage subtotal excludes some elementwise/fused/nonlinear work; no full compute parity.'})
                if not quick:
                    inp=tmp/'inputs.pt';torch.save(batch(tt,slice(0,4)),inp);output=tmp/'actual.npy'
                    code="""import sys,torch,numpy as np
assert not any(n.startswith(('tac_osm','scripts')) for n in sys.modules)
torch.set_num_threads(1)
x=torch.load(sys.argv[1],weights_only=True)
c=torch.jit.load(sys.argv[2]).eval();m=torch.jit.load(sys.argv[3]).eval()
o=torch.cat((x[0][:,[32,64,96]],x[2].unsqueeze(-1)),-1)
with torch.no_grad():np.save(sys.argv[4],m(x[0],x[1],o,c(*x))[:,-1].sigmoid().numpy())
assert not any(n.startswith(('tac_osm','scripts')) for n in sys.modules)
"""
                    subprocess.run([sys.executable,'-c',code,str(inp),str(out/context_name),str(out/(st['name']+'.jit.pt')),str(output)],
                        check=True,cwd=tmp,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    pp=np.load(output);np.testing.assert_allclose(pp,expected[:4],rtol=0,atol=1e-6)
                    fresh.append({'name':st['name'],'forecasts':4,'max_abs_error':float(np.max(np.abs(pp-expected[:4])))})
        write_json(out/'inherited-context-exports.json',exports)
        assert evaluate(work,data,roles,r)==r['evaluation']
        exported=0;epaths=0
        for p in work.glob('*-predictions.npy'):
            np.testing.assert_array_equal(np.load(p),np.load(out/p.name));exported+=len(np.load(p));epaths+=1
        corrupt=tmp/'corrupt';corrupt.mkdir()
        for p in out.iterdir():
            if p.is_file():(corrupt/p.name).symlink_to(p)
        bad=corrupt/(r['cells'][0]['students'][0]['name']+'.pt');bad.unlink();bad.write_bytes(b'corrupt')
        proc=subprocess.run([sys.executable,'-m','scripts.me033_audit','--out',str(corrupt),'--verify-only'],capture_output=True,text=True)
        assert proc.returncode and 'artifact hash mismatch' in proc.stderr
    result={'status':'PASS','rebuilt_context_values':feature_count,'optimizer_RNG_receipts':receipts,
        'checkpoint_forecasts':checkpoint_forecasts,'bit_exact_checkpoint_paths':checkpoint_exact,
        'export_reference_forecasts':exported,'bit_exact_export_reference_paths':epaths,
        'fresh_Torch_only_pipelines':fresh,'resource_subtotals':resources,'streaming_checks':stream_checks,
        'frozen_latent_reconstruction_from_common':redundancy,'corruption_rejected_before_source':True,
        'historical_DEV_scored':False,'reserved_cohorts_opened':False}
    write_json(out/('verification.json' if quick else 'replay.json'),result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/me033');p.add_argument('--quick',action='store_true');p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    if a.verify_only:verify(Path(a.out))
    else:audit(Path(a.out),a.quick)
