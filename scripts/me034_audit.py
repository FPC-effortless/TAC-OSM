"""Independent ME034 checkpoint, raw pipeline, statistics and provenance audit."""
import argparse
import json
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from tac_osm.me034_runtime import StablePipeline
from tac_osm.me032_conditional_memory import FrozenContext
from scripts.me034_stable_research import (
    ROOT,SEEDS,ARMS,COUNTS,construct,verify_frozen,subset,roles_manifest,load_source,
    auxiliary,context_features,restore_context,offered_items,predict,targets,
    file_sha,write_json,batch,scores,tree_predict,fit_only_popularity,EWMABank
)


def verify(out):
    r=verify_frozen(out)
    index=out/'original-final-payloads.json'
    if r['complete'] and not index.exists():
        write_json(index,{p.name:file_sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('frozen-audit.json',)})
    if index.exists():
        for name,digest in json.loads(index.read_text()).items():
            if file_sha(out/name)!=digest:raise ValueError('final artifact hash mismatch: '+name)
    exports=out/'pipeline-exports.json'
    if exports.exists():
        for name,digest in json.loads(exports.read_text()).items():
            if file_sha(out/name)!=digest:raise ValueError('pipeline artifact hash mismatch: '+name)
    return r


def check_receipt(out,st,ck,rows):
    assert ck['step']==300 and ck['seed']==st['seed'] and ck['arm']==st['arm']
    assert {float(v['step']) for v in ck['optimizer']['state'].values()}=={300.}
    gen=torch.Generator().manual_seed(st['seed']+300)
    for _ in range(300):torch.randint(rows,(128,),generator=gen)
    assert torch.equal(gen.get_state(),ck['sampler_rng'])
    trace=json.loads((out/(st['name']+'-trace.json')).read_text())
    assert [v['step'] for v in trace]==list(range(1,301))
    assert all(np.isfinite(v['real_loss']) and v['cpu_seconds']>=0 for v in trace)
    assert sum(v['embedding_gradient_after_clip']>0 for v in trace)==st['positive_embedding_gradient_steps']
    assert sum(v['memory_gradient_after_clip']>0 for v in trace)==st['positive_memory_gradient_steps']
    assert st['charged_cpu_seconds']<=180
    return trace


def frozen_audit(out):
    r=verify(out);roles=json.loads((out/'roles.json').read_text());checked=0
    for cell in r['cells']:
        inc=torch.load(out/(cell['tag']+'-inherited.pt'),weights_only=True)
        for st in cell['students']:
            ck=torch.load(out/(st['name']+'.pt'),weights_only=True)
            check_receipt(out,st,ck,roles[cell['rotation']]['readout']['windows'])
            net=construct(st['arm'],inc,st['seed']);net.load_state_dict(ck['state_dict'],strict=True)
            assert sum(v.numel() for v in net.parameters() if v.requires_grad)==COUNTS[st['arm']]
            if st['arm'] in ('native_frozen','history_frozen','gru_frozen'):
                table=net.memory.core.emb if st['arm']=='native_frozen' else net.memory.emb
                assert not table.weight.requires_grad and torch.equal(table.weight,inc['embedding'])
                assert not st['positive_embedding_gradient_steps']
            if st['arm']=='native_adapt':
                assert net.memory.core.emb.weight.requires_grad and not torch.equal(net.memory.core.emb.weight,inc['embedding'])
                assert st['positive_embedding_gradient_steps']>0
            if st['arm'] in ('native_frozen','native_adapt','history_frozen','gru_frozen'):
                assert all(torch.equal(v,inc['common_state'][k]) for k,v in net.common.state_dict().items())
            assert cell['shared_charged_cpu_seconds']+st['charged_cpu_seconds']<=300
            checked+=1
    assert checked==63 and len(r['cells'])==9
    result={'status':'PASS','optimizer_RNG_parameter_freeze_receipts':checked,
        'tree_models':len(r['cells']),'final_labels_opened':r['final_labels_opened'],
        'protocol_sha256':r['protocol_sha256'],'frozen_payload_sha256':file_sha(out/'frozen-payloads.json')}
    write_json(out/'frozen-audit.json',result);print(json.dumps(result),flush=True)
    return result


def eager_context(state):
    c=FrozenContext(state['emb.weight'],float(state['duration_mean']),float(state['duration_scale']),
        state['item_popularity'],state['feature_mean'],state['feature_scale'])
    c.load_state_dict(state,strict=True);return c.eval()


def statistics_replay(out,r,data):
    from sklearn.metrics import roc_auc_score
    y=data['y'].numpy().ravel();preds={s:{} for s in SEEDS}
    for s in SEEDS:
        for a in (*ARMS,'boosted'):
            per=[np.load(out/f'final-{a}-r{rot}-s{s}-predictions.npy') for rot in range(3)]
            p=sum((v/3. for v in per),np.zeros(len(y)))
            np.testing.assert_array_equal(p,np.load(out/f'final-ensemble-{a}-s{s}.npy'))
            assert scores(p,y)==r['final_evaluation']['pooled_per_seed'][str(s)][a]
            preds[s][a]=p
    users,inv=np.unique(data['users'],return_inverse=True);counts=np.bincount(inv)
    delta=np.mean([(preds[s]['native_frozen']-y)**2-(preds[s]['common_adapt']-y)**2 for s in SEEDS],axis=0)
    sums=np.bincount(inv,weights=delta);rng=np.random.default_rng(34034);b=[];a=[]
    for _ in range(3000):
        draw=rng.integers(len(users),size=len(users));w=np.bincount(draw,minlength=len(users))[inv]
        b.append(sums[draw].sum()/counts[draw].sum())
        a.append(np.mean([roc_auc_score(y,preds[s]['native_frozen'],sample_weight=w)-roc_auc_score(y,preds[s]['common_adapt'],sample_weight=w) for s in SEEDS]))
    primary=r['final_evaluation']['primary']
    np.testing.assert_allclose(np.quantile(b,[.0125,.9875]),primary['Brier_Bonferroni_CI975'],rtol=0,atol=1e-14)
    np.testing.assert_allclose(np.quantile(a,[.0125,.9875]),primary['AUC_Bonferroni_CI975'],rtol=0,atol=1e-14)
    return {'all_seed_metrics_exact':True,'primary_bootstrap_intervals_reproduced':True,'draws':3000}


def audit(out,quick=False):
    torch.set_num_threads(1);r=verify(out);assert r['complete'];start=time.process_time()
    frozen=frozen_audit(out)
    data=load_source(ROOT/'results/me022')['train'];roles=roles_manifest(data)
    assert roles==json.loads((out/'roles.json').read_text())
    final=torch.load(out/'final-data.pt',weights_only=True)['data']
    assert all(u%16==6 for u in final['users']) and not set(data['users'])&set(final['users'])
    assert all(1597363200<=t<1599350400 for t in final['timestamps'])
    counts={'checkpoint_forecasts':0,'bit_exact_checkpoint_forecasts':0,
        'scripted_forecasts':0,'bit_exact_scripted_forecasts':0,'pipeline_forecasts':0,
        'bit_exact_pipeline_forecasts':0,'tree_forecasts':0,'feature_values_rebuilt':0}
    resources=[];sharing=[];fresh=[];exports={};diagnostics=[]
    with tempfile.TemporaryDirectory() as folder:
        tmp=Path(folder)
        for cell in r['cells']:
            tag=cell['tag'];role=roles[cell['rotation']]
            inc=torch.load(out/(tag+'-inherited.pt'),weights_only=True)
            context=restore_context(inc['context_state'])
            rr=subset(data,torch.tensor(role['representation']['indices']))
            cs=inc['context_state']
            old=torch.load(ROOT/'results/me031'/f'encoder-r{cell["rotation"]}-s{cell["seed"]}.pt',weights_only=True)
            assert torch.equal(old['state_dict']['emb.weight'],inc['embedding'])
            assert torch.equal(fit_only_popularity(rr),cs['item_popularity'])
            assert float(rr['duration'].mean())==float(cs['duration_mean'])
            assert max(float(rr['duration'].std(unbiased=False)),1e-5)==float(cs['duration_scale'])
            raw=eager_context(cs)
            raw.feature_mean.zero_();raw.feature_scale.fill_(1.)
            rf=context_features(torch.jit.script(raw),rr)[...,:26].reshape(-1,26)
            torch.testing.assert_close(rf.mean(0),cs['feature_mean'],rtol=0,atol=1e-7)
            torch.testing.assert_close(rf.std(0,unbiased=False).clamp_min(.05),cs['feature_scale'],rtol=0,atol=1e-7)
            ec=auxiliary(rr,context_features(context,rr))
            bank=EWMABank(cs['item_popularity'],torch.zeros(12),torch.ones(12))
            with torch.no_grad():ef=bank.features(rr['items'],rr['metadata'],offered_items(rr),ec).reshape(-1,12)
            torch.testing.assert_close(ef.mean(0),inc['ewma_mean'],rtol=0,atol=1e-7)
            torch.testing.assert_close(ef.std(0,unbiased=False).clamp_min(.05),inc['ewma_scale'],rtol=0,atol=1e-7)
            assert float(torch.logit(rr['y'].mean().clamp(.0001,.9999)))==inc['global_logit']
            t=subset(data,torch.tensor(role['evaluation']['indices']))
            for label,part in (('S',subset(data,torch.tensor(role['readout']['indices']))),('T',t)):
                rebuilt=context_features(context,part)
                torch.testing.assert_close(rebuilt,inc['context_'+label],rtol=0,atol=1e-7)
                counts['feature_values_rebuilt']+=rebuilt.numel()
            contexts={'cv':auxiliary(t,inc['context_T']),
                'final':auxiliary(final,context_features(context,final))}
            for st in cell['students']:
                ck=torch.load(out/(st['name']+'.pt'),weights_only=True)
                net=construct(st['arm'],inc,st['seed']);net.load_state_dict(ck['state_dict'],strict=True);net.eval()
                scripted=torch.jit.script(net).eval()
                export=torch.jit.load(str(out/(st['name']+'.jit.pt'))).eval()
                share=st['arm'] in ('native_frozen','history_frozen','gru_frozen')
                pipeline=StablePipeline(eager_context(inc['context_state']),net,share)
                pipe=torch.jit.script(pipeline).eval()
                pname='pipeline-'+st['name']+'.jit.pt'
                if not (out/pname).exists():pipe.save(str(out/pname))
                runtime=torch.jit.load(str(out/pname)).eval();exports[pname]=file_sha(out/pname)
                if share:
                    plain=StablePipeline(eager_context(inc['context_state']),net)
                    before=sum(p.numel()*p.element_size() for p in plain.parameters())
                    after=sum(p.numel()*p.element_size() for p in runtime.parameters())
                    assert before-after==512000
                    table=runtime.student.memory.core.emb if st['arm']=='native_frozen' else runtime.student.memory.emb
                    assert table.weight.data_ptr()==runtime.context.emb.weight.data_ptr()
                    sharing.append({'name':st['name'],'original_tensor_weight_bytes':before,
                        'shared_tensor_weight_bytes':after,'saved_bytes':before-after,'buffers_excluded':True})
                for label,part in (('cv',t),('final',final)):
                    expected=np.load(out/(label+'-'+st['arm']+'-'+tag+'-predictions.npy'))
                    p=predict(scripted,part,contexts[label]);np.testing.assert_allclose(p,expected,rtol=0,atol=1e-7)
                    counts['checkpoint_forecasts']+=len(p);counts['bit_exact_checkpoint_forecasts']+=int(np.array_equal(p,expected))*len(p)
                    q=predict(export,part,contexts[label]);np.testing.assert_allclose(q,expected,rtol=0,atol=1e-7)
                    counts['scripted_forecasts']+=len(q);counts['bit_exact_scripted_forecasts']+=int(np.array_equal(q,expected))*len(q)
                    if not quick:
                        with torch.no_grad():
                            raw=torch.cat([runtime(*batch(part,slice(i,i+256)))[:,-1].sigmoid() for i in range(0,len(part['y']),256)]).numpy().astype(np.float64)
                        np.testing.assert_allclose(raw,expected,rtol=0,atol=1e-7)
                        counts['pipeline_forecasts']+=len(raw);counts['bit_exact_pipeline_forecasts']+=int(np.array_equal(raw,expected))*len(raw)
                if not quick:
                    inputs=tmp/'inputs.pt';output=tmp/'actual.npy';torch.save(batch(final,slice(0,4)),inputs)
                    code="""import sys,torch,numpy as np
assert not any(x.startswith(('tac_osm','scripts')) for x in sys.modules)
torch.set_num_threads(1)
x=torch.load(sys.argv[1],weights_only=True);m=torch.jit.load(sys.argv[2]).eval()
with torch.no_grad():p=m(*x)[:,-1].sigmoid().numpy()
np.save(sys.argv[3],p)
assert not any(x.startswith(('tac_osm','scripts')) for x in sys.modules)
"""
                    subprocess.run([sys.executable,'-c',code,str(inputs),str((out/pname).resolve()),str(output)],cwd=tmp,check=True,capture_output=True)
                    expected=np.load(out/('final-'+st['arm']+'-'+tag+'-predictions.npy'))[:4]
                    actual=np.load(output).astype(np.float64);np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-6)
                    fresh.append({'name':st['name'],'forecasts':4,'max_abs_error':float(np.max(np.abs(actual-expected)))})
                if not quick and cell['rotation']==0 and cell['seed']==14001:
                    x=batch(final,slice(0,128));samples=[]
                    with torch.no_grad():
                        for _ in range(3):runtime(*x)
                        for _ in range(9):
                            clock=time.perf_counter();runtime(*x);samples.append(time.perf_counter()-clock)
                    resources.append({'arm':st['arm'],'B128_full_pipeline_latency_median_seconds':float(np.median(samples)),
                        'latency_samples_seconds':samples,'unique_parameter_bytes':sum(p.numel()*p.element_size() for p in runtime.parameters()),
                        'buffer_bytes':sum(p.numel()*p.element_size() for p in runtime.buffers()),
                        'context_cache_excluded':True,'recurrence_state_bytes':256 if st['arm'].startswith('native') else 144 if st['arm']=='gru_frozen' else None})
                if not quick and st['arm'] in ('native_frozen','history_frozen'):
                    # Isolate recurrence order from already-available context. A
                    # pooled identity here says nothing about signal in the data.
                    d=subset(final,torch.arange(min(256,len(final['y']))));ctx=contexts['final'][:len(d['y'])]
                    g=torch.Generator().manual_seed(34034)
                    perm=torch.cat([torch.randperm(32,generator=g)+32*j for j in range(4)])
                    shuffled={**d,'items':d['items'][:,perm],'metadata':d['metadata'][:,perm]}
                    a=predict(export,d,ctx);b=predict(export,shuffled,ctx)
                    with torch.no_grad():
                        x=ctx[...,:26]
                        zero=(ctx[...,26]+export.common(x))[:,-1].sigmoid().numpy().astype(np.float64)
                    diagnostics.append({'name':st['name'],'rows':len(a),'sequence_only_shuffle_Brier_delta':scores(b,d['y'].numpy())['Brier']-scores(a,d['y'].numpy())['Brier'],
                        'zero_state_Brier_delta':scores(zero,d['y'].numpy())['Brier']-scores(a,d['y'].numpy())['Brier'],
                        'max_shuffle_probability_change':float(np.max(abs(b-a))),'OOD_diagnostic_not_information_test':True})
            for label,part in (('cv',t),('final',final)):
                actual=tree_predict(out/(cell['tree']['name']+'.pickle'),contexts[label])
                expected=np.load(out/(label+'-boosted-'+tag+'-predictions.npy'));np.testing.assert_array_equal(actual,expected)
                counts['tree_forecasts']+=len(actual)
        write_json(out/'pipeline-exports.json',exports)
        stats=statistics_replay(out,r,final)
        corrupt=tmp/'corrupt';corrupt.mkdir()
        for p in out.iterdir():
            if p.is_file():(corrupt/p.name).symlink_to(p.resolve())
        victim=corrupt/(r['cells'][0]['students'][0]['name']+'.pt');victim.unlink();victim.write_bytes(b'corrupt')
        proc=subprocess.run([sys.executable,'-m','scripts.me034_audit','--out',str(corrupt),'--verify-only'],cwd=ROOT,capture_output=True,text=True)
        assert proc.returncode and 'artifact hash mismatch' in proc.stderr
    result={'status':'PASS',**counts,'optimizer_RNG_freeze_receipts':frozen,'fresh_torch_only_pipelines':fresh,
        'shared_frozen_tables':sharing,'statistics_replay':stats,'full_pipeline_resources':resources,
        'history_diagnostics':diagnostics,'corruption_rejected_before_source':True,
        'audit_CPU_seconds':time.process_time()-start,'full_compute_parity':False}
    write_json(out/('verification.json' if quick else 'replay.json'),result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/me034');p.add_argument('--frozen-only',action='store_true');p.add_argument('--verify-only',action='store_true');p.add_argument('--quick',action='store_true');a=p.parse_args()
    if a.verify_only:verify(Path(a.out))
    elif a.frozen_only:frozen_audit(Path(a.out))
    else:audit(Path(a.out),a.quick)
