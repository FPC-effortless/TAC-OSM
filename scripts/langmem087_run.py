#!/usr/bin/env python3
"""Stage A: original086 continuation, immutable checkpoints, fresh-source attribution.
This runner never tunes on DEV or changes the trained model's equations.
"""
from __future__ import annotations
import argparse,hashlib,json,random,resource,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'research/mathematical-engineering/langmem086'),str(ROOT/'research/mathematical-engineering/langmem087')]
from scripts import langmem086_run as old
from diagnostic_tac087 import DiagnosticTAC087
from probe_adapter import adapter086
from tac_osm.research_probe import probe_memory,probe_dependencies
from scripts.validate_research_parity import check_manifest
REG=ROOT/'research/mathematical-engineering/langmem087'
SEEDS=old.SEEDS;BATCH=4;LENGTH=256;CHECKPOINTS=(160,640,1280)
EVAL_STOP=2048;EVAL_BEGIN=256
LESIONS=('fast_zero','slow_zero','both_zero','permuted_read','old_slow_zero','old_slow_permuted','current_slow_write_zero')
digest=old.digest;weight_hash=old.weight_hash

def batches(docs,seed,steps):
    rng=np.random.default_rng(seed);names=sorted(docs)
    for step in range(steps):
        row=[]
        for _ in range(BATCH):
            owner=names[int(rng.integers(len(names)))];raw=docs[owner];start=int(rng.integers(len(raw)-LENGTH))
            row.append((owner,start,raw[start:start+LENGTH+1]))
        yield step,row

def schedule(docs,seed,steps):
    a=hashlib.sha256();b=hashlib.sha256()
    for step,row in batches(docs,seed,steps):
        for owner,start,raw in row:a.update(f'{owner}:{start};'.encode());b.update(raw)
    return {'seed':seed,'steps':steps,'batch':BATCH,'context':LENGTH,'targets':steps*BATCH*LENGTH,
        'sample_order_sha256':a.hexdigest(),'sample_bytes_sha256':b.hexdigest()}

def plan():
    docs,sha=old.fit_sources086()
    result={'fit_source_sha256':sha,'original086_sampler_prefix_exact':True,'by_seed':[]}
    frozen=json.loads((old.REG/'FIT_PLAN.json').read_text())
    for seed in SEEDS:
        assert schedule(docs,seed,160)==next(r for r in frozen['by_seed'] if r['seed']==seed)
        result['by_seed'].append({'seed':seed,'checkpoints':[schedule(docs,seed,n) for n in CHECKPOINTS]})
    return result

def preflight(model):
    manifest=json.loads((REG/'FAIR_COMPARISON.json').read_text())
    if check_manifest(manifest,'preflight')['status']!='PASS_PREFIT_SINGLE_MODEL':raise ValueError('prefit gate')
    if model.counts()['trainable']!=100521 or model.counts()['deployed']!=100521:raise ValueError('actual100k count')
    if old.model_source_hash()!=manifest['implementation_sha256']:raise ValueError('086 source graph changed')
    return manifest

def continue_fit(seed,previous,target,out):
    if out.exists():raise FileExistsError('no checkpoint/evidence overwrite')
    if target not in (640,1280):raise ValueError('registered duration only')
    start=160 if target==640 else 640
    torch.set_num_threads(2)
    path=previous/'MODEL.pt';cp=torch.load(path,map_location='cpu',weights_only=True)
    fit=json.loads((previous/'FIT_RECEIPT.json').read_text())
    parent_sha=digest(path.read_bytes())
    if cp['seed']!=seed or cp['steps']!=start or fit['checkpoint_sha256']!=parent_sha:raise ValueError('original parent identity')
    if start==160:
        pin=next(r for r in json.loads((REG/'PARENTS.json').read_text())['by_seed'] if r['seed']==seed)
        if parent_sha!=pin['checkpoint_sha256'] or cp['study']!='LANGMEM086':raise ValueError('original160 archive pin')
        if fit['trained_steps']!=160 or not fit['cold_replay_bitwise']:raise ValueError('original160 replay')
    elif cp['study']!='LANGMEM087' or fit['trained_steps']!=640:raise ValueError('original640 required')
    model=old.CompactDualClippedTAC086();model.load_state_dict(cp['model'],strict=True);manifest=preflight(model)
    if cp['model_source_sha256']!=old.model_source_hash() or weight_hash(model)!=fit['trained_model_hash']:raise ValueError('original source/model hash')
    docs,sha=old.fit_sources086();sched=schedule(docs,seed,target)
    frozen=json.loads((REG/'FIT_PLAN.json').read_text())
    required=next(r for r in frozen['by_seed'] if r['seed']==seed)['checkpoints']
    if sha!=frozen['fit_source_sha256'] or cp['fit_source_sha256']!=sha or sched!=next(r for r in required if r['steps']==target) or cp['sample_receipt']!=schedule(docs,seed,start):raise ValueError('FIT/schedule prefix mismatch')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);optimizer.load_state_dict(cp['optimizer'])
    if any(int(s['step'])!=start for s in optimizer.state.values()):raise ValueError('all original optimizer step counters')
    # Original086 global NumPy RNG was unused by the local default_rng sampler.
    np.random.seed(seed)
    if start==160:
        if np.random.get_state()[1].tolist()!=cp['numpy_rng']:raise ValueError('original unused global NumPy seed state')
    else:
        st=cp['numpy_rng_full'];np.random.set_state((st[0],np.array(st[1],dtype=np.uint32),st[2],st[3],st[4]))
    torch.set_rng_state(cp['torch_rng']);random.setstate(cp['python_rng'])
    fixture=torch.tensor([list(next(iter(docs.values()))[:16])])
    poison=fixture.clone();poison[:,8:]=(poison[:,8:]+113)%256
    with torch.no_grad():
        a,_=model(fixture,owners=('FIT-poison',));b,_=model(poison,owners=('FIT-poison',))
    if not torch.equal(a[:,:8],b[:,:8]):raise ValueError('actual FIT future suffix poison')
    initial_hash=weight_hash(model);counts=torch.tensor(cp['fit_byte_counts'],dtype=torch.int64)
    losses=list(cp['fit_loss_trace']);gradients=[];actual_order=hashlib.sha256();actual_bytes=hashlib.sha256()
    started=time.process_time();out.mkdir(parents=True)
    for step,row in batches(docs,seed,target):
        for owner,offset,raw in row:actual_order.update(f'{owner}:{offset};'.encode());actual_bytes.update(raw)
        if step<start:continue
        x=torch.tensor([list(raw[:-1]) for _,_,raw in row]);y=torch.tensor([list(raw[1:]) for _,_,raw in row])
        counts+=torch.bincount(y.flatten(),minlength=256)
        optimizer.zero_grad(set_to_none=True)
        logits,_=model(x,owners=tuple(f'fit-row-{i}' for i in range(BATCH)))
        loss=F.cross_entropy(logits.flatten(0,1),y.flatten());loss.backward()
        if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('actual loss/connected finite gradient failure')
        if step in (start,target-1):
            gradients.append({'step':step,'all_parameter_tensors_connected_finite':True,'by_module_l2':{n:float(torch.linalg.vector_norm(torch.cat([p.grad.flatten() for k,p in model.named_parameters() if k.split('.')[0]==n]))) for n in sorted({n.split('.')[0] for n,p in model.named_parameters()})}})
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        if step%80==0:print(json.dumps({'phase':'FIT','seed':seed,'step':step,'nll':losses[-1]}),flush=True)
    cpu=time.process_time()-started
    if actual_order.hexdigest()!=sched['sample_order_sha256'] or actual_bytes.hexdigest()!=sched['sample_bytes_sha256'] or len(losses)!=target or int(counts.sum())!=target*1024:raise ValueError('actual executed schedule')
    if any(int(s['step'])!=target for s in optimizer.state.values()):raise ValueError('final optimizer steps')
    model.eval();final_hash=weight_hash(model)
    with torch.no_grad():reference,reference_state=model(fixture,owners=('cold-replay',))
    ns=np.random.get_state()
    checkpoint={'study':'LANGMEM087','seed':seed,'steps':target,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
        'torch_rng':torch.get_rng_state(),'numpy_rng_full':[ns[0],ns[1].tolist(),int(ns[2]),int(ns[3]),float(ns[4])],'python_rng':random.getstate(),
        'model_counts':model.counts(),'model_source_sha256':old.model_source_hash(),'fit_source_sha256':sha,'sample_receipt':sched,
        'fit_byte_counts':counts.tolist(),'fit_loss_trace':losses,'parent_checkpoint_sha256':parent_sha,
        'runner_sha256':digest(Path(__file__).read_bytes())}
    dest=out/'MODEL.pt';torch.save(checkpoint,dest);sealed=digest(dest.read_bytes())
    saved=torch.load(dest,map_location='cpu',weights_only=True);cold=old.CompactDualClippedTAC086().eval();cold.load_state_dict(saved['model'],strict=True)
    with torch.no_grad():restored,restored_state=cold(fixture,owners=('cold-replay',))
    if not torch.equal(reference,restored) or any(not torch.equal(reference_state.tensors[k],restored_state.tensors[k]) for k in reference_state.tensors):raise ValueError('cold bitwise logits/state replay')
    cold_optimizer=torch.optim.AdamW(cold.parameters(),lr=.001,weight_decay=.01);cold_optimizer.load_state_dict(saved['optimizer'])
    if any(int(s['step'])!=target for s in cold_optimizer.state.values()):raise ValueError('cold optimizer replay')
    live=cold.counts();manifest['models']=[{**r,'measured':{'trainable_parameters':live['trainable'],'deployed_parameters':live['deployed'],'checkpoint_sha256':sealed}} for r in manifest['models'] if r['seed']==seed]
    final_gate=check_manifest(manifest,'final')
    sample={'tokens':fixture[:,:9],'owners':('source-FIT-diagnostic',)}
    with torch.no_grad():memory=probe_memory(cold,sample,adapter086())
    dependencies=probe_dependencies(cold,sample,adapter086())
    receipt={'study':'LANGMEM087','seed':seed,'trained_steps':target,'additional_steps':target-start,'targets_seen':target*1024,
        'sample_receipt':sched,'fit_source_sha256':sha,'checkpoint_sha256':sealed,'parent_checkpoint_sha256':parent_sha,
        'initial_model_hash':initial_hash,'trained_model_hash':final_hash,'model_counts':live,'model_source_sha256':old.model_source_hash(),
        'cold_replay_bitwise':True,'cold_state_replay_bitwise':True,'cold_optimizer_step_count':target,'unseen_DEV_accessed_before_seal':False,
        'segment_fit_cpu_seconds':cpu,'cumulative_fit_cpu_seconds':fit.get('cumulative_fit_cpu_seconds',fit.get('fit_cpu_seconds',0))+cpu,
        'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'actual_FIT_future_poison_before_continuation':'PASS',
        'gradient_receipts':gradients,'fit_loss_segment_first':losses[start],'fit_loss_segment_last':losses[-1],
        'final_parameter_gate':final_gate,'memory_probe':memory,'dependency_probe':dependencies,'gpu_minutes':0,'protected_TEST':'NOT_ACCESSED'}
    (out/'FIT_RECEIPT.json').write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'phase':'SEALED','seed':seed,'updates':target,'checkpoint_sha256':sealed}),flush=True)

@torch.no_grad()
def evaluate(seed,trained,documents,out):
    if out.exists():raise FileExistsError('do not overwrite first DEV')
    torch.set_num_threads(2);fit=json.loads((trained/'FIT_RECEIPT.json').read_text());path=trained/'MODEL.pt'
    cp=torch.load(path,map_location='cpu',weights_only=True);steps=cp['steps']
    if steps not in CHECKPOINTS or cp['seed']!=seed or fit['seed']!=seed or fit['trained_steps']!=steps or fit['unseen_DEV_accessed_before_seal'] is not False or fit['checkpoint_sha256']!=digest(path.read_bytes()):raise ValueError('sealed checkpoint')
    m=DiagnosticTAC087().eval();m.load_state_dict(cp['model'],strict=True);preflight(m);before=weight_hash(m)
    native=old.CompactDualClippedTAC086().eval();native.load_state_dict(cp['model'],strict=True)
    fixture=torch.arange(12).view(1,-1)
    actual,actual_state=m(fixture,owners=('normal-equation-parity',));expected,expected_state=native(fixture,owners=('normal-equation-parity',))
    if not torch.equal(actual,expected) or any(not torch.equal(actual_state.tensors[k],expected_state.tensors[k]) for k in expected_state.tensors):raise ValueError('diagnostic default differs from original trained graph')
    if before!=fit['trained_model_hash'] or cp['model_source_sha256']!=old.model_source_hash():raise ValueError('frozen model/source identity')
    reg=json.loads((REG/'NEW_DEV_SOURCE.json').read_text());docs={}
    for entry in reg['documents']:
        raw=(documents/Path(entry['path']).name).read_bytes();blob=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
        if len(raw)!=entry['bytes'] or blob!=entry['git_blob']:raise ValueError('new publisher blob pin')
        docs[entry['path']]=raw
    prior=(torch.tensor(cp['fit_byte_counts'],dtype=torch.float64)+.5)/(steps*1024+128)
    rows=[];autonomous=[];target_sha=hashlib.sha256();prefix_error=0.;poison_pairs=0;model_wall=0.;ring_wall=0.;split_error=0.;started=time.process_time()
    for owner,raw in sorted(docs.items()):
        state=m.start((owner,));ring=old.PackedRing086(1928);keys={}
        for t in range(EVAL_STOP):
            token=torch.tensor([[raw[t]]]);prior_state=state;tic=time.perf_counter()
            logits,state,trace=m(token,state=state,owners=(owner,),return_trace=True);model_wall+=time.perf_counter()-tic
            if state.tensor_bytes()!=1920 or state.clock!=t+1:raise ValueError('actual state budget')
            ring.append(int(raw[t]));f=trace[0]
            split_error=max(split_error,float((f['old_slow_component']+f['current_slow_component']-state.tensors['slow_memory']).abs().max()))
            if t>=4:keys[t]=f['writer'][0].clone()
            if t==127:
                full,bulk=m(torch.tensor([list(raw[:128])]),owners=(owner,));err=float((full[:,-1]-logits[:,0]).abs().max());prefix_error=max(prefix_error,err)
                if err>3e-5:raise ValueError('source stream/bulk128 parity')
            if t<EVAL_BEGIN:continue
            alternatives={}
            for lesion in LESIONS:
                other,other_state=m(token,state=prior_state,owners=(owner,),lesion=lesion)
                # Pure read attribution must preserve every written state except downstream upper.
                if any(not torch.equal(other_state.tensors[k],state.tensors[k]) for k in state.tensors if k!='upper'):raise ValueError('attribution changed original stored non-upper state')
                alternatives[lesion]=other
            tic=time.perf_counter();rp,matches=ring.predict(prior);ring_wall+=time.perf_counter()-tic
            if t==EVAL_BEGIN:
                mutated=bytearray(raw);mutated[t+1:]=bytes((v+113)%256 for v in mutated[t+1:])
                pl,ps=m(torch.tensor([[mutated[t]]]),state=prior_state,owners=(owner,))
                if not torch.equal(pl,logits) or any(not torch.equal(ps.tensors[k],state.tensors[k]) for k in state.tensors):raise ValueError('actualsource suffix poison')
                poison_pairs+=1
            if t%16==0:
                anchors=list(range(max(4,t-256),t));scores=torch.stack([keys[a] for a in anchors])@f['query'][0];selected=anchors[int(scores.argmax())]
                positives=[a for a in anchors if raw[a-4:a]==raw[t-3:t+1]]
                autonomous.append({'owner':owner,'clock':t,'past_context_exists':bool(positives),'top1_context_hit':selected in positives,'slow_mix':float(f['mix'][0,1]),'slow_eta':float(f['slow_eta'][0,0]),'slow_rho':float(f['slow_rho'])})
            # First reveal target only after every model/control prediction.
            y=int(raw[t+1]);lp=logits[0,0].log_softmax(-1)
            row={'owner':owner,'clock':t,'actual_next_byte':y,'full_nll':-float(lp[y]),'full_correct':int(int(lp.argmax())==y),
                 'FIT_prior_nll':-float(prior[y].log()),'FIT_prior_correct':int(int(prior.argmax())==y),'ring1928_nll':-float(rp[y].log()),'ring_matching_count':matches,'slow_mix':float(f['mix'][0,1]),'slow_eta':float(f['slow_eta'][0,0])}
            for name,log in alternatives.items():row[name+'_nll']=-float(log[0,0].log_softmax(-1)[y])
            rows.append(row);target_sha.update(owner.encode()+b':'+str(t).encode()+b':'+bytes([y])+b';')
    if len(rows)!=3584 or weight_hash(m)!=before or split_error>1e-6:raise ValueError('complete frozen evaluation/split identity')
    fields=[k for k in rows[0] if k not in ('owner','clock','actual_next_byte')];metrics={k:sum(r[k] for r in rows)/len(rows) for k in fields}
    for lesion in LESIONS:metrics[lesion+'_minus_full_nll']=metrics[lesion+'_nll']-metrics['full_nll']
    positives=[r for r in autonomous if r['past_context_exists']]
    result={'study':'LANGMEM087','seed':seed,'trained_steps':steps,'checkpoint_sha256':fit['checkpoint_sha256'],'trained_model_hash':before,
        'model_source_sha256':old.model_source_hash(),'diagnostic_source_sha256':digest((REG/'diagnostic_tac087.py').read_bytes()),'model_counts':m.counts(),
        'original_graph_default_bitwise_parity':True,'all_three_durations_fitted_before_workflow_DEV_acquisition':True,
        'fit_source_sha256':fit['fit_source_sha256'],'sample_receipt':fit['sample_receipt'],
        'new_document_hashes':{k:{'sha256':digest(v),'bytes':len(v)} for k,v in docs.items()},'evaluation_nextbyte_target_sha256':target_sha.hexdigest(),
        'nextbyte_targets_per_checkpoint':len(rows),'persistent_tensor_bytes_per_owner':1920,'clock_bytes':8,'shared_weights_bytes':402084,'fixed_buffer_bytes':8192,
        'raw_ring_bytes':1928,'ring_prior_shared_float64_bytes':2048,'oldslow_split_max_abs_error':split_error,'metrics':metrics,'rows':rows,
        'autonomous_query_metrics':{'queries':len(autonomous),'contexts_available':len(positives),'context_top1_hits':sum(r['top1_context_hit'] for r in positives)},'autonomous_query_rows':autonomous,
        'original_prefix_max_abs_error':prefix_error,'actual_source_future_poison_pairs':poison_pairs,'weights_unchanged_during_evaluation':True,
        'evaluator_old_keys_fed_to_model':False,'model_receives_only_observed_byte_and_own_state':True,'evaluation_cpu_seconds':time.process_time()-started,
        'model_full_inference_wall_seconds':model_wall,'model_observed_steps':4096,'ring_query_wall_seconds':ring_wall,'ring_queries':len(rows),
        'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'gpu_minutes':0,'protected_TEST':'NOT_ACCESSED','independently_qualified':False}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'seed':seed,'updates':steps,'metrics':metrics,'autonomous':result['autonomous_query_metrics']}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('plan','continue','evaluate'));p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--trained',type=Path);p.add_argument('--steps',type=int);p.add_argument('--documents',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.phase=='plan':
        if a.output.exists():raise FileExistsError('plan exists')
        a.output.write_text(json.dumps(plan(),indent=2,sort_keys=True)+'\n')
    elif a.phase=='continue':
        if a.documents is not None:raise ValueError('FIT-only interface excludes DEV paths')
        continue_fit(a.seed,a.trained,a.steps,a.output)
    else:evaluate(a.seed,a.trained,a.documents,a.output)
if __name__=='__main__':main()
