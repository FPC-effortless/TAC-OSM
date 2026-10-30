#!/usr/bin/env python3
"""Gated100521p single-family FIT -> sealed replay -> first new Rust DEV.
No old weights and no evaluator keys enter the model's inference interface.
"""
from __future__ import annotations
import argparse,hashlib,json,math,random,resource,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'research/mathematical-engineering/langmem086'),str(ROOT/'research/mathematical-engineering/langbase024')]
from compact_tac import CompactDualClippedTAC086
from probe_adapter086 import adapter086
from byte_next_reference import read_sha_source
from scripts.validate_research_parity import check_manifest
from tac_osm.research_probe import probe_memory,probe_dependencies

SEEDS=(1701,1702,1703);STEPS=160;BATCH=4;LENGTH=256
EVAL_STOP=1024;EVAL_BEGIN=128
REG=ROOT/'research/mathematical-engineering/langmem086'

def digest(data):return hashlib.sha256(data).hexdigest()
def weight_hash(m):
    h=hashlib.sha256()
    for k,v in sorted(m.state_dict().items()):h.update(k.encode()+b'\0'+v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def fit_sources086():
    receipt=json.loads((ROOT/'research/mathematical-engineering/language_source023/SOURCE_RECEIPT.json').read_text())
    if receipt['commit']!='60403a5409ff2c3f3b07dd2ca91a7a3e096839c7':raise ValueError('publisher FIT version')
    docs={e['path']:read_sha_source(ROOT,e['path'],e) for e in receipt['files'] if e['role']=='FIT'}
    supplement=json.loads((ROOT/'research/mathematical-engineering/language_source023/UTF8_SUPPLEMENT_RECEIPT.json').read_text())
    docs[supplement['path']]=read_sha_source(ROOT,supplement['path'],supplement)
    if len(docs)!=14:raise ValueError('exact original14 FIT owners required')
    h=hashlib.sha256()
    for owner,raw in sorted(docs.items()):h.update(owner.encode()+b'\0'+bytes.fromhex(digest(raw)))
    return docs,h.hexdigest()

def batches(docs,seed):
    rng=np.random.default_rng(seed);names=sorted(docs)
    for step in range(STEPS):
        row=[]
        for _ in range(BATCH):
            owner=names[int(rng.integers(len(names)))];raw=docs[owner]
            start=int(rng.integers(len(raw)-LENGTH))
            row.append((owner,start,raw[start:start+LENGTH+1]))
        yield step,row

def schedule(docs,seed):
    h=hashlib.sha256();b=hashlib.sha256()
    for step,row in batches(docs,seed):
        for owner,start,raw in row:h.update(f'{owner}:{start};'.encode());b.update(raw)
    return {'seed':seed,'steps':STEPS,'batch':BATCH,'context':LENGTH,'targets':STEPS*BATCH*LENGTH,
            'sample_order_sha256':h.hexdigest(),'sample_bytes_sha256':b.hexdigest()}

def plan():
    docs,sha=fit_sources086()
    return {'fit_source_sha256':sha,'fit_owners':sorted(docs),'by_seed':[schedule(docs,s) for s in SEEDS]}

def model_source_hash():return digest((REG/'compact_tac.py').read_bytes())

def preflight(seed,model):
    manifest=json.loads((REG/'FAIR_COMPARISON.json').read_text());result=check_manifest(manifest,'preflight')
    if result['status']!='PASS_PREFIT_SINGLE_MODEL':raise ValueError('actual prefit contract')
    if model.counts()['trainable']!=100521 or model.counts()['deployed']!=100521:raise ValueError('actual100k counts')
    if model_source_hash()!=manifest['implementation_sha256']:raise ValueError('model source differs from frozen recipe')
    row=next(x for x in manifest['models'] if x['seed']==seed)
    if row['trainable_parameters']!=100521 or row['deployed_parameters']!=100521:raise ValueError('preregistered count')
    return manifest

def train(seed,out):
    if out.exists():raise FileExistsError('no overwrite original trained evidence')
    torch.set_num_threads(2);torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    model=CompactDualClippedTAC086();manifest=preflight(seed,model)
    docs,sha=fit_sources086();sched=schedule(docs,seed)
    frozen=json.loads((REG/'FIT_PLAN.json').read_text())
    if frozen['fit_source_sha256']!=sha or sched!=next(x for x in frozen['by_seed'] if x['seed']==seed):raise ValueError('actual source/schedule differs')
    out.mkdir(parents=True);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    initial_hash=weight_hash(model);counts=torch.zeros(256,dtype=torch.int64)
    prefix=torch.tensor([list(next(iter(docs.values()))[:16])],dtype=torch.long)
    changed=prefix.clone();changed[:,8:]=(changed[:,8:]+113)%256
    with torch.no_grad():
        original_logits,_=model(prefix,owners=('real-FIT-poison',))
        poisoned_logits,_=model(changed,owners=('real-FIT-poison',))
    if not torch.equal(original_logits[:,:8],poisoned_logits[:,:8]):raise ValueError('actual FIT future poison before optimizer')
    started=time.process_time();losses=[];gradient_receipts=[];sample_hash=hashlib.sha256()
    for step,row in batches(docs,seed):
        tokens=torch.tensor([list(raw[:-1]) for _,_,raw in row],dtype=torch.long)
        targets=torch.tensor([list(raw[1:]) for _,_,raw in row],dtype=torch.long)
        for owner,start,raw in row:sample_hash.update(f'{owner}:{start};'.encode())
        counts+=torch.bincount(targets.flatten(),minlength=256)
        optimizer.zero_grad(set_to_none=True)
        logits,_=model(tokens,owners=tuple(f'fit-row-{i}' for i in range(BATCH)))
        loss=F.cross_entropy(logits.flatten(0,1),targets.flatten());loss.backward()
        if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('actual loss/gradient failure')
        if step in (0,STEPS-1):
            gradient_receipts.append({'step':step,'all_parameter_tensors_connected_finite':True,
                'by_module_l2':{name:float(torch.linalg.vector_norm(torch.cat([p.grad.flatten() for n,p in model.named_parameters() if n.split('.')[0]==name]))) for name in sorted({n.split('.')[0] for n,p in model.named_parameters()})}})
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step();losses.append(float(loss.detach()))
        if step%20==0:print(json.dumps({'phase':'FIT','seed':seed,'step':step,'nll':losses[-1]}),flush=True)
    if sample_hash.hexdigest()!=sched['sample_order_sha256']:raise ValueError('actual fitted sample order')
    final_hash=weight_hash(model)
    if final_hash==initial_hash:raise ValueError('model did not train')
    model.eval();fixture=torch.tensor([list(next(iter(docs.values()))[:16])])
    with torch.no_grad():reference,_=model(fixture,owners=('cold-replay',))
    checkpoint={'study':'LANGMEM086','seed':seed,'steps':STEPS,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
        'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state()[1].tolist(),'python_rng':random.getstate(),
        'model_counts':model.counts(),'model_source_sha256':model_source_hash(),'fit_source_sha256':sha,'sample_receipt':sched,
        'fit_byte_counts':counts.tolist(),'fit_loss_trace':losses}
    path=out/'MODEL.pt';torch.save(checkpoint,path);checkpoint_sha=digest(path.read_bytes())
    replay=torch.load(path,map_location='cpu',weights_only=True);cold=CompactDualClippedTAC086().eval();cold.load_state_dict(replay['model'],strict=True)
    with torch.no_grad():restored,_=cold(fixture,owners=('cold-replay',))
    if not torch.equal(reference,restored):raise ValueError('cold replay logits differ')
    cold_optimizer=torch.optim.AdamW(cold.parameters(),lr=.001,weight_decay=.01)
    cold_optimizer.load_state_dict(replay['optimizer'])
    if not cold_optimizer.state:raise ValueError('optimizer/RNG checkpoint missing')
    # Final gate checks actual live counts/checkpoint before any new Rust content exists.
    live=model.counts()
    manifest['models']=[{**r,'measured':{'trainable_parameters':live['trainable'],'deployed_parameters':live['deployed'],'checkpoint_sha256':checkpoint_sha}} for r in manifest['models'] if r['seed']==seed]
    final_gate=check_manifest(manifest,'final')
    sample={'tokens':fixture[:,:9],'owners':('source-FIT-diagnostic',)}
    with torch.no_grad():memory=probe_memory(cold,sample,adapter086())
    dependencies=probe_dependencies(cold,sample,adapter086())
    receipt={'study':'LANGMEM086','seed':seed,'trained_steps':STEPS,'targets_seen':sched['targets'],
        'sample_receipt':sched,'fit_source_sha256':sha,'checkpoint_sha256':checkpoint_sha,
        'initial_model_hash':initial_hash,'trained_model_hash':final_hash,'model_counts':model.counts(),
        'model_source_sha256':model_source_hash(),'cold_replay_bitwise':True,'unseen_DEV_accessed_before_seal':False,
        'fit_cpu_seconds':time.process_time()-started,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'actual_FIT_future_poison_before_first_update':'PASS','gradient_receipts':gradient_receipts,'fit_loss_first':losses[0],'fit_loss_last':losses[-1],
        'final_parameter_gate':final_gate,'memory_probe':memory,'dependency_probe':dependencies,
        'gpu_minutes':0,'protected_TEST':'NOT_ACCESSED'}
    (out/'FIT_RECEIPT.json').write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'phase':'SEALED','seed':seed,'checkpoint_sha256':checkpoint_sha}),flush=True)

def read_documents(directory):
    reg=json.loads((REG/'NEW_DEV_SOURCE.json').read_text());out={}
    if reg['publisher_commit']!='2771df165f092a4fbb9f46b5b2e7450f19010adb':raise ValueError('Rust immutable version')
    for item in reg['documents']:
        raw=(directory/Path(item['path']).name).read_bytes()
        blob=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
        if len(raw)!=item['bytes'] or blob!=item['git_blob']:raise ValueError('new publisher Git blob/size mismatch')
        out[item['path']]=raw
    return out

@torch.no_grad()
def evaluate(seed,trained,documents,out):
    if out.exists():raise FileExistsError('no overwrite original first evaluation')
    torch.set_num_threads(2)
    fit=json.loads((trained/'FIT_RECEIPT.json').read_text());path=trained/'MODEL.pt'
    if fit['seed']!=seed or fit['trained_steps']!=STEPS or fit['unseen_DEV_accessed_before_seal'] is not False or digest(path.read_bytes())!=fit['checkpoint_sha256']:raise ValueError('sealed trained model first')
    cp=torch.load(path,map_location='cpu',weights_only=True)
    m=CompactDualClippedTAC086().eval();m.load_state_dict(cp['model'],strict=True)
    before=weight_hash(m)
    if before!=fit['trained_model_hash']:raise ValueError('original trained model source hash')
    if cp['seed']!=seed or cp['steps']!=STEPS or cp['model_source_sha256']!=model_source_hash() or cp['fit_source_sha256']!=fit['fit_source_sha256']:raise ValueError('model/seed/source receipt mismatch')
    docs=read_documents(documents) # FIRST new DEV read after this seed's fit/seal.
    prior=(torch.tensor(cp['fit_byte_counts'],dtype=torch.float64)+.5)/(STEPS*BATCH*LENGTH+128)
    rows=[];target_sha=hashlib.sha256();prefix_error=0.;poison_pairs=0;autonomous=[];started=time.process_time()
    model_wall=0.;ring_wall=0.
    for owner,raw in sorted(docs.items()):
        if len(raw)<EVAL_STOP+1:raise ValueError('frozen new source too short')
        state=m.start((owner,));keys={};ring=PackedRing086(1928)
        for t in range(EVAL_STOP):
            token=torch.tensor([[raw[t]]],dtype=torch.long);prior_state=state
            start=time.perf_counter()
            logits,state,trace=m(token,state=state,owners=(owner,),return_trace=True)
            model_wall+=time.perf_counter()-start
            if state.tensor_bytes()!=1920 or state.clock!=t+1:raise ValueError('measured native state payload')
            ring.append(int(raw[t]));f=trace[0]
            if t>=4:keys[t]=f['writer'][0].clone()
            if t==127:
                full,bulk=m(torch.tensor([list(raw[:128])]),owners=(owner,))
                err=float((full[:,-1]-logits[:,0]).abs().max());prefix_error=max(prefix_error,err)
                if err>3e-5:raise ValueError('actual trained streaming/full parity')
            if t<EVAL_BEGIN:continue
            alternatives={}
            for lesion in ('fast_zero','slow_zero','both_zero','permuted_read'):
                alternatives[lesion]=m(token,state=prior_state,owners=(owner,),lesion=lesion)[0]
            start=time.perf_counter();ring_probs,match_count=ring.predict(prior);ring_wall+=time.perf_counter()-start
            if t==128:
                mutated=bytearray(raw);mutated[t+1:]=bytes((v+113)%256 for v in mutated[t+1:])
                ptoken=torch.tensor([[mutated[t]]],dtype=torch.long)
                plogits,pstate=m(ptoken,state=prior_state,owners=(owner,))
                if not torch.equal(plogits,logits) or any(not torch.equal(pstate.tensors[k],state.tensors[k]) for k in state.tensors):raise ValueError('actual trained source future poison')
                poison_pairs+=1
            ranked=None
            if t%8==0:
                anchors=list(range(max(4,t-256),t));stack=torch.stack([keys[a] for a in anchors]);q=f['query'][0]
                scores=stack@q;selected=anchors[int(scores.argmax())]
                context=raw[t-3:t+1];positives=[a for a in anchors if raw[a-4:a]==context]
                # No source historical key is supplied to the model; these are evaluator labels only.
                ranked={'owner':owner,'clock':t,'past_context_exists':bool(positives),
                    'model_current_query_context_hit':selected in positives,'selected_historical_clock':selected,
                    'matching_count':len(positives),'query_key_top_cosine':float(scores.max()),'slow_mix':float(f['mix'][0,1])}
            # Source actual future byte is revealed only after every prediction/control.
            y=int(raw[t+1]);lp=logits[0,0].log_softmax(-1)
            row={'owner':owner,'clock':t,'actual_next_byte':y,'full_nll':-float(lp[y]),'full_correct':int(int(lp.argmax())==y),
                'FIT_prior_nll':-float(prior[y].log()),'FIT_prior_correct':int(int(prior.argmax())==y),
                'ring1928_nll':-float(ring_probs[y].log()),'ring_matching_count':match_count}
            for lesion,other in alternatives.items():row[lesion+'_nll']=-float(other[0,0].log_softmax(-1)[y])
            rows.append(row);target_sha.update(owner.encode()+b':'+str(t).encode()+b':'+bytes([y])+b';')
            if ranked is not None:
                ranked['selected_historical_byte_equals_future']=int(raw[ranked['selected_historical_clock']]==y)
                autonomous.append(ranked)
    if len(rows)!=2*(EVAL_STOP-EVAL_BEGIN) or weight_hash(m)!=before:raise ValueError('complete first evaluation/frozen weights')
    fields=[k for k in rows[0] if k not in ('owner','clock','actual_next_byte')]
    metrics={k:sum(x[k] for x in rows)/len(rows) for k in fields}
    positives=[x for x in autonomous if x['past_context_exists']]
    return {'study':'LANGMEM086','seed':seed,'checkpoint_sha256':fit['checkpoint_sha256'],'trained_model_hash':before,
        'model_counts':m.counts(),'trained_steps':STEPS,'fit_source_sha256':fit['fit_source_sha256'],'sample_receipt':fit['sample_receipt'],
        'new_document_hashes':{k:{'sha256':digest(v),'bytes':len(v)} for k,v in docs.items()},
        'new_document_Git_blob_registry_frozen_before_fit':True,'raw_SHA256_recorded_after_seal':True,
        'evaluation_nextbyte_target_sha256':target_sha.hexdigest(),'nextbyte_targets_per_seed':len(rows),
        'model_receives_only_observed_byte_and_own_state':True,'evaluator_old_keys_fed_to_model':False,
        'persistent_tensor_bytes_per_owner':1920,'clock_bytes':8,'owner_utf8_bytes':{k:len(k.encode()) for k in docs},
        'shared_learned_weight_bytes':100521*4,'shared_fixed_buffer_bytes':8192,'last_logits_persisted':False,
        'same_payload_ring_total_bytes':1928,'same_payload_ring_capacity':1912,
        'original_prefix_max_abs_error':prefix_error,'actual_source_future_poison_pairs':poison_pairs,
        'weights_unchanged_during_evaluation':True,'metrics':metrics,'rows':rows,'autonomous_query_rows':autonomous,
        'autonomous_query_metrics':{'queries':len(autonomous),'exact_past_context_coverage':len(positives)/len(autonomous),
             'top1_context_hit_when_available':sum(x['model_current_query_context_hit'] for x in positives)/len(positives) if positives else None},
        'evaluation_cpu_seconds':time.process_time()-started,
        'model_full_inference_wall_seconds':model_wall,'model_full_observed_steps':2*EVAL_STOP,
        'ring_query_wall_seconds':ring_wall,'ring_queries':len(rows),
        'peak_evaluation_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'gpu_minutes':0,'protected_TEST':'NOT_ACCESSED',
        'independently_qualified':False,'limits':['New Rust documents from same publisher as085, not broad independent qualification',
            'Single trained100521p family; no equally trained alternative architecture comparison or clipping superiority claim',
            'Read lesions hold original prior state fixed; not full historical counterfactual replay',
            'External evaluator key table is for diagnostics only; never supplied to deployed model',
            'Memory query context matching is a byte-context retrieval proxy, not factual semantic correctness',
            'Tensor+clock+owner payload excludes allocator/metadata object overhead and transient workspace; peak RSS recorded separately',
            'Full original computational graph IEEE/AD proof and independent reward/goal certification OPEN']}

class PackedRing086:
    def __init__(self,total):
        if total!=1928:raise ValueError('same tensor+clock payload budget')
        self.data=bytearray(total)
    def append(self,byte):
        import struct
        clock,count=struct.unpack_from('<QQ',self.data);cap=len(self.data)-16
        self.data[16+clock%cap]=byte;struct.pack_into('<QQ',self.data,0,clock+1,min(count+1,cap))
    def get(self,a):
        import struct
        clock,count=struct.unpack_from('<QQ',self.data)
        return self.data[16+a%(len(self.data)-16)] if clock-count<=a<clock else None
    def predict(self,prior):
        import struct
        clock,count=struct.unpack_from('<QQ',self.data);t=clock-1;start=clock-count
        context=bytes(self.get(i) for i in range(t-3,t+1));counts=torch.zeros(256,dtype=torch.float64)
        for a in range(max(start+4,4),t):
            if bytes(self.get(i) for i in range(a-4,a))==context:counts[self.get(a)]+=1
        n=int(counts.sum());return (counts+8*prior)/(n+8),n

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('plan','train','evaluate'))
    p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trained',type=Path);p.add_argument('--documents',type=Path);a=p.parse_args()
    if a.phase=='plan':
        if a.output.exists():raise FileExistsError('frozen plan exists')
        a.output.write_text(json.dumps(plan(),indent=2,sort_keys=True)+'\n')
    elif a.phase=='train':
        if a.documents is not None or a.trained is not None:raise ValueError('FIT-only interface excludes unseen source paths')
        train(a.seed,a.output)
    else:
        if a.trained is None or a.documents is None:raise ValueError('sealed trained source paths required')
        r=evaluate(a.seed,a.trained,a.documents,a.output);a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+'\n')
        print(json.dumps({'seed':a.seed,'metrics':r['metrics'],'autonomous':r['autonomous_query_metrics']}))
if __name__=='__main__':main()
