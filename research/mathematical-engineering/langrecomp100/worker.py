"""Actual paired-policy16-update continuation worker, FIT only, original retained."""
import argparse,gc,importlib.util,json,os,resource,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
_spec=importlib.util.spec_from_file_location('langrecomp100_common',Path(__file__).with_name('common.py'));C=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=C;_spec.loader.exec_module(C)

def proc_memory():
    values={}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:','VmHWM:')):
            k,v,*_=line.split();values[k[:-1]]=int(v)*1024
    return values

def reset_hwm():
    try:
        Path('/proc/self/clear_refs').write_text('5\n');return True
    except OSError:return False

class Saves:
    def __init__(self,model):
        self.parameter_storages={p.untyped_storage().data_ptr() for p in model.parameters()};self.logical=0;self.events=0;self.nonparameter={}
    def pack(self,t):
        self.logical+=t.numel()*t.element_size();self.events+=1
        if t.numel():
            storage=t.untyped_storage();pointer=storage.data_ptr()
            if pointer not in self.parameter_storages:self.nonparameter[pointer]=max(self.nonparameter.get(pointer,0),storage.nbytes())
        # Packed values are data views; retaining grad_fn can create cycles.
        return t.detach()
    def unpack(self,t):return t
    def report(self):return {'logical_requested_bytes':self.logical,'save_requests':self.events,
        'distinct_nonparameter_requested_storage_bytes':sum(self.nonparameter.values()),
        'scope':'Forward save-hook requests/identifiers only; not live peak or physical VRAM; parameter aliases may appear in logical totals.'}

def cold(args):
    C.C.setup(args.seed);payload=torch.load(args.checkpoint,map_location='cpu',weights_only=True);model=C.C.make(args.seed);model.load_state_dict(payload['model']);model.eval()
    x=torch.tensor(C.read(args.inputs),dtype=torch.long)
    with torch.inference_mode():y,s=model(x,owners=tuple('cold-'+str(i) for i in range(len(x))))
    with args.output.open('xb') as f:np.savez(f,logits=y.numpy(),**{n:t.numpy() for n,t in s.tensors.items()})

def run(args):
    reg=C.read(C.HERE/'REGISTER.json')
    if reg['continuation_enabled'] is not True:raise ValueError('continuation disabled pending source/proof checks')
    if reg['math_gate'].get('status')!='PASS':raise ValueError('scoped math gate required')
    if args.seed not in C.SEEDS or args.policy not in C.POLICIES:raise ValueError('fixed seed/policy matrix required')
    source=C.source_hashes(args.source_commit)
    if args.output.exists():raise FileExistsError('never overwrite original trial')
    args.output.mkdir(parents=True);history=[];completed=0;started=time.perf_counter()
    model,optimizer,payload,parent=C.load_parent(args.parent,args.seed);model.train()
    before=C.C.state_digest(model);initial_optimizer=C.optimizer_digest(optimizer,model);parent_rng=C.tensor_digest({'torch':torch.random.get_rng_state()})
    corpus,windows,episodes,_=C.C.load_fit(args.data,C.read(C.C.HERE/'REGISTER.json'))
    all_plan=C.C.schedule(args.seed,2048+reg['updates'],windows,episodes)
    journal=C.read(args.parent/'SAMPLING_JOURNAL.json')
    if all_plan[:2048]!=journal['steps'] or C.digest(all_plan[:2048])!=parent['sample_sha256']:raise ValueError('original2048-step data order differs')
    plan=all_plan[2048:];C.save(args.output/'SAMPLING_JOURNAL.json',{'original_prefix_sha256':parent['sample_sha256'],'continuation':plan,'sha256':C.digest(plan)})
    del payload;gc.collect()
    # Matched native/recompute warmup: two F/B passes per data mode, zero updates.
    activation_profiles={}
    for item in (plan[0],plan[3],plan[0],plan[3]):
        optimizer.zero_grad(set_to_none=True);collector=Saves(model)
        with torch.autograd.graph.saved_tensors_hooks(collector.pack,collector.unpack):
            value,logits,state=C.loss(model,C.C.batch(item,corpus,windows,episodes),args.policy)
        value.backward();C.gradients(model)
        activation_profiles[item['mode']]={**collector.report(),'loss':float(value.detach()),'measurement':'second zero-update original-parent warmup batch; separate from timing/RSS updates'}
        del value,logits,state,collector
    optimizer.zero_grad(set_to_none=True);gc.collect()
    if C.C.state_digest(model)!=before or C.optimizer_digest(optimizer,model)!=initial_optimizer:raise ValueError('zero-update warmup changed original weights/moments')
    observed_lifetime_peak=proc_memory()['VmHWM']
    try:
        for number,item in enumerate(plan):
            if time.perf_counter()-started>reg['max_worker_seconds']:raise TimeoutError('registered continuation budget exceeded')
            values=C.C.batch(item,corpus,windows,episodes);optimizer.zero_grad(set_to_none=True)
            counter={};baseline=proc_memory();observed_lifetime_peak=max(observed_lifetime_peak,baseline['VmHWM']);hwm_reset=reset_hwm()
            tick=time.perf_counter()
            # Actual timed updates have no saved-tensor instrumentation hooks.
            value,logits,state=C.loss(model,values,args.policy,counter)
            forward_seconds=time.perf_counter()-tick
            tick=time.perf_counter();value.backward();backward_seconds=time.perf_counter()-tick
            after_backward=proc_memory();raw_hash,norms=C.gradients(model)
            outputs={'logits':logits,**state.tensors};output_hash=C.tensor_digest(outputs);loss_value=float(value.detach())
            if not bool(torch.isfinite(value)):raise ValueError('nonfinite actual FIT loss')
            clip_tick=time.perf_counter();clip=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False)
            if not bool(torch.isfinite(clip)):raise ValueError('nonfinite actual global gradient norm')
            clip_seconds=time.perf_counter()-clip_tick
            tick=time.perf_counter();optimizer.step();optimizer_seconds=time.perf_counter()-tick;completed+=1
            memory=proc_memory();peak=max(after_backward['VmHWM'],memory['VmHWM']);observed_lifetime_peak=max(observed_lifetime_peak,peak)
            row={'step':completed,'absolute_step':2048+completed,'mode':item['mode'],'loss':loss_value,'outputs_state_sha256':output_hash,
                'raw_all_parameter_gradient_sha256':raw_hash,'parameter_gradient_norms':norms,'gradient_norm_before_clip':float(clip),
                'model_sha256':C.tensor_digest(model.state_dict()),'optimizer_sha256':C.optimizer_digest(optimizer,model),
                'torch_rng_sha256':C.tensor_digest({'torch':torch.random.get_rng_state()}),'forward_calls_including_recompute':counter.get('calls',0),
                'forward_seconds':forward_seconds,'backward_seconds':backward_seconds,'optimizer_seconds':optimizer_seconds,
                'clip_seconds':clip_seconds,'forward_backward_optimizer_seconds':forward_seconds+backward_seconds+clip_seconds+optimizer_seconds,
                'memory':{'baseline_rss_bytes':baseline['VmRSS'],'whole_process_step_peak_rss_bytes':peak if hwm_reset else None,
                          'hwm_reset_supported':hwm_reset,'kernel_hwm_after_step_bytes':memory['VmHWM'],'observed_lifetime_peak_rss_bytes':observed_lifetime_peak},'exposure':values[-1]}
            history.append(row);del value,logits,state,outputs,values
        if len(optimizer.state)!=17 or any(float(s['step'])!=2048+reg['updates'] for s in optimizer.state.values()):raise ValueError('incomplete actual AdamW continuation')
        if any(not bool(torch.isfinite(t).all()) for t in model.parameters()):raise ValueError('nonfinite actual continuation parameter')
        if C.sha(args.parent/'MODEL.pt')!=parent['checkpoint_sha256']:raise ValueError('original parent archive was altered')
        model.eval();p=args.output/'MODEL.pt';meta={'study':'LANGRECOMP100','seed':args.seed,'policy':args.policy,'optimizer_steps':2048+completed,
            'new_optimizer_steps':completed,'source_commit':args.source_commit,'source_hashes':source,'counts':model.counts(),'parent_checkpoint_sha256':parent['checkpoint_sha256']}
        final_sha=C.C.checkpoint(p,model,optimizer,meta)
        tokens=C.C.batch(plan[0],corpus,windows,episodes)[0];inputs=args.output/'COLD_FIT_INPUTS.json';C.save(inputs,tokens.tolist())
        with torch.inference_mode():y,s=model(tokens,owners=tuple('cold-'+str(i) for i in range(len(tokens))))
        with (args.output/'COLD_ORIGINAL.npz').open('xb') as f:np.savez(f,logits=y.numpy(),**{n:t.numpy() for n,t in s.tensors.items()})
        subprocess.run([sys.executable,str(Path(__file__)),'--mode','cold','--seed',str(args.seed),'--checkpoint',str(p),'--inputs',str(inputs),'--output',str(args.output/'COLD_REPLAY.npz')],check=True)
        with np.load(args.output/'COLD_ORIGINAL.npz',allow_pickle=False) as a,np.load(args.output/'COLD_REPLAY.npz',allow_pickle=False) as b:
            if set(a.files)!=set(b.files) or any(not np.array_equal(a[n],b[n]) for n in a.files):raise ValueError('fresh original model/state replay changed')
        result={'status':'COMPLETE_FIT_ONLY_CONTINUATION','seed':args.seed,'policy':args.policy,'steps':completed,'absolute_optimizer_steps':2048+completed,
            'trainable_parameters':model.counts()['trainable'],'deployed_parameters':model.counts()['deployed'],'parent_checkpoint_sha256':parent['checkpoint_sha256'],
            'initial_model_sha256':before,'initial_optimizer_sha256':initial_optimizer,'initial_torch_rng_sha256':parent_rng,
            'sample_sha256':C.digest(plan),'activation_profiles':activation_profiles,'source_commit':args.source_commit,'source_hashes':source,'history':history,'checkpoint_sha256':final_sha,
            'cold_replay_bitwise':True,'hardware':C.C.hardware(),'elapsed_seconds':time.perf_counter()-started,'DEV_opened':False,'GPU_minutes':0}
        C.save(args.output/'FIT_RECEIPT.json',result);print({'seed':args.seed,'policy':args.policy,'steps':completed,'status':result['status']},flush=True)
    except BaseException as error:
        C.save(args.output/'FAILED.json',{'seed':args.seed,'policy':args.policy,'completed_steps':completed,'history':history,'reason':repr(error)})
        C.C.checkpoint(args.output/'PARTIAL_MODEL.pt',model,optimizer,{'status':'FAILED_RETAINED','new_optimizer_steps':completed});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('fit','cold'),default='fit');p.add_argument('--seed',type=int,required=True);p.add_argument('--policy',choices=C.POLICIES)
    p.add_argument('--parent',type=Path);p.add_argument('--data',type=Path);p.add_argument('--source-commit');p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--inputs',type=Path)
    a=p.parse_args();cold(a) if a.mode=='cold' else run(a)
