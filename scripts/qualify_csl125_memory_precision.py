"""Bounded CPU numerical qualification of opt-in R4 memory precision; NO FIT."""
from copy import deepcopy
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.audit_csl_scan_cancellation import (
    CASES,CHUNKS,OBJECTIVES,NAMES,fixture,objective,serial_torch,evaluate,difference)
from scripts.profile_csl125_r2_scales import configs
from tac_osm.csl125_r2 import selective_affine_scan
from tac_osm.csl125_r3 import CSL125R3Config
from tac_osm.csl125_tool_model import CSL125R4
from tac_osm.csl125_tool_runtime import last_features,legal_byte_logits
from tac_osm.csl125_memory_precision import enable_precise_memory_,precise_affine_scan
OUT=ROOT/'ci-artifacts/csl125-memory-precision'


def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def operator_audit():
    protocol=json.loads((ROOT/'research/csl_scan_audit_20261106/PROTOCOL.json').read_text())
    for path,expected in protocol['source_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('original operator source changed: '+path)
    rows,arrays=[],{}
    arms={'original_fp32':selective_affine_scan,'memory_precision':precise_affine_scan}
    for case in CASES:
        args=fixture(case)
        for chunk in CHUNKS:
            for obj in OBJECTIVES:
                ref=evaluate(serial_torch,args,chunk,obj,torch.float64)
                ident=f'{case}_{chunk}_{obj}'
                for arg,g in zip(NAMES,ref[2]): arrays[ident+'_reference_'+arg]=g.numpy()
                for arm,fn in arms.items():
                    leaves=[a.detach().clone().requires_grad_(True) for a in args]
                    out,end=fn(*leaves,chunk)
                    loss=objective(out,obj)
                    grads=torch.autograd.grad(loss,leaves)
                    checks={name:difference(g,rg,5e-6,5e-5) for name,g,rg in zip(NAMES,grads,ref[2])}
                    checks['forward']=difference(out.detach(),ref[0],1e-5,1e-5)
                    checks['final']=difference(end.detach(),ref[1],1e-5,1e-5)
                    failed=[k for k,v in checks.items() if v['status']!='PASS']
                    for name,g in zip(NAMES,grads): arrays[ident+'_'+arm+'_'+name]=g.detach().numpy()
                    rows.append({'case':case,'chunk':chunk,'objective':obj,'arm':arm,
                        'operator_dtype':str(out.dtype),'objective_dtype':str(loss.dtype),
                        'failed_checks':failed,'checks':checks})
    result={'rows_per_arm':84,'rows':rows,'tolerance':{'atol':5e-6,'rtol':5e-5},
        'failed_rows':{name:sum(bool(r['failed_checks']) for r in rows if r['arm']==name) for name in arms},
        'original_sources_unchanged':True,'all_original_rows_retained':len(rows)==168,
        'scope':'FP64 memory includes normalized diagnostic read/reduction; not FP32-objective clearance'}
    write('OPERATOR_AUDIT.json',result)
    np.savez_compressed(OUT/'OPERATOR_GRADIENTS.npz',**arrays)
    return {k:v for k,v in result.items() if k!='rows'}


def full_model_gradient():
    torch.manual_seed(71)
    model=enable_precise_memory_(CSL125R4()).train()
    tokens=torch.randint(0,model.config.vocab_size,(1,8))
    started=time.perf_counter()
    outputs,state=model(tokens,all_rungs=True,emit_repair_logits=True,stream_ids=('gradient',))
    loss=0.
    for o in outputs.values():
        target=torch.randint(0,model.config.vocab_size,(1,7))
        for name in ('token_logits','repair_logits'):
            loss=loss+F.cross_entropy(o[name][:,:-1].transpose(1,2),target)
        loss=loss+F.cross_entropy(o['tool_logits'],torch.tensor([1]))
        loss=loss+F.cross_entropy(o['field_logits'].transpose(1,2),torch.zeros((1,8),dtype=torch.long))
        loss=loss+F.binary_cross_entropy_with_logits(o['accept_logit'],torch.ones_like(o['accept_logit']))
        loss=loss+F.binary_cross_entropy_with_logits(o['error_logit'],torch.ones_like(o['error_logit']))
    loss.backward()
    parameters=list(model.named_parameters())
    missing=[n for n,p in parameters if p.grad is None]
    nonfinite=[n for n,p in parameters if p.grad is not None and not torch.isfinite(p.grad).all()]
    zero=[n for n,p in parameters if p.grad is not None and not torch.any(p.grad!=0)]
    result={'parameters':model.parameter_count(),'tensors':len(parameters),'missing':missing,
        'nonfinite':nonfinite,'zero':zero,'loss':float(loss.detach()),
        'wall_seconds':time.perf_counter()-started,
        'all_weights_fp32':all(p.dtype==torch.float32 for _,p in parameters),
        'all_memory_fp64':all(s.fast.dtype==s.fast_mass.dtype==torch.float64 for s in state.layers),
        'optimizer_steps':0,'scope':'actual full-size random-weight supervised graph, not learning'}
    del loss,outputs,state,model,parameters
    gc.collect()
    write('FULL_MODEL_GRADIENT.json',result)
    return result


def memory_bytes(state):
    return sum(getattr(s,k).numel()*getattr(s,k).element_size()
        for s in state.layers for k in ('fast','slow','fast_mass','slow_mass'))


def benchmark():
    results={}
    for scale in ('1m','10m','125m'):
        torch.manual_seed(72)
        rank=32 if scale=='125m' else 16
        cfg=CSL125R3Config(**asdict(configs()[scale]),correction_rank=rank)
        base=CSL125R4(cfg).eval(); precise=enable_precise_memory_(deepcopy(base))
        ids=torch.arange(256,dtype=torch.long);tokens=torch.randint(0,cfg.vocab_size,(1,64))
        times={'original':[],'memory_precision':[]}; models={'original':base,'memory_precision':precise}
        outputs={};states={}
        with torch.inference_mode():
            for name,m in models.items():
                for _ in range(2):
                    z,s=last_features(m,tokens,depth=cfg.layers)
                    legal_byte_logits(m,z,ids)
            for rep in range(7):
                order=('original','memory_precision') if rep%2==0 else ('memory_precision','original')
                for name in order:
                    start=time.perf_counter_ns()
                    z,s=last_features(models[name],tokens,depth=cfg.layers)
                    outputs[name]=legal_byte_logits(models[name],z,ids)
                    times[name].append((time.perf_counter_ns()-start)/1e6)
                    states[name]=s
        results[scale]={'parameters':base.parameter_count(),'rungs':list(cfg.rungs),
            'samples_ms':times,'median_ms':{k:statistics.median(v) for k,v in times.items()},
            'memory_state_bytes':{k:memory_bytes(s) for k,s in states.items()},
            'last_byte_logits':difference(outputs['memory_precision'],outputs['original'],3e-4,3e-5),
            'parameter_count_unchanged':precise.parameter_count()==base.parameter_count()}
        if scale=='125m':
            with torch.inference_mode():
                x=tokens[:,:24]
                full,fullstate=precise(x,all_rungs=True,stream_ids=('stream',))
                state=None; parts={r:[] for r in cfg.rungs}
                for lo,hi in ((0,7),(7,13),(13,24)):
                    out,state=precise(x[:,lo:hi],all_rungs=True,state=state,stream_ids=('stream',))
                    for r in cfg.rungs:parts[r].append(out[r]['token_logits'])
                streaming={str(r):difference(torch.cat(parts[r],1),full[r]['token_logits'],3e-4,3e-5)
                           for r in cfg.rungs}
                poisoned=x.clone();poisoned[:,16:]=(poisoned[:,16:]+13)%cfg.vocab_size
                changed,_=precise(poisoned,all_rungs=True,stream_ids=('stream',))
                future={str(r):bool(torch.equal(changed[r]['token_logits'][:,:16],full[r]['token_logits'][:,:16])) for r in cfg.rungs}
                results[scale]['full_geometry_streaming']=streaming
                results[scale]['future_poison_prefix_identical']=future
                del x,full,fullstate,state,out,parts,changed,poisoned
        del base,precise,models,tokens,outputs,states,z,s
        gc.collect()
    result={'torch':torch.__version__,'cpu':platform.processor(),'threads':torch.get_num_threads(),
        'device':'cpu','batch':1,'prefill_tokens':64,'repetitions':7,'warmups':2,
        'includes':'same-weight last_features + 256 byte logits',
        'excludes':['grammar','full autoregressive generation','actual tool effects','GPU','mobile','trained quality'],
        'scales':results}
    write('BENCHMARK.json',result)
    return result


def main():
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():raise RuntimeError('CPU-only diagnostic')
    OUT.mkdir(parents=True,exist_ok=True)
    op=operator_audit();print('OPERATOR',json.dumps(op),flush=True)
    full=full_model_gradient();print('FULL_MODEL',json.dumps(full),flush=True)
    bench=benchmark();print('BENCHMARK',json.dumps(bench),flush=True)
    failures=[]
    if op['failed_rows']['memory_precision']!=0:failures.append('operator')
    if full['missing'] or full['nonfinite'] or full['zero']:failures.append('full_gradients')
    if not full['all_weights_fp32'] or not full['all_memory_fp64']:failures.append('precision_boundary')
    for scale,row in bench['scales'].items():
        if row['last_byte_logits']['status']!='PASS':failures.append(scale+'_forward')
    for rung,row in bench['scales']['125m']['full_geometry_streaming'].items():
        if row['status']!='PASS':failures.append('stream_'+rung)
    if not all(bench['scales']['125m']['future_poison_prefix_identical'].values()):failures.append('future_poison')
    summary={'status':'PASS_ENGINEERING_ONLY' if not failures else 'FAIL',
        'failed_gates':failures,'operator':op,'full_model':full,
        'benchmark_medians':{s:v['median_ms'] for s,v in bench['scales'].items()},
        'optimizer_steps':0,'gpu_minutes':0,'training_authorized':False,
        'remaining':['source-qualified data/splits','independent evaluator signatures and ACL evidence',
                     'current v2 fair-registration','GPU/mobile kernel cost','learned tool accuracy']}
    write('SUMMARY.json',summary);print('SUMMARY',json.dumps(summary),flush=True)
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
