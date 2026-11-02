"""Constructed source-equation/carry/gradient tests, never model capability."""
import argparse,hashlib,json,sys,math
from dataclasses import replace
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import models,reference

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def compare(a,b,dtype):
    atol,rtol=(2e-5,2e-5) if dtype==torch.float32 else (1e-8,1e-7)
    assert a.shape==b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
    assert torch.allclose(a,b,atol=atol,rtol=rtol),float((a-b).detach().abs().max())
    return float((a-b).detach().abs().max()) if a.numel() else 0.

def source_case(row,dtype,length):
    torch.manual_seed(11201)
    m=models.build(row).to(dtype).eval();before=models.preflight.parameter_digest(m)
    x=(torch.arange(2*length).reshape(2,length)*17+3)%256
    y,s=m(x);params=dict(m.named_parameters())
    yr,sr=reference.tac(x,params,depth=8)
    err=compare(y,yr,dtype)
    serr=max(compare(s.tensors[n],sr[n],dtype) for n in sr)
    def objective(z,states):return z.square().mean()+.01*sum(t.square().mean() for t in states.values())
    g=torch.autograd.grad(objective(y,s.tensors),tuple(params.values()))
    gr=torch.autograd.grad(objective(yr,sr),tuple(params.values()))
    gerr=max(compare(a,b,dtype) for a,b in zip(g,gr))
    with torch.no_grad():
        ys,ss=m(x,parallel=False);compare(y,ys,dtype)
        for n in sr:compare(s.tensors[n],ss.tensors[n],dtype)
        if length>1:
            cut=length//2;a,left=m(x[:,:cut]);b,right=m(x[:,cut:],state=left)
            compare(y,torch.cat((a,b),1),dtype)
            for n in sr:compare(s.tensors[n],right.tensors[n],dtype)
            poison=x.clone();poison[:,cut:]=(poison[:,cut:]+101)%256
            compare(y[:,:cut],m(poison)[0][:,:cut],dtype)
        single=torch.cat([m(x[i:i+1])[0] for i in range(2)])
        compare(y,single,dtype)
        wrong=dict(params);wrong['head.out.bias']=params['head.out.bias']+.1
        wrong_y,_=reference.tac(x,wrong,depth=8)
        assert not torch.allclose(y,wrong_y,atol=2e-5,rtol=2e-5)
        try:m(x,state=s,owners=('wrong-a','wrong-b'))
        except ValueError:owner_rejected=True
        else:raise AssertionError('wrong owners accepted')
        # Carry is checked against independently unrolled reference at nonzero clock.
        yp,sp=m(x,state=s)
        rp,rsp=reference.tac(x,params,depth=8,state=s.tensors,clock=s.clock)
        compare(yp,rp,dtype)
        for n in rsp:compare(sp.tensors[n],rsp[n],dtype)
    # Differentiation through nonzero carried state, including matrix and mass.
    leaves={n:t.detach().clone().requires_grad_(True) for n,t in s.tensors.items()}
    carried=replace(s,tensors=leaves)
    yc,sc=m(x,state=carried)
    yrc,src=reference.tac(x,params,depth=8,state=leaves,clock=s.clock)
    ga=torch.autograd.grad(objective(yc,sc.tensors),tuple(leaves.values()))
    gb=torch.autograd.grad(objective(yrc,src),tuple(leaves.values()))
    state_gerr=max(compare(a,b,dtype) for a,b in zip(ga,gb))
    assert models.preflight.parameter_digest(m)==before
    return {'scale':row['scale'],'arm':row['arm'],'dtype':str(dtype),'length':length,'batch':2,
        'forward_max_abs':err,'state_max_abs':serr,'parameter_gradient_max_abs':gerr,
        'carried_state_gradient_max_abs':state_gerr,'parameter_tensors':len(params),
        'checks':['serial_parallel','chunk_carry','nonzero_clock_reference','future_poison','entity_isolation','wrong_readout_rejected','wrong_owner_rejected','all_parameter_gradients','all_state_gradients','weights_unchanged']}

def inventory():
    out=[]
    for row in models.configurations():
        with torch.device('meta'):m=models.build(row)
        n=sum(p.numel() for p in m.parameters());r={**row,'trainable_parameters':n,'deployed_parameters':n}
        if row['arm'].startswith('tac'):
            r.update(head_width=m.head.width,head_parameters=sum(p.numel() for p in m.head.parameters()),
                recurrent_state_floats=m.start(('x',)).tensor_bytes()//4,
                nonlexical_state_floats=sum(math.prod(s) for k,s in m.shapes.items() if k!='lexical'))
        elif row['arm']=='stacked_gru':r['recurrent_state_floats']=row['depth']*row['width']
        out.append(r)
    for scale in ('small','medium'):
        ns=[r['trainable_parameters'] for r in out if r['scale']==scale]
        assert (max(ns)-min(ns))/max(ns)<=.01
    return out

def constructor_equivalence():
    rows=[]
    for scale,d,target in [('small',256,1_000_000),('medium',992,10_000_000)]:
        torch.manual_seed(11201);models.NATIVE.TARGET=target
        old=models.NATIVE.StackedCandidate(depth=8,d=d)
        torch.manual_seed(11201);new=models.ReallocatedTAC(target=target,d=d,c=16,k=4,v=8)
        assert models.preflight.parameter_digest(old)==models.preflight.parameter_digest(new)
        rows.append({'scale':scale,'default_geometry_initialization_byte_identical':True})
    return rows

def run(output):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    report={'schema':'langscale001.no-fit-graph.v1','classification':'CONSTRUCTED_ENGINEERING_NOT_TRAINED_CAPABILITY',
        'torch':torch.__version__,'optimizer_updates':0,'dataset_access':False,'dev_test_access':False,
        'historical_sources':models.preflight.checked_sources(),'source_files_sha256':{p.name:digest(p) for p in [HERE/'models.py',HERE/'reference.py',HERE/'verify.py']},
        'inventory':inventory(),'constructor_equivalence':constructor_equivalence(),'cases':[]}
    for row in models.configurations():
        if row['arm'] not in ('tac_memory_scaled','tac_head_to_state'):continue
        for dtype in (torch.float32,torch.float64):
            for length in (1,7,17,128):
                case=source_case(row,dtype,length);report['cases'].append(case)
                print(json.dumps({k:case[k] for k in ('arm','scale','dtype','length')}),flush=True)
    report['status']='PASS_BOUNDED_NO_FIT_CHECKS_GLOBAL_PROOFS_OPEN'
    output.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out)
