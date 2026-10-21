"""Mathematical falsifiers only; generated inputs are not capability evidence."""
import types
import pytest
import torch
from tac_osm.coupled_native import CoupledNative
from tac_osm.me025_feature_parity import KernelNative,TinyCausalTransformer,kernel_summaries
from tac_osm.native_forecast_runtime import NativeForecastRuntime,kernel_segment,compose_kernel_segments


def arguments():
    torch.manual_seed(26026)
    meta=torch.rand(3,128,3); meta[:,:,1]=(meta[:,:,1]>.5).float()
    return torch.randint(16000,(3,128)),meta,torch.randint(16000,(3,)),torch.rand(3,1)


def source(kernel):
    torch.manual_seed(26)
    net=KernelNative(.7,.05,torch.linspace(.2,.8,16000)) if kernel else CoupledNative(0.,True,.7,.05)
    with torch.no_grad(): net.head.weight.normal_(std=.1); net.head.bias.fill_(.13)
    return net.eval()


def test_active_head_original_compiled_and_streaming_equality(tmp_path):
    torch.set_num_threads(1); args=arguments()
    for kernel in [False,True]:
        old=source(kernel); runtime=NativeForecastRuntime(old,kernel).eval()
        assert sum(p.numel() for p in runtime.parameters())==(135056 if kernel else 135050)
        compiled=torch.jit.script(runtime)
        with torch.no_grad():
            expected=old(*args)
            torch.testing.assert_close(compiled(*args),expected,atol=1e-6,rtol=0)
            state=torch.zeros(3,runtime.state_width)
            for t in range(128): state=compiled.stream_step(args[0][:,t],args[1][:,t],state)
            prior=(args[1][:,-32:,1].sum(-1,keepdim=True)+1)/34
            torch.testing.assert_close(compiled.readout(state,args[2],args[3],prior),expected,atol=1e-6,rtol=0)
            path=tmp_path/f'{kernel}.pt'; torch.jit.save(compiled,str(path))
            torch.testing.assert_close(torch.jit.load(str(path))(*args),expected,atol=1e-6,rtol=0)


def test_kernel_segment_identity_associativity_and_irregular_chunks():
    torch.manual_seed(27); feedback=torch.rand(4,128,dtype=torch.float64)
    a,b,c=[kernel_segment(feedback[:,x:y]) for x,y in [(0,11),(11,97),(97,128)]]
    left=compose_kernel_segments(compose_kernel_segments(a,b),c)
    right=compose_kernel_segments(a,compose_kernel_segments(b,c))
    direct=kernel_segment(feedback); empty=kernel_segment(feedback[:,:0])
    for segment in [left,right,compose_kernel_segments(empty,direct),compose_kernel_segments(direct,empty)]:
        for actual,expected in zip(segment,direct):
            torch.testing.assert_close(actual,expected,atol=2e-14,rtol=2e-14)
    torch.testing.assert_close(left[1]/left[2],kernel_summaries(feedback),atol=2e-15,rtol=2e-15)


def test_kernel_impulse_derivatives_match_normalized_exponentials():
    feedback=torch.rand(1,17,dtype=torch.float64,requires_grad=True)
    def normalize(x):
        _,s,w=kernel_segment(x[None]); return (s/w)[0]
    j=torch.autograd.functional.jacobian(normalize,feedback[0])
    ages=torch.arange(16,-1,-1,dtype=torch.float64)
    weights=torch.exp2(-ages[None,:]/torch.tensor([2.,8.,16.,32.,64.],dtype=torch.float64)[:,None])
    torch.testing.assert_close(j,weights/weights.sum(-1,keepdim=True),atol=1e-15,rtol=1e-14)
    torch.testing.assert_close(j.sum(-1),torch.ones(5,dtype=torch.float64),atol=1e-15,rtol=0)


def causal_errors(net,args):
    base=net.prefix_logits(*args); errors=[]
    for col,depth in enumerate([32,64,96]):
        meta=args[1].clone(); meta[:,depth:,1]=1-meta[:,depth:,1]
        items=args[0].clone(); items[:,depth+1:]=(items[:,depth+1:]+77)%16000
        meta[:,depth+1:,0]=meta[:,depth+1:,0]+.5
        trial=net.prefix_logits(items,meta,args[2],args[3])
        errors.append(float((base[:,col]-trial[:,col]).abs().max().detach()))
    return errors


def test_causality_with_active_heads_and_deliberate_unmasked_falsifier():
    args=arguments(); native=source(True)
    torch.manual_seed(28); attention=TinyCausalTransformer(.7,.05,torch.linspace(.2,.8,16000)).eval()
    with torch.no_grad(): attention.head.weight.normal_(std=.1)
    assert max(causal_errors(native,args))<=3e-6
    assert max(causal_errors(attention,args))<=3e-6
    def unmasked(self,items,metadata):
        v=self.in_proj(torch.cat((self.emb(items),metadata),-1))+self.positions.unsqueeze(0)
        return self.to_state(self.final_norm(self.blocks(v)))
    attention.causal_states=types.MethodType(unmasked,attention)
    assert max(causal_errors(attention,args))>1e-4


def test_compiled_readout_requires_correct_state_and_observed_kernel_counts():
    runtime=torch.jit.script(NativeForecastRuntime(source(True),True))
    with pytest.raises((ValueError,torch.jit.Error)):
        runtime.readout(torch.zeros(3,74),torch.zeros(3,dtype=torch.long),torch.zeros(3,1),torch.full((3,1),.5))
    with pytest.raises((ValueError,torch.jit.Error)):
        runtime.stream_step(torch.zeros(3,dtype=torch.long),torch.zeros(3,3),torch.zeros(3,64))


def test_compiled_forecast_preserves_input_derivatives():
    args=arguments()
    for kernel in [False,True]:
        old=source(kernel).double(); runtime=NativeForecastRuntime(old,kernel).double()
        x=args[1].double().requires_grad_(); d=args[3].double().requires_grad_()
        a=old(args[0],x,args[2],d).sum(); ga=torch.autograd.grad(a,(x,d))
        b=runtime(args[0],x,args[2],d).sum(); gb=torch.autograd.grad(b,(x,d))
        for lhs,rhs in zip(ga,gb): torch.testing.assert_close(lhs,rhs,atol=1e-12,rtol=1e-10)
