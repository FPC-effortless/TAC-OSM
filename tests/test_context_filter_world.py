import torch
from tac_osm.variants.context_filter_world import ContextFilterWorld,affine_prefix
from tac_osm.variants.median_anchored_world import MedianAnchoredWorld
from scripts.me095_train_world import load_initial


def test_affine_scan_matches_independent_serial_values_and_gradients():
    torch.manual_seed(9595);torch.set_num_threads(1)
    a=torch.rand(2,37,28,dtype=torch.float64,requires_grad=True);b=torch.randn_like(a,requires_grad=True);initial=torch.randn(2,28,dtype=a.dtype,requires_grad=True)
    value=initial;parts=[]
    for t in range(37):value=a[:,t]*value+b[:,t];parts.append(value)
    serial=torch.stack(parts,1);parallel=affine_prefix(a,b,initial)
    torch.testing.assert_close(parallel,serial,rtol=1e-12,atol=1e-12)
    expected=torch.autograd.grad(serial.square().mean(),(a,b,initial),retain_graph=True);actual=torch.autograd.grad(parallel.square().mean(),(a,b,initial))
    for u,v in zip(expected,actual):torch.testing.assert_close(u,v,rtol=1e-10,atol=1e-10)
    values=torch.randn_like(a);bounded=affine_prefix(a.detach(),(1-a.detach())*values,initial.detach())
    prefix=torch.cat((initial.detach()[:,None],values),1)
    for t in range(37):
        assert (bounded[:,t]>=prefix[:,:t+2].amin(1)-1e-12).all()
        assert (bounded[:,t]<=prefix[:,:t+2].amax(1)+1e-12).all()


def inputs(t=7):
    torch.manual_seed(9596);torch.set_num_threads(1);x=torch.randn(2,t,14,dtype=torch.float64)*.01
    return x,torch.randn_like(x)*.01,torch.rand(2,t,1,dtype=x.dtype)*.1+.01,torch.rand(2,t,1,32,32,dtype=x.dtype),torch.tensor([0,1]),torch.rand(2,t,6,dtype=x.dtype)


def test_whole_model_budget_retained_weights_and_active_filter_context_gradients():
    m=ContextFilterWorld().double();parent=MedianAnchoredWorld().double();load_initial(m,parent.state_dict())
    assert sum(p.numel() for p in m.parameters())==104689
    for name,value in parent.state_dict().items():torch.testing.assert_close(m.state_dict()[name],value,rtol=0,atol=0)
    x,a,dt,im,task,context=inputs();p,s=m(x,a,elapsed=dt,images=im,task_ids=task,clock_context=context,return_state=True)
    assert len(s)==15 and sum(v[0].numel() for v in s)==1359
    from tac_osm.variants.median_anchored_distribution import log_density,mixture_crps
    y=x[:,:,None]+.001;loss=(mixture_crps(p,y)-.2*log_density(p,y)+20*(p['median']-y).abs()).mean()+p['motion_logit'].square().mean();loss.backward()
    for name,par in m.named_parameters():assert par.grad is not None and torch.isfinite(par.grad).all(),name
    for name in ['filter_rates.weight','filter_context.weight','clock_encoder.weight','task_embedding.weight']:assert dict(m.named_parameters())[name].grad.abs().sum()>0,name


def test_context_causality_full_chunk_and_streaming_equivalence():
    m=ContextFilterWorld().double()
    with torch.no_grad():m.task_embedding.weight.normal_(std=.03);m.clock_encoder.weight.normal_(std=.01);m.filter_rates.weight.normal_(std=.03)
    x,a,dt,im,task,ctx=inputs();whole,final=m(x,a,elapsed=dt,images=im,task_ids=task,clock_context=ctx,return_state=True)
    xx=x.clone();xx[:,4:]+=1;ii=im.clone();ii[:,4:]=1-ii[:,4:];cc=ctx.clone();cc[:,4:]+=2
    alt=m(xx,a,elapsed=dt,images=ii,task_ids=task,clock_context=cc)
    for k in whole:torch.testing.assert_close(whole[k][:,:4],alt[k][:,:4],rtol=1e-10,atol=1e-10)
    _,s=m(x[:,:3],a[:,:3],elapsed=dt[:,:3],images=im[:,:3],task_ids=task,clock_context=ctx[:,:3],return_state=True)
    end,s=m(x[:,3:],a[:,3:],s,elapsed=dt[:,3:],images=im[:,3:],task_ids=task,clock_context=ctx[:,3:],return_state=True)
    stream=None;parts=[]
    for t in range(x.shape[1]):
        out,stream=m.step(x[:,t],a[:,t],stream,elapsed=dt[:,t],images=im[:,t],task_ids=task,clock_context=ctx[:,t]);parts.append(out)
    last=m.forecast_last(x,a,elapsed=dt,images=im,task_ids=task,clock_context=ctx)
    for k in whole:
        torch.testing.assert_close(whole[k][:,3:],end[k],rtol=1e-8,atol=1e-8)
        torch.testing.assert_close(whole[k],torch.stack([r[k] for r in parts],1),rtol=1e-8,atol=1e-8)
        torch.testing.assert_close(whole[k][:,-1],last[k],rtol=1e-8,atol=1e-8)
    for u,v,w in zip(final,s,stream):torch.testing.assert_close(u,v,rtol=1e-10,atol=1e-10);torch.testing.assert_close(u,w,rtol=1e-10,atol=1e-10)
