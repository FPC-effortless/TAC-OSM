"""Real FIT mathematical/gradient graph proof, no optimizer updates or REPORT."""
import torch,json
from pathlib import Path
import scripts.me095_train_world as R
from tac_osm.gradient_surgery import task_gradient_matrix,projected_task_gradients
from tac_osm.variants.median_anchored_distribution import mixture_crps,log_density,implicit_median
from torch.nn import functional as F

def main():
    R.configure('me095');R.integrity();torch.set_num_threads(1)
    _,fit,_,identified,_=R.load_original(Path('/tmp/me077-original-source'))
    m=R.ContextFilterWorld(identified);old=torch.load(R.visual_parent(R.SEEDS[0]),weights_only=True,map_location='cpu');R.load_initial(m,old['weights'])
    entries,ww,_=R.schedule(fit,identified,R.SEEDS[0]);x,a,y,valid,dt,images,task_ids,context=R.batch(fit,entries[0][:5]);weights=torch.tensor(ww[0][:5]);normalizer=float((m.delta_scale*m.state_scale).mean())
    p=m(x,a,elapsed=dt,images=images,task_ids=task_ids,clock_context=context);tasks=R.task_losses(p,y,x,valid,weights,m.state_scale,normalizer)
    h=y.new_tensor((1,4,16))[None,None,:,None];factual=(((y-x[:,:,None]).abs()*m.state_scale).mean(-1)>=.025).float()
    rawscore=(mixture_crps(p,y)*m.state_scale/(normalizer*h)).mean(-1)+.2*(-log_density(p,y)+m.state_scale.log()).mean(-1)+20*((implicit_median(p)-y.double()).abs()*m.state_scale/(normalizer*h)).mean(-1)+.2*F.binary_cross_entropy_with_logits(p['motion_logit'],factual,reduction='none')
    unsplit=(weights*(rawscore*valid).sum((1,2))/valid.sum((1,2)).clamp_min(1)).mean()
    torch.testing.assert_close(sum(tasks),unsplit,rtol=1e-5,atol=1e-6)
    expected=torch.cat([g.reshape(-1).double() for g in torch.autograd.grad(unsplit,tuple(m.parameters()),retain_graph=True)])
    matrix=task_gradient_matrix(tasks,tuple(m.parameters()));torch.testing.assert_close(matrix.sum(0),expected,rtol=1e-5,atol=1e-6)
    p=m(x,a,elapsed=dt,images=images,task_ids=task_ids,clock_context=context);stats=projected_task_gradients(R.task_losses(p,y,x,valid,weights,m.state_scale,normalizer),m.parameters())
    grads={n:{'finite':v.grad is not None and bool(torch.isfinite(v.grad).all()),'nonzero':v.grad is not None and bool(v.grad.abs().sum()>0)} for n,v in m.named_parameters()}
    if not all(v['finite'] and v['nonzero'] for v in grads.values()):raise RuntimeError('Inactive/nonfinite actual FIT parameter path')
    result={'status':'PASS','optimizer_updates':0,'parameters':sum(p.numel() for p in m.parameters()),'source':'5authentic FIT32framewindows,noREPORT','independent_unsplit_loss_and_gradient_partition':'PASS','gradient_rtol':1e-5,'gradient_atol':1e-6,'max_absolute_gradient_difference':float((matrix.sum(0)-expected).abs().max()),'projection':stats,'parameter_gradients':grads}
    R.write(R.ROOT/'REAL_FIT_OPTIMIZATION_PREFLIGHT.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
