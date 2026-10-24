"""No-fit real source proof: original score partition and batched task VJP."""
import json
from pathlib import Path
import torch
import scripts.me092_train_world as R
from scripts.me087_train_world import training_loss as original_proper_loss
from tac_osm.gradient_surgery import task_gradient_matrix,projected_task_gradients


def main():
    R.configure('me092');torch.set_num_threads(1)
    _,fit,_,identified,_=R.load_original(Path('/tmp/me077-original-source'))
    m=R.VisualWorld(identified)
    native=torch.load('research/replay-bundles/ME087/seed-87001/final.pt',weights_only=True,map_location='cpu')
    visual=torch.load('research/replay-bundles/ME091/seed-91001/final.pt',weights_only=True,map_location='cpu')
    R.load_initial(m,native['weights'],visual['weights'])
    picks,ww,_=R.schedule(fit,identified,R.SEEDS[0]);x,a,y,valid,dt,images=R.batch(fit,picks[0][:4]);weights=torch.tensor(ww[0][:4])
    norm=float((m.delta_scale*m.state_scale).mean());raw=m(x,a,elapsed=dt,images=images)
    tasks=R.task_losses(raw,y,x,valid,weights,m.state_scale,norm)
    old=original_proper_loss(raw,y,x,valid,weights,m.state_scale,norm)
    torch.testing.assert_close(sum(tasks),old,rtol=1e-10,atol=1e-10)
    expected=torch.cat([g.reshape(-1).double() for g in torch.autograd.grad(old,tuple(m.parameters()),retain_graph=True)])
    matrix=task_gradient_matrix(tasks,tuple(m.parameters()))
    torch.testing.assert_close(matrix.sum(0),expected,rtol=1e-5,atol=1e-6)
    raw=m(x,a,elapsed=dt,images=images);tasks=R.task_losses(raw,y,x,valid,weights,m.state_scale,norm)
    stats=projected_task_gradients(tasks,m.parameters())
    gradients={name:{'finite':p.grad is not None and bool(torch.isfinite(p.grad).all()),'nonzero':p.grad is not None and bool(p.grad.abs().sum()>0),'norm':None if p.grad is None else float(p.grad.norm())} for name,p in m.named_parameters()}
    if not all(d['finite'] for d in gradients.values()):raise RuntimeError('Projected original-source gradients nonfinite')
    result={'status':'PASS','optimizer_updates':0,'real_FIT_windows':4,'observed_history':32,'parameters':sum(p.numel() for p in m.parameters()),'original_unsplit_score_partition':'PASS','batched_task_gradient_sum_vs_independent_original_score_gradient':'PASS','gradient_rtol':1e-5,'gradient_atol':1e-6,'maximum_absolute_gradient_difference':float((matrix.sum(0)-expected).abs().max()),'task_dot_products':(matrix@matrix.T).tolist(),'projection':stats,'parameter_gradients':gradients,'scope':'Actual original FIT observations only; no optimization step, REPORT score, learned capability or gate pass. One shared forward and batched3task VJP; CPU/backend speed not inferred.'}
    R.write(R.ROOT/'REAL_FIT_OPTIMIZATION_PREFLIGHT.json',result);print(json.dumps({k:result[k] for k in ['status','parameters','optimizer_updates','maximum_absolute_gradient_difference','projection']}),flush=True)

if __name__=='__main__':main()
