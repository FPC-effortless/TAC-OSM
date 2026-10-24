"""No-fit real-input whole graph and active-gradient census for visual native model."""
import json
from pathlib import Path
import torch
import scripts.me091_train_world as R

class RawGraph(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x,a,dt,images):
        out,state=self.model.visual_tensor_graph(x,a,dt,images)
        return tuple(out.values())+tuple(state)

def main():
    torch.set_num_threads(1);R.configure('me091')
    _,fit,_,identified,evidence=R.load_original(Path('/tmp/me077-original-source'))
    m=R.VisualWorld(identified);source=torch.load('research/replay-bundles/ME087/seed-87001/final.pt',map_location='cpu',weights_only=True);R.load_initial(m,source['weights'])
    picks,weights,_=R.schedule(fit,identified,R.SEEDS[0]);x,a,y,valid,dt,images=R.batch(fit,picks[0])
    raw=m(x,a,elapsed=dt,images=images)
    loss=R.training_loss(raw,y,x,valid,torch.tensor(weights[0]),m.state_scale,float((m.delta_scale*m.state_scale).mean()),3*torch.maximum(m.quiet_noise,.01*m.delta_scale))
    loss.backward();grad={n:{'elements':p.numel(),'finite':p.grad is not None and bool(torch.isfinite(p.grad).all()),'nonzero':p.grad is not None and bool(p.grad.abs().sum()>0),'gradient_norm':None if p.grad is None else float(p.grad.norm())} for n,p in m.named_parameters()}
    if not all(g['finite'] and g['nonzero'] for g in grad.values()):raise RuntimeError('Inactive/nonfinite whole graph parameter')
    graph=torch.export.export(RawGraph(m),(x[:2],a[:2],dt[:2],images[:2])).graph_module
    count=len(list(graph.graph.nodes));R.write(R.ROOT/'REAL_FIT_GRAPH_AUDIT.json',{'no_optimizer_step':True,'actual_trainable_parameters':sum(p.numel() for p in m.parameters()),'parameter_tensors':len(grad),'gradient_census':grad,'exported_raw_whole_graph_nodes':count,'profile':{'batch':2,'history':32,'image':[1,32,32]},'input_source_sha256':evidence['original_160_episode_source_sha256'],'visual_source_sha256':evidence['visual_source']['canonical_sha256'],'parallel':'shared CNN, numeric input projections, causal convolutions, and shared horizon/coordinate readouts are batched; nonlinear native state evolution remains sequential','reuse':'one CNN embedding per observed frame, encoded-history cache reused in incremental mode; no future frame, autoregressive invented observation or hidden mutable input'})
    (R.ROOT/'REAL_FIT_RAW_GRAPH.txt').write_text(str(graph.graph)+'\n')
    print(json.dumps({'whole_model_active_gradient_tensors':len(grad),'graph_nodes':count,'parameters':sum(p.numel() for p in m.parameters()),'status':'PASS'}),flush=True)

if __name__=='__main__':main()
