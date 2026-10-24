"""Trace real inputs through the full parameter-differentiable tensor graph, no fit."""
import json
from collections import Counter
from pathlib import Path
import torch
from torch import nn
import scripts.me087_train_world as R
from tac_osm.variants.bounded_value_world import BoundedValueWorld


def main():
    torch.set_num_threads(1);R.configure('me087')
    _,fit,_,identified,evidence=R.load_original(Path('/tmp/me077-original-source'))
    picks=R.schedule(fit,identified,R.SEEDS[0])[0][0][:2]
    x,a,y,valid,elapsed=R.batch(fit,picks)
    class Graph(nn.Module):
        def __init__(self):super().__init__();self.model=BoundedValueWorld(identified)
        def forward(self,x,a,elapsed):
            p,state=self.model.tensor_graph(x,a,elapsed)
            return p,tuple(state)
    torch.manual_seed(87087);graph=Graph().eval()
    exported=torch.export.export(graph,(x,a,elapsed),strict=True)
    original=graph(x,a,elapsed);replayed=exported.module()(x,a,elapsed)
    for key in original[0]:torch.testing.assert_close(original[0][key],replayed[0][key])
    for u,v in zip(original[1],replayed[1]):torch.testing.assert_close(u,v)
    loss=R.training_loss(original[0],y,x,valid,torch.ones(2),graph.model.state_scale,float((graph.model.delta_scale*graph.model.state_scale).mean()))
    loss.backward()
    coverage={name:{'parameters':p.numel(),'finite_gradient':bool(torch.isfinite(p.grad).all()) if p.grad is not None else False,
                     'gradient_norm':float(p.grad.norm()) if p.grad is not None else None} for name,p in graph.named_parameters()}
    if not all(v['finite_gradient'] and v['gradient_norm']>0 for v in coverage.values()):raise RuntimeError('disconnected or invalid parameter path')
    ops=Counter(str(n.target) for n in exported.graph.nodes if n.op=='call_function')
    root=Path('research/bounded-value-search')
    (root/'FULL_TENSOR_GRAPH.txt').write_text(str(exported.graph)+'\n')
    info={'stage':'UNFITTED_REAL_FIT_INPUT_ENGINEERING_TRACE','source_sha256':evidence['original_160_episode_source_sha256'],
        'input_shape':list(x.shape),'known_command_shape':list(a.shape),'actual_parameters':sum(p.numel() for p in graph.parameters()),
        'exported_nodes':len(list(exported.graph.nodes)),'operators':dict(ops),'parameter_gradient_coverage':coverage,
        'full_export_output_equivalence':'PASS','data_dependent_Python_in_pure_graph':False,
        'parallel_axes':['batch','observation_encoder_time','depthwise_causal_filters_time_and_channel','observed_change_prefix_max','horizon_query','distribution_quantile'],
        'sequential_dependency':'native nonlinear fast/slow/associative recurrence; not an SSM conversion or parallel-scan claim',
        'no_optimizer_updates':True,'no_learned_intelligence_claim':True}
    (root/'GRAPH_TRACE.json').write_text(json.dumps(info,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'parameters':info['actual_parameters'],'exported_nodes':info['exported_nodes'],'parameter_tensors_with_finite_nonzero_gradient':len(coverage),'equivalence':'PASS'}))

if __name__=='__main__':main()
