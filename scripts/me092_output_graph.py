"""Export complete qualified-reference prefill/step graphs on real FIT observations."""
import argparse,json
from pathlib import Path
from collections import Counter
import torch
from torch import nn
import scripts.me092_train_world as R
from tac_osm.variants.visual_world import VisualWorld
from tac_osm.variants.reusable_execution import mixture_quantiles


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',choices=tuple(R.CONFIG),default='me092');args=parser.parse_args()
    R.configure(args.study);R.integrity();torch.set_num_threads(1)
    _,fit,_,identified,evidence=R.load_original(Path('/tmp/me077-original-source'))
    entries=R.schedule(fit,identified,R.SEEDS[0])[0][0][:2];x,a,_,_,dt,images=R.batch(fit,entries)
    checkpoint=R.OUT/f'seed-{R.SEEDS[0]}'/'final.pt'
    m=VisualWorld(identified)
    m.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['weights'],strict=True);m.double().eval()
    class EndToEnd(nn.Module):
        def __init__(self,model):super().__init__();self.model=model
        def forward(self,x,a,elapsed,images,state):
            raw,updated=self.model.visual_tensor_graph(x,a,elapsed,images,state,last_only=True)
            q=mixture_quantiles(raw)
            return {'lower':q[0],'median':q[1],'upper':q[2],'factual_motion_probability':raw['motion_logit'].sigmoid()},tuple(updated)
    wrapper=EndToEnd(m);reports={}
    with torch.no_grad():
        _,carried=m(x[:,:-1].double(),a[:,:-1].double(),return_state=True,elapsed=dt[:,:-1].double(),images=images[:,:-1].double())
        for label,inputs in [('prefill32',(x.double(),a.double(),dt.double(),images.double(),m.initial_state(x[:,0].double()))),
                             ('incremental_step',(x[:,-1:].double(),a[:,-1:].double(),dt[:,-1:].double(),images[:,-1:].double(),carried))]:
            program=torch.export.export(wrapper,inputs,strict=True)
            expected=wrapper(*inputs);actual=program.module()(*inputs)
            for key in expected[0]:torch.testing.assert_close(expected[0][key],actual[0][key],rtol=1e-5,atol=1e-6)
            for u,v in zip(expected[1],actual[1]):torch.testing.assert_close(u,v,rtol=1e-5,atol=1e-6)
            ops=Counter(str(n.target) for n in program.graph.nodes if n.op=='call_function')
            path=R.ROOT/f'{label.upper()}_OUTPUT_GRAPH.txt';path.write_text(str(program.graph)+'\n')
            reports[label]={'nodes':len(list(program.graph.nodes)),'operators':dict(ops),'actual_shape':list(inputs[0].shape),'full_value_equivalence':'PASS'}
    result={'stage':'trained-first-original-real-FIT-reference-graph','study':R.STUDY,'original_checkpoint_sha256':R.sha(checkpoint),
      'reference_dtype':'float64','source_sha256':evidence['original_160_episode_source_sha256'],'graphs':reports,
      'state_tensor_values_per_entity':sum(t[0].numel() for t in carried),'cache_excludes':'model weights, forecasts, tensor/container overhead',
      'graphs_include':'actual final medians, quantiles, factual risk, all thirteen typed state outputs; no neural historical decoder work if only final forecast is needed',
      'training_or_new_TEST_scoring':False,'qualification':'execution graph verification only; learned quality must pass separate original gates'}
    R.write(R.ROOT/'END_TO_END_GRAPH.json',result)
    print(json.dumps({'study':R.STUDY,'prefill_nodes':reports['prefill32']['nodes'],'step_nodes':reports['incremental_step']['nodes'],'cache_values':result['state_tensor_values_per_entity'],'status':'PASS'}))

if __name__=='__main__':main()
