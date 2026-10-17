"""Stream public receipts through a restored DEV056 research model.

The matched GRU is the default reference. Predictions do not constitute a
safety certificate. This interface accepts no simulator or hidden labels.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from tac_osm.development.learned_stack import PublicEvent,COSTS
from tac_osm.development.public_memory_deadline import DeadlineModel,public_features,ARMS


def parse_event(value):
    if set(value)!={'action','sensors','exact','tick','hazard'}:raise ValueError('exact public receipt fields required')
    if (type(value['action']) is not int or value['action'] not in range(5)
        or type(value['tick']) is not int or not 0<=value['tick']<24
        or type(value['hazard']) not in (float,int) or value['hazard'] not in (.05,.25)
        or any(not isinstance(value[k],list) or len(value[k])!=2 or
               any(type(x) is not int or x not in (0,1) for x in value[k]) for k in ('sensors','exact'))):
        raise ValueError('invalid public receipt')
    return PublicEvent(value['action'],tuple(value['sensors']),tuple(value['exact']),value['tick']),value['hazard']


@torch.no_grad()
def run(checkpoint,arm,lines):
    torch.set_num_threads(1)
    model=DeadlineModel(0,arm).eval()
    model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu'))
    state=model.initial(torch.zeros(1,11));declared=None
    for tick,line in enumerate(lines):
        event,hazard=parse_event(json.loads(line))
        if event.tick!=tick:raise ValueError('stream must start at zero and advance one tick at a time')
        if declared is not None and hazard!=declared:raise ValueError('public hazard must stay fixed within a world')
        declared=hazard
        state=model.update(state,torch.tensor([public_features(event,hazard)],dtype=torch.float32))
        expanded=tuple(s.expand(5,*s.shape[1:]) if s is not None else None for s in state)
        probability=model.predict(expanded,torch.arange(5),torch.full((5,),tick)).sigmoid()
        costs=torch.tensor(COSTS)+4*probability.sum(-1)
        print(json.dumps({'tick':tick,'arm':arm,'post_fault_probabilities':probability.tolist(),
                          'predicted_costs':costs.tolist(),'chosen_action':int(costs.argmin()),
                          'backend_sample_updates':model.last_write_count}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--arm',choices=ARMS,default='gru_matched');parser.add_argument('--events')
    args=parser.parse_args()
    if args.events:
        with open(args.events) as stream:run(args.checkpoint,args.arm,stream)
    else:run(args.checkpoint,args.arm,sys.stdin)
