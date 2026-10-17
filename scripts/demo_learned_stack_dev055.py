"""Restore DEV-055 and stream public JSON events into action forecasts.

Input: one object per line, exactly action/sensors/exact/tick. Output: learned
post-action fault probabilities and the registered one-step cost decision.
No simulator, hidden state, targets or training labels are accepted here.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from tac_osm.development.learned_stack import LearnedStack, PublicEvent, COSTS


def parse_event(value):
    if set(value) != {'action','sensors','exact','tick'}:
        raise ValueError('exactly action, sensors, exact, tick required')
    if (type(value['action']) is not int or value['action'] not in range(5)
        or type(value['tick']) is not int or not 0 <= value['tick'] < 20
        or any(not isinstance(value[k],list) or len(value[k])!=2
               or any(type(x) is not int or x not in (0,1) for x in value[k]) for k in ('sensors','exact'))):
        raise ValueError('invalid public event')
    return PublicEvent(value['action'],tuple(value['sensors']),tuple(value['exact']),value['tick'])


@torch.no_grad()
def run(checkpoint,arm,events):
    torch.set_num_threads(1)
    model=LearnedStack(0,arm).eval()
    model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu'))
    state=model.initial(torch.zeros(1,10))
    for expected_tick,line in enumerate(events):
        event=parse_event(json.loads(line))
        if event.tick != expected_tick: raise ValueError('stream must start at zero and advance one tick at a time')
        state=model.update(state,torch.tensor([event.tensor()],dtype=torch.float32))
        expanded=tuple(s.expand(5,*s.shape[1:]) if s is not None else None for s in state)
        p=model.predict(expanded,torch.arange(5),torch.full((5,),event.tick)).sigmoid()
        costs=torch.tensor(COSTS)+4*p.sum(-1)
        print(json.dumps({'tick':event.tick,'fault_probabilities':p.tolist(),
                          'chosen_action':int(costs.argmin()),'predicted_costs':costs.tolist(),
                          'work':model.last_work}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--arm',default='mtsk')
    parser.add_argument('--events',help='JSONL public events; defaults to stdin')
    args=parser.parse_args()
    if args.events:
        with open(args.events) as stream: run(args.checkpoint,args.arm,stream)
    else: run(args.checkpoint,args.arm,sys.stdin)
