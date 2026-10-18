"""Synthetic public data-acquisition accounting, separate from fitting/evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from run_structured_memory_dev061 import PROTOCOL,ResearchWorld,teacher
from run_learned_stack_dev055 import write,digest


def run(directory,output):
    p=json.loads(PROTOCOL.read_text());result={'protocol_sha256':digest(PROTOCOL),'splits':{},
       'scope':'One shared synthetic corpus, not reacquired per epoch or architecture. Observation and physically executed collection actions only; no terminal-task charge on unclosed training snapshots. Private instrumentation/analytic counterfactual labels are free in this simulator; real labeling/acquisition costs UNKNOWN. No commercial amortization or monetary CPU conversion.'}
    for name,key in [('train','train_seed'),('validation','validation_seed')]:
        ds=torch.load(Path(directory)/'_datasets'/f'{name}.pt',weights_only=True);cost=observations=actions=0.;counts={'inspect':0,'repair':0}
        for i in range(len(ds['current'])):
            n=(2,4,6)[i%3];h=(16,32,64,96)[(i//3)%4];w=ResearchWorld(p['data'][key],i,n,h)
            rng=np.random.default_rng(np.random.SeedSequence([p['data'][key],i,613]))
            if rng.random()<1/3:
                for _ in range(n+4):
                    if not w.bits.any():break
                    commands=w.candidates()[4];w.act(commands[int(teacher(w.bits,w.parents,commands).argmax())])
                for _ in range(int(rng.integers(n+3))):w.act(('inspect',int(rng.integers(n))))
            else:
                for _ in range(int(rng.integers(n+1))):
                    commands=[c for c in w.candidates()[4] if c[0]!='stop'];w.act(commands[int(rng.integers(len(commands)))])
            np.testing.assert_array_equal(np.stack(w.events),ds['events'][i,ds['present'][i]].numpy())
            np.testing.assert_array_equal(w.bits,ds['current'][i,:n].numpy())
            for a in w.actions:counts[a['kind']]+=1
            cost+=w.cost;observations+=len(w.events);actions+=w.action_cost
        content=hashlib.sha256()
        for field in sorted(ds):
            value=ds[field];content.update(field.encode());content.update(str(value.dtype).encode());content.update(json.dumps(list(value.shape)).encode());content.update(value.numpy().tobytes())
        result['splits'][name]={'tensor_content_sha256':content.hexdigest(),'snapshots':len(ds['current']),'collection_cost':cost,'observation_cost':.01*observations,'action_cost':actions,'observations':int(observations),'actions':counts,'all_public_histories_and_current_labels_exactly_reconstructed':True}
    result['total_unique_collection_cost']=sum(v['collection_cost'] for v in result['splits'].values());write(output,result)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--directory',required=True);a.add_argument('--output',required=True);args=a.parse_args();run(args.directory,args.output)
