"""Fresh fixed-query semantic shuffle diagnostics; no training or promotion."""
import argparse
import json
from pathlib import Path
import torch
from data_stable_semantic_dev067 import batch
from intervene_stable_semantic_dev067 import restore
from run_learned_stack_dev055 import write


@torch.no_grad()
def run(directory, output):
    torch.set_num_threads(1);root=Path(directory)
    assert json.loads((root/'summary.json').read_text())['status']=='COMPLETE'
    records=torch.load(root/'test-fixed.pt',weights_only=True);result=dict(status='RUNNING',by_seed={},checkpoints=0)
    for path in sorted(root.glob('group-*/*/report.json')):
        report,model=restore(path.parent);chunks=[]
        gate=__import__('copy').deepcopy(model.gate).float()
        for start in range(0,len(records),64):
            x=batch(records,range(start,min(start+64,len(records))),gate)
            x={n:v.double() if v.is_floating_point() else v for n,v in x.items()}
            o=model(x['events'],x['present'],x['keys'],x['candidates'],x['prices'])
            chunks.append((x,o))
        predicted=torch.cat([o['posterior'] for _,o in chunks]);public=torch.cat([x['posterior'] for x,_ in chunks])
        graphs=public.reshape(-1,4,3).sum(1).argmax(-1);donors=list(range(len(records)))
        for graph in range(3):
            indices=(graphs==graph).nonzero(as_tuple=True)[0].tolist()
            for position,index in enumerate(indices):donors[index]=indices[(position+1)%len(indices)]
        shuffled=predicted[donors];distinct=(public-public[donors]).abs().sum(-1)>1e-6
        item=dict(queries=len(records),identity_flips=0,exact_flips=0,shuffle_flips=0,
                  public_posterior_distinct_donors=int(distinct.sum()),distinct_shuffle_flips=0)
        offset=0
        for x,original in chunks:
            n=len(x['keys']);base=model.choice(original)
            for name,q in [('identity',predicted[offset:offset+n]),('exact',public[offset:offset+n]),('shuffle',shuffled[offset:offset+n])]:
                changed=model.act_from_belief(q,x['keys'],x['candidates'],x['prices']);flips=model.choice(changed)!=base
                item[name+'_flips']+=int(flips.sum())
                if name=='identity':
                    torch.testing.assert_close(changed['logits'],original['logits'],atol=0,rtol=0)
                    assert not flips.any()
                if name=='shuffle':item['distinct_shuffle_flips']+=int((flips&distinct[offset:offset+n]).sum())
            offset+=n
        result['by_seed'][report['arm']+'/'+str(report['seed'])]=item;result['checkpoints']+=1
    assert result['checkpoints']==120
    result.update(status='COMPLETE',scope='Forced fixed-query head outputs only, public-same-graph cyclic donor. Distinct posterior is not necessarily a different optimal action. No alternative action executed, and no maximize-flips/selection objective.')
    write(output,result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.directory,a.output)
