"""Fresh frozen-policy semantic/clock probes; forced histories, no training."""
import argparse
from functools import lru_cache
import gzip
import json
from pathlib import Path
import time
import numpy as np
import torch
from data_belief_dev063 import BeliefWorld,relation
from data_belief_dev062 import PriceBelief
from run_belief_dev063 import make
from run_learned_stack_dev055 import write,digest
from audit_identity_dev0615 import compare
MODES=('identity','exact','shuffled','blurred','clock_shift','clock_scale')


@lru_cache(maxsize=32768)
def teacher(parents,error,inspection,posterior):
    b=PriceBelief(np.array(parents),error=error,inspection=inspection);b.p=np.array(posterior);return int(b.policy().argmax())


def restore(folder):
    r=json.loads((folder/'report.json').read_text());assert digest(folder/'agent.pt')==r['checkpoint_sha256'];weights=torch.load(folder/'agent.pt',weights_only=True);gate={k.removeprefix('gate.'):v.float() for k,v in weights.items() if k.startswith('gate.')};m=make(r['arm'],r['spec'],r['seed'],gate).double().eval();m.load_state_dict(weights);return r,m


@torch.no_grad()
def run(directory,output,details,smoke=False):
    torch.set_num_threads(1);root=Path(directory);assert json.loads((root/'summary.json').read_text())['status']==('SMOKE_ONLY' if smoke else 'COMPLETE');start=time.process_time();result={'status':'RUNNING','checkpoints':0,'cells':0,'queries':0,'by_arm':{},'by_seed':{},'identity_drift':{'values':0,'max_abs':0.},'recorded_drift':{'values':0,'max_abs':0.},'scope':'Forced recorded learned-policy histories. Counterfactual head outputs only, no changed actions/outcomes, weight tuning or deployed oracle.','details':str(details)};Path(details).parent.mkdir(parents=True,exist_ok=True);write(output,result)
    try:
        with gzip.open(details,'wt',encoding='utf-8') as stream:
            for path in sorted(root.glob('group-*/*/report.json')):
                if time.process_time()-start>1200:raise TimeoutError('Registered1200CPU-second intervention cap')
                r,m=restore(path.parent);cells=r['validation'] if smoke else json.loads((path.parent/'test.json').read_text())['cells'][:4];arm=result['by_arm'].setdefault(r['arm'],{'queries':0,'eligible_queries':0,'modes':{name:{'flips':0,'eligible_flips':0,'max_logit_change':0.} for name in MODES}});seedkey=str(r['seed'])+'/'+r['arm'];perseed={'queries':0,'eligible_queries':0,'flips':{name:0 for name in MODES}};result['by_seed'][seedkey]=perseed
                for c in cells:
                    worlds=[BeliefWorld(c['seed'],i,c['history'],c['error'],c['inspection']) for i in range(c['worlds'])];beliefs=[w.reference() for w in worlds];keys=torch.tensor(np.stack([w.identities for w in worlds]),dtype=torch.float64);state=m.initial(keys)
                    for i,w in enumerate(worlds):
                        e=torch.tensor(np.stack(w.events),dtype=torch.float64);take=m.gate(e[:,:6]).squeeze(-1)>=0;keep=e[take][-64:];state['events'][i,:len(keep)]=keep;state['present'][i,:len(keep)]=True
                    for step in range(c['horizon']):
                        if time.process_time()-start>1200:raise TimeoutError('Registered intervention CPU cap')
                        active=[i for i,row in enumerate(c['rows']) if step<len(row['actions'])]
                        if not active:break
                        payload=[w.candidates() for w in worlds];d=torch.tensor(np.stack([x[0] for x in payload]),dtype=torch.float64);prices=keys.new_tensor([[w.error,w.inspection] for w in worlds]);base=m.decide(state,d,prices);baseline=m.choice(base);public=base['posterior'].clone();donors=list(range(len(worlds)))
                        for i in active:
                            target=np.zeros((4,3));target[:,relation(worlds[i])]=beliefs[i].p;public[i]=torch.tensor(target.flatten());row=c['rows'][i];compare(base['posterior'][i].tolist(),row['posterior'][step],result['recorded_drift']);compare(public[i].tolist(),row['public_posterior'][step],result['recorded_drift']);assert int(payload[i][1][int(baseline[i])])==row['actions'][step]['action']
                        for graph in range(3):
                            indices=[i for i in active if relation(worlds[i])==graph]
                            for position,i in enumerate(indices):donors[i]=indices[(position+1)%len(indices)]
                        targets={'identity':base['posterior'],'exact':public,'shuffled':base['posterior'][donors],'blurred':.5*base['posterior']+.5/12};outputs={}
                        for mode,target in targets.items():
                            hook=m.belief.register_forward_hook(lambda module,args,value,target=target:target.log())
                            try:outputs[mode]=m.decide(state,d,prices)
                            finally:hook.remove()
                        compare(outputs['identity']['logits'].tolist(),base['logits'].tolist(),result['identity_drift']);assert torch.equal(m.choice(outputs['identity']),baseline)
                        for mode in ('clock_shift','clock_scale'):
                            changed={**state,'events':state['events'].clone()};t=changed['events'][:,:,42];changed['events'][:,:,42]=torch.where(state['present'],t+10000 if mode=='clock_shift' else t*1000,t);outputs[mode]=m.decide(changed,d,prices)
                            if r['spec']['clock_mode']=='event_gap':
                                compare(outputs[mode]['logits'].tolist(),base['logits'].tolist(),result['identity_drift']);assert torch.equal(m.choice(outputs[mode]),baseline),'Relative clock decision drift'
                        choices={name:m.choice(o) for name,o in outputs.items()}
                        for i in active:
                            w=worlds[i];j=donors[i];b=beliefs[i];parents=tuple(int(v) for v in w.parents);eligible=teacher(parents,w.error,w.inspection,tuple(b.p))!=teacher(parents,w.error,w.inspection,tuple(beliefs[j].p));arm['queries']+=1;perseed['queries']+=1;result['queries']+=1;arm['eligible_queries']+=eligible;perseed['eligible_queries']+=eligible;item={'arm':r['arm'],'seed':r['seed'],'history':c['history'],'index':i,'step':step,'donor_index':j,'baseline_index':int(baseline[i]),'decision_relevant_shuffle':bool(eligible),'choices':{},'max_logit_changes':{}}
                            for name,o in outputs.items():
                                index=int(choices[name][i]);flipped=index!=int(baseline[i]);change=float((o['logits'][i]-base['logits'][i]).abs().max());arm['modes'][name]['flips']+=flipped;arm['modes'][name]['eligible_flips']+=flipped and eligible;arm['modes'][name]['max_logit_change']=max(arm['modes'][name]['max_logit_change'],change);perseed['flips'][name]+=flipped;item['choices'][name]=index;item['max_logit_changes'][name]=change
                            stream.write(json.dumps(item)+'\n')
                        pending=[]
                        for i,w in enumerate(worlds):
                            if i not in active:pending.append([]);continue
                            action=c['rows'][i]['actions'][step]['action'];before=len(w.events);receipt=w.act(action);beliefs[i].update(action,receipt);pending.append(w.events[before:])
                        for tick in range(max(map(len,pending),default=0)):
                            present=torch.tensor([tick<len(e) for e in pending]);e=keys.new_tensor(np.stack([v[tick] if tick<len(v) else np.zeros(46) for v in pending]));state=m.observe(state,e,present)
                    result['cells']+=1
                result['checkpoints']+=1;write(output,result);print('INTERVENED',path.parent.name,r['seed'],result['queries'],flush=True)
        assert result['checkpoints']==(7 if smoke else 140) and result['cells']==(7 if smoke else 560);result['status']='COMPLETE'
    except Exception:
        import traceback
        result['status']='FAILED';result['failure']=traceback.format_exc();raise
    finally:
        result['CPU_seconds']=time.process_time()-start
        if Path(details).exists():result['details_sha256']=digest(details)
        write(output,result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True);p.add_argument('--details',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();run(a.directory,a.output,a.details,a.smoke)
