"""Freeze DEV-060 reference outcomes, never fit a learned policy."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from tac_osm.development.uncertain_world import UncertainWorld,COMMANDS
from run_learned_stack_dev055 import write,digest,git
PROTOCOL=ROOT/'research/dev060/protocol-foundations.json'


def calibration(probabilities,labels,bins=10):
    p=np.asarray(probabilities,float);y=np.asarray(labels,float)
    if not len(p):return {'count':0,'brier':None,'ece':None,'bins':[]}
    if p.shape!=y.shape or not np.isfinite(p).all() or not np.isfinite(y).all() or (np.any((p<0)|(p>1)) or np.any((y!=0)&(y!=1))):
        raise ValueError('aligned finite probabilities in [0,1] and binary observed labels required')
    values=[];ece=0.;assigned=np.minimum((p*bins).astype(int),bins-1)
    for index in range(bins):
        lower=index/bins;mask=assigned==index
        count=int(mask.sum())
        if not count:continue
        mean=float(p[mask].mean());rate=float(y[mask].mean());ece+=count*abs(mean-rate)/len(p)
        z=1.959963984540054;den=1+z*z/count;center=(rate+z*z/(2*count))/den
        radius=z*np.sqrt(rate*(1-rate)/count+z*z/(4*count*count))/den
        values.append({'lower':float(lower),'count':count,'mean_probability':mean,'empirical_frequency':rate,'wilson95':[float(center-radius),float(center+radius)]})
    return {'count':len(p),'brier':float(((p-y)**2).mean()),'ece':ece,'bins':values,
            'limitation':'Visited decisions are correlated within worlds. Bin intervals are descriptive conditional counts, not clustered deployment guarantees.'}


def evaluate(seed=6062003,count=256):
    result={'source':git('rev-parse','HEAD'),'protocol_sha256':digest(PROTOCOL),'scope':'References only; no neural policy results. Public planner exact belief enumeration, bounded depth2; gap includes planning approximation.','arms':{}}
    for arm in ('public_belief','privileged','latest_shortcut'):
        rows=[];probabilities=[];labels=[]
        for i in range(count):
            w=UncertainWorld(seed,i);b=w.reference();lookup={tuple(k):j for j,k in enumerate(w.identities.tolist())}
            for _ in range(8):
                probabilities.append(float(b.p[0]));labels.append(float(not w.bits.any()))
                if arm=='public_belief':choice=b.choose(2)
                elif arm=='privileged':
                    candidates=[j for j,(kind,e) in enumerate(COMMANDS) if kind=='repair' and w.bits[e] and (w.parents[e]<0 or not w.bits[w.parents[e]])]
                    choice=candidates[0] if candidates else 0
                else:
                    latest={}
                    for r in w.events:
                        if r['source'] in ('initial','inspect','repair'):latest[lookup[tuple(r['subject'])]]=r['value']
                    candidates=[j for j,(kind,e) in enumerate(COMMANDS) if kind=='repair' and latest.get(e,0)]
                    choice=candidates[0] if candidates else 0
                receipt=w.act(choice);b.update(choice,receipt)
                if w.stopped:break
            rows.append(w.finish())
        result['arms'][arm]={'summary':{'worlds':count,'completion':float(np.mean([r['completion'] for r in rows])),
            'physical_repair':float(np.mean([r['physical_repair'] for r in rows])),
            'faults':sum(r['faults'] for r in rows),'premature_stop':sum(r['stopped'] and r['faults']>0 for r in rows),
            'missing_stop':sum(not r['stopped'] for r in rows),'mean_cost':float(np.mean([r['cost'] for r in rows])),
            'actions_per_world':float(np.mean([len(r['actions']) for r in rows])),
            'inspections':sum(a['kind']=='inspect' for r in rows for a in r['actions']),
            'cost_components':{k:float(np.mean([r['costs'][k] for r in rows])) for k in ('observations','inspections','repairs','terminal')},
            'public_completion_belief_calibration':calibration(probabilities,labels)},'rows':rows}
    private=result['arms']['privileged']['summary']
    private['interventional_filter_diagnostic']=private.pop('public_completion_belief_calibration')
    private['calibration_limitation']='Private action selection depends on hidden state; filter ignores that extra action-selection information. This diagnostic is not an exact conditional posterior calibration claim.'
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',required=True);args=ap.parse_args()
    result=evaluate();write(args.output,result)
    for arm,x in result['arms'].items():print(arm,json.dumps(x['summary']),flush=True)
