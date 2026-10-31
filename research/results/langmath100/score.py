"""Score sealed frozen predictions; no model import, inference, optimizer or fitting."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def decode(raw):
 if b';' not in raw:return None
 raw=raw.split(b';',1)[0]
 if len(raw)<3 or raw[0] not in b'123456' or raw[1] not in b'123456':return None
 if len(raw[2:])!=(raw[0]-48)*(raw[1]-48) or any(v not in b'0123456789' for v in raw[2:]):return None
 return raw
def interval(values,clusters):
 names=sorted(set(clusters));means=np.array([np.mean([v for v,c in zip(values,clusters) if c==n]) for n in names])
 rng=np.random.default_rng(100001);draw=means[rng.integers(0,len(names),(5000,len(names)))].mean(1)
 # Cluster sign flips test a symmetric zero-effect null; small clusters limit resolution.
 if len(names)<=16:
  exact=np.array([abs(np.mean(means*np.array(s))) for s in itertools.product((-1,1),repeat=len(names))])
  p=float(np.mean(exact>=abs(means.mean())-1e-12))
 else:p=None
 return {'mean':float(np.mean(values)),'equal_cluster_mean':float(means.mean()),
  'cluster_CI95':np.quantile(draw,[.025,.975]).tolist(),'clusters':len(names),'paired_cluster_signflip_p':p}

def run(a):
 assert not a.output.exists(),'Preserve results'
 judge=read(a.data/'DEV_JUDGE.json');actors=read(a.data/'DEV_ACTOR.json')
 records=[];sets=[]
 for seal in sorted(a.predictions.rglob('PREDICTIONS_SEALED.json')):
  if 'smoke' in seal.parts:continue
  for entry in read(seal)['predictions']:
   p=seal.parent/entry['file'];assert sha(p)==entry['sha256']
   with np.load(p,allow_pickle=False) as z:
    assert list(z['ids'])==[r['id'] for r in actors]
    for i,r in enumerate(actors):
     g=judge[r['id']];target=bytes(g['target_bytes']);ok=bool(z['available'][i]);raw=bytes(z['generated'][i,:r['generation_limit']])
     row={'seed':entry['seed'],'condition':entry['condition'],'id':r['id'],'domain':r['domain'],'group':g['group'],'cluster':g['cluster_id'],'available':ok}
     if ok:
      lp=z['log_probs'][i].astype(float);assert np.isfinite(lp).all() and abs(np.logaddexp.reduce(lp))<1e-4
      assert raw[0]==lp.argmax()
      valid=True if r['domain']=='lexical' else decode(raw) is not None
      exact=raw==target if r['domain']=='lexical' else valid and decode(raw)==decode(target)
      row.update(nll=-float(lp[target[0]]),first_accuracy=int(raw[0]==target[0]),exact=int(exact),format_valid=int(valid),
       prefix_accuracy=[int(raw[:j]==target[:j]) for j in range(1,min(4,len(target))+1)])
     records.append(row)
   sets.append((entry['seed'],entry['condition']))
 seeds=sorted({s for s,_ in sets});conditions=sorted({c for _,c in sets})
 assert seeds==[9901,9902,9903] and len(conditions)==46
 assert len(sets)==len(set(sets))==138
 summary=[];lookup={(r['seed'],r['condition'],r['id']):r for r in records}
 cohorts=[('lexical','positive'),('lexical','negative'),('lexical','uniform'),('arc','unknown_rule')]
 for cond in conditions:
  for domain,group in cohorts:
   rr=[r for r in records if r['condition']==cond and r['domain']==domain and r['group']==group];ok=[r for r in rr if r['available']]
   summary.append({'condition':cond,'domain':domain,'group':group,'seed_case_pairs':len(rr),'unavailable':len(rr)-len(ok),
    'NLL':float(np.mean([r['nll'] for r in ok])) if ok else None,
    'exact_rate':float(np.mean([r['exact'] for r in ok])) if ok else None,
    'first_byte_accuracy':float(np.mean([r['first_accuracy'] for r in ok])) if ok else None,
    'format_rate':float(np.mean([r['format_valid'] for r in ok])) if ok else None,
    'per_seed':[{'seed':s,'exact':sum(r['exact'] for r in ok if r['seed']==s),'cases':sum(r['seed']==s for r in ok)} for s in seeds]})
 effects=[]
 for domain,group in cohorts:
  ids=[r['id'] for r in actors if r['domain']==domain and judge[r['id']]['group']==group]
  for cond in conditions:
   if cond=='reset_00':continue
   for metric in ('nll','exact'):
    vals=[];clusters=[];missing=[]
    for case in ids:
     paired=[(lookup[(s,cond,case)],lookup[(s,'reset_00',case)]) for s in seeds]
     if any(not x['available'] or not y['available'] for x,y in paired):missing.append(case);continue
     # Positive means intact carry better: ablated-minus-carry NLL, carry-minus-ablated exact.
     vals.append(float(np.mean([(x[metric]-y[metric])*(1 if metric=='nll' else -1) for x,y in paired])));clusters.append(judge[case]['cluster_id'])
    effects.append({'domain':domain,'group':group,'condition':cond,'metric':metric,
     'status':'MEASURED' if not missing else 'INCONCLUSIVE_FULL_COHORT',
     'cases':len(ids),'unavailable_cases':len(missing),'supported_subset':interval(vals,clusters) if vals else None})
 # Holm correction across the entire registered exploratory effect family.
 pp=[(i,e['supported_subset']['paired_cluster_signflip_p']) for i,e in enumerate(effects) if e['supported_subset'] and e['supported_subset']['paired_cluster_signflip_p'] is not None]
 running=0
 for rank,(i,p) in enumerate(sorted(pp,key=lambda x:x[1])):
  running=max(running,min(1.,p*(len(pp)-rank)));effects[i]['Holm_adjusted_p']=running
 interactions=[]
 for domain,group in cohorts:
  ids=[r['id'] for r in actors if r['domain']==domain and judge[r['id']]['group']==group]
  for j,k in itertools.combinations(range(5),2):
   vals=[];clusters=[]
   for case in ids:
    masks=[0,1<<j,1<<k,(1<<j)|(1<<k)];ms=[np.mean([lookup[(s,f'reset_{m:02d}',case)]['nll'] for s in seeds]) for m in masks]
    vals.append(ms[3]-ms[1]-ms[2]+ms[0]);clusters.append(judge[case]['cluster_id'])
   interactions.append({'domain':domain,'group':group,'removed_factors':[j,k],
    'metric':'NLL_double_reset_minus_single_resets_plus_intact','effect':interval(vals,clusters)})
 gates=[]
 for domain,group in [('lexical','positive'),('arc','unknown_rule')]:
  intact=next(x for x in summary if (x['condition'],x['domain'],x['group'])==('reset_00',domain,group))
  for cond in ('reset_08','reset_16','reset_24','off_fast','off_slow','off_both'):
   ee=[e for e in effects if (e['condition'],e['domain'],e['group'])==(cond,domain,group)]
   nll=next(e for e in ee if e['metric']=='nll');ex=next(e for e in ee if e['metric']=='exact')
   gate=all(p['exact']>0 for p in intact['per_seed']) and nll['supported_subset']['cluster_CI95'][0]>0 and ex['supported_subset']['cluster_CI95'][0]>0
   gates.append({'domain':domain,'condition':cond,'status':'PASS' if gate else 'FAIL',
    'nonzero_intact_exact_each_seed':all(p['exact']>0 for p in intact['per_seed']),
    'note':'Exploratory small exposed DEV; this gate alone cannot provide independent qualification'})
 result={'study_id':'LANGMATH100-001-002-008','qualification':'FROZEN_INTERNAL_EXPOSED_DEV_EXPLORATORY','optimizer_steps':0,
  'seeds':seeds,'conditions':len(conditions),'prediction_sets':len(sets),'seed_case_condition_rows':len(records),
  'summary':summary,'effects':effects,'interactions':interactions,'capability_gates':gates,
  'not_measured':['new trained A1-A6','new hidden-cue task','oracle-query solvability','teacher-forced full-target NLL','independent protected replication','energy'],
  'scope':'First-byte effects and exact greedy outcomes reported separately. Reset-after-acquisition can repopulate during query; off_* continuously clamps matrix/mass. Same-trained-model inference surgery is not matched architecture training.'}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'rows':len(records),'sets':len(sets),'gates':gates},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
