import torch,json,numpy as np
from pathlib import Path
import scripts.me092_train_world as R
R.configure('me092');R.integrity();R.frozen();torch.set_num_threads(1)
_,_,report,identified,_=R.load_original(Path('/tmp/me077-original-source'));queries,_=R.evaluation_origins(report,identified)
stats={}
with torch.no_grad():
 for seed in R.SEEDS:
  m=R.VisualWorld(identified);m.load_state_dict(torch.load(R.OUT/f'seed-{seed}/final.pt',map_location='cpu',weights_only=True)['weights']);m.double();groups={g:[] for g in ['moving','stationary']}
  for i in range(0,len(queries),24):
   x,a,y,valid,dt,images=R.batch(report,queries[i:i+24],8);raw=m(x.double(),a.double(),elapsed=dt.double(),images=images.double());raw={k:v[:,-1,0] for k,v in raw.items()};target=y[:,-1,0].double();current=x[:,-1].double();physical=m.state_scale
   moving=((target-current).abs()*physical).mean(-1)>=.025
   for j in range(len(x)):
    g='moving' if moving[j] else 'stationary';groups[g].append({'quiet_center_mae':float(((raw['current'][j]-target[j]).abs()*physical).mean()),'quiet_center_shift_from_observation':float(((raw['current'][j]-current[j]).abs()*physical).mean()),'quiet_scale_physical':float((raw['quiet_scale'][j]*physical).mean()),'moving_scale_physical':float((raw['moving_scale'][j]*physical).mean()),'mixture_probability_mean':float(raw['mixture_logit'][j].sigmoid().mean()),'gripper_mixture_probability':raw['mixture_logit'][j,[6,13]].sigmoid().tolist()})
  stats[str(seed)]={g:{k:np.mean([r[k] for r in rows],axis=0).tolist() for k in rows[0]} for g,rows in groups.items()}
result={'no_fit':True,'source':'Same exposed REPORT, five original Git-frozen models; descriptive output trace only, not ablation or new independent evidence','stats':stats}
(R.ROOT/'READOUT_TRACE.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
