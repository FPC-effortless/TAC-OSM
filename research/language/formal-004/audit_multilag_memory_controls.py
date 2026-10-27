"""Cold no-fit original/new frozen checkpoints on authentic Vim DEV.
Controls: zero memory, adjacent off-key, eight-step off-key and FIT prior.
No training, no TEST; developer exposed DEV reused explicitly.
"""
import argparse,hashlib,json,sys,math
from pathlib import Path
import torch,numpy as np
import torch.nn.functional as F
from train_multilag_real_fit import chars,sha,batches

def audit(a):
  archive=Path(a.archive);out=Path(a.out);source=Path(a.source)
  torch.set_num_threads(1)
  protocol=json.loads((archive/'PROTOCOL.json').read_text())
  new=json.loads((out/'RESULTS.json').read_text())
  original=json.loads((archive/'RESULTS.json').read_text())
  if len(new['runs'])!=10:raise ValueError('all ten complete checkpoint runs required')
  for name,hash_expected in protocol['source']['byte_sha256'].items():
    if sha(source/name)!=hash_expected:raise ValueError('source hash drift')
  docs=protocol['source']['doc_partitions'];fit=[chars((source/f).read_bytes()) for f in docs['fit']];dev=batches([chars((source/f).read_bytes()) for f in docs['dev']]);V=128
  counts=np.full(V,.5,dtype=np.float64)
  for d in fit:counts+=np.bincount(d,minlength=V)
  fitp=torch.tensor(counts/counts.sum(),dtype=torch.float32)
  sys.path.insert(0,str(archive));from parallel_tac_language_gate import ParallelTACLanguage
  results=[]
  for arm in ('original_64only','new_64and96'):
    for row in (original['runs'] if arm=='original_64only' else new['runs']):
      mode=row['mode'];seed=row['seed']
      file=(archive/row['checkpoint_file'] if arm=='original_64only' else out/row['checkpoint'])
      expected=(row['checkpoint_sha256'])
      if sha(file)!=expected:raise ValueError('checkpoint hash drift')
      ck=torch.load(file,weights_only=False,map_location='cpu')
      net=ParallelTACLanguage(decoder=75,position_mode='multiscale',gate_mode=mode).eval()
      net.load_state_dict(ck['state'] if arm=='original_64only' else ck['model'],strict=True)
      measure={'arm':arm,'seed':seed,'mode':mode,'checkpoint_sha256':expected,'lag':{}}
      with torch.no_grad():
       for lag in (64,96):
        totals={'correct':0.,'zero':0.,'prior':0.,'wrong1':0.,'wrong8':0.};n={'correct':0,'zero':0,'prior':0,'wrong1':0,'wrong8':0};acc=0
        for x,y in dev:
          _,trace=net(x,return_trace=True)
          h,t=net.historical_logits(x,trace,lag)
          totals['correct']+=F.cross_entropy(h.reshape(-1,V),t.reshape(-1),reduction='sum').item()
          n['correct']+=t.numel();acc+=(h.argmax(-1)==t).sum().item()
          z=torch.zeros_like(trace['aligned'][:,lag:,0,:,0]);zero=F.linear(net.hist_project(z),net.embedding.weight)
          totals['zero']+=F.cross_entropy(zero.reshape(-1,V),t.reshape(-1),reduction='sum').item();n['zero']+=t.numel()
          totals['prior']+=(-fitp.log()[t]).sum().item();n['prior']+=t.numel()
          for offset in (1,8):
            w,_=net.historical_logits(x,trace,lag+offset)
            match=t[:,offset:]
            totals['wrong'+str(offset)]+=F.cross_entropy(w.reshape(-1,V),match.reshape(-1),reduction='sum').item()
            n['wrong'+str(offset)]+=match.numel()
        rec={k+'_nll':totals[k]/n[k] for k in totals};rec['correct_accuracy']=acc/n['correct'];rec['samples']=n
        if arm=='original_64only' and abs(rec['correct_nll']-row['historical'][str(lag)]['nll'])>1e-5:raise ValueError('original loss mismatch')
        if arm=='new_64and96' and abs(rec['correct_nll']-row['dev']['hist'+str(lag)+'_nll'])>1e-5:raise ValueError('new loss mismatch')
        measure['lag'][str(lag)]=rec
      results.append(measure)
  agg={}
  for arm in ('original_64only','new_64and96'):
   agg[arm]={}
   for lag in ('64','96'):
    d=[r['lag'][lag] for r in results if r['arm']==arm]
    agg[arm][lag]={k:float(np.mean([r[k] for r in d])) for k in ('correct_nll','correct_accuracy','zero_nll','prior_nll','wrong1_nll','wrong8_nll')}
  doc={'scope':'original authentic Vim exposed DEV, no untouched TEST','source_manifest_sha256':sha(archive/'PROTOCOL.json'),'n_checkpoints_each_arm':10,'results':results,'aggregates':agg}
  (out/'MEMORY_CONTROLS.json').write_text(json.dumps(doc,indent=2,sort_keys=True))
  print('MEMORY_CONTROLS',json.dumps(agg,indent=2,sort_keys=True),flush=True)
if __name__=='__main__':
  ap=argparse.ArgumentParser();ap.add_argument('--archive',required=True);ap.add_argument('--source',required=True);ap.add_argument('--out',required=True);audit(ap.parse_args())
