"""Fresh-process replay of original initialization and exported state, no fits."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch

p=argparse.ArgumentParser();p.add_argument('--source-model',type=Path,required=True)
p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();b=a.source_model.read_bytes()
assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()=='1636ec81fe0a23e631f7f4da67914d365883d3ef'
spec=importlib.util.spec_from_file_location('replay_original098',a.source_model)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
torch.set_num_threads(1);rows=[]
with torch.inference_mode():
 for seed in (98001,98002,98003):
  r=torch.load(a.results/f'INITIAL_AND_REPLAY_{seed}.pt',weights_only=True)
  m=mod.ContextualAssociativeTAC098().eval();m.load_state_dict(r['model_state_dict'])
  h=hashlib.sha256()
  for k,v in sorted(m.state_dict().items()):h.update(k.encode());h.update(v.numpy().tobytes())
  assert h.hexdigest()==r['weight_sha256']
  state=mod.State098(r['owners'],r['saved_state']['clock'],r['saved_state']['tensors'])
  out,_=m(r['query'],state=state,owners=r['owners'])
  assert torch.equal(out,r['expected_logits'])
  _,reconstructed=m(r['history'],owners=r['owners'])
  assert all(torch.equal(v,reconstructed.tensors[k]) for k,v in state.tensors.items())
  rows.append({'seed':seed,'weight_sha256':h.hexdigest(),'exact_logits':True,'exact_acquired_state':True})
assert not a.output.exists()
a.output.write_text(json.dumps({'status':'PASS_FRESH_PROCESS_REPLAY','rows':rows,'optimizer_steps':0},indent=2)+'\n')
print(a.output.read_text())
