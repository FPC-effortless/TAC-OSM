"""Actual variant adapter smoke and contamination falsifier; constructed inputs."""
import argparse,json,sys
from dataclasses import replace
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import models
from probe_adapters import adapter_for
from tac_osm.research_probe import probe_dependencies,probe_entity_isolation,probe_memory

def run(out):
 torch.set_num_threads(2);rows=[]
 for row in models.configurations():
  if row['arm'] not in ('tac_memory_scaled','tac_head_to_state'):continue
  torch.manual_seed(11201);m=models.build(row).double().eval()
  sample={'history':torch.tensor([[1,2,3,4],[9,8,7,6]]),'query':torch.tensor([[5],[10]]),'owners':('a','b')}
  a=adapter_for(row['arm'],m);a.validate(m,sample);s=a.start(m,sample)
  with torch.no_grad():
   for t in range(4):s=a.advance(m,s,sample,t)
   y=a.forecast(m,s,sample)['next_byte_logits'];expected=m(torch.cat((sample['history'],sample['query']),1))[0][:,-1:]
   assert torch.allclose(y,expected,atol=1e-8,rtol=1e-7)
  isolation=probe_entity_isolation(m,sample,a,tolerance=1e-8);assert isolation['status']=='PASS'
  def wrong(m,s,x):return {'next_byte_logits':a.forecast(m,s,x)['next_byte_logits']+x['query'].double().mean()/256}
  bad=probe_entity_isolation(m,sample,replace(a,forecast=wrong),tolerance=1e-8);assert bad['status']=='FAIL'
  dep=probe_dependencies(m,sample,a)
  assert all(v['connected'] and v['finite'] for head in dep['heads'].values() for v in head.values())
  memory=probe_memory(m,sample,a);assert memory['query_does_not_write_state']
  rows.append({'scale':row['scale'],'arm':row['arm'],'actual_source_adapter':a.name,'source_forecast_agrees':True,
    'entity_isolation':isolation,'contamination_negative_control':bad,'dependencies':dep,'memory':memory})
 out.write_text(json.dumps({'status':'PASS_CONSTRUCTED_ADAPTER_CHECKS','optimizer_updates':0,'dataset_access':False,'cases':rows},indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out)
