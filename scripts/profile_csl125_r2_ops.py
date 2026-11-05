"""No-fit CPU operator attribution; source-owned bounded reference only."""
import json,torch
from pathlib import Path
from torch.profiler import profile, ProfilerActivity
from tac_osm.csl125_r1 import CSL125R1
from tac_osm.csl125_r2 import CSL125R2
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1);torch.manual_seed(1253)
x=torch.randint(0,32768,(1,64))
summary={}
for name,cls in [('R1',CSL125R1),('R2',CSL125R2)]:
    m=cls().eval()
    with torch.inference_mode():
        for _ in range(2):m(x,depth=20,emit_token_logits=False)
        with profile(activities=[ProfilerActivity.CPU],record_shapes=False) as p:
            m(x,depth=20,emit_token_logits=False)
    entries=[]
    for ev in sorted(p.key_averages(),key=lambda v:v.self_cpu_time_total,reverse=True)[:20]:
        entries.append({'operator':ev.key,'self_cpu_ms':round(ev.self_cpu_time_total/1000,4),'calls':ev.count})
    summary[name]=entries
    del m
path=ROOT/'research/mathematical-engineering/csl125_r2/OPERATOR_PROFILE_NO_FIT.json'
path.write_text(json.dumps({'stage':'SINGLE_CPU_OPERATOR_PROFILE_NO_FIT','threads':1,'batch':1,'length':64,'depth':20,'no_vocab_head':True,'optimizer_steps':0,'operators':summary},indent=2)+'\n')
print(json.dumps({k:v[:8] for k,v in summary.items()},indent=2))
