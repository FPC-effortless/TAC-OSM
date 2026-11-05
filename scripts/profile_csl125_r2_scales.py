"""No-fit multiscale CSL125R2 parameter and CPU screening; independent models.

Tokenizer vocabulary sizes are capacities, no fitted BPE files. These models
are NOT depth slices of the 123.8M checkpoint, nor quality comparisons.
"""
import json,statistics,time,torch
from pathlib import Path
from tac_osm.csl125_r2 import CSL125R2,CSL125R2Config
ROOT=Path(__file__).resolve().parents[1]


def configs():
    return {
      '1m':CSL125R2Config(vocab_size=4096,width=112,layers=8,ff_width=128,state_width=16,
          role_count=4,query_heads=4,kv_heads=1,attention_layers=(4,8),rungs=(2,4,8),attention_window=64),
      '10m':CSL125R2Config(vocab_size=8192,width=256,layers=12,ff_width=576,state_width=80,
          role_count=8,query_heads=8,kv_heads=2,attention_layers=(4,8,12),rungs=(2,6,12),attention_window=64),
      '125m':CSL125R2Config(),
    }


def profile():
    torch.set_num_threads(1);torch.manual_seed(11250)
    out={'stage':'NO_FIT_SIZE_LADDER_CPU_REFERENCE','device':'cpu','threads':1,'batch':1,'sequence':32,
         'optimizer_steps':0,'corpus':'NONE','tokenizers_fitted':False,'independent_parameterizations':True,
         'controls':{},'limitations':'not trained, GPU, energy or cross-scale intelligence evidence'}
    for name,cfg in configs().items():
        with torch.device('meta'):meta=CSL125R2(cfg,device='meta')
        n=meta.parameter_count()
        del meta
        torch.manual_seed(1234)
        model=CSL125R2(cfg).eval()
        assert n==model.parameter_count()
        x=torch.randint(0,cfg.vocab_size,(1,32))
        row={'parameters':n,'vocab_size':cfg.vocab_size,'width':cfg.width,'layers':cfg.layers,
             'rungs':{},'ff_width':cfg.ff_width,'state_width':cfg.state_width,'attention_layers':cfg.attention_layers}
        with torch.inference_mode():
            for depth in cfg.rungs:
                for _ in range(2):model(x,depth=depth,emit_token_logits=False)
                samples=[]
                for _ in range(3):
                    t=time.perf_counter();o,s=model(x,depth=depth,emit_token_logits=False)
                    samples.append((time.perf_counter()-t)*1000)
                row['rungs'][str(depth)]={'ms_samples':samples,'median_ms':statistics.median(samples),
                   'prefill_tok_per_s':32000/statistics.median(samples)}
        out['controls'][name]=row
        del model
    p=ROOT/'research/mathematical-engineering/csl125_r2/SCALING_NO_FIT.json'
    p.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:{'params':x['parameters'],'prefill':{r:round(v['prefill_tok_per_s']) for r,v in x['rungs'].items()}}
                      for k,x in out['controls'].items()},indent=2))
    return out

if __name__=='__main__':profile()
