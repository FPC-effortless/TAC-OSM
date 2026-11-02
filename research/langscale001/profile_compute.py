"""No-fit CPU profiler coverage audit: partial kernel FLOPs, NOT full training FLOPs."""
import argparse,json,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import models

def run(out):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    rows=[];x=(torch.arange(512).reshape(4,128)*13+5)%256
    for row in models.configurations():
        torch.manual_seed(11201);m=models.build(row).train()
        before=models.preflight.parameter_digest(m)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True,record_shapes=True) as prof:
            y=m(x);y=y[0] if isinstance(y,tuple) else y
            y.square().mean().backward()
        events=prof.key_averages()
        counted=[{'operator':e.key,'calls':e.count,'flops':e.flops} for e in events if e.flops]
        zero=[{'operator':e.key,'calls':e.count} for e in events if not e.flops]
        assert models.preflight.parameter_digest(m)==before
        rows.append({'configuration':row,'counted_kernel_flops':sum(e['flops'] for e in counted),'counted_ops':counted,'unaccounted_ops':zero,
            'weights_unchanged':True,'optimizer_updates':0})
        print(row['scale'],row['arm'],rows[-1]['counted_kernel_flops'],flush=True)
    report={'status':'PARTIAL_FLOPS_COVERAGE_NOT_QUALIFIED_COMPUTE_PARITY','torch':torch.__version__,
        'batch':4,'length':128,'real_data_access':False,'optimizer_updates':0,
        'objective':'squared logits, no language targets; not the registered CE/AdamW step',
        'limits':['Uncounted elementary/transcendental/reduction/scan/backward operators remain','No optimizer step was executed or charged','Not a training or capability result','No speed claims: shared local host, not a controlled benchmark'], 'profiles':rows}
    out.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out)
