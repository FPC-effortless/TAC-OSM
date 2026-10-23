"""Report every original numeric-contract failure; never relax its tolerance."""
import json
import os
from pathlib import Path
import platform
import numpy as np
import torch
from scripts.diagkey004_aligned_recall import ARMS,SEEDS,OUT,verify_fit,model_for,evaluate


def main():
    torch.set_num_threads(1);rows=[]
    for arm in ARMS:
        for seed in SEEDS:
            fit=verify_fit(arm,seed)
            for cp in fit['checkpoints']:
                model=model_for(arm)
                model.load_state_dict(torch.load(cp['path'],weights_only=False,map_location='cpu')['weights'])
                actual=evaluate(model,seed,'TEST')
                original=json.loads((OUT/f'{arm}-s{seed}'/f"TEST-step-{cp['step']}.json").read_text())
                a=np.asarray(original['logits']);b=np.asarray(actual['logits']);error=np.abs(a-b)
                close=np.isclose(a,b,atol=1e-5,rtol=1e-5)
                rows.append({'arm':arm,'seed':seed,'step':cp['step'],
                    'checkpoint_sha256':cp['sha256'],'logit_count':int(a.size),
                    'logits_outside_registered_tolerance':int((~close).sum()),
                    'maximum_absolute_logit_error':float(error.max()),
                    'different_classifications':sum(x!=y for x,y in zip(original['correct_bits'],actual['correct_bits'])),
                    'absolute_NLL_error':abs(original['NLL']-actual['NLL'])})
    strict=all(r['logits_outside_registered_tolerance']==0 and r['different_classifications']==0 and r['absolute_NLL_error']<=1e-5 for r in rows)
    report={'status':'PASS_REGISTERED_NUMERICS' if strict else 'FAIL_REGISTERED_NUMERICS_PRESERVED',
        'registered_atol':1e-5,'registered_rtol':1e-5,'registered_NLL_absolute_tolerance':1e-5,
        'original_checkpoint_hashes_and_live_receipts_verified':True,'new_optimizer_updates':0,
        'exact_all_classifications':all(r['different_classifications']==0 for r in rows),
        'all_NLL_within_original_contract':all(r['absolute_NLL_error']<=1e-5 for r in rows),
        'runtime':{'cpu_capability':torch.backends.cpu.get_cpu_capability(),'platform':platform.platform(),
                   'torch':torch.__version__,'numpy':np.__version__,
                   'MKL_ENABLE_INSTRUCTIONS':os.environ.get('MKL_ENABLE_INSTRUCTIONS'),
                   'ATEN_CPU_CAPABILITY':os.environ.get('ATEN_CPU_CAPABILITY')},'rows':rows}
    path=OUT/'CROSS_CPU_AUDIT.json';path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':report['status'],'exact_classifications':report['exact_all_classifications'],
        'NLL_within_contract':report['all_NLL_within_original_contract'],
        'logits_outside_tolerance':sum(r['logits_outside_registered_tolerance'] for r in rows),
        'maximum_absolute_error':max(r['maximum_absolute_logit_error'] for r in rows)},indent=2))


if __name__=='__main__':main()
