#!/usr/bin/env python3
"""Diagnose failed TAOPILOT003 independent replay WITHOUT touching DEV."""
from __future__ import annotations
import hashlib, json, os, platform, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'research/mathematical-engineering/langmem086')]
import torch
from compact_tac import CompactDualClippedTAC086
from scripts.tao003_pilot import ARMS, model_hash, optimizer_for

def probe():
    base=ROOT/'research/evidence/taopilot003/seed1701'
    out={'study':'TAOPILOT003','phase':'NO_DEV_CROSS_HOST_REPLAY_DIAGNOSTIC',
         'protected_TEST':'NOT_ACCESSED','dev_read':False,'fit_steps':0,
         'torch_version':str(torch.__version__),'platform':platform.platform(),
         'processor':platform.processor(),'results':{}}
    torch.set_num_threads(2)
    for arm in ARMS:
        path=base/arm/'MODEL.pt'
        if not path.exists():raise FileNotFoundError(path)
        cp=torch.load(path,map_location='cpu',weights_only=True)
        p_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        receipt=json.loads((base/arm/'FIT_RECEIPT.json').read_text())
        if receipt['checkpoint_sha256']!=p_hash:raise ValueError('original checkpoint SHA drift')
        model=CompactDualClippedTAC086().eval();model.load_state_dict(cp['model'],strict=True)
        exact_weights=(model_hash(model)==cp['final_weight_hash'])
        o=optimizer_for(arm,model);o.load_state_dict(cp['optimizer'])
        state_steps=sorted({int(x['step']) for x in o.state.values()})
        with torch.no_grad():
            l,_=model(cp['cold_fixture'],owners=('cold-check',))
            m=model.train();lt,_=m(cp['cold_fixture'],owners=('cold-check',))
        original=cp['cold_logits']
        delta=(l-original).abs()
        diff=(l!=original)
        scale=original.abs().clamp_min(1.)
        max_relative=float((delta/scale).max())
        row={'exact_weight_sha':exact_weights,
             'optimizer_steps':state_steps,'cold_reference_shape':list(original.shape),
             'bitwise_logits':bool(torch.equal(l,original)),
             'max_abs_error':float(delta.max()), 'mean_abs_error':float(delta.mean()),
             'max_scaled_relative_error':max_relative,
             'nonidentical_logit_elements':int(diff.sum()),
             'identical_top1':bool(torch.equal(l.argmax(-1),original.argmax(-1))),
             'finite':bool(torch.isfinite(l).all()),
             'train_eval_logits_identical':bool(torch.equal(l,lt)),
             'small_preexisting_3e-5_abs_1e-4_rel_contract':bool(torch.allclose(l,original,atol=3e-5,rtol=1e-4))}
        out['results'][arm]=row
        print(json.dumps({'arm':arm,**row},sort_keys=True),flush=True)
    if not all(r['exact_weight_sha'] and r['optimizer_steps']==[32] and r['finite'] and r['identical_top1'] for r in out['results'].values()):
        out['hard_identity_gate']='FAIL'
    else:
        out['hard_identity_gate']='PASS_WEIGHTS_OPTIMIZER_TOP1_ONLY'
    dest=base/'INDEPENDENT_REPLAY_DIAG.json'
    if dest.exists():raise FileExistsError('do not rewrite first diagnosis')
    dest.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
    print('REPLAY_DIAG_SAVED',dest,flush=True)
if __name__=='__main__':probe()
