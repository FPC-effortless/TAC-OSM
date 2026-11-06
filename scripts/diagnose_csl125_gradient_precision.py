"""No-model attribution of inherited scan/normalized-objective gradient errors.

Does not patch production source or relax the original tolerances. The diagnostic
FP64 scan is an independently named formula copy, not a qualified deployment.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts.audit_csl_scan_cancellation import (
    CASES, CHUNKS, OBJECTIVES, NAMES, fixture, objective, serial_torch, evaluate, difference,
)
from tac_osm.csl125_r2 import selective_affine_scan


def diagnostic_scan64(z, carry, rate, gate, chunk_size):
    r = rate.double().view(1, 1, *rate.shape)
    state, outputs = carry.double(), []
    for begin in range(0, z.shape[1], chunk_size):
        src = z[:,begin:begin+chunk_size].double()
        rg = r * gate[:,begin:begin+chunk_size].double().unsqueeze(-1)
        eta, beta = -torch.expm1(-rg), torch.exp(-rg)
        A, B = beta, eta.unsqueeze(-1)*src
        shift = 1
        while shift < src.shape[1]:
            old_A, old_B = A, B
            A = torch.cat((old_A[:,:shift],old_A[:,shift:]*old_A[:,:-shift]),dim=1)
            B = torch.cat((old_B[:,:shift],old_B[:,shift:]+old_A[:,shift:].unsqueeze(-1)*old_B[:,:-shift]),dim=1)
            shift *= 2
        result = A.unsqueeze(-1)*state.unsqueeze(1)+B
        outputs.append(result)
        state = result[:,-1]
    return torch.cat(outputs,dim=1), state


def run():
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError('this diagnostic is CPU only')
    parent = json.loads((ROOT/'research/csl_scan_audit_20261106/PROTOCOL.json').read_text())
    for path, expected in parent['source_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != expected:
            raise RuntimeError('frozen original source changed: '+path)
    protocol = json.loads((ROOT/'research/csl125_schema_runtime/GRADIENT_PRECISION_PROTOCOL.json').read_text())
    atol, rtol = protocol['tolerance']['gradient_atol'], protocol['tolerance']['gradient_rtol']
    arms = {
        'source_scan32_loss32': (selective_affine_scan, torch.float32),
        'source_scan32_loss64': (selective_affine_scan, torch.float64),
        'diagnostic_scan64_loss32': (diagnostic_scan64, torch.float32),
        'diagnostic_scan64_loss64': (diagnostic_scan64, torch.float64),
    }
    rows, arrays = [], {}
    for case in CASES:
        args = fixture(case)
        for chunk in CHUNKS:
            for loss in OBJECTIVES:
                ref = evaluate(serial_torch, args, chunk, loss, torch.float64)
                identity = f'{case}_{chunk}_{loss}'
                for name, grad in zip(NAMES, ref[2]):
                    arrays[identity+'_reference_'+name] = grad.numpy()
                for arm, (scan, loss_dtype) in arms.items():
                    leaves = [a.detach().clone().requires_grad_(True) for a in args]
                    out, _ = scan(*leaves, chunk)
                    scalar = objective(out.to(loss_dtype), loss)
                    grads = torch.autograd.grad(scalar, leaves)
                    checks = {n: difference(g, expected, atol, rtol)
                              for n,g,expected in zip(NAMES,grads,ref[2])}
                    failed = [n for n,v in checks.items() if v['status'] != 'PASS']
                    for name, g in zip(NAMES, grads):
                        arrays[identity+'_'+arm+'_'+name] = g.detach().numpy()
                    rows.append({'case': case, 'chunk': chunk, 'objective': loss,
                                 'arm': arm, 'failed_gradients': failed,
                                 'forward_max_abs': float((out.detach().double()-ref[0]).abs().max()),
                                 'checks': checks})
    assert len(rows) == 7*4*3*4
    summary = {'kind': 'operator_precision_attribution_only', 'torch': torch.__version__,
               'case_objective_chunk_rows_per_arm': 84, 'total_rows': len(rows),
               'failed_rows_by_arm': {a: sum(bool(r['failed_gradients']) for r in rows if r['arm']==a) for a in arms},
               'optimizer_steps': 0, 'models_constructed': 0, 'production_patch_authorized': False,
               'original_source_unchanged': True, 'objective_probe_rounding_included': True}
    outdir = ROOT/'ci-artifacts/csl125-schema'
    outdir.mkdir(parents=True,exist_ok=True)
    (outdir/'GRADIENT_PRECISION.json').write_text(json.dumps({'summary':summary,'rows':rows},indent=2)+'\n')
    np.savez_compressed(outdir/'GRADIENT_ARRAYS.npz',**arrays)
    print(json.dumps(summary,indent=2,sort_keys=True))


if __name__ == '__main__':
    run()
