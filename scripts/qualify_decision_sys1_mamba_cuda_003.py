#!/usr/bin/env python3
"""Explicit authorized-CUDA no-fit check; NEVER installs/falls back/trains.

A PASS is only bounded FP32 output/gradient/cache agreement at the pinned module.
It is not transitive dependency, performance, source-data or training admission.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import torch
from tac_osm.decision_sys1_mamba2_003 import Mamba2RiskModel, official_mamba_class


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--authorized-cuda', action='store_true',
                        help='Required acknowledgment of separately approved compute')
    args = parser.parse_args()
    result = {'study':'DECISION-SYS1-003', 'status':'BLOCKED', 'optimizer_steps':0,
              'real_model_predictions':0, 'training_admitted':False,
              'official_transitive_dependencies_qualified':False}
    code = 2
    try:
        if not args.authorized_cuda:
            raise RuntimeError('no explicit authorized CUDA execution flag')
        official_mamba_class()  # Reject absent CUDA/package/wrong source before work.
        torch.manual_seed(6101)
        ref = Mamba2RiskModel('reference').cuda().float()
        off = Mamba2RiskModel('official').cuda().float()
        off.load_state_dict(ref.state_dict(), strict=True)
        result['devices'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        result['torch_version'] = torch.__version__
        result['source_files'] = {}
        for module in ('mamba_ssm.modules.mamba2', 'mamba_ssm.ops.triton.ssd_combined',
                       'mamba_ssm.ops.triton.layernorm_gated', 'mamba_ssm.ops.triton.selective_state_update'):
            import importlib
            path = Path(inspect.getfile(importlib.import_module(module)))
            result['source_files'][module] = hashlib.sha256(path.read_bytes()).hexdigest()
        # Record bounded FP32 differences; do not widen thresholds after failure.
        x = torch.randn(2, 33, 212, device='cuda', requires_grad=True)
        xx = x.detach().clone().requires_grad_()
        y, state_ref = ref(x)
        yy, _ = off(xx)
        torch.testing.assert_close(y, yy, rtol=5e-4, atol=5e-5)
        y.square().mean().backward()
        yy.square().mean().backward()
        errors = {}
        for (name,p), (oname,q) in zip(ref.named_parameters(), off.named_parameters()):
            if name != oname or p.grad is None or q.grad is None:
                raise RuntimeError('parameter/gradient correspondence missing')
            torch.testing.assert_close(p.grad, q.grad, rtol=2e-3, atol=1e-5)
            errors[name] = float((p.grad-q.grad).abs().max())
        torch.testing.assert_close(x.grad, xx.grad, rtol=2e-3, atol=1e-5)
        with torch.no_grad():
            state = off.initial(2)
            values = []
            for t in range(x.shape[1]):
                value, state = off.step(x[:,t], state)
                values.append(value)
            streamed = torch.stack(values,1)
            torch.testing.assert_close(yy, streamed, rtol=5e-4, atol=5e-5)
            for a,b in zip(state_ref,state):
                torch.testing.assert_close(a.conv,b.conv,rtol=5e-4,atol=5e-5)
                torch.testing.assert_close(a.ssm,b.ssm,rtol=5e-4,atol=5e-5)
        result.update(status='PASS_BOUNDED_NUMERICAL_ONLY',
                      full_output_max_abs=float((y-yy).detach().abs().max()),
                      full_cache_max_abs=float((yy-streamed).detach().abs().max()),
                      parameter_gradient_max_abs=errors)
        code = 0
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
