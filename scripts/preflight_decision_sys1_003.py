#!/usr/bin/env python3
"""No-fit source/count/parity receipt. Does not admit training or benchmark speed."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import platform
from pathlib import Path
import torch
from tac_osm.decision_sys1_001 import NativeRiskStateModel
from tac_osm.decision_sys1_baselines_002 import GRURiskModel, TransformerRiskModel
from tac_osm.decision_sys1_mamba2_003 import Mamba2RiskModel, count_and_state_receipt


def receipt() -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(6101)
    model = Mamba2RiskModel('reference').double()
    second = copy.deepcopy(model)
    x = torch.randn(2, 17, 212, dtype=torch.float64, requires_grad=True)
    xx = x.detach().clone().requires_grad_()
    full, full_state = model(x)
    state = second.initial(2)
    values = []
    for t in range(x.shape[1]):
        value, state = second.step(xx[:, t], state)
        values.append(value)
    streamed = torch.stack(values, 1)
    (full.square().mean() + sum(s.ssm.square().mean() for s in full_state)).backward()
    (streamed.square().mean() + sum(s.ssm.square().mean() for s in state)).backward()
    gradient_max = 0.0
    tensors = 0
    for (name, p), (other_name, q) in zip(model.named_parameters(), second.named_parameters()):
        if name != other_name or p.grad is None or q.grad is None:
            raise RuntimeError('missing/mismatched actual parameter gradient')
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-9, atol=1e-11)
        gradient_max = max(gradient_max, float((p.grad-q.grad).abs().max()))
        tensors += 1
    torch.testing.assert_close(full, streamed, rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(x.grad, xx.grad, rtol=1e-9, atol=1e-11)
    models = {'native': NativeRiskStateModel(), 'gru': GRURiskModel(),
              'transformer': TransformerRiskModel(), 'mamba2_reference': model}
    counts = {name: sum(p.numel() for p in m.parameters()) for name, m in models.items()}
    source_files = [Path(__file__).relative_to(Path.cwd()),
                    Path('src/tac_osm/decision_sys1_mamba2_003.py'),
                    Path('src/tac_osm/decision_sys1_ssd_reference_003.py')]
    sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    result = count_and_state_receipt()
    result.update(study='DECISION-SYS1-003', torch_version=torch.__version__,
                  python=platform.python_version(), threads=torch.get_num_threads(),
                  counts=counts, relative_max_count_gap=(max(counts.values())-min(counts.values()))/min(counts.values()),
                  fp32_state_bytes_per_entity=4*result['state_scalars_per_entity'],
                  fixture={'seed':6101, 'shape':[2,17,212], 'dtype':'float64',
                           'full_step_output_max_abs':float((full-streamed).detach().abs().max()),
                           'full_step_state_max_abs':max(float((a.ssm-b.ssm).detach().abs().max()) for a,b in zip(full_state,state)),
                           'parameter_gradient_max_abs':gradient_max,
                           'parameter_gradient_tensors_checked':tensors,
                           'input_gradient_max_abs':float((x.grad-xx.grad).abs().max())},
                  source_sha256=sources, training_admitted=False, real_model_predictions=0,
                  numerical_scope='CPU reference paths only; not upstream CUDA/IEEE/whole-model proof',
                  official_transitive_kernel_qualification='NOT_RUN')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = receipt()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
