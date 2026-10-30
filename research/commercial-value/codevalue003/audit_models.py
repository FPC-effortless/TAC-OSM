#!/usr/bin/env python3
"""Constructed-only no-fit audit of inherited model implementations.

No publisher reads, optimizer steps, GPU allocation, or capability evidence.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location(
    'codevalue002_train_audit', HERE.parent / 'codevalue002/train.py')
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)


def run():
    torch.set_num_threads(2)
    torch.manual_seed(930003)
    tokens = torch.randint(0, 256, (2, 32))
    poisoned = tokens.clone()
    poisoned[:, 16:] = (poisoned[:, 16:] + 127) % 256
    result = {
        'study': 'CODEVALUE003', 'scope': 'CONSTRUCTED_UNTRAINED_MODELS_ONLY',
        'training_updates': 0, 'publisher_bytes_read': 0, 'gpu_minutes': 0,
        'capability_evidence': False, 'torch_version': str(torch.__version__),
        'models': {},
    }
    for arm in parent.ARMS:
        model = parent.make(arm).eval()
        with torch.no_grad():
            first = model(tokens)
            changed = model(poisoned)
            repeated = model(tokens)
        delta = float((first[:, :16] - changed[:, :16]).abs().max())
        deterministic_delta = float((first - repeated).abs().max())
        if delta > 1e-6 or deterministic_delta > 1e-6:
            raise ValueError(f'{arm}: prefix causality/determinism failure')
        if first.shape != (2, 32, 256) or not torch.isfinite(first).all():
            raise ValueError(f'{arm}: logits shape/finite failure')
        model.train()
        loss = torch.nn.functional.cross_entropy(
            model(tokens[:, :-1]).reshape(-1, 256), tokens[:, 1:].reshape(-1))
        loss.backward()  # gradient connectivity audit; no optimizer exists
        disconnected = [n for n, p in model.named_parameters() if p.grad is None]
        nonfinite = [n for n, p in model.named_parameters()
                     if p.grad is not None and not torch.isfinite(p.grad).all()]
        zero = [n for n, p in model.named_parameters()
                if p.grad is not None and torch.count_nonzero(p.grad) == 0]
        if disconnected or nonfinite:
            raise ValueError(f'{arm}: disconnected/nonfinite gradients')
        result['models'][arm] = {
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'deployed_parameters': sum(p.numel() for p in model.parameters()),
            'future_suffix_poison_prefix_max_abs_difference': delta,
            'repeat_prediction_max_abs_difference': deterministic_delta,
            'disconnected_gradients': disconnected,
            'nonfinite_gradients': nonfinite,
            'zero_gradient_tensors_on_this_fixture': zero,
            'status': 'PASS_CONSTRUCTED_ONLY',
        }
    files = [HERE.parent / 'codevalue002/train.py']
    files.extend(ROOT / 'research/mathematical-engineering' / path for path in (
        'langscale070/scaled_recurrence.py', 'context026/context_transition.py',
        'langlayer038/recurrence_only.py', 'langhead094/model094.py',
        'langhead093/model093.py', 'langmem086/compact_tac.py'))
    result['source_sha256'] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return result


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, allow_nan=False))
