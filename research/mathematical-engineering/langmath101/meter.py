"""Actual-source dense arithmetic meter for exact LANGMATH101 owner lanes.

The native source OperatorLedger is reused unchanged. All observed operations,
including uncharged ones, are fingerprinted. No full FLOP or energy claim.
This module never fits a model automatically.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
    spec.loader.exec_module(mod);return mod


def _native_budget():
    # Always use the implementation included in the frozen source manifest.
    # An unrelated installed package must not silently override the bound copy.
    name='langmath101_original_objective_budget'
    if name not in sys.modules:
        _load(name,HERE/'src__tac_osm__research_objective_budget.py')
    return sys.modules[name]


def _owner():
    name='langmath101_owner_gate'
    if name not in sys.modules:
        _load(name,HERE/'owner_two_lane.py')
    return sys.modules[name]


def verify_common_bindings(model, common_state_digest, *, expected=None):
    """Verify the profiler and fit worker hash the same actual initial tensors."""
    native=_native_budget().state_dict_digest(model.state_dict())
    common=common_state_digest(model)
    if native!=common:
        raise ValueError('Profiler and fit-worker initialized tensor digests differ')
    if expected is not None and native!=expected:
        raise ValueError('Actual initial tensors differ from the frozen profile')
    return {'state_digest':native,'native_and_common_match':True}


def profile_without_update(model, loss_function, *, architecture, seed, input_shape):
    """Profile one actual source batch/backward while preserving every weight.

    loss_function(model) must use the preregistered authentic source batch.
    This produces no optimizer update. The caller seals original initialization
    bytes and binds the batch/schedule/recipe outside this local graph meter.
    """
    import torch
    B=_native_budget();G=_owner()
    before=B.state_dict_digest(model.state_dict())
    model.zero_grad(set_to_none=True)
    with B.OperatorLedger('forward_backward') as ledger:
        loss=loss_function(model)
        if not bool(torch.isfinite(loss)):
            raise ValueError('actual source loss is nonfinite')
        loss.backward()
    gradients={name:float(p.grad.detach().abs().sum()) if p.grad is not None else None
               for name,p in model.named_parameters() if p.requires_grad}
    if any(v is None or not torch.isfinite(torch.tensor(v)) for v in gradients.values()):
        raise ValueError('actual active parameter tensor lacks a finite gradient')
    after=B.state_dict_digest(model.state_dict())
    if before!=after:
        raise ValueError('zero-update source profile changed model weights/buffers')
    rows=ledger.rows();charge=G.counted_dense_units(rows)
    result={'architecture':architecture,'seed':seed,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'deployed_parameters':sum(p.numel() for p in model.parameters()),
            'parameter_tensor_count':len(gradients),'finite_gradient_tensor_count':len(gradients),
            'nonzero_gradient_tensor_count':sum(v>0 for v in gradients.values()),
            'gradient_L1':gradients,'input_shape':list(input_shape),
            'weights_before':before,'weights_after':after,'optimizer_steps':0,
            'operators':rows,'operator_sha256':charge['operators_sha256'],
            **charge,'loss':float(loss.detach())}
    model.zero_grad(set_to_none=True)
    return result


class ActualTrainingLedger:
    """Use around the whole authorized forward/backward/clip/optimizer step.

    Merely constructing this recorder does not train. Caller code decides whether
    training is allowed, performs the actual step and records source exposure.
    """
    def __enter__(self):
        self._ledger=_native_budget().OperatorLedger('actual_training_step')
        self._ledger.__enter__();return self

    def __exit__(self,*args):
        result=self._ledger.__exit__(*args)
        self.operators=self._ledger.rows()
        self.charge=_owner().counted_dense_units(self.operators)
        return result

    def receipt(self):
        if not hasattr(self,'charge'):
            raise ValueError('actual step ledger is not complete')
        return {'operator_sha256':self.charge['operators_sha256'],
                'counted_dense_units':self.charge['counted_dense_units']}
