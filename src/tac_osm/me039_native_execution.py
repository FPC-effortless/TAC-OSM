"""Optional explicit execution backend for the ME038 native architecture.

The core equations, parameter names and learned weights are retained. Scripted
backward may round differently: this is not a promise of bit-exact retraining.
"""
import torch
from tac_osm.me038_innovation_forecast import build_model as native_model


def compile_native_core(model):
    """Compile a newly constructed native core before creating its optimizer."""
    if model.arm!='innovation_tac':raise ValueError('only the native ME038 core is supported')
    if not isinstance(model.memory,torch.jit.ScriptModule):
        model.memory=torch.jit.script(model.memory)
    return model


def build_model(horizon,seed,backend='scripted'):
    if backend not in ('eager','scripted'):raise ValueError('unregistered execution backend')
    model=native_model('innovation_tac',horizon,seed)
    return compile_native_core(model) if backend=='scripted' else model
