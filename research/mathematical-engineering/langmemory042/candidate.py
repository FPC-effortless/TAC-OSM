"""Only editable experiment surface: two frozen position-feature controls.
No changed source bytes, ordinal keys, basis denominator or learned weights.
"""
from contextlib import contextmanager
import torch

MODES=('native','cap_log','cap_all')


def transform(features,mode):
    if mode not in MODES or features.shape[-1]!=7 or not features.is_floating_point():
        raise ValueError('registered mode and seven floating positional features required')
    if mode=='native': return features
    # Same dtype/kernel as original phi_position(log1p(t)), not Python rounding.
    t=features.new_tensor(127.)
    cap=torch.log1p(t)
    if mode=='cap_log':
        return torch.cat((features[...,:1].clamp_max(cap),features[...,1:]),-1)
    at127=torch.stack((torch.log1p(t),torch.sin(t/16),torch.cos(t/16),torch.sin(t/64),torch.cos(t/64),torch.sin(t/256),torch.cos(t/256)))
    return torch.where((features[...,:1]>cap),at127,features)


@contextmanager
def controlled_position(model,mode):
    if mode not in MODES: raise ValueError('unregistered trial')
    if mode=='native':
        yield;return
    handle=model.pos.register_forward_pre_hook(lambda module,args:(transform(args[0],mode),))
    try: yield
    finally: handle.remove()
