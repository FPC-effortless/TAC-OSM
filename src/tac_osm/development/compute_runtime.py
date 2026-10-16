"""DEV-030 explicit FP32 execution backend; no task/optimization changes.

The classical table's rule-based public state updates stay on CPU. Its learned
head runs on the requested device. Other arms keep state and parameters on that
device. The adapter preserves the original state_dict names and public API.
"""
from __future__ import annotations

import os
import platform
import torch


def configure(device):
    device=torch.device(device)
    if device.type not in ('cpu','cuda'):
        raise ValueError('only CPU and CUDA are registered')
    torch.set_num_threads(1)
    if device.type=='cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; refusing silent CPU fallback')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True)
    return device


class DeviceBoundary:
    """Move PUBLIC tensors at explicit boundaries, including sensing receipts."""
    def __init__(self,model,device='cpu'):
        self.device=configure(device)
        self.model=model.to(self.device)
        self.state_device=torch.device('cpu') if model.arm=='event_table' else self.device

    def __getattr__(self,name):
        return getattr(self.model,name)

    def encode(self,*inputs):
        return self.model.encode(*(x.to(self.state_device) for x in inputs))

    def update(self,state,*inputs):
        return self.model.update(state.to(self.state_device),*(x.to(self.state_device) for x in inputs))

    def decode(self,state,queries):
        return self.model.decode(state.to(self.device),queries.to(self.device))

    def synchronize(self):
        if self.device.type=='cuda':
            torch.cuda.synchronize(self.device)

    def metadata(self):
        return dict(device=str(self.device),state_device=str(self.state_device),dtype='float32',
            torch_version=str(torch.__version__),python_version=platform.python_version(),
            cuda_version=torch.version.cuda,
            gpu_name=torch.cuda.get_device_name(self.device) if self.device.type=='cuda' else None,
            tf32=False,amp=False,compilation=False,batch_size_unchanged=True)
