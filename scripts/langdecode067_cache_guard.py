"""LANGDECODE067: fail-closed identity/version guard on the existing cached pos.

This is a separate inference-only candidate, not a change to original
TAC 043, cached 064 or a trained model. Defends against ordinary accidental
cache replacement and in-place PyTorch API edits in the 7-scalar cache.

PyTorch .data / external raw-storage writes can bypass Tensor._version;
this is explicitly NOT adversarial tamper resistance or universal IEEE proof.
"""
import torch

from scripts.langdecode064_cached_cap import CachedCapPositionTAC064


class VersionGuardedCapTAC067(CachedCapPositionTAC064):
    """Same actual math as 064; O(1) local cache identity/version check."""

    def __init__(self, model, position_mode="native"):
        super().__init__(model, position_mode)
        ref=self._cap_all_ref
        if position_mode=="cap_all":
            if not torch.is_tensor(ref):
                raise ValueError("cache is not a tensor")
            if not hasattr(ref,"_version"):
                raise ValueError("PyTorch tensor version metadata unavailable")
            self._cache_integrity=(id(ref),ref._version,tuple(ref.shape),
                                   ref.dtype,ref.device,ref.requires_grad)
        else:
            self._cache_integrity=None

    def _check_model(self):
        # The base class calls _check_model() from its __init__ BEFORE 064's
        # cache has been constructed. The guard is installed atomically after
        # inherited initialization; post-construction entrypoints check it.
        super()._check_model()
        if "_cache_integrity" not in self.__dict__:
            return
        reference=self._cache_integrity
        current=self.__dict__.get("_cap_all_ref")
        if reference is None:
            if current is not None:
                raise ValueError("native-mode cache unexpectedly present")
            return
        if (not torch.is_tensor(current) or not hasattr(current,"_version")
                or (id(current),current._version,tuple(current.shape),
                    current.dtype,current.device,current.requires_grad)!=reference):
            raise ValueError("cached positional features replaced or modified")
