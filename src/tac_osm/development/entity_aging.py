"""DEV-023: independent public coordinate-isolation and aging-scope factors."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .retention_controls import RetentionControl

ARMS = ("native_global", "masked_global", "native_local", "masked_local", "schema41", "memoryless")


class EntityAgingControl(RetentionControl):
    def __init__(self, factorial_arm, seed=0):
        if factorial_arm not in ARMS:
            raise ValueError("unregistered entity/aging intervention")
        base = factorial_arm if factorial_arm in ("schema41", "memoryless") else (
            "entity" if factorial_arm.startswith("masked") else "native")
        super().__init__(base, seed)
        self.factorial_arm = factorial_arm
        self.local_aging = factorial_arm.endswith("local")

    def _update_encoded(self, state, x, entities, values):
        # These corners exactly preserve DEV-022 native and bundled updates.
        if self.arm != "tac_joint" or self.factorial_arm in ("native_global", "masked_local"):
            return super()._update_encoded(state, x, entities, values)
        m = self.memory
        encoded = F.one_hot(values.long(), 2).to(x.dtype)
        joint = torch.cat((x, encoded), -1)
        mask = self.entity_mask(entities, x)[:, None, None, :]
        k = m.keys(x)
        if self.entity_indexed:
            k = F.normalize(k * mask.squeeze(2), dim=-1)
        erase = m.erase(joint).reshape_as(k).sigmoid()
        decay = m.decay(joint).sigmoid()[..., None, None]
        assignment = m.assignment(x).softmax(-1)
        write = m.write(joint).reshape(-1, 4, 8).sigmoid()
        v = m.value(encoded)
        erased = torch.einsum("bevk,bek->bev", state, erase * k)
        contribution = assignment[..., None, None] * (
            -erased[..., None] * k[:, :, None, :] +
            (write * v[:, None, :])[..., None] * k[:, :, None, :])
        if self.local_aging:
            # Crucial: restrict ONLY decay. Learned erase/write keys remain full.
            aged = state + (decay - 1) * state * mask
        else:
            # Masked keys isolate erase/write/read, but all coordinates still age.
            aged = decay * state
        return aged + contribution
