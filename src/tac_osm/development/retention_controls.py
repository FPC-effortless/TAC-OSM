"""DEV-022 interventions; public inputs only, unchanged 768-scalar TAC state."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .memory_baselines import MemoryBaseline

ARMS = ("native", "no_decay", "no_erase", "neither", "entity", "entity_no_decay",
        "aligned", "aligned_no_decay", "schema41", "memoryless")


class RetentionControl(MemoryBaseline):
    def __init__(self, intervention, seed=0):
        if intervention not in ARMS:
            raise ValueError("unregistered retention intervention")
        super().__init__(intervention if intervention in ("schema41", "memoryless") else "tac_joint", seed)
        self.intervention = intervention
        self.disable_decay = intervention in ("no_decay", "neither", "entity_no_decay", "aligned_no_decay")
        self.disable_erase = intervention in ("no_erase", "neither")
        self.entity_indexed = intervention.startswith("entity")
        self.aligned = intervention.startswith("aligned")
        if self.arm == "tac_joint":
            for name in ("decay", "erase"):
                if getattr(self, "disable_" + name):
                    for p in getattr(self.memory, name).parameters():
                        p.requires_grad_(False)
            if self.aligned:
                for p in self.read_key.parameters():
                    p.requires_grad_(False)

    @staticmethod
    def entity_mask(entities, reference):
        # Eight public entities occupy disjoint 3-coordinate key slices.
        return F.one_hot(entities.long(), 8).to(reference.dtype).repeat_interleave(3, -1)

    def _update_encoded(self, state, x, entities, values):
        if self.arm != "tac_joint" or self.intervention == "native":
            return super()._update_encoded(state, x, entities, values)
        m = self.memory
        encoded = F.one_hot(values.long(), 2).to(x.dtype)
        joint = torch.cat((x, encoded), -1)
        k = m.keys(x)
        mask = self.entity_mask(entities, x)[:, None, None, :]
        if self.entity_indexed:
            k = F.normalize(k * mask.squeeze(2), dim=-1)
        erase = (torch.zeros_like(k) if self.disable_erase else
                 m.erase(joint).reshape_as(k).sigmoid())
        decay = (x.new_ones((len(x), 4)) if self.disable_decay else m.decay(joint).sigmoid())
        assignment = m.assignment(x).softmax(-1)
        write = m.write(joint).reshape(-1, 4, 8).sigmoid()
        v = m.value(encoded)
        erased = torch.einsum("bevk,bek->bev", state, erase * k)
        proposal = decay[..., None, None] * state + assignment[..., None, None] * (
            -erased[..., None] * k[:, :, None, :] +
            (write * v[:, None, :])[..., None] * k[:, :, None, :])
        # Entity treatment isolates both writes and aging to the addressed slot.
        return state + (proposal - state) * mask if self.entity_indexed else proposal

    def decode(self, state, queries):
        if self.arm != "tac_joint" or not (self.entity_indexed or self.aligned):
            return super().decode(state, queries)
        if queries.shape[1:] != (8, 3, 14):
            raise ValueError("public entity/action/time queries required")
        b = len(state)
        q = self.query_encoder(queries).reshape(b, 24, 12)
        projection = self.memory.key if self.aligned else self.read_key
        keys = projection(q).reshape(b, 24, 4, 24)
        if self.entity_indexed:
            entities = queries[..., :8].argmax(-1).reshape(b, 24)
            keys = keys * self.entity_mask(entities, keys)[:, :, None, :]
        keys = F.normalize(keys, dim=-1)
        read = torch.einsum("bevk,bqek->bqev", state, keys).flatten(2)
        return self.head(torch.cat((read, q), -1)).reshape(b, 8, 3, 2)
