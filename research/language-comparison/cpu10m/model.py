"""Native CSL byte language model. Instantiate after setting the intended random seed."""
from dataclasses import replace

import torch
from torch import nn
from tac_osm.byte100k_001 import byte100k_config
from tac_osm.csl125_memory_precision import enable_precise_memory_
from tac_osm.csl125_r2 import CSL125R2


class NativeByte(nn.Module):
    def __init__(self, depth, width, state_width, ff):
        super().__init__()
        cfg = replace(byte100k_config(), layers=depth, rungs=(depth,), width=width,
                      state_width=state_width, ff_width=ff, query_heads=8, kv_heads=4,
                      attention_layers=(1,) if depth == 1 else tuple(range(2, depth + 1, 2)), max_parameters=10_500_000)
        self.core = CSL125R2(cfg)
        self.core.tool_head = nn.Identity()
        self.core.field_head = nn.Identity()
        self.core.accept_head = nn.Identity()
        enable_precise_memory_(self.core)

    def forward(self, ids, state=None, *, stream_ids=None):
        # Preserve the single-request timing API. Batched streams can carry
        # state across calls when the caller supplies the same ordered identities.
        if stream_ids is None and ids.ndim == 2 and ids.shape[0] == 1:
            stream_ids = ('request',)
        outputs, state = self.core(ids, state=state, stream_ids=stream_ids)
        return outputs[self.core.config.layers]['token_logits'], state


def selected_native_byte():
    """10,103,943 learned parameters. Tied byte matrix initialized at inverse width."""
    model = NativeByte(1, 896, 128, 2304)
    with torch.no_grad():
        model.core.embedding.weight.mul_(1 / model.core.config.width)
    return model
