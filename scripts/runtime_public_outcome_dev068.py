"""Torch-only inference. Requires public JSON; no simulator or teacher."""
import hashlib
import json
from pathlib import Path
import sys
import torch
from agent_outcome import PublicOutcomeAgent

torch.set_num_threads(1)
# Supported eager interface sets the same strict deterministic-kernel flag
# without importing PyTorch's optional compiler stack at runtime.
torch.set_deterministic_debug_mode('error')
assert torch.are_deterministic_algorithms_enabled()
root = Path(__file__).resolve().parent
meta = json.loads((root/'metadata.json').read_text())
spec = meta['spec']
model = PublicOutcomeAgent(spec['backend'], spec['width'], spec['aux']).double().eval()
assert hashlib.file_digest((root/'agent.pt').open('rb'), 'sha256').hexdigest() == meta['checkpoint_sha256']
model.load_state_dict(torch.load(root/'agent.pt', weights_only=True, map_location='cpu'))
state = None


def tensor(value, shape):
    x = torch.tensor(value, dtype=torch.float64)
    if list(x.shape) != shape or not torch.isfinite(x).all():
        raise ValueError(f'Finite public tensor {shape} required')
    return x[None]


with torch.inference_mode():
    for line in sys.stdin:
        request = json.loads(line)
        if request.get('reset') or state is None:
            keys = tensor(request['keys'], [2, 16])
            if (keys.norm(dim=-1) < 1e-12).any(): raise ValueError('Nonzero entity keys required')
            state = model.initial(keys)
        for event in request['events']:
            state = model.observe(state, tensor(event, [46]))
        remaining = tensor([request['remaining_fraction']], [1])
        if not bool(((remaining >= 0) & (remaining <= 1)).all()):
            raise ValueError('Remaining public budget fraction outside [0,1]')
        out = model.decide(state, tensor(request['candidates'], [5, 19]),
            tensor(request['prices'], [2]), remaining)
        print(json.dumps(dict(index=int(model.choice(out)[0]), scores=out['logits'][0].tolist(),
            paired_measurement_probabilities=out['posterior'][0].tolist(),
            future_verification_probabilities=out['future'][0].softmax(-1).tolist(),
            normalized_physical_cost_predictions=out['cost'][0].tolist()), allow_nan=False), flush=True)
