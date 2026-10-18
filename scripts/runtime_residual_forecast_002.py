"""JSON-lines point forecasts. Only CPU PyTorch beyond the standard library."""
import hashlib
import json
from pathlib import Path
import sys
import torch
from model import ResidualForecast
root=Path(__file__).resolve().parent;meta=json.loads((root/'metadata.json').read_text());weights=root/'agent.pt'
assert hashlib.sha256(weights.read_bytes()).hexdigest()==meta['checkpoint_sha256']
torch.set_num_threads(1);m=ResidualForecast(**meta['spec'],seed=0).double().eval();m.load_state_dict(torch.load(weights,weights_only=True))
with torch.no_grad():
    for line in sys.stdin:
        request=json.loads(line);x=torch.tensor(request['history'],dtype=torch.float64)
        if x.shape!=(24,8) or not torch.isfinite(x).all():raise ValueError('Expected finite24x8 raw analog context')
        forecast=m(x[None])[0]
        if not torch.isfinite(forecast).all():raise ValueError('Non-finite forecast')
        print(json.dumps({'forecast':forecast.tolist(),'features':meta['features'],'qualification':'RESEARCH_ONLY'},allow_nan=False),flush=True)
