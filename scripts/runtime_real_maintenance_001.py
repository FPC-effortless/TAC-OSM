"""Torch-only bounded sensor forecast; no health/repair decision override."""
import hashlib
import json
from pathlib import Path
import sys
import torch
from agent_model import MaintenanceForecast

torch.set_num_threads(1);root=Path(__file__).resolve().parent;meta=json.loads((root/'metadata.json').read_text());assert hashlib.file_digest((root/'agent.pt').open('rb'),'sha256').hexdigest()==meta['checkpoint_sha256'];s=meta['spec'];m=MaintenanceForecast(meta['arm'],s['width'],s['aux']).double().eval();m.load_state_dict(torch.load(root/'agent.pt',weights_only=True,map_location='cpu'))
with torch.inference_mode():
    for line in sys.stdin:
        r=json.loads(line);x=torch.tensor(r['history'],dtype=torch.float64)
        if list(x.shape)!=[24,15] or not torch.isfinite(x).all():raise ValueError('24 complete ten-minute sensor vectors required')
        o=m(x[None]);print(json.dumps({'next_sensor_mean':o['mean'][0].tolist(),'next_sensor_std':(o['log_variance'][0].mul(.5).exp()*m.scale).tolist(),'scope':'observational forecast; no repair-success claim'},allow_nan=False),flush=True)
