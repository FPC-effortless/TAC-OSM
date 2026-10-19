"""Torch-only point forecasts from a supplied public eight-reading context."""
import hashlib
import json
from pathlib import Path
import sys
import torch
from model import ThermalForecast

torch.set_num_threads(1)
root=Path(__file__).resolve().parent;metadata=json.loads((root/'metadata.json').read_text())
assert hashlib.file_digest((root/'model.pt').open('rb'),'sha256').hexdigest()==metadata['checkpoint_sha256']
model=ThermalForecast(**metadata['spec']).double().eval()
model.load_state_dict(torch.load(root/'model.pt',weights_only=True,map_location='cpu'))

with torch.inference_mode():
    for line in sys.stdin:
        request=json.loads(line)
        values=torch.tensor(request['values'],dtype=torch.float64)
        minutes=torch.tensor(request['minute_of_day'],dtype=torch.float64)
        if values.shape!=(8,12) or minutes.shape!=(8,) or not torch.isfinite(values).all() or not torch.isfinite(minutes).all():
            raise ValueError('Finite8x12publicpastreadings and8minute-of-day values required')
        prediction=model(values[None],minutes[None])[0]
        print(json.dumps(dict(channels=metadata['channels'],horizon_minutes=[30,60],
            temperature_F=prediction.tolist()),allow_nan=False),flush=True)
