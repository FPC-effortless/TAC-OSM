"""Standalone Torch median forecast from recorded technical fields only."""
import hashlib
import json
import math
from pathlib import Path
import sys
import torch
from model import RepairOutcomeForecaster

torch.set_num_threads(1); torch.set_deterministic_debug_mode('error')
root = Path(__file__).resolve().parent
meta = json.loads((root/'metadata.json').read_text()); schema = meta['schema']
assert hashlib.file_digest((root/'model.pt').open('rb'), 'sha256').hexdigest() == meta['checkpoint_sha256']
model = RepairOutcomeForecaster(meta['features'], [0.]*meta['features'], [1.]*meta['features'], 1.)
model.load_state_dict(torch.load(root/'model.pt', map_location='cpu', weights_only=True)); model.eval()
missing_values = ('', 'na', 'n/a', 'null', 'missing', 'nan')


def normalize(value):
    return ' '.join(str(value).strip().casefold().split()) if value is not None else ''


def encode(row):
    result = []
    for name in schema['categorical']:
        value = normalize(row.get(name)); vocab = schema['vocabulary'][name]
        result.extend(float(value == known) for known in vocab)
        result += [float(value in missing_values), float(value not in vocab and value not in missing_values)]
    for name in schema['numeric']:
        try: value = float(row.get(name, ''))
        except (ValueError, TypeError): value = float('nan')
        missing = not math.isfinite(value) or value < 0
        value = schema['numeric_imputation'][name] if missing else value
        result += [math.log1p(value), float(missing)]
    return result


with torch.inference_mode():
    for line in sys.stdin:
        records = json.loads(line)['records']
        if not records: raise ValueError('At least one technical record required')
        output = model(torch.tensor([encode(row) for row in records], dtype=torch.float64))
        if not torch.isfinite(output).all(): raise ValueError('Nonfinite model prediction; no silent clipping')
        print(json.dumps(dict(time_to_fill_20L_seconds=output.tolist()), allow_nan=False), flush=True)
