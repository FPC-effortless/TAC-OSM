"""Independent final-horizon CPU worker after Weather releases the second CPU."""
import time
from pathlib import Path
import torch
import scripts.me036_research as base
from scripts.me036b_continue import continue_fit

if __name__=='__main__':
    torch.set_num_threads(1)
    while not (base.ROOT/'results/me036b/weather-h720-ridge/fit.json').exists():time.sleep(.25)
    parent=base.json.loads((base.OUT/'frozen.json').read_text())
    z,meta=base.load_data('electricity')
    for r in parent['fits']:
        if r['dataset']=='electricity' and r['horizon']==720 and r['status']=='PASS':continue_fit(r,z,meta)
