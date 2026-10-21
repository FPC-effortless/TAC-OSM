"""Second single-thread CPU worker; same registered fit functions and budgets.

The coordinator must pause before entering Electricity until this worker exits.
No shared in-progress fit may be run by both processes.
"""
import torch
import scripts.me036_research as base
from scripts.me036b_continue import continue_fit, ridge_fit

if __name__=='__main__':
    torch.set_num_threads(1)
    parent=base.json.loads((base.OUT/'frozen.json').read_text())
    z,meta=base.load_data('electricity')
    for r in parent['fits']:
        if r['dataset']=='electricity' and r['status']=='PASS':continue_fit(r,z,meta)
    for h in base.HORIZONS:ridge_fit('electricity',h,z,meta)
