"""Source-pinned carried-state credit assignment study; internal FIT/DEV only."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
def older(name):
    for p in (ROOT.parent/name,ROOT.parent/(name+'_work')):
        if (p/name/'common.py').is_file():return p
    raise FileNotFoundError(name)
OLD=older('collect014');sys.path.insert(0,str(OLD))
from collect014.common import OLD as C12,canonical,digest,require,verify_data,internal_gate,source_paths as c14_sources
from collect014.train import fingerprint,freeze_tree
ARMS=('full_bptt','detach_128','current_only')
SEEDS=(16001,16002,16003)
EXPECTED={'full_bptt':99971,'detach_128':99971,'current_only':99907}
def build(arm,seed):
    import torch
    from .model import CarriedTAC
    require(arm in ARMS,'unknown arm');torch.manual_seed(seed)
    m=CarriedTAC(arm)
    require(sum(p.numel() for p in m.parameters())==EXPECTED[arm],'live parameter count differs')
    return m
def source_paths():
    paths=c14_sources()
    for n in ('__init__.py','common.py','model.py','data.py','reference.py','train.py','evaluate.py','test_prefit.py','register.py','verify_launch.py'):
        paths['research/collect016/collect016/'+n]=ROOT/'collect016'/n
    paths['research/collect016/WORKFLOW.yml']=ROOT/'WORKFLOW.yml'
    return paths
