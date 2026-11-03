"""COLLECT014: source-identified factorial repairs on the frozen 100K graph."""
from pathlib import Path
import hashlib,json,sys

ROOT=Path(__file__).resolve().parents[1]
def older_root():
    for p in (ROOT.parent/'collect012',ROOT.parent/'collect012_work'):
        if (p/'collect012/common.py').is_file():return p
    raise FileNotFoundError('COLLECT012 runtime missing')
OLD=older_root();sys.path.insert(0,str(OLD))
from collect012.common import canonical,digest,require,verify_data,internal_gate,source_paths as old_sources

SPECS={'balance':dict(read_after=False,balance=.01),
       'write_first':dict(read_after=True,balance=0.),
       'combined':dict(read_after=True,balance=.01)}
EXPECTED={a:99971 for a in SPECS}

def build(arm,seed):
    import torch
    from .model import StudyTAC
    require(arm in SPECS,'unknown registered repair')
    torch.manual_seed(seed);m=StudyTAC(arm,**SPECS[arm]).float()
    require(sum(p.numel() for p in m.parameters())==99971,'actual model size differs')
    return m

def source_paths():
    paths={'research/collect011/'+p:f for p,f in old_sources().items()}
    for n in ('__init__.py','common.py','reference104.py','reference110.py','reference.py','train.py','restore_data.py'):
        paths['research/collect012/collect012/'+n]=OLD/'collect012'/n
    for n in ('__init__.py','common.py','model.py','reference.py','train.py','evaluate.py','assays.py','test_prefit.py','register.py','verify_launch.py'):
        paths['research/collect014/collect014/'+n]=ROOT/'collect014'/n
    paths['research/collect014/WORKFLOW.yml']=ROOT/'WORKFLOW.yml'
    return paths
