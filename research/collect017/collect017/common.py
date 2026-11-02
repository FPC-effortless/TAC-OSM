"""Additional post-null native-attention learnability control; no winner claim."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT.parent/'collect016',ROOT.parent/'collect016_work'):
    if (p/'collect016/common.py').is_file():C16=p;break
else:raise FileNotFoundError('COLLECT016 source missing')
sys.path.insert(0,str(C16))
from collect016.common import canonical,digest,require,verify_data,internal_gate,fingerprint,freeze_tree,source_paths as prior_sources
ARMS=('attention',);SEEDS=(16001,16002,16003);EXPECTED={'attention':99875}
def build(arm,seed):
    import torch
    from .model import AttentionSanity
    require(arm=='attention','unknown model');torch.manual_seed(seed);m=AttentionSanity()
    require(sum(p.numel() for p in m.parameters())==99875,'live active size differs');return m
def source_paths():
    paths=prior_sources()
    for n in ('__init__.py','common.py','model.py','train.py','evaluate.py','test_prefit.py','register.py','verify_launch.py'):
        paths['research/collect017/collect017/'+n]=ROOT/'collect017'/n
    paths['research/collect017/WORKFLOW.yml']=ROOT/'WORKFLOW.yml'
    return paths
