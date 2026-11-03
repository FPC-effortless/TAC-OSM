"""COLLECT018 source-native acquisition study; old rounds remain immutable."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT.parent/'collect016',ROOT.parent/'collect016_work'):
    if (p/'collect016/common.py').is_file():C16=p;break
else:raise FileNotFoundError('COLLECT016 source required')
sys.path.insert(0,str(C16))
from collect016.common import canonical,digest,require,verify_data,internal_gate,fingerprint,freeze_tree,source_paths as old_sources
FAMILIES=('phase','context','gru')
OBJECTIVES=('joint','first')
ARMS=tuple(a+'_'+b for a in FAMILIES for b in OBJECTIVES)
SEEDS=(18001,18002,18003)
EXPECTED={'phase':99971,'context':99971,'gru':100323}
def build(arm,seed):
    import torch
    from .model import ContextPhase,GRUControl
    from collect016.common import build as original
    family=arm.split('_')[0];require(arm in ARMS,'unknown arm');torch.manual_seed(seed)
    m=GRUControl() if family=='gru' else ContextPhase() if family=='context' else original('full_bptt',seed)
    n=sum(p.numel() for p in m.parameters());require(n==EXPECTED[family] and all(p.requires_grad for p in m.parameters()),'live active count differs')
    require(95000<=n<=105000,'absolute count fails');return m
def source_paths():
    paths=old_sources()
    for name in ('__init__','common','model','reference','data','train','evaluate','test_prefit','register','verify_launch','probe'):
        paths['research/collect018/collect018/'+name+'.py']=ROOT/'collect018'/f'{name}.py'
    paths['research/collect018/WORKFLOW.yml']=ROOT/'WORKFLOW.yml'
    for p in sorted((ROOT/'probe_vendor').glob('*.py')):paths['research/collect018/probe_vendor/'+p.name]=p
    return paths
