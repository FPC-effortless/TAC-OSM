"""Source-pinned COLLECT012 internal learnability study; no qualification flag."""
from pathlib import Path
import hashlib,json,sys

ROOT=Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def require(ok,message):
    if not ok:raise ValueError(message)

def runtime_root():
    for p in (ROOT/'runtime',ROOT.parent/'collect011',ROOT.parent/'collect011_work'):
        if (p/'collect011/model.py').is_file():return p
    raise FileNotFoundError('pinned COLLECT011 runtime missing')

sys.path.insert(0,str(runtime_root()))
from collect011.model import CollectiveTAC

SPECS={
    'attention':dict(width=16,ffn_dim=38,depth=4),
    'phase_p':dict(width=16,ffn_dim=59,depth=4),
    'hybrid_hard_p':dict(width=16,ffn_dim=57,depth=4),
}
EXPECTED={'attention':99875,'phase_p':99879,'hybrid_hard_p':99971}

def build(arm,seed):
    import torch
    require(arm in SPECS,'unregistered model')
    torch.manual_seed(seed)
    m=CollectiveTAC(arm,**SPECS[arm]).float()
    n=sum(p.numel() for p in m.parameters() if p.requires_grad)
    require(n==EXPECTED[arm] and n==sum(p.numel() for p in m.parameters()),'actual model count mismatch')
    require(95000<=n<=105000,'model outside standing size band')
    return m

def source_paths():
    rt=runtime_root()
    files=[rt/'collect011/__init__.py',rt/'collect011/model.py',rt/'collect011/memory.py']
    files+=sorted((rt/'vendor').rglob('*.py'))
    return {str(p.relative_to(rt)):p for p in files if '__pycache__' not in p.parts}

def verify_data():
    manifest=json.loads((ROOT/'data/SPLIT_MANIFEST.json').read_text())
    out={}
    for part in ('train','dev'):
        path=ROOT/'data'/f'{part}.bin'
        require(path.is_file() and digest(path)==manifest['parts'][part]['sha256'],'original '+part+' split changed')
        require(path.stat().st_size==manifest['parts'][part]['bytes'],'original split length changed')
        out[part]=path.read_bytes()
    return out,manifest

def internal_gate(manifest,phase='preflight'):
    import importlib.util
    sys.path.insert(0,str(ROOT/'policy/src'))
    spec=importlib.util.spec_from_file_location('collect012_pinned_parity',ROOT/'policy/scripts/validate_research_parity.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    result=mod.check_manifest(manifest,phase)
    require(result.get('reward_integrity')=='NOT_INDEPENDENTLY_CERTIFIED_INTERNAL_RESEARCH','wrong execution level')
    require(not result.get('eligible_for_architecture_superiority'),'internal result falsely promoted')
    return result
