"""Explicit mapping of the actual TAC/phase state to repository diagnostics."""
from dataclasses import dataclass,replace
from pathlib import Path
import importlib.util,json,sys
import torch
from .model import CollectiveTAC,CollectState,FFN_DIM as FFN,named_state_tensors as tensors
from .memory import PhaseState

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"governance/src"))
from tac_osm import research_probe as source

@dataclass(frozen=True)
class Sample:
    tokens:torch.Tensor
    owners:tuple[str,...]

def patch(state,name,value):
    p=state.payload();parts=name.split(".")
    if parts[0]=="native":p["native"]["native"]["layers"][int(parts[1])][parts[2]]=value
    elif parts[0]=="ssa":p["native"]["ssa"][parts[1]]=value
    else:p["phase"][int(parts[1])][parts[2]]=value
    return CollectState.from_payload(p)

def state_views(state):
    # This adapter covers bounded native banks/rotary/SSA/phase only. The native
    # attention control's variable-length KV cache is tested separately by the
    # full-graph chunk/causality/owner tests. Never disguise absent KV as a scalar.
    return {n:v for n,v in tensors(state).items() if not n.endswith((".recent_key",".recent_value"))}

def gradient_graph(model,state,sample):
    leaves={n:t.detach().clone().requires_grad_() for n,t in state_views(state).items()}
    s=state
    for n,v in leaves.items():s=patch(s,n,v)
    y,_=model(sample.tokens[:,-1:],state=s,owners=sample.owners)
    return {"next_byte_logits":y},leaves

def validate(model,sample):
    if not isinstance(model,CollectiveTAC) or sample.tokens.dtype!=torch.long or sample.tokens.ndim!=2 or sample.tokens.shape[1]<2:
        raise ValueError("actual CollectiveTAC and observed Long[B,T>=2] sample required")
    model.core._owners(sample.owners)

ADAPTER=source.ResearchProbeAdapter(name="COLLECT011.actual_tac_lm.v1",validate=validate,
    steps=lambda s:s.tokens.shape[1]-1,start=lambda m,s:m.start(s.owners),
    advance=lambda m,state,s,t:m(s.tokens[:,t:t+1],state=state,owners=s.owners)[1],
    forecast=lambda m,state,s:{"next_byte_logits":m(s.tokens[:,-1:],state=state,owners=s.owners)[0]},
    state_views=state_views,patch_state=patch,batch_size=lambda s:s.tokens.shape[0],
    batch_item=lambda s,i:Sample(s.tokens[i:i+1],s.owners[i:i+1]),gradient_graph=gradient_graph)

def main():
    torch.set_num_threads(1);rows=[]
    for arm in CollectiveTAC.ARMS:
        torch.manual_seed(1);m=CollectiveTAC(arm,ffn_dim=FFN[arm]).double()
        sample=Sample(torch.arange(16).reshape(2,8),("a","b"))
        dep=source.probe_dependencies(m,sample,ADAPTER)
        isolation=source.probe_entity_isolation(m,sample,ADAPTER,tolerance=1e-10)
        with torch.no_grad():
            state,_=source._replay(ADAPTER,m,sample)
            _,pure=source._forecast_pure(ADAPTER,m,state,sample)
        rows.append(dict(arm=arm,dependencies=dep,isolation=isolation,pure_forecast=pure,
            state_coverage="native fast/slow banks, rotary, SSA, phase; variable native KV tested in separate full-graph tests",
            full_zero_patch_memory_probe="INCONCLUSIVE: zero precision is outside the SPD state contract",
            candidate_permutation="INCONCLUSIVE: no candidate-order axis in byte logits"))
    root=Path(__file__).resolve().parents[1]
    (root/"results/RESEARCH_PROBE.json").write_text(json.dumps(dict(status="NO_FIT_ENGINEERING",adapter=ADAPTER.name,rows=rows),indent=2))
    print(json.dumps([{k:r[k] for k in ("arm","isolation","pure_forecast")} for r in rows],indent=2))

if __name__=="__main__":main()
