"""Real subprocesses reject research imports while running persistent control."""
from pathlib import Path
import sys
import json
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


@pytest.mark.parametrize('family',['identity','belief'])
def test_clean_portable_runtime_has_no_research_dependency(tmp_path,family):
    torch.set_num_threads(1)
    if family=='identity':
        from run_identity_dev0615 import make,setup_gate
        from tac_osm.development.robust_identity import configurations
        from audit_identity_dev0615 import standalone
        arm='split_robust';spec=configurations()[arm];m=make(arm,spec,199615,setup_gate()).double().eval()
    else:
        from run_belief_dev062 import make,gate_fit
        from tac_osm.development.belief_agent import configurations
        from audit_belief_dev062 import standalone
        arm='tac';spec=configurations()[arm];m=make(arm,spec,199620,gate_fit()).double().eval()
    folder=tmp_path/'checkpoint';folder.mkdir();torch.save(m.state_dict(),folder/'agent.pt');report={'arm':arm,'spec':spec,'seed':199615 if family=='identity' else 199620};destination=tmp_path/'portable'
    with torch.no_grad():count=standalone(folder,report,m,destination,demo_seed=1996201)
    assert count>=1
    assert not list(destination.glob('*world*'))
    expected=[json.loads(x) for x in (destination/'demo-expected.jsonl').read_text().splitlines()]
    assert count==len(expected)
