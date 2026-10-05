import json
from pathlib import Path


def test_contract_split_is_disjoint():
    c=json.loads(Path("contracts/TACOSM-PLM-AMORTIZED-PROBE-POLICY-018A.json").read_text())
    t=set(c["data_split"]["training_seeds"])
    v=set(c["data_split"]["validation_seeds"])
    e=set(c["data_split"]["test_seeds"])
    assert not (t & v or t & e or v & e)

def test_test_teacher_hidden():
    c=json.loads(Path("contracts/TACOSM-PLM-AMORTIZED-PROBE-POLICY-018A.json").read_text())
    assert c["data_split"]["test_seed_teacher_outputs_used_for_training"] is False
    assert c["data_split"]["checkpoint_selected_without_test_data"] is True
