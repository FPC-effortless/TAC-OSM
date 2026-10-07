import json,inspect
import pytest

torch = pytest.importorskip("torch")
from tac_osm.integrated_e2e_ff_baseline import FeedForwardMultimodal
def test_parameter_match():
    m=FeedForwardMultimodal();assert abs(m.parameter_count()-74126)/74126<=.01
def test_no_persistence_components():
    s=inspect.getsource(FeedForwardMultimodal);assert "ExplicitEntityState" not in s and "FixedCASM" not in s and "post_action_update" not in s
def test_contract():
    c=json.load(open("contracts/TACOSM-PLM-MATCHED-FF-001.json"));assert c["primary_endpoint"]=="heldout_q2_accuracy";assert c["status"]=="pre-registered"

# CI trigger: scientific contract is unchanged; this commit only re-runs the registered gate.
