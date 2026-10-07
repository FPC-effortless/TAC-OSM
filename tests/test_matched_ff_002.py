import json,inspect
import pytest
pytest.importorskip("torch")
from tac_osm.integrated_e2e_ff_feedback_baseline import FeedForwardFeedbackMultimodal
def test_parameter_match():
 m=FeedForwardFeedbackMultimodal();assert abs(m.parameter_count()-74126)/74126<=.01
def test_no_persistence():
 s=inspect.getsource(FeedForwardFeedbackMultimodal);assert "ExplicitEntityState" not in s and "FixedCASM" not in s and "post_action_update" not in s
def test_contract():
 c=json.load(open("contracts/TACOSM-PLM-MATCHED-FF-002.json"));assert c["status"]=="pre-registered" and c["primary_endpoint"]=="heldout_q2_accuracy"

# CI trigger only: contract and protocol unchanged.
