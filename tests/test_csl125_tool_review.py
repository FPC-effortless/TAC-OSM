"""Host permission checks cannot be bypassed by a schema-valid model output."""
import pytest

from tac_osm.csl125_tool_protocol import catalogue, validate_tool_proposal, ToolContractError
from tac_osm.csl125_tool_review import review_proposal


def setup():
    tools=catalogue([
      {"name":"get_weather","description":"Read forecast","effect":"read",
       "parameters":{"type":"object","properties":{
          "city":{"type":"string"}},"required":["city"],"additionalProperties":False}},
      {"name":"unlock_door","description":"Unlock an entry","effect":"high_impact",
       "parameters":{"type":"object","properties":{
          "door":{"type":"string"}},"required":["door"],"additionalProperties":False}}
    ])
    proposal=validate_tool_proposal({"function_calls":[
        {"name":"get_weather","arguments":{"city":"Lagos"}},
        {"name":"unlock_door","arguments":{"door":"front"}}]},tools)
    return tools,proposal


def test_explicit_permission_and_confirmation_are_separate_from_model():
    tools,proposal=setup()
    review=review_proposal(proposal,tools,permitted_tool_names=frozenset(["get_weather","unlock_door"]))
    assert [r.status for r in review]==["candidate_only","confirmation_required"]
    assert review[0].reason.endswith("execution")
    only=review_proposal(proposal,tools,permitted_tool_names=frozenset(["get_weather"]))
    assert only[1].status=="denied"
    both=review_proposal(proposal,tools,permitted_tool_names=frozenset(["get_weather","unlock_door"]),
                         confirmed_call_indices=frozenset([1]))
    assert both[1].status=="candidate_only"
    assert not hasattr(both[1],"execute")
    assert not hasattr(both[1],"confidence")


def test_malformed_permission_input_fails_closed():
    tools,proposal=setup()
    with pytest.raises(ToolContractError,match="explicit"):
        review_proposal(proposal,tools,permitted_tool_names=["get_weather"])
    with pytest.raises(ToolContractError,match="explicit"):
        review_proposal(proposal,tools,permitted_tool_names=frozenset(["get_weather"]),confirmed_call_indices=[1])


def test_abstention_produces_no_plan_even_when_tool_is_whitelisted():
    tools,_=setup()
    p=validate_tool_proposal('{"function_calls":[]}',tools)
    assert review_proposal(p,tools,permitted_tool_names=frozenset(["get_weather"]))==()
