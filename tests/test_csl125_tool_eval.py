"""Evaluation correctness controls: no neural scores or trained examples."""
import json
import pytest
from tac_osm.csl125_tool_protocol import catalogue, ToolContractError
from tac_osm.csl125_tool_eval import score_tool_case, aggregate_tool_scores


def ts():
    return catalogue([
        {"name":"weather","description":"get weather","parameters":{
            "type":"object","properties":{"city":{"type":"string"}},
            "required":["city"],"additionalProperties":False}},
        {"name":"clock","description":"get time","parameters":{
            "type":"object","properties":{"zone":{"type":"string"}},
            "required":["zone"],"additionalProperties":False}}
    ])


def call(name,field,value):
    return {"name":name,"arguments":{field:value}}


def js(calls):
    return json.dumps({"function_calls":calls},sort_keys=True,separators=(",",":"))


def test_exact_args_and_order_vs_reversed_multi_call():
    x=js([call("weather","city","Lagos"),call("clock","zone","Africa/Lagos")])
    y=js([call("clock","zone","Africa/Lagos"),call("weather","city","Lagos")])
    correct=score_tool_case(x,x,ts())
    reverse=score_tool_case(y,x,ts())
    assert correct.full_exact and correct.sequence_exact and correct.arguments_exact
    assert reverse.valid_prediction and not reverse.sequence_exact and not reverse.full_exact


def test_argument_error_and_no_call_false_positive():
    x=js([call("weather","city","Lagos")])
    y=js([call("weather","city","Kano")])
    s=score_tool_case(y,x,ts())
    assert s.sequence_exact and not s.arguments_exact and not s.full_exact
    false=score_tool_case(x,js([]),ts())
    assert false.false_call and not false.abstention_correct
    abstain=score_tool_case(js([]),js([]),ts())
    assert abstain.abstention_correct and abstain.full_exact


def test_malformed_prediction_is_not_relabelled_as_abstention():
    invalid=score_tool_case('{"function_calls":[{"name":"bad","arguments":{}}]}',js([]),ts())
    assert not invalid.valid_prediction and not invalid.abstention_correct and invalid.false_call
    assert not invalid.full_exact
    real=score_tool_case(js([]),js([]),ts())
    agg=aggregate_tool_scores([invalid,real])
    assert agg["count"]==2 and agg["exact_ast"]==0.5
    assert agg["invalid_fraction"]==0.5
    assert agg["abstention_correct_count"]==1
    assert agg["invalid_not_counted_as_abstention"]
    with pytest.raises(ToolContractError,match="empty"):
        aggregate_tool_scores([])
