"""R4 protocol/grammar/scientific boundary tests: source-only, never capability proof."""
import copy
import json
import pytest

from tac_osm.csl125_tool_protocol import (
    ByteFallbackCodec, ToolContractError, build_tool_prompt, catalogue,
    prepare_tool_supervision, tool_observation, validate_tool_proposal,
)
from tac_osm.csl125_tool_grammar import _HEADER, allowed_byte_ids, prefix_state


def tools():
    return catalogue([
        {"name": "get_weather", "description": "Weather for a city",
         "parameters": {"type": "object", "properties": {
             "city": {"type": "string", "minLength": 2, "maxLength": 80},
             "units": {"type": "string", "enum": ["celsius","fahrenheit"]}},
             "required": ["city"], "additionalProperties": False}},
        {"name": "set_brightness", "description": "Change phone brightness",
         "parameters": {"type": "object", "properties": {
             "value": {"type":"integer", "minimum":0, "maximum":100}},
             "required":["value"], "additionalProperties":False},
         "effect": "write"},
        {"name": "check_flights", "description": "List flight paths",
         "parameters": {"type": "object", "properties": {
             "cities": {"type": "array", "items": {"type":"string","maxLength":80},
                        "minItems": 2, "maxItems": 4},
             "direct": {"type": "boolean"}},
             "required": ["cities"], "additionalProperties": False}},
    ])


def test_valid_no_call_abstention_and_ordered_calls():
    ts=tools()
    no=validate_tool_proposal('{"function_calls":[]}',ts)
    assert no.status=="abstain" and no.function_calls==()
    q = {"function_calls":[
        {"name":"get_weather","arguments":{"city":"Lagos"}},
        {"name":"set_brightness","arguments":{"value":30}}]}
    proposal=validate_tool_proposal(json.dumps(q),ts)
    assert proposal.status=="proposed"
    assert [c["name"] for c in proposal.function_calls]==["get_weather","set_brightness"]
    assert proposal.raw_json==json.dumps(q,sort_keys=True,separators=(',',':'))


@pytest.mark.parametrize("raw",[
    '{"function_calls":[{"name":"invented","arguments":{}}]}',
    '{"function_calls":[{"name":"set_brightness","arguments":{"value":true}}]}',
    '{"function_calls":[{"name":"set_brightness","arguments":{"value":101}}]}',
    '{"function_calls":[{"name":"set_brightness","arguments":{"value":"30"}}]}',
    '{"function_calls":[{"name":"get_weather","arguments":{}}]}',
    '{"function_calls":[{"name":"get_weather","arguments":{"city":"Lagos","secret":"X"}}]}',
    '{"function_calls":[{"name":"check_flights","arguments":{"cities":["A","B"],"direct":"yes"}}]}',
    '{"function_calls":[{"name":"check_flights","arguments":{"cities":["A"]}}]}',
    '{"function_calls":[{"name":"get_weather","arguments":{"city":"Lagos","city":"Abuja"}}]}',
    '{"function_calls":[],"execute":true}',
    '{"function_calls":[],"function_calls":[]}',
    '{"function_calls":[{"name":"get_weather","arguments":{"city":NaN}}]}',
    '{"function_calls":[{"name":"get_weather","arguments":{"city":"x"}}]}',
    '{"function_calls":null}',
    '{"function_calls":[true]}',
    '{"function_calls":[]',
])
def test_invalid_proposals_fail_closed(raw):
    with pytest.raises(ToolContractError):
        validate_tool_proposal(raw,tools())


def test_invalid_catalogue_schema_and_untrusted_effect_rejected():
    good = [{"name":"foo", "description":"Example",
             "parameters":{"type":"object", "properties":{},
                           "required":[], "additionalProperties":False}}]
    assert catalogue(good)[0].name=="foo"
    for mutation in [
        lambda x: x[0].update(name="bad.tool"),
        lambda x: x[0].update(effect="ignore_acl"),
        lambda x: x[0].update(parameters={"type":"object","properties":{}}),
        lambda x: x[0].update(parameters={"type":"object","properties":{},
                                          "additionalProperties":True}),
        lambda x: x[0].update(parameters={"type":"object","properties":{},
                                          "additionalProperties":False,"$ref":"self"}),
        lambda x: x.append(copy.deepcopy(x[0])),
    ]:
        bad=copy.deepcopy(good); mutation(bad)
        with pytest.raises(ToolContractError):
            catalogue(bad)


def test_prompt_canonical_serialization_and_anti_leakage_training_mask():
    ts=tools()
    labeled=[{"name":"get_weather","arguments":{"city":"Lagos"}}]
    example=prepare_tool_supervision("Weather Lagos",ts,labeled)
    assert len(example.token_ids)==len(example.target_mask)+1
    assert example.target_mask.count(True)==example.output_bytes
    assert not any(example.target_mask[:example.prompt_bytes-1])
    assert all(example.target_mask[example.prompt_bytes-1:])
    prompt=ByteFallbackCodec.decode(example.token_ids[:example.prompt_bytes])
    completion=ByteFallbackCodec.decode(example.token_ids[example.prompt_bytes:])
    assert "Weather Lagos" in prompt and "ASSISTANT_TOOL_JSON:" in prompt
    assert 'get_weather' in completion and '"city":"Lagos"' in completion
    assert 'ASSISTANT_TOOL_JSON:' not in completion
    assert validate_tool_proposal(completion,ts).status=="proposed"


def test_observation_is_explicit_bounded_data_not_an_executed_action():
    ts=tools()
    receipt=tool_observation("get_weather", {"temp": 27, "city":"Lagos"},ts)
    prompt=build_tool_prompt("Check it",ts,[receipt])
    assert '"result":{"city":"Lagos","temp":27}' in prompt
    with pytest.raises(ToolContractError):
        tool_observation("unregistered", {"ok":1},ts)
    with pytest.raises(ToolContractError):
        tool_observation("get_weather","x"*2000,ts)


def test_utf8_byte_codec_roundtrip_and_invalid_byte_rejected():
    text="Go to Jos 🗺️"
    assert ByteFallbackCodec.decode(ByteFallbackCodec.encode(text))==text
    with pytest.raises(ToolContractError):
        ByteFallbackCodec.decode([999])
    with pytest.raises(ToolContractError):
        ByteFallbackCodec.decode([0xFF,0xFE])


@pytest.mark.parametrize("response", [
    '{"function_calls":[]}',
    '{"function_calls":[{"name":"get_weather","arguments":{"city":"Lagos"}}]}',
    '{"function_calls":[{"name":"set_brightness","arguments":{"value":32}}]}',
    '{"function_calls":[{"name":"check_flights","arguments":{"direct":true,"cities":["Jos","Lagos"]}}]}',
])
def test_grammar_has_complete_prefix_path_for_valid_canonical_proposals(response):
    assert response.startswith(_HEADER)
    for i, char in enumerate(response):
        assert ord(char) in allowed_byte_ids(response[:i]), (i,char)
    assert prefix_state(response).complete
    assert allowed_byte_ids(response)==()


@pytest.mark.parametrize("bad", [
    '{"function_calls":[}', '{"function_calls":[1,]}',
    '{"function_calls":[{"name":"foo","arguments":{"x":01}}]}',
    '{"function_calls":[{"name":"foo","arguments":{"x":truz}}]}',
    '{"function_calls":[{"name":"foo","arguments":{"x":"bad\\q"}}]}',
    '{"function_calls":[]"other":123}',
])
def test_syntax_grammar_rejects_malformed_prefixes(bad):
    assert not prefix_state(bad).valid or not prefix_state(bad).complete


def test_grammar_does_not_claim_schema_guarantee():
    # Valid JSON syntax but illegal function_calls shape; strict post-validator rejects it.
    raw='{"function_calls":[true]}'
    assert prefix_state(raw).complete
    with pytest.raises(ToolContractError):
        validate_tool_proposal(raw,tools())


def test_bounded_number_literals_and_tokenizer_fallback():
    for response in ('{"function_calls":[{"name":"x","arguments":{"a":-3.1e2}}]}',
                     '{"function_calls":[{"name":"x","arguments":{"a":null}}]}'):
        assert prefix_state(response).complete
    assert prefix_state('{"function_calls":[{"a":1e').valid
    assert not prefix_state('{"function_calls":[{"a":1e+X').valid
