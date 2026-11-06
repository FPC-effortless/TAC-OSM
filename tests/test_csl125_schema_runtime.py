"""Pipeline verification only: fixtures and random weights are not learned skill."""
import copy
import json
import pytest
import torch
from jsonschema import Draft202012Validator

from tac_osm.csl125_tool_protocol import catalogue, ToolContractError, ByteFallbackCodec
from tac_osm.csl125_tool_schema import CompiledToolSchema, prepare_schema_supervision, schema_target
from tac_osm.csl125_tool_runtime import last_features, legal_byte_logits, propose_schema_tools
from tac_osm.csl125_r3 import CSL125R3Config
from tac_osm.csl125_tool_model import CSL125R4


def tools():
    return catalogue([
        {'name': 'weather', 'description': 'Weather for a city', 'parameters': {
            'type': 'object', 'properties': {
                'city': {'type': 'string', 'minLength': 2, 'maxLength': 40},
                'units': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}},
            'required': ['city'], 'additionalProperties': False}},
        {'name': 'brightness', 'description': 'Set brightness', 'effect': 'write', 'parameters': {
            'type': 'object', 'properties': {'value': {'type': 'integer', 'minimum': 0, 'maximum': 100}},
            'required': ['value'], 'additionalProperties': False}},
    ])


@pytest.fixture(scope='module')
def compiled():
    return CompiledToolSchema(tools())


def feed(compiled, raw):
    session = compiled.session()
    for token in raw.encode():
        allowed = session.allowed()
        if token not in allowed.tolist():
            return False
        session.accept(token)
    return session.complete


@pytest.mark.parametrize('calls', [
    [], [{'name': 'weather', 'arguments': {'city': 'Jos'}}],
    [{'name': 'weather', 'arguments': {'city': 'Lagos', 'units': 'celsius'}}],
    [{'name': 'brightness', 'arguments': {'value': 0}}],
    [{'name': 'brightness', 'arguments': {'value': 100}}],
    [{'name': 'weather', 'arguments': {'city': 'Jos'}}, {'name': 'brightness', 'arguments': {'value': 20}}],
])
def test_valid_target_every_byte_is_permitted(compiled, calls):
    raw = schema_target(calls, tools())
    assert feed(compiled, raw)
    Draft202012Validator(compiled.schema).validate(json.loads(raw))


@pytest.mark.parametrize('raw', [
    '{"function_calls":[{"name":"invented","arguments":{}}]}',
    '{"function_calls":[{"name":"weather","arguments":{}}]}',
    '{"function_calls":[{"name":"weather","arguments":{"city":"Jos","secret":"x"}}]}',
    '{"function_calls":[{"name":"weather","arguments":{"city":"Jos","city":"Kano"}}]}',
    '{"function_calls":[{"name":"weather","arguments":{"city":7}}]}',
    '{"function_calls":[{"name":"weather","arguments":{"city":"Jos","units":"kelvin"}}]}',
    '{"function_calls":[{"name":"brightness","arguments":{"value":true}}]}',
    '{"function_calls":[{"name":"brightness","arguments":{"value":"30"}}]}',
    '{"function_calls":[{"name":"brightness","arguments":{"value":101}}]}',
    '{"function_calls":[{"name":"brightness","arguments":{"value":-1}}]}',
    '{"function_calls":[],"execute":true}',
    '{"function_calls":[true]}',
])
def test_schema_invalid_proposal_is_not_decodable(compiled, raw):
    assert not feed(compiled, raw)


def test_catalogue_snapshot_and_matcher_session_isolation():
    original = tools()
    g = CompiledToolSchema(original)
    original[0].parameters['properties']['city']['type'] = 'integer'
    copy_out = g.tools
    copy_out[0].parameters['properties']['city']['type'] = 'boolean'
    assert g.tools[0].parameters['properties']['city']['type'] == 'string'
    a, b = g.session(), g.session()
    a.accept(ord('{'))
    assert bytes(a.bytes) == b'{' and bytes(b.bytes) == b''
    with pytest.raises(ToolContractError):
        b.accept(999)


def test_call_limit_and_empty_call_are_separate_from_model_ability(compiled):
    call = {'name': 'brightness', 'arguments': {'value': 10}}
    four = schema_target([call] * 4, tools())
    assert feed(compiled, four)
    five = json.dumps({'function_calls': [call] * 5}, separators=(',', ':'))
    assert not feed(compiled, five)
    none_only = CompiledToolSchema(tools(), max_calls=0)
    assert feed(none_only, '{"function_calls":[]}')
    assert not feed(none_only, schema_target([call], tools()))


def test_supervision_matches_grammar_and_no_label_in_prompt(compiled):
    calls = [{'name': 'weather', 'arguments': {'city': 'Jos'}}]
    item = prepare_schema_supervision('Weather in Jos', tools(), calls)
    prompt = ByteFallbackCodec.decode(item.token_ids[:item.prompt_bytes])
    target = ByteFallbackCodec.decode(item.token_ids[item.prompt_bytes:])
    assert target not in prompt and feed(compiled, target)
    assert target.index('name') < target.index('arguments')
    assert len(item.token_ids) == len(item.target_mask) + 1
    assert sum(item.target_mask) == item.output_bytes
    assert not any(item.target_mask[:item.prompt_bytes-1])


def small():
    return CSL125R3Config(vocab_size=512, width=32, ff_width=64, layers=4,
        role_count=4, state_width=8, rungs=(2,4), attention_layers=(2,4),
        query_heads=4, kv_heads=2, attention_window=16, correction_rank=4)


@pytest.mark.parametrize('depth', [2, 4])
def test_legal_row_projection_and_state_equal_parent(depth):
    torch.manual_seed(26)
    model = CSL125R4(small()).eval()
    tokens = torch.randint(0,256,(1,24))
    with torch.inference_mode():
        full, ref = model(tokens, depth=depth, stream_ids=('schema-request',))
        z, got = last_features(model, tokens, depth=depth)
        ids = torch.tensor([32, 34, 48, 97, 123, 255], dtype=torch.long)
        scores = legal_byte_logits(model, z, ids)
        torch.testing.assert_close(scores, full[depth]['token_logits'][0,-1,ids], atol=3e-5, rtol=3e-5)
        for a, b in zip(ref.layers, got.layers):
            for field in ('fast', 'slow', 'fast_mass', 'slow_mass', 'conv_tail'):
                assert torch.equal(getattr(a, field), getattr(b, field))
        state = None
        for begin, end in ((0,7),(7,13),(13,24)):
            zz, state = last_features(model, tokens[:,begin:end], depth=depth, state=state)
        torch.testing.assert_close(zz, z, atol=3e-5, rtol=3e-5)
        with pytest.raises(ToolContractError):
            last_features(model, tokens[:,:1], depth=depth, state=state, stream_id='different')


def test_runtime_random_weights_only_and_fail_closed(compiled):
    torch.manual_seed(27)
    model = CSL125R4(small()).eval()
    r = propose_schema_tools(model, 'Weather in Jos', compiled, depth=2, max_new_bytes=4)
    assert r.proposal.status == 'incomplete'
    assert r.generated_bytes == 4 and r.trained_quality == 'NOT_MEASURED'
    assert r.grammar_fingerprint == compiled.fingerprint
    assert not hasattr(r, 'execute')
    with pytest.raises(ToolContractError):
        propose_schema_tools(model.train(), 'x', compiled)


def test_no_added_model_parameters():
    model = CSL125R4(device='meta')
    assert model.parameter_count() == 123_873_263
    assert model.parameter_count() < 125_000_000
