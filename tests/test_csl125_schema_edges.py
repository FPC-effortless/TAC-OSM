"""Additional schema edge fixtures, never a neural accuracy benchmark."""
import json
import pytest
from jsonschema import Draft202012Validator
from tac_osm.csl125_tool_protocol import catalogue
from tac_osm.csl125_tool_schema import CompiledToolSchema, schema_target


def bounded_tools():
    return catalogue([{'name':'configure','description':'Configure a bounded device record',
        'parameters':{'type':'object','properties':{
            'label':{'type':'string','minLength':2,'maxLength':8},
            'enabled':{'type':'boolean'},
            'offset':{'type':'number','minimum':-4,'maximum':4},
            'ports':{'type':'array','minItems':1,'maxItems':2,'items':{'type':'integer','minimum':0,'maximum':10}},
            'optional_null':{'type':'null'},
            'nested':{'type':'object','properties':{'unit':{'type':'string','enum':['C','F']}},
                      'required':['unit'],'additionalProperties':False}},
            'required':['label'],'additionalProperties':False}}])


@pytest.fixture(scope='module')
def grammar():
    return CompiledToolSchema(bounded_tools())


def accepts(g, text):
    s=g.session()
    for token in text.encode('utf-8'):
        if token not in s.allowed().tolist():
            return False
        s.accept(token)
    return s.complete


@pytest.mark.parametrize('arguments',[
    {'label':'Jos'}, {'label':'Èkó'}, {'label':'AB😀'},
    {'label':'Lagos','enabled':False,'optional_null':None},
    {'label':'Jos','ports':[0,10],'nested':{'unit':'C'}},
    {'label':'Jos','offset':-0.5}, {'label':'Jos','offset':4.0},
    {'label':'Jos','offset':-4.0}, {'label':'Jos','offset':0.0},
])
def test_valid_nested_and_unicode_targets(grammar, arguments):
    text=schema_target([{'name':'configure','arguments':arguments}],bounded_tools())
    Draft202012Validator(grammar.schema).validate(json.loads(text))
    assert accepts(grammar,text)


@pytest.mark.parametrize('arguments',[
    {'label':'X'}, {'label':'123456789'}, {'label':'Jos','enabled':1},
    {'label':'Jos','ports':[]}, {'label':'Jos','ports':[1,2,3]},
    {'label':'Jos','ports':[11]}, {'label':'Jos','nested':{}},
    {'label':'Jos','offset':4.5}, {'label':'Jos','offset':-4.5},
    {'label':'Jos','optional_null':False},
])
def test_invalid_value_constraints_rejected(grammar,arguments):
    ordered={k:arguments[k] for k in sorted(arguments)}
    text=json.dumps({'function_calls':[{'name':'configure','arguments':ordered}]},ensure_ascii=True,separators=(',',':'))
    assert not accepts(grammar,text)
