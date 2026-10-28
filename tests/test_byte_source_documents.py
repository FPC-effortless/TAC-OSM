"""Synthetic boundary fixtures; source admission is separately replayed."""
from dataclasses import fields
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import pytest
from tac_osm.byte_source_documents import ByteDocument,ByteObservation,observe_at,next_target,encode_bytes,decode_bytes


def test_all256_bytes_roundtrip_and_no_unknown_symbol():
    raw=bytes(range(256))
    assert encode_bytes(raw)==tuple(range(256))
    assert decode_bytes(encode_bytes(raw))==raw


def test_prediction_input_excludes_future_and_target_field():
    d=ByteDocument('Doc/tutorial/a.rst',b'abcdef')
    o=observe_at(d,2,history=2)
    assert o.tokens==(98,99) and next_target(d,2)==100
    assert o.first_position==1 and o.last_position==2
    assert 'target' not in {f.name for f in fields(ByteObservation)}
    assert observe_at(ByteDocument(d.owner,b'abcXYZ'),2,history=2)==o
    assert next_target(ByteDocument(d.owner,b'abcXYZ'),2)==88


def test_owner_boundaries_are_explicit_not_model_state_proof():
    a=observe_at(ByteDocument('a.rst',b'abc'),0)
    b=observe_at(ByteDocument('b.rst',b'abc'),0)
    assert a.tokens==b.tokens and a.owner!=b.owner


@pytest.mark.parametrize('position',[-1,2,True,1.5])
def test_invalid_prediction_position(position):
    with pytest.raises(ValueError):observe_at(ByteDocument('a.rst',b'abc'),position)


@pytest.mark.parametrize('owner',['../a','/a','a//b','a\\b'])
def test_invalid_owner(owner):
    with pytest.raises(ValueError):ByteDocument(owner,b'ab')


@pytest.mark.parametrize('tokens',[(256,),(-1,),(True,),(1.5,)])
def test_invalid_codec_values(tokens):
    with pytest.raises(ValueError):decode_bytes(tokens)
