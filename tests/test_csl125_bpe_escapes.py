"""Regression for upstream bounded-string escape failure; bounds stay enforced."""
import json
import pytest
from tac_osm.csl125_bpe_schema import escaped_schema_ebnf

@pytest.mark.parametrize('text',[
    '"a"','"😀"','"\\n"','"\\\""','"\\\\"','"\\t"','"\\u0000"','"\\u00e9"','"\\ud83d\\ude00"',
])
def test_one_decoded_scalar_with_escape_or_unicode(text):
    import xgrammar as xgr
    grammar=xgr.Grammar.from_ebnf(escaped_schema_ebnf({'type':'string','minLength':1,'maxLength':1}))
    info=xgr.TokenizerInfo([bytes([i]) for i in range(256)],xgr.VocabType.RAW,vocab_size=256,stop_token_ids=[])
    m=xgr.GrammarMatcher(xgr.GrammarCompiler(info,max_threads=1).compile_grammar(grammar),terminate_without_stop_token=True)
    assert m.accept_string(text) and m.is_terminated()
    assert len(json.loads(text))==1

@pytest.mark.parametrize('text',[
    '""','"ab"','"😀x"','"\\nX"','"\\u00e9x"','"\\ud83d\\ude00x"','"\\ud83d"','"\\ude00"','"raw\ttab"','"\\q"',
])
def test_overlength_raw_controls_and_bad_escapes_rejected(text):
    import xgrammar as xgr
    grammar=xgr.Grammar.from_ebnf(escaped_schema_ebnf({'type':'string','minLength':1,'maxLength':1}))
    info=xgr.TokenizerInfo([bytes([i]) for i in range(256)],xgr.VocabType.RAW,vocab_size=256,stop_token_ids=[])
    m=xgr.GrammarMatcher(xgr.GrammarCompiler(info,max_threads=1).compile_grammar(grammar),terminate_without_stop_token=True)
    assert not m.accept_string(text) or not m.is_terminated()

def test_unrecognized_string_constraints_fail_closed():
    from tac_osm.csl125_training_data import DataContractError
    with pytest.raises(DataContractError):escaped_schema_ebnf({'type':'string','pattern':'[a-z]','maxLength':1})
