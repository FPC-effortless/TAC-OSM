"""Bounded JSON strings with escapes for the fitted BPE path.

Opt-in adapter around XGrammar 0.2.8, whose bounded-string converter dropped
JSON escapes in our retained run42264533850. Do not remove length bounds.
Preserve all other schema conversion and final independent validation.
"""
from __future__ import annotations
import copy,json
from tac_osm.csl125_training_data import ToolBPE,BPEToolSession,DataContractError,digest

# One production consumes ONE decoded Unicode scalar, even when escaped.
# Isolated UTF-16 surrogates and raw JSON control characters are rejected.
_CHARACTER_RULES = r'''
csl_json_scalar ::= [^"\\\x00-\x1f\uD800-\uDFFF] | "\\" csl_json_escape
csl_json_escape ::= ["\\/bfnrt] | "u" csl_json_unicode
csl_json_unicode ::= [0-9a-cA-Ce-fE-F] [0-9a-fA-F]{3} | [dD] [0-7] [0-9a-fA-F]{2} | [dD] [89aAbB] [0-9a-fA-F]{2} "\\u" [dD] [c-fC-F] [0-9a-fA-F]{2}
'''

def escaped_schema_ebnf(schema):
    """Replace only unconstrained-string VALUE productions with bounded atoms.

    A unique sentinel const obtains the exact surrounding schema grammar.
    Its quoted literal is replaced with quote + scalar{min,max} + quote.
    No schema keyword is discarded: string enums/consts remain untouched;
    object fields, numeric bounds, array sizes and alternatives use upstream.
    Fails closed if upstream printing changes or a new keyword appears.
    """
    import xgrammar as xgr
    original=json.dumps(schema,sort_keys=True)
    replacements={}
    def visit(node):
        if isinstance(node,list):return [visit(x) for x in node]
        if not isinstance(node,dict):return node
        if node.get('type')=='string' and 'enum' not in node and 'const' not in node:
            if set(node)-{'type','description','minLength','maxLength'}:
                raise DataContractError('unsupported_string_constraint_for_escape_adapter')
            lo,hi=node.get('minLength',0),node.get('maxLength',256)
            if type(lo)is not int or type(hi)is not int or not 0<=lo<=hi<=512:
                raise DataContractError('invalid_decoded_string_bounds')
            marker=f'CSLBOUND_{len(replacements)}_{digest(original)[:20]}'
            if marker in original:raise DataContractError('sentinel_collision')
            replacements[marker]=(lo,hi)
            return {'const':marker}
        return {k:visit(v) for k,v in node.items()}
    transformed=visit(copy.deepcopy(schema))
    ebnf=str(xgr.Grammar.from_json_schema(transformed,any_whitespace=False,
              separators=(',',':'),strict_mode=True,any_order=False))
    if 'csl_json_scalar' in ebnf:raise DataContractError('grammar_symbol_collision')
    quote=json.dumps('"')
    for marker,(lo,hi) in replacements.items():
        literal=json.dumps(json.dumps(marker))
        if literal not in ebnf:
            excerpt=[line for line in ebnf.splitlines() if marker in line]
            raise DataContractError('upstream_sentinel_print_changed:'+repr(excerpt))
        atom=f'{quote} csl_json_scalar{{{lo},{hi}}} {quote}'
        ebnf=ebnf.replace(literal,atom)
    return ebnf+_CHARACTER_RULES

class EscapedBPEToolGrammar:
    """Same fitted BPE and tool contract; repaired escaped-string production."""
    def __init__(self,codec,tools):
        import xgrammar as xgr
        from tokenizers import Tokenizer
        from tac_osm.csl125_tool_schema import snapshot_tools,tool_response_schema
        from tac_osm.csl125_tool_protocol import catalogue
        self.codec=ToolBPE(Tokenizer.from_str(codec.backend.to_str()))
        self.tools=snapshot_tools(catalogue(tools))
        self.size=self.codec.vocab_size
        self.schema=tool_response_schema(self.tools)
        self.ebnf=escaped_schema_ebnf(self.schema)
        vocab=[self.codec.backend.id_to_token(i) for i in range(self.size)]
        info=xgr.TokenizerInfo(vocab,xgr.VocabType.BYTE_LEVEL,vocab_size=self.size,
                              stop_token_ids=[self.codec.eos_id],add_prefix_space=False)
        self.compiled=xgr.GrammarCompiler(info,max_threads=1).compile_grammar(self.ebnf)
        self.xgr=xgr
        self.fingerprint=digest({'tokenizer':self.codec.backend.to_str(),'schema':self.schema,'ebnf':self.ebnf})
    def session(self):return BPEToolSession(self)
