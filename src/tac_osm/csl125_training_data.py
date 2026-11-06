"""TRAIN-only preparation for the existing R4 tool caller, no neural optimizer.

Source labels are synthetic. Fixed supported catalogue is independent of labels.
Unknown contracts, null values, result references and >4 calls are quarantined.
No silently coerced or repaired gold arguments. Exact-group split != semantic OOD.
"""
from __future__ import annotations
import collections
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

PAD = '<|csl_pad|>'
EOS = '<|csl_eos|>'
RESERVED = (PAD, EOS)
SOURCE_REVISION = '42563ae614280d2891d57f1e7057c4bc50dd27bd'
SOURCE_HASHES = {
    'README.md':'08a6c5cfa655e1ecd774a41a9a815e845ad0afd47b289378bc1d6ee66c81dda9',
    'annotated_api.jsonl':'29c4791f7a496e587af74b1a1398864bd6c6de9db767cb97606525d62ec05951',
    'DroidCall_train.jsonl':'4cb2d5691c1b95b0908c59efcb361c8a9c9b12f0a4d182acfc2df0ccc92e6d3b',
}

class DataContractError(ValueError):
    pass

def canonical(obj) -> str:
    return json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)

def digest(obj) -> str:
    raw=obj if isinstance(obj,bytes) else canonical(obj).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()

def no_duplicates(pairs):
    out={}
    for key,value in pairs:
        if key in out:
            raise DataContractError('duplicate_json_key')
        out[key]=value
    return out

def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line,object_pairs_hook=no_duplicates,
                parse_constant=lambda _: (_ for _ in ()).throw(DataContractError('nonfinite_json')))
                for line in f if line.strip()]

def query_group(query: str) -> str:
    if type(query) is not str or not query.strip():
        raise DataContractError('empty_or_nonstring_request')
    return digest(' '.join(unicodedata.normalize('NFKC',query).casefold().split()))

def split_for(group: str) -> str:
    return 'DEV' if int(hashlib.sha256(('CSL125|125|'+group).encode()).hexdigest()[:16],16)%10==0 else 'FIT'

def strings(x):
    if isinstance(x,str):
        yield x
    elif isinstance(x,list):
        for v in x: yield from strings(v)
    elif isinstance(x,dict):
        for v in x.values(): yield from strings(v)

def api_catalogue(apis):
    """Adapt only publisher-declared primitive/list types, no label inference."""
    tools=[]; excluded=[]
    type_map={'str':'string','bool':'boolean','int':'integer'}
    for api in apis:
        try:
            if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}',api['name']):
                raise DataContractError('invalid_tool_name')
            props={}; required=[]
            for key,field in api['arguments'].items():
                typ=field['type']
                if typ.startswith('Optional[') and typ.endswith(']'):
                    typ=typ[9:-1]
                if typ in type_map:
                    schema={'type':type_map[typ]}
                    if typ=='str': schema['maxLength']=512
                elif typ in ('List[str]','list[str]'):
                    schema={'type':'array','items':{'type':'string','maxLength':512},'maxItems':16}
                else:
                    raise DataContractError('unsupported_argument_type:'+field['type'])
                description=field.get('description','')
                if type(description) is not str or len(description)>320:
                    raise DataContractError('description_outside_runtime_contract')
                if description: schema['description']=description
                props[key]=schema
                if field['required'] is True: required.append(key)
            if len(api['description'])>400: raise DataContractError('tool_description_outside_runtime_contract')
            tools.append({'name':api['name'],'description':api['description'],
                          'parameters':{'type':'object','properties':props,'required':required,'additionalProperties':False},
                          'effect':'high_impact'})
        except (KeyError,TypeError,DataContractError) as exc:
            excluded.append({'name':api.get('name'),'source_sha256':digest(api),'reason':str(exc)})
    if not tools or len({t['name'] for t in tools})!=len(tools):
        raise DataContractError('empty_or_duplicate_catalogue')
    return sorted(tools,key=lambda x:x['name']),excluded

def prepare_records(rows,tools):
    from jsonschema import Draft202012Validator
    validators={t['name']:Draft202012Validator(t['parameters']) for t in tools}
    groups=collections.defaultdict(list)
    for index,row in enumerate(rows,1):
        try: group=query_group(row.get('query'))
        except DataContractError: group='invalid:'+str(index)
        groups[group].append((index,row))
    prepared=[]; quarantine=[]; duplicates=[]
    audit=collections.Counter()
    for group,items in sorted(groups.items()):
        labels={digest(r.get('answers')) for _,r in items}
        if len(labels)>1:
            for idx,r in items:
                quarantine.append({'line':idx,'source_sha256':digest(r),'reasons':['conflicting_normalized_request_labels']})
                audit.update(['conflicting_normalized_request_labels'])
            continue
        for idx,row in items[1:]:
            duplicates.append({'line':idx,'source_sha256':digest(row),'kept_line':items[0][0]})
        idx,row=items[0]; reasons=[]
        query=row.get('query'); answers=row.get('answers')
        if type(query) is not str or not query.strip() or len(query.encode())>4096:
            reasons.append('request_outside_runtime_contract')
        if type(answers) is not list or not 0<=len(answers)<=4:
            reasons.append('call_count_outside_runtime_contract')
        calls=[]
        if isinstance(answers,list):
            for a in answers:
                if type(a) is not dict or set(a)-{'id','name','arguments'} or not {'name','arguments'}<=set(a):
                    reasons.append('malformed_source_call');continue
                if a['name'] not in validators:
                    reasons.append('tool_not_in_supported_catalogue');continue
                args=a['arguments']
                if any(re.search(r'#\s*\d+',s) for s in strings(args)):
                    reasons.append('unresolved_result_reference')
                if any(s in text for text in strings(args) for s in RESERVED):
                    reasons.append('reserved_token_literal')
                errors=list(validators[a['name']].iter_errors(args))
                if errors: reasons.append('arguments_violate_declared_runtime_schema')
                calls.append({'name':a['name'],'arguments':args})
        if type(query) is str and any(s in query for s in RESERVED): reasons.append('reserved_token_literal')
        if reasons:
            quarantine.append({'line':idx,'source_sha256':digest(row),'reasons':sorted(set(reasons))})
            audit.update(set(reasons));continue
        prepared.append({'id':f'DroidCall_train:{idx}','source_line':idx,'source_sha256':digest(row),
                         'group':group,'split':split_for(group),'query':query,'calls':calls,
                         'catalogue_sha256':digest(tools)})
    summary={'raw_rows':len(rows),'accepted_rows':len(prepared),'quarantined_rows':len(quarantine),
             'deduplicated_rows':len(duplicates),'exclusion_reason_counts':dict(audit),
             'split_counts':dict(collections.Counter(x['split'] for x in prepared)),
             'accepted_call_count_histogram':dict(collections.Counter(len(x['calls']) for x in prepared)),
             'zero_call_examples':sum(not x['calls'] for x in prepared),
             'all_examples_share_label_independent_catalogue':True,
             'catalogue_size':len(tools),'catalogue_sha256':digest(tools),
             'semantic_template_leakage':'NOT_RESOLVED_BY_EXACT_GROUP_SPLIT',
             'unseen_tool_evaluation':False,'neural_training_authorized':False}
    assert len(rows)==len(prepared)+len(quarantine)+len(duplicates)
    return prepared,quarantine,duplicates,summary

def render_example(record,tools):
    from tac_osm.csl125_tool_protocol import catalogue,build_tool_prompt
    from tac_osm.csl125_tool_schema import schema_target
    if record['catalogue_sha256']!=digest(tools): raise DataContractError('catalogue_hash_mismatch')
    specs=catalogue(tools)
    return build_tool_prompt(record['query'],specs),schema_target(record['calls'],specs)

@dataclass(frozen=True)
class EncodedExample:
    ids: tuple[int,...]
    target_mask: tuple[bool,...]
    prompt_tokens: int

class ToolBPE:
    """Fitted byte-level BPE; no unknown-byte loss or implicit normalization."""
    def __init__(self,backend):
        self.backend=backend
        self.pad_id=backend.token_to_id(PAD);self.eos_id=backend.token_to_id(EOS)
        if self.pad_id is None or self.eos_id is None or self.pad_id==self.eos_id:
            raise DataContractError('missing_or_duplicate_control_ids')
    @property
    def vocab_size(self): return self.backend.get_vocab_size()
    @classmethod
    def fit(cls,records,tools,vocab_size):
        from tokenizers import Tokenizer,models,pre_tokenizers,decoders,trainers
        if not records or any(r['split']!='FIT' for r in records):
            raise DataContractError('tokenizer_fit_requires_nonempty_FIT_only')
        if type(vocab_size) is not int or not 258<=vocab_size<=32768:
            raise DataContractError('tokenizer_capacity_outside_registered_range')
        tok=Tokenizer(models.BPE())
        tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder=decoders.ByteLevel()
        trainer=trainers.BpeTrainer(vocab_size=vocab_size,min_frequency=2,show_progress=False,
            special_tokens=list(RESERVED),initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        def corpus():
            for r in sorted(records,key=lambda x:x['id']):
                yield from render_example(r,tools)
        tok.train_from_iterator(corpus(),trainer=trainer)
        return cls(tok)
    def encode(self,text):
        if any(s in text for s in RESERVED): raise DataContractError('reserved_token_literal')
        ids=self.backend.encode(text,add_special_tokens=False).ids
        if self.backend.decode(ids,skip_special_tokens=False)!=text:
            raise DataContractError('tokenizer_roundtrip_failure')
        return ids
    def supervised(self,prompt,target,max_tokens=8192):
        p,y=self.encode(prompt),self.encode(target)
        ids=p+y+[self.eos_id]
        if len(p)<2 or not y or len(ids)>max_tokens: raise DataContractError('context_limit_no_truncation')
        return EncodedExample(tuple(ids),tuple([False]*(len(p)-1)+[True]*(len(y)+1)),len(p))
    def save(self,path): self.backend.save(str(path))
    @classmethod
    def load(cls,path):
        from tokenizers import Tokenizer
        return cls(Tokenizer.from_file(str(path)))

def collate(examples,pad_id):
    import torch
    if not examples: raise DataContractError('empty_batch')
    lengths=torch.tensor([len(e.ids) for e in examples],dtype=torch.long)
    tokens=torch.full((len(examples),int(lengths.max())),pad_id,dtype=torch.long)
    mask=torch.zeros((len(examples),tokens.shape[1]-1),dtype=torch.bool)
    for i,e in enumerate(examples):
        if len(e.target_mask)!=len(e.ids)-1: raise DataContractError('target_alignment')
        tokens[i,:len(e.ids)]=torch.tensor(e.ids)
        mask[i,:len(e.target_mask)]=torch.tensor(e.target_mask)
    return tokens,mask,lengths

def padded_tool_loss(model,tokens,mask,lengths):
    """Right-padding-aware all-rung SFT loss. No packing or state reuse."""
    import torch
    from torch.nn import functional as F
    if tokens.dtype!=torch.long or mask.dtype!=torch.bool or tokens.ndim!=2 or mask.shape!=(tokens.shape[0],tokens.shape[1]-1):
        raise DataContractError('invalid_batch_shape')
    if lengths.shape!=(tokens.shape[0],) or lengths.dtype!=torch.long: raise DataContractError('invalid_lengths')
    for row,n in zip(mask,lengths.tolist()):
        if not 3<=n<=tokens.shape[1] or row[n-1:].any(): raise DataContractError('padded_target_leakage')
        selected=torch.where(row[:n-1])[0]
        if not len(selected) or selected[0]<1 or not row[selected[0]:n-1].all():
            raise DataContractError('noncontiguous_response_target')
    outputs,_=model(tokens,all_rungs=True)
    losses=[F.cross_entropy(outputs[r]['token_logits'][:,:-1][mask].float(),tokens[:,1:][mask])
            for r in model.config.rungs]
    return sum(losses)/len(losses)

class BPEToolGrammar:
    """Use the fitted BPE vocabulary, never interpret BPE IDs as UTF-8 bytes."""
    def __init__(self,codec,tools):
        import xgrammar as xgr
        from tokenizers import Tokenizer
        from tac_osm.csl125_tool_schema import snapshot_tools,tool_response_schema
        from tac_osm.csl125_tool_protocol import catalogue
        self.codec=ToolBPE(Tokenizer.from_str(codec.backend.to_str()))
        self.tools=snapshot_tools(catalogue(tools))
        self.size=self.codec.vocab_size
        vocab=[self.codec.backend.id_to_token(i) for i in range(self.size)]
        info=xgr.TokenizerInfo(vocab,xgr.VocabType.BYTE_LEVEL,
                              vocab_size=self.size,stop_token_ids=[self.codec.eos_id],add_prefix_space=False)
        self.compiled=xgr.GrammarCompiler(info,max_threads=1).compile_json_schema(
            tool_response_schema(self.tools),any_whitespace=False,separators=(',',':'),
            strict_mode=True,any_order=False)
        self.xgr=xgr
        self.fingerprint=digest({'tokenizer':self.codec.backend.to_str(),'schema':tool_response_schema(self.tools)})
    def session(self): return BPEToolSession(self)

class BPEToolSession:
    def __init__(self,grammar):
        self.grammar=grammar
        self.matcher=grammar.xgr.GrammarMatcher(grammar.compiled,terminate_without_stop_token=True)
        self.mask=grammar.xgr.allocate_token_bitmask(1,grammar.size)
        self.ids=[]
    @property
    def complete(self): return self.matcher.is_terminated()
    def allowed(self):
        import torch
        if self.complete: return torch.empty(0,dtype=torch.long)
        self.matcher.fill_next_token_bitmask(self.mask)
        ids=torch.arange(self.grammar.size,dtype=torch.long)
        words=self.mask[0].long()[ids//32]
        valid=((words>>(ids%32))&1).bool()
        valid[self.grammar.codec.pad_id]=False
        valid[self.grammar.codec.eos_id]=False
        return ids[valid]
    def accept(self,token):
        if (type(token)is not int or not 0<=token<self.grammar.size or self.complete or
            token in (self.grammar.codec.pad_id,self.grammar.codec.eos_id) or len(self.ids)>=2048):
            raise DataContractError('invalid_BPE_grammar_token_or_limit')
        if not self.matcher.accept_token(token): raise DataContractError('token_violates_tool_schema')
        self.ids.append(token)
    def validate(self):
        from tac_osm.csl125_tool_protocol import validate_tool_proposal
        if not self.complete: raise DataContractError('incomplete_BPE_schema')
        raw=self.grammar.codec.backend.decode(self.ids,skip_special_tokens=False)
        return validate_tool_proposal(raw,self.grammar.tools)

def propose_bpe_tools(model,user_text,grammar,*,depth=None,max_new_tokens=256,prefill_chunk=128):
    """Bounded model-conditioned BPE generation, proposal-only; not learned proof."""
    import torch
    from torch.nn import functional as F
    from tac_osm.csl125_tool_runtime import last_features
    from tac_osm.csl125_tool_protocol import build_tool_prompt,ToolProposal
    if model.training or model.config.vocab_size!=grammar.size:
        raise DataContractError('eval_model_and_exact_tokenizer_vocabulary_required')
    if type(max_new_tokens)is not int or not 1<=max_new_tokens<=1024:
        raise DataContractError('invalid_generation_token_budget')
    if type(prefill_chunk)is not int or not 1<=prefill_chunk<=512:
        raise DataContractError('invalid_prefill_chunk')
    d=model.config.layers if depth is None else depth
    prompt=grammar.codec.encode(build_tool_prompt(user_text,grammar.tools))
    if len(prompt)+max_new_tokens>8192: raise DataContractError('BPE_context_limit')
    device=model.embedding.weight.device
    if device.type=='meta': raise DataContractError('meta_model_cannot_generate')
    state=None;session=grammar.session()
    with torch.inference_mode():
        for i in range(0,len(prompt),prefill_chunk):
            z,state=last_features(model,torch.tensor([prompt[i:i+prefill_chunk]],device=device),
                                  depth=d,state=state,stream_id='csl-bpe-proposal')
        for _ in range(max_new_tokens):
            allowed=session.allowed().to(device)
            if not allowed.numel(): break
            logits=F.linear(z,model.embedding.weight.index_select(0,allowed))[0,0]
            if not torch.isfinite(logits).all(): raise DataContractError('nonfinite_BPE_logits')
            token=int(allowed[logits.argmax()])
            session.accept(token)
            if session.complete: return session.validate()
            z,state=last_features(model,torch.tensor([[token]],device=device),depth=d,state=state,
                                  stream_id='csl-bpe-proposal')
    return ToolProposal((),'incomplete','bounded BPE decode did not complete',
        grammar.codec.backend.decode(session.ids,skip_special_tokens=False))
