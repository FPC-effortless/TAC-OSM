"""Build an auditable TRAIN-only source/tokenizer bundle. Neural fit forbidden."""
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
from tac_osm.csl125_training_data import (
    SOURCE_HASHES,SOURCE_REVISION,read_jsonl,digest,api_catalogue,prepare_records,
    render_example,ToolBPE,DataContractError)
from tac_osm.csl125_bpe_schema import EscapedBPEToolGrammar as BPEToolGrammar

def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def jsonl(path,rows):path.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows))
def describe(values):
    s=sorted(values)
    return {'count':len(s),'total':sum(s),'mean':statistics.mean(s),'median':statistics.median(s),
            'p95':s[min(len(s)-1,int(.95*len(s)))],'maximum':max(s)}

def build(out):
    pub=out/'publisher'
    receipt=json.loads((out/'SOURCE_RECEIPT.json').read_text())
    if receipt['revision']!=SOURCE_REVISION:raise DataContractError('unregistered_source_revision')
    for name,expected in SOURCE_HASHES.items():
        if digest((pub/name).read_bytes())!=expected:raise DataContractError('source_hash_mismatch:'+name)
    rows=read_jsonl(pub/'DroidCall_train.jsonl');apis=read_jsonl(pub/'annotated_api.jsonl')
    tools,excluded=api_catalogue(apis)
    records,quarantine,duplicates,audit=prepare_records(rows,tools)
    fit=[r for r in records if r['split']=='FIT'];dev=[r for r in records if r['split']=='DEV']
    assert not set(r['group'] for r in fit)&set(r['group'] for r in dev)
    audit['original_answer_catalogue_exact_rows']=sum(
        {a['name'] for a in r['answers']}=={t['name'] for t in r['tools']} for r in rows)
    audit['original_no_call_rows']=sum(not r['answers'] for r in rows)
    audit['source_readme_license']='publisher declares apache-2.0; not independent legal/outcome certification'
    audit['official_TEST_opened']=False
    save(out/'CATALOGUE.json',tools);save(out/'EXCLUDED_APIS.json',excluded)
    save(out/'DATA_AUDIT.json',audit)
    jsonl(out/'FIT.jsonl',fit);jsonl(out/'DEV.jsonl',dev)
    jsonl(out/'QUARANTINE.jsonl',quarantine);jsonl(out/'DUPLICATES.jsonl',duplicates)
    rendered={r['id']:render_example(r,tools) for r in records}
    results={};failures=[]
    for capacity in (4096,8192,16384):
        codec=ToolBPE.fit(fit,tools,capacity)
        path=out/f'tokenizer_{capacity}.json';codec.save(path)
        grammar=BPEToolGrammar(codec,tools)
        (out/f'grammar_{capacity}.ebnf').write_text(grammar.ebnf)
        lengths={'FIT':{'prompt':[],'response':[],'total':[]},'DEV':{'prompt':[],'response':[],'total':[]}}
        too_long=[];checked=0
        for r in records:
            prompt,label=rendered[r['id']]
            p,y=codec.encode(prompt),codec.encode(label)
            stats=lengths[r['split']]
            stats['prompt'].append(len(p));stats['response'].append(len(y)+1);stats['total'].append(len(p)+len(y)+1)
            if len(p)+len(y)+1>8192:too_long.append(r['id'])
            session=grammar.session()
            try:
                for token in y:session.accept(token)
                proposal=session.validate()
                if list(proposal.function_calls)!=r['calls']:raise DataContractError('gold_roundtrip_changed_arguments')
                checked+=1
            except (ValueError,RuntimeError) as exc:
                failures.append({'capacity':capacity,'id':r['id'],'reason':str(exc)})
        stats={split:{k:describe(v) for k,v in values.items()} for split,values in lengths.items()}
        for split in lengths:
            stats[split]['fraction_over_2048']=sum(x>2048 for x in lengths[split]['total'])/len(lengths[split]['total'])
            stats[split]['fraction_over_4096']=sum(x>4096 for x in lengths[split]['total'])/len(lengths[split]['total'])
        raw_bytes=sum(len(p.encode('utf-8'))+len(y.encode('utf-8')) for p,y in rendered.values())
        import torch
        from tac_osm.csl125_r3 import CSL125R3Config
        from tac_osm.csl125_tool_model import CSL125R4
        with torch.device('meta'):
            meta=CSL125R4(CSL125R3Config(vocab_size=codec.vocab_size),device='meta')
        parameters=meta.parameter_count()
        assert parameters==98_707_439+codec.vocab_size*768 and parameters<=125_000_000
        results[str(capacity)]={'requested_capacity':capacity,'actual_vocab_size':codec.vocab_size,
            'tokenizer_sha256':digest(path.read_bytes()),'tokenizer_file_bytes':path.stat().st_size,
            'fit_record_count':len(fit),'FIT_ids_hash':digest(sorted(r['id'] for r in fit)),
            'DEV_used_in_tokenizer_fit':False,'statistics':stats,'oversized_no_truncation':too_long,
            'exact_roundtrip_and_gold_grammar_accepted':checked,'source_rows_checked':len(records),
            'all_record_raw_bytes':raw_bytes,'all_record_BPE_tokens_with_EOS':sum(lengths['FIT']['total'])+sum(lengths['DEV']['total']),
            'full_20L_model_parameters_meta_verified':parameters,'grammar_fingerprint':grammar.fingerprint}
        save(out/'TOKENIZER_RESULTS.json',results);save(out/'GRAMMAR_FAILURES.json',failures)
        print('TOKENIZER_RESULT',capacity,json.dumps(results[str(capacity)],sort_keys=True),flush=True)
    status={'stage':'DATA_TOKENIZER_PREPARATION_COMPLETE' if not failures else 'COMPLETE_WITH_GRAMMAR_FAILURES',
            'neural_optimizer_steps':0,'neural_checkpoint_trained':False,'training_admitted':False,
            'source_revision':SOURCE_REVISION,'synthetic_data':True,'official_TEST_opened':False,
            'blocked_by':['no authored no-call coverage in this source','no observed execution outcome labels',
                         'independent signed source/evaluator/ACL evidence not supplied',
                         'complete matched-baseline FAIR_COMPARISON preflight not supplied',
                         'GPU allocation and CUDA precision qualification not supplied'],
            'not_blocked_by':['removed 100K ceiling','unexplained inherited gradient failure'],
            'grammar_failures':len(failures),'accepted_rows':len(records),'FIT_rows':len(fit),'DEV_rows':len(dev)}
    save(out/'TRAINING_ADMISSION.json',status)
    print('DATA_AUDIT',json.dumps(audit,sort_keys=True),flush=True)
    print('TRAINING_ADMISSION',json.dumps(status,sort_keys=True),flush=True)
    if failures:raise RuntimeError('gold grammar bridge failed; retained all failures')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='ci-artifacts/csl125-training-data');a=parser.parse_args()
    build(Path(a.out))
