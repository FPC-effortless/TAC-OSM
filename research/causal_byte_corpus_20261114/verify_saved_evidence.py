"""Stdlib verification of saved corpus evidence; never imports/calls a model."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[2]
ZIP_PIN='9054ef6516a09498f873cf9e26a2239b604f392c91f30c74063603d5e4ede8cf'


def check(condition,message):
    if not condition:raise ValueError(message)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def main():
    study=ROOT/'research/causal_byte_corpus_20261114'
    protocol=json.loads((study/'PROTOCOL.json').read_text())
    manifest_raw=(study/'CORPUS.json').read_bytes()
    check(sha(manifest_raw)==protocol['corpus_manifest_sha256'],'manifest pin')
    manifest=json.loads(manifest_raw);raw_docs={}
    for entry in [manifest['license'],*manifest['files']]:
        raw=(ROOT/'research/source-bundles/LANGUAGE-SOURCE023'/entry['path']).read_bytes()
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        check(len(raw)==entry['bytes'] and sha(raw)==entry['sha256'] and blob==entry['git_blob'],'original file '+entry['path'])
        if 'role' in entry:raw_docs[entry['path']]=(entry,raw)
    archive=ROOT/'research/evidence/causal_byte_corpus_20261114/HOSTED_AUDIT.zip'
    check(sha(archive.read_bytes())==ZIP_PIN,'original ZIP pin')
    with ZipFile(archive) as z:
        result_raw=z.read('audit/RESULTS.json');result=json.loads(result_raw)
        check(result_raw==(study/'LOCAL_RESULTS.json').read_bytes(),'local/hosted exact receipt')
        for path,pin in result['source_code_sha256'].items():
            check(sha((ROOT/path).read_bytes())==pin,'unchanged audit source '+path)
        raw_ledger=z.read('audit/CHUNKS.jsonl')
        check(sha(raw_ledger)==result['chunk_ledger_sha256'],'ledger pin')
        rows=list(map(json.loads,raw_ledger.splitlines()));groups=defaultdict(list)
        for r in rows:
            check(r['chunk_length'] in (128,512) and r['role'] in ('FIT','DEV'),'registered cases only')
            entry,raw=raw_docs[r['document']]
            check(r['role']==('DEV' if entry['role']=='DEV' else 'FIT'),'frozen role')
            a,b=r['input_begin'],r['input_end']
            check(0<=a<b<=len(raw)-1 and b-a<=r['chunk_length'],'input bounds')
            check((r['target_begin'],r['target_end'])==(a+1,b+1),'shifted observed target')
            check(r['reset_state']==(a==0),'reset exactly at document start')
            check(r['source_sha256']==entry['sha256'] and r['input_sha256']==sha(raw[a:b]) and r['target_sha256']==sha(raw[a+1:b+1]),'saved original byte identities')
            groups[(r['chunk_length'],r['role'],r['document'])].append(r)
        check(len(groups)==36 and len(rows)==2905,'all18documents x2chunk lengths')
        for (_,_,name),members in groups.items():
            check(members[0]['input_begin']==0,'first input offset')
            check(all(x['input_end']==y['input_begin'] for x,y in zip(members,members[1:])),'no gaps/repeated targets')
            check(members[-1]['input_end']==len(raw_docs[name][1])-1,'full original denominator')
        check(len(result['cases'])==4,'complete registered case matrix')
        for case in result['cases']:
            length,role=case['chunk_length'],case['role']
            chosen=[r for r in rows if (r['chunk_length'],r['role'])==(length,role)]
            selected=[(e,b) for e,b in raw_docs.values() if ('DEV' if e['role']=='DEV' else 'FIT')==role]
            check(len(chosen)==case['chunks'] and sum(r['input_end']-r['input_begin'] for r in chosen)==case['within_document_next_byte_pairs'],'counts')
            check(sum(r['reset_state'] for r in chosen)==case['document_state_resets']==len(selected)==case['documents'],'document resets')
            check(sum(len(b) for _,b in selected)==case['source_bytes'] and sum(v>=128 for _,b in selected for v in b[1:])==case['nonascii_targets'],'source/rare byte denominators')
        suites=ET.fromstring(z.read('TESTS.xml')).findall('testsuite')
        check(sum(int(s.attrib['tests']) for s in suites)==51,'software test count')
        check(all(int(s.attrib[k])==0 for s in suites for k in ('failures','errors','skipped')),'software outcomes')
        check(z.read('CPU_PLAN.json')==(study/'COMPUTE_DECISION.json').read_bytes(),'original CPU plan')
        check(not result['source_synthetic'] and not result['independently_qualified'] and not result['training_authorized'] and not result['protected_TEST_opened'],'scope')
        check(result['model_executions']==result['optimizer_updates']==result['tokenizer_fit_steps']==result['GPU_minutes']==0,'no model/fit/GPU')
    print('PASS: original corpus/license/source pins, saved2905chunks/full pair coverage, 51tests and exact local/hosted receipt; no model or adapter execution')


if __name__=='__main__':main()
