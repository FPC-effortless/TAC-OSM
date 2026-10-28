#!/usr/bin/env python3
"""Original publisher source/codec admission; no model fit or prediction score."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tac_osm.byte_source_documents import ByteDocument,observe_at,next_target,encode_bytes,decode_bytes
from tac_osm.replay_intake import safe_file
from scripts.validate_research_parity import check_manifest

STUDY=ROOT/'research/mathematical-engineering/language_source023'
BUNDLE=ROOT/'research/source-bundles/LANGUAGE-SOURCE023'


def sha(raw):return hashlib.sha256(raw).hexdigest()


def git_blob(raw):return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()


def audit(bundle=BUNDLE,*,refetch=False):
    spec=json.loads((STUDY/'FROZEN_SOURCE.json').read_text())
    receipt=json.loads((STUDY/'SOURCE_RECEIPT.json').read_text())
    manifest=json.loads((STUDY/'FAIR_COMPARISON.json').read_text())
    check_manifest(manifest,'preflight')
    if sha((STUDY/'FROZEN_SOURCE.json').read_bytes())!=manifest['source_selection_sha256']:
        raise ValueError('preregistered source selection changed')
    if len(spec['files'])!=17 or len(receipt['files'])!=17 or spec['implementation_commit']!=receipt['commit']:
        raise ValueError('publisher file coverage/commit changed')
    whole=hashlib.sha256();rows=[];lines={'FIT':set(),'DEV':set()}
    for index,(frozen,row) in enumerate(zip(spec['files'],receipt['files'])):
        if frozen!={k:v for k,v in row.items() if k!='sha256'} or row['role']!=('DEV' if index%4==3 else 'FIT'):
            raise ValueError('original identities or role assignment changed')
        raw=safe_file(bundle,row['path']).read_bytes()
        if sha(raw)!=row['sha256'] or git_blob(raw)!=row['git_blob'] or len(raw)!=row['bytes']:
            raise ValueError('publisher original bytes differ: '+row['path'])
        if refetch:
            url='https://raw.githubusercontent.com/python/cpython/'+spec['implementation_commit']+'/'+row['path']
            with urllib.request.urlopen(url,timeout=30) as response:published=response.read()
            if published!=raw:raise ValueError('fresh publisher bytes differ')
        text=raw.decode('utf-8',errors='strict')
        if decode_bytes(encode_bytes(raw))!=raw:raise ValueError('byte256 codec changed original data')
        document=ByteDocument(row['path'],raw)
        cutoffs=sorted({0,min(127,len(raw)-2),(len(raw)-2)//2,len(raw)-2})
        controls=[]
        for t in cutoffs:
            observed=observe_at(document,t)
            poisoned=ByteDocument(row['path'],raw[:t+1]+bytes(x^255 for x in raw[t+1:]))
            visible=ByteDocument(row['path'],raw[:t]+bytes([raw[t]^255])+raw[t+1:])
            if observe_at(poisoned,t)!=observed or next_target(poisoned,t)==next_target(document,t):
                raise ValueError('future-only negative control failed')
            if observe_at(visible,t)==observed:raise ValueError('source-visible positive control inert')
            if decode_bytes(observed.tokens)!=raw[observed.first_position:t+1]:
                raise ValueError('observation does not match actual original prefix')
            controls.append({'position':t,'prefix_sha256':sha(decode_bytes(observed.tokens)),
                'future_source_changed':True,'future_target_changed':True,'feature_prefix_invariant':True,
                'current_byte_positive_changes_features':True})
        lines[row['role']].update(line.strip() for line in raw.splitlines() if line.strip())
        rows.append(dict(path=row['path'],role=row['role'],sha256=row['sha256'],git_blob=row['git_blob'],
            original_bytes=len(raw),unicode_codepoints=len(text),nonascii_bytes=sum(x>=128 for x in raw),
            distinct_byte_values=len(set(raw)),utf8='PASS_STRICT',byte256_roundtrip='PASS_EXACT',
            historical_map_changed_bytes=sum(x>=128 for x in raw),one_step_positions=len(raw)-1,
            controls=controls))
        whole.update(raw)
    license_raw=safe_file(bundle,'LICENSE').read_bytes()
    if sha(license_raw)!=receipt['license_sha256'] or git_blob(license_raw)!=spec['license']['git_blob']:
        raise ValueError('publisher license bytes changed')
    if refetch:
        with urllib.request.urlopen('https://raw.githubusercontent.com/python/cpython/'+spec['implementation_commit']+'/LICENSE',timeout=30) as response:
            if response.read()!=license_raw:raise ValueError('fresh publisher license differs')
    if whole.hexdigest()!=receipt['source_sha256'] or whole.hexdigest()!=manifest['real_world_source']['sha256']:
        raise ValueError('actual source-byte sequence hash differs')
    totals={role:{key:sum(row[key] for row in rows if row['role']==role)
                  for key in ('original_bytes','nonascii_bytes','historical_map_changed_bytes','one_step_positions')}
            for role in ('FIT','DEV')}
    return dict(study_id=spec['study_id'],evidence='INTERNAL_SOURCE_ADMISSION_NOT_MODEL_CAPABILITY',
        publisher_commit=spec['implementation_commit'],source_sha256=whole.hexdigest(),
        source_sha256_scope=receipt['source_sha256_scope'],license_sha256=receipt['license_sha256'],
        fresh_publisher_refetch_verified=refetch,files=rows,totals=totals,
        duplicate_whole_documents=len(rows)-len({r['sha256'] for r in rows}),
        shared_nonblank_line_types=len(lines['FIT']&lines['DEV']),
        source_prefix_controls=sum(len(r['controls']) for r in rows),
        controls_are_synthetic_not_authentic_outcomes=True,
        model_state_AD_isolation='INCONCLUSIVE_NOT_INTEGRATED',
        training='NOT_RUN',predictive_NLL='NOT_RUN',protected_TEST=False,independently_qualified=False)


def audit_supplement(bundle=BUNDLE,*,refetch=False):
    row=json.loads((STUDY/'UTF8_SUPPLEMENT_RECEIPT.json').read_text())
    m=json.loads((ROOT/'research/mathematical-engineering/language_source023b/FAIR_COMPARISON.json').read_text())
    check_manifest(m,'preflight')
    if (row['path']!='Doc/howto/unicode.rst' or row['role']!='FIT_SUPPLEMENT'
        or row['publisher_commit']!='60403a5409ff2c3f3b07dd2ca91a7a3e096839c7'):
        raise ValueError('separately preregistered supplement identity differs')
    raw=safe_file(bundle,row['path']).read_bytes()
    if sha(raw)!=row['sha256'] or sha(raw)!=m['real_world_source']['sha256'] or git_blob(raw)!=row['git_blob'] or len(raw)!=row['bytes']:
        raise ValueError('publisher supplement bytes differ')
    license_raw=safe_file(bundle,'LICENSE').read_bytes()
    receipt=json.loads((STUDY/'SOURCE_RECEIPT.json').read_text())
    if sha(license_raw)!=receipt['license_sha256']:raise ValueError('publisher license differs')
    if refetch:
        base='https://raw.githubusercontent.com/python/cpython/'+row['publisher_commit']+'/'
        for name,expected in ((row['path'],raw),('LICENSE',license_raw)):
            with urllib.request.urlopen(base+name,timeout=30) as response:
                if response.read()!=expected:raise ValueError('fresh publisher supplement/license differs')
    text=raw.decode('utf-8',errors='strict')
    if decode_bytes(encode_bytes(raw))!=raw:raise ValueError('supplement codec roundtrip differs')
    d=ByteDocument(row['path'],raw);controls=[]
    for t in sorted({0,min(127,len(raw)-2),(len(raw)-2)//2,len(raw)-2}):
        o=observe_at(d,t)
        future=ByteDocument(d.owner,raw[:t+1]+bytes(x^255 for x in raw[t+1:]))
        visible=ByteDocument(d.owner,raw[:t]+bytes([raw[t]^255])+raw[t+1:])
        if observe_at(future,t)!=o or next_target(future,t)==next_target(d,t) or observe_at(visible,t)==o:
            raise ValueError('supplement boundary controls failed')
        controls.append(dict(position=t,prefix_sha256=sha(decode_bytes(o.tokens)),feature_prefix_invariant=True,
            future_target_changed=True,current_byte_positive_changes_features=True))
    return dict(study_id=row['study_id'],path=row['path'],role=row['role'],publisher_commit=row['publisher_commit'],
        sha256=row['sha256'],git_blob=row['git_blob'],license_sha256=receipt['license_sha256'],
        original_bytes=len(raw),unicode_codepoints=len(text),nonascii_bytes=sum(x>=128 for x in raw),
        distinct_byte_values=len(set(raw)),one_step_positions=len(raw)-1,utf8='PASS_STRICT',byte256_roundtrip='PASS_EXACT',
        fresh_publisher_refetch_verified=refetch,controls=controls,controls_are_synthetic_not_authentic_outcomes=True,
        original17_document_split_unchanged=True,training='NOT_RUN',predictive_NLL='NOT_RUN',
        model_state_AD_isolation='INCONCLUSIVE_NOT_INTEGRATED',protected_TEST=False,independently_qualified=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--refetch',action='store_true');p.add_argument('--supplement',action='store_true');args=p.parse_args()
    report=(audit_supplement if args.supplement else audit)(refetch=args.refetch)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('study_id','totals','source_prefix_controls','shared_nonblank_line_types','original_bytes','nonascii_bytes','distinct_byte_values','predictive_NLL') if k in report}))


if __name__=='__main__':main()
