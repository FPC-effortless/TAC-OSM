"""Publisher TRAIN-only source acquisition and inspection. No neural fit."""
from __future__ import annotations
import argparse, collections, hashlib, json, re, urllib.request
from pathlib import Path

ALLOWED = ('README.md','annotated_api.jsonl','DroidCall_train.jsonl')
REPO = 'mllmTeam/DroidCall'

def sha(b):
    return hashlib.sha256(b).hexdigest()

def fetch(url, maximum=32_000_000):
    req=urllib.request.Request(url,headers={'User-Agent':'CSL125-source-audit/1.0'})
    with urllib.request.urlopen(req,timeout=90) as response:
        data=response.read(maximum+1)
    if len(data)>maximum:
        raise ValueError('publisher source exceeds registered byte budget')
    return data

def acquire(out: Path, revision=None):
    raw=out/'publisher'; raw.mkdir(parents=True,exist_ok=True)
    metadata=fetch('https://huggingface.co/api/datasets/'+REPO)
    info=json.loads(metadata)
    if info.get('private') or info.get('gated'):
        raise RuntimeError('source is private/gated; no mirror or implicit terms acceptance')
    revision=revision or info['sha']
    if not re.fullmatch('[0-9a-f]{40}',revision):
        raise ValueError('require immutable full source commit')
    (raw/'hub_metadata.json').write_bytes(metadata)
    receipt={'repo':REPO,'revision':revision,'official_test_opened':False,
             'model_optimizer_steps':0,'synthetic':True,'files':{}}
    for name in ALLOWED:
        url=f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{name}'
        content=fetch(url); (raw/name).write_bytes(content)
        receipt['files'][name]={'url':url,'sha256':sha(content),'bytes':len(content)}
    card=(raw/'README.md').read_text()
    if not re.search(r'license:\s*apache-2\.0',card,re.I):
        raise RuntimeError('publisher license declaration changed; review required')
    receipt['declared_license']='apache-2.0'
    receipt['source_qualification']='PUBLISHER_BYTES_HASHED_NOT_INDEPENDENT_OUTCOME_ATTESTATION'
    (out/'SOURCE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

def inspect(out):
    raw=out/'publisher'
    rows=[json.loads(x) for x in (raw/'DroidCall_train.jsonl').read_text().splitlines() if x.strip()]
    apis=[json.loads(x) for x in (raw/'annotated_api.jsonl').read_text().splitlines() if x.strip()]
    shape=lambda x:{k:type(v).__name__ for k,v in x.items()}
    queries=[str(r.get('query','')).strip().casefold() for r in rows]
    hist=collections.Counter(len(r.get('answers',[])) for r in rows)
    types=collections.Counter()
    for api in apis:
        for arg in api.get('arguments',{}).values():
            types[str(arg.get('type'))]+=1
    summary={'source_rows':len(rows),'catalogue_rows':len(apis),
             'record_shape':shape(rows[0]),'api_shape':shape(apis[0]),
             'answer_count_histogram':dict(hist),
             'unique_normalized_queries':len(set(queries)),
             'duplicate_query_rows':len(queries)-len(set(queries)),
             'argument_type_histogram':dict(types),
             'source_api_example':apis[0],
             'answer_shape':shape(next(a for r in rows for a in r.get('answers',[]))),
             'tool_names':[a.get('name') for a in apis],
             'source_label_metadata_keys':sorted({k for r in rows for a in r.get('answers',[]) for k in a}),
             'neural_optimizer_steps':0,'official_test_opened':False}
    (out/'SOURCE_INSPECTION.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='ci-artifacts/csl125-training-data');p.add_argument('--revision');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    acquire(out,a.revision);inspect(out)
