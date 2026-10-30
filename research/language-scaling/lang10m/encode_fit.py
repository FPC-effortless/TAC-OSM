"""Verify original FIT/tokenizer bytes, encode document-owned arrays for profiling."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm


def encode(manifest,model,out):
    data=json.loads(manifest.read_text())
    if data['partition']!='FIT' or any(r['partition']!='FIT' for r in data['sources']):raise ValueError('FIT only')
    if out.exists():raise FileExistsError('never overwrite token receipts')
    out.mkdir(parents=True)
    tokenizer=spm.SentencePieceProcessor(model_file=str(model))
    if tokenizer.vocab_size()!=4096:raise ValueError('wrong vocabulary')
    arrays={};rows=[]
    for i,r in enumerate(data['sources']):
        p=manifest.parent/r['path'];raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=r['sha256']:raise ValueError('FIT bytes changed')
        text=raw.decode('utf-8',errors='strict');tokens=tokenizer.encode(text)
        if tokenizer.decode(tokens)!=text:raise ValueError('non-lossless FIT encoding')
        if any(t<0 or t>=4096 for t in tokens):raise ValueError('invalid token ID')
        key=f'doc_{i:03d}';arrays[key]=np.asarray(tokens,dtype=np.uint16)
        rows.append({'array':key,'source':r['path'],'modality':r['modality'],
            'source_sha256':r['sha256'],'tokens':len(tokens),'utf8_bytes':len(raw)})
    np.savez_compressed(out/'FIT_TOKENS.npz',**arrays)
    report={'variant':'LANG10M001','partition':'FIT','files':rows,'vocabulary':4096,
      'total_tokens':sum(r['tokens'] for r in rows),'total_utf8_bytes':sum(r['utf8_bytes'] for r in rows),
      'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
      'tokenizer_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
      'arrays_sha256':hashlib.sha256((out/'FIT_TOKENS.npz').read_bytes()).hexdigest(),
      'dev_accessed':False,'test_accessed':False,'neural_optimizer_updates':0}
    (out/'TOKENS_RECEIPT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='files'}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();encode(a.manifest,a.model,a.out)
