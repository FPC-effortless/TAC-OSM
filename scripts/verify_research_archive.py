"""Verify original research delivery bytes and indexed payloads without fitting."""
import argparse,hashlib,json,tarfile
from pathlib import Path


def verify(archive,expected):
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest!=expected:raise ValueError('archive SHA-256 mismatch')
    with tarfile.open(archive,'r:gz') as tar:
        index=json.load(tar.extractfile('INDEX.json'))
        for row in index['payloads']:
            member=tar.getmember(row['path'])
            if not member.isfile() or member.size!=row['bytes']:raise ValueError(row['path'])
            if hashlib.sha256(tar.extractfile(member).read()).hexdigest()!=row['sha256']:raise ValueError(row['path'])
    print(json.dumps({'status':'PASS','archive_sha256':digest,'native_models':index.get('native_models',len(index['models'])),
        'indexed_payloads':len(index['payloads']),'weights_retrained':False,'forecast_replayed_on_this_host':False},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('sha256');a=p.parse_args();verify(a.archive,a.sha256)
