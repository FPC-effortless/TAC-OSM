"""Download every registered archive; never rely on a first-page glob listing."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import urllib.error
import urllib.parse
import urllib.request
import zipfile

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

def api(repository, suffix, *, redirect=False):
    url = 'https://api.github.com/repos/'+repository+'/'+suffix
    request = urllib.request.Request(url, headers={
        'Authorization':'Bearer '+os.environ['GH_TOKEN'],
        'Accept':'application/vnd.github+json','User-Agent':'LANGMATH101-artifact-audit'})
    opener = urllib.request.build_opener(NoRedirect())
    try:
        with opener.open(request, timeout=45) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if not redirect or error.code not in (301,302,303,307,308): raise
        destination = error.headers['Location']
        if urllib.parse.urlsplit(destination).scheme != 'https':
            raise ValueError('Artifact redirect must use HTTPS')
        # The GitHub token is sent only to api.github.com, never to the blob host.
        with urllib.request.urlopen(destination, timeout=60) as response:
            data = response.read(128*1024*1024+1)
        if len(data)>128*1024*1024: raise ValueError('Artifact exceeds bounded archive size')
        return data

def records(repository, run_id, prefix, expected, registry=None):
    if registry is not None:
        assert registry['run_id'] == run_id
        rows = [{'id':r['artifact_id'],'name':r['name'],'digest':'sha256:'+r['archive_sha256'],
                 'initial_checkpoint_sha256':r['initial_checkpoint_sha256']} for r in registry['archives']]
    else:
        rows=[]; page=1
        while True:
            payload=api(repository,f'actions/runs/{run_id}/artifacts?per_page=100&page={page}')
            entries=payload['artifacts']; rows.extend(r for r in entries if r['name'].startswith(prefix))
            if page*100>=payload['total_count']: break
            if not entries or page>=100: raise ValueError('Incomplete artifact pagination')
            page+=1
    if len(rows)!=expected or len({r['name'] for r in rows})!=expected:
        raise ValueError(f'Expected exactly{expected} distinct archives; found{len(rows)}')
    return rows

def extract_verified(raw, row, output):
    if 'sha256:'+hashlib.sha256(raw).hexdigest()!=row['digest']:
        raise ValueError('Original artifact ZIP digest differs: '+row['name'])
    if PurePosixPath(row['name']).name!=row['name']:
        raise ValueError('Invalid artifact name')
    destination=Path(output)/row['name'];destination.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for member in archive.infolist():
            path=PurePosixPath(member.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in member.filename:
                raise ValueError('Unsafe archive member')
            if (member.external_attr>>16)&0o170000==0o120000:
                raise ValueError('Archive symlink prohibited')
            if member.file_size>128*1024*1024: raise ValueError('Oversized archive member')
            target=destination.joinpath(*path.parts)
            if member.is_dir(): target.mkdir(parents=True,exist_ok=True)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as stream: stream.write(archive.read(member))
    if 'initial_checkpoint_sha256' in row:
        if hashlib.sha256((destination/'INITIAL_MODEL.pt').read_bytes()).hexdigest()!=row['initial_checkpoint_sha256']:
            raise ValueError('Original initial checkpoint bytes differ')
    return {'artifact_id':row['id'],'name':row['name'],'digest':row['digest']}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repository',required=True);parser.add_argument('--run-id',type=int,required=True)
    parser.add_argument('--prefix',default='');parser.add_argument('--expected',type=int,required=True)
    parser.add_argument('--registry',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    registry=json.loads(args.registry.read_bytes()) if args.registry else None
    rows=records(args.repository,args.run_id,args.prefix,args.expected,registry)
    def download(row):
        if row.get('expired'): raise ValueError('Original artifact expired')
        raw=api(args.repository,'actions/artifacts/'+str(row['id'])+'/zip',redirect=True)
        return extract_verified(raw,row,args.output)
    with ThreadPoolExecutor(max_workers=6) as pool: receipts=list(pool.map(download,rows))
    print(json.dumps({'verified_artifact_count':len(receipts),'run_id':args.run_id,'archives':receipts}),flush=True)

if __name__=='__main__':main()
