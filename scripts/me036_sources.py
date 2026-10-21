"""Acquire exactly the registered unmodified publisher/mirror bytes."""
import concurrent.futures, hashlib, json, urllib.request
from pathlib import Path
import scripts.me036_research as base


def fetch(item):
    name,record=item;path=base.DATA/name
    if Path(name).name!=name:raise ValueError('invalid source basename')
    if path.exists():
        assert base.sha(path)==record['sha256'] and path.stat().st_size==record['bytes']
        return name,'verified existing'
    with urllib.request.urlopen(record['url'],timeout=120) as response:
        data=response.read(record['bytes']+1)
    if len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']:
        raise ValueError('registered source changed: '+name)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_bytes(data);temp.replace(path)
    return name,'downloaded and verified'


if __name__=='__main__':
    base.DATA.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((base.REPORT/'sources.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for row in pool.map(fetch,manifest.items()):print(*row,flush=True)
