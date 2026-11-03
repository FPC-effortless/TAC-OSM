"""Extract exact public FIT/DEV bytes from the pinned source archive only."""
from pathlib import Path
import json,zipfile
from .common import ROOT,verify_data,digest,require

def main():
    path=ROOT/'data/FIT_DEV_SOURCE.zip'
    expected=(ROOT/'data/FIT_DEV_SOURCE.sha256').read_text().strip()
    require(digest(path)==expected,'source payload archive changed')
    allowed={'train.bin','dev.bin'}
    with zipfile.ZipFile(path) as z:
        require(set(z.namelist())==allowed,'unexpected source payload member')
        for name in sorted(allowed):
            data=z.read(name);(ROOT/'data'/name).write_bytes(data)
    _,manifest=verify_data()
    print(json.dumps(dict(status='PASS_ORIGINAL_FIT_DEV_RESTORED',test_bytes_read=False,
        hashes={p:manifest['parts'][p]['sha256'] for p in ('train','dev')})))

if __name__=='__main__':main()
