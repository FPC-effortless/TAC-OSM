"""Download connector-returned original artifact URLs privately; validate identity."""
import argparse,hashlib,json,stat,urllib.request,zipfile
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);p.add_argument('--dest',type=Path,required=True);a=p.parse_args()
    requests=json.loads(a.request.read_text());a.dest.mkdir(parents=True,exist_ok=True);registry=[]
    for r in requests:
        assert r['source_sha']=='2209e1328e3b42db4dd48cd702a21a55ffad24af'
        z=a.dest/(r['name']+'.zip');expected=r['digest'].removeprefix('sha256:')
        if not z.exists():
            with urllib.request.urlopen(r['url'],timeout=90) as response,z.open('wb') as out:
                while block:=response.read(1024*1024):out.write(block)
        assert z.stat().st_size==r['expected_size'],'artifact ZIP size mismatch'
        sha=hashlib.sha256(z.read_bytes()).hexdigest();assert sha==expected,'artifact ZIP digest mismatch'
        folder=a.dest/r['folder'];folder.mkdir(exist_ok=True)
        with zipfile.ZipFile(z) as archive:
            assert archive.testzip() is None,'artifact ZIP CRC mismatch'
            for info in archive.infolist():
                path=Path(info.filename)
                assert not path.is_absolute() and '..' not in path.parts and not stat.S_ISLNK(info.external_attr>>16),'unsafe ZIP path'
            archive.extractall(folder)
        record={k:v for k,v in r.items() if k!='url'};record.update(zip_path=str(z.resolve()),extract_dir=str(folder.resolve()),sha256=sha,crc_verified=True)
        registry.append(record);print(json.dumps(dict(artifact_id=r['artifact_id'],name=r['name'],sha256=sha,verified=True)),flush=True)
    path=a.dest/'ARTIFACT_REGISTRY.json';old=json.loads(path.read_text()) if path.exists() else []
    merged={r['artifact_id']:r for r in old}
    merged.update({r['artifact_id']:r for r in registry});path.write_text(json.dumps(list(merged.values()),indent=2));a.request.unlink()
if __name__=='__main__':main()
