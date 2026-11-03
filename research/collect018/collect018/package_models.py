"""Atomic generated-evidence bundles; repository code/data remain in source control."""
import hashlib,json,os,zipfile
from pathlib import Path
from .common import ROOT,require
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bundle(path,entries,metadata):
    require(not path.exists(),'bundle already exists; inspect rather than overwrite')
    temporary=path.with_suffix(path.suffix+'.building')
    contents=[dict(member=name,size=p.stat().st_size,sha256=sha(p)) for name,p in entries]
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for name,p in entries:z.write(p,name)
        z.writestr('CONTENTS.json',json.dumps(dict(metadata=metadata,files=contents),indent=2))
    with temporary.open('rb') as f:os.fsync(f.fileno())
    os.replace(temporary,path)
    return dict(filename=path.name,size=path.stat().st_size,sha256=sha(path),members=len(contents)+1)
def main():
    downloads=ROOT.parent/'collect018_downloads';dest=ROOT.parent/'collect018_deliverables';dest.mkdir(exist_ok=True)
    registry=json.loads((downloads/'ARTIFACT_REGISTRY.json').read_text());require(len(registry)==19,'not all18originals+prefit')
    bundles=[]
    for family in ('phase','context','gru'):
        records=[r for r in registry if r['folder'].startswith(family+'_')];require(len(records)==6,'family originals incomplete')
        entries=[]
        for r in records:
            p=downloads/(r['name']+'.zip');require(sha(p)==r['sha256'] and r['crc_verified'],'original changed');entries.append(('original_artifacts/'+p.name,p))
        for p in sorted((ROOT/'results').glob('smoke_'+family+'_*/*')):
            if p.is_file():entries.append(('smoke/'+p.parent.name+'/'+p.name,p))
        meta=dict(study='COLLECT018',family=family,original_source='19f69b6ced3fa8baa4320f53e2b6dc2914b73308',graph_source='7b9af0d5dccc1f52c5055db850f8f8c28c3ae70c',run=41561294525,full_fits=6,full_optimizer_updates=6144,full_snapshots=24,smoke_fits=2,smoke_updates=16,smoke_snapshots=4,qualification=False,original_artifacts=[{k:v for k,v in r.items() if k not in ('zip_path','extract_dir')} for r in records])
        bundles.append(bundle(dest/('COLLECT018_'+family.upper()+'_CHECKPOINTS.zip'),entries,meta))
    (dest/'MODEL_BUNDLES.json').write_text(json.dumps(bundles,indent=2));print(json.dumps(bundles,indent=2))
if __name__=='__main__':main()
