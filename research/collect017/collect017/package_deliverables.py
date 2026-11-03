"""Package generated originals and audits; omit Git-backed source/datasets."""
import argparse,hashlib,json,os,zipfile
from pathlib import Path
from .common import ROOT,C16,require
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()
def bundle(path,items,description):
    manifest=dict(description=description,entries=[],source_location='GitHub draft PRs TAC-OSM#700,#701 and pnds-lean#107; source and dataset not copied here.')
    require(not path.exists(),'bundle already exists; verify original instead of silently replacing')
    temporary=path.with_suffix('.building')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5,allowZip64=True) as z:
        for source,name in items:
            require(source.is_file(),'missing packaged original '+str(source));data=source.read_bytes()
            manifest['entries'].append(dict(path=name,size=len(data),sha256=hashlib.sha256(data).hexdigest()))
            z.writestr(name,data,compress_type=zipfile.ZIP_STORED if source.suffix=='.zip' else zipfile.ZIP_DEFLATED)
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
        z.writestr('README.txt',description+'\n\nAll generated originals retained; no selected best seed. Internal research, no protected TEST or architecture-superiority claim.\nNested original workflow ZIPs retain fit/evaluation receipts, model and AdamW/RNG checkpoints, raw predictions and their original hashes.\n')
    with temporary.open('rb') as f:os.fsync(f.fileno())
    os.replace(temporary,path)
    return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=sha(path),entries=len(items))
def verify(path):
    with zipfile.ZipFile(path) as z:
        require(z.testzip() is None,'bundle CRC failed');m=json.loads(z.read('MANIFEST.json'))
        require(set(z.namelist())=={e['path'] for e in m['entries']}|{'MANIFEST.json','README.txt'},'archive inventory differs')
        for e in m['entries']:
            data=z.read(e['path']);require(len(data)==e['size'] and hashlib.sha256(data).hexdigest()==e['sha256'],'bundle member differs '+e['path'])
    return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=sha(path),crc_verified=True,all_member_hashes_verified=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--c16-originals',type=Path,required=True);p.add_argument('--c17-originals',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    if a.verify_only:
        rows=[verify(q) for q in sorted(a.output_dir.glob('*.zip'))];require(len(rows)==3,'expected three bundles');print(json.dumps(rows,indent=2));(ROOT/'results/PACKAGING_VERIFICATION.json').write_text(json.dumps(rows,indent=2));return
    a.output_dir.mkdir(exist_ok=True,parents=True);registry16=json.loads((a.c16_originals/'ARTIFACT_REGISTRY.json').read_text());registry17=json.loads((a.c17_originals/'ARTIFACT_REGISTRY.json').read_text())
    models_a=[(Path(r['zip_path']),'originals/'+Path(r['zip_path']).name) for r in registry16 if any(arm in r['name'] for arm in ('full_bptt','detach_128'))]
    models_b=[(Path(r['zip_path']),'originals/'+Path(r['zip_path']).name) for r in registry16 if 'current_only' in r['name']]+[(Path(r['zip_path']),'originals/'+Path(r['zip_path']).name) for r in registry17 if 'attention' in r['name']]
    require(len(models_a)==6 and len(models_b)==6,'incomplete original fit archive set')
    receipts=[bundle(a.output_dir/'COLLECT016_017_MODELS_A.zip',models_a,'Six original connected/detached phase workflow artifacts; 24 immutable full-fit snapshots, three seeds each.'),
        bundle(a.output_dir/'COLLECT016_017_MODELS_B.zip',models_b,'Six original current-value/native-attention workflow artifacts; 24 immutable full-fit snapshots, three seeds each.')]
    items=[]
    for name,work,registry in [('collect016',C16,registry16),('collect017',ROOT,registry17)]:
        for q in sorted((work/'results').rglob('*')):
            if not q.is_file() or q.suffix not in ('.json','.jsonl','.log','.pt','.sha256'):continue
            if q.name.startswith('PACKAGING'):continue
            items.append((q,name+'/results/'+q.relative_to(work/'results').as_posix()))
        original_root=a.c16_originals if name=='collect016' else a.c17_originals
        items.append((original_root/'ARTIFACT_REGISTRY.json',name+'/ARTIFACT_REGISTRY.json'))
        for r in registry:
            if 'prefit' in r['name']:items.append((Path(r['zip_path']),name+'/prefit/'+Path(r['zip_path']).name))
    for q in sorted(a.output_dir.glob('*.png')):items.append((q,'figures/'+q.name))
    receipts.append(bundle(a.output_dir/'COLLECT016_017_EVIDENCE.zip',items,'Generated trained-source audits, read-only diagnostics, all raw supplemental/position predictions, failure logs, smoke snapshots and scientific figures. Zero additional optimizer updates in these diagnostics.'))
    (ROOT/'results/PACKAGING_RECEIPT.json').write_text(json.dumps(receipts,indent=2));print(json.dumps(receipts,indent=2))
if __name__=='__main__':main()
