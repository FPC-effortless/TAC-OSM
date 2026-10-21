"""Bundle original evidence plus complete offline ME034 audit dependencies."""
import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path
from scripts.me034_stable_research import ROOT,SEEDS,file_sha,write_json
from scripts.me034_audit import verify


def build(destination):
    out=ROOT/'results/me034';report=ROOT/'research/mathematical-engineering/me034'
    r=verify(out);assert r['complete'] and json.loads((out/'replay.json').read_text())['status']=='PASS'
    circular={'ARTIFACT.md','archive.json','archive-index.json','restoration.log','restoration.json'}
    source=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    paths={p:'repository_source' for p in source if (ROOT/p).is_file() and not
        (p.startswith('research/mathematical-engineering/me034/') and Path(p).name in circular)}
    for p in out.iterdir():
        if p.is_file() and p.name!='verification.json':paths[str(p.relative_to(ROOT))]='original_ME034_evidence'
    old=json.loads((ROOT/'results/me022/results.json').read_text())
    support=['results/me022/results.json','results/me022/data-snapshot.pt']
    support += ['results/me022/'+row['checkpoint'] for row in old['models']]
    for c in r['cells']:
        support += [f'results/me031/encoder-r{c["rotation"]}-s{c["seed"]}.pt',
                    f'results/me032/teacher-r{c["rotation"]}-s{c["seed"]}.pt']
    support += ['results/'+study+'/results.json' for study in ('me031','me032','me033')]
    for p in support:paths[p]='original_offline_audit_support'
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    index={'study':r['study'],'source_snapshot_commit':head,'fit_source_freeze':r['source_commit'],
        'scope':'Original fits/optimizer/RNG/predictions/runtimes and self-contained offline ME034 replay support; no refit or new final access.',
        'payloads':[{'path':p,'sha256':file_sha(ROOT/p),'bytes':(ROOT/p).stat().st_size,'kind':paths[p]} for p in sorted(paths)]}
    write_json(report/'archive-index.json',index)
    data=(json.dumps(index,indent=2,sort_keys=True)+'\n').encode()
    with tarfile.open(destination,'w:gz',compresslevel=6) as tar:
        for item in index['payloads']:tar.add(ROOT/item['path'],arcname=item['path'],recursive=False)
        info=tarfile.TarInfo('ARCHIVE-INDEX.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
    receipt={'path':str(destination),'bytes':destination.stat().st_size,'sha256':file_sha(destination),
        'indexed_payloads':len(index['payloads']),'source_snapshot_commit':head,'fit_source_freeze':r['source_commit'],
        'offline_audit_self_contained':True,'source_ZIP_included':False,'source_arrays_and_pinned_original_parent_checkpoints_included':True}
    write_json(report/'archive.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('/workspace/me034-stable-input-checkpoints-and-evidence.tar.gz'));a=p.parse_args();build(a.out)
