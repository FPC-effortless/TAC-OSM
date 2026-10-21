"""Local self-contained archive of original ME036 models, not a public dataset release."""
import io, json, subprocess, tarfile
from pathlib import Path
import scripts.me036_research as base


def run(destination):
    report=base.REPORT;out=base.ROOT/'results/me036b'
    assert json.loads((out/'replay.json').read_text())['status']=='PASS'
    frozen=json.loads((out/'frozen.json').read_text())
    for r in frozen['fits']:
        for e in r['endpoints']:assert base.sha(base.ROOT/e['checkpoint'])==e['sha256']
    for r in frozen['ridge']:assert base.sha(base.ROOT/r['coefficient'])==r['sha256']
    excluded={'ARTIFACT.md','archive.json','archive-index.json','restoration.json','restoration.log'}
    paths=[p for p in subprocess.check_output(['git','ls-files'],cwd=base.ROOT,text=True).splitlines()
        if (base.ROOT/p).is_file() and not (p.startswith('research/mathematical-engineering/me036/') and Path(p).name in excluded)]
    for folder in (base.ROOT/'results/me036',out,base.ROOT/'results/me036c',base.DATA):
        paths += [str(p.relative_to(base.ROOT)) for p in folder.rglob('*') if p.is_file() and not p.name.endswith('.tmp')]
    paths=sorted(set(paths));head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.ROOT,text=True).strip()
    index={'source_snapshot_commit':head,'original_models_no_refit':True,'local_research_delivery':True,
        'publisher_sources':'Unmodified ETT CSVs/license; THUML Weather/Electricity CSVs and original mirror dataset card. Not committed to public repository.',
        'payloads':[{'path':p,'bytes':(base.ROOT/p).stat().st_size,'sha256':base.sha(base.ROOT/p)} for p in paths]}
    base.dump(report/'archive-index.json',index)
    content=json.dumps(index,indent=2).encode()
    with tarfile.open(destination,'w:gz',compresslevel=3) as tar:
        for r in index['payloads']:tar.add(base.ROOT/r['path'],arcname=r['path'],recursive=False)
        item=tarfile.TarInfo('ARCHIVE-INDEX.json');item.size=len(content);tar.addfile(item,io.BytesIO(content))
    receipt={'path':str(destination),'bytes':destination.stat().st_size,'sha256':base.sha(destination),
        'indexed_payloads':len(paths),'source_snapshot_commit':head,
        'original_checkpoint_endpoints':sum(len(r['endpoints']) for r in frozen['fits']),
        'native_selected_models':len(json.loads((out/'runtime.json').read_text())['exports']),
        'native_TorchScript_exports':2*len(json.loads((out/'runtime.json').read_text())['exports']),
        'registered_ridge_coefficients':len(frozen['ridge']),
        'posthoc_corrected_ridge_coefficients':20,'public_GitHub_Actions_artifact':False}
    base.dump(report/'archive.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':run(Path('/workspace/me036-full-horizon-models-and-evidence.tar.gz'))
