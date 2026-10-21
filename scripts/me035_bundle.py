"""Package original trained forecasters and unmodified licensed source."""
import io,json,subprocess,tarfile
from pathlib import Path
from scripts.me035_architecture_research import ROOT,file_sha,write_json
from scripts.me035_audit import verify as verify_a
from scripts.me035b_audit import verify as verify_b


def run(destination):
    a,b=ROOT/'results/me035',ROOT/'results/me035b';verify_a(a);verify_b(b)
    assert json.loads((a/'replay.json').read_text())['status']=='PASS'
    assert json.loads((b/'replay.json').read_text())['forecast_replay_status']=='PASS'
    excluded={'ARTIFACT.md','archive.json','archive-index.json','restoration.json','restoration.log'}
    paths=[p for p in subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines() if (ROOT/p).is_file()
        and not(p.startswith('research/mathematical-engineering/me035/') and Path(p).name in excluded)]
    paths+= [str(p.relative_to(ROOT)) for folder in (a,b) for p in folder.iterdir() if p.is_file() and p.name!='verification.json']
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    index={'source_snapshot_commit':head,'contains_original_models_no_refit':True,'publisher_source':'unmodified original ETTh1.csv and CC BY-ND4.0license',
        'initial_equivalence_receipt_pass':False,'payloads':[{'path':p,'sha256':file_sha(ROOT/p),'bytes':(ROOT/p).stat().st_size} for p in sorted(set(paths))]}
    report=ROOT/'research/mathematical-engineering/me035';write_json(report/'archive-index.json',index)
    data=(json.dumps(index,indent=2,sort_keys=True)+'\n').encode()
    with tarfile.open(destination,'w:gz',compresslevel=6) as t:
        for row in index['payloads']:t.add(ROOT/row['path'],arcname=row['path'],recursive=False)
        item=tarfile.TarInfo('ARCHIVE-INDEX.json');item.size=len(data);t.addfile(item,io.BytesIO(data))
    receipt={'path':str(destination),'bytes':destination.stat().st_size,'sha256':file_sha(destination),
        'indexed_payloads':len(index['payloads']),'source_snapshot_commit':head,'raw_selected_models':3,
        'original_checkpoint_endpoints':51,'offline_replay_self_contained':True,'initial_equivalence_receipt_pass':False}
    write_json(report/'archive.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':run(Path('/workspace/me035-native-architecture-models-and-evidence.tar.gz'))
