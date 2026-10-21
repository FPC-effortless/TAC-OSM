"""Commit-sized original engineering records, no retraining or seed substitution."""
import io,json,tarfile
from pathlib import Path
import scripts.me039_research as b


def archive(name,paths,metadata):
    paths=sorted(set(paths));destination=b.ROOT/'research/model-artifacts'/name
    index={**metadata,'models':[],'payloads':[{'path':str(p.relative_to(b.ROOT)),'bytes':p.stat().st_size,'sha256':b.sha(p)} for p in paths]}
    with tarfile.open(destination,'w:gz',compresslevel=3) as tar:
        for p in paths:tar.add(p,arcname=str(p.relative_to(b.ROOT)),recursive=False)
        data=json.dumps(index,indent=2).encode();info=tarfile.TarInfo('INDEX.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
    assert destination.stat().st_size<100_000_000,'Do not silently add over-limit Git blobs'
    return {'path':str(destination.relative_to(b.ROOT)),'bytes':destination.stat().st_size,'sha256':b.sha(destination),'indexed_payloads':len(paths),'kind':metadata['kind']}


def run():
    receipts=[]
    for dataset in ('ETTh1','electricity'):
        paths=[p for folder in (b.OUT/'training').glob(dataset+'-*') for p in folder.iterdir() if p.is_file()]
        assert len(paths)==120
        receipts.append(archive(f'me039-{dataset}-numerical-records.tar.gz',paths,{'kind':'engineering_optimizer_gradient_replays','dataset':dataset,'continuations':24,'new_capability_models':0}))
    paths=[p for p in (b.OUT/'exports').rglob('*') if p.is_file()]
    assert len(paths)==180
    receipts.append(archive('me039-experimental-serving-exports.tar.gz',paths,{'kind':'experimental_scripted_core_exports','native_models':60,'recommended_for_serving':False}))
    b.dump(b.ROOT/'research/model-artifacts/me039-archives.json',{'archives':receipts,'original_records_no_refit':True})
    print(json.dumps(receipts,indent=2))

if __name__=='__main__':run()
