"""Explicitly supplemental balanced probe; leaves frozen primary assay intact."""
import argparse,collections,hashlib,json,random
from pathlib import Path
import torch
from .common import ROOT,build,require,digest
from . import assays
from collect012.train import fingerprint

def episodes():
    rng=random.Random(140002);rows=[]
    for owner in range(12):
        keys=[]
        while len(keys)<6:
            k=''.join(rng.choices('ijklmnopqrstuvwxyz',k=4))
            if k not in keys:keys.append(k)
        latest=[(owner*4+j)%8 for j in range(4)];initial=latest.copy()
        initial[0]=(initial[0]+3)%8;initial[1]=(initial[1]+5)%8
        records=[(keys[j],initial[j]) for j in range(4)]+[(keys[j],latest[j]) for j in range(2)]
        prefix='Use ? if absent.\n'+''.join(f'The mark for {k} is {chr(97+v)}.\n' for k,v in records)
        rows.append(dict(keys=keys,latest=latest,initial=initial,prefix=prefix))
    counts=collections.Counter(v for r in rows for v in r['latest']);assert counts=={i:6 for i in range(8)}
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--control',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);receipt=json.loads((a.run/'FIT_RECEIPT.json').read_text());ck=a.run/receipt['checkpoint']
    require(receipt['optimizer_updates']==1024 and digest(ck)==receipt['checkpoint_sha256'],'original checkpoint differs')
    if a.control:
        from collect012.common import build as original
        model=original('hybrid_hard_p',receipt['seed'])
    else:model=build(receipt['arm'],receipt['seed'])
    saved=torch.load(ck,map_location='cpu',weights_only=True);model.load_state_dict(saved['model']);model.eval();before=fingerprint(model)
    original_episodes=assays.episode_spec
    try:
        assays.episode_spec=episodes
        with torch.inference_mode():result=assays.retrieval(model)
    finally:assays.episode_spec=original_episodes
    require(before==fingerprint(model),'supplemental probe changed parameters')
    result.update(variant='control' if a.control else receipt['arm'],seed=receipt['seed'],
        checkpoint_sha256=digest(ck),cohort='SUPPLEMENTAL_BALANCED_140002',
        protocol=json.loads((ROOT/'SUPPLEMENTAL_BALANCED_PROTOCOL.json').read_text()),
        protocol_sha256=digest(ROOT/'SUPPLEMENTAL_BALANCED_PROTOCOL.json'),weights_unchanged=True,optimizer_updates=0)
    a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(variant=result['variant'],seed=result['seed'],summaries=result['summaries']),indent=2))

if __name__=='__main__':main()
