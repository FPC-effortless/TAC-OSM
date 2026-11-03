"""Preserve C16 primary and append post-null C17/position diagnostics."""
import argparse,json,statistics
from pathlib import Path
import torch
from .common import ROOT,C16,SEEDS,require,digest
from collect016.collect_completed import recompute
def main():
    p=argparse.ArgumentParser();p.add_argument('--c16-originals',type=Path,required=True);p.add_argument('--c17-originals',type=Path,required=True);a=p.parse_args()
    c16=json.loads((C16/'results/SUMMARY.json').read_text());c17=json.loads((ROOT/'results/SUMMARY.json').read_text())
    require(len(c16['records'])==9 and len(c17['records'])==3,'missing campaign records')
    records=c16['records']+c17['records'];positions=[];pairs=[]
    for r in records:
        work=ROOT if r['arm']=='attention' else C16;orig=a.c17_originals if r['arm']=='attention' else a.c16_originals
        name=f"{r['arm']}_{r['seed']}";d=json.loads((work/'results'/name/'POSITION_AND_TRAIN.json').read_text())
        require(d['checkpoint_sha256']==r['checkpoint_sha256'] and d['weights_unchanged'] and d['optimizer_updates']==0,'position model differs')
        require(len(d['recent_train'])==16 and {x['prediction_index'] for x in d['recent_train']}=={380},'recent TRAIN cohort differs')
        for block in d['position_matched']:recompute(block)
        positions.append(dict(arm=r['arm'],seed=r['seed'],recent_train_first_accuracy=statistics.mean(x['first_correct'] for x in d['recent_train']),
            recent_train_predictions=sorted({x['prediction'] for x in d['recent_train']}),matched=[dict(key_family=v['key_family'],normal=next(z for z in v['summary'] if z['condition']=='normal')) for v in d['position_matched']]))
        audit=json.loads((work/'results'/f'audit_{name}.json').read_text())
        r['exposure_sequence_sha256']=audit['exposure_sequence_sha256']
        gen=json.loads((work/'results'/name/'GENERATION.json').read_text());r['generation_payloads']=[v['payload'] for v in gen['rows']]
    for seed in SEEDS:
        require(len({r['exposure_sequence_sha256'] for r in records if r['seed']==seed})==1,'attention/phase input schedule differs')
        original=torch.load(a.c16_originals/f'full_bptt_{seed}/checkpoint_0000.pt',weights_only=True,map_location='cpu')['model']
        for arm in ('detach_128','current_only','attention'):
            path=(a.c17_originals if arm=='attention' else a.c16_originals)/f'{arm}_{seed}'/'checkpoint_0000.pt'
            other=torch.load(path,weights_only=True,map_location='cpu')['model'];full=[];slices=[]
            for n,v in other.items():
                if n not in original:continue
                w=original[n]
                if v.shape==w.shape:require(torch.equal(v,w),'compatible initializer differs '+n);full.append(n)
                elif '.ffn_' in n:
                    shape=tuple(min(i,j) for i,j in zip(v.shape,w.shape));sl=tuple(slice(0,z) for z in shape)
                    require(torch.equal(v[sl],w[sl]),'FFN shared slice differs '+n);slices.append(n)
            pairs.append(dict(seed=seed,arm=arm,equal_complete_tensors=len(full),equal_ffn_slices=len(slices)))
    out=dict(status='COMPLETE_COMBINED_INTERNAL_RESEARCH',original_primary='COLLECT016; unchanged',post_null_control='COLLECT017; preregistered before its own first full fit',
        records=records,aggregates=c16['aggregates']+c17['aggregates'],post_null_position=positions,initializer_pairs=pairs,all_four_arm_exposures_equal=True,
        full_updates=6144,smoke_updates=32,full_snapshots=48,smoke_snapshots=8,presented_training_bytes=4718592,generation_samples=24,gpu_minutes=0,
        optimizer_seconds=sum(r['optimizer_seconds'] for r in records),no_architecture_superiority_claim=True,independent_qualification=False,
        current_only_is_history_free=False,full_training_flops='OPEN',global_attention_replacement=False,full_Lean_source_IEEE_AD='OPEN',
        conclusion='All fixed-budget neural arms fail primary retrieval. Local/train-position failure means no established capacity ceiling or useful architecture discriminator.')
    (ROOT/'results/COMBINED_SUMMARY.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ('records','post_null_position','initializer_pairs')},indent=2))
if __name__=='__main__':main()
