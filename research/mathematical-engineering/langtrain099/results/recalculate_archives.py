"""Recalculate archived GHA predictions; never execute a neural model or fit."""
from pathlib import Path
import argparse,json,hashlib,numpy as np,torch
parser=argparse.ArgumentParser();parser.add_argument('--artifacts',type=Path,required=True);parser.add_argument('--judge',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
root=args.artifacts;out={};evalroot=root/'langtrain099-complete-evaluation'
if evalroot.exists():
    official=json.loads((evalroot/'RESULTS.json').read_text());episodes=json.loads((evalroot/'SCORED_EPISODES.json').read_text());language=json.loads((evalroot/'SCORED_LANGUAGE.json').read_text());judge=json.loads(args.judge.read_text())
    lexical=[]
    for seed in (9901,9902,9903):
        for stage in ('initial','trained'):
            with np.load(evalroot/f'{stage}-{seed}/original.npz',allow_pickle=False) as p:
                meta=json.loads(str(p['metadata'].item()));ids=meta['case_ids']
                for ci,condition in enumerate(('carry','reset','donor','corrupt')):
                    for group in ('positive','negative','uniform'):
                        indices=[i for i,case in enumerate(ids) if judge[case]['domain']=='lexical' and judge[case]['group']==group]
                        successes=sum(bytes(p['generated_bytes'][ci,i,:4])==bytes(judge[ids[i]]['target_bytes']) for i in indices)
                        nll=np.mean([-float(p['log_probs'][ci,i,judge[ids[i]]['target_bytes'][0]]) for i in indices])
                        stored=[r for r in episodes if (r['seed'],r['checkpoint_stage'],r['domain'],r['group'],r['condition'])==(seed,stage,'lexical',group,condition)]
                        assert sum(r['exact'] for r in stored)==successes and abs(np.mean([r['nll'] for r in stored])-nll)<1e-10
                        lexical.append(dict(seed=seed,stage=stage,group=group,condition=condition,correct=successes,cases=len(indices),nll=float(nll)))
    out['lexical_recomputed_from_original_predictions']=lexical
    out['language_by_seed']=[]
    for stage in ('initial','trained'):
        for seed in (9901,9902,9903):
            for modality in sorted({r['modality'] for r in language}):
                group=[r for r in language if (r['stage'],r['seed'],r['modality'])==(stage,seed,modality)]
                out['language_by_seed'].append({'stage':stage,'seed':seed,'modality':modality,'nll':float(np.mean([r['nll'] for r in group])),'accuracy':sum(r['correct'] for r in group)/(256*len(group)),'bigram_nll':float(np.mean([r['bigram_nll'] for r in group]))})
    out['capability_gates']=official['capability_gates'];out['effects']=official['effects']
receipts=[]
for folder in sorted(p for p in root.glob('langtrain099-original-seed-*') if p.is_dir() and (p/'FIT_RECEIPT.json').exists()):
    r=json.loads((folder/'FIT_RECEIPT.json').read_text());trained=torch.load(folder/'MODEL.pt',map_location='cpu',weights_only=True);initial=torch.load(folder/'INITIAL_MODEL.pt',map_location='cpu',weights_only=True)
    assert hashlib.sha256((folder/'MODEL.pt').read_bytes()).hexdigest()==r['checkpoint_sha256']
    assert hashlib.sha256((folder/'INITIAL_MODEL.pt').read_bytes()).hexdigest()==r['initial_checkpoint_sha256']
    assert sum(v.numel() for k,v in trained['model'].items() if k not in ('bank_min','bank_max'))==10000000
    assert all(float(x['step'])==2048 for x in trained['optimizer']['state'].values()) and len(trained['optimizer']['state'])==17
    assert all(torch.isfinite(v).all() for v in trained['model'].values())
    assert any(not torch.equal(trained['model'][k],initial['model'][k]) for k in trained['model'])
    with np.load(folder/'COLD_FIT_ORIGINAL.npz',allow_pickle=False) as a,np.load(folder/'COLD_FIT_REPLAY.npz',allow_pickle=False) as b:
        assert set(a.files)==set(b.files);delta=max(np.abs(a[n]-b[n]).max() for n in a.files);assert delta<=1e-6
    summary={k:r[k] for k in ('seed','steps','trainable_parameters','checkpoint_sha256','initial_checkpoint_sha256','elapsed_seconds','exposure','hardware','source_commit')}
    summary['original_optimizer_slot_count']=17;summary['all_original_optimizer_slots_steps']=2048;summary['cold_replay_recomputed_max_abs_difference']=float(delta)
    for mode in ('language','episodic'):
        history=[x['loss'] for x in r['history'] if x['mode']==mode]
        summary[mode+'_first32_loss_mean']=float(np.mean(history[:32]));summary[mode+'_last32_loss_mean']=float(np.mean(history[-32:]));summary[mode+'_updates']=len(history)
    receipts.append(summary)
out['original_fit_audits']=receipts;out['method']='Read-only original archive/logit arithmetic; zero new neural inference or optimizer updates'
args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
