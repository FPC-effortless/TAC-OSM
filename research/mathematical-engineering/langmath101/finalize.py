"""Fail-closed all-fit barrier before any LANGMATH101 candidate DEV evaluation.

Rebinds actual initial/final checkpoint bytes, AdamW slots, source/profile,
ordered exposure and measured operator ledgers for every preregistered job.
No model selection, training, DEV or protected TEST access occurs here.
"""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import torch
import numpy as np

HERE = Path(__file__).resolve().parent
sp = importlib.util.spec_from_file_location('langmath101_finalize_common', HERE/'common.py')
C = importlib.util.module_from_spec(sp); sys.modules[sp.name] = C; sp.loader.exec_module(C)


def require(ok, message):
    if not ok: raise ValueError(message)


def _identity(value):
    return value.get('arm'), value.get('seed'), value.get('lane')


def _rng(payload, label):
    rng = payload.get('rng')
    require(isinstance(rng, dict) and set(rng) == {'torch', 'python', 'numpy_keys'}, label+' complete RNG fields absent')
    require(isinstance(rng['torch'], torch.Tensor) and rng['torch'].dtype == torch.uint8 and rng['torch'].numel() > 0, label+' torch RNG absent')
    require(isinstance(rng['python'], tuple) and len(rng['python']) == 3, label+' Python RNG absent')
    require(isinstance(rng['numpy_keys'], torch.Tensor) and rng['numpy_keys'].shape == (624,), label+' NumPy RNG absent')


def _semantic_snapshot(snapshot):
    """Rebase only process-local BPE parameter object IDs/version counters."""
    out = copy.deepcopy(snapshot)
    fields = out.get('fields', {})
    guard = fields.get('cache_guard', ())
    if guard:
        require(out.get('dataclass','').endswith('.AdaptiveState'), 'volatile guard outside registered adaptive state')
        require(isinstance(guard,tuple) and guard[0][0] == 'tokenizer' and len(guard[0]) == 2, 'invalid raw tokenizer guard')
        for item in guard[1:]:
            require(isinstance(item,tuple) and len(item) == 5 and isinstance(item[1],int) and (item[2] is None or isinstance(item[2],int)), 'invalid raw parameter guard')
        fields['cache_guard'] = (guard[0],)+tuple((item[0],item[3],item[4]) for item in guard[1:])
    return out


def verify_checkpoint(directory, receipt, row, profile, manifest):
    directory = Path(directory)
    require(not any((directory/n).is_symlink() for n in ('MODEL.pt','INITIAL_MODEL.pt','FIT_RECEIPT.json')), 'checkpoint symlink forbidden')
    final_path, initial_path = directory/'MODEL.pt', directory/'INITIAL_MODEL.pt'
    require(C.sha(final_path) == receipt.get('checkpoint_sha256'), 'actual MODEL.pt SHA differs')
    require(C.sha(initial_path) == receipt.get('initial_checkpoint_sha256'), 'actual INITIAL_MODEL.pt SHA differs')
    require(receipt['checkpoint_sha256'] != receipt['initial_checkpoint_sha256'], 'initial/final checkpoint bytes identical')
    initial = torch.load(initial_path, map_location='cpu', weights_only=True)
    final = torch.load(final_path, map_location='cpu', weights_only=True)
    expected = row['architecture'], row['seed'], row['lane']
    for label, payload, stage, steps in (('initial',initial,'INITIAL_BEFORE_DEV',0),('final',final,'SEALED_BEFORE_DEV',receipt['actual_steps'])):
        require(set(payload) == {'model','optimizer','metadata','rng'}, label+' complete checkpoint payload absent')
        meta = payload['metadata']; require(_identity(meta) == expected, label+' arm/seed/lane differs')
        require(meta.get('stage') == stage and meta.get('optimizer_steps') == steps, label+' checkpoint stage/steps differs')
        require(meta.get('source_hashes') == profile['source_hashes'], label+' executable source differs')
        require(meta.get('implementation_sha256') == profile['implementation_sha256'], label+' implementation differs')
        require(meta.get('recipe') == C.RECIPE and meta.get('training_recipe_sha256') == manifest['training_recipe_sha256'], label+' recipe differs')
        require(meta.get('tokenizer_receipt_sha256') == profile['tokenizer_receipts_sha256'][str(row['seed'])], label+' tokenizer artifact identity differs')
        tok = meta.get('tokenizer_receipt'); require(isinstance(tok,dict), label+' FIT tokenizer receipt absent')
        require(tok.get('partition') == 'FIT', label+' tokenizer not FIT-only')
        C.adaptive().FrozenBPE.from_receipt(tok)
        _rng(payload,label)
    require(initial['metadata']['tokenizer_receipt'] == final['metadata']['tokenizer_receipt'], 'initial/final tokenizer differs')
    require(initial['optimizer'] is None, 'initial archive already has optimizer state')
    imodel = C.load_model(initial); before = C.state_digest(imodel)
    model = C.load_model(final); after = C.state_digest(model)
    require(before == receipt['initial_weights_sha256'], 'actual initialization differs from fit receipt')
    require(after == receipt['final_weights_sha256'] and before != after, 'actual final weights did not change or receipt differs')
    require(all(bool(torch.isfinite(p).all()) for p in model.state_dict().values()), 'nonfinite actual final weights')
    counts = model.counts()
    require((counts['trainable'],counts['deployed']) == (row['trainable_parameters'],row['deployed_parameters']), 'actual model parameter count differs')
    optimizer = final.get('optimizer'); require(isinstance(optimizer,dict) and set(optimizer) == {'state','param_groups'}, 'actual AdamW optimizer slots absent')
    groups = optimizer['param_groups']; require(len(groups) == 1, 'exact one AdamW parameter group required')
    group = groups[0]
    for key, expected_value in (('lr',C.RECIPE['learning_rate']),('betas',tuple(C.RECIPE['betas'])),('eps',C.RECIPE['eps']),('weight_decay',C.RECIPE['weight_decay']),('foreach',False)):
        require(group.get(key) == expected_value, 'saved AdamW policy differs: '+key)
    params = list(model.parameters()); ids = group['params']; slots = optimizer['state']
    require(len(ids) == len(params) == len(slots) and len(set(ids)) == len(ids) and set(ids) == set(slots), 'actual optimizer parameter/slot coverage differs')
    slot_steps = []
    for ident, parameter in zip(ids,params):
        slot = slots[ident]; require(set(slot) == {'step','exp_avg','exp_avg_sq'}, 'actual AdamW slot schema differs')
        step = slot['step']; require(isinstance(step,torch.Tensor) and step.numel() == 1 and float(step) == receipt['actual_steps'], 'actual optimizer slot step differs')
        slot_steps.append(int(step))
        for key in ('exp_avg','exp_avg_sq'):
            moment = slot[key]
            require(isinstance(moment,torch.Tensor) and moment.shape == parameter.shape and moment.dtype == parameter.dtype and bool(torch.isfinite(moment).all()), 'invalid actual optimizer moment: '+key)
        require(bool((slot['exp_avg_sq'] >= 0).all()), 'negative AdamW second moment')
    require(slot_steps == receipt.get('optimizer_slot_steps'), 'optimizer slot receipt differs from actual checkpoint')
    require(receipt.get('cold_replay') == 'EXACT_SEMANTIC_STATE' and receipt.get('cold_replay_scope') == 'all_persistent_content_exact; BPE_object_id_and_parameter_version_verified_per_model_then_rebased', 'complete semantic cold replay not recorded')
    require(receipt.get('cold_continuation') == 'EXACT_LOGITS_AND_SEMANTIC_STATE_ON_NEXT_FOUR_FIT_OBSERVATIONS', 'actual cold continuation not recorded')
    for filename,key in (('COLD_RELOADED_STATE.pt','cold_reloaded_state_sha256'),('COLD_CONTINUATION_STATE.pt','cold_continuation_state_sha256')):
        require(C.sha(directory/filename) == receipt.get(key), 'cold replay/continuation state archive changed: '+filename)
    require((directory/'COLD_CONTINUATION.npz').is_file(), 'actual cold continuation prediction archive absent')
    require(C.sha(directory/'COLD_FIT_STATE.pt') == receipt.get('cold_state_sha256'), 'cold FIT state archive changed')
    require((directory/'COLD_FIT.npz').is_file() and (directory/'LOSSES.json').is_file(), 'complete FIT replay/loss archives missing')
    T = C.local('train')
    original_state = torch.load(directory/'COLD_FIT_STATE.pt',map_location='cpu',weights_only=True)
    reloaded_state = torch.load(directory/'COLD_RELOADED_STATE.pt',map_location='cpu',weights_only=True)
    T.assert_state_equal(_semantic_snapshot(original_state),_semantic_snapshot(reloaded_state),'actual_saved_cold_semantic_state')
    continuation = torch.load(directory/'COLD_CONTINUATION_STATE.pt',map_location='cpu',weights_only=True)
    require(set(continuation) == {'original','reloaded'}, 'complete cold continuation states absent')
    T.assert_state_equal(_semantic_snapshot(continuation['original']),_semantic_snapshot(continuation['reloaded']),'actual_saved_continuation_semantic_state')
    for filename in ('COLD_FIT.npz','COLD_CONTINUATION.npz'):
        with np.load(directory/filename,allow_pickle=False) as replay:
            require('tokens' in replay and replay['tokens'].dtype.kind in 'iu' and bool(((replay['tokens'] >= 0)&(replay['tokens'] < 256)).all()), 'invalid observed FIT replay bytes')
            for name in replay.files:
                require(bool(np.isfinite(replay[name]).all()), 'nonfinite saved cold FIT array')
    losses = C.read(directory/'LOSSES.json')
    require(len(losses) == receipt['actual_steps'] and all(isinstance(x,(float,int)) and torch.isfinite(torch.tensor(x)).item() for x in losses), 'loss archive is nonfinite or incomplete')
    return {'arm':row['architecture'],'seed':row['seed'],'lane':row['lane'],
            'checkpoint_sha256':C.sha(final_path),'initial_checkpoint_sha256':C.sha(initial_path),
            'checkpoint_path':str(final_path.resolve()), 'fit_receipt_sha256':C.sha(directory/'FIT_RECEIPT.json'),
            'actual_steps':receipt['actual_steps'],'final_weights_sha256':after,'initial_weights_sha256':before,
            'optimizer_parameter_slots':len(slots), 'cold_state_sha256':receipt['cold_state_sha256']}


def finalize(args):
    require(not args.output.exists(), 'retain original finalization outputs')
    args.output.mkdir(parents=True)
    manifest = C.read(args.manifest); gate = C.local('owner_two_lane')
    gate.validate_program(manifest, 'preflight', root=C.ROOT)
    expected = {(r['architecture'],r['seed'],r['lane']) for r in manifest['arms']}
    exact = {(arm,seed,lane) for arm in C.arm_names() for seed in C.SEEDS for lane in gate.LANES}
    require(expected == exact and len(expected) == 138, 'complete exact 23-arm x 3-seed x 2-lane matrix required')
    profile = C.read(C.ROOT/manifest['operator_profile_path'])
    require(C.source_hashes() == profile['source_hashes'], 'actual executable source differs from frozen profile')
    require(C.digest(C.RECIPE) == manifest['training_recipe_sha256'], 'actual runtime recipe differs')
    receipts = {}; paths = {}
    for path in sorted(args.fitsdir.rglob('FIT_RECEIPT.json')):
        require(not path.is_symlink(), 'fit receipt symlink forbidden')
        value = C.read(path); key = _identity(value)
        require(key in expected and key not in receipts, 'unexpected or duplicate fit receipt: '+str(key))
        require(value.get('status') == 'SEALED_BEFORE_DEV' and value.get('protected_TEST_access') is False, 'fit is not sealed before DEV')
        receipts[key],paths[key] = value,path
    require(set(receipts) == expected, 'not all 138 original FIT receipts are present: missing='+str(sorted(expected-set(receipts))))
    require(not list(args.fitsdir.rglob('FAILURE.json')), 'unresolved FIT failure receipt present; retain and resolve in separate preregistered run')
    measured = copy.deepcopy(manifest); checked = []
    for row in measured['arms']:
        key = row['architecture'],row['seed'],row['lane']; receipt = receipts[key]
        checked.append(verify_checkpoint(paths[key].parent,receipt,row,profile,manifest))
        row['measured'] = receipt
        print(json.dumps({'checked':len(checked),'jobs':138,'arm':key[0],'seed':key[1],'lane':key[2]}),flush=True)
    final_gate = gate.validate_program(measured,'final',root=C.ROOT)
    C.save(args.output/'FINAL_MEASURED_MANIFEST.json',measured)
    C.save(args.output/'FINAL_OWNER_GATE.json',final_gate)
    seal = {'schema':'langmath101.all-fits-seal.v1','status':'SEALED_ALL_FITS_BEFORE_DEV','jobs':138,
            'manifest_sha256':C.sha(args.manifest),'measured_manifest_sha256':C.sha(args.output/'FINAL_MEASURED_MANIFEST.json'),
            'final_gate_sha256':C.sha(args.output/'FINAL_OWNER_GATE.json'), 'implementation_sha256':profile['implementation_sha256'],
            'source_hashes':profile['source_hashes'],'checkpoints':checked,
            'no_DEV_data_read':True,'protected_TEST_access':False,'eligible_for_architecture_superiority':False,
            'independent_replication':False,'full_model_FLOPs':'INCONCLUSIVE_NOT_CLAIMED'}
    C.save(args.output/'SEALED_ALL_FITS.json',seal)
    return seal


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--fitsdir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try: finalize(a)
    except Exception as e:
        if a.output.exists() and not (a.output/'FINALIZATION_FAILURE.json').exists():
            C.save(a.output/'FINALIZATION_FAILURE.json',{'status':'BARRIER_REJECTED_NO_DEV_PERMITTED','exception':type(e).__name__,'message':str(e)})
        raise
