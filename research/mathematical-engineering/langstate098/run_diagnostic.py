"""No optimizer, no capability score: exact-source state separation diagnostics.

Run: python run_diagnostic.py --output results/DIAGNOSTIC.json
Use --source-model PATH to test the pinned original file in a full checkout.
"""
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
EXPECTED_BLOB = "1636ec81fe0a23e631f7f4da67914d365883d3ef"
SEEDS = (98001, 98002, 98003)


def digest_model(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def close(x, y):
    return bool(torch.allclose(x, y, atol=2e-5, rtol=2e-5))


def maxdiff(x, y):
    return float((x-y).abs().max())


def state_diff(a, b):
    return {k: maxdiff(a.tensors[k], b.tensors[k]) for k in a.tensors}


def clone_state(state, names=None, zero=False):
    chosen = set(state.tensors) if names is None else set(names)
    return replace(state, tensors={k: (torch.zeros_like(v) if zero and k in chosen
                                       else v.detach().clone()) for k, v in state.tensors.items()})


def snapshot(state):
    """Bytes owned by caller, without Python pickle objects or target fields."""
    buff = io.BytesIO()
    torch.save({'owners': state.owners, 'clock': state.clock,
                'tensors': {k: v.detach().clone() for k, v in state.tensors.items()}}, buff)
    return buff.getvalue()


def restore(blob, state_type):
    payload = torch.load(io.BytesIO(blob), weights_only=True)
    return state_type(tuple(payload['owners']), payload['clock'], payload['tensors'])


def timed(fn):
    for _ in range(5): fn()
    values = []
    for _ in range(30):
        start = time.perf_counter_ns(); fn()
        values.append((time.perf_counter_ns()-start)/1e6)
    return {'p50_ms': float(np.percentile(values, 50)),
            'p95_ms': float(np.percentile(values, 95)), 'samples_ms': values}


def profile(fn):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                with_flops=True, record_shapes=True) as prof:
        fn()
    events = prof.key_averages()
    return {'supported_operator_FLOPs': sum(e.flops for e in events),
            'counted_operators': [{'name': e.key, 'calls': e.count, 'flops': e.flops}
                                  for e in events if e.flops],
            'uncounted_operator_names': [e.key for e in events if not e.flops],
            'scope': 'ATen profiler supported operations only; not total model or hardware FLOPs'}


def run(args):
    path = args.source_model or ROOT/'source/model.py'
    source = path.read_bytes()
    blob = hashlib.sha1(b'blob '+str(len(source)).encode()+b'\0'+source).hexdigest()
    if blob != EXPECTED_BLOB:
        raise ValueError('Original source Git blob mismatch; refuse model substitution')
    spec = importlib.util.spec_from_file_location('state_diag_original098', path)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    checks, observations, weights = [], [], []

    def check(seed, name, passed, evidence=None):
        checks.append({'seed': seed, 'name': name, 'status': 'PASS' if passed else 'FAIL',
                       'evidence': evidence})

    with torch.inference_mode():
        for seed in SEEDS:
            torch.manual_seed(seed)
            m = module.ContextualAssociativeTAC098().eval()
            before = digest_model(m)
            owners = ('paired-left', 'paired-right')
            # Only the first byte differs; the last 31 observed bytes are identical.
            history = torch.full((2, 32), 46, dtype=torch.long)
            history[:, 0] = torch.tensor([65, 66])
            query = torch.tensor([[63], [63]])
            _, carried = m(history, owners=owners)
            full, updated = m(query, state=carried, owners=owners)
            fresh_logits, fresh_state = m(query, owners=owners)
            reset = clone_state(carried, zero=True)
            reset_logits, _ = m(query, state=reset, owners=owners)
            check(seed, 'parameter_and_state_counts',
                  m.counts()['trainable'] == 100370 and m.start(('s',)).tensor_bytes() == 1728,
                  {'parameters': m.counts(), 'state_bytes_per_stream': 1728})
            # Batched identical rows may differ by SIMD rounding. Use the already
            # registered FP32 tolerance, plus exact singleton/argmax controls.
            single_a,_=m(query[:1],owners=('identical',))
            single_b,_=m(query[1:],owners=('identical',))
            check(seed, 'current_only_has_equal_predictions_for_conflicting_histories',
                  close(fresh_logits[0],fresh_logits[1]) and torch.equal(single_a,single_b)
                  and int(fresh_logits[0].argmax())==int(fresh_logits[1].argmax()),
                  {'balanced_binary_accuracy_ceiling': .5, 'batched_max_delta':maxdiff(fresh_logits[0],fresh_logits[1]),
                   'singleton_bit_identical':torch.equal(single_a,single_b),
                   'reason': 'Identical actual model inputs; opposite labels. No learning result.'})
            check(seed, 'full_reset_preserving_clock_erases_pair_distinction',
                  close(reset_logits[0], reset_logits[1])
                  and int(reset_logits[0].argmax())==int(reset_logits[1].argmax()),
                  {'clock': reset.clock, 'max_pair_delta': maxdiff(reset_logits[0], reset_logits[1])})
            delta = maxdiff(full[0], full[1])
            check(seed, 'historical_byte_changes_carried_prediction', delta > 1e-8,
                  {'max_logit_delta': delta, 'not_evidence_of_correct_recall': True})
            packed = snapshot(carried); loaded = restore(packed, module.State098)
            external, external_updated = m(query, state=loaded, owners=owners)
            check(seed, 'external_snapshot_reproduces_prediction_and_state_exactly',
                  torch.equal(full, external) and all(v == 0 for v in state_diff(updated, external_updated).values()),
                  {'serialized_bytes_two_streams_including_container': len(packed)})
            replay, replay_state = m(torch.cat((history, query), 1), owners=owners)
            err = maxdiff(full, replay[:, -1:])
            check(seed, 'raw_external_history_replay_matches_streaming', close(full, replay[:, -1:])
                  and all(close(updated.tensors[k], replay_state.tensors[k]) for k in updated.tensors),
                  {'max_logit_delta': err, 'state_deltas': state_diff(updated, replay_state)})
            # Decoder-only mutation: final state must not depend on the predictive head.
            backup = {k: v.detach().clone() for k,v in m.state_dict().items()}
            m.ffn_out.weight.mul_(-2); m.ffn_out.bias.add_(.5)
            altered, altered_state = m(query, state=carried, owners=owners)
            check(seed, 'decoder_parameters_do_not_enter_state_update',
                  all(v == 0 for v in state_diff(updated, altered_state).values())
                  and maxdiff(full, altered) > 1e-8,
                  {'prediction_change': maxdiff(full, altered), 'state_deltas': state_diff(updated, altered_state)})
            m.load_state_dict(backup)
            # Predictor interface: the complete 162D feature, not memory alone.
            captured = []
            hook = m.norm.register_forward_pre_hook(lambda _, inp: captured.append(inp[0]))
            original, _ = m(query, state=carried, owners=owners); hook.remove()
            z = m.norm(captured[0])
            head_only = m.ffn_out(m.ffn_up(z)*torch.nn.functional.silu(m.ffn_gate(z)))
            check(seed, 'explicit_stateless_head_matches_source', torch.equal(original, head_only),
                  {'feature_width': captured[0].shape[-1], 'head_parameters': sum(p.numel() for name,p in m.named_parameters()
                   if name.startswith(('norm.', 'ffn_up.', 'ffn_gate.', 'ffn_out.')))})
            _, nowrite = m(query, state=carried, owners=owners,
                            write_mask=torch.zeros_like(query, dtype=torch.bool))
            diffs = state_diff(carried, nowrite)
            check(seed, 'no_bank_writes_does_not_mean_stateless',
                  diffs['memory'] == 0 and diffs['mass'] == 0
                  and any(diffs[k] > 0 for k in ('context','lexical','previous_key')),
                  {'state_deltas': diffs, 'clock_advance': nowrite.clock-carried.clock})
            changes = {}
            for label, names in [('all', None), ('banks', ['memory','mass']),
                                 ('context', ['context']), ('lexical', ['lexical']),
                                 ('previous_key', ['previous_key'])]:
                out, _ = m(query, state=clone_state(carried,names,True), owners=owners)
                changes[label] = maxdiff(full,out)
            donor = replace(carried, tensors={k:v.flip(0).clone() for k,v in carried.tensors.items()})
            swapped, _ = m(query,state=donor,owners=owners)
            check(seed, 'same_clock_donor_swap_swaps_predictions', close(swapped,full.flip(0)),
                  {'max_logit_delta': maxdiff(full,swapped)})
            singles=[]
            for i in range(2):
                ss=module.State098((owners[i],),carried.clock,{k:v[i:i+1] for k,v in carried.tensors.items()})
                singles.append(m(query[i:i+1],state=ss,owners=(owners[i],))[0])
            check(seed, 'batch_entity_isolation', close(full,torch.cat(singles)),
                  {'max_logit_delta': maxdiff(full,torch.cat(singles))})
            poisoned_owners=('renamed-a','renamed-b')
            renamed=replace(carried,owners=poisoned_owners)
            named,_=m(query,state=renamed,owners=poisoned_owners)
            check(seed, 'owner_names_do_not_encode_predictions', torch.equal(named,full))
            # Future bytes cannot affect any pre-future output/state.
            future_a=torch.cat((history,query,torch.zeros((2,12),dtype=torch.long)),1)
            future_b=future_a.clone();future_b[:,33:]=255
            a,_=m(future_a,owners=owners);b,_=m(future_b,owners=owners)
            check(seed, 'future_byte_poison_preserves_all_earlier_logits',torch.equal(a[:,:33],b[:,:33]))
            repeat,_=m(query,state=carried,owners=owners)
            check(seed, 'deterministic_no_hidden_model_owned_state',torch.equal(full,repeat))
            check(seed, 'bank_invariant',bool((carried.tensors['memory'].abs() <= carried.tensors['mass'][...,None,:]+2e-6).all()))
            check(seed, 'weights_unchanged',digest_model(m)==before)
            weights.append({'seed':seed,'weight_sha256':before})
            args.output.parent.mkdir(parents=True,exist_ok=True)
            torch.save({'seed':seed,'model_state_dict':m.state_dict(),
                        'weight_sha256':before,'history':history,'query':query,
                        'owners':owners,'saved_state':{'clock':carried.clock,'tensors':carried.tensors},
                        'expected_logits':full},args.output.parent/f'INITIAL_AND_REPLAY_{seed}.pt')
            observations.append({'seed':seed,'intervention_max_logit_changes':changes,
                                 'history_pair_logit_change':delta})

        # Reuse the final initialization only for cost accounting, no model selection.
        costs=[];o=('cost-stream',)
        for length in (64,256,1024):
            h=(torch.arange(length).reshape(1,-1)%256)
            q=torch.tensor([[63]])
            _,s=m(h,owners=o)
            packed=snapshot(s)
            operations={
                'A_current_only':lambda:m(q,owners=o),
                'C_carried_step':lambda:m(q,state=s,owners=o),
                'B_external_snapshot_load_and_step':lambda:m(q,state=restore(packed,module.State098),owners=o),
                'B_raw_history_replay':lambda:m(torch.cat((h,q),1),owners=o),
                'prefill_history':lambda:m(h,owners=o)}
            for mode,fn in operations.items():
                costs.append({'history_bytes':length,'mode':mode,**timed(fn),
                              **profile(fn), 'state_payload_bytes':1728,
                              'persisted_model_state_bytes':1728 if mode in ('C_carried_step','B_external_snapshot_load_and_step') else 0,
                              'external_raw_history_bytes':length if mode=='B_raw_history_replay' else 0,
                              'snapshot_container_bytes':len(packed)})

    # Autograd outside inference_mode; one future-only loss, zero optimizer steps.
    gradients=[]
    for seed in SEEDS:
        torch.manual_seed(seed);m=module.ContextualAssociativeTAC098()
        o=('gradient-stream',);h=torch.tensor([[65]+[46]*31]);q=torch.tensor([[63]])
        _,s=m(h,owners=o)
        for name,v in s.tensors.items():
            if v.requires_grad:v.retain_grad()
        out,_=m(q,state=s,owners=o)
        loss=torch.nn.functional.cross_entropy(out[:,0],torch.tensor([66]))
        loss.backward()
        state_grad={k:float(v.grad.abs().sum()) if v.grad is not None else 0. for k,v in s.tensors.items()}
        params={k:float(p.grad.abs().sum()) if p.grad is not None else 0. for k,p in m.named_parameters()}
        check(seed,'future_loss_reaches_acquired_state',sum(state_grad.values())>0
              and all(np.isfinite(v) for v in state_grad.values()),state_grad)
        m.zero_grad(set_to_none=True)
        _,s=m(h,owners=o); original_leaves=list(s.tensors.values())
        for v in original_leaves:v.retain_grad()
        detached=clone_state(s)
        out,_=m(q,state=detached,owners=o)
        torch.nn.functional.cross_entropy(out[:,0],torch.tensor([66])).backward()
        check(seed,'detached_state_blocks_acquisition_graph',all(v.grad is None for v in original_leaves))
        gradients.append({'seed':seed,'state_gradient_L1':state_grad,'parameter_gradient_L1':params,
                          'scope':'Gradient connectivity only, not evidence of learned retention'})

    cpu=next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines()
              if l.startswith('model name')),'unknown')
    result={'study_id':'LANGMATH098-STATE-SEPARATION-DIAG001',
            'qualification':'UNTRAINED_MECHANISM_ONLY_NOT_CAPABILITY',
            'source_commit':'427db08e1d2c971b4fb4c3c579214e289211762e',
            'source_git_blob':blob,'source_sha256':hashlib.sha256(source).hexdigest(),
            'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest(),
            'optimizer_steps':0,'trained_checkpoints_loaded':False,'protected_TEST_access':False,
            'cpu':cpu,'platform':platform.platform(),'torch':torch.__version__,'threads':1,
            'dtype':'FP32','seeds':SEEDS,'checks':checks,'observations':observations,
            'gradients':gradients,'initial_weight_identities':weights,'costs':costs,
            'D_predictive_state_status':'NOT_IMPLEMENTED_OR_TRAINED_IN_PINNED_SOURCE',
            'learning_transfer_closed_loop_intelligence':'NOT_TESTED',
            'passed':sum(c['status']=='PASS' for c in checks),
            'failed':sum(c['status']=='FAIL' for c in checks)}
    if args.output.exists():raise FileExistsError('Preserve original diagnostic receipt')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'passed':result['passed'],'failed':result['failed'],
                      'observations':observations,'costs':[{k:r[k] for k in
                       ('history_bytes','mode','p50_ms','p95_ms','supported_operator_FLOPs')} for r in costs]}))
    return int(result['failed']>0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-model',type=Path);sys.exit(run(p.parse_args()))
