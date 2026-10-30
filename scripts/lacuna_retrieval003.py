#!/usr/bin/env python3
"""Preregistered frozen-source read interventions. No model fits or TEST access."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import struct
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.replay_langmem086087 import digest, fit_sources, load_source, original_checkpoints, weights
from tac_osm.research_probe import probe_memory, probe_dependencies

REG = ROOT / 'research/mathematical-engineering/lacuna_retrieval003'
REFERENCE_SHA256 = '05c1383ce4ed853c730f33c5fb2b0a5d31a6438314aa8213d9e2467929123a4c'
BINS = ((8, 32), (32, 128), (128, 240))
ARMS = ('A', 'B', 'C', 'D', 'matrix_wrong', 'raw_wrong',
        'A_shuffle', 'B_shuffle', 'C_shuffle', 'D_shuffle')


class RawRing:
    """256 actual payload bytes: uint64 clock +248 observed bytes. No side index."""
    def __init__(self):
        self.data = bytearray(256)

    @property
    def clock(self):
        return struct.unpack_from('<Q', self.data)[0]

    def append(self, byte):
        t = self.clock
        self.data[8 + t % 248] = byte
        struct.pack_into('<Q', self.data, 0, t + 1)

    def byte(self, t):
        if not max(0, self.clock - 248) <= t < self.clock:
            raise IndexError('outside retained raw history')
        return self.data[8 + t % 248]

    def context(self, j):
        return bytes(self.byte(i) for i in range(j - 4, j))

    def candidates(self):
        # Last appended byte has clock self.clock-1; exclude its new write.
        return list(range(max(4, self.clock - 248 + 4), self.clock - 1))


def source_docs():
    receipt = json.loads((REG / 'SOURCE_RECEIPT.json').read_text())
    docs = {}
    for row in receipt['documents']:
        raw = (ROOT / 'research/source-bundles/LACUNA-RETRIEVAL003' / row['path']).read_bytes()
        blob = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
        if len(raw) != row['bytes'] or digest(raw) != row['sha256'] or blob != row['git_blob']:
            raise ValueError('publisher blob/hash mismatch')
        if row['path'] != 'LICENSE':
            docs[row['path']] = raw
    if len(docs) != 2:
        raise ValueError('frozen two-owner cohort')
    return docs


def reconstruct(model, ring, indices):
    """Rebuild keys/values from ring only, without external source bytes/state."""
    device = model.embedding.weight.device
    context = torch.tensor([list(ring.context(j)) for j in indices], device=device)
    lexical = model.embedding(context).tanh()
    local = lexical[:, 3].clone()
    for lag in range(3):
        local = local + model.context_lag[lag].tanh() * lexical[:, 2 - lag]
    keys = F.normalize(model.writer(local.tanh()), dim=-1, eps=1e-6)
    tokens = torch.tensor([ring.byte(j) for j in indices], device=device)
    t = torch.tensor(indices, dtype=torch.float32, device=device)
    pos = torch.stack((torch.log1p(t) / 8, torch.sin(t / 16), torch.cos(t / 16),
                       torch.sin(t / 64), torch.cos(t / 64),
                       torch.sin(t / 256), torch.cos(t / 256)), -1)
    e = (model.embedding(tokens) + model.position(pos)).tanh()
    values = F.normalize(model.byte_codes[tokens] + .1 * model.value_residual(e).tanh(), dim=-1, eps=1e-6)
    return keys.double(), values.double()


def attention(keys, values, query):
    w = (8 * (keys @ query)).softmax(0)
    return w @ values, w


def magnitude_match(vector, reference):
    norm = vector.norm()
    return torch.zeros_like(vector) if float(norm) < 1e-12 else vector * (reference.norm() / norm)


def byte_read(vector, codes):
    direction = F.normalize(vector, dim=0, eps=1e-12)
    return (8 * (codes @ direction)).log_softmax(0)


@contextmanager
def override_slow_read(model, trace, replacement):
    """Patch every original native-read consumer, never model weights or time state."""
    native = trace['mix'][:, :1] * trace['read_fast'] + trace['mix'][:, 1:] * trace['read_slow']
    delta = trace['mix'][:, 1:] * (replacement.reshape(1, 8) - trace['read_slow'])
    hits = {'fast': 0, 'slow': 0, 'norm': 0}

    def hook(name):
        def apply(module, args):
            x = args[0]
            sl = slice(96, 104) if name == 'norm' else slice(-8, None)
            if x.ndim != 2 or not torch.allclose(x[:, sl], native, atol=3e-6, rtol=3e-6):
                raise ValueError('original source native-read consumer changed')
            out = x.clone()
            out[:, sl] = out[:, sl] + delta.to(x.dtype)
            hits[name] += 1
            return (out,)
        return apply

    handles = [getattr(model, name).register_forward_pre_hook(hook(name)) for name in hits]
    try:
        yield hits
        if set(hits.values()) != {1}:
            raise ValueError('single-clock read consumer coverage')
    finally:
        for handle in handles:
            handle.remove()


def patched_logits(model, token, prior, owner, trace, read):
    with override_slow_read(model, trace, read):
        return model(token, state=prior, owners=(owner,))[0][0, 0]


def prepare_reads(model, ring, trace, old_bank, current, keys, values, coefficients, indices, j):
    """Prediction preparation has no future label argument or source-document access."""
    q = trace['query'][0].double()
    rk, rv = reconstruct(model, ring, indices)
    selected = indices.index(j)
    a = rk[selected]
    wrongs = [i for i, b in enumerate(indices)
              if abs(b - j) >= 5 and not b - 4 <= j < b and ring.context(b) != ring.context(j)]
    if not wrongs:
        return None
    wi = max(wrongs, key=lambda i: (float(rk[i] @ a), -indices[i]))
    wrong = rk[wi]
    raw_q, wq = attention(rk, rv, q)
    raw_a, wa = attention(rk, rv, a)
    old = old_bank @ q
    shuffled_bank = values.roll(1, dims=0).T @ coefficients
    raw_shift = rv.roll(1, dims=0)
    reads = {'A': old, 'B': old_bank @ a, 'C': raw_q, 'D': raw_a,
             'matrix_wrong': old_bank @ wrong, 'raw_wrong': attention(rk, rv, wrong)[0],
             'A_shuffle': shuffled_bank @ q, 'B_shuffle': shuffled_bank @ a,
             'C_shuffle': attention(rk, raw_shift, q)[0],
             'D_shuffle': attention(rk, raw_shift, a)[0]}
    replacements = {name: (magnitude_match(read, old) + current @ q).float() for name, read in reads.items()}
    cosine = rk @ a
    singular = torch.linalg.svdvals(old_bank)
    p = singular / singular.sum().clamp_min(1e-30)
    info = {'wrong_clock': indices[wi], 'raw_records': len(indices),
            'native_old_read_norm': float(old.norm()),
            'native_query_target_attention': float(wq[selected]),
            'supplied_key_target_attention': float(wa[selected]),
            'supplied_key_equal_key_count': int((cosine > 1 - 1e-6).sum()),
            'native_query_target_rank': 1 + int(((rk @ q) > rk[selected] @ q + 1e-6).sum()),
            'matrix_singular_values': singular.tolist(),
            'matrix_effective_rank': float((-(p * p.clamp_min(1e-30).log()).sum()).exp()),
            'near_zero_read_count': sum(float(v.norm()) < 1e-12 for v in reads.values())}
    target_read = values[j-1] * (coefficients[j-1] @ a)
    interference = reads['B'] - target_read
    info.update(target_event_correct_key_read_norm=float(target_read.norm()),
                interference_correct_key_read_norm=float(interference.norm()),
                interference_to_target_norm_ratio=float(interference.norm()/target_read.norm().clamp_min(1e-12)),
                correct_key_read_cosine_target_value=float(F.cosine_similarity(reads['B'], values[j-1], dim=0)),
                different_context_keys_cosine_over_point9=sum(float(rk[i]@a)>.9 and ring.context(b)!=ring.context(j) for i,b in enumerate(indices)))
    return reads, replacements, info, rk, rv


def summarize(rows, availability):
    result = []
    for lo, hi in BINS:
        rr = [r for r in rows if r['bin_begin'] == lo and r['status'] == 'MEASURED']
        cells = {(r['owner'], r['clock']) for r in rr}
        summary = {'bin_begin': lo, 'bin_end_exclusive': hi, 'measurements': len(rr),
                   'distinct_source_query_cells': len(cells),
                   'unavailable_or_missing_control': sum(a['bin_begin'] == lo for a in availability) - len(rr),
                   'arms': {}, 'provisional_diagnostic_signals': []}
        for arm in ARMS:
            metrics = {field: sum(r['arms'][arm][field] for r in rr) / len(rr) if rr else None
                       for field in ('past_byte_nll', 'past_byte_correct', 'next_byte_nll', 'original_minus_arm_nll')}
            summary['arms'][arm] = metrics
        for arm in ('A', 'B', 'C', 'D'):
            wrong = 'matrix_wrong' if arm in ('A', 'B') else 'raw_wrong'
            means = [sum(r['arms'][arm]['original_minus_arm_nll'] for r in rr if r[field] == value) /
                     sum(r[field] == value for r in rr)
                     for field, vals in (('seed', (1701, 1702, 1703)), ('owner', sorted({r['owner'] for r in rows})))
                     for value in vals if any(r[field] == value for r in rr)]
            m = summary['arms'][arm]['original_minus_arm_nll']
            missing_control = any(r['status'] != 'MEASURED' for r in rows if r['bin_begin'] == lo)
            if not missing_control and len(cells) >= 20 and len(means) == 5 and min(means) > 0 and m >= .001 and all(
                    m > summary['arms'][control]['original_minus_arm_nll'] for control in (wrong, arm + '_shuffle')):
                summary['provisional_diagnostic_signals'].append(arm)
        summary['support'] = 'ENOUGH_FOR_REGISTERED_SCREEN' if len(cells) >= 20 else 'INCONCLUSIVE_LOW_SUPPORT'
        if rr:
            for field in ('native_query_target_attention', 'supplied_key_target_attention',
                          'native_query_target_rank', 'supplied_key_equal_key_count', 'matrix_effective_rank',
                          'interference_to_target_norm_ratio', 'correct_key_read_cosine_target_value',
                          'different_context_keys_cosine_over_point9'):
                summary[field] = sum(r[field] for r in rr) / len(rr)
            summary['historical_continuation_matches_next_byte'] = sum(r['historical_continuation_matches_next_byte'] for r in rr) / len(rr)
        result.append(summary)
    return result


@torch.no_grad()
def run(output):
    if output.exists():
        raise FileExistsError('preserve original results')
    torch.set_num_threads(2)
    docs = source_docs()
    manifest = json.loads((ROOT / 'research/replication/langmem086087/MANIFEST.json').read_text())
    cls = load_source(ROOT, manifest)
    adapter_spec = importlib.util.spec_from_file_location('retrieval003_native_adapter', ROOT/'research/mathematical-engineering/langmem086/probe_adapter086.py')
    adapter_module = importlib.util.module_from_spec(adapter_spec)
    adapter_spec.loader.exec_module(adapter_module)
    ref_spec = importlib.util.spec_from_file_location('retrieval003_independent_reference', REG/'pnds_reference.py')
    if digest((REG/'pnds_reference.py').read_bytes()) != REFERENCE_SHA256:
        raise ValueError('independent reference source changed')
    ref = importlib.util.module_from_spec(ref_spec)
    ref_spec.loader.exec_module(ref)
    fit_docs, _ = fit_sources(ROOT)
    rows, availability, checks = [], [], []
    started = time.process_time()
    for checkpoint, cp, fit in original_checkpoints(ROOT, manifest):
        if cp['steps'] != 640:
            continue
        model = cls().eval()
        model.load_state_dict(cp['model'], strict=True)
        before = weights(model)
        if model.counts()['trainable'] != 100521:
            raise ValueError('original parameters')
        sample = {'tokens': torch.tensor([list(next(iter(fit_docs.values()))[:9])]), 'owners': ('FIT-adapter-probe',)}
        memory_probe = probe_memory(model, sample, adapter_module.adapter086())
        with torch.enable_grad():
            dependency_probe = probe_dependencies(model, sample, adapter_module.adapter086())
        for owner, raw in docs.items():
            state = model.start((owner,))
            ring = RawRing()
            coef = torch.zeros((4096, 8), dtype=torch.float64)
            vals = torch.zeros_like(coef)
            actual_keys = torch.zeros_like(coef)
            mx = {'transport_error': 0., 'reconstructed_key_error': 0., 'reconstructed_value_error': 0.,
                  'identity_logit_error': 0., 'zero_read_logit_error': 0., 'future_poison_error': 0.,
                  'independent_attention_error': 0., 'independent_norm_match_error': 0.}
            original_nll = []
            gradient = None
            for t in range(4096):
                token = torch.tensor([[raw[t]]])
                prior = state
                logits, state, traces = model(token, state=prior, owners=(owner,), return_trace=True)
                tr = traces[0]
                if state.tensor_bytes() != 1920:
                    raise ValueError('original state bytes')
                ring.append(raw[t])
                e = (model.embedding(token) + model.position(model.positions(t, 1, 1, torch.float32, token.device))).tanh()
                eta = float((.01 * model.slow_gate(e).sigmoid())[0, 0, 0])
                rho = float(.995 + .004 * model.slow_decay.sigmoid())
                k, v = tr['writer'][0].double(), tr['value'][0].double()
                n = max(0, t - 1)
                if t:
                    coef[:n] = rho * (coef[:n] - eta * (coef[:n] @ k)[:, None] * k)
                    current = eta * v[:, None] * k[None, :]
                    old_bank = state.tensors['slow_memory'][0].double() - current
                pending = []
                if t >= 256 and t % 16 == 0:
                    indices = ring.candidates()
                    rk, rv = reconstruct(model, ring, indices)
                    mx['reconstructed_key_error'] = max(mx['reconstructed_key_error'], float((rk - actual_keys[indices]).abs().max()))
                    mx['reconstructed_value_error'] = max(mx['reconstructed_value_error'], float((rv - vals[torch.tensor(indices) - 1]).abs().max()))
                    for lo, hi in BINS:
                        matching = [j for j in indices if lo <= t - j < hi and ring.context(j) == raw[t-3:t+1]]
                        j = max(matching, default=None)
                        availability.append({'seed': cp['seed'], 'owner': owner, 'clock': t, 'bin_begin': lo, 'available': j is not None})
                        if j is None:
                            continue
                        prepared = prepare_reads(model, ring, tr, old_bank, current, actual_keys, vals[:n], coef[:n], indices, j)
                        if prepared is None:
                            rows.append({'seed': cp['seed'], 'owner': owner, 'clock': t, 'bin_begin': lo, 'status': 'INCONCLUSIVE_NO_CLEAN_WRONG_ADDRESS'})
                            continue
                        reads, replacements, info, _, _ = prepared
                        preds = {name: patched_logits(model, token, prior, owner, tr, replacement)
                                 for name, replacement in replacements.items()}
                        lp = {name: byte_read(read, model.byte_codes.double()) for name, read in reads.items()}
                        mx['identity_logit_error'] = max(mx['identity_logit_error'], float((preds['A'] - logits[0, 0]).abs().max()))
                        row = {'seed': cp['seed'], 'owner': owner, 'clock': t, 'bin_begin': lo,
                               'bin_end_exclusive': hi, 'historical_clock': j, 'status': 'MEASURED', **info}
                        pending.append((row, preds, lp, rv[indices.index(j)]))
                        if gradient is None:
                            expected, _ = ref.attention(rk.tolist(), rv.tolist(), tr['query'][0].double().tolist())
                            mx['independent_attention_error'] = max(abs(a-b) for a,b in zip(expected,reads['C'].tolist()))
                            expected_match = ref.match_norm(reads['C'].tolist(), reads['A'].tolist())
                            actual_match = magnitude_match(reads['C'], reads['A']).tolist()
                            mx['independent_norm_match_error'] = max(abs(a-b) for a,b in zip(expected_match,actual_match))
                            zero = patched_logits(model, token, prior, owner, tr, torch.zeros(8))
                            source_zero = model(token, state=prior, owners=(owner,), lesion='slow_zero')[0][0, 0]
                            mx['zero_read_logit_error'] = float((zero - source_zero).abs().max())
                            # Poison future source bytes and independently recompute native prefix state.
                            poisoned = bytearray(raw[:t + 17])
                            poisoned[t+1:] = bytes((b + 113) % 256 for b in poisoned[t+1:])
                            p_logits, p_state, p_trace = model(torch.tensor([list(poisoned[:t+1])]), owners=(owner,), return_trace=True)
                            ps = p_state.tensors['slow_memory'][0].double() - current
                            pring = RawRing()
                            for byte in poisoned[:t+1]:
                                pring.append(byte)
                            pp = prepare_reads(model, pring, p_trace[-1], ps, current, actual_keys, vals[:n], coef[:n], indices, j)
                            mx['future_poison_error'] = max(float((p_logits[0, -1] - logits[0, 0]).abs().max()),
                                max(float((pp[1][name] - replacements[name]).abs().max()) for name in ARMS))
                            label = int(logits[0, 0].argmax())
                            with torch.enable_grad():
                                leaf = tr['read_slow'][0].detach().clone().requires_grad_(True)
                                loss = patched_logits(model, token, prior, owner, tr, leaf).log_softmax(0)[label]
                                grad = torch.autograd.grad(loss, leaf)[0]
                            direction = torch.ones(8) / math.sqrt(8)
                            h = .001
                            plus = patched_logits(model, token, prior, owner, tr, leaf.detach()+h*direction).log_softmax(0)[label]
                            minus = patched_logits(model, token, prior, owner, tr, leaf.detach()-h*direction).log_softmax(0)[label]
                            err = abs(float((grad*direction).sum()) - float((plus-minus)/(2*h)))
                            gradient = {'error': err, 'gradient_norm': float(grad.norm()), 'label': 'prefix_argmax_not_future_target'}
                            if err > .002 or not torch.isfinite(grad).all():
                                raise ValueError('actual read VJP/finite difference')
                if t:
                    coef[t-1] = eta * k
                    vals[t-1] = v
                    actual_keys[t] = k
                err = float((vals[:t].T @ coef[:t] - state.tensors['slow_memory'][0].double()).abs().max())
                mx['transport_error'] = max(mx['transport_error'], err)
                if err > 1e-5:
                    raise ValueError('conditional transport reconstruction')
                # First use of the future target is after every arm/control prediction.
                y = raw[t+1]
                full_nll = -float(logits[0, 0].log_softmax(0)[y])
                if t >= 256:
                    original_nll.append(full_nll)
                for row, preds, lp, oracle in pending:
                    target = raw[row['historical_clock']]
                    row.update(target_byte=target, next_byte=y, original_nll=full_nll,
                               historical_continuation_matches_next_byte=int(target == y),
                               isolated_encoded_value_correct=int(byte_read(oracle, model.byte_codes.double()).argmax()) == target,
                               arms={})
                    for name in ARMS:
                        next_nll = -float(preds[name].log_softmax(0)[y])
                        row['arms'][name] = {'past_byte_nll': -float(lp[name][target]),
                                             'past_byte_correct': int(int(lp[name].argmax()) == target),
                                             'next_byte_nll': next_nll, 'original_minus_arm_nll': full_nll-next_nll}
                    if not all(math.isfinite(x) for a in row['arms'].values() for x in a.values()):
                        raise ValueError('nonfinite scores')
                    rows.append(row)
            if mx['transport_error'] > 1e-5 or any(v > 3e-5 for k, v in mx.items() if k != 'transport_error'):
                raise ValueError('preregistered source correspondence tolerance: ' + str(mx))
            checks.append({'seed': cp['seed'], 'owner': owner, **mx, 'local_read_VJP': gradient,
                           'native_adapter_memory_probe': memory_probe,
                           'native_adapter_dependency_probe': dependency_probe,
                           'uninstrumented_boundaries': ['full_parameter_AD', 'IEEE_proof', 'independent_source_qualification'],
                           'native_all_positions_nll': sum(original_nll)/len(original_nll),
                           'raw_bank_bytes': len(ring.data), 'native_slow_bank_bytes': 256,
                           'native_tensor_state_bytes': 1920, 'native_clock_bytes': 8,
                           'actual_extra_reference_ring_bytes': 256,
                           'decomposition_workspace_bytes': sum(x.numel()*x.element_size() for x in (coef, vals, actual_keys))})
        if weights(model) != before:
            raise ValueError('frozen weights changed')
    result = {'study': 'LACUNA-RETRIEVAL003', 'status': 'PASS_INTEGRITY_FROZEN_NEW_DOCUMENT_DIAGNOSTIC',
              'summary': summarize(rows, availability), 'rows': rows, 'availability': availability,
              'source_checks': checks, 'source_commit': __import__('subprocess').check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'new_training_steps': 0, 'protected_TEST': False, 'GPU_minutes': 0,
              'original_failed_gates_unchanged': True, 'training_authorized_by_result': False,
              'whole_model_mathematical_equivalence': 'OPEN', 'independently_qualified': False,
              'cpu_seconds': time.process_time()-started,
              'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({'summary': result['summary'], 'cpu_seconds': result['cpu_seconds']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
