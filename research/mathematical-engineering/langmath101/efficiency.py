"""Bounded CPU inference measurements of already sealed LANGMATH101 checkpoints.

No optimizer, FIT/DEV/TEST data or model selection. The calling workflow must
seal every planned fit before invoking this single-checkpoint measurement.
All models emit byte logits: output tokens/sec therefore means bytes/sec.
Adaptive representation events and BPE recomputation are reported separately.
"""
import argparse
from dataclasses import fields, is_dataclass
import gc
import hashlib
import importlib.util
import math
from pathlib import Path
import signal
import statistics
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
sp = importlib.util.spec_from_file_location('langmath101_common', HERE/'common.py')
C = importlib.util.module_from_spec(sp)
sys.modules[sp.name] = C
sp.loader.exec_module(C)

OWNERS = ('efficiency-stream-0',)
PREFILLS = (64, 256, 1024)
GENERATED = 64
WARMUPS = 1
REPEATS = 3
FIXTURE = b'def tiny_function(value):\n    return value + 1\n'


class BudgetExceeded(TimeoutError):
    pass


def fixture(length):
    raw = (FIXTURE*((length+len(FIXTURE)-1)//len(FIXTURE)))[:length]
    return torch.tensor(list(raw), dtype=torch.long).reshape(1, length)


def state_memory(value):
    """Reachable state payload; backing storage also catches retained views.

    Python sizes are an estimate of reachable objects, including shared Python
    singletons, not incremental allocations or an isolated resident-set delta.
    Tensor wrapper sizes exclude the separately counted underlying storage.
    """
    seen, storage_seen = set(), set()
    out = {'tensor_objects': 0, 'tensor_logical_bytes': 0,
           'unique_tensor_backing_storage_bytes': 0,
           'estimated_python_object_bytes': 0,
           'non_tensor_bytes_payload_bytes': 0, 'string_utf8_payload_bytes': 0}

    def walk(v):
        if id(v) in seen:
            return
        seen.add(id(v))
        out['estimated_python_object_bytes'] += sys.getsizeof(v)
        if isinstance(v, torch.Tensor):
            out['tensor_objects'] += 1
            out['tensor_logical_bytes'] += v.numel()*v.element_size()
            storage = v.untyped_storage()
            key = (str(v.device), storage.data_ptr(), storage.nbytes())
            if key not in storage_seen:
                storage_seen.add(key)
                out['unique_tensor_backing_storage_bytes'] += storage.nbytes()
        elif is_dataclass(v) and not isinstance(v, type):
            for f in fields(v):
                walk(getattr(v, f.name))
            # Dataclass attribute-table allocation is absent from getsizeof(v).
            if hasattr(v, '__dict__'):
                out['estimated_python_object_bytes'] += sys.getsizeof(v.__dict__)
        elif isinstance(v, dict):
            for k, x in v.items():
                walk(k)
                walk(x)
        elif isinstance(v, (tuple, list)):
            for x in v:
                walk(x)
        elif isinstance(v, bytes):
            out['non_tensor_bytes_payload_bytes'] += len(v)
        elif isinstance(v, str):
            out['string_utf8_payload_bytes'] += len(v.encode('utf-8'))
        elif v is not None and type(v) not in (int, float, bool):
            raise TypeError('Uncounted state object: '+str(type(v)))

    walk(value)
    out['estimated_state_bytes_with_backing_storage'] = (
        out['estimated_python_object_bytes']+out['unique_tensor_backing_storage_bytes'])
    out['estimate_scope'] = 'reachable Python objects plus unique tensor storage; not an RSS delta'
    return out


def state_signature(state):
    if not is_dataclass(state) or not isinstance(state.tensors, dict):
        raise TypeError('Expected source state dataclass with tensors dictionary')
    if state.owners != OWNERS or type(state.clock) is not int or state.clock < 0:
        raise ValueError('Returned stream ownership/clock is invalid')
    signature = {'type': type(state).__module__+'.'+type(state).__qualname__,
                 'owners': list(state.owners), 'clock': state.clock, 'tensors': {}}
    for name, tensor in state.tensors.items():
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu':
            raise TypeError('CPU tensor state required')
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError('Nonfinite carried state: '+name)
        signature['tensors'][name] = {'shape': list(tensor.shape),
                                     'dtype': str(tensor.dtype),
                                     'logical_bytes': tensor.numel()*tensor.element_size()}
    signature['metadata_fields'] = [f.name for f in fields(state) if f.name != 'tensors']
    signature['source_tensor_bytes_method'] = state.tensor_bytes()
    return signature


def adaptive_receipts(receipts, seconds):
    if not receipts:
        return None
    counters = {name: sum(int(r.get(name, 0)) for r in receipts) for name in
                ('source_bytes', 'byte_readouts', 'recurrent_updates',
                 'candidate_recurrent_calls', 'reused_prefix_hidden_states')}
    kinds = {}
    represented_bytes = 0
    for receipt in receipts:
        for event in receipt.get('events', ()):
            kind = {0: 'byte', 1: 'discrete', 2: 'latent'}[event['kind']]
            kinds[kind] = kinds.get(kind, 0)+1
            represented_bytes += int(event['bytes'])
    counters.update({'representation_events_by_kind': kinds,
                     'committed_representation_events': sum(kinds.values()),
                     'committed_representation_source_bytes': represented_bytes,
                     'executed_recurrent_updates_per_second': counters['recurrent_updates']/seconds,
                     'candidate_recurrent_calls_per_second': counters['candidate_recurrent_calls']/seconds,
                     'last_call': receipts[-1],
                     'bpe_update_semantics': 'BPE recurrent updates include recomputed token suffixes; not newly emitted tokens',
                     'representation_tokens_per_second': sum(kinds.values())/seconds if kinds else None})
    return counters


def measured_call(model, arm, tokens, state=None):
    kwargs = {'owners': OWNERS, 'state': state}
    if arm.startswith('a5_'):
        kwargs['execution'] = 'inference'
    logits, next_state = model(tokens, **kwargs)
    if logits.shape != (1, tokens.shape[1], 256):
        raise ValueError('Unexpected next-byte logit shape')
    return logits, next_state


def state_record(state, arm, observed, external_window=None):
    signature = state_signature(state)
    stateless = arm in ('stateless_current', 'stateless_window4')
    if signature['clock'] != (0 if stateless else observed):
        raise ValueError('Returned state clock does not match observed source bytes')
    out = {'signature': signature, 'model_state_memory': state_memory(state)}
    if external_window is not None:
        out['external_input_window_memory'] = state_memory(external_window)
        out['external_input_source_bytes'] = external_window.numel()
        out['combined_state_and_external_window_memory'] = state_memory((state, external_window))
    return out


def one_prefill(model, arm, length, check):
    tokens = fixture(length)
    check()
    start = time.perf_counter_ns()
    logits, state = measured_call(model, arm, tokens)
    elapsed = (time.perf_counter_ns()-start)/1e9
    check()
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('Nonfinite prefill predictions')
    receipt = getattr(model, 'last_receipt', None)
    return {'seconds': elapsed, 'state': state_record(state, arm, length),
            'adaptive': adaptive_receipts([receipt] if receipt else [], elapsed),
            'logits_sha256': hashlib.sha256(logits.contiguous().numpy().tobytes()).hexdigest()}


def one_decode(model, arm, check):
    prefix = fixture(64)
    logits, state = measured_call(model, arm, prefix)
    last = logits[:, -1]
    window = 4 if arm == 'stateless_window4' else 1 if arm == 'stateless_current' else None
    external = prefix[:, -window:].clone() if window else None
    emitted, receipts = [], []
    check()
    start = time.perf_counter_ns()
    for _ in range(GENERATED):
        check()
        new_byte = last.argmax(-1, keepdim=True)
        emitted.append(new_byte)
        if window:
            external = torch.cat((external, new_byte), 1)[:, -window:].clone()
            logits, state = measured_call(model, arm, external)
        else:
            logits, state = measured_call(model, arm, new_byte, state)
        last = logits[:, -1]
        if arm.startswith('a5_'):
            receipts.append(model.last_receipt)
    elapsed = (time.perf_counter_ns()-start)/1e9
    check()
    if not bool(torch.isfinite(last).all()):
        raise ValueError('Nonfinite decoding predictions')
    raw = bytes(torch.cat(emitted, 1).flatten().tolist())
    return {'seconds': elapsed, 'emitted_source_bytes': len(raw),
            'emitted_sha256': hashlib.sha256(raw).hexdigest(),
            'state': state_record(state, arm, 64+GENERATED, external),
            'adaptive': adaptive_receipts(receipts, elapsed)}


def summarize(samples, source_bytes, direction):
    times = [s['seconds'] for s in samples]
    p50 = statistics.median(times)
    if len(set(s.get('emitted_sha256', s.get('logits_sha256')) for s in samples)) != 1:
        raise ValueError('Repeated deterministic workload produced different predictions')
    return {'status': 'MEASURED', 'samples_seconds': times, 'p50_seconds': p50,
            'min_seconds': min(times), 'max_seconds': max(times),
            'source_bytes_per_trial': source_bytes,
            direction+'_source_bytes_per_second_at_p50': source_bytes/p50,
            'model_byte_output_tokens_per_second_at_p50': source_bytes/p50,
            'model_output_token_definition': 'one token is one of 256 byte IDs',
            'repeated_predictions': 'EXACT', 'trials': samples}


def measure(a):
    if a.output.exists():
        raise FileExistsError('Never overwrite original efficiency output')
    if not math.isfinite(a.max_seconds) or not 1 <= a.max_seconds <= 60:
        raise ValueError('Benchmark budget must be between 1 and 60 seconds')
    started = time.perf_counter()
    payload = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    meta = payload['metadata']
    if meta.get('stage') != 'SEALED_BEFORE_DEV' or int(meta.get('optimizer_steps', 0)) <= 0:
        raise ValueError('Only genuinely fitted SEALED_BEFORE_DEV checkpoints may be measured')
    C.setup(meta['seed'])
    if meta.get('source_hashes') != C.source_hashes():
        raise ValueError('Inference source differs from the exact checkpoint training source')
    model = C.load_model(payload)
    del payload
    gc.collect()
    if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('This frozen timing fixture requires CPU FP32 parameters')
    before = C.state_digest(model)
    arm = meta['arm']
    hardware = C.hardware()
    result = {'status': 'MEASURED', 'arm': arm, 'seed': meta['seed'], 'lane': meta['lane'],
              'checkpoint_sha256': C.sha(a.checkpoint), 'model_sha256': before,
              'checkpoint_stage': meta['stage'], 'checkpoint_optimizer_steps': meta['optimizer_steps'],
              'source_hashes': meta['source_hashes'], 'parameters': model.counts(),
              'parameter_tensor_bytes': sum(p.numel()*p.element_size() for p in model.parameters()),
              'fixed_buffer_tensor_bytes': sum(b.numel()*b.element_size() for b in model.buffers()),
              'hardware': hardware, 'precision': 'FP32', 'batch_size': 1,
              'inference_mode': True, 'warmup_trials': WARMUPS, 'measured_trials': REPEATS,
              'max_seconds': a.max_seconds, 'timing_clock': 'perf_counter_ns, CPU synchronous',
              'timing_fixture': {'kind': 'constructed code-like bytes; no capability data',
                                 'repeated_literal_hex': FIXTURE.hex()},
              'global_all_fit_seal': 'MUST_BE_ENFORCED_BY_CALLING_WORKFLOW; not established by one checkpoint',
              'full_model_FLOPs': 'INCONCLUSIVE_NOT_MEASURED', 'energy': 'NOT_MEASURED',
              'optimizer_steps_performed': 0, 'protected_TEST_access': False,
              'prefill': {}, 'decode': None,
              'decode_policy': 'greedy 64 emitted bytes after 64 observed prefix bytes; includes observing all 64 emissions',
              'decode_timing_scope': 'excludes prefix prefill; includes argmax, model updates and external-window/counter bookkeeping',
              'process_peak_rss_scope': 'whole process high-water mark, including checkpoint load and allocator retention; not isolated model memory'}
    if arm in ('stateless_current', 'stateless_window4'):
        result['external_input_policy'] = ('rolling last '+str(4 if arm == 'stateless_window4' else 1)+
                                           ' observed source bytes, supplied afresh with no carried native state')
    elif arm == 'transformer':
        result['history_policy'] = 'explicit full observed-byte history; full-prefix attention recomputation; no KV cache'
    elif arm == 'a5_fixed_bpe':
        result['history_policy'] = 'explicit raw prefix, token IDs and unchanged-token-prefix GRU hidden cache; changed suffix recomputed'
    else:
        result['history_policy'] = 'source native carried state; A5 selected-route inference where applicable'
    if hasattr(model, 'configuration'):
        result['configuration'] = model.configuration()

    deadline = started+a.max_seconds

    def check():
        if time.perf_counter() >= deadline:
            raise BudgetExceeded('Frozen per-model wall-clock budget exhausted')

    def alarm_handler(signum, frame):
        raise BudgetExceeded('Frozen per-model wall-clock budget exhausted')

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, alarm_handler)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.perf_counter()))
    current = None
    try:
        with torch.inference_mode():
            for length in PREFILLS:
                current = ('prefill', str(length))
                result['prefill'][str(length)] = {'status': 'STARTED'}
                for _ in range(WARMUPS):
                    one_prefill(model, arm, length, check)
                samples = [one_prefill(model, arm, length, check) for _ in range(REPEATS)]
                result['prefill'][str(length)] = summarize(samples, length, 'processed')
            current = ('decode', None)
            result['decode'] = {'status': 'STARTED'}
            for _ in range(WARMUPS):
                one_decode(model, arm, check)
            samples = [one_decode(model, arm, check) for _ in range(REPEATS)]
            result['decode'] = summarize(samples, GENERATED, 'emitted')
    except BudgetExceeded as e:
        result['status'] = 'INCONCLUSIVE_BUDGET'
        result['budget_message'] = str(e)
        if current and current[0] == 'prefill':
            result['prefill'][current[1]] = {'status': 'INCONCLUSIVE_BUDGET'}
        elif current:
            result['decode'] = {'status': 'INCONCLUSIVE_BUDGET'}
    except Exception as e:
        result['status'] = 'FAILED'
        result['failure'] = {'exception': type(e).__name__, 'message': str(e)}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if previous_timer[0] or previous_timer[1]:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)
    for length in PREFILLS:
        result['prefill'].setdefault(str(length), {'status': 'NOT_MEASURED_BUDGET' if result['status'] == 'INCONCLUSIVE_BUDGET' else 'NOT_MEASURED'})
    if result['decode'] is None:
        result['decode'] = {'status': 'NOT_MEASURED_BUDGET' if result['status'] == 'INCONCLUSIVE_BUDGET' else 'NOT_MEASURED'}
    result['weights_unchanged'] = C.state_digest(model) == before
    if not result['weights_unchanged']:
        result['status'] = 'FAILED'
        result['failure'] = {'exception': 'ValueError', 'message': 'Inference changed parameter bytes'}
    result['elapsed_seconds_including_checkpoint_load'] = time.perf_counter()-started
    result['peak_rss_bytes'] = C.hardware()['peak_rss_bytes']
    result['hardware']['peak_rss_bytes'] = result['peak_rss_bytes']
    C.save(a.output, result)
    print(C.digest(result), result['status'], arm, flush=True)
    return 0 if result['status'] == 'MEASURED' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=60.)
    sys.exit(measure(parser.parse_args()))
