"""Equivalent replay logger: cache public timestamps instead of rescanning noise.

Only diagnostic label extraction changes. Model calls, inputs, actions, effects,
state updates, costs and reported metrics must match frozen canonical records.
Never used for fitting or claimed as a model inference speed improvement.
"""
import hashlib
import numpy as np
import torch
from run_structured_memory_dev061 import ResearchWorld


@torch.no_grad()
def evaluate(model, seed, n, h, count=32, family='clean'):
    worlds = [ResearchWorld(seed, i, n, h, family) for i in range(count)]
    latest_public = []
    for world in worlds:
        stamps = {}
        for e in world.events:
            if e[0] > 0 or e[2] > 0:
                key = int(e[4:12].argmax())
                stamps[key] = max(stamps.get(key, -1e9), float(e[21]))
        latest_public.append(stamps)
    model.eval()
    state = model.initial(count)
    writes = np.zeros(count, dtype=int)
    telemetry = [{'completion_probability': [], 'stop_probability': [], 'healthy': [], 'current_brier': [], 'stale_reads': 0, 'latest_hits': 0, 'wrong_reads': 0, 'query_reads': 0} for _ in worlds]
    work = {k: 0 for k in ('read_candidates', 'retained_read_slots', 'operator_calls', 'latent_transitions')}
    for t in range(h):
        state, trace = model.observe(state, torch.from_numpy(np.stack([w.events[t] for w in worlds])))
        writes += trace['gate'].numpy().astype(int)
    active = list(range(count))
    for _ in range(2 * n + 4):
        if not active:
            break
        public = [worlds[i].candidates() for i in active]
        values = [torch.from_numpy(np.stack([p[k] for p in public])) for k in range(4)]
        selected = (state[0][active], state[1][active])
        out = model(selected, *values)
        choices = out['logits'].argmax(-1).tolist()
        prob = out['logits'].softmax(-1)
        for k in work:
            work[k] += model.last_work[k]
        remaining = []
        receipts = []
        for j, i in enumerate(active):
            w = worlds[i]
            t = telemetry[i]
            stop = next((k for k, c in enumerate(public[j][4]) if c[0] == 'stop'))
            t['completion_probability'].append(float(out['completion'][j].sigmoid()))
            t['stop_probability'].append(float(prob[j, stop]))
            t['healthy'].append(bool(not w.bits.any()))
            if 'current_logits' in out:
                predictions = out['current_logits'][j, :n].sigmoid().numpy()
                t['current_brier'].append(float(np.mean((predictions - w.bits) ** 2)))
            if 'state_read_logits' in out:
                chosen = out['state_read_logits'][j, :n, 0].argmax(-1).tolist()
                raw = selected[0][j, :, -22:].numpy()
                for entity, s in enumerate(chosen):
                    key = int(w.keys[entity])
                    latest = latest_public[i][key]
                    t['query_reads'] += 1
                    matching = s < 32 and (raw[s, 0] > 0 or raw[s, 2] > 0) and (raw[s, 4 + key] > 0.5)
                    if matching:
                        if float(raw[s, 21]) < latest - 1e-07:
                            t['stale_reads'] += 1
                        else:
                            t['latest_hits'] += 1
                    else:
                        t['wrong_reads'] += 1
            receipt = w.act(public[j][4][choices[j]])
            for e in receipt:
                if e[0] > 0 or e[2] > 0:
                    key = int(e[4:12].argmax())
                    latest_public[i][key] = max(latest_public[i].get(key, -1e9), float(e[21]))
            if not w.stopped:
                remaining.append(i)
                receipts.append(receipt)
        memory, occupied = state
        for t in range(max(map(len, receipts), default=0)):
            present = torch.tensor([t < len(r) for r in receipts])
            events = torch.from_numpy(np.stack([r[t] if t < len(r) else np.zeros(22, np.float32) for r in receipts]))
            update, trace = model.observe((memory[remaining], occupied[remaining]), events, present)
            memory = memory.index_copy(0, torch.tensor(remaining), update[0])
            occupied = occupied.index_copy(0, torch.tensor(remaining), update[1])
            writes[remaining] += trace['gate'].numpy().astype(int)
        state = (memory, occupied)
        active = remaining
    rows = []
    for i, w in enumerate(worlds):
        r = w.finish(int(writes[i]))
        initial = r.pop('initial_events')
        r.pop('keys')
        r.pop('parents')
        r['public_initial_sha256'] = hashlib.sha256(np.asarray(initial, np.float32).tobytes()).hexdigest()
        r['family'] = family
        r.update(telemetry[i])
        r['physical_repair'] = r['final_faults'] == 0
        r['economic_cost'] = r['cost'] - r['write_cost']
        r['unnecessary_repairs'] = sum((a['kind'] == 'repair' and (not a['before'][a['target']]) for a in r['actions']))
        r['healthy_nonstop'] = sum((a['kind'] != 'stop' and (not any(a['before'])) for a in r['actions']))
        rows.append(r)
    return {'seed': seed, 'entities': n, 'history': h, 'family': family, 'worlds': count, 'rows': rows, 'work': work, 'state_tensor_bytes': sum((v.numel() * v.element_size() for v in state)) // count}
