#!/usr/bin/env python3
"""Registered DEV-029 balanced engineering pilot; validation only, no test path."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import torch
import torch.nn.functional as F
from tac_osm.development.optimization_program import ARMS, EventTablePredictor, domain
from tac_osm.development.memory_baselines import MemoryBaseline
from tac_osm.development.entity_aging import EntityAgingControl
from tac_osm.development.schema_maintenance import make_pair, ROLES, DEADLINES, execute, exact_public_targets
from tac_osm.development.delayed_maintenance import Action
from tac_osm.development.noisy_memory import parameter_hash
from run_learned_schema_dev import public_stream_tensors, public_queries, labels
from run_memory_controls_dev import physical, sensed


def build(arm, seed):
    if arm == 'event_table':
        return EventTablePredictor(seed)
    if arm == 'masked_local':
        return EntityAgingControl('masked_local', seed)
    return MemoryBaseline('tac_joint' if arm == 'native' else arm, seed)


def training_cases(step, smoke=False):
    return make_pair(311000000 + step, 32 if smoke else 512,
                     (32, 128, 512)[(step // 4) % 3], ROLES[step % 4],
                     DEADLINES[(step // 12) % 3], domain=domain('train'))


@torch.no_grad()
def validate(model, smoke):
    rows = []
    begin = time.monotonic()
    for d in ((32,) if smoke else (32, 128, 512)):
        for role in ROLES:
            for deadline in (('at',) if smoke else DEADLINES):
                for rep in range(1 if smoke else 2):
                    cases = make_pair(900000 + rep, 32 if smoke else 512, d, role,
                                      deadline, domain=domain('validation'))
                    inp = public_stream_tensors([c.observations for c in cases])
                    q = public_queries([c.tick for c in cases], [c.horizon for c in cases])
                    state = model.encode(*inp)
                    logits = model.decode(state, q)
                    probs = logits.softmax(-1)[..., 1].tolist()
                    selected, matched = sensed(model, state, q, cases, logits)
                    for b, c in enumerate(cases):
                        if exact_public_targets(c.observations, c.tick, c.horizon) != c.wait_targets:
                            raise AssertionError('public upper bound disagrees with evaluator')
                        none = execute(c.world, c.horizon, physical(probs[b]), 0)
                        rows.append(dict(fingerprint=c.fingerprint, due_delay=d, role=role,
                            deadline=deadline, repetition=rep, partner=b, target_entity=c.target_entity,
                            predictions=logits[b].argmax(-1).tolist(), logits=logits[b].tolist(),
                            labels=labels([c])[0].tolist(), none=none, selected=selected[b],
                            matched_random=matched[b], exact_rule_upper_bound=list(c.wait_targets)))
    primary = [int(r['predictions'][r['target_entity']][0] == r['labels'][r['target_entity']][0])
               for r in rows if r['deadline'] != 'before']
    ce = F.cross_entropy(torch.tensor([r['logits'] for r in rows]).reshape(-1, 2),
                         torch.tensor([r['labels'] for r in rows]).reshape(-1)).item()
    return dict(split='validation', primary_accuracy=sum(primary) / len(primary),
                cross_entropy=ce, rows=rows, seconds=time.monotonic()-begin,
                selected_minus_none_cost=sum(r['selected']['cost_per_entity']-r['none']['cost_per_entity'] for r in rows)/len(rows))


def run(arm, output, smoke):
    torch.set_num_threads(1)
    model = build(arm, 29311)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.002, weight_decay=.0001)
    initial = parameter_hash(model)
    fps = []
    steps = 2 if smoke else 216
    wall, cpu = time.monotonic(), time.process_time()
    for step in range(steps):
        cases = training_cases(step, smoke)
        fps.extend(c.fingerprint for c in cases)
        inp = public_stream_tensors([c.observations for c in cases])
        q = public_queries([c.tick for c in cases], [c.horizon for c in cases])
        target = labels(cases)
        receipts = [(c.world.act(Action('inspect', random.Random(f'dev029:{step}:{b}').randrange(8))),) for b, c in enumerate(cases)]
        obs = public_stream_tensors(receipts)
        optimizer.zero_grad(set_to_none=True)
        state = model.encode(*inp)
        pre = model.decode(state, q)
        post = model.decode(model.update(state, *(x[:, 0] for x in obs)), q)
        loss = (F.cross_entropy(pre.reshape(-1, 2), target.reshape(-1)) + F.cross_entropy(post.reshape(-1, 2), target.reshape(-1))) / 2
        if not torch.isfinite(loss):
            raise AssertionError('nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step % 36 == 0:
            print(json.dumps(dict(arm=arm, step=step, loss=float(loss.detach()))), flush=True)
    train_wall, train_cpu = time.monotonic()-wall, time.process_time()-cpu
    model.eval()
    validation = validate(model, smoke)
    if set(fps) & {r['fingerprint'] for r in validation['rows']}:
        raise AssertionError('train/validation overlap')
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / f'{arm}.pt'
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), updates=steps,
                    arm=arm, source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()), checkpoint)
    report = dict(experiment='DEV-029', arm=arm, seed=311, initialization_seed=29311,
        learning_rate=.002, curriculum='fixed512', updates=steps, smoke=smoke,
        protocol_sha256=hashlib.sha256((ROOT/'research/dev029/protocol.json').read_bytes()).hexdigest(),
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        initial_hash=initial, final_hash=parameter_hash(model),
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        state_scalars=model.state_scalars, state_bytes=model.state_scalars*4,
        observed_events=steps*(32 if smoke else 512)*2, training_receipts=steps*2,
        training_seconds=train_wall, training_cpu_seconds=train_cpu, final_loss=float(loss.detach()),
        training_fingerprints=fps, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        checkpoint_bytes=checkpoint.stat().st_size, validation=validation,
        scientific_selection_eligible=False, test_accessed=False)
    (out/f'{arm}.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(arm=arm, validation_accuracy=validation['primary_accuracy'], seconds=train_wall)), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--output', required=True); p.add_argument('--smoke', action='store_true')
    a=p.parse_args(); run(a.arm, a.output, a.smoke)
