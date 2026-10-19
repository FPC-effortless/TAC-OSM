"""Training instrumentation uses only paid public receipts and final verifier.

Additional exact inspections are withheld from the student's shadow history.
They are separately charged acquisition instrumentation, NOT free policy input.
The simulator is isolated in PublicMaintenance; learners never receive it.
"""
from pathlib import Path
import json
import numpy as np
import torch
from data_belief_dev062 import BeliefWorld
from tac_osm.development.uncertain_world import COMMANDS


class PublicMaintenance:
    def __init__(self, seed, index, history, error=.1, inspection=.1):
        self._world = BeliefWorld(seed, index, history, error, inspection)

    def observations(self):
        return np.array(self._world.events, dtype=np.float32).copy()

    def keys(self):
        return self._world.identities.copy()

    def choices(self):
        return self._world.candidates()

    def fees(self):
        # Provisional final hidden-fault penalties are not public transition fees.
        return {k: float(self._world.costs[k]) for k in ('observations', 'inspections', 'repairs')}

    def execute(self, action):
        start = len(self._world.events)
        receipt = self._world.act(action)
        return list(receipt), np.array(self._world.events[start:], dtype=np.float32).copy()

    def verify(self):
        value = self._world.finish()
        return {k: value[k] for k in ('completion', 'physical_repair', 'faults', 'stopped', 'cost', 'costs')}


def graph_from_public(events, keys):
    graph = 0
    for event in events:
        if event[1] > .5:
            child = int(np.argmax(keys @ event[6:22]))
            parent = int(np.argmax(keys @ event[22:38]))
            if child == parent:
                raise ValueError('Invalid public relationship')
            graph = 1 if child == 1 else 2
    return graph


def measurements(env):
    before = sum(env.fees().values())
    labels, trace = [], []
    for entity in range(2):
        action = COMMANDS.index(('inspect', entity))
        receipt, events = env.execute(action)
        if len(receipt) != 1 or receipt[0] not in (0, 1):
            raise ValueError('Paid exact measurement contract violated')
        labels.append(int(receipt[0]))
        trace.append(dict(action=action, receipt=receipt, events=events.tolist()))
    return 2*labels[0]+labels[1], dict(actions=trace, fee=sum(env.fees().values())-before)


def corpus(seed, count, gate, horizon=10):
    rng = np.random.default_rng(seed)
    states, transitions, episodes = [], [], []
    index = 0
    while len(transitions) < count:
        history = (32, 128)[index % 2]
        error = (.1, .2)[(index//2) % 2]
        price = (.05, .1, .2)[index % 3]
        env = PublicMaintenance(seed, index, history, error, price)
        keys = env.keys(); shadow = list(env.observations())
        graph = graph_from_public(shadow, keys)
        episode = dict(seed=seed, index=index, history=history, error=error, inspection=price,
                       probes=[], main=[], state_ids=[])
        label, probe = measurements(env); episode['probes'].append(probe)

        def save_state(label, remaining):
            events = torch.tensor(np.array(shadow), dtype=torch.float32)
            with torch.no_grad():
                keep = gate(events[:, :6].to(next(gate.parameters()).dtype)).squeeze(-1) >= 0
            choices, permutation = env.choices()
            state_id = len(states)
            states.append(dict(events=events[keep][-64:], keys=torch.tensor(keys),
                candidates=torch.tensor(choices), prices=torch.tensor([error, price]),
                remaining=torch.tensor([remaining/horizon]), label=label*3+graph))
            episode['state_ids'].append(state_id)
            return state_id, permutation

        current, permutation = save_state(label, horizon)
        for depth in range(horizon):
            # Uniform factual exploration: never consult measurement labels.
            action = int(rng.integers(5))
            chosen = int(np.flatnonzero(permutation == action)[0])
            before = sum(env.fees().values())
            receipt, events = env.execute(action)
            fee = sum(env.fees().values())-before
            shadow.extend(events)
            done = action == COMMANDS.index(('stop', -1)) or depth == horizon-1
            episode['main'].append(dict(action=action, receipt=receipt, fee=fee, events=events.tolist()))
            if done:
                verified = env.verify(); episode['verified'] = verified
                future_label = 12+3*verified['faults']+graph
                physical = fee+verified['costs']['terminal']
                train_cost = physical+20*int(not verified['completion'])
                next_id = current  # Masked out of every Bellman continuation.
            else:
                label, probe = measurements(env); episode['probes'].append(probe)
                future_label = label*3+graph
                next_id, next_permutation = save_state(label, horizon-depth-1)
                physical = train_cost = fee
            transitions.append(dict(state=current, next=next_id, choice=chosen,
                future_label=future_label, done=done, physical_cost=physical, train_cost=train_cost))
            if done:
                break
            current, permutation = next_id, next_permutation
        episodes.append(episode); index += 1
    # Finish the last episode, even when its last transition crosses the request.
    length = max(1, max(len(s['events']) for s in states))
    data = {key: torch.stack([s[key] for s in states])
            for key in ('keys', 'candidates', 'prices', 'remaining')}
    data['events'] = torch.zeros(len(states), length, 46)
    data['present'] = torch.zeros(len(states), length, dtype=torch.bool)
    for i, state in enumerate(states):
        n = len(state['events']); data['events'][i, :n] = state['events']; data['present'][i, :n] = True
    data['label'] = torch.tensor([s['label'] for s in states])
    for key in transitions[0]:
        data[key] = torch.tensor([t[key] for t in transitions])
    coverage = dict(seed=seed, requested_transitions=count, transitions=len(transitions), states=len(states),
        episodes=len(episodes), label_probe_actions=sum(len(p['actions']) for e in episodes for p in e['probes']),
        label_probe_fees=sum(p['fee'] for e in episodes for p in e['probes']),
        main_actions=sum(len(e['main']) for e in episodes),
        main_fees=sum(a['fee'] for e in episodes for a in e['main']),
        terminal_costs=sum(e['verified']['costs']['terminal'] for e in episodes),
        initial_observation_fees=sum(e['history']*.01 for e in episodes),
        private_targets=False, source='paid exact inspection receipts and final public verifier')
    return data, episodes, coverage


def teacher_targets(episodes, state_count):
    """Explicit TRAINING-ONLY known-law PUBLIC control; never primary input."""
    result = torch.zeros(state_count, 12, dtype=torch.float64)
    for episode in episodes:
        world = BeliefWorld(episode['seed'], episode['index'], episode['history'],
                            episode['error'], episode['inspection'])
        belief = world.reference()
        graph = graph_from_public(world.events, world.identities)
        for i, state_id in enumerate(episode['state_ids']):
            if i:
                previous = episode['main'][i-1]
                belief.update(previous['action'], previous['receipt'])
            result[state_id].reshape(4, 3)[:, graph] = torch.tensor(belief.p)
    assert torch.allclose(result.sum(-1), torch.ones(state_count, dtype=torch.float64))
    return result


def state_batch(data, ids, dtype):
    result = {k: data[k][ids] for k in ('events', 'present', 'keys', 'candidates', 'prices', 'remaining', 'label')}
    length = max(1, int(result['present'].sum(-1).max()))
    result['events'] = result['events'][:, :length]
    result['present'] = result['present'][:, :length]
    return {k: v.to(dtype) if v.is_floating_point() else v for k, v in result.items()}
