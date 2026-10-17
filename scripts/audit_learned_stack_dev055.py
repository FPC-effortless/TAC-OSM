"""Separate-process restored inference, physical replay and optional native replay.

Native replay checks Veritas execution/accounting against newly authored public
synthetic trajectories. It shares the physics generator and does not constitute
an independent physical model or real-world outcome verification.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from run_learned_stack_dev055 import ROOT, PROTOCOL, digest, write, evaluate
from tac_osm.development.learned_stack import LearnedStack, PhysicalWorld, COSTS

PIN = 'b21d829b62d3cab2811e746267ca305e110ed004'


def native_replay(row):
    from tac_osm.development.veritas_public_challenge import make_episode
    from investigation_world.operational.models import (
        ActionKind, HiddenActionEffect, PublicActionSpec, StateAssertion, EpisodeSubmission)
    from investigation_world.operational.runtime import OperationalRuntime
    ep = make_episode(row['seed'],row['index'])
    world = PhysicalWorld(row['seed'],row['index'],row['family'])
    initial = {'e0.fault':int(world.bits[0]),'e1.fault':int(world.bits[1]),'world.tick':0}
    effects = []
    faults = 0
    for tick, recorded in enumerate(row['actions']):
        before = world.bits.copy()
        event,cost,label = world.step(recorded['action'])
        if cost != recorded['cost'] or label['after'] != recorded['after'] or event.tensor() != recorded['public']:
            raise AssertionError('physical replay mismatch')
        faults += sum(label['after'])
        effects.append(HiddenActionEffect(
            action_name=f'action{recorded["action"]}',required_parameters={'tick':tick},
            required_state=[StateAssertion(object_id='world',field_name='tick',expected_value=tick),
                            *[StateAssertion(object_id=f'e{e}',field_name='fault',expected_value=int(before[e])) for e in range(2)]],
            set_state={'e0.fault':int(world.bits[0]),'e1.fault':int(world.bits[1]),'world.tick':tick+1},
            observable_result={'sensors':list(event.sensors),'exact':list(event.exact),'tick':event.tick}))
    ep.task.available_actions = [PublicActionSpec(name=f'action{a}',kind=ActionKind.EXECUTE,
        system='plant',description=f'DEV-055 action {a}',parameter_names=['tick'],cost=int(round(COSTS[a]*20))) for a in range(5)]
    ep.task.constraints = ['Exactly twenty public actions; each receipt advances the physical clock.']
    ep.oracle.initial_state = initial
    ep.oracle.action_effects = effects
    ep.oracle.target_state = [StateAssertion(object_id=f'e{e}',field_name='fault',expected_value=0) for e in range(2)]
    ep.oracle.invariants = []
    ep.oracle.forbidden_actions = []
    ep.oracle.required_action_counts = {}
    ep.oracle.max_cost = 1000; ep.oracle.max_tool_calls = 20
    runtime = OperationalRuntime(ep)
    for tick, recorded in enumerate(row['actions']):
        receipt = runtime.act(f'action{recorded["action"]}',tick=tick)
        if any(receipt[k] != effects[tick].observable_result[k] for k in ('sensors','exact','tick')):
            raise AssertionError('native public receipt mismatch')
    native_total = runtime.budget.spent/20+4*faults
    if abs(native_total-row['cost']) > 1e-8 or faults != row['fault_exposure']:
        raise AssertionError('native cost/fault accounting mismatch')
    if any(event.blocked for event in runtime.events): raise AssertionError('native blocked action')
    verified = runtime.submit(EpisodeSubmission(conclusion='DEV-055 synthetic replay'))
    if bool(verified.state == 1) != bool(sum(world.bits) == 0): raise AssertionError('native final state mismatch')
    return {'actions':len(row['actions']),'residual_faults':int(sum(world.bits)),'state_verified':bool(verified.state == 1)}


def run(directory, output, veritas_root=None):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    out = Path(directory)
    summary=json.loads((out/'summary.json').read_text())
    p=json.loads(PROTOCOL.read_text())
    if summary['smoke']: raise ValueError('smoke is not registered research evidence')
    if digest(PROTOCOL) != summary['protocol_sha256']: raise AssertionError('protocol changed')
    for name, value in json.loads((out/'manifest.json').read_text()).items():
        if digest(out/name) != value: raise AssertionError(f'hash mismatch: {name}')
    reports=json.loads((out/'reports.json').read_text())
    controls=json.loads((out/'controls.json').read_text())
    tables=np.load(out/'filters.npz')
    native = veritas_root is not None
    if native:
        vr=Path(veritas_root).resolve()
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=vr,text=True).strip()!=PIN: raise AssertionError('wrong Veritas commit')
        sys.path.insert(0,str(vr/'src'))
        import investigation_world
        if not Path(investigation_world.__file__).resolve().is_relative_to(vr): raise AssertionError('wrong Veritas import')
    result={'status':'RUNNING','restored_checkpoints':0,'physical_worlds':0,'physical_actions':0,
            'native_worlds':0,'native_actions':0,'native_unrepaired_worlds':0,'checkpoint_forecast_lesions':{},
            'native_veritas_commit':PIN if native else None,
            'limitation':'Same author and shared physics/model code; development replay, not independent replication. Native effect tables encode recorded synthetic trajectories.'}
    def check_rows(rows):
        for row in rows:
            w=PhysicalWorld(row['seed'],row['index'],row['family'])
            cost,faults=0.,0
            for action in row['actions']:
                event,c,label=w.step(action['action'])
                assert event.tensor()==action['public'] and label['after']==action['after'] and c==action['cost']
                cost+=c; faults+=sum(label['after'])
            assert cost==row['cost'] and faults==row['fault_exposure']
            result['physical_worlds']+=1; result['physical_actions']+=len(row['actions'])
            if native:
                replay=native_replay(row)
                result['native_worlds']+=1; result['native_actions']+=replay['actions']
                result['native_unrepaired_worlds']+=int(not replay['state_verified'])
    for arm, evaluations in controls.items():
        for family, evaluation in evaluations.items(): check_rows(evaluation['rows'])
    for seed in p['model_seeds']:
        for arm in p['arms']:
            report=reports[str(seed)][arm]
            if report['status']!='COMPLETE': raise AssertionError(f'incomplete {seed}/{arm}')
            checkpoint=out/f'{seed}-{arm}.pt'
            if digest(checkpoint)!=report['checkpoint_sha256']: raise AssertionError('checkpoint hash')
            model=LearnedStack(seed,arm).eval()
            model.load_state_dict(torch.load(checkpoint,weights_only=True))
            for family,evaluation in report['evaluation'].items():
                replay=evaluate(model,p['splits'][family],family,tables['transitions'],tables['independent'])
                if replay['rows'] != evaluation['rows']: raise AssertionError(f'restored inference mismatch {seed}/{arm}/{family}')
                check_rows(evaluation['rows'])
            # Operational lesion on fresh disjoint worlds, not a new test endpoint.
            if arm=='mtsk':
                spec=[8551900+seed,16]
                carried=evaluate(model,spec,'coupling',tables['transitions'],tables['independent'])
                neutral=evaluate(model,spec,'coupling',tables['transitions'],tables['independent'],neutral_memory=True)
                result['checkpoint_forecast_lesions'][str(seed)]={'carried_cost':carried['cost'],'zero_memory_cost':neutral['cost'],
                    'changed_actions':sum(a['action']!=b['action'] for x,y in zip(carried['rows'],neutral['rows']) for a,b in zip(x['actions'],y['actions']))}
            result['restored_checkpoints']+=1
            print('audited',seed,arm,flush=True)
    result.update(status='PASS',seconds=time.monotonic()-started,source=summary['source'],
                  audit_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write(output,result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--veritas-root')
    args=parser.parse_args(); run(args.directory,args.output,args.veritas_root)
