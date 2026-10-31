"""Auditor software fixtures only: no original model prediction or optimizer step."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


checkpoints = load('langcontrol098_readonly_checkpoint_tests', 'audit_checkpoints.py')
outcomes = load('langcontrol098_readonly_outcome_tests', 'audit_outcomes.py')


def loss_receipt(objective='full'):
    rows = []
    for step in range(1, 5):
        rows.append(dict(step=step, loss=5. if objective == 'full' else 3.,
            future_loss=3., prefix_loss=2., prefix_lambda=1. if objective == 'full' else 0.,
            per_domain_losses={'lexical': {'future': 2., 'prefix': 1.},
                               'arc': {'future': 4., 'prefix': 3.}},
            gradient_norm_before_clip=.5 if step % 2 == 0 else 2.,
            gradient_clipped=step % 2 == 1))
    return dict(objective_control=checkpoints.expected_control(objective), history=rows,
                gradient_clipped_steps=2, gradient_clip_frequency=.5)


@pytest.mark.parametrize('objective', ['full', 'query'])
def test_registered_scalar_decomposition_and_clipping_receipts(objective):
    result = checkpoints.audit_loss_history(loss_receipt(objective), objective, steps=4)
    assert result['status'] == 'PASS_SAVED_SCALAR_COEFFICIENT_DECOMPOSITION'
    assert result['max_float32_component_sum_difference'] == 0.
    assert result['gradient_clipped_steps'] == 2


@pytest.mark.parametrize('mutation', ['future_coefficient', 'normalization', 'domain_weight',
    'future_sum', 'prefix_sum', 'total_loss', 'lambda', 'clip_flag', 'clip_count', 'history'])
def test_scalar_coefficient_or_update_history_mutation_fails(mutation):
    receipt = loss_receipt()
    row = receipt['history'][0]
    if mutation == 'future_coefficient':
        receipt['objective_control']['future_component_coefficient'] = 2.
    elif mutation == 'normalization':
        receipt['objective_control']['normalize_by_total_coefficient_mass'] = True
    elif mutation == 'domain_weight':
        receipt['objective_control']['domain_weights']['arc'] = .8
    elif mutation == 'future_sum':
        row['future_loss'] += .1
    elif mutation == 'prefix_sum':
        row['prefix_loss'] += .1
    elif mutation == 'total_loss':
        row['loss'] += .1
    elif mutation == 'lambda':
        row['prefix_lambda'] = 0.
    elif mutation == 'clip_flag':
        row['gradient_clipped'] = False
    elif mutation == 'clip_count':
        receipt['gradient_clipped_steps'] = 1
    else:
        receipt['history'].pop()
    with pytest.raises(checkpoints.AuditError):
        checkpoints.audit_loss_history(receipt, 'full', steps=4)


def ledger():
    entries = [dict(stage='optimizer', signature=checkpoints.canonical(
        ['aten::mul', [['torch.float32', [2]]], [['torch.float32', [2]]]]).decode(), calls=3)]
    for name in sorted(checkpoints.ZERO_WORK_PROFILER):
        entries.append(dict(stage='optimizer', signature=checkpoints.canonical([name, [], []]).decode(), calls=2))
    return entries


def test_serialization_markers_retain_balanced_counts_without_filtering_work():
    rows = ledger()
    sha, calls = checkpoints.check_ledger(rows, 'optimizer')
    assert sha == checkpoints.digest(rows) and calls == 7


@pytest.mark.parametrize('mutation', ['unknown', 'tensor', 'balance', 'count_type', 'duplicate', 'shape', 'onlymarkers'])
def test_independent_ledger_parser_fails_closed(mutation):
    rows = ledger()
    if mutation == 'unknown':
        rows[1]['signature'] = checkpoints.canonical(['custom::hidden_arithmetic', [], []]).decode()
    elif mutation == 'tensor':
        name = json.loads(rows[1]['signature'])[0]
        rows[1]['signature'] = checkpoints.canonical([name, [['torch.float32', [2]]], []]).decode()
    elif mutation == 'balance':
        rows[1]['calls'] += 1
    elif mutation == 'count_type':
        rows[1]['calls'] = True
    elif mutation == 'duplicate':
        rows.append(deepcopy(rows[1]))
    elif mutation == 'shape':
        rows[0]['signature'] = checkpoints.canonical(['aten::mul', [['torch.float32', [-1]]], []]).decode()
    else:
        rows = rows[1:]
    with pytest.raises(checkpoints.AuditError):
        checkpoints.check_ledger(rows, 'optimizer')


def optimizer_fixture():
    # A constructed topology and fabricated slots test the auditor, not fitting.
    model = torch.nn.Linear(2, 3)
    named = list(model.named_parameters())
    group = dict(params=list(range(len(named))), lr=.001, weight_decay=.01, eps=1e-8,
        foreach=False, fused=False, amsgrad=False, maximize=False, capturable=False,
        differentiable=False, betas=(.9, .999))
    state = {i: dict(step=torch.tensor(256.), exp_avg=torch.zeros_like(value),
                    exp_avg_sq=torch.ones_like(value)) for i, (_, value) in enumerate(named)}
    training = dict(steps=256, learning_rate=.001, weight_decay=.01, eps=1e-8, betas=[.9, .999])
    return model, dict(state=state, param_groups=[group]), training


def test_constructed_optimizer_slots_validate_without_executing_optimizer():
    model, slots, training = optimizer_fixture()
    report = checkpoints.check_optimizer(model, slots, training)
    assert report['all_slot_steps'] == [256] and report['slot_count'] == 2


@pytest.mark.parametrize('mutation', ['step', 'shape', 'missing', 'negative_square', 'optimizer_recipe'])
def test_optimizer_slots_cannot_hide_wrong_updates_or_missing_parameters(mutation):
    model, slots, training = optimizer_fixture()
    if mutation == 'step':
        slots['state'][0]['step'] -= 1
    elif mutation == 'shape':
        slots['state'][0]['exp_avg'] = torch.zeros(1)
    elif mutation == 'missing':
        del slots['state'][0]
    elif mutation == 'negative_square':
        slots['state'][0]['exp_avg_sq'].flatten()[0] = -1.
    else:
        slots['param_groups'][0]['lr'] = .01
    with pytest.raises(checkpoints.AuditError):
        checkpoints.check_optimizer(model, slots, training)


def test_rng_archive_requires_all_three_original_rng_states():
    current = np.random.get_state()
    rng = dict(python=random.getstate(), numpy=(current[0], current[1].tolist(), *current[2:]),
               torch=torch.get_rng_state())
    checkpoints.check_rng(rng)
    for field in rng:
        changed = deepcopy(rng)
        del changed[field]
        with pytest.raises(checkpoints.AuditError):
            checkpoints.check_rng(changed)


def prediction_fixture():
    actor, judge = [], {}
    for i in range(72):
        domain = 'lexical' if i < 64 else 'arc'
        ident = f'097-{i:064x}'
        group = ('positive' if i < 32 else 'negative' if i < 48 else 'uniform') if domain == 'lexical' else 'unknown_rule'
        actor.append(dict(id=ident, domain=domain, generation_limit=4 if domain == 'lexical' else 40))
        judge[ident] = dict(domain=domain, group=group, cluster_id=f'code-{i % 4}' if domain == 'lexical' else f'arc-{i}',
            donor_available=i < 67, binding_shuffle_available=domain == 'arc',
            target_bytes=[1, 2, 3, 4] if domain == 'lexical' else list(b'111;'))
    available = np.ones((5, 72), dtype=bool)
    available[4, :64] = False
    available[2, 67:] = False
    payload = dict(log_probs=np.full((5, 72, 256), -np.log(256.), dtype=np.float32),
                   generated_bytes=np.zeros((5, 72, 40), dtype=np.uint8), available=available,
                   lengths=np.array([4] * 64 + [40] * 8))
    metadata = dict(checkpoint=dict(family='tac', objective='query', seed=1701),
                    conditions=list(outcomes.CONDITIONS), case_ids=[r['id'] for r in actor])
    return payload, metadata, actor, judge


def test_independent_scoring_preserves_all_rows_and_unavailable_donors():
    payload, metadata, actor, judge = prediction_fixture()
    main, binding = outcomes.scored_rows(payload, metadata, actor, judge)
    assert len(main) == 288 and len(binding) == 8
    assert sum(row['status'] == 'INCONCLUSIVE' for row in main) == 5
    assert all(row['nll'] is None for row in main if row['status'] == 'INCONCLUSIVE')
    assert all(row['exact_match'] == 0 for row in main if row['status'] == 'MEASURED')
    assert all(row['format_valid'] == 0 for row in binding)


@pytest.mark.parametrize('mutation', ['donor_available', 'greedy_byte', 'length', 'nonfinite', 'case_order'])
def test_saved_prediction_boundary_mutations_fail(mutation):
    payload, metadata, actor, judge = prediction_fixture()
    if mutation == 'donor_available':
        payload['available'][2, 71] = True
    elif mutation == 'greedy_byte':
        payload['generated_bytes'][0, 0, 0] = 2
    elif mutation == 'length':
        payload['lengths'][0] = 40
    elif mutation == 'nonfinite':
        payload['log_probs'][0, 0, 0] = float('nan')
    else:
        metadata['case_ids'].reverse()
    with pytest.raises(outcomes.AuditError):
        outcomes.scored_rows(payload, metadata, actor, judge)


def test_bootstrap_clusters_do_not_treat_seed_duplicates_as_new_cases():
    matrix = np.tile([1., 1., -1., -1.], (5, 1))
    a = outcomes.interval(matrix, ['a', 'a', 'b', 'b'])
    b = outcomes.interval(np.repeat(matrix, 2, axis=0), ['a', 'a', 'b', 'b'])
    for field in ('mean', 'ci95_lower', 'ci95_upper', 'cases', 'clusters'):
        assert a[field] == b[field]
    matrix[:, 0] = np.nan
    result = outcomes.interval(matrix, ['a', 'a', 'b', 'b'])
    assert result['status'] == 'INCONCLUSIVE' and result['unavailable_cases'] == 1


def test_all_null_saved_predictions_fail_registered_learning_gates():
    payload, metadata, actor, judge = prediction_fixture()
    table, bindings = {}, {}
    for objective in outcomes.OBJECTIVES:
        for seed in outcomes.SEEDS:
            metadata['checkpoint'].update(objective=objective, seed=seed)
            main, alternate = outcomes.scored_rows(payload, metadata, actor, judge)
            table.update({outcomes.row_key(row): row for row in main})
            bindings.update({outcomes.row_key(row): row for row in alternate})
    raw = {case: dict(nll=0., exact_match=0) for case, row in judge.items() if row['domain'] == 'lexical'}
    arc = {case: dict(exact_match=0) for case, row in judge.items() if row['domain'] == 'arc'}
    report = outcomes.reconstruct_family('tac', table, bindings, judge, raw, arc)
    assert report['joint_learning_gate'] == report['practical_reference_quality_gate'] == 'FAIL'
    assert report['domains']['arc']['effects']['carry_minus_donor_nll']['status'] == 'INCONCLUSIVE'


def test_audit_source_is_pinned_and_rejects_another_fit_source():
    assert checkpoints.EXPECTED_SOURCE == outcomes.EXPECTED_SOURCE == '50bf7f2818f38c31f7c4d8280c1352c06f33bd4f'
    assert checkpoints.expected_control('full') == outcomes.expected_control('full')
    assert checkpoints.expected_control('query') == outcomes.expected_control('query')
    with pytest.raises(checkpoints.AuditError):
        checkpoints.frozen_file(Path('.').resolve(), 'missing.py', '0' * 40)
