"""Additive, unqualified synthetic diagnostic-fit admission. No model imports.

Approval/release anchors are supplied by the trusted execution controller, not
by the study manifest. These are consistency checks on that controller's
attestation, not cryptographic proof of user authorship. Old policy paths do not
call this module. A scope-only registration check never authorizes execution.
"""
import hashlib
import json
import math
import re

SCHEMA = 'tac-osm.synthetic-diagnostic-fit.v1'
FALSE_FLAGS = ('architecture_superiority', 'architecture_ranking_or_selection', 'real_world_capability', 'population_generalization',
               'deployment_readiness', 'safety_qualification', 'promotion', 'protected_TEST',
               'GPU', 'paid_compute', 'automatic_retry', 'adaptive_tuning', 'early_success_stop')
PIN_KEYS = ('source_manifest_sha256', 'code_manifest_sha256', 'wrapper_manifest_sha256',
            'protocol_sha256', 'policy_manifest_sha256', 'runtime_pin_sha256')

def demand(condition, message):
    if not condition:
        raise ValueError('synthetic diagnostic admission: ' + message)

def digest(value, name):
    demand(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None,
           name + ' must be SHA256')

def number(value, name, *, whole=False, zero=False):
    demand(isinstance(value, (int, float)) and not isinstance(value, bool)
           and math.isfinite(value) and (value >= 0 if zero else value > 0), name + ' invalid number')
    if whole:
        demand(isinstance(value, int), name + ' must be an integer')

def decoded(raw, name, anchor=None):
    demand(isinstance(raw, bytes), name + ' original bytes required')
    if anchor is not None:
        digest(anchor, name + ' external anchor')
        demand(hashlib.sha256(raw).hexdigest() == anchor, name + ' external anchor mismatch')
    value = json.loads(raw)
    demand(isinstance(value, dict), name + ' object required')
    return value

def same(a, b):
    return json.dumps(a, sort_keys=True, separators=(',', ':'), allow_nan=False) == json.dumps(b, sort_keys=True, separators=(',', ':'), allow_nan=False)

def validate_scope(m):
    """Validate registration only. Intentionally never returns fit permission."""
    demand(isinstance(m, dict) and m.get('schema_version') == SCHEMA, 'unsupported schema')
    demand(m.get('study_type') == 'synthetic_diagnostic_fit', 'explicit new category required')
    demand(isinstance(m.get('study_id'), str) and m['study_id'].startswith('SYNTHDIAG-'), 'new study ID required')
    demand(m.get('registered_before_first_fit') is True, 'preregistration required')
    for field in ('diagnostic_question', 'objective', 'auxiliary_loss_disclosure'):
        demand(isinstance(m.get(field), str) and len(m[field].strip()) >= 16, field + ' required')
    demand(isinstance(m.get('source_commit'), str) and re.fullmatch('[0-9a-f]{40}', m['source_commit']) is not None, 'exact model source commit required')
    demand(m.get('classification') == 'SYNTHETIC_SINGLE_SEED_EXPLORATORY_NONCOMPARATIVE_UNQUALIFIED', 'classification mismatch')
    demand(set(m.get('prohibited_claims_and_actions', {})) == set(FALSE_FLAGS), 'complete exact prohibition set required')
    for name in FALSE_FLAGS:
        demand(m.get('prohibited_claims_and_actions', {}).get(name) is False, name + ' must be false')
    source = m.get('source', {})
    demand(source.get('generated_synthetic_bytes') is True and source.get('observed_real_outcomes') is False,
           'truthful synthetic provenance required')
    for name in ('recipe_sha256', 'training_fixture_sha256', 'heldout_fixture_sha256', 'ordered_exposure_sha256'):
        digest(source.get(name), 'source.' + name)
    for name in PIN_KEYS:
        if name != 'protocol_sha256':  # Protocol contains this registration; grant binds its later hash.
            digest(m.get(name), name)
    number(m.get('seed'), 'seed', whole=True)
    demand(m.get('stage') in ('fixed', 'varying') and m.get('regime') in ('default', 'larger_epsilon'), 'declared stage/regime required')
    arms = m.get('arms')
    demand(isinstance(arms, list) and 1 <= len(arms) <= 3, 'one to three declared controls required')
    names = [a.get('family') for a in arms]
    demand(all(isinstance(x, str) and x for x in names) and len(set(names)) == len(names), 'unique families required')
    for arm in arms:
        for field in ('active_parameters', 'deployed_parameters'):
            number(arm.get(field), field, whole=True)
            demand(95000 <= arm[field] <= 105000, '95k to 105k actual parameter target required')
        demand(arm['active_parameters'] <= arm['deployed_parameters'], 'active exceeds deployed')
        for field in ('updates', 'input_tokens', 'training_examples', 'task_examples', 'first_targets'):
            number(arm.get(field), field, whole=True)
        number(arm.get('seed'), 'arm.seed', whole=True)
        demand(arm['seed'] == m['seed'], 'one common seed required')
        demand(arm.get('ordered_exposure_sha256') == source['ordered_exposure_sha256'], 'ordered exposure mismatch')
    for field in ('active_parameters', 'deployed_parameters'):
        values = [a[field] for a in arms]
        demand((max(values)-min(values))/max(values) <= .01, 'parameter match exceeds 1%')
    for field in ('updates', 'input_tokens', 'training_examples', 'task_examples', 'first_targets', 'ordered_exposure_sha256'):
        demand(all(a[field] == arms[0][field] for a in arms), 'unmatched ' + field)
    optimizer = m.get('optimizer', {})
    for field in ('learning_rate', 'epsilon', 'clip_norm'):
        number(optimizer.get(field), 'optimizer.' + field)
    number(optimizer.get('weight_decay'), 'optimizer.weight_decay', zero=True)
    demand(isinstance(optimizer.get('betas'), list) and len(optimizer['betas']) == 2, 'betas required')
    for beta in optimizer['betas']:
        number(beta, 'beta', zero=True)
        demand(beta < 1, 'beta must be below one')
    demand(optimizer.get('foreach') is False, 'frozen foreach setting required')
    demand(m.get('tuning_trials') == 0 and type(m.get('tuning_trials')) is int, 'no tuning trials')
    demand(m.get('selection') == 'fixed_checkpoints_no_selection', 'fixed checkpoint policy required')
    points = m.get('checkpoints')
    demand(isinstance(points, list) and points and all(type(n) is int and n >= 0 for n in points)
           and points == sorted(set(points)) and points[0] == 0 and points[-1] == arms[0]['updates'], 'checkpoint schedule invalid')
    demand(m.get('heldout_policy') == 'after_all_fits_sealed_no_tuning', 'heldout boundary required')
    demand(m.get('state_policy') == 'fresh_each_batch_query_forks_immutable_prefix', 'state ownership policy required')
    demand(m.get('optimizer_regime_control') == 'paired_initializers_exposures_only_declared_epsilon_differs', 'paired optimizer contract required')
    demand(m.get('unchanged_family_auxiliary_losses_disclosed') is True, 'auxiliary loss disclosure required')
    demand(m.get('compute_parity_claim') is False, 'this category cannot certify compute parity')
    counts = m.get('expected_counts', {})
    required_counts = {'model_constructors', 'optimizer_constructors', 'candidate_forward_calls', 'reference_forward_calls', 'candidate_backward_calls', 'reference_backward_calls', 'optimizer_updates', 'optimizer_attempts', 'future_poison_forward_calls'}
    demand(isinstance(counts, dict) and set(counts) == required_counts, 'complete exact call/update counter set required')
    for field, value in counts.items():
        number(value, 'count.' + field, whole=True, zero=True)
    demand(counts.get('optimizer_updates') == sum(a['updates'] for a in arms)
           and counts.get('optimizer_attempts') == counts['optimizer_updates'], 'aggregate updates mismatch')
    caps = m.get('caps', {})
    demand(isinstance(caps, dict), 'caps required')
    for field in ('total_CPU', 'total_wall', 'outer_CPU', 'parent_CPU', 'child_CPU', 'child_stop_CPU', 'parent_stop_CPU', 'monitor_wall', 'poll', 'inventory_interval', 'cleanup'):
        number(caps.get(field), 'cap.' + field)
    demand(caps['outer_CPU'] + caps['parent_CPU'] + caps['child_CPU'] <= caps['total_CPU'], 'CPU sub-budgets exceed aggregate')
    demand(caps['child_stop_CPU'] < caps['child_CPU'] and caps['parent_stop_CPU'] < caps['parent_CPU']
           and caps['monitor_wall'] < caps['total_wall'], 'early stop headroom required')
    memory = m.get('memory_limits', {})
    for field in ('address_space_bytes', 'per_file_bytes', 'child_output_bytes', 'total_output_bytes', 'child_file_count', 'total_file_count'):
        number(memory.get(field), 'memory.' + field, whole=True)
    for field in ('runner_path', 'admission_path'):
        path = m.get(field)
        demand(isinstance(path, str) and path and not path.startswith('/') and '..' not in path.split('/'), field + ' must be repository-relative')
    return {'status': 'PASS_SYNTHETIC_REGISTRATION_ONLY_NOT_AUTHORIZED', 'study_id': m['study_id'],
            'eligible_for_architecture_superiority': False, 'qualified': False, 'promotion_allowed': False}

def validate_synthetic_diagnostic(m, phase='preflight', *, approval_bytes=None, grant_bytes=None,
                                  review_bytes=None, trusted_approval_sha256=None,
                                  trusted_grant_sha256=None, actual=None):
    validate_scope(m)
    demand(phase in ('preflight', 'final'), 'phase invalid')
    digest(trusted_approval_sha256, 'external policy approval anchor')
    digest(trusted_grant_sha256, 'external one-shot release anchor')
    approval = decoded(approval_bytes, 'scope approval', trusted_approval_sha256)
    grant = decoded(grant_bytes, 'one-shot grant', trusted_grant_sha256)
    demand(approval.get('schema') == 'tac-osm.synthetic-scope-approval.v1'
           and approval.get('repository') == 'FPC-effortless/TAC-OSM'
           and approval.get('approved_scope') == 'bounded_synthetic_diagnostic_fits'
           and approval.get('answer') == 'Yes', 'approval scope mismatch')
    demand(approval.get('verification') == 'trusted_root_conversation_attestation_not_cryptographic_authorship', 'approval provenance required')
    for field in ('question_message_id', 'answer_message_id', 'question'):
        demand(isinstance(approval.get(field), str) and len(approval[field]) >= 8, 'approval ' + field + ' missing')
    demand(approval.get('retains_real_data_and_comparison_requirements') is True
           and approval.get('permits_superiority_or_readiness') is False, 'approval scope not narrow')
    demand(grant.get('authorized') is True and grant.get('retry_authorized') is False, 'one-shot release required')
    demand(grant.get('approval_sha256') == trusted_approval_sha256, 'release approval mismatch')
    demand(grant.get('study_id') == m['study_id'], 'release study mismatch')
    registration_fingerprint = hashlib.sha256(json.dumps(m, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    demand(grant.get('registration_fingerprint') == registration_fingerprint, 'release registration fingerprint mismatch')
    for field in ('stage', 'regime', 'seed', 'caps', 'expected_counts', 'memory_limits'):
        demand(same(grant.get(field), m[field]), 'release ' + field + ' mismatch')
    demand(grant.get('families') == [a['family'] for a in m['arms']], 'release family order mismatch')
    demand(same(grant.get('updates_per_arm'), {a['family']: a['updates'] for a in m['arms']}), 'release update scope mismatch')
    for name in PIN_KEYS:
        digest(grant.get(name), 'release.' + name)
        if name != 'protocol_sha256':
            demand(grant[name] == m[name], 'release ' + name + ' mismatch')
    digest(grant.get('review_sha256'), 'review pin')
    demand(grant.get('bounded_launcher_review_sha256') == grant['review_sha256'], 'launcher review required')
    review = decoded(review_bytes, 'independent review', grant['review_sha256'])
    demand(review.get('status') == 'PASS_REVIEWED_SYNTHETIC_DIAGNOSTIC_SOURCE'
           and review.get('execution_release_signoff') is True, 'independent source signoff missing')
    for name in PIN_KEYS:
        demand(review.get('pins', {}).get(name) == grant[name], 'review ' + name + ' mismatch')
    demand(review.get('registration_fingerprint') == registration_fingerprint, 'review registration mismatch')
    if phase == 'final':
        validate_final(m, actual)
    return {'status': 'PASS_SYNTHETIC_DIAGNOSTIC_PREFLIGHT_UNQUALIFIED' if phase == 'preflight' else 'PASS_SYNTHETIC_FINAL_INTEGRITY_RESOURCE_PENDING',
            'phase': phase, 'study_id': m['study_id'], 'eligible_for_architecture_superiority': False,
            'qualified': False, 'promotion_allowed': False, 'real_world_evidence': False,
            'approval_sha256': trusted_approval_sha256, 'grant_sha256': trusted_grant_sha256,
            'review_sha256': grant['review_sha256'], 'trust_model': 'externally anchored trusted-root attestation, not cryptographic user-authorship proof'}

def validate_final(m, actual):
    demand(isinstance(actual, dict), 'actual final receipt required')
    demand(same(actual.get('counts'), m['expected_counts']), 'actual call/update counts mismatch')
    expected = {a['family']: a['updates'] for a in m['arms']}
    demand(same(actual.get('updates'), expected) and same(actual.get('attempted_updates'), expected),
           'incomplete, extra or uncertain optimizer updates')
    demand(actual.get('implementation_gate_pass') is True, 'implementation gate failure retained')
    demand(actual.get('source_manifest_unchanged') is True and actual.get('heldout_weights_unchanged') is True, 'source/weight integrity missing')
    for arm in m['arms']:
        family = arm['family']; initial = actual.get('initial_parameter_counts', {}).get(family, {})
        final = actual.get('artifact_verification', {}).get(family, {})
        for field in ('active_parameters', 'deployed_parameters'):
            demand(type(initial.get(field)) is int and initial[field] == arm[field], 'actual initial ' + field + ' mismatch')
            demand(type(final.get(field)) is int and final[field] == arm[field], 'actual final ' + field + ' mismatch')
        digest(final.get('checkpoint_sha256'), 'actual checkpoint')
        demand(type(final.get('every_AdamW_step')) is int and final['every_AdamW_step'] == arm['updates'] and type(final.get('restored_optimizer_updates_executed')) is int and final['restored_optimizer_updates_executed'] == 0, 'actual moment step mismatch')
        for field in ('finite_moments', 'model_optimizer_rng_payload_exact', 'restored_optimizer_state_exact', 'serialization_logits_state_bitwise_exact'):
            demand(final.get(field) is True, 'final ' + field + ' missing')
        exposure = actual.get('exposures', {}).get(family, {})
        for field in ('updates', 'input_tokens', 'training_examples', 'task_examples', 'first_targets', 'ordered_exposure_sha256'):
            demand(same(exposure.get(field), arm[field]), 'actual exposure mismatch: ' + field)
    number(actual.get('child_CPU_at_final_gate'), 'actual child CPU', zero=True)
    demand(actual['child_CPU_at_final_gate'] <= m['caps']['child_CPU'], 'actual child CPU cap exceeded')
    return True

def validate_resource_receipt(m, receipt, *, trusted_grant_sha256=None):
    validate_scope(m)
    digest(trusted_grant_sha256, 'external current resource grant anchor')
    demand(isinstance(receipt, dict) and receipt.get('status') == 'COMPLETE_WITH_CAPTURED_RESOURCE_ACCEPTANCE', 'complete resource receipt required')
    demand(receipt.get('grant_sha256') == trusted_grant_sha256 and receipt.get('wrapper_manifest_sha256') == m['wrapper_manifest_sha256'], 'stale or mismatched resource run')
    for key in ('study_id', 'stage', 'regime'):
        demand(receipt.get(key) == m[key], 'resource ' + key + ' mismatch')
    demand(receipt.get('semantic_status') == 'COMPLETE_FIXED_BUDGET_MATRIX' and receipt.get('child_final_admission_status') == 'PASS_SYNTHETIC_FINAL_INTEGRITY_RESOURCE_PENDING', 'final child integrity incomplete')
    demand(receipt.get('owned_group_children_reaped') is True and receipt.get('adopted_children') == [], 'owned process cleanup incomplete')
    fields = ('full_wait4_parent_and_reaped_descendants_CPU', 'child_complete_wait4_CPU', 'parent_complete_CPU', 'outer_CPU_before_final_receipt', 'adopted_complete_CPU', 'captured_total_CPU', 'wall_before_final_receipt')
    for name in fields:
        number(receipt.get(name), name, zero=True)
    full = receipt['full_wait4_parent_and_reaped_descendants_CPU']
    child = receipt['child_complete_wait4_CPU']; parent = receipt['parent_complete_CPU']
    outer = receipt['outer_CPU_before_final_receipt']; adopted = receipt['adopted_complete_CPU']
    demand(adopted == 0 and full >= child, 'wait4 component inconsistency')
    demand(math.isclose(full, child + parent, rel_tol=0, abs_tol=1e-9)
           and math.isclose(receipt['captured_total_CPU'], full + outer + adopted, rel_tol=0, abs_tol=1e-9), 'wait4 component sum mismatch')
    for value, cap in ((child, 'child_CPU'), (parent, 'parent_CPU'), (outer, 'outer_CPU'),
                       (receipt['captured_total_CPU'], 'total_CPU'), (receipt['wall_before_final_receipt'], 'total_wall')):
        demand(value <= m['caps'][cap], 'resource cap exceeded: ' + cap)
    demand(receipt.get('retry_authorized') is False and receipt.get('outer_final_write_exit_tail_included') is False, 'resource scope disclosure missing')
    return {'status': 'PASS_SYNTHETIC_RESOURCE_ACCEPTANCE_UNQUALIFIED', 'qualified': False,
            'promotion_allowed': False, 'eligible_for_architecture_superiority': False,
            'grant_sha256': trusted_grant_sha256,
            'captured_tail_caveat': 'Outer final receipt/exit tail is outside captured CPU/wall measurement.'}
