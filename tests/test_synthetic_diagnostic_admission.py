"""Constructed metadata falsifiers only: no real approval, model or optimizer."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
policy=load('synthetic_policy_under_test',ROOT/'scripts/synthetic_diagnostic_policy.py')
canonical=load('canonical_gate_under_test',ROOT/'scripts/validate_research_parity.py')
no_torch_probe=load('synthetic_no_torch_probe',Path(__file__).with_name('synthetic_no_torch_probe.py'))
def raw(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(value):return hashlib.sha256(raw(value)).hexdigest()

def fixture():
    m={'schema_version':policy.SCHEMA,'study_type':'synthetic_diagnostic_fit','study_id':'SYNTHDIAG-CONSTRUCTED-TEST-ONLY','registered_before_first_fit':True,
       'classification':'SYNTHETIC_SINGLE_SEED_EXPLORATORY_NONCOMPARATIVE_UNQUALIFIED',
       'diagnostic_question':'Constructed metadata test, not a fitted diagnostic.', 'objective':'Fixed synthetic objective metadata for software tests.',
       'auxiliary_loss_disclosure':'Family-specific auxiliary terms remain disclosed.', 'source_commit':'a'*40,
       'prohibited_claims_and_actions':{k:False for k in policy.FALSE_FLAGS},
       'source':{'generated_synthetic_bytes':True,'observed_real_outcomes':False,**{k:'a'*64 for k in ('recipe_sha256','training_fixture_sha256','heldout_fixture_sha256','ordered_exposure_sha256')}},
       **{k:'b'*64 for k in policy.PIN_KEYS if k!='protocol_sha256'},'seed':18001,'stage':'fixed','regime':'default',
       'arms':[{'family':f,'seed':18001,'active_parameters':100000,'deployed_parameters':100000,'updates':128,'input_tokens':65536,'training_examples':512,'task_examples':384,'first_targets':384,'ordered_exposure_sha256':'a'*64} for f in ('gru','phase','context')],
       'optimizer':{'learning_rate':.003,'epsilon':1e-8,'clip_norm':1.,'weight_decay':.1,'betas':[.9,.95],'foreach':False},
       'tuning_trials':0,'selection':'fixed_checkpoints_no_selection','checkpoints':[0,32,64,112,128],
       'heldout_policy':'after_all_fits_sealed_no_tuning','state_policy':'fresh_each_batch_query_forks_immutable_prefix',
       'optimizer_regime_control':'paired_initializers_exposures_only_declared_epsilon_differs','unchanged_family_auxiliary_losses_disclosed':True,'compute_parity_claim':False,
       'expected_counts':dict(model_constructors=6,optimizer_constructors=6,candidate_forward_calls=795,reference_forward_calls=6,candidate_backward_calls=402,reference_backward_calls=18,optimizer_updates=384,optimizer_attempts=384,future_poison_forward_calls=3),
       'caps':dict(total_CPU=360,total_wall=600,outer_CPU=5,parent_CPU=10,child_CPU=345,child_stop_CPU=340,parent_stop_CPU=8,monitor_wall=590,poll=1,inventory_interval=10,cleanup=5),
       'memory_limits':dict(address_space_bytes=8*1024**3,per_file_bytes=8*1024**2,child_output_bytes=32*1024**2,total_output_bytes=36*1024**2,child_file_count=64,total_file_count=80),
       'runner_path':'research/synthetic_diagnostics/constructed/code/runner.py','admission_path':'research/synthetic_diagnostics/constructed/code/admission.py'}
    approval={'schema':'tac-osm.synthetic-scope-approval.v1','repository':'FPC-effortless/TAC-OSM','approved_scope':'bounded_synthetic_diagnostic_fits','answer':'Yes','verification':'trusted_root_conversation_attestation_not_cryptographic_authorship','question_message_id':'CONSTRUCTED-QUESTION-NOT-REAL','answer_message_id':'CONSTRUCTED-ANSWER-NOT-REAL','question':'Constructed software-test permission, not an actual user message.','retains_real_data_and_comparison_requirements':True,'permits_superiority_or_readiness':False}
    pins={k:m.get(k,'c'*64) for k in policy.PIN_KEYS}
    review={'status':'PASS_REVIEWED_SYNTHETIC_DIAGNOSTIC_SOURCE','execution_release_signoff':True,'pins':pins,'registration_fingerprint':sha(m),'constructed_test_only':True}
    grant={'authorized':True,'retry_authorized':False,'approval_sha256':sha(approval),'study_id':m['study_id'],'registration_fingerprint':sha(m),**{k:m[k] for k in ('stage','regime','seed','caps','expected_counts','memory_limits')},'families':[a['family'] for a in m['arms']],'updates_per_arm':{a['family']:a['updates'] for a in m['arms']},**pins,'review_sha256':sha(review),'bounded_launcher_review_sha256':sha(review),'constructed_test_only':True}
    evidence=dict(approval_bytes=raw(approval),grant_bytes=raw(grant),review_bytes=raw(review),trusted_approval_sha256=sha(approval),trusted_grant_sha256=sha(grant))
    return m,approval,review,grant,evidence

def actual(m):
    counts={a['family']:a['updates'] for a in m['arms']}
    return {'counts':copy.deepcopy(m['expected_counts']),'updates':counts,'attempted_updates':dict(counts),'implementation_gate_pass':True,'source_manifest_unchanged':True,'heldout_weights_unchanged':True,
       'initial_parameter_counts':{a['family']:{k:a[k] for k in ('active_parameters','deployed_parameters')} for a in m['arms']},
       'artifact_verification':{a['family']:{**{k:a[k] for k in ('active_parameters','deployed_parameters')},'checkpoint_sha256':'d'*64,'every_AdamW_step':a['updates'],'restored_optimizer_updates_executed':0,'finite_moments':True,'model_optimizer_rng_payload_exact':True,'restored_optimizer_state_exact':True,'serialization_logits_state_bitwise_exact':True} for a in m['arms']},
       'exposures':{a['family']:{k:a[k] for k in ('updates','input_tokens','training_examples','task_examples','first_targets','ordered_exposure_sha256')} for a in m['arms']},'child_CPU_at_final_gate':40.}

def resource(m, grant_sha):
    return {'status':'COMPLETE_WITH_CAPTURED_RESOURCE_ACCEPTANCE','grant_sha256':grant_sha,'wrapper_manifest_sha256':m['wrapper_manifest_sha256'],**{k:m[k] for k in ('study_id','stage','regime')},'semantic_status':'COMPLETE_FIXED_BUDGET_MATRIX','child_final_admission_status':'PASS_SYNTHETIC_FINAL_INTEGRITY_RESOURCE_PENDING','owned_group_children_reaped':True,'adopted_children':[],'full_wait4_parent_and_reaped_descendants_CPU':41.,'child_complete_wait4_CPU':40.,'parent_complete_CPU':1.,'outer_CPU_before_final_receipt':.1,'adopted_complete_CPU':0.,'captured_total_CPU':41.1,'wall_before_final_receipt':45.,'retry_authorized':False,'outer_final_write_exit_tail_included':False}

class AdmissionTests(unittest.TestCase):
    def test_valid_metadata_stays_unqualified(self):
        m,_,_,_,e=fixture();out=canonical.check_manifest(m,**e)
        self.assertEqual(out['status'],'PASS_SYNTHETIC_DIAGNOSTIC_PREFLIGHT_UNQUALIFIED')
        self.assertFalse(out['qualified']);self.assertFalse(out['promotion_allowed']);self.assertFalse(out['eligible_for_architecture_superiority'])
        self.assertIn('NOT_AUTHORIZED',policy.validate_scope(m)['status'])
        no_torch_probe.assert_isolated_no_torch(self, __file__)
    def test_external_anchors_cannot_come_from_manifest(self):
        m,_,_,_,e=fixture();m['approved']=True;m['trusted_approval_sha256']=e['trusted_approval_sha256']
        for field in ('trusted_approval_sha256','trusted_grant_sha256','approval_bytes','grant_bytes','review_bytes'):
            with self.subTest(field=field),self.assertRaises(ValueError):policy.validate_synthetic_diagnostic(m,**{**e,field:None})
    def test_receipt_substitution_and_missing_review(self):
        m,a,r,g,e=fixture()
        for field,replacement in [('approval_bytes',raw({**a,'answer':'No'})),('grant_bytes',raw({**g,'seed':999})),('review_bytes',raw({**r,'execution_release_signoff':False}))]:
            with self.subTest(field=field),self.assertRaises(ValueError):policy.validate_synthetic_diagnostic(m,**{**e,field:replacement})
    def test_reanchored_wrong_scope_or_release_still_fails(self):
        m,a,r,g,e=fixture()
        a['approved_scope']='unbounded synthetic training'
        ee={**e,'approval_bytes':raw(a),'trusted_approval_sha256':sha(a)}
        with self.assertRaises(ValueError):policy.validate_synthetic_diagnostic(m,**ee)
        for field,bad in [('stage','varying'),('regime','larger_epsilon'),('families',['context']),('seed',42),('registration_fingerprint','f'*64),('policy_manifest_sha256','f'*64),('caps',{**m['caps'],'poll':True})]:
            gg={**g,field:bad}
            with self.subTest(field=field),self.assertRaises(ValueError):policy.validate_synthetic_diagnostic(m,**{**e,'grant_bytes':raw(gg),'trusted_grant_sha256':sha(gg)})
    def test_invalid_numeric_budgets(self):
        for bad in (True,False,0,-1,float('nan'),float('inf'),1.5):
            m,*_=fixture();m['arms'][0]['updates']=bad
            with self.subTest(bad=bad),self.assertRaises(ValueError):policy.validate_scope(m)
        for bad in (True,False,0,-1,float('nan'),float('inf')):
            m,*_=fixture();m['caps']['total_CPU']=bad
            with self.subTest(cap=bad),self.assertRaises(ValueError):policy.validate_scope(m)
    def test_forbidden_claims_and_actions(self):
        for field in policy.FALSE_FLAGS:
            m,*_=fixture();m['prohibited_claims_and_actions'][field]=True
            with self.subTest(field=field),self.assertRaises(ValueError):policy.validate_scope(m)
    def test_missing_question_counter_runtime_or_memory(self):
        for field in ('diagnostic_question','runtime_pin_sha256','memory_limits'):
            m,*_=fixture();del m[field]
            with self.subTest(field=field),self.assertRaises(ValueError):policy.validate_scope(m)
        m,*_=fixture();m['expected_counts']={'optimizer_updates':384,'optimizer_attempts':384}
        with self.assertRaises(ValueError):policy.validate_scope(m)
    def test_parameter_band_relative_match_and_no_legacy_escape(self):
        for study,bad in [('SYNTHDIAG-SOFTWARE',94999),('SYNTHDIAG-SOFTWARE',105001),('ME001_REUSED',100000)]:
            m,*_=fixture();m['study_id']=study;m['arms'][0]['active_parameters']=bad
            with self.subTest(study=study,bad=bad),self.assertRaises(ValueError):policy.validate_scope(m)
        m,*_=fixture();m['arms'][0]['active_parameters']=98000
        with self.assertRaises(ValueError):policy.validate_scope(m)
    def test_boolean_arm_seed_does_not_equal_integer_one(self):
        m,*_=fixture();m['seed']=1
        for a in m['arms']:a['seed']=1
        m['arms'][0]['seed']=True
        with self.assertRaises(ValueError):policy.validate_scope(m)
    def test_mismatched_exposures_or_checkpoint_selection(self):
        for field,bad in [('input_tokens',65535),('ordered_exposure_sha256','b'*64)]:
            m,*_=fixture();m['arms'][0][field]=bad
            with self.subTest(field=field),self.assertRaises(ValueError):policy.validate_scope(m)
        m,*_=fixture();m['selection']='best heldout checkpoint'
        with self.assertRaises(ValueError):policy.validate_scope(m)
    def test_final_integrity_is_not_resource_completion(self):
        m,_,_,_,e=fixture();out=policy.validate_synthetic_diagnostic(m,'final',actual=actual(m),**e)
        self.assertEqual(out['status'],'PASS_SYNTHETIC_FINAL_INTEGRITY_RESOURCE_PENDING')
        self.assertFalse(out['qualified'])
    def test_partial_extra_or_uncertain_updates_fail_final(self):
        m,_,_,_,e=fixture()
        for key,n in [('updates',127),('updates',129),('attempted_updates',129)]:
            a=actual(m);a[key]['gru']=n
            with self.subTest(key=key,n=n),self.assertRaises(ValueError):policy.validate_synthetic_diagnostic(m,'final',actual=a,**e)
    def test_actual_model_moments_checkpoint_and_exposures_required(self):
        m,_,_,_,e=fixture()
        for key,bad in [('active_parameters',99999),('every_AdamW_step',127),('finite_moments',False),('checkpoint_sha256',None),('serialization_logits_state_bitwise_exact',False),('restored_optimizer_updates_executed',False)]:
            a=actual(m);a['artifact_verification']['gru'][key]=bad
            with self.subTest(key=key),self.assertRaises(ValueError):policy.validate_synthetic_diagnostic(m,'final',actual=a,**e)
        a=actual(m);a['exposures']['gru']['first_targets']=383
        with self.assertRaises(ValueError):policy.validate_synthetic_diagnostic(m,'final',actual=a,**e)
        a=actual(m);a['initial_parameter_counts']={}
        with self.assertRaises(ValueError):policy.validate_synthetic_diagnostic(m,'final',actual=a,**e)
    def test_final_failures_nonfinite_or_overcap_are_not_pass(self):
        m,_,_,_,e=fixture()
        for key,bad in [('implementation_gate_pass',False),('source_manifest_unchanged',False),('child_CPU_at_final_gate',float('inf')),('child_CPU_at_final_gate',346.)]:
            a=actual(m);a[key]=bad
            with self.subTest(key=key),self.assertRaises(ValueError):policy.validate_synthetic_diagnostic(m,'final',actual=a,**e)
    def test_complete_resource_acceptance_requires_exact_run(self):
        m,_,_,_,e=fixture();r=resource(m,e['trusted_grant_sha256'])
        self.assertFalse(policy.validate_resource_receipt(m,r,trusted_grant_sha256=e['trusted_grant_sha256'])['qualified'])
        for field,bad in [('grant_sha256','f'*64),('wrapper_manifest_sha256','f'*64),('captured_total_CPU',42.),('child_complete_wait4_CPU',None),('parent_complete_CPU',float('nan')),('owned_group_children_reaped',False),('child_final_admission_status','HOLD'),('wall_before_final_receipt',601.),('outer_CPU_before_final_receipt',True)]:
            with self.subTest(field=field),self.assertRaises(ValueError):policy.validate_resource_receipt(m,{**r,field:bad},trusted_grant_sha256=e['trusted_grant_sha256'])
    def test_old_synthetic_fit_and_fake_no_fit_still_reject(self):
        m={'schema_version':'tac-osm.parity.v1','study_id':'NEW_SINGLE_SOFTWARE_TEST','study_type':'single_model_training','registered_before_first_fit':True,'real_world_source':{'observed_real_outcomes':False}}
        with self.assertRaisesRegex(ValueError,'authentic real source outcomes'):canonical.check_manifest(m)
        d={'schema_version':'tac-osm.parity.v1','study_id':'NEW_DIAGNOSTIC_SOFTWARE_TEST','study_type':'noncomparative_diagnostic','no_architecture_superiority_claim':True,'scientific_purpose':'A constructed no-fit software diagnostic','no_new_model_fits':False}
        with self.assertRaisesRegex(ValueError,'no-fit'):canonical.check_manifest(d)
        d['no_new_model_fits']=True;result=canonical.check_manifest(d)
        self.assertEqual(result['status'],'PASS_DIAGNOSTIC_ONLY');self.assertEqual(result['reward_integrity'],'NOT_APPLICABLE_HISTORICAL_OR_NO_FIT')
    def test_old_parity_function_ast_unchanged(self):
        tree=ast.parse((ROOT/'scripts/validate_research_parity.py').read_bytes())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_check_manifest_parity_only')
        self.assertEqual(hashlib.sha256(ast.dump(function,include_attributes=False).encode()).hexdigest(),'52812d3389ec2c52a3d20be2564e1625a9a8a57cc354845361689ccce58a8432')

if __name__=='__main__':unittest.main()
