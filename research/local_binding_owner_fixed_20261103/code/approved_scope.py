"""One explicitly approved synthetic diagnostic, with all original gates retained.

This is an operator-scoped exception, not real-data qualification. No model,
optimizer, network operation or import of a neural library is performed here.
"""
import hashlib
import json
from pathlib import Path

AUTH_SHA='69c73cd73e58822b14049da7bea2ce8e733b8432c2f7596436836eb75d2a9750'
EXCEPTION_SHA='ac57dd42176da92adc9a88e85fe3f6ec37839b251bc0daa81cbc9d4e12410c2a'
SOURCE_SHA='5aa2c227e4ee3a58eadd96260fd74d7409a5ced56a1db1f86b99797794589787'
SCIENTIFIC_SHA='cc5692a66213a44644e9004d4bcb711a216538cdf9ba2e2dcc7f76228a2ab4ec'
INHERITED_VALIDATOR_SHA='fcfb5c2122129559b620eed27feb450f5b69d0a31437436e35824d6cfe52a78e'
UNCHANGED_CODE_SHA='fa4dcb751099bd41f63ea0611375fea0db444eeaeba21e28f0671d9f3d765c01'


class ScopeError(ValueError):pass


def require(condition,message):
    if not condition:raise ScopeError(message)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':')).encode()


def validate(root,protocol,stage,regime,inherited):
    root=Path(root)
    require(stage=='fixed' and regime=='default','OUTSIDE_ONE_RUN_SCOPE')
    require(digest(root/'OWNER_AUTHORIZATION.json')==AUTH_SHA,'OWNER_APPROVAL_BYTES_CHANGED')
    require(digest(root/'APPROVED_OWNER_EXCEPTION.json')==EXCEPTION_SHA,'EXCEPTION_BYTES_CHANGED')
    auth=json.loads((root/'OWNER_AUTHORIZATION.json').read_bytes())
    exc=json.loads((root/'APPROVED_OWNER_EXCEPTION.json').read_bytes())
    require(auth['user_response']=='Yes','EXPLICIT_OWNER_APPROVAL_MISSING')
    scope=auth['approved_scope']
    require(scope['updates_per_model']==128 and scope['total_updates']==384,'UPDATE_LIMIT_CHANGED')
    require(scope['max_aggregate_CPU_seconds']==360 and scope['max_wall_seconds']==600,'RESOURCE_LIMIT_CHANGED')
    require(auth['no_additional_stage_retry_or_budget_approved'] is True,'EXTRA_EXECUTION_NOT_APPROVED')
    require(digest(root/'SOURCE_MANIFEST.json')==SOURCE_SHA,'MODEL_SOURCE_MANIFEST_CHANGED')
    require(digest(root/protocol['admission']['validator_path'])==INHERITED_VALIDATOR_SHA,
            'INHERITED_VALIDATOR_CHANGED')
    subset={key:protocol[key] for key in exc['scientific_protocol_keys']}
    require(hashlib.sha256(canonical(subset)).hexdigest()==SCIENTIFIC_SHA,'SCIENTIFIC_PROTOCOL_CHANGED')
    require(protocol['families']==['gru','phase','context'] and protocol['seed']==18001,'ARM_OR_SEED_CHANGED')
    require(protocol['parameter_counts']=={'gru':100323,'phase':99971,'context':99971},'PARAMETER_DECLARATION_CHANGED')
    counts=list(protocol['parameter_counts'].values())
    require(all(95000<=n<=105000 for n in counts) and max(counts)/min(counts)-1<=.01,'PARAMETER_LIMIT')
    require(digest(root/'UNCHANGED_SCIENTIFIC_CODE.json')==UNCHANGED_CODE_SHA,'SCIENTIFIC_CODE_REGISTRY_CHANGED')
    for name,meta in json.loads((root/'UNCHANGED_SCIENTIFIC_CODE.json').read_bytes()).items():
        require(digest(root/name)==meta['sha256'],'SCIENTIFIC_CODE_CHANGED '+name)
    require(inherited['status']=='BLOCKED_EXISTING_POLICY','INHERITED_RESULT_NOT_PRESERVED')
    require(len(inherited['rows'])==3,'INHERITED_ARM_COUNT_CHANGED')
    for row in inherited['rows']:
        require(row['admitted'] is False and row['reason']=='authentic real source outcomes required',
                'UNEXPECTED_INHERITED_FAILURE')
        manifest=json.loads((root/row['manifest']).read_bytes())
        require(manifest['real_world_source']['observed_real_outcomes'] is False,'SYNTHETIC_AS_REAL')
        require(manifest['synthetic_source']['generated_fixture'] is True,'FIXTURE_PROVENANCE_CHANGED')
    return {'status':'PASS_ONE_OFF_OWNER_SYNTHETIC_DIAGNOSTIC','scope_id':exc['scope_id'],
            'owner_authorization_sha256':AUTH_SHA,'exception_sha256':EXCEPTION_SHA,
            'stage':stage,'regime':regime,'total_updates_allowed':384,
            'inherited_status':inherited['status'],'inherited_rejections_preserved':True,
            'real_world_outcomes':False,'eligible_for_architecture_superiority':False,
            'independently_qualified':False,'promotion_authorized':False,'protected_TEST_authorized':False,
            'automatic_retry_authorized':False,'new_model_executions_at_admission':0}
