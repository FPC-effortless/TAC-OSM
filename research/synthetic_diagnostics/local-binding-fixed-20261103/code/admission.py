"""Exact approved canonical synthetic policy; externally anchored run evidence.

No scope-only check or manifest boolean authorizes a fit. This module imports
no Torch. The old rejected v1 admission and all historical sources are retained.
"""
import hashlib,json,types
from pathlib import Path

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load_policy(root,protocol):
    info=protocol['admission']
    for name,row in info['files'].items():
        path=root/name;assert sha(path)==row['sha256'] and path.stat().st_size==row['bytes']
    p=root/'policy/scripts/synthetic_diagnostic_policy.py';module=types.ModuleType('canonical_synthetic_policy');module.__file__=str(p);exec(compile(p.read_bytes(),str(p),'exec',dont_inherit=True),module.__dict__)
    m=json.loads((root/'FAIR_COMPARISON.json').read_bytes())
    return module,m

def inputs(root,protocol,grant,approval_sha,grant_sha):
    grant_bytes=Path(grant['grant_path']).read_bytes();assert hashlib.sha256(grant_bytes).hexdigest()==grant_sha
    assert json.dumps(json.loads(grant_bytes),sort_keys=True,allow_nan=False)==json.dumps(grant,sort_keys=True,allow_nan=False)
    assert grant['protocol_sha256']==sha(root/'PROTOCOL.json') and json.loads((root/'PROTOCOL.json').read_bytes())==protocol
    assert grant['policy_manifest_sha256']==sha(root/'policy/MANIFEST.json')
    module,m=load_policy(root,protocol)
    assert grant['protocol']=='LOCAL_BINDING_LEARNABILITY_V2'
    assert list(protocol['stages'])==['fixed'] and list(protocol['regimes'])==['default']
    assert grant['stage']==m['stage']=='fixed' and grant['regime']==m['regime']=='default'
    assert Path(grant['output_directory']).resolve()==root/'runs/fixed/default'
    spec=protocol['stages']['fixed']
    for key,expected in [('seed',protocol['seed']),('source_commit',protocol['source_commit']),('caps',spec['caps']),('checkpoints',spec['checkpoints']),('expected_counts',spec['expected_counts']),('objective',protocol['objective']),('optimizer',dict(protocol['optimizer_other_settings'],epsilon=protocol['regimes']['default']['epsilon']))]:assert module.same(m[key],expected)
    assert [a['family'] for a in m['arms']]==protocol['families']
    for arm in m['arms']:
        assert arm['updates']==spec['updates_per_arm'] and arm['active_parameters']==arm['deployed_parameters']==protocol['parameter_counts'][arm['family']]
    assert m['source']['recipe_sha256']==sha(root/'code/fixtures.py')
    for field,name in [('training_fixture_sha256','fixed_train_cases.json'),('schedule_sha256','fixed_schedule.json'),('heldout_fixture_sha256',spec['heldout_file'])]:assert m['source'][field]==protocol['fixture_files'][name]['sha256']
    assert module.same(m['memory_limits'],dict(address_space_bytes=8*1024**3,per_file_bytes=8*1024**2,child_output_bytes=32*1024**2,total_output_bytes=36*1024**2,child_file_count=64,total_file_count=80))
    assert grant['policy_manifest_sha256']==sha(root/'policy/MANIFEST.json')
    assert grant['runtime_pin_sha256']==sha(root/'RUNTIME_PIN.json')
    assert grant['protocol_sha256']==sha(root/'PROTOCOL.json')
    for name,key in [('SOURCE_MANIFEST.json','source_manifest_sha256'),('CODE_MANIFEST.json','code_manifest_sha256'),('wrapper/MANIFEST.json','wrapper_manifest_sha256')]:assert grant[key]==sha(root/name)
    evidence={'approval_bytes':Path(grant['approval_path']).read_bytes(),'grant_bytes':Path(grant['grant_path']).read_bytes(),'review_bytes':Path(grant['review_path']).read_bytes(),'trusted_approval_sha256':approval_sha,'trusted_grant_sha256':grant_sha}
    assert json.loads(evidence['grant_bytes'])==grant
    return module,m,evidence

def enforce(root,protocol,grant,out,approval_sha,grant_sha):
    module,m,evidence=inputs(root,protocol,grant,approval_sha,grant_sha)
    result=module.validate_synthetic_diagnostic(m,'preflight',**evidence)
    if out is not None:
        with (out/'ADMISSION_PREFLIGHT.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    return result

def final(root,protocol,grant,out,actual,approval_sha,grant_sha):
    module,m,evidence=inputs(root,protocol,grant,approval_sha,grant_sha)
    result=module.validate_synthetic_diagnostic(m,'final',actual=actual,**evidence)
    with (out/'ADMISSION_FINAL_INTEGRITY.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    return result

def resources(root,protocol,grant,actual,approval_sha,grant_sha):
    module,m,evidence=inputs(root,protocol,grant,approval_sha,grant_sha)
    module.validate_synthetic_diagnostic(m,'preflight',**evidence)
    return module.validate_resource_receipt(m,actual,trusted_grant_sha256=grant_sha)
