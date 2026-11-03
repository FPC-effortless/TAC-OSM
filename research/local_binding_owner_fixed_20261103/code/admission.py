"""Truthful inherited admission check. Synthetic fits currently fail closed.

This is not a waiver. No draft policy proposal is imported by this module.
"""
import hashlib,json,sys,types
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def inspect_inherited(root,protocol,stage,regime):
    info=protocol['admission']
    for name,row in info['files'].items():
        p=root/name;assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
    path=root/info['validator_path'];module=types.ModuleType('binding_inherited_policy');module.__file__=str(path)
    previous_path=list(sys.path)
    try:exec(compile(path.read_bytes(),str(path),'exec',dont_inherit=True),module.__dict__)
    finally:sys.path[:]=previous_path
    rows=[]
    for family in protocol['families']:
        name=f'admission/registrations/{stage}/{regime}/{family}/FAIR_COMPARISON.json';m=json.loads((root/name).read_bytes())
        assert m['real_world_source']['observed_real_outcomes'] is False and m['synthetic_source']['generated_fixture'] is True
        try:
            result=module.check_manifest(m,'preflight')
            valid=result.get('reward_integrity')=='NOT_INDEPENDENTLY_CERTIFIED_INTERNAL_RESEARCH' and result.get('eligible_for_architecture_superiority') is False
            rows.append({'family':family,'manifest':name,'manifest_sha256':sha(root/name),'admitted':valid,'validator_result':result})
        except module.ParityError as error:rows.append({'family':family,'manifest':name,'manifest_sha256':sha(root/name),'admitted':False,'error_type':type(error).__name__,'reason':str(error)})
    return {'status':'PASS_EXISTING_ADMISSION' if all(r['admitted'] for r in rows) else 'BLOCKED_EXISTING_POLICY','stage':stage,'regime':regime,'policy_commit':info['policy_commit'],'validator_sha256':sha(path),'rows':rows,'policy_change_approved':False,'model_executions':0,'optimizer_updates':0}

def inspect(root,protocol,stage,regime):
    inherited=inspect_inherited(root,protocol,stage,regime)
    if stage!='fixed' or regime!='default':return inherited
    path=root/'code/approved_scope.py'
    entry=json.loads((root/'CODE_MANIFEST.json').read_bytes())['files']['code/approved_scope.py']
    if sha(path)!=entry['sha256']:raise ValueError('APPROVED_SCOPE_CODE_CHANGED')
    module=types.ModuleType('binding_owner_scope');module.__file__=str(path)
    exec(compile(path.read_bytes(),str(path),'exec',dont_inherit=True),module.__dict__)
    result=module.validate(root,protocol,stage,regime,inherited)
    result['inherited_receipt']=inherited
    result['model_executions']=0;result['optimizer_updates']=0
    return result

def enforce(root,protocol,grant,out):
    result=inspect(root,protocol,grant['stage'],grant['regime'])
    with (out/'ADMISSION_RESULT.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    if result['status']!='PASS_ONE_OFF_OWNER_SYNTHETIC_DIAGNOSTIC':raise RuntimeError('ADMISSION_BLOCKED: synthetic fits need an explicitly approved narrow policy path; source review and budget release alone do not waive the inherited gate')
    return result

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1];protocol=json.loads((root/'PROTOCOL.json').read_bytes())
    assert 'torch' not in sys.modules
    result={'status':'BLOCKED_EXISTING_POLICY','source_only_policy_check':True,'matrices':[inspect(root,protocol,s,r) for s in protocol['stages'] for r in protocol['regimes']]}
    assert 'torch' not in sys.modules
    (root/'admission/ADMISSION_PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
