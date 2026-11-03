"""Exact LABOPS026 authority; unchanged inherited real-data rejection retained."""
import hashlib,json,sys,types
from pathlib import Path
AUTH_SHA='5b94944d46b06fb125e2d5b7beab15b11623d6c727d336ed7a749e76a472b176'
EXCEPTION_SHA='fc6034cc838b8f1969a03c7393c531c96b6fc5a3ea09ece74cbee6695f9fe852'
SOURCE_SHA='5aa2c227e4ee3a58eadd96260fd74d7409a5ced56a1db1f86b99797794589787'
SCIENTIFIC_SHA='6c6ea4934e3f4ad9e386014613abe58de1962037b1a3257b793d0abc0a13f0c5'
VALIDATOR_SHA='fcfb5c2122129559b620eed27feb450f5b69d0a31437436e35824d6cfe52a78e'
class ScopeError(ValueError):pass
def require(c,m):
    if not c:raise ScopeError(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def inspect(root,protocol,stage,regime):
    root=Path(root)
    require(stage=='fixed' and regime=='default','OUTSIDE_FRESH_GRU512_SCOPE')
    require(sha(root/'OWNER_AUTHORIZATION.json')==AUTH_SHA,'OWNER_AUTHORITY_BYTES_CHANGED')
    require(sha(root/'APPROVED_OWNER_EXCEPTION.json')==EXCEPTION_SHA,'EXCEPTION_BYTES_CHANGED')
    auth=json.loads((root/'OWNER_AUTHORIZATION.json').read_bytes())
    exc=json.loads((root/'APPROVED_OWNER_EXCEPTION.json').read_bytes())
    require(auth['user_response']=='Execution authority given','EXPLICIT_AUTHORITY_MISSING')
    require(sha(root/'SOURCE_MANIFEST.json')==SOURCE_SHA,'SOURCE_CHANGED')
    subset={k:protocol[k] for k in exc['scientific_protocol_keys']}
    require(hashlib.sha256(canonical(subset)).hexdigest()==SCIENTIFIC_SHA,'SCIENTIFIC_PROTOCOL_CHANGED')
    require(protocol['families']==['gru'] and protocol['seed']==18001,'FAMILY_SEED_CHANGED')
    require(protocol['stages']['fixed']['updates_per_arm']==512,'UPDATE_BUDGET_CHANGED')
    require(protocol['parameter_counts']=={'gru':100323},'MODEL_SCALE_CHANGED')
    require(protocol['retry_authorized'] is False and protocol['protected_TEST_authorized'] is False,'FORBIDDEN_AUTHORITY')
    require(protocol['fixture_files']['no_heldout_groups.json']['sha256']==hashlib.sha256(b'[]\n').hexdigest(),'HELDOUT_ADDED')
    for path,entry in protocol['admission']['files'].items():
        require(sha(root/path)==entry['sha256'] and (root/path).stat().st_size==entry['bytes'],'ADMISSION_SOURCE_CHANGED '+path)
    validator=root/protocol['admission']['validator_path']
    require(sha(validator)==VALIDATOR_SHA,'INHERITED_VALIDATOR_CHANGED')
    module=types.ModuleType('labops026_inherited_policy');module.__file__=str(validator)
    previous=list(sys.path)
    try:exec(compile(validator.read_bytes(),str(validator),'exec',dont_inherit=True),module.__dict__)
    finally:sys.path[:]=previous
    path=root/'admission/registrations/fixed/default/gru/FAIR_COMPARISON.json'
    manifest=json.loads(path.read_bytes())
    require(manifest['synthetic_source']['generated_fixture'] is True and manifest['real_world_source']['observed_real_outcomes'] is False,'SYNTHETIC_MISLABELLED')
    try:module.check_manifest(manifest,'preflight')
    except module.ParityError as error:
        require(str(error)=='authentic real source outcomes required','UNEXPECTED_INHERITED_FAILURE '+str(error))
        inherited={'status':'BLOCKED_EXISTING_POLICY','validator_sha256':VALIDATOR_SHA,'admitted':False,'reason':str(error),'manifest_sha256':sha(path)}
    else:raise ScopeError('INHERITED_REJECTION_NOT_PRESERVED')
    return {'status':'PASS_ONE_OFF_OWNER_SYNTHETIC_DIAGNOSTIC','scope_id':'LABOPS026_GRU512_20261103','owner_authorization_sha256':AUTH_SHA,'exception_sha256':EXCEPTION_SHA,'stage':stage,'regime':regime,'total_updates_allowed':512,'inherited_receipt':inherited,'inherited_rejections_preserved':True,'real_world_outcomes':False,'eligible_for_architecture_superiority':False,'independently_qualified':False,'promotion_authorized':False,'protected_TEST_authorized':False,'automatic_retry_authorized':False,'new_model_executions_at_admission':0}
def enforce(root,protocol,grant,out):
    require(grant['allocation_id']=='LABOPS026_GRU512_20261103','WRONG_ALLOCATION')
    result=inspect(root,protocol,grant['stage'],grant['regime'])
    with (out/'ADMISSION_RESULT.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    return result
