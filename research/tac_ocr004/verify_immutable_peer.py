"""Phase-one OCR004 immutable byte contracts, not dispatcher activation.

No pin defaults or moving branch URLs. This module imports no model or math
reference. A caller must supply a separately reviewed immutable metadata pin.
It does not verify pin authority, source rights, execution, Lean or training.
"""
import hashlib
import json
import re
from pathlib import Path

SOURCE = '84d0801d902b44c28f355b5a484e56b34751e713'
SOURCE_HASHES = {'research/tac_ocr001/model.py': 'af541347b34fd58b4b67657fb162c04327e35cb93ba7a61ee6e29c5e752b60d6', 'research/tac_ocr001/steady.py': '6fac8a5a1a1303374a0c2380983bed003187b17484469ed9c11e449922f029d3', 'research/tac_ocr001/encoder.py': '5cde42ec290880f869c2c44025df9f04285729f3fe28e79b0885614741c13d9e', 'research/tac_ocr001/tokenizer.py': '09e35bff24a091f4648c67ca47256f9db6c2f52a179b482a16c7322bbda686ec', 'research/tac_ocr001/presets.py': 'e225eda514fe56f4efd6f21f366ae0db8082e8aa16a440418dec522bed7d82d3', 'research/tac_ocr001/probe_adapter.py': '8b988dfebb8a2369013aa685a520eacc5ded743f20df0560336251e7ce2afd88', 'research/tac_ocr001/pipeline.py': 'd0534993d94b864a339b6cccaae0afd8746df875ec177a77f66c2375b9368d95', 'research/langcache104/model.py': '84507fd17ace7eb2e880d5f15ee7a3a80c41543041eee103ca926966bcd1e882', 'research/langcache104/kernels.py': '1c2782bbaaac7aa54722f420bbbad8c5215f260ed591a6cef154c0540f41b8d1'}
MATH_HASHES = {'PNDS/TACOCR004Lifecycle.lean': '1b828a562221760f455aaabb44019887e79d1c973cdbff299b131d1209cbdb3c', 'foundation/tac_ocr004/lifecycle_reference.py': '9e3fd171f9515582155242a368098578cd200c808316ca56184d8a7157364b73', 'foundation/tac_ocr004/verify_bindings.py': '350456fe054b2a4752552df46b28b37d2fb8792c90301ba8f3fbbe12b2c800d0'}
OLD_SOURCE = 'd2700f9b3864dab0d71a59a96643bbb72635f144'
OLD_SOURCE_HASHES = {'research/tac_ocr001/model.py': 'db17b400af293c4c03ffe98f9fb87abe07446a1b41305205def2eeb103138fcb', 'research/tac_ocr001/steady.py': '52aabae41fb93b52214ea1cf227e19264f1e6df2d121cfe90af635242c8118d6', 'research/tac_ocr001/encoder.py': '5cde42ec290880f869c2c44025df9f04285729f3fe28e79b0885614741c13d9e', 'research/tac_ocr001/tokenizer.py': '09e35bff24a091f4648c67ca47256f9db6c2f52a179b482a16c7322bbda686ec', 'research/tac_ocr001/presets.py': 'e225eda514fe56f4efd6f21f366ae0db8082e8aa16a440418dec522bed7d82d3', 'research/tac_ocr001/probe_adapter.py': '8b988dfebb8a2369013aa685a520eacc5ded743f20df0560336251e7ce2afd88', 'research/tac_ocr001/pipeline.py': 'd0534993d94b864a339b6cccaae0afd8746df875ec177a77f66c2375b9368d95', 'research/langcache104/model.py': '84507fd17ace7eb2e880d5f15ee7a3a80c41543041eee103ca926966bcd1e882', 'research/langcache104/kernels.py': '1c2782bbaaac7aa54722f420bbbad8c5215f260ed591a6cef154c0540f41b8d1'}
OLD_MATH_HASHES = {'PNDS/TACOCR001.lean': '4f8e307938d9f962369fd76ec3cd80a8e63e7231aa55c427cf2f45c3d805d1fb', 'foundation/tac_ocr001/FORMULATION.md': '9f548bc99c25384506dcd6dc11fc2731122aef050a680ab9a68895c14cd0e88e', 'foundation/tac_ocr001/math_reference.py': 'c87951af2081c7db36eaf47c7cb90e0e3b35e7562fc356f3209ca704fcf5fbd4', 'foundation/tac_ocr001/test_math_reference.py': '95fe5bdcb295387c35cd791f5b17a9bd96b02b3598d9b453e3e34e5167dc3e4f', 'foundation/tac_ocr001/test_reciprocal_candidate.py': '782b1a7be3c0ff3b59f93130a7fd06e4593e34bad846b3e76665c0511f7257a2'}
EXPECTED_MAPS = {'tac': {'schema': 'pnds.tac_ocr004.prospective_source_map.v1', 'variant_id': 'TAC-OCR-004-TAC', 'tac_repository': 'FPC-effortless/TAC-OSM', 'tac_commit': '84d0801d902b44c28f355b5a484e56b34751e713', 'tac_tree': '461643ae9e62990c591b3ca67d5a93b5c79e1e3e', 'ancestry': ['TAC-OCR-001-TAC'], 'family': 'tac', 'prior_map_sha256': 'fa2bc697e2ddba109429e07973c3fcbab78f21d6182a298fbf085e85f4242144', 'prior_pnds_commit': '60597e5b0f2690eab19af6d1f1278f1b8eb9e80f', 'tac_source_sha256': {'research/tac_ocr001/model.py': 'af541347b34fd58b4b67657fb162c04327e35cb93ba7a61ee6e29c5e752b60d6', 'research/tac_ocr001/steady.py': '6fac8a5a1a1303374a0c2380983bed003187b17484469ed9c11e449922f029d3', 'research/tac_ocr001/encoder.py': '5cde42ec290880f869c2c44025df9f04285729f3fe28e79b0885614741c13d9e', 'research/tac_ocr001/tokenizer.py': '09e35bff24a091f4648c67ca47256f9db6c2f52a179b482a16c7322bbda686ec', 'research/tac_ocr001/presets.py': 'e225eda514fe56f4efd6f21f366ae0db8082e8aa16a440418dec522bed7d82d3', 'research/tac_ocr001/probe_adapter.py': '8b988dfebb8a2369013aa685a520eacc5ded743f20df0560336251e7ce2afd88', 'research/tac_ocr001/pipeline.py': 'd0534993d94b864a339b6cccaae0afd8746df875ec177a77f66c2375b9368d95', 'research/langcache104/model.py': '84507fd17ace7eb2e880d5f15ee7a3a80c41543041eee103ca926966bcd1e882', 'research/langcache104/kernels.py': '1c2782bbaaac7aa54722f420bbbad8c5215f260ed591a6cef154c0540f41b8d1'}, 'changed_source_paths': ['research/tac_ocr001/model.py', 'research/tac_ocr001/steady.py'], 'lean_path_proposed': 'PNDS/TACOCR004Lifecycle.lean', 'python_path_proposed': 'foundation/tac_ocr004/lifecycle_reference.py', 'status': 'DRAFT_LOCAL_BYTES_NOT_PUBLIC_PEER_AUTHORITY', 'lean_compiled': False, 'source_lifecycle_projection_checked': False, 'full_source_equivalence_verified': False, 'whole_model_mathematical_readiness': False, 'training_authorized': False, 'open_obligations': ['WHOLE_ENCODER_BRIDGE_CORE_SOURCE_LOWERING', 'ALL_PARAMETER_AND_STATE_GRADIENT_CORRESPONDENCE', 'FINITE_PRECISION_STABILITY', 'POOLING_INFORMATION_RETENTION', 'FROZEN_GRAPH_UNIVERSAL_EQUIVALENCE', 'REAL_IMAGE_TO_TRAINED_TRANSCRIPTION', 'SOURCE_QUALIFIED_MATCHED_TRAINING_AND_HELDOUT_OCR', 'RECIPROCAL_BOTH_REPOSITORY_REGISTRATION', 'LEAN_BUILD_AND_AXIOM_AUDIT_NEW_LIFECYCLE', 'ALL_SOURCE_LIFETIME_MODES_AND_SHAPES', 'UUID_FRESHNESS_PREMISE_NOT_CRYPTOGRAPHIC_PROOF'], 'previous_evidence_inherited_as_history_only': True, 'historical_LABOPS005_draft_mathematical_files_sha256': {'TACOCR004Lifecycle.lean': '1b828a562221760f455aaabb44019887e79d1c973cdbff299b131d1209cbdb3c', 'lifecycle_reference.py': '9e3fd171f9515582155242a368098578cd200c808316ca56184d8a7157364b73', 'check_correspondence.py': '55a5c58cfa0c47eb2a182102a72ec2d65c886b56c40bad6ffa4e7aad3dce8c75'}, 'mathematical_files_sha256': {'PNDS/TACOCR004Lifecycle.lean': '1b828a562221760f455aaabb44019887e79d1c973cdbff299b131d1209cbdb3c', 'foundation/tac_ocr004/lifecycle_reference.py': '9e3fd171f9515582155242a368098578cd200c808316ca56184d8a7157364b73', 'foundation/tac_ocr004/verify_bindings.py': '350456fe054b2a4752552df46b28b37d2fb8792c90301ba8f3fbbe12b2c800d0'}, 'prior_source_lifecycle_projection': 'LABOPS005 historical27source combinations only; not rerun', 'tac_source_paths': ['research/tac_ocr001/model.py', 'research/tac_ocr001/steady.py', 'research/tac_ocr001/encoder.py', 'research/tac_ocr001/tokenizer.py', 'research/tac_ocr001/presets.py', 'research/tac_ocr001/probe_adapter.py', 'research/tac_ocr001/pipeline.py', 'research/langcache104/model.py', 'research/langcache104/kernels.py'], 'lean_module': 'PNDS/TACOCR004Lifecycle.lean', 'python_reference': 'foundation/tac_ocr004/lifecycle_reference.py'}, 'transformer': {'schema': 'pnds.tac_ocr004.prospective_source_map.v1', 'variant_id': 'TAC-OCR-004-TRANSFORMER', 'tac_repository': 'FPC-effortless/TAC-OSM', 'tac_commit': '84d0801d902b44c28f355b5a484e56b34751e713', 'tac_tree': '461643ae9e62990c591b3ca67d5a93b5c79e1e3e', 'ancestry': ['TAC-OCR-001-TRANSFORMER'], 'family': 'transformer', 'prior_map_sha256': '3b167b539a9a415d48e5ccb84f8154ec32fd86e6d14708abe8918d166eac013e', 'prior_pnds_commit': '60597e5b0f2690eab19af6d1f1278f1b8eb9e80f', 'tac_source_sha256': {'research/tac_ocr001/model.py': 'af541347b34fd58b4b67657fb162c04327e35cb93ba7a61ee6e29c5e752b60d6', 'research/tac_ocr001/steady.py': '6fac8a5a1a1303374a0c2380983bed003187b17484469ed9c11e449922f029d3', 'research/tac_ocr001/encoder.py': '5cde42ec290880f869c2c44025df9f04285729f3fe28e79b0885614741c13d9e', 'research/tac_ocr001/tokenizer.py': '09e35bff24a091f4648c67ca47256f9db6c2f52a179b482a16c7322bbda686ec', 'research/tac_ocr001/presets.py': 'e225eda514fe56f4efd6f21f366ae0db8082e8aa16a440418dec522bed7d82d3', 'research/tac_ocr001/probe_adapter.py': '8b988dfebb8a2369013aa685a520eacc5ded743f20df0560336251e7ce2afd88', 'research/tac_ocr001/pipeline.py': 'd0534993d94b864a339b6cccaae0afd8746df875ec177a77f66c2375b9368d95', 'research/langcache104/model.py': '84507fd17ace7eb2e880d5f15ee7a3a80c41543041eee103ca926966bcd1e882', 'research/langcache104/kernels.py': '1c2782bbaaac7aa54722f420bbbad8c5215f260ed591a6cef154c0540f41b8d1'}, 'changed_source_paths': ['research/tac_ocr001/model.py', 'research/tac_ocr001/steady.py'], 'lean_path_proposed': 'PNDS/TACOCR004Lifecycle.lean', 'python_path_proposed': 'foundation/tac_ocr004/lifecycle_reference.py', 'status': 'DRAFT_LOCAL_BYTES_NOT_PUBLIC_PEER_AUTHORITY', 'lean_compiled': False, 'source_lifecycle_projection_checked': False, 'full_source_equivalence_verified': False, 'whole_model_mathematical_readiness': False, 'training_authorized': False, 'open_obligations': ['WHOLE_ENCODER_BRIDGE_CORE_SOURCE_LOWERING', 'ALL_PARAMETER_AND_STATE_GRADIENT_CORRESPONDENCE', 'FINITE_PRECISION_STABILITY', 'POOLING_INFORMATION_RETENTION', 'FROZEN_GRAPH_UNIVERSAL_EQUIVALENCE', 'REAL_IMAGE_TO_TRAINED_TRANSCRIPTION', 'SOURCE_QUALIFIED_MATCHED_TRAINING_AND_HELDOUT_OCR', 'RECIPROCAL_BOTH_REPOSITORY_REGISTRATION', 'LEAN_BUILD_AND_AXIOM_AUDIT_NEW_LIFECYCLE', 'ALL_SOURCE_LIFETIME_MODES_AND_SHAPES', 'UUID_FRESHNESS_PREMISE_NOT_CRYPTOGRAPHIC_PROOF'], 'previous_evidence_inherited_as_history_only': True, 'historical_LABOPS005_draft_mathematical_files_sha256': {'TACOCR004Lifecycle.lean': '1b828a562221760f455aaabb44019887e79d1c973cdbff299b131d1209cbdb3c', 'lifecycle_reference.py': '9e3fd171f9515582155242a368098578cd200c808316ca56184d8a7157364b73', 'check_correspondence.py': '55a5c58cfa0c47eb2a182102a72ec2d65c886b56c40bad6ffa4e7aad3dce8c75'}, 'mathematical_files_sha256': {'PNDS/TACOCR004Lifecycle.lean': '1b828a562221760f455aaabb44019887e79d1c973cdbff299b131d1209cbdb3c', 'foundation/tac_ocr004/lifecycle_reference.py': '9e3fd171f9515582155242a368098578cd200c808316ca56184d8a7157364b73', 'foundation/tac_ocr004/verify_bindings.py': '350456fe054b2a4752552df46b28b37d2fb8792c90301ba8f3fbbe12b2c800d0'}, 'prior_source_lifecycle_projection': 'LABOPS005 historical27source combinations only; not rerun', 'tac_source_paths': ['research/tac_ocr001/model.py', 'research/tac_ocr001/steady.py', 'research/tac_ocr001/encoder.py', 'research/tac_ocr001/tokenizer.py', 'research/tac_ocr001/presets.py', 'research/tac_ocr001/probe_adapter.py', 'research/tac_ocr001/pipeline.py', 'research/langcache104/model.py', 'research/langcache104/kernels.py'], 'lean_module': 'PNDS/TACOCR004Lifecycle.lean', 'python_reference': 'foundation/tac_ocr004/lifecycle_reference.py'}, 'gru': {'schema': 'pnds.tac_ocr004.prospective_source_map.v1', 'variant_id': 'TAC-OCR-004-GRU', 'tac_repository': 'FPC-effortless/TAC-OSM', 'tac_commit': '84d0801d902b44c28f355b5a484e56b34751e713', 'tac_tree': '461643ae9e62990c591b3ca67d5a93b5c79e1e3e', 'ancestry': ['TAC-OCR-001-GRU'], 'family': 'gru', 'prior_map_sha256': '56d51a50bbf07accdbe9ddaa252fd6211e0e698bfdb7c9a89b49bf035d8be9d9', 'prior_pnds_commit': '60597e5b0f2690eab19af6d1f1278f1b8eb9e80f', 'tac_source_sha256': {'research/tac_ocr001/model.py': 'af541347b34fd58b4b67657fb162c04327e35cb93ba7a61ee6e29c5e752b60d6', 'research/tac_ocr001/steady.py': '6fac8a5a1a1303374a0c2380983bed003187b17484469ed9c11e449922f029d3', 'research/tac_ocr001/encoder.py': '5cde42ec290880f869c2c44025df9f04285729f3fe28e79b0885614741c13d9e', 'research/tac_ocr001/tokenizer.py': '09e35bff24a091f4648c67ca47256f9db6c2f52a179b482a16c7322bbda686ec', 'research/tac_ocr001/presets.py': 'e225eda514fe56f4efd6f21f366ae0db8082e8aa16a440418dec522bed7d82d3', 'research/tac_ocr001/probe_adapter.py': '8b988dfebb8a2369013aa685a520eacc5ded743f20df0560336251e7ce2afd88', 'research/tac_ocr001/pipeline.py': 'd0534993d94b864a339b6cccaae0afd8746df875ec177a77f66c2375b9368d95', 'research/langcache104/model.py': '84507fd17ace7eb2e880d5f15ee7a3a80c41543041eee103ca926966bcd1e882', 'research/langcache104/kernels.py': '1c2782bbaaac7aa54722f420bbbad8c5215f260ed591a6cef154c0540f41b8d1'}, 'changed_source_paths': ['research/tac_ocr001/model.py', 'research/tac_ocr001/steady.py'], 'lean_path_proposed': 'PNDS/TACOCR004Lifecycle.lean', 'python_path_proposed': 'foundation/tac_ocr004/lifecycle_reference.py', 'status': 'DRAFT_LOCAL_BYTES_NOT_PUBLIC_PEER_AUTHORITY', 'lean_compiled': False, 'source_lifecycle_projection_checked': False, 'full_source_equivalence_verified': False, 'whole_model_mathematical_readiness': False, 'training_authorized': False, 'open_obligations': ['WHOLE_ENCODER_BRIDGE_CORE_SOURCE_LOWERING', 'ALL_PARAMETER_AND_STATE_GRADIENT_CORRESPONDENCE', 'FINITE_PRECISION_STABILITY', 'POOLING_INFORMATION_RETENTION', 'FROZEN_GRAPH_UNIVERSAL_EQUIVALENCE', 'REAL_IMAGE_TO_TRAINED_TRANSCRIPTION', 'SOURCE_QUALIFIED_MATCHED_TRAINING_AND_HELDOUT_OCR', 'RECIPROCAL_BOTH_REPOSITORY_REGISTRATION', 'LEAN_BUILD_AND_AXIOM_AUDIT_NEW_LIFECYCLE', 'ALL_SOURCE_LIFETIME_MODES_AND_SHAPES', 'UUID_FRESHNESS_PREMISE_NOT_CRYPTOGRAPHIC_PROOF'], 'previous_evidence_inherited_as_history_only': True, 'historical_LABOPS005_draft_mathematical_files_sha256': {'TACOCR004Lifecycle.lean': '1b828a562221760f455aaabb44019887e79d1c973cdbff299b131d1209cbdb3c', 'lifecycle_reference.py': '9e3fd171f9515582155242a368098578cd200c808316ca56184d8a7157364b73', 'check_correspondence.py': '55a5c58cfa0c47eb2a182102a72ec2d65c886b56c40bad6ffa4e7aad3dce8c75'}, 'mathematical_files_sha256': {'PNDS/TACOCR004Lifecycle.lean': '1b828a562221760f455aaabb44019887e79d1c973cdbff299b131d1209cbdb3c', 'foundation/tac_ocr004/lifecycle_reference.py': '9e3fd171f9515582155242a368098578cd200c808316ca56184d8a7157364b73', 'foundation/tac_ocr004/verify_bindings.py': '350456fe054b2a4752552df46b28b37d2fb8792c90301ba8f3fbbe12b2c800d0'}, 'prior_source_lifecycle_projection': 'LABOPS005 historical27source combinations only; not rerun', 'tac_source_paths': ['research/tac_ocr001/model.py', 'research/tac_ocr001/steady.py', 'research/tac_ocr001/encoder.py', 'research/tac_ocr001/tokenizer.py', 'research/tac_ocr001/presets.py', 'research/tac_ocr001/probe_adapter.py', 'research/tac_ocr001/pipeline.py', 'research/langcache104/model.py', 'research/langcache104/kernels.py'], 'lean_module': 'PNDS/TACOCR004Lifecycle.lean', 'python_reference': 'foundation/tac_ocr004/lifecycle_reference.py'}}
LEGACY_ROLE_DIGESTS = {'tac': 'df606d2fc7ca47f83b6637ea397c80e664820a514475dc536d91aca3119f8a5d', 'pnds': '7345ecf98ef988c9dae7162c5113c843a9d307d4c055b8692c7d164629810cd4'}
FAMILIES=('tac','transformer','gru')
REPOS={'tac':'FPC-effortless/TAC-OSM','pnds':'FPC-effortless/pnds-lean'}
REGISTRY={'tac':'research/pnds-variant-math-links.json','pnds':'foundation/v1/variant-math-links.json'}

def require(value,message):
    if not value:raise ValueError(message)

def digest(data):return hashlib.sha256(data).hexdigest()

def read_local(root,path):
    root=Path(root).resolve();target=(root/path).resolve()
    require(target.is_relative_to(root),'repository path escape')
    return target.read_bytes()

def url(repo,commit,path):
    require(repo in REPOS.values(),'unrecognized repository')
    require(isinstance(commit,str) and re.fullmatch('[0-9a-f]{40}',commit),'immutable commit required')
    require(isinstance(path,str) and path and not path.startswith('/') and '..' not in Path(path).parts,'remote path escape')
    return 'https://raw.githubusercontent.com/'+repo+'/'+commit+'/'+path

def map_path(role,family):
    return ('research/tac_ocr004/'+family+'/MATH_SOURCE_MAP.json') if role=='tac' else ('foundation/tac_ocr004/'+family+'/SOURCE_MAP.json')

def expected_row(family):
    return {'id':'TAC-OCR-004-'+family.upper(),'tac_source_commit':SOURCE,
      'source_map_path':map_path('pnds',family),'tac_source_map_path':map_path('tac',family),
      'lean_path':'PNDS/TACOCR004Lifecycle.lean','python_path':'foundation/tac_ocr004/lifecycle_reference.py','status':'OPEN'}

def validate_map(data,family):
    require(data==EXPECTED_MAPS[family],'frozen map fields or premises drift')
    require(data.get('schema')=='pnds.tac_ocr004.prospective_source_map.v1','map schema')
    require(data.get('variant_id')=='TAC-OCR-004-'+family.upper() and data.get('family')==family,'map family')
    require(data.get('tac_repository')==REPOS['tac'] and data.get('tac_commit')==SOURCE,'source identity')
    require(data.get('status')=='DRAFT_LOCAL_BYTES_NOT_PUBLIC_PEER_AUTHORITY','unearned map status')
    require(data.get('tac_source_paths')==list(SOURCE_HASHES) and data.get('tac_source_sha256')==SOURCE_HASHES,'canonical source closure')
    require(data.get('lean_module')=='PNDS/TACOCR004Lifecycle.lean' and data.get('python_reference')=='foundation/tac_ocr004/lifecycle_reference.py','canonical formulation paths')
    require(data.get('lean_path_proposed')==data['lean_module'] and data.get('python_path_proposed')==data['python_reference'],'proposed/canonical path drift')
    require(data.get('mathematical_files_sha256')==MATH_HASHES,'frozen mathematical closure')
    for key in ['lean_compiled','source_lifecycle_projection_checked','full_source_equivalence_verified','whole_model_mathematical_readiness','training_authorized']:
        require(data.get(key) is False,'unearned qualification flag: '+key)

def select_rows(registry):
    rows=registry.get('prospective_candidates')
    require(isinstance(rows,list) and all(isinstance(x,dict) for x in rows),'candidate array')
    ids=[x.get('id') for x in rows]
    require(len(ids)==len(set(ids)),'duplicate candidate')
    selected=[x for x in rows if str(x.get('id','')).startswith('TAC-OCR-004-')]
    require({x.get('id') for x in selected}=={'TAC-OCR-004-'+f.upper() for f in FAMILIES} and len(selected)==3,'missing or unknown OCR004 candidate')
    for family in FAMILIES:require(expected_row(family) in selected,'reciprocal row contract')
    return selected

def verify_immutable_peer(local_root,role,pin,read_remote):
    require(role in REPOS,'unknown local role')
    peer='pnds' if role=='tac' else 'tac'
    require(isinstance(pin,dict) and set(pin)=={'schema','peer_repository','metadata_commit','map_sha256'},'pin fields')
    require(pin['schema']=='pnds.tac_ocr004.immutable_peer.v1' and pin['peer_repository']==REPOS[peer],'pin schema/repository')
    commit=pin['metadata_commit'];url(REPOS[peer],commit,REGISTRY[peer])
    require(isinstance(pin['map_sha256'],dict) and set(pin['map_sha256'])==set(FAMILIES),'pin map inventory')
    local=json.loads(read_local(local_root,REGISTRY[role]))
    remote=json.loads(read_remote(url(REPOS[peer],commit,REGISTRY[peer])))
    left,right=dict(local),dict(remote);left.pop('prospective_candidates',None);right.pop('prospective_candidates',None)
    # Existing roles have distinct independently versioned LANGCAP103 metadata.
    # Preserve every original field per role, not a silent ignored-field subset.
    canonical=lambda value:json.dumps(value,sort_keys=True,separators=(',',':')).encode()
    require(digest(canonical(left))==LEGACY_ROLE_DIGESTS[role] and digest(canonical(right))==LEGACY_ROLE_DIGESTS[peer],'historical per-role registry snapshot drift')
    require(left.get('variants')==right.get('variants') and len(left.get('variants',[]))==17,'original seventeen variants drift')
    require(select_rows(local)==select_rows(remote),'prospective row drift')
    for family in FAMILIES:
        raw=read_local(local_root,map_path(role,family))
        require(raw==read_remote(url(REPOS[peer],commit,map_path(peer,family))),'peer map bytes drift')
        require(re.fullmatch('[0-9a-f]{64}',str(pin['map_sha256'][family])) is not None and digest(raw)==pin['map_sha256'][family],'peer map digest')
        validate_map(json.loads(raw),family)
    for path,sha in SOURCE_HASHES.items():
        actual=read_remote(url(REPOS['tac'],SOURCE,path))
        require(digest(actual)==sha,'immutable source bytes: '+path)
        if role=='tac':require(digest(read_local(local_root,path))==sha,'local current source drift: '+path)
    for path,sha in MATH_HASHES.items():
        actual=read_local(local_root,path) if role=='pnds' else read_remote(url(REPOS['pnds'],commit,path))
        require(digest(actual)==sha,'mathematical bytes: '+path)
    return {'status':'IMMUTABLE_PEER_BYTE_CORRESPONDENCE_ONLY','variants':3,'source_files':9,
      'other_candidates_verified':False,'global_existing_snapshot_parity_established':False,'canonical_dispatch_changed':False,'peer_activated':False,
      'lean_compiled':False,'whole_model_proved':False,'training_authorized':False}

def verify_historical_bytes(source_root,math_root):
    """Check full nine-file old source/five-file math closure, not old CLI execution."""
    for path,sha in OLD_SOURCE_HASHES.items():require(digest(read_local(source_root,path))==sha,'historical source mismatch: '+path)
    for path,sha in OLD_MATH_HASHES.items():require(digest(read_local(math_root,path))==sha,'historical math mismatch: '+path)
    return {'historical_source_commit':OLD_SOURCE,'source_files':9,'math_files':5,
      'scope':'SOURCE_AND_MATH_BYTES_NOT_ORIGINAL_VERIFIER_EXECUTION','original_legacy_verifier_executed':False,'whole_model_proved':False}
