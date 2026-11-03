"""Draft local two-root OCR004 binding check; no network/model/proof authority."""
import hashlib
import json
from pathlib import Path

FAMILIES = ('tac', 'transformer', 'gru')
SOURCE = '84d0801d902b44c28f355b5a484e56b34751e713'
SOURCE_SHA256 = {'research/tac_ocr001/model.py': 'af541347b34fd58b4b67657fb162c04327e35cb93ba7a61ee6e29c5e752b60d6', 'research/tac_ocr001/steady.py': '6fac8a5a1a1303374a0c2380983bed003187b17484469ed9c11e449922f029d3', 'research/tac_ocr001/encoder.py': '5cde42ec290880f869c2c44025df9f04285729f3fe28e79b0885614741c13d9e', 'research/tac_ocr001/tokenizer.py': '09e35bff24a091f4648c67ca47256f9db6c2f52a179b482a16c7322bbda686ec', 'research/tac_ocr001/presets.py': 'e225eda514fe56f4efd6f21f366ae0db8082e8aa16a440418dec522bed7d82d3', 'research/tac_ocr001/probe_adapter.py': '8b988dfebb8a2369013aa685a520eacc5ded743f20df0560336251e7ce2afd88', 'research/tac_ocr001/pipeline.py': 'd0534993d94b864a339b6cccaae0afd8746df875ec177a77f66c2375b9368d95', 'research/langcache104/model.py': '84507fd17ace7eb2e880d5f15ee7a3a80c41543041eee103ca926966bcd1e882', 'research/langcache104/kernels.py': '1c2782bbaaac7aa54722f420bbbad8c5215f260ed591a6cef154c0540f41b8d1'}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(root, relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('repository path escape')
    return path

def verify(tac_root, pnds_root):
    tac_root,pnds_root=Path(tac_root),Path(pnds_root)
    registries=[json.loads(load(root,path).read_text()) for root,path in
        ((tac_root,'research/pnds-variant-math-links.json'),
         (pnds_root,'foundation/v1/variant-math-links.json'))]
    rows=[]
    for family in FAMILIES:
        variant='TAC-OCR-004-'+family.upper()
        matches=[[r for r in registry['prospective_candidates'] if r['id']==variant] for registry in registries]
        if len(matches[0])!=1 or len(matches[1])!=1 or matches[0]!=matches[1]:
            raise ValueError('reciprocal row mismatch')
        row=matches[0][0]
        expected_row={'id':variant,'tac_source_commit':SOURCE,'source_map_path':'foundation/tac_ocr004/'+family+'/SOURCE_MAP.json','tac_source_map_path':'research/tac_ocr004/'+family+'/MATH_SOURCE_MAP.json','lean_path':'PNDS/TACOCR004Lifecycle.lean','python_path':'foundation/tac_ocr004/lifecycle_reference.py','status':'OPEN'}
        if row != expected_row:
            raise ValueError('reciprocal path/schema changed')
        if row['tac_source_commit']!=SOURCE or row['status']!='OPEN':
            raise ValueError('source/status changed')
        a=load(tac_root,row['tac_source_map_path'])
        b=load(pnds_root,row['source_map_path'])
        if a.read_bytes()!=b.read_bytes():
            raise ValueError('source map differs')
        data=json.loads(a.read_text())
        if data['schema']!='pnds.tac_ocr004.prospective_source_map.v1' or data['family']!=family or data['status']!='DRAFT_LOCAL_BYTES_NOT_PUBLIC_PEER_AUTHORITY':
            raise ValueError('map schema/family/status changed')
        if data['variant_id']!=variant or data['tac_commit']!=SOURCE:
            raise ValueError('source map identity changed')
        if any(data[key] is not False for key in ('lean_compiled','full_source_equivalence_verified','whole_model_mathematical_readiness','training_authorized')):
            raise ValueError('unearned qualification flag')
        if data['tac_source_sha256'] != SOURCE_SHA256:
            raise ValueError('source closure differs from externally frozen reviewed pins')
        for path,sha in data['tac_source_sha256'].items():
            if digest(load(tac_root,path))!=sha:
                raise ValueError('actual source changed: '+path)
        expected_math={'PNDS/TACOCR004Lifecycle.lean': '1b828a562221760f455aaabb44019887e79d1c973cdbff299b131d1209cbdb3c', 'foundation/tac_ocr004/lifecycle_reference.py': '9e3fd171f9515582155242a368098578cd200c808316ca56184d8a7157364b73'}
        expected_math['foundation/tac_ocr004/verify_bindings.py']=digest(Path(__file__).resolve())
        if data['mathematical_files_sha256'] != expected_math:
            raise ValueError('required frozen mathematical closure changed')
        for path,sha in data['mathematical_files_sha256'].items():
            if digest(load(pnds_root,path))!=sha:
                raise ValueError('math bytes changed: '+path)
        rows.append(variant)
    return {'status':'LOCAL_DRAFT_BINDING_BYTES_MATCH_NOT_PUBLIC_PEER_COMMIT_OR_PROOF',
            'variants':rows,'source_files_per_variant':9,'whole_model_proved':False,
            'lean_build':'UNRUN','new_models':0,'new_fits':0}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--tac-root',required=True);p.add_argument('--pnds-root',required=True)
    a=p.parse_args();print(json.dumps(verify(a.tac_root,a.pnds_root),indent=2))
