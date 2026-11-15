"""Stdlib byte/receipt verifier; no Torch, forward, backward or new model score."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path,PurePosixPath
import struct
import xml.etree.ElementTree as ET
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'research/evidence/byte100k_001'
PINS={
 'LOCAL_ORIGINAL':'705ebc623ef629a25d694bc49f31d51789a9db83ec370128e10be8851c9cbe43',
 'HOSTED_ORIGINAL':'e8d46da4bd8318c398c7b4e6d1e48cc4c4dc9b2c50291be6d218e221319eb8c3',
 'HOSTED_REPLAY':'ceb8c19484d5315b21e1122de646501691bb816ab7e6df86711496bbbaf35ba2',
}


def check(condition,message):
    if not condition:raise ValueError(message)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def archives():
    manifest=json.loads((BASE/'ARCHIVES.json').read_text())['archives']
    result={}
    check(set(manifest)==set(PINS),'all original archives')
    for role,pin in PINS.items():
        pieces=[]
        for i,part in enumerate(manifest[role]['parts']):
            check(part['path']==role+'.zip.part'+str(i).zfill(3),'fixed ordered archive parts')
            raw=(BASE/part['path']).read_bytes()
            check(len(raw)==part['bytes'] and sha(raw)==part['sha256'],'part integrity')
            pieces.append(raw)
        raw=b''.join(pieces)
        check(sha(raw)==pin==manifest[role]['sha256'] and len(raw)==manifest[role]['bytes'],'lossless original ZIP')
        result[role]=raw
    return result


def inspect(z,prefix):
    raw_index=z.read(prefix+'INDEX.json');index=json.loads(raw_index)
    check(index['variant_id']=='BYTE100K-001' and not index['trained'],'untrained variant')
    for path,pin in index['source_sha256'].items():
        check(sha((ROOT/path).read_bytes())==pin,'frozen source '+path)
    for seed,entry in index['models'].items():
        folder=prefix+'models/'+seed+'/'
        raw=z.read(folder+'manifest.json');manifest=json.loads(raw)
        check(sha(raw)==entry['manifest_sha256'],'original inference manifest pin')
        raw=z.read(folder+'tensors.safetensors')
        check(sha(raw)==manifest['tensor_sha256'] and len(raw)==manifest['tensor_bytes'],'original model/state payload')
        header=json.loads(raw[8:8+struct.unpack('<Q',raw[:8])[0]])
        check(sum(math.prod(v['shape']) for k,v in header.items() if k.startswith('parameter.'))==99050,'original actual model count')
        check(not any(any(x in k for x in ('tool_head','field_head','accept_head')) for k in header if k.startswith('parameter.')),'no learned tool head capacity')
    for name,pin in index['observed'].items():
        check(sha(z.read(prefix+'observed/'+name))==pin,'original diagnostic tensor bytes')
    result=json.loads(z.read(prefix+'RESULTS.json'))
    check(result['status']=='PASS' and not result['trained'] and not result['independently_qualified'],'receipt status/scope')
    check(result['optimizer_updates']==result['GPU_minutes']==0 and not result['DEV_model_scoring'] and not result['protected_TEST_opened'],'no fit/GPU/DEV scoring/TEST')
    check(len(result['models'])==3 and len(result['cases'])==6,'full3seed2document matrix')
    for m in result['models']:
        check(m['trainable']==m['deployed']==99050 and m['unchanged'] and m['model_identity_before']==m['model_identity_after']==index['models'][str(m['seed'])]['model_identity'],'unchanged original model')
    for r in result['cases']:
        check(r['status']=='PASS' and all(r['checks'].values()) and r['gradient_tensors']==80 and len(r['gradient_rows'])==80,'all source/causal/gradient checks')
        check(all(v['connected'] and v['finite_nonzero'] and v['within_tolerance'] for v in r['gradient_rows'].values()),'full parameter tensor coverage')
    return raw_index,index,result


def main(extract_original=None):
    payloads=archives()
    loaded={}
    for role,raw in payloads.items():
        with ZipFile(io.BytesIO(raw)) as z:
            check(z.testzip() is None,'ZIP CRC')
            loaded[role]=inspect(z,'capture/' if role=='HOSTED_ORIGINAL' else '')
            if role=='HOSTED_ORIGINAL':
                suites=ET.fromstring(z.read('TESTS.xml')).findall('testsuite')
                check(sum(int(s.attrib['tests']) for s in suites)==59,'original59software tests')
                check(all(int(s.attrib[k])==0 for s in suites for k in ('errors','failures','skipped')),'software outcomes')
                check(z.read('CPU_PLAN.json')==(ROOT/'research/byte100k_001/COMPUTE_DECISION.json').read_bytes(),'original CPU plan')
    original,index,_=loaded['HOSTED_ORIGINAL'];replay,replay_index,result=loaded['HOSTED_REPLAY']
    check(original==replay,'cross-run original model and observed tensor INDEX byte equality')
    check(all(m['different_prototype_restored'] for m in result['models']),'different prototypes restored')
    check(len(result['original_tensor_replay'])==6 and all(r['status']=='PASS' and r['tensors']==105 and r['maximum_absolute_error']==0 for r in result['original_tensor_replay']),'all6x105original tensor replay')
    check((BASE/'LOCAL_REPLAY_INDEX.json').read_bytes()==loaded['LOCAL_ORIGINAL'][0],'local original INDEX replay identity')
    local=json.loads((BASE/'LOCAL_REPLAY_RESULTS.json').read_text())
    check(local['status']=='PASS' and len(local['original_tensor_replay'])==6 and all(r['maximum_absolute_error']==0 for r in local['original_tensor_replay']),'local original replay receipt')
    failed=json.loads((ROOT/'research/byte100k_001/initial/FAILURE.json').read_text())
    for source,saved in [('src/tac_osm/byte100k_001.py','model.py'),('research/byte100k_001/checks.py','checks.py'),('tests/test_byte100k_001.py','tests.py')]:
        check(sha((ROOT/'research/byte100k_001/initial'/saved).read_bytes())==failed['source_sha256'][source],'original failed source preserved')
    check('2 failed, 6 passed' in (ROOT/'research/byte100k_001/initial/TEST_REPRODUCTION.log').read_text(),'initial reproduced failure remains')
    math_link=json.loads((ROOT/'research/byte100k_001/MATH_LINK.json').read_text())
    check(sha((ROOT/math_link['tac_model_path']).read_bytes())==math_link['tac_model_sha256'],'math/source pin')
    check(sha((ROOT/'research/byte100k_001/ByteLanguage100K.lean').read_bytes())==math_link['lean_sha256'],'exact compiled Lean copy')
    check(sha((ROOT/'research/byte100k_001/math_reference.py').read_bytes())==math_link['python_reference_sha256'],'independent Python copy')
    axiom_log=(ROOT/'research/byte100k_001/PNDS_AXIOM_EXCERPT.log').read_text()
    check('Built PNDS.ByteLanguage100K' in axiom_log and 'sorryAx' not in axiom_log,'explicit scoped compilation/axiom excerpt')
    for declaration in math_link['lean_declarations']:
        check(declaration in axiom_log,'compiled/audited declaration')
    if extract_original is not None:
        destination=Path(extract_original)
        check(not destination.exists() and not destination.is_symlink(),'extract only to NEW owned folder')
        with ZipFile(io.BytesIO(payloads['HOSTED_ORIGINAL'])) as z:
            for name in z.namelist():
                p=PurePosixPath(name)
                check(not p.is_absolute() and '..' not in p.parts and '\\' not in name,'safe original archive path')
            destination.mkdir(parents=True)
            for name in z.namelist():
                p=destination/name
                if name.endswith('/'):p.mkdir(parents=True,exist_ok=True)
                else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
        print('Original capture extracted:',str(destination/'capture'),'INDEX_SHA256='+sha(original))
    print('PASS: lossless original ZIPs/model/observed bytes, actual99050counts, 59tests, 6x105zero-error original replays and initial failure; no model executed')


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--extract-original',type=Path)
    main(cli.parse_args().extract_original)
