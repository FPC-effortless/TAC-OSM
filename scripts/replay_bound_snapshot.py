#!/usr/bin/env python3
"""Two-job UNTRAINED fixture persistence, never trained/source qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
import torch
from safetensors.torch import save,load
from tac_osm.bound_snapshot import save_bound_snapshot,restore_bound_snapshot
from tac_osm.stream_binding import capture_stream_binding
from research.bound_snapshot_20261112.fixture import model,prefix,context,continuation,values

CASES=('fp32','fp64')
SOURCE_PATHS=('src/tac_osm/bound_snapshot.py','scripts/replay_bound_snapshot.py',
              'research/bound_snapshot_20261112/fixture.py')


def check_source():
    p=json.loads((ROOT/'research/bound_snapshot_20261112/PROTOCOL.json').read_text())
    if not all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in p['source_pins'].items()):
        raise ValueError('inherited source changed')


def capture(output):
    if output.exists():raise FileExistsError('preserve prior fixture')
    check_source();output.mkdir(parents=True)
    index={'scope':'UNTRAINED_ENGINEERING_FIXTURE_ONLY','source_sha':{},'cases':{}}
    with torch.no_grad():
        for name in CASES:
            m=model(11,name=='fp64');s=prefix(m)
            pin=save_bound_snapshot(output/name,m,s,**context())
            # Original model predicts after save; no target labels/data/fit.
            expected=values(continuation(m,s))
            raw=save({n:t.contiguous().clone() for n,t in expected.items()})
            (output/name/'expected.safetensors').write_bytes(raw)
            index['cases'][name]={'manifest_sha256':pin,'expected_sha256':hashlib.sha256(raw).hexdigest(),
                                  'original_binding':capture_stream_binding(m,s,**context())}
    for path in SOURCE_PATHS:
        index['source_sha'][path]=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    raw=json.dumps(index,indent=2,sort_keys=True).encode()
    (output/'INDEX.json').write_bytes(raw)
    print(json.dumps({'stage':'capture','cases':2,'index_sha256':hashlib.sha256(raw).hexdigest(),
                      'trained':False,'optimizer_steps':0,'GPU_minutes':0,'TEST_opened':False}))


def replay(folder,pin,output):
    if output.exists():raise FileExistsError('preserve prior replay')
    check_source()
    raw=(folder/'INDEX.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('trusted index pin mismatch')
    index=json.loads(raw)
    if (set(index)!= {'scope','source_sha','cases'} or index['scope']!='UNTRAINED_ENGINEERING_FIXTURE_ONLY' or
        set(index['cases'])!=set(CASES) or set(index['source_sha'])!=set(SOURCE_PATHS)):
        raise ValueError('unexpected fixture index')
    if not all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in index['source_sha'].items()):
        raise ValueError('snapshot source mismatch')
    rows=[]
    with torch.no_grad():
        for name in CASES:
            proto=model(101,name=='fp64');proto.attention['2'].inv_freq.add_(.125)
            before=capture_stream_binding(proto,prefix(proto,4,0),**context(4,0))
            spec=index['cases'][name]
            restored,s=restore_bound_snapshot(folder/name,proto,
                expected_manifest_sha256=spec['manifest_sha256'],**context())
            binding=capture_stream_binding(restored,s,**context())
            different=before['model']!=spec['original_binding']['model']
            prototype_unchanged=capture_stream_binding(proto,prefix(proto,4,0),**context(4,0))==before
            binding_exact=binding==spec['original_binding']
            # Compute first. Expected evaluation payload is not a model input.
            actual=values(continuation(restored,s))
            reference_raw=(folder/name/'expected.safetensors').read_bytes()
            if hashlib.sha256(reference_raw).hexdigest()!=spec['expected_sha256']:
                raise ValueError('expected fixture hash changed')
            expected=load(reference_raw)
            checks={};errors={};exact={}
            if actual.keys()!=expected.keys():raise ValueError('output coverage mismatch')
            for key,a in actual.items():
                b=expected[key]
                exact[key]=(a.dtype==b.dtype and a.shape==b.shape and bool(torch.equal(
                    a.contiguous().reshape(-1).view(torch.uint8),
                    b.contiguous().reshape(-1).view(torch.uint8))))
                checks[key]=bool(torch.isfinite(a).all() and torch.isfinite(b).all() and
                                 torch.allclose(a,b,atol=1e-5,rtol=1e-5)) if a.is_floating_point() else exact[key]
                errors[key]=float((a.double()-b.double()).abs().max()) if a.numel() else 0.
            passed=different and prototype_unchanged and binding_exact and all(checks.values())
            rows.append({'case':name,'status':'PASS' if passed else 'FAIL',
                         'initial_model_identity_different':different,'prototype_unchanged':prototype_unchanged,
                         'loaded_binding_byte_exact':binding_exact,'all_output_checks':checks,
                         'output_byte_exact':exact,'max_abs_errors':errors})
    result={'scope':'UNTRAINED_ENGINEERING_FIXTURE_ONLY','status':'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',
            'rows':rows,'index_sha256':pin,'torch':torch.__version__,
            'independently_qualified':False,'trained':False,'optimizer_steps':0,'GPU_minutes':0,'TEST_opened':False,
            'historical_trained_checkpoint_replay':'NOT_RUN','physical_device':'NOT_RUN',
            'limits':'New untrained fixture files only. Exact loaded tensor/config/state binding; prediction tolerances1e-5 fixed. No authentic-data quality or independent authority/whole-IEEE/AD proof.'}
    output.mkdir(parents=True);(output/'REPLAY.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'stage':'replay','status':result['status'],'cases':2,'trained':False}))
    if result['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['capture','replay'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--input',type=Path);p.add_argument('--expected-index-sha')
    args=p.parse_args()
    if args.stage=='capture':capture(args.output)
    else:replay(args.input,args.expected_index_sha,args.output)
