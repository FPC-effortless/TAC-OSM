"""Exact-pinned one-matrix runner. Only a fresh reviewed root grant permits work."""
import time
START=time.monotonic()
import hashlib,json,os,resource,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=None
DIAGNOSTIC=None
CAPS=None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(phase):
    assert time.process_time()<CAPS['child_stop_CPU'],'CHILD_CPU_EARLY_STOP'
    assert time.monotonic()-START<CAPS['total_wall']-20,'CHILD_WALL_EARLY_STOP'
    (OUT/'PROGRESS.json').write_text(json.dumps({'phase':phase,'CPU':time.process_time(),'wall':time.monotonic()-START})+'\n')
def main():
    global OUT,DIAGNOSTIC,CAPS
    assert __debug__ and len(sys.argv)==3
    grant_path=Path(sys.argv[1]).resolve();OUT=Path(sys.argv[2]).resolve();grant=json.loads(grant_path.read_bytes())
    assert grant['authorized'] is True and grant['protocol']=='LOCAL_BINDING_GRU512_V1' and grant['retry_authorized'] is False
    assert grant['stage'] in ('fixed','varying') and grant['regime'] in ('default','larger_epsilon')
    assert OUT==HERE.parent/'runs'/grant['stage']/grant['regime']/'output'/'child' and OUT.is_dir()
    assert grant['wrapper_manifest_sha256']==sha(HERE/'MANIFEST.json')
    wm=json.loads((HERE/'MANIFEST.json').read_bytes())
    for name,row in wm['files'].items():assert sha(HERE/name)==row['sha256'] and (HERE/name).stat().st_size==row['bytes']
    sp=HERE.parent/'SOURCE_MANIFEST.json';pp=HERE.parent/'PROTOCOL.json';cp=HERE.parent/'CODE_MANIFEST.json'
    assert grant['source_manifest_sha256']==sha(sp) and grant['protocol_sha256']==sha(pp) and grant['code_manifest_sha256']==sha(cp)
    sm=json.loads(sp.read_bytes());protocol=json.loads(pp.read_bytes());cm=json.loads(cp.read_bytes());CAPS=protocol['stages'][grant['stage']]['caps'];assert grant['caps']==CAPS
    assert resource.getrlimit(resource.RLIMIT_CPU)==(CAPS['child_CPU'],)*2
    assert resource.getrlimit(resource.RLIMIT_AS)==(8*1024**3,)*2 and resource.getrlimit(resource.RLIMIT_FSIZE)==(8*1024**2,)*2
    (OUT/'CONSUMED_ONCE').open('x').close();(OUT/'runtime-cache').mkdir();(OUT/'runtime-tmp').mkdir()
    os.environ.update(HOME=str(OUT/'runtime-cache'),XDG_CACHE_HOME=str(OUT/'runtime-cache'),TMPDIR=str(OUT/'runtime-tmp'))
    raw=(HERE/'guard.py').read_bytes();guard=types.ModuleType('learnability_guard');exec(compile(raw,str(HERE/'guard.py'),'exec',dont_inherit=True),guard.__dict__)
    allowed=[str(HERE/n) for n in wm['files']]+[str(HERE/'MANIFEST.json'),str(sp),str(pp),str(cp),str(grant_path)]
    allowed += [str(HERE.parent/'source'/p) for p in sm['files']]
    allowed += [str(HERE.parent/p) for p in cm['files']]
    allowed += [str(HERE.parent/'fixtures'/p) for p in protocol['fixture_files']]
    allowed += [str(HERE.parent/p) for p in protocol['admission']['files']]
    if grant['regime']=='larger_epsilon':
        paired=HERE.parent/'runs'/grant['stage']/'default'/'output'/'child'/'PAIRING_RECEIPT.json'
        accounting=HERE.parent/'runs'/grant['stage']/'default'/'outer-output'/'COMPLETE_ACCOUNTING.json'
        assert Path(grant['paired_default_receipt_path']).resolve()==paired
        assert Path(grant['paired_default_accounting_path']).resolve()==accounting
        allowed.extend([str(paired),str(accounting)])
    guard.guard(OUT,allowed)
    for name,row in cm['files'].items():assert sha(HERE.parent/name)==row['sha256'] and (HERE.parent/name).stat().st_size==row['bytes']
    check('source_pins_verified_before_torch')
    for name in ('admission','core','fixtures','source_probes','runner'):
        p=HERE.parent/'code'/(name+'.py');raw=p.read_bytes();module=types.ModuleType(name);module.__file__=str(p);sys.modules[name]=module;exec(compile(raw,str(p),'exec',dont_inherit=True),module.__dict__)
        if name=='runner':DIAGNOSTIC=module
    return DIAGNOSTIC.run(HERE.parent,protocol,grant,OUT,check)
if __name__=='__main__':
    code=2;result={}
    try:result=main();code=0
    except BaseException as e:
        frames=[];tb=e.__traceback__
        while tb is not None:frames.append({'file':Path(tb.tb_frame.f_code.co_filename).name,'line':tb.tb_lineno,'function':tb.tb_frame.f_code.co_name});tb=tb.tb_next
        result={'status':'HOLD_LEARNABILITY_EXCEPTION','error_type':type(e).__name__,'error_text':str(e)[:1000],'frames':frames[-20:]}
        if hasattr(e,'safe_diagnostic'):result['guard_denial']=e.safe_diagnostic
    finally:
        result.update(CPU_before_receipt=time.process_time(),wall_before_receipt=time.monotonic()-START,retry_authorized=False)
        if DIAGNOSTIC is not None:
            result['observed_counts']=dict(getattr(DIAGNOSTIC,'COUNTS',{}));result['observed_updates']=dict(getattr(DIAGNOSTIC,'UPDATES',{}));result['observed_attempted_updates']=dict(getattr(DIAGNOSTIC,'ATTEMPTED_UPDATES',{}))
            result['optimizer_step_outcome_uncertain']=any(n>result['observed_updates'].get(k,0) for k,n in result['observed_attempted_updates'].items())
        if OUT is not None and OUT.is_dir():
            with (OUT/'RESULT.json').open('x') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
        else:sys.stderr.write(json.dumps(result)+'\n')
    raise SystemExit(code)
