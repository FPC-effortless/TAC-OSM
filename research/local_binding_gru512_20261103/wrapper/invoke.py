"""One-shot supervisor: source-pinned subprocess, safe wait4, fixed caps."""
import time
START=time.monotonic()
import ctypes,hashlib,json,os,resource,signal,subprocess,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
PYTHON=HERE.parent/'runtime/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    with p.open('x') as f:f.write(json.dumps(v,indent=2,allow_nan=False)+'\n')
def limit(signum,frame):raise TimeoutError('SUPERVISOR_CPU_LIMIT')
def main():
    assert __debug__ and len(sys.argv)==2
    grant_path=Path(sys.argv[1]).resolve();grant=json.loads(grant_path.read_bytes())
    assert grant['authorized'] is True and grant['protocol']=='LOCAL_BINDING_GRU512_V1'
    assert grant['retry_authorized'] is False
    assert grant['wrapper_manifest_sha256']==sha(HERE/'MANIFEST.json')
    manifest=json.loads((HERE/'MANIFEST.json').read_bytes())
    for name,r in manifest['files'].items():
        p=HERE/name;assert sha(p)==r['sha256'] and p.stat().st_size==r['bytes']
    assert grant['source_manifest_sha256']==sha(HERE.parent/'SOURCE_MANIFEST.json')
    assert grant['protocol_sha256']==sha(HERE.parent/'PROTOCOL.json')
    protocol=json.loads((HERE.parent/'PROTOCOL.json').read_bytes())
    assert grant['stage'] in ('fixed','varying') and grant['regime'] in ('default','larger_epsilon')
    caps=protocol['stages'][grant['stage']]['caps'];assert grant['caps']==caps
    assert grant['code_manifest_sha256']==sha(HERE.parent/'CODE_MANIFEST.json')
    assert grant['bounded_launcher_review_sha256']==grant['review_sha256']
    review=Path(grant['review_path']);assert sha(review)==grant['review_sha256']
    runroot=HERE.parent/'runs'/grant['stage']/grant['regime']
    out=runroot/'output';out.mkdir();(out/'CONSUMED_ONCE').open('x').close();childout=out/'child';childout.mkdir()
    resource.setrlimit(resource.RLIMIT_CPU,(10,caps['total_CPU']));resource.setrlimit(resource.RLIMIT_CORE,(0,0));signal.signal(signal.SIGXCPU,limit)
    wd=types.ModuleType('binding_watchdog');exec(compile((HERE/'watchdog.py').read_bytes(),str(HERE/'watchdog.py'),'exec'),wd.__dict__)
    process=None;monitor=None;usage=None;reaped=False;report={'status':'HOLD_NOT_STARTED'};exitcode=2
    try:
        env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','LC_ALL':'C.UTF-8','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','TORCHDYNAMO_DISABLE':'1'}
        expected_parent=os.getpid();libc=ctypes.CDLL(None,use_errno=True)
        def child_limits():
            if libc.prctl(1,signal.SIGKILL,0,0,0)!=0:os._exit(120)
            if os.getppid()!=expected_parent:os._exit(121)
            # Parent hard cap remains at the registered aggregate cap, above the child cap.
            resource.setrlimit(resource.RLIMIT_CPU,(caps['child_CPU'],caps['child_CPU']));resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
            resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**2,)*2);resource.setrlimit(resource.RLIMIT_CORE,(0,0));signal.alarm(caps['total_wall']-15)
        def inspect():
            files=[p for p in childout.rglob('*') if p.is_file()];assert len(files)<=64
            assert all(not p.is_symlink() and p.stat().st_size<=8*1024**2 for p in files)
            assert sum(p.stat().st_size for p in files)<=32*1024**2
        assert time.process_time()<8 and time.monotonic()-START<5
        with (out/'child.stdout').open('xb') as so,(out/'child.stderr').open('xb') as se:
            process=subprocess.Popen([str(PYTHON),'-I','-B',str(HERE/'child.py'),str(grant_path),str(childout)],stdin=subprocess.DEVNULL,stdout=so,stderr=se,env=env,preexec_fn=child_limits)
            write(out/'OWNED_CHILD_PID.json',{'pid':process.pid,'pgrp':os.getpgrp(),'parent':expected_parent,'parent_death_signal':'SIGKILL','race_checked':True})
            monitor=wd.monitor(process,start_wall=START,parent_initial_CPU=0.,child_cap=caps['child_CPU'],child_stop=caps['child_stop_CPU'],parent_cap=10.,parent_stop=8.,wall=caps['monitor_wall'],cleanup=5.,poll=1.,inventory_interval=10.,inspect_output=inspect,process_group=False)
        reaped=monitor['child_reaped'];write(out/'SUPERVISOR.json',monitor)
        result=json.loads((childout/'RESULT.json').read_bytes()) if (childout/'RESULT.json').is_file() else None
        ok=monitor['status']=='PASS_BOUNDED_AUDIT_PROCESS' and result is not None and result['status']=='COMPLETE_FIXED_BUDGET_MATRIX'
        report={'status':'COMPLETE_BOUNDED_DIAGNOSTIC' if ok else 'HOLD_DIAGNOSTIC','semantic_status':result.get('status') if result else None,'child_complete_wait4_CPU':monitor['child_CPU_seconds'],'child_reaped':reaped,'child_returncode':process.returncode,'result_sha256':sha(childout/'RESULT.json') if result else None,'retry_authorized':False}
        exitcode=0 if ok else 2
    except BaseException as e:report.update(status='HOLD_SUPERVISOR_EXCEPTION',error_type=type(e).__name__,error_text=str(e)[:500])
    finally:
        if process is not None and process.returncode is None:
            try:os.kill(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                try:
                    pid,status,usage=os.wait4(process.pid,os.WNOHANG)
                    if pid==process.pid:process.returncode=os.waitstatus_to_exitcode(status);reaped=True;break
                except ChildProcessError:break
                time.sleep(.01)
            report['fallback_reaped']=reaped
            if not reaped:report['quarantine']='UNCONFIRMED_STOP_NO_RETRY'
            if usage is not None:report['fallback_complete_child_CPU']=usage.ru_utime+usage.ru_stime
        report.update(parent_CPU_at_receipt=time.process_time(),wall_at_receipt=time.monotonic()-START,retry_authorized=False)
        if report['parent_CPU_at_receipt']>10 or report['wall_at_receipt']>caps['total_wall']-5:report['status']='HOLD_FINAL_RESOURCE_CAP';exitcode=2
        write(out/'PARENT_RECEIPT.json',report)
    return exitcode
if __name__=='__main__':raise SystemExit(main())
