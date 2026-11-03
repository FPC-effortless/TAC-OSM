"""Complete wait4 accounting around the once-only supervisor."""
import time
START=time.monotonic()
import ctypes,hashlib,json,os,resource,signal,subprocess,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert __debug__ and len(sys.argv)==4
    approval_sha=sys.argv[-2];grant_sha=sys.argv[-1];grant_path=Path(sys.argv[1]).resolve();grant=json.loads(grant_path.read_bytes());assert sha(grant_path)==grant_sha and grant['grant_path']==str(grant_path);assert sha(Path(grant['approval_path']))==approval_sha
    assert grant['authorized'] is True and grant['wrapper_manifest_sha256']==sha(HERE/'MANIFEST.json')
    assert grant['protocol']=='LOCAL_BINDING_LEARNABILITY_V2' and grant['protocol_sha256']==sha(HERE.parent/'PROTOCOL.json')
    protocol=json.loads((HERE.parent/'PROTOCOL.json').read_bytes())
    assert grant['stage'] in ('fixed','varying') and grant['regime'] in ('default','larger_epsilon')
    caps=protocol['stages'][grant['stage']]['caps'];assert grant['caps']==caps
    runroot=HERE.parent/'runs'/grant['stage']/grant['regime'];assert Path(grant['output_directory']).resolve()==runroot.resolve()
    manifest=json.loads((HERE/'MANIFEST.json').read_bytes())
    for name,r in manifest['files'].items():assert sha(HERE/name)==r['sha256']
    assert grant['retry_authorized'] is False
    runroot.parent.mkdir(parents=True,exist_ok=True);runroot.mkdir()
    out=runroot/'outer-output';out.mkdir();(out/'CONSUMED_ONCE').open('x').close()
    # Keep the inherited hard cap high enough for registered child/parent caps.
    resource.setrlimit(resource.RLIMIT_CPU,(5,caps['total_CPU']));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    def limit(s,f):raise TimeoutError('OUTER_CPU_LIMIT')
    signal.signal(signal.SIGXCPU,limit)
    libc=ctypes.CDLL(None,use_errno=True)
    assert libc.prctl(36,1,0,0,0)==0,'SUBREAPER_SETUP_FAILED'
    observed=ctypes.c_int()
    assert libc.prctl(37,ctypes.byref(observed),0,0,0)==0 and observed.value==1
    wd=types.ModuleType('outer_watchdog');exec(compile((HERE/'watchdog.py').read_bytes(),str(HERE/'watchdog.py'),'exec'),wd.__dict__)
    def inspect():
        files=[p for p in (runroot/'output').rglob('*') if p.is_file()]
        assert len(files)<=80 and all(not p.is_symlink() and p.stat().st_size<=8*1024**2 for p in files)
        assert sum(p.stat().st_size for p in files)<=36*1024**2
    expected_outer=os.getpid()
    def supervisor_guard():
        if libc.prctl(1,signal.SIGKILL,0,0,0)!=0:os._exit(122)
        if os.getppid()!=expected_outer:os._exit(123)
    with (out/'parent.stdout').open('xb') as so,(out/'parent.stderr').open('xb') as se:
        p=subprocess.Popen([sys.executable,'-I','-B',str(HERE/'invoke.py'),str(grant_path),approval_sha,grant_sha],stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True,preexec_fn=supervisor_guard,env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','LC_ALL':'C.UTF-8','PYTHONDONTWRITEBYTECODE':'1'})
        publication_error=None
        try:(out/'OWNED_PARENT_PID.json').write_text(json.dumps({'pid':p.pid,'pgrp':p.pid})+'\n')
        except BaseException as error:
            publication_error=type(error).__name__
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        finally:
            monitor=wd.monitor(p,start_wall=START,parent_initial_CPU=0.,child_cap=caps['total_CPU']-5,child_stop=caps['total_CPU']-10,parent_cap=5.,parent_stop=4.,wall=caps['total_wall']-2,cleanup=2.,poll=1.,inventory_interval=10.,inspect_output=inspect)
        if publication_error is not None:
            monitor['status']='HOLD_PID_PUBLICATION_FAILURE';monitor['publication_error']=publication_error
    # If the supervisor failed before reaping, its child is now our child in
    # the exact group we created. wait4(-owned_pgid) never reaps unrelated groups.
    adopted=[];group_closed=False;deadline=min(START+caps['total_wall'],time.monotonic()+2)
    while time.monotonic()<deadline:
        try:pid,status,usage=os.wait4(-p.pid,os.WNOHANG)
        except ChildProcessError:group_closed=True;break
        if pid>0:
            if pid==p.pid:
                assert not monitor['child_reaped']
                p.returncode=os.waitstatus_to_exitcode(status)
                monitor.update(child_reaped=True,child_returncode=p.returncode,child_CPU_seconds=usage.ru_utime+usage.ru_stime,late_supervisor_reap=True,status='HOLD_LATE_SUPERVISOR_REAP')
            else:adopted.append({'pid':pid,'exitcode':os.waitstatus_to_exitcode(status),'complete_CPU':usage.ru_utime+usage.ru_stime})
        else:
            # A live child in this exact owned group proves the group still
            # exists and cannot yet have had its ID recycled for unrelated work.
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            time.sleep(.01)
    monitor['adopted_children']=adopted;monitor['owned_group_children_reaped']=group_closed
    (out/'SUPERVISOR.json').write_text(json.dumps(monitor,indent=2)+'\n')
    receipt=runroot/'output/PARENT_RECEIPT.json';inner=json.loads(receipt.read_bytes()) if receipt.is_file() else {}
    policy_preflight_error=None;policy_module=None;registration=None
    try:
        code_path=HERE.parent/'CODE_MANIFEST.json';assert sha(code_path)==grant['code_manifest_sha256'];code_manifest=json.loads(code_path.read_bytes())
        path=HERE.parent/'code/admission.py';row=code_manifest['files']['code/admission.py'];assert sha(path)==row['sha256'] and path.stat().st_size==row['bytes']
        admission=types.ModuleType('outer_admission');admission.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec',dont_inherit=True),admission.__dict__)
        policy_module,registration,evidence=admission.inputs(HERE.parent,protocol,grant,approval_sha,grant_sha)
        policy_module.validate_synthetic_diagnostic(registration,'preflight',**evidence)
    except BaseException as error:policy_preflight_error={'type':type(error).__name__,'text':str(error)[:1000]}
    cpu=monitor['child_CPU_seconds'];adopted_cpu=sum(r['complete_CPU'] for r in adopted);own=time.process_time();total=None if cpu is None else cpu+adopted_cpu+own;wall=time.monotonic()-START
    ok=policy_preflight_error is None and monitor['status']=='PASS_BOUNDED_AUDIT_PROCESS' and group_closed and not adopted and inner.get('status')=='COMPLETE_BOUNDED_DIAGNOSTIC' and total<=caps['total_CPU'] and wall<=caps['total_wall']
    report={'status':'COMPLETE_WITH_CAPTURED_RESOURCE_ACCEPTANCE' if ok else 'HOLD_OUTER_RESOURCE_OR_PROCESS','full_wait4_parent_and_reaped_descendants_CPU':cpu,'adopted_children':adopted,'adopted_complete_CPU':adopted_cpu,'owned_group_children_reaped':group_closed,'outer_CPU_before_final_receipt':own,'captured_total_CPU':total,'wall_before_final_receipt':wall,'child_complete_wait4_CPU':inner.get('child_complete_wait4_CPU'),'parent_complete_CPU':None if cpu is None or inner.get('child_complete_wait4_CPU') is None else cpu-inner['child_complete_wait4_CPU'],'semantic_status':inner.get('semantic_status'),'wrapper_manifest_sha256':sha(HERE/'MANIFEST.json'),'grant_sha256':sha(grant_path),'outer_final_write_exit_tail_included':False,'retry_authorized':False}
    report['policy_preflight_error']=policy_preflight_error
    report['uncaptured_tail']='Final scalar policy validation, receipt write and process exit only; model work and policy evidence loading are captured.'
    report.update(study_id=grant['study_id'],stage=grant['stage'],regime=grant['regime'],child_final_admission_status=inner.get('child_final_admission_status'))
    if ok:
        try:
            report['policy_resource_gate']=policy_module.validate_resource_receipt(registration,report,trusted_grant_sha256=grant_sha)
        except BaseException as error:
            ok=False;report.update(status='HOLD_FINAL_POLICY_RESOURCE_GATE',policy_error_type=type(error).__name__,policy_error=str(error)[:1000])
    (out/'COMPLETE_ACCOUNTING.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))
    return 0 if ok else 2
if __name__=='__main__':raise SystemExit(main())
