"""Linux owned-child CPU watchdog, independent of kernel RLIMIT enforcement.

Not an execution authority: caller pins a fixed reviewed saved-auditor command.
Polled CPU ceiling is not a hard quota; all overshoots cause HOLD at final reap.
"""
import os
from pathlib import Path
import resource
import signal
import subprocess
import time


def proc_cpu(pid, ticks):
    assert type(pid) is int and pid > 0 and type(ticks) is int and ticks > 0
    # Only this directly spawned child's public process accounting, not arbitrary paths.
    with open('/proc/'+str(pid)+'/stat', 'rb') as stream:
        raw=stream.read(8193)
    assert 0 < len(raw) <= 8192
    prefix, fields=raw.rsplit(b') ',1)
    assert int(prefix.split(b' ',1)[0]) == pid
    values=fields.split()
    assert len(values)>=20
    user,system,start=map(int,(values[11],values[12],values[19]))
    assert min(user,system)>=0 and start>0
    return (user+system)/ticks,start


def _signal(pid, process_group):
    try:(os.killpg if process_group else os.kill)(pid,signal.SIGKILL)
    except ProcessLookupError:pass


def monitor(process, *, start_wall, parent_initial_CPU, child_cap=30., child_stop=27.,
            parent_cap=4., parent_stop=3.5, wall=120., cleanup=5., poll=.05, inspect_output=None, inventory_interval=None, process_group=True):
    """Own wait4; matching PID alone establishes reap. Never calls wait()/poll()."""
    ticks=None
    if inventory_interval is None: inventory_interval=poll
    next_inventory=start_wall
    phase_CPU={key:0. for key in ('wait_reap','output_scan','proc_sample','kill_signal')}
    phase_calls={key:0 for key in phase_CPU}
    def measured(key,call,*args):
        began=time.process_time()
        try:return call(*args)
        finally:
            phase_CPU[key]+=time.process_time()-began
            phase_calls[key]+=1
    reaped=False;usage=None;status=None;reason=None;identity=None;peak=0.;samples=0;kill_sent=False
    def take_wait():
        nonlocal reaped,usage,status
        pid,found_status,found_usage=measured('wait_reap',os.wait4,process.pid,os.WNOHANG)
        assert pid in (0,process.pid)
        if pid==process.pid:
            reaped=True;status=found_status;usage=found_usage
            process.returncode=os.waitstatus_to_exitcode(status)
    try:
        assert 0 < child_stop < child_cap and 0 < parent_stop < parent_cap
        assert 0 < poll <= 1. and poll <= inventory_interval <= 10. and 0 < cleanup <= 5 and 0 < wall <= 1500
        ticks=os.sysconf('SC_CLK_TCK');assert type(ticks) is int and ticks>0
        while True:
            take_wait()
            if reaped:break
            if time.monotonic()-start_wall >= wall:
                reason='WALL_STOP';break
            if time.process_time()-parent_initial_CPU >= parent_stop:
                reason='PARENT_CPU_STOP';break
            if inspect_output is not None and time.monotonic() >= next_inventory:
                measured('output_scan',inspect_output)
                next_inventory=time.monotonic()+inventory_interval
            try:
                cpu,start=measured('proc_sample',proc_cpu,process.pid,ticks)
            except FileNotFoundError:
                take_wait()
                if reaped:break
                raise
            if identity is None:identity=start
            assert start==identity,'OWNED_CHILD_IDENTITY_CHANGED'
            peak=max(peak,cpu);samples+=1
            if cpu>=child_stop:
                reason='CHILD_CPU_STOP';break
            time.sleep(min(poll,max(0.,wall-(time.monotonic()-start_wall))))
    except BaseException as error:
        reason='MONITOR_'+type(error).__name__
    finally:
        if not reaped:
            try:measured('kill_signal',_signal,process.pid,process_group);kill_sent=True
            except BaseException as error:reason=reason or 'KILL_'+type(error).__name__
            deadline=time.monotonic()+cleanup
            while not reaped and time.monotonic()<deadline:
                try:take_wait()
                except BaseException as error:
                    reason=reason or 'CLEANUP_'+type(error).__name__;break
                if not reaped:
                    try:time.sleep(.01)
                    except BaseException as error:
                        reason=reason or 'CLEANUP_SLEEP_'+type(error).__name__;break
    if reaped and inspect_output is not None:
        try:
            measured('output_scan',inspect_output)
        except BaseException as error:
            reason = reason or 'TERMINAL_OUTPUT_' + type(error).__name__
    child_CPU=usage.ru_utime+usage.ru_stime if reaped else None
    parent_CPU=time.process_time()-parent_initial_CPU
    if not reaped:reason=reason or 'UNCONFIRMED_STOP_QUARANTINE'
    elif child_CPU>child_cap:reason=reason or 'CHILD_CPU_CAP_EXCEEDED'
    if parent_CPU>parent_cap:reason=reason or 'PARENT_CPU_CAP_EXCEEDED'
    if process.returncode!=0:reason=reason or 'CHILD_NONZERO'
    elapsed=time.monotonic()-start_wall
    if elapsed>=wall:reason=reason or 'TERMINAL_WALL_CAP_EXCEEDED'
    return {'status':'PASS_BOUNDED_AUDIT_PROCESS' if reason is None else 'HOLD_AUDIT_RESOURCE_OR_PROCESS',
            'error_code':reason,'child_reaped':reaped,'child_returncode':process.returncode,
            'child_CPU_seconds':child_CPU,'parent_CPU_seconds':parent_CPU,
            'wall_seconds':elapsed,'child_CPU_observed_peak':peak,
            'parent_phase_CPU':phase_CPU,'parent_phase_calls':phase_calls,
            'parent_CPU_outside_measured_phase_calls':parent_CPU-sum(phase_CPU.values()),
            'parent_phase_scope':'Exclusive aggregate call intervals, including failedcalls; residual includes callerpre-monitor imports/admission, scheduling/receipt/control overhead; no per-poll logs',
            'child_CPU_samples':samples,'proc_starttime_ticks':identity,'clock_ticks_per_second':ticks,
            'kill_requested':kill_sent,'limits':{'child_CPU_cap':child_cap,'child_CPU_polled_stop':child_stop,
                'parent_CPU_cap':parent_cap,'parent_CPU_polled_stop':parent_stop,'wall':wall,'cleanup':cleanup,'poll':poll,'inventory_interval':inventory_interval},
            'CPU_enforcement':'polled_proc_ticks_plus_final_wait4; RLIMIT_not_assumed_effective',
            'overshoot_policy':'Any measured CPU cap breach fails acceptance. CPU/wall detection may lag by poll plus calls/scheduling; output inventory may lag by inventory_interval plus calls/scheduling. No hard cumulative-write quota. Unchanged guard, FSIZE/AS and cooperative child checks remain separate controls.',
            'retry_authorized':False,'unreaped_policy':'QUARANTINE_NO_RETRY_UNTIL_CONFIRMED_STOP',
            'descendants':'Reviewed child must not spawn; '+('owned process-group' if process_group else 'exact owned PID')+' kill requested. Outer adopted-child accounting is separate.'}
