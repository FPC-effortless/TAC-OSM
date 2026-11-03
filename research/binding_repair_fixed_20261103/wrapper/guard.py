"""Owned probe guard: exact Linux null-sink compatibility, otherwise fail closed."""
import os
import stat
import hashlib
import sys
import sysconfig
from pathlib import Path

class GuardDenied(RuntimeError):
    def __init__(self, code, event, path=None):
        super().__init__(code)
        self.safe_diagnostic={'code':code,'event':event,
            'path_class':'LINUX_NULL_DEVICE' if path == Path('/dev/null') else ('REDACTED_PATH' if path is not None else 'NO_PATH'),
            'path_sha256':hashlib.sha256(os.fsencode(str(path))).hexdigest() if path is not None else None}

def null_sink(path):
    if path != Path('/dev/null'):return False
    info=path.stat()
    return stat.S_ISCHR(info.st_mode) and os.major(info.st_rdev)==1 and os.minor(info.st_rdev)==3

def guard(out,allowed):
    roots={Path(sys.prefix).resolve(),Path(sysconfig.get_path('stdlib')).resolve(),Path(sysconfig.get_path('platstdlib')).resolve(),out.resolve()}
    allowed={str(Path(x).resolve()) for x in allowed}
    def owned(p):return p==out or out in p.parents
    def hook(event,args):
        if event in ('socket.connect','socket.bind','subprocess.Popen','os.system','os.fork','os.posix_spawn','os.symlink','os.link'):raise GuardDenied('SIDE_EFFECT_DENIED',event)
        if event in ('os.remove','os.rmdir','os.mkdir','os.chmod','os.chown','os.rename'):
            for path in args[:2] if event=='os.rename' else args[:1]:
                if not isinstance(path,(str,bytes,os.PathLike)) or not owned(Path(os.fsdecode(path)).resolve()):raise GuardDenied('MUTATION_DENIED',event)
        if event=='open':
            if isinstance(args[0],int):return
            p=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]
            writing=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
            if writing:
                if owned(p) or null_sink(p):return
                raise GuardDenied('WRITE_DENIED',event,p)
            if str(p) in allowed or any(p==r or r in p.parents for r in roots):return
            if str(p) in ('/dev/null','/dev/urandom','/proc/self/maps','/proc/self/statm','/proc/cpuinfo','/etc/localtime'):return
            raise GuardDenied('READ_DENIED',event,p)
    sys.addaudithook(hook)
