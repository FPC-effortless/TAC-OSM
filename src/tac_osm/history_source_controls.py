"""Explicitly derived FIT-only source controls; never authentic outcomes.

Original archive SHA stays the provenance parent. Derived archives have their
OWN hashes. The unchanged original reader runs with that declared hash only in
an isolated research process; its default original-byte guard is tested first.
No protected numeric rows are parsed or copied into control archives.
"""
from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
import hashlib
import io
from pathlib import Path
import zipfile


NUMERIC=("PM2.5","PM10","SO2","NO2","CO","O3","TEMP","PRES","DEWP","RAIN","WSPM")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_fit_members(path, expected_sha):
    if sha(path)!=expected_sha:raise ValueError('original publisher archive hash differs')
    members={}
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            if not name.endswith('.csv'):continue
            with z.open(name) as f:
                header=f.readline();cols=header.decode('utf-8-sig').strip().split(',')
                if not set(('year','month','day','hour','station','wd',*NUMERIC))<=set(cols):raise ValueError('unknown source header')
                if cols[:5]!=['No','year','month','day','hour']:raise ValueError('source date prefix changed')
                rows=[]
                for line in f:
                    # Date prefix only: never parse protected numeric fields.
                    prefix=line.split(b',',5)
                    year=int(prefix[1])
                    if year>=2015:break
                    when=dt.datetime(*(int(x) for x in prefix[1:5]))
                    rows.append((when,line))
                if not rows:raise ValueError('empty FIT source')
                # Stops original reader before numeric parsing; never an observed row.
                stub=[b'PROTECTED_NUMERICS_NOT_READ']*len(cols)
                for key,value in (('No',b'0'),('year',b'2015'),('month',b'1'),('day',b'1'),('hour',b'0')):
                    stub[cols.index(key)]=value
                members[name]=(header,cols,rows,b','.join(stub)+b'\n')
    if len(members)!=12:raise ValueError('exactly twelve original station members required')
    return members


def make_control(members,path,*,cutoff=None,mode='identity',positive=None):
    if mode not in ('identity','values','missingness','positive'):raise ValueError('unregistered source mutation')
    if mode in ('values','missingness') and cutoff is None:raise ValueError('future boundary required')
    if mode=='positive' and positive is None:raise ValueError('source-visible positive-control identity required')
    changes=0;prefix=hashlib.sha256();future_before=hashlib.sha256();future_after=hashlib.sha256()
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,(header,cols,rows,stub) in sorted(members.items()):
            data=[header]
            for when,line in rows:
                future=cutoff is not None and when>cutoff
                visible_positive=mode=='positive' and name.endswith(f'PRSA_Data_{positive[0]}_20130301-20170228.csv') and when==positive[1]
                rewritten=line
                if future or visible_positive:
                    # Only authorized pre2015 rows reach this branch.
                    fields=line.rstrip(b'\r\n').split(b',')
                    if visible_positive:
                        fields[cols.index('PM2.5')]=b'50000'
                    elif mode in ('values','missingness'):
                        for k in NUMERIC:
                            j=cols.index(k)
                            if mode=='values':fields[j]=b'1000000' if fields[j]!=b'NA' else b'NA'
                            else:fields[j]=b'NA' if fields[j]!=b'NA' else b'1000000'
                        j=cols.index('wd')
                        fields[j]=(b'E' if fields[j]!=b'NA' else b'NA') if mode=='values' else (b'NA' if fields[j]!=b'NA' else b'E')
                    rewritten=b','.join(fields)+b'\n'
                if future:
                    future_before.update(name.encode()+line);future_after.update(name.encode()+rewritten)
                else:
                    if rewritten!=line and not visible_positive:raise ValueError('forbidden source-prefix edit')
                    prefix.update(name.encode()+line)
                changes+=rewritten!=line
                data.append(rewritten)
            data.append(stub)
            info=zipfile.ZipInfo(name,date_time=(2000,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,b''.join(data))
    path=Path(path);path.write_bytes(out.getvalue())
    if mode!='identity' and not changes:raise ValueError('inert source poison control')
    return dict(mode=mode,sha256=sha(path),changed_FIT_rows=changes,
        original_prefix_line_sha256=prefix.hexdigest(),future_before_sha256=future_before.hexdigest(),
        future_after_sha256=future_after.hexdigest(),synthetic_negative_control_not_observed_outcomes=mode!='identity',
        contains_protected_numeric_rows=False)


@contextmanager
def declared_control_reader(source, control_path, control_sha, *, original_sha):
    """Explicit diagnostic configuration, never a forged original provenance hash.

    Original guard is restored even on exception. Caller must separately verify
    the publisher parent and the prefix-preserving mutation before entering.
    """
    if source.SOURCE_SHA!=original_sha:raise ValueError('original reader hash configuration already altered')
    if sha(control_path)!=control_sha:raise ValueError('declared derived archive hash differs')
    source.SOURCE_SHA=control_sha
    try:yield
    finally:source.SOURCE_SHA=original_sha
