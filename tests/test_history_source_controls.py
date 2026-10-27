"""Constructed source and original-code fixtures; never capability evidence."""
import csv
import io
import hashlib
from pathlib import Path
from types import SimpleNamespace
import datetime as dt
import os
import subprocess
import sys
import zipfile
import pytest
from tac_osm.history_source_controls import NUMERIC,read_fit_members,make_control,declared_control_reader


def fixture_source(tmp_path,quoted=False):
    p=tmp_path/'original.zip'
    cols=['No','year','month','day','hour',*NUMERIC,'wd','station']
    with zipfile.ZipFile(p,'w') as z:
        for site in range(12):
            lines=[','.join(cols)]
            for hour in (0,1):lines.append(','.join(map(str,[hour+1,2014,5,1,hour,*([5]*len(NUMERIC)),'N',f'S{site}'])))
            lines.append(','.join(map(str,[3,2015,1,1,0,*(['DO_NOT_PARSE_PROTECTED']*len(NUMERIC)),'NA',f'S{site}'])))
            if quoted:
                out=io.StringIO(newline='');writer=csv.writer(out,quoting=csv.QUOTE_ALL,lineterminator='\r\n')
                for i,line in enumerate(lines):
                    fields=line.split(',')
                    if i==2:fields[5]='NA'
                    writer.writerow(fields)
                data=out.getvalue()
            else:data='\n'.join(lines)+'\n'
            z.writestr(f'PRSA_Data_S{site}_20130301-20170228.csv',data)
    return p,hashlib.sha256(p.read_bytes()).hexdigest()


def test_genuine_hash_checked_before_reading_fixture(tmp_path):
    p,h=fixture_source(tmp_path)
    assert len(read_fit_members(p,h))==12
    with pytest.raises(ValueError,match='publisher'):read_fit_members(p,'0'*64)


def test_fit_only_derivative_never_copies_protected_numeric_payload(tmp_path):
    p,h=fixture_source(tmp_path);members=read_fit_members(p,h)
    out=tmp_path/'control.zip';r=make_control(members,out)
    assert not r['contains_protected_numeric_rows'] and r['sha256']!=h
    with zipfile.ZipFile(out) as z:
        for n in z.namelist():
            b=z.read(n)
            assert b'DO_NOT_PARSE_PROTECTED' not in b
            assert b'PROTECTED_NUMERICS_NOT_READ' in b
            assert b'2015,1,1,0' in b


@pytest.mark.parametrize('mode',['values','missingness'])
def test_future_only_mutations_keep_fixture_prefix_bytes(tmp_path,mode):
    p,h=fixture_source(tmp_path);members=read_fit_members(p,h)
    out=tmp_path/'poison.zip';r=make_control(members,out,cutoff=dt.datetime(2014,5,1),mode=mode)
    assert r['changed_FIT_rows']==12 and r['future_before_sha256']!=r['future_after_sha256']
    with zipfile.ZipFile(out) as z:
        for name,(_,_,rows,_) in members.items():
            lines=z.read(name).splitlines(keepends=True)
            assert lines[1]==rows[0][1]
            assert lines[2]!=rows[1][1]


def test_explicit_derived_hash_restores_parent_even_on_exception(tmp_path):
    p,h=fixture_source(tmp_path);out=tmp_path/'fit.zip';r=make_control(read_fit_members(p,h),out)
    source=SimpleNamespace(SOURCE_SHA=h)
    with pytest.raises(RuntimeError):
        with declared_control_reader(source,out,r['sha256'],original_sha=h):
            assert source.SOURCE_SHA==r['sha256']
            raise RuntimeError('control reader failed')
    assert source.SOURCE_SHA==h
    with pytest.raises(ValueError,match='derived'):
        with declared_control_reader(source,out,'0'*64,original_sha=h):pass
    assert source.SOURCE_SHA==h


def test_inert_or_unregistered_source_poison_fails(tmp_path):
    p,h=fixture_source(tmp_path);members=read_fit_members(p,h)
    with pytest.raises(ValueError,match='inert'):make_control(members,tmp_path/'x.zip',cutoff=dt.datetime(2014,6,1),mode='values')
    with pytest.raises(ValueError,match='unregistered'):make_control(members,tmp_path/'x.zip',mode='rewrite_all')


def test_actual_original_probe_uses_real_writer_and_native_read_graph():
    root=os.environ.get('TAC_OSM_CONTENT_QUERY_SOURCE')
    if not root:pytest.skip('dedicated CPU workflow supplies pinned original code')
    code=r'''
import sys
from pathlib import Path
sys.path[:0]=[str(Path.cwd()),str(Path.cwd()/'src')]
import torch
from tac_osm.history_source_probe import HistorySourceWindow,adapter
from scripts.audit_history_source_reload import probe,first_divergence
from scripts.original_history_trace import verify_implementation,load_original_modules
root=verify_implementation(Path(sys.argv[1]),'fd201207cb76b30470f64e1fc1baecf9de44560d')
source,original=load_original_modules(root)
torch.set_num_threads(1);torch.manual_seed(71)
s=HistorySourceWindow(torch.randn(2,88,13),torch.ones(2,88,13,dtype=torch.bool),torch.zeros(2,88,13),torch.zeros(2,88,13),torch.tensor([0,1]),torch.arange(88).expand(2,-1),source.calendar(list(range(23,87,8))*2).reshape(2,8,6),source.calendar([93,93])[:,None])
for arm in ('assoc_historic_query','fast_slow_historic_query'):
 m=original.create_model(26016,arm).eval()
 with torch.no_grad():
  a,b=probe(m,s),probe(m,s)
 assert first_divergence(a,b) is None
 assert len(a['timeline'])==88 and a['geometry'][0]['query_hour']==87
 assert a['update_reconstruction_max_abs_error']<=2e-6 and a['read_reconstruction_max_abs_error']<=2e-6
 assert set(a['final_state'])=={'fast','slow','associative','entities','active'}
 assert all(p.grad is None for p in m.parameters())
 try:adapter().validate(m,{'raw_target':torch.zeros(1)})
 except ValueError:pass
 else:raise AssertionError('untyped target dictionary accepted')
 print('PASS_ORIGINAL_SOURCE_GRAPH_CONSTRUCTED_FIXTURE',arm)
'''
    r=subprocess.run([sys.executable,'-c',code,root],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
    assert r.stdout.count('PASS_ORIGINAL_SOURCE_GRAPH_CONSTRUCTED_FIXTURE')==2


def test_positive_source_edit_uses_exact_owner_not_substring(tmp_path):
    p,h=fixture_source(tmp_path);members=read_fit_members(p,h)
    out=tmp_path/'positive.zip';r=make_control(members,out,mode='positive',positive=('S1',dt.datetime(2014,5,1)))
    assert r['changed_FIT_rows']==1
    with zipfile.ZipFile(out) as z:
        assert z.read('PRSA_Data_S10_20130301-20170228.csv').splitlines()[1]==members['PRSA_Data_S10_20130301-20170228.csv'][2][0][1].strip()


@pytest.mark.parametrize('mode',['values','missingness'])
def test_publisher_quoted_csv_header_dates_wind_and_missing_cells(tmp_path,mode):
    p,h=fixture_source(tmp_path,quoted=True);members=read_fit_members(p,h)
    out=tmp_path/'quoted-control.zip'
    make_control(members,out,cutoff=dt.datetime(2014,5,1),mode=mode)
    with zipfile.ZipFile(out) as z:
        n='PRSA_Data_S0_20130301-20170228.csv'
        lines=z.read(n).splitlines(keepends=True)
        assert lines[0]==members[n][0] and lines[1]==members[n][2][0][1]
        row=next(csv.reader([lines[2].decode()]))
        assert row[5]==('NA' if mode=='values' else '1000000')
        assert row[-2]==('E' if mode=='values' else 'NA')
