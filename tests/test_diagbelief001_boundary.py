"""Actual CSV/transform implementation, synthetic files; no capability evidence."""
import csv
import datetime as dt
import hashlib
import io
import zipfile
import numpy as np
from scripts import me100_sensor_data as source
from scripts.me102_gap_data import augmentation, floor_rates


def archive(tmp_path, monkeypatch, poison=False):
    path=tmp_path/('poison.zip' if poison else 'clean.zip')
    epoch=dt.datetime(2014,12,31)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for sid,site in enumerate(source.SITES):
            s=io.StringIO();w=csv.DictWriter(s,fieldnames=['year','month','day','hour','station',*source.NUMERIC,'wd']);w.writeheader()
            for t in range(48):
                now=epoch+dt.timedelta(hours=t)
                row=dict(year=now.year,month=now.month,day=now.day,hour=now.hour,station=site,wd='N')
                row.update({name:float(1+sid+c+t+(1000 if poison and t>=36 else 0)) for c,name in enumerate(source.NUMERIC)})
                w.writerow(row)
            # Actual loader must stop before attempting any numeric conversion.
            row=dict(year=2016,month=1,day=1,hour=0,station=site,wd='NO_PARSE')
            row.update({name:'DENIED_NUMERIC_SENTINEL' for name in source.NUMERIC});w.writerow(row)
            z.writestr(f'PRSA_Data_{site}_20130301-20170228.csv',s.getvalue())
    monkeypatch.setattr(source,'ARCHIVE',path);monkeypatch.setattr(source,'SHA',hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(source,'EPOCH',epoch);monkeypatch.setattr(source,'FIT_STOP',24);monkeypatch.setattr(source,'REPORT_STOP',48)
    return source.load('fit'),source.load('report')


def test_source_report_poison_cannot_refit_or_enter_earlier_inputs(tmp_path,monkeypatch):
    fit,report=archive(tmp_path,monkeypatch)
    pf,pr=archive(tmp_path,monkeypatch,True)
    for key in ('raw','values','observed','ages','mean','std'):
        np.testing.assert_array_equal(fit[key],pf[key])
    for key in ('mean','std'):np.testing.assert_array_equal(report[key],pr[key])
    a=source.batch(report,[(0,30)],8);b=source.batch(pr,[(0,30)],8)
    for i in range(6):np.testing.assert_array_equal(a[i],b[i])
    assert not np.array_equal(a[6],b[6])
    assert fit['raw'].shape==(12,24,13) and report['raw'].shape==(12,48,13)


def test_direct_plus_six_target_poison_preserves_augmented_inputs(tmp_path,monkeypatch):
    _,report=archive(tmp_path,monkeypatch)
    # Explicit fixture FIT tagging permits 48-hour artificial data only.
    d=dict(report,phase='fit');alternate=dict(d,raw=d['raw'].copy())
    alternate['raw'][0,36,:6]+=100
    a=augmentation(d,[(0,30)],8,np.random.default_rng(7))
    b=augmentation(alternate,[(0,30)],8,np.random.default_rng(7))
    for i in range(6):np.testing.assert_array_equal(a[i],b[i])
    assert not np.array_equal(a[6],b[6])


def test_phase_label_does_not_authenticate_raw_array_provenance(tmp_path,monkeypatch):
    fit,report=archive(tmp_path,monkeypatch)
    # Documentary regression: helper alone accepts a mislabeled longer array.
    # It must never be claimed to verify origin provenance.
    false_fit=dict(report,phase='fit')
    assert sum(floor_rates(fit)['counts'])==0
    assert sum(floor_rates(false_fit)['counts'])>0
