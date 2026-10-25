"""Pinned source loader. FIT and REPORT only; later numerical rows never parsed."""
import csv,io,zipfile,datetime,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'research/source-bundles/ME100/PRSA2017_original.zip'
SHA='d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8'
SITES=('Aotizhongxin','Changping','Dingling','Dongsi','Guanyuan','Gucheng','Huairou','Nongzhanguan','Shunyi','Tiantan','Wanliu','Wanshouxigong')
NUMERIC=('PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM')
WIND=('N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW')
EPOCH=datetime.datetime(2013,3,1)
FIT_STOP=int((datetime.datetime(2015,1,1)-EPOCH).total_seconds()/3600)
REPORT_STOP=int((datetime.datetime(2016,1,1)-EPOCH).total_seconds()/3600)

def load(phase='fit'):
    if phase not in ('fit','report'):raise ValueError('protected numerical cohort unavailable')
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()!=SHA:raise RuntimeError('original source digest mismatch')
    stop=2015 if phase=='fit' else 2016
    z=zipfile.ZipFile(ARCHIVE);raw=[];hours=[];cal=[]
    for site in SITES:
        matches=[p for p in z.namelist() if p.endswith(f'PRSA_Data_{site}_20130301-20170228.csv')]
        if len(matches)!=1:raise RuntimeError('source entity file missing/duplicated')
        vals=[];ts=[];clocks=[]
        reader=csv.DictReader(io.TextIOWrapper(z.open(matches[0]),encoding='utf-8-sig'))
        for row in reader:
            if int(row['year'])>=stop:break # NO later numeric conversion.
            if row['station']!=site:raise RuntimeError('entity mismatch')
            dt=datetime.datetime(*(int(row[k]) for k in ['year','month','day','hour']))
            t=int((dt-EPOCH).total_seconds()/3600)
            v=[float(row[k]) if row[k]!='NA' else np.nan for k in NUMERIC]
            w=row['wd']
            if w=='NA':v.extend((np.nan,np.nan))
            else:
                if w not in WIND:raise RuntimeError('unknown recorded wind tag')
                a=2*np.pi*WIND.index(w)/16;v.extend((np.sin(a),np.cos(a)))
            if any(np.isinf(v)):raise RuntimeError('infinite source values')
            if any(x<0 for x in v[:6] if np.isfinite(x)):raise RuntimeError('negative pollutant source value')
            vals.append(v);ts.append(t)
            a=2*np.pi*np.array([dt.hour/24,dt.weekday()/7,(dt.timetuple().tm_yday-1)/366])
            clocks.append(np.concatenate((np.sin(a),np.cos(a))))
        if ts!=list(range(len(ts))):raise RuntimeError('source hourly ordering/gaps changed')
        raw.append(np.asarray(vals,dtype=np.float32));hours.append(ts);cal.append(clocks)
    data=np.stack(raw);hour=np.asarray(hours,dtype=np.int64);clock=np.asarray(cal,dtype=np.float32)
    obs=np.isfinite(data);transformed=data.copy();transformed[...,:6]=np.log1p(transformed[...,:6])
    fit=transformed[:,:FIT_STOP]
    mean=np.nanmean(fit.astype(np.float64),axis=(0,1));std=np.nanstd(fit.astype(np.float64),axis=(0,1)).clip(.0001)
    values=np.nan_to_num((transformed-mean)/std,nan=0).clip(-8,8).astype(np.float32)
    # Clocks/ages may reveal staleness, NEVER carry an old numeric value into inputs.
    last_t=np.full((12,13),-10**9,dtype=np.int64);last_v=np.zeros((12,13),dtype=np.float32)
    age=np.empty_like(values);last=np.empty_like(values)
    for t in range(data.shape[1]):
        last_t=np.where(obs[:,t],t,last_t);last_v=np.where(obs[:,t],values[:,t],last_v)
        dt=t-last_t
        age[:,t]=np.log1p(np.minimum(dt,256))/np.log(257)
        last[:,t]=np.where(dt<=255,last_v,0)
    # Recover un-clipped integer age for primary cohort selection only.
    pm_age=np.empty(data.shape[:2],dtype=np.int64);prev=np.full(12,-10**9,dtype=np.int64)
    for t in range(data.shape[1]):
        prev=np.where(obs[:,t,0],t,prev);pm_age[:,t]=t-prev
    start=255 if phase=='fit' else FIT_STOP;end=FIT_STOP-6 if phase=='fit' else REPORT_STOP-6
    general=[];gap=[]
    for sid in range(12):
        ix=np.arange(start,end)
        eligible=obs[sid,ix+6,0]
        ix=ix[eligible]
        if phase=='report':
            # Fixed daily anchor plus EVERY natural-gap query; no value-based selection.
            day=ix[ix%24==12]
        else:day=ix
        gp=ix[(pm_age[sid,ix]>=8)&(pm_age[sid,ix]<=255)]
        general.extend((sid,int(t)) for t in day);gap.extend((sid,int(t)) for t in gp)
    return dict(raw=data,values=values,observed=obs,ages=age,clock=clock,hours=hour,last=last,pm_age=pm_age,
                mean=mean.astype(np.float32),std=std.astype(np.float32),general=general,gap=gap,phase=phase)

def batch(d,origins,length):
    sid=np.asarray([p[0] for p in origins],dtype=np.int64);at=np.asarray([p[1] for p in origins],dtype=np.int64)
    index=at[:,None]-np.arange(length-1,-1,-1)[None]
    take=lambda name:d[name][sid[:,None],index]
    y=d['raw'][sid,at+6,:6];target=np.nan_to_num(y,nan=0).astype(np.float32)
    return (take('values'),take('observed'),take('ages'),take('clock'),sid,d['hours'][sid[:,None],index],target,np.isfinite(y))

def ridge_features(d,origins):
    sid=np.asarray([p[0] for p in origins]);t=np.asarray([p[1] for p in origins])
    return np.concatenate((np.ones((len(t),1)),d['values'][sid,t],d['observed'][sid,t].astype(float),d['ages'][sid,t],
                           d['clock'][sid,t],np.eye(12)[sid],d['last'][sid,t]),-1)
