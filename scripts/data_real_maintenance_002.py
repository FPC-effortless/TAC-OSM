"""New MetroPT2 stream preprocessing and one-way sealed-A access guard."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
FEATURES=['TP2','TP3','H1','DV_pressure','Reservoirs','Oil_temperature','Flowmeter','Motor_current']
SOURCE_MD5='056e04fd1874eaef4697d6b6657502c3'
EPOCH=datetime(1970,1,1)


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def seconds(value):return int((datetime.fromisoformat(value)-EPOCH).total_seconds())


def split_windows(times,values):
    result={name:{'x':[],'y':[],'time':[]} for name in ('train','validation','sealed-a','sealed-b')}
    for j in range(24,len(times)):
        if times[j]%3600 or not np.all(np.diff(times[j-24:j+1])==600):continue
        t=times[j];name='train' if t<seconds('2022-06-01') else 'validation' if t<seconds('2022-06-15') else 'sealed-a' if t<seconds('2022-07-01') else 'sealed-b'
        result[name]['x'].append(values[j-24:j]);result[name]['y'].append(values[j]);result[name]['time'].append(t)
    return {name:{k:torch.tensor(np.asarray(v),dtype=torch.int64 if k=='time' else torch.float32) for k,v in rows.items()} for name,rows in result.items()}


def prepare(source,destination):
    start=time.process_time();source=Path(source);target=Path(destination);target.mkdir()
    md5=hashlib.md5();sha=hashlib.sha256()
    with source.open('rb') as stream:
        while chunk:=stream.read(1024*1024):md5.update(chunk);sha.update(chunk)
    assert source.stat().st_size==1209663443 and md5.hexdigest()==SOURCE_MD5,'Incomplete or wrong publisher object'
    bins={};count=0;out_of_order=0;last=None
    with source.open(newline='',encoding='utf-8-sig') as stream:
        reader=csv.reader(stream);header=next(reader);assert header[0]=='timestamp';indices=[header.index(k) for k in FEATURES]
        for row in reader:
            timestamp=seconds(row[0]);end=(timestamp//600+1)*600;values=np.array([float(row[i]) for i in indices],np.float64);assert np.isfinite(values).all(),'Non-finite analog row; registration requires fail, not silent imputation'
            out_of_order+=int(last is not None and timestamp<last);last=timestamp;count+=1
            if end not in bins:bins[end]=[values,1]
            else:bins[end][0]+=values;bins[end][1]+=1
    times=np.array(sorted(k for k,v in bins.items() if v[1]>=300),np.int64);values=np.array([bins[t][0]/bins[t][1] for t in times],np.float32);data=split_windows(times,values)
    assert all(len(v['x'])>0 for v in data.values()),'Incomplete date coverage; do not move boundaries'
    for name,rows in data.items():torch.save(rows,target/(name+'.pt'))
    metadata={'source':'https://zenodo.org/records/7766691','doi':'10.5281/zenodo.7766691','creators':['Bruno Veloso','João Gama','Rita Ribeiro','Pedro Pereira'],'license':'CC BY4.0','raw_sha256':sha.hexdigest(),'raw_md5':md5.hexdigest(),'raw_bytes':source.stat().st_size,'source_rows':count,'out_of_order_rows':out_of_order,'accepted_bins':len(times),'rejected_sparse_bins':len(bins)-len(times),'features':FEATURES,'split_counts':{k:len(v['x']) for k,v in data.items()},'data_hashes':{k:digest(target/(k+'.pt')) for k in data},'preprocessing_cpu_seconds':time.process_time()-start,'exposure':'Values streamed only for fixed preprocessing; no held-out metrics. B is not loaded by fitter/auditor A.'};write(target/'manifest.json',metadata);write(target/'seal-state.json',{'phase':'PREPARED','A_metrics_opened':False,'B_metrics_opened':False});return metadata


def open_a(data_root,run_root):
    d=Path(data_root);r=Path(run_root);s=json.loads((r/'summary.json').read_text());choice=json.loads((r/'selected-arm.json').read_text())
    assert s['status']=='FITS_COMPLETE' and s['fits']==80 and len(choice['checkpoint_hashes'])==80,'Finish/freeze all seeds and selection before A'
    for name,expected in choice['checkpoint_hashes'].items():assert digest(r/name)==expected,'Checkpoint changed after selection'
    manifest=json.loads((d/'manifest.json').read_text());assert digest(d/'sealed-a.pt')==manifest['data_hashes']['sealed-a']
    write(d/'seal-state.json',{'phase':'A_EVALUATION','A_metrics_opened':True,'B_metrics_opened':False,'selection_sha256':digest(r/'selected-arm.json'),'data_manifest_sha256':digest(d/'manifest.json')});return torch.load(d/'sealed-a.pt',weights_only=True)
