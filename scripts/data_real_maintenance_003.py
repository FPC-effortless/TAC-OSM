"""Frozen measured-data cohort with no test-value read before an exposure marker."""
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import torch

ROOMS=[f'T_Room_{i}_Avg' for i in (*range(102,107),*range(202,207))]
EXOGENOUS=['T_out_Avg','Net_Total_Power']
SOURCE_SHA='1e6cd3df991abf865f7567e6cc0ac246968299eb02dfe9464d887727f930dde0'
CONTEXT=8; HORIZONS=(2,4)


def write(path, value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def digest(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def timestamp(text):
    text=text.strip()
    try:return datetime.fromisoformat(text)
    except ValueError:
        for pattern in ('%m/%d/%Y %H:%M','%m/%d/%Y %H:%M:%S'):
            try:return datetime.strptime(text,pattern)
            except ValueError:pass
    raise ValueError('Unsupported public timestamp format; do not silently reorder times')


def inventory(source):
    assert digest(source)==SOURCE_SHA,'Wrong or incomplete measured ORNL source archive'
    with ZipFile(source) as archive:
        names=sorted(n for n in archive.namelist() if not n.startswith('Weather_'))
    assert len(names)==63 and all('/' not in n and n.endswith('.csv') for n in names)
    splits={'train':[],'validation':[],'test':[]}
    for name in names:
        date=datetime.strptime(name[:6],'%y%m%d')
        which='train' if date<datetime(2018,8,1) else 'validation' if date<datetime(2019,1,1) else 'test'
        splits[which].append(name)
    assert len(splits['test'])==11
    return names,splits


def windows(source,names,all_names):
    results={key:[] for key in ('x','y','minutes','day','origin_seconds')};counts=[]
    with ZipFile(source) as archive:
        for name in names:
            rows=list(csv.DictReader(io.StringIO(archive.read(name).decode('utf-8-sig'))))
            assert len(rows)==96,name
            assert {'TIMESTAMP',*ROOMS,*EXOGENOUS}<=set(rows[0]),name
            times=[timestamp(row['TIMESTAMP']) for row in rows]
            def number(value):
                try:return float(value)
                except (TypeError,ValueError):return float('nan')
            values=np.array([[number(row[k]) for k in ROOMS+EXOGENOUS] for row in rows],dtype=np.float64)
            count=0; invalid=0;gaps=0;boundary=0
            expected=datetime.strptime(name[:6],'%y%m%d').date()
            for end in range(CONTEXT-1,len(rows)-max(HORIZONS)):
                start=end-CONTEXT+1;future=[end+h for h in HORIZONS]
                if any((b-a).total_seconds()!=900 for a,b in zip(times[start:future[-1]],times[start+1:future[-1]+1],strict=True)):
                    gaps+=1;continue
                if any(t.date()!=expected for t in times[start:future[-1]+1]):boundary+=1;continue
                x=values[start:end+1];y=values[future,:len(ROOMS)]
                if not np.isfinite(x).all() or not np.isfinite(y).all():invalid+=1;continue
                results['x'].append(x);results['y'].append(y)
                results['minutes'].append([t.hour*60+t.minute for t in times[start:end+1]])
                results['day'].append(all_names.index(name))
                results['origin_seconds'].append(int((times[end]-datetime(1970,1,1)).total_seconds()))
                count+=1
            counts.append(dict(file=name,valid_windows=count,nonfinite_windows=invalid,
                               noncontiguous_windows=gaps,crossday_windows=boundary))
            assert count>=32,('DATA_QUALITY_REJECTED',counts[-1])
    return {key:torch.tensor(np.array(values),dtype=torch.int64 if key in ('day','origin_seconds') else torch.float64)
            for key,values in results.items()},counts


def prepare(source,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    names,splits=inventory(source)
    assert len(splits['train'])>=12 and len(splits['validation'])>=4
    reports={}
    for name in ('train','validation'):
        data,counts=windows(source,splits[name],names)
        torch.save(data,destination/(name+'.pt'));reports[name]=counts
    metadata=dict(source_sha256=SOURCE_SHA,source_url='https://data.openei.org/submissions/909',
        doi='10.25984/1862682',license='CC-BY-4.0',
        attribution='Kim, Janghyun; Leach, Matt; Im, Piljae. Curated Test Fault Data Set, NREL2019.',
        target_channels=ROOMS,past_exogenous=EXOGENOUS,files=splits,
        context=CONTEXT,horizon_steps=HORIZONS,minutes_per_step=15,
        train_validation_coverage=reports,phase='TRAIN_VALIDATION_PREPARATION',test_sensor_values_parsed=False,
        data_sha256={name:digest(destination/(name+'.pt')) for name in ('train','validation')},
        scope='63CSV experiment-days in one physical building, not63independent machines or fault configurations. Weather files and fault names are never numerical model features. Test dates/files were inventoried, not their sensor values.')
    write(destination/'source.json',metadata)
    return metadata


def open_test(source,root):
    root=Path(root);summary=json.loads((root/'summary.json').read_text())
    selection=json.loads((root/'selection.json').read_text())
    assert summary['status']=='VALIDATION_ELIGIBLE_FITS_COMPLETE' and summary['fits']==80
    assert len(selection['checkpoint_hashes'])==80
    for name,sha in selection['checkpoint_hashes'].items():assert digest(root/name)==sha,name
    assert not (root/'TEST-EXPOSURE.json').exists()
    with (root/'TEST-EXPOSURE.json').open('x') as stream:
        json.dump(dict(selection_sha256=digest(root/'selection.json'),source_sha256=SOURCE_SHA,
                       reason='All80fits and registered validation entry gate complete; opening2019sensor values once'),stream)
    summary.update(status='EVALUATING',test_opened=True);write(root/'summary.json',summary)
    names,splits=inventory(source);assert len(splits['test'])>=6
    data,counts=windows(source,splits['test'],names)
    torch.save(data,root/'_datasets/test.pt');write(root/'test-coverage.json',counts)
    metadata=json.loads((root/'_datasets/source.json').read_text())
    metadata.update(phase='TEST_GENERATED',test_sensor_values_parsed=True)
    metadata['data_sha256']['test']=digest(root/'_datasets/test.pt')
    write(root/'_datasets/source.json',metadata)
    return data
