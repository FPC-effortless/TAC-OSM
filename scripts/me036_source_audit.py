"""Describe publisher timestamps/sentinels without changing benchmark values."""
import csv, json
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
import scripts.me036_research as base


def run():
    result=[]
    for name in base.DATASETS:
        _,meta=base.load_data(name)
        with open(base.DATA/(name+'.csv')) as f:
            reader=csv.reader(f);next(reader);rows=list(reader)
        dates=[datetime.fromisoformat(r[0]) for r in rows]
        cadence=Counter(int((b-a).total_seconds()) for a,b in zip(dates,dates[1:]))
        a=np.asarray([[float(v) for v in r[1:]] for r in rows],dtype=np.float64)
        tr,va,end=meta['split_edges']
        result.append({'dataset':name,'source_sha256':meta['source']['sha256'],
            'numeric_rows':len(a),'channels':a.shape[1],'timestamp_deltas_seconds':dict(cadence),
            'timestamps_strictly_increasing':all(b>a for a,b in zip(dates,dates[1:])),
            'minus9999_by_channel':dict(zip(meta['columns'],(a==-9999).sum(0).tolist())),
            'minus9999_by_split':{k:int((a[first:last]==-9999).sum()) for k,first,last in
                [('train',0,tr),('validation',tr,va),('test',va,end)]},
            'benchmark_policy':'Preserve publisher finite sentinel values as preregistered. '
                'Weather training normalization includes these codes; its test targets contain none in this source. Standardized benchmark MSE is not error in physical units.'})
    base.dump(base.REPORT/'source-audit.json',result)


if __name__=='__main__':run()
