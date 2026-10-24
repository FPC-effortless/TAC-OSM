"""Known operation type and current observed capture-clock phases only.

No episode IDs, relative curator episode age, total duration, future times/actions,
case ending or outcome labels. Periods are fixed before fitting, not selected.
"""
from pathlib import Path
import numpy as np
import torch
import pyarrow.parquet as pq
from scripts.me076_train_system1_real import digest

PERIODS=(1.,4.,16.)


def attach_context(cache,fit,report):
    tasks=sorted({task for task,_ in fit})
    if len(tasks)!=5 or {task for task,_ in report}!=set(tasks):raise RuntimeError('Exactly original five known operation types required')
    mapping={name:i for i,name in enumerate(tasks)};receipts={}
    for role,rows in [('FIT',fit),('REPORT',report)]:
        records={}
        for (task,ep),row in rows.items():
            path=cache/task/f'data/chunk-{ep//1000:03d}'/f'episode_{ep:06d}.parquet'
            clock=np.asarray(pq.read_table(path,columns=['capture_time'])['capture_time'].to_numpy(),dtype=np.float64)
            if len(clock)!=len(row['state']) or not np.isfinite(clock).all():raise RuntimeError('Observed capture clock differs')
            values=[]
            for period in PERIODS:
                phase=np.remainder(clock,period)*(2*np.pi/period);values.extend([np.sin(phase),np.cos(phase)])
            context=np.stack(values,-1).astype(np.float32)
            row['clock_context']=torch.from_numpy(context);row['task_id']=mapping[task]
            records[f'{task}:{ep}']={'frames':len(clock),'task_id':mapping[task],'clock_context_sha256':__import__('hashlib').sha256(context.tobytes()).hexdigest()}
        receipts[role]={'cases':len(records),'context_records_sha256':digest(records)}
    return {'operation_type_mapping':mapping,'clock_period_seconds':list(PERIODS),'features':'sin/cos of current capture_time modulo1/4/16seconds, float64 preprocessing thenfloat32; no absoluteepoch scalar or episodebirth/end data','roles':receipts,'availability':'recorded observation capture clock and externally known requested operation; reference action cutoff remains t-1','comparison_scope':'Additional known metadata access compared with old models, plus different parameter/exposure budgets; no equal-information architecture winner claim'}
