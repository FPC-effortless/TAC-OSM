import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from scripts.me095_context import attach_context


def test_context_is_current_clock_periodic_and_cannot_read_future_labels(tmp_path):
    fit={};report={}
    for i in range(5):
        task=f'operation-{i}';folder=tmp_path/task/'data/chunk-000';folder.mkdir(parents=True)
        for ep,role in [(1,fit),(2,report)]:
            path=folder/f'episode_{ep:06d}.parquet'
            pq.write_table(pa.table({'capture_time':[1000.25,1000.5,1000.75,1001.],'future_outcome':[0,1,0,1]}),path)
            role[(task,ep)]={'state':torch.zeros(4,14)}
    original=attach_context(tmp_path,fit,report);before=report[('operation-0',2)]['clock_context'].clone()
    path=tmp_path/'operation-0/data/chunk-000/episode_000002.parquet'
    pq.write_table(pa.table({'capture_time':[1000.25,1000.5,1002.75,1003.],'future_outcome':[100,100,100,100]}),path)
    attach_context(tmp_path,fit,report)
    torch.testing.assert_close(before[:2],report[('operation-0',2)]['clock_context'][:2],rtol=0,atol=0)
    pq.write_table(pa.table({'capture_time':[1016.25,1016.5,1016.75,1017.],'future_outcome':[900,900,900,900]}),path)
    attach_context(tmp_path,fit,report)
    torch.testing.assert_close(before,report[('operation-0',2)]['clock_context'],rtol=0,atol=0)
    assert original['clock_period_seconds']==[1.,4.,16.]
    assert fit[('operation-0',1)]['task_id']==0 and fit[('operation-4',1)]['task_id']==4
