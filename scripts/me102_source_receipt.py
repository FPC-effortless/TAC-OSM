"""Check saved panel eligibility/labels against original publisher rows; no fits."""
from pathlib import Path
import json,hashlib,numpy as np,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts import me100_sensor_data as data
OUT=ROOT/'research/replay-bundles/ME102'

def audit():
    d=data.load('report');a=np.load(OUT/'FORECASTS.npz');rows={}
    for cohort in ('gap','general'):
        orig=np.asarray(d[cohort]);np.testing.assert_array_equal(a[cohort+'_origins'],orig)
        truth=d['raw'][orig[:,0],orig[:,1]+6,:6];mask=np.isfinite(truth)
        np.testing.assert_array_equal(a[cohort+'_observed'],mask);np.testing.assert_array_equal(a[cohort+'_truth'],np.nan_to_num(truth,nan=0))
        keys=[k for k in a.files if k.startswith(cohort+'_') and any(k.startswith(cohort+'_'+r+'_') for r in ('light_floor','strong_reconstruction','physical_score'))]
        if len(keys)!=15:raise RuntimeError('missingoriginalforecasts')
        for key in keys:
            q=a[key]
            if not np.isfinite(q).all() or (q<0).any() or (np.diff(q,axis=-1)<0).any():raise RuntimeError('invalidoriginalpredictivedistributionquantiles')
        rows[cohort]=dict(queries=len(orig),global_calendar_weeks=len(np.unique(orig[:,1]//168)),source_stations=len(np.unique(orig[:,0])),truth_sha256=hashlib.sha256(np.nan_to_num(truth,nan=0).tobytes()).hexdigest(),all15_forecasts_finite_nonnegative_ordered=True)
    result=dict(status='PASS',source_sha256=data.SHA,original_raw_labels_recomputed=True,cohorts=rows,protected_2016_2017_scored=False,no_new_fits=True)
    p=OUT/'RAW_SOURCE_RECEIPT.json'
    if p.exists():
        if json.loads(p.read_text())!=result:raise RuntimeError('rawsourcereplaychanged')
    else:p.write_text(json.dumps(result,indent=2)+'\n')
    print('ME102_ORIGINAL_RAW_SOURCE_AUDIT=PASS')
if __name__=='__main__':audit()
