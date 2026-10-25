"""No-fit sharpness audit of saved original forecasts, not a new acceptance gate."""
from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/replay-bundles/ME101'

def report():
    old=np.load(ROOT/'research/replay-bundles/ME100/FORECASTS.npz');new=np.load(OUT/'FORECASTS.npz');rows=[]
    for cohort in ('gap','general'):
        if not np.array_equal(old[cohort+'_origins'],new[cohort+'_origins']):raise RuntimeError('cohort mismatch')
        keep=new[cohort+'_observed']
        for seed in range(101001,101006):
            parent=seed-1000
            for name,arr in [('ME100',old[f'{cohort}_{parent}_256']),('ME101',new[f'{cohort}_{seed}_256'])]:
                width=arr[:,:,2]-arr[:,:,0]
                rows.append(dict(cohort=cohort,seed=seed,model=name,mean_width_by_channel=[float(width[:,i][keep[:,i]].mean()) for i in range(6)],p95_width_by_channel=[float(np.quantile(width[:,i][keep[:,i]],.95)) for i in range(6)]))
    result=dict(status='POSTHOC_SHARPNESS_AUDIT_ONLY',no_new_fits=True,no_new_data_access=True,acceptance_gates_changed=False,rows=rows)
    dest=OUT/'UNCERTAINTY_REPORT.json'
    if dest.exists():
        expected=json.loads(dest.read_text())
        for a,b in zip(rows,expected['rows']):
            np.testing.assert_allclose(a['mean_width_by_channel'],b['mean_width_by_channel'],rtol=1e-5,atol=1e-6)
            np.testing.assert_allclose(a['p95_width_by_channel'],b['p95_width_by_channel'],rtol=1e-5,atol=1e-6)
    else:dest.write_text(json.dumps(result,indent=2)+'\n')
    print('ME101_ORIGINAL_SHARPNESS_AUDIT=PASS')
if __name__=='__main__':report()
