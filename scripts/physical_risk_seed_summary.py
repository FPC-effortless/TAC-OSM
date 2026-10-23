"""No-fit per-original-model risk scores; avoids calling a five-fit mean a100k model."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.me078_report_capability import ranking

def main(study):
    out=Path('research/replay-bundles')/study
    models=[]
    for path in sorted(out.glob('*/seed-*/ORIGINAL_PREDICTIONS.json')):
        rows=json.loads(path.read_text());h1=[r for r in rows if r['horizon']==1]
        models.append({'architecture':path.parts[-3],'seed':int(path.parent.name.split('-')[1]),'risk':ranking(h1)})
    old=[]
    for path in sorted(Path('research/replay-bundles/ME077').glob('seed-*/ORIGINAL_PREDICTIONS.json')):
        rows=json.loads(path.read_text());old.append(ranking([{'moving':r['motion_rich'],'motion_probability':r['motion_probability']} for r in rows]))
    summary={arm:{k:float(np.mean([m['risk'][k] for m in models if m['architecture']==arm])) for k in ('auc','ap','top10pct_motion_capture')} for arm in ('native','gru','transformer')}
    result={'study':study,'interpretation':'Means of five separate100k-model metrics, NOT an ensemble deployment. Earlier CAPABILITY_RESULTS horizon merged event_ranking is separately an averaged-probability diagnostic.',
      'by_architecture_mean_individual_seed':summary,'original_models':models,
      'frozen_ME077_mean_individual_seed':{k:float(np.mean([r[k] for r in old])) for k in ('auc','ap','top10pct_motion_capture')},
      'exposure':'EXPOSED_DEV','original_real_queries':960,'positive_outcomes':84,'episode_clusters':40,'no_new_fits':True}
    dest=Path('research/mathematical-engineering')/study.lower()/'RISK_INDIVIDUAL_SEED_RESULTS.json';dest.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',choices=['ME078','ME079'],required=True);a=p.parse_args();main(a.study)
