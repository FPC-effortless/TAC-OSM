"""Descriptive alternating-run timing uncertainty; no predictive scoring or fit."""
import json
from pathlib import Path
import numpy as np
from scripts.me076_train_system1_real import sha,write
root=Path('research/mathematical-engineering/me094');profiles={};sources={}
for name in ['COMPILED_GRAPH_AUDIT','PHYSICAL_BOUNDARY_GRAPH_AUDIT']:
    path=root/(name+'.json');sources[str(path)]=sha(path)
    for label,p in json.loads(path.read_text())['profiles'].items():
        a=np.asarray(p['raw_timings_ms']['eager']);b=np.asarray(p['raw_timings_ms']['compiled'])
        if len(a)!=31 or len(b)!=31 or (a<=0).any() or (b<=0).any():raise RuntimeError('Expected original31positive paired alternating timing rounds')
        draws=np.random.default_rng(94094).integers(len(a),size=(5000,len(a)))
        profiles[label]={'rounds':31,'eager_median_ms':float(np.median(a)),'compiled_median_ms':float(np.median(b)),'median_speedup':float(np.median(a)/np.median(b)),'paired_round_bootstrap95interval':np.quantile(np.median(a[draws],1)/np.median(b[draws],1),[.025,.975]).tolist()}
point=root/'POINT_QUERY_GRAPH_AUDIT.json'
if point.exists():
    sources[str(point)]=sha(point)
    for label,p in json.loads(point.read_text())['profiles'].items():
        a=np.asarray(p['raw_timings_ms']['complete']);b=np.asarray(p['raw_timings_ms']['point_only']);draws=np.random.default_rng(94094).integers(len(a),size=(5000,len(a)))
        profiles['point_query_'+label]={'rounds':31,'complete_median_ms':float(np.median(a)),'point_only_median_ms':float(np.median(b)),'median_speedup_for_requested_subset':float(np.median(a)/np.median(b)),'paired_round_bootstrap95interval':np.quantile(np.median(a[draws],1)/np.median(b[draws],1),[.025,.975]).tolist(),'different_output_scope_not_architecture_comparison':True}
write(root/'TIMING_INTERVALS.json',{'no_fit':True,'source_sha256':sources,'bootstrap_draws':5000,'rng_seed':94094,'profiles':profiles,'scope':'Descriptive conditional31alternating within-process rounds on shared/unpinned CPU during training; autocorrelation, thermal/load changes, hardware/site variability are not modeled. No architecture-rival or hardware-general latency claim.'})
