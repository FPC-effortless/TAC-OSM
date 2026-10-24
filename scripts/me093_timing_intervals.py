"""Descriptive alternating-run timing uncertainty; no predictive scoring or fit."""
import json
from pathlib import Path
import numpy as np
from scripts.me076_train_system1_real import sha,write
root=Path('research/mathematical-engineering/me093');profiles={};sources={}
for name in ['COMPILED_GRAPH_AUDIT','PHYSICAL_BOUNDARY_GRAPH_AUDIT']:
    path=root/(name+'.json');sources[str(path)]=sha(path)
    for label,p in json.loads(path.read_text())['profiles'].items():
        a=np.asarray(p['raw_timings_ms']['eager']);b=np.asarray(p['raw_timings_ms']['compiled'])
        if len(a)!=31 or len(b)!=31 or (a<=0).any() or (b<=0).any():raise RuntimeError('Expected original31positive paired alternating timing rounds')
        draws=np.random.default_rng(93093).integers(len(a),size=(5000,len(a)))
        profiles[label]={'rounds':31,'eager_median_ms':float(np.median(a)),'compiled_median_ms':float(np.median(b)),'median_speedup':float(np.median(a)/np.median(b)),'paired_round_bootstrap95interval':np.quantile(np.median(a[draws],1)/np.median(b[draws],1),[.025,.975]).tolist()}
write(root/'TIMING_INTERVALS.json',{'no_fit':True,'source_sha256':sources,'bootstrap_draws':5000,'rng_seed':93093,'profiles':profiles,'scope':'Descriptive conditional31alternating within-process rounds on shared/unpinned CPU during training; autocorrelation, thermal/load changes, hardware/site variability are not modeled. No architecture-rival or hardware-general latency claim.'})
