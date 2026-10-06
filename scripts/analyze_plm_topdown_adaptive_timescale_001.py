#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
def boot(a,b,seed=2991,rounds=20000):
    d=a-b; rng=np.random.default_rng(seed); x=rng.choice(d,size=(rounds,len(d)),replace=True).mean(axis=1)
    return float(d.mean()),float(np.quantile(x,.025)),float(np.quantile(x,.975))
p=argparse.ArgumentParser(); p.add_argument("result"); a=p.parse_args()
x=json.loads(Path(a.result).read_text())
rows=x["per_seed"]["rows"]
v=lambda arm:np.asarray([r["success"] for r in rows if r["H"]==256 and r["arm"]==arm],dtype=float)
ad,fx,no=v("three_adaptive"),v("three_fixed"),v("no_state")
d,lo,hi=boot(ad,fx)
assert np.isclose(d,x["endpoints"]["primary"]["adaptive_minus_fixed_mean"])
assert np.allclose([lo,hi],x["endpoints"]["primary"]["seed_bootstrap_95ci"])
assert np.isclose(ad.mean()-no.mean(),x["endpoints"]["primary"]["adaptive_minus_no_state_mean"])
print(json.dumps({"adaptive_mean":float(ad.mean()),"fixed_mean":float(fx.mean()),"no_state_mean":float(no.mean()),"difference":d,"bootstrap_95ci":[lo,hi],"adaptive_alphas_by_seed":x["endpoints"]["learned_alphas_by_seed"]},indent=2,sort_keys=True))
