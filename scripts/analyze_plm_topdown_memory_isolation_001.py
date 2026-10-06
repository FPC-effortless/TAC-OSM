#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
def boot(a,b,seed=3991,rounds=20000):
    d=np.asarray(a)-np.asarray(b)
    rng=np.random.default_rng(seed)
    z=rng.choice(d,size=(rounds,len(d)),replace=True).mean(axis=1)
    return float(d.mean()),float(np.quantile(z,.025)),float(np.quantile(z,.975))
p=argparse.ArgumentParser(); p.add_argument("result"); args=p.parse_args()
x=json.loads(Path(args.result).read_text(encoding="utf-8")); rows=x["per_seed"]["rows"]
v=lambda arm:np.asarray([r["success"] for r in rows if r["H"]==256 and r["arm"]==arm],dtype=float)
m,s,n=v("mtsk"),v("single_timescale"),v("no_state")
d,lo,hi=boot(m,s)
assert np.isclose(d,x["endpoints"]["primary"]["mtsk_minus_single_mean"])
assert np.allclose([lo,hi],x["endpoints"]["primary"]["seed_bootstrap_95ci"])
print(json.dumps({"mtsk_mean":float(m.mean()),"single_mean":float(s.mean()),"no_state_mean":float(n.mean()),"difference":d,"bootstrap_95ci":[lo,hi]},indent=2,sort_keys=True))
