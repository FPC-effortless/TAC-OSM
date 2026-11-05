"""No-fit oracle-gate upper-bound counterexample; NOT trained performance."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tac_osm.csl125_r2_math_reference import selective_serial


def run():
    t=513
    z=np.zeros((1,t,1,1,2),dtype=np.float64)
    z[0,0,0,0]=[1,1]
    z[0,1:,0,0]=[-1,1]
    carry=np.zeros((1,1,1,2),dtype=np.float64)
    rate=np.array([[1/64]],dtype=np.float64)
    full=np.ones((1,t,1))
    protected=np.ones((1,t,1));protected[:,1:,:]=0
    row={}
    for name,gate in (('always_write',full),('oracle_ignore_distractors',protected)):
        out,final=selective_serial(z,carry,rate,gate)
        n,d=final[0,0,0]
        row[name]={'final_numerator':float(n),'final_mass':float(d),
                   'normalized_value':float(n/(d+1e-8)),
                   'first_write_gate':float(gate[0,0,0]),
                   'distractor_gate':float(gate[0,1,0])}
    return {'stage':'NO_FIT_ORACLE_GATE_UPPER_BOUND','used_neural_model':False,
      'fitted_gating_policy':False,'optimizer_steps':0,'original_value':1,
      'distractors':512,'distractor_value':-1,'timescale':64,'results':row,
      'scope':'A policy that *already knows* which writes to ignore can protect state. No learned routing or ability to identify irrelevant tokens is demonstrated.'}

if __name__=='__main__':
    obj=run();p=ROOT/'research/mathematical-engineering/csl125_r2/ADMISSION_COUNTEREXAMPLE_NO_FIT.json'
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps(obj['results'],indent=2))
