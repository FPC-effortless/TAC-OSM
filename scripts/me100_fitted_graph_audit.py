"""No-fit audit: full serial/parallel refinement on original fitted FIT queries."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts import me100_entity_sensor as study
from scripts import me100_sensor_data as data
from scripts.me100_graph_reference import serial
from tac_osm.variants.entity_sensor_world import EntitySensorWorld

def audit():
    torch.set_num_threads(1);study.budget_audit();d=data.load('fit');rows=[]
    for seed in study.SEEDS:
        cp=torch.load(study.OUT/f'seed-{seed}'/'final.pt',map_location='cpu',weights_only=False)
        for dtype,tol in ((torch.float32,2e-5),(torch.float64,1e-9)):
            m=EntitySensorWorld().to(dtype);m.load_state_dict(cp['model']);x=study.tensors(d,d['gap'][:2],64)
            x=[v.to(dtype) if v.is_floating_point() else v for v in x];p,s=m(*x[:6]);r=serial(m,*x[:5]);maximum=0.
            for got,want in zip((p.log_location,p.log_scale,s.source_sum,s.source_mass,s.evidence_sum,s.memory),r):
                torch.testing.assert_close(got,want,rtol=tol,atol=tol*.1);maximum=max(maximum,float((got-want).abs().max().detach()))
            gp=torch.autograd.grad(p.log_location.square().sum()+p.log_scale.square().sum(),tuple(m.parameters()))
            gr=torch.autograd.grad(r[0].square().sum()+r[1].square().sum(),tuple(m.parameters()))
            for a,b in zip(gp,gr):torch.testing.assert_close(a,b,rtol=tol*10,atol=tol)
            _,part=m(*(v[:,:32] if v.ndim>=2 and v.shape[1]==64 else v for v in x[:6]))
            end,_=m(*(v[:,32:] if v.ndim>=2 and v.shape[1]==64 else v for v in x[:6]),state=part)
            torch.testing.assert_close(p.log_location,end.log_location,rtol=tol,atol=tol*.1)
            rows.append(dict(seed=seed,dtype=str(dtype),max_output_state_abs_error=maximum,all_parameter_gradients_compared=len(gp),full_chunk_agreement=True))
    result=dict(status='PASS',real_FIT_origins=d['gap'][:2],rows=rows,no_retraining=True,report_values_used=False)
    path=study.OUT/'FITTED_GRAPH_AUDIT.json'
    if path.exists():
        original=json.loads(path.read_text())
        if original['status']!='PASS' or len(original['rows'])!=len(rows):raise RuntimeError('prior audit mismatch')
    else:study.write(path,result)
    print(json.dumps(result),flush=True)
if __name__=='__main__':audit()
