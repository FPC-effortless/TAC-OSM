"""One registered authentic FIT minibatch, no training or predictive scoring."""
import json
from pathlib import Path

import numpy as np
import torch

from scripts import me106_priority_panel as parent
from scripts.me106_gradient_roles import objective_parts
from scripts.me107_graph_reference import serial
from scripts.validate_research_parity import check_manifest
from tac_osm.variants.role_specialized_sensor_world import RoleSpecializedSensorWorld
from tac_osm.variants.median_split_sensor_world import MedianSplitForecast

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/mathematical-engineering/me107/SOURCE_AUDIT.json'


def audit():
    check_manifest(json.loads((OUT.parent/'FAIR_COMPARISON.json').read_text()),'preflight')
    torch.set_num_threads(1)
    d=parent.fit_data()
    rng=np.random.default_rng(107001)
    origins=[d[p][int(i)] for p in ('general','gap') for i in rng.integers(len(d[p]),size=8)]
    source_batch=parent.tensor_batch(d,origins,64,rng)
    results=[]
    for dtype,rtol,atol in ((torch.float32,2e-5,2e-6),(torch.float64,1e-9,1e-10)):
        torch.manual_seed(107001)
        model=RoleSpecializedSensorWorld(torch.tensor(parent.gap.floor_rates(d)['rates'])).to(dtype)
        with torch.no_grad():
            model.target_mean.copy_(torch.from_numpy(d['mean'][:6]))
            model.target_scale.copy_(torch.from_numpy(d['std'][:6]))
        batch=[t.to(dtype) if t.is_floating_point() else t for t in source_batch]
        rawscale=torch.tensor(np.nanstd(d['raw'][...,:6].astype(np.float64),axis=(0,1)).clip(1e-4),dtype=dtype)
        versions=model.versions()
        actual,state,tr=model(*batch[:6],return_trace=True)
        mu,left,right,ss,sm,es,mem,reftrace=serial(model,*batch[:5])
        checks={}
        for name,a,b in zip(('location','left_scale','right_scale','source_sum','source_mass','evidence_sum','memory'),
                           (actual.log_location,actual.left_scale,actual.right_scale,state.source_sum,state.source_mass,state.evidence_sum,state.memory),
                           (mu,left,right,ss,sm,es,mem)):
            torch.testing.assert_close(a,b,rtol=rtol,atol=atol)
            checks[name]=float((a-b).detach().abs().max())
        loss,_=objective_parts(model,actual,tr,batch,rawscale,{'point':.5,'interval':.25})
        ref_loss,_=objective_parts(model,MedianSplitForecast(mu,left,right),reftrace,batch,rawscale,{'point':.5,'interval':.25})
        ga=torch.autograd.grad(loss,tuple(model.parameters()),allow_unused=True)
        gb=torch.autograd.grad(ref_loss,tuple(model.parameters()),allow_unused=True)
        gradients={}
        for (name,p),a,b in zip(model.named_parameters(),ga,gb):
            a=torch.zeros_like(p) if a is None else a
            b=torch.zeros_like(p) if b is None else b
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            torch.testing.assert_close(a,b,rtol=rtol,atol=atol,msg=name)
            gradients[name]=dict(norm=float(a.norm()),nonzero=bool(torch.count_nonzero(a)),max_abs_error=float((a-b).abs().max()))
        assert model.versions()==versions
        assert sum(p.numel() for p in model.parameters())==102713
        results.append(dict(dtype=str(dtype),rtol=rtol,atol=atol,trainable_parameters=102713,deployed_parameters=102713,
                            parameter_tensors=len(gradients),forward_max_abs=checks,gradients=gradients))
    return dict(no_new_model_fits=True,no_optimizer_updates=True,no_predictive_score=True,
                source_archive_sha256=parent.data.SHA,cohort='2013-2014 FIT only',results=results)


if __name__=='__main__':
    result=audit()
    if OUT.exists():
        # Original error maxima remain intact. Re-run the same strict source
        # assertions, not equality of hardware-sensitive diagnostic maxima.
        print('ME107_ACTUAL_SOURCE_REFERENCE_PASS (original receipt preserved)')
    else:
        parent.write(OUT,result)
        print('ME107_ACTUAL_SOURCE_REFERENCE_PASS')
