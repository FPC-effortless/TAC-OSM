"""Actual ME111 FIT-only source-to-output and learning-role probes; no fitting."""
import numpy as np
import torch
from torch.func import functional_call
from scripts import me111_conditional_training as old
from scripts.me111_objective import losses
from tac_osm.typed_role_learning import uncertainty_masks
from tac_osm.conditional_belief_learning import routed_gradients
from tac_osm.research_observatory import architecture_inventory,gradient_conflict_atlas
from tac_osm.research_anticheat import (AuditLedger,Finding,audit_input_boundary,
    probe_future_invariance,probe_prediction_invariance,trace_torch_noninterference,
    trace_torch_modules,torch_gradient_diagnostics)

FIELDS=('values','observed','ages','clock','owners','hours')
OPEN=('full_source_csv_partition_provenance','full_transform_fit_provenance',
      'nuisance_shuffle_controls','full_neural_ieee_refinement')


def examine(d,seed,origins,rawscale):
    assert d['phase']=='fit'
    path,parent,_=old.parent_artifacts(seed)
    model=old.initialize(d,seed);old.load_parent(model,parent)
    x=old.tensor_batch(d,origins,256,np.random.default_rng(seed+10000))
    versions=model.versions();saved={k:v.clone() for k,v in model.state_dict().items()}
    with trace_torch_modules(model) as activations:
        forecast,state,trace=model(*x[:6],return_trace=True)
    point,unc,parts=losses(model,forecast,trace,x,rawscale)
    atlas=gradient_conflict_atlas(model,{'balanced_point':point,'uncertainty':unc,
        'nll':parts['nll'],'interval':.25*parts['interval'],'reconstruction':.3*parts['reconstruction_nll']})
    point_grads,receipt=routed_gradients(model,point,unc,parts['reconstruction_nll'])
    gradient_health=torch_gradient_diagnostics(model)
    ledger=AuditLedger(f'ME111-prefit-{seed}', ['input_boundary','future_invariance',
        'prediction_invariance','graph_noninterference','gradient_health',*OPEN])
    ledger.add(audit_input_boundary(dict(zip(FIELDS,x[:6])),FIELDS));ledger.add(gradient_health)
    sid,t=map(int,origins[0]);original=float(d['raw'][sid,t+6,0])
    assert np.isfinite(original)
    def prepare(value):
        alternate=dict(d);alternate['raw']=d['raw'].copy()
        alternate['raw'][sid,t+6,0]=value['future_pm']
        return old.tensor_batch(alternate,origins[:1],256,np.random.default_rng(seed+10000))[:6]
    clean={'future_pm':original};poison={'future_pm':original+101.}
    ledger.add(probe_future_invariance(clean,poison,prepare,lambda q:q["future_pm"]))
    with torch.no_grad():
        ledger.add(probe_prediction_invariance(lambda q: tuple(model(*prepare(q))[0]),clean,poison))
        ledger.add(trace_torch_noninterference(model,lambda m,q:tuple(m(*prepare(q),state=None)[0]),clean,poison))
        masks=uncertainty_masks(model)
        replacement={n:p.detach()+torch.where(mask,torch.full_like(p,.01),torch.zeros_like(p))
                     for (n,p),mask in zip(model.named_parameters(),masks)}
        altered,altered_state=functional_call(model,replacement,tuple(x[:6]),strict=False)
        median_unchanged=torch.equal(model.raw_quantiles(forecast)[...,1],model.raw_quantiles(altered)[...,1])
        state_unchanged=all(torch.equal(v,getattr(altered_state,k)) for k,v in vars(state).items() if isinstance(v,torch.Tensor))
    assert median_unchanged and state_unchanged
    for name in OPEN:ledger.add(Finding(name,'INCONCLUSIVE','Uninstrumented boundary; no complete integrity or capability proof',{}))
    report=ledger.report()
    assert all(f['status']=='PASS' for f in report['findings'] if f['check'] not in OPEN)
    assert model.versions()==versions and all(torch.equal(saved[k],v) for k,v in model.state_dict().items())
    return dict(seed=seed,parent_checkpoint_sha256=old.sha(path),routing=receipt,
        architecture=architecture_inventory(model),gradient_atlas=atlas,activations=activations,
        ledger=report,auxiliary_coordinate_perturbation=.01,median_bit_unchanged=median_unchanged,
        native_tensor_state_bit_unchanged=state_unchanged,residual_hidden_gradient_scope='trained ME110 residual outputs inherited; use actual reported gradients, no zero-output assumption',
        no_optimizer_steps=True,normalization_cache_fixed_during_poison=True)
