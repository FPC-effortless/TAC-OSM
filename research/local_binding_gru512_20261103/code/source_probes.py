"""Bound actual source-adapter probes inside the single admitted job."""
import json

def run(torch, model, out, specification):
    from collect018.probe import run_source_probes
    seen={};activity={'forward_calls':0,'autograd_grad_calls':0}
    rng=torch.get_rng_state().clone()
    def forward(module,args):
        seen[id(module)]=module;activity['forward_calls']+=1
        if activity['forward_calls']>specification['neural_forward_calls']:
            raise RuntimeError('SOURCE_PROBE_FORWARD_CAP')
    hook=model.register_forward_pre_hook(forward)
    original=torch.autograd.grad
    def grad(*args,**kwargs):
        activity['autograd_grad_calls']+=1
        if activity['autograd_grad_calls']>specification['autograd_grad_calls']:
            raise RuntimeError('SOURCE_PROBE_AD_CAP')
        return original(*args,**kwargs)
    torch.autograd.grad=grad
    try: report=run_source_probes(model)
    finally:
        torch.autograd.grad=original;hook.remove()
    if activity['forward_calls']!=specification['neural_forward_calls'] or activity['autograd_grad_calls']!=1:
        raise RuntimeError('SOURCE_PROBE_COUNTS_DIFFER')
    if len(seen)-1!=specification['model_deepcopies']:
        raise RuntimeError('SOURCE_PROBE_MODEL_COPY_COUNT_DIFFERS')
    if not torch.equal(rng,torch.get_rng_state()):raise RuntimeError('SOURCE_PROBE_CHANGED_RNG')
    if report['entity_isolation']['status']!='PASS' or report['forecast_state_purity']!='PASS':
        raise RuntimeError('SOURCE_PROBE_FAILED')
    for head in report['dependencies']['heads'].values():
        for result in head.values():
            if not result['connected'] or not result['finite']:raise RuntimeError('SOURCE_PROBE_DEPENDENCY_FAILED')
    receipt={'status':'PASS_ACTUAL_SOURCE_ADAPTER_FINITE_PROBES','report':report,'activity':activity,
             'model_deepcopies':len(seen)-1,'RNG_unchanged':True,'optimizer_updates':0,
             'limit':'Finite initial-state source probe; no global causal/IEEE/full-graph proof.'}
    with (out/'SOURCE_PROBE_RECEIPT.json').open('x') as f:f.write(json.dumps(receipt,indent=2)+'\n')
    return receipt
