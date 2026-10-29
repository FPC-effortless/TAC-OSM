"""Actual upper-block adapter; inherited full-core memory probes stay OPEN."""
import torch
from tac_osm.research_probe import ResearchProbeAdapter
from recurrence_only import RecurrenceOnlyUpperBlock


def validate(model,sample):
    if not isinstance(model,RecurrenceOnlyUpperBlock): raise ValueError('actual038 upper block required')
    if set(sample)!={'observed_features','query_features'}: raise ValueError('source feature allowlist differs')
    z,q=sample['observed_features'],sample['query_features']
    if z.ndim!=3 or q.ndim!=2 or z.shape[0]!=q.shape[0] or z.shape[-1]!=q.shape[-1] or z.shape[-1]!=model.width:
        raise ValueError('invalid source upper feature shapes')
    if not all(x.is_floating_point() and torch.isfinite(x).all() for x in (z,q)): raise ValueError('nonfinite source features')


def graph(model,state,sample):
    h=state.detach().clone().requires_grad_(True)
    q=sample['query_features'].detach().clone().requires_grad_(True)
    return {'residual_features':model.read_from_prior(q,h)},{'prior_upper_state':h,'current_upper_features':q}


UPPER_PROBE=ResearchProbeAdapter(
    name='LANGLAYER038_UPPER_ONLY_SOURCE_STEP_NOT_INHERITED_FULL_CORE',validate=validate,
    steps=lambda sample:sample['observed_features'].shape[1],
    start=lambda model,sample:sample['query_features'].new_zeros((sample['query_features'].shape[0],model.state_width)),
    advance=lambda model,state,sample,t:model.observe(sample['observed_features'][:,t],state),
    forecast=lambda model,state,sample:{'residual_features':model.read_from_prior(sample['query_features'],state)},
    state_views=lambda state:{'upper_recurrence':state},
    patch_state=lambda state,name,value:value.detach().clone() if name=='upper_recurrence' and value.shape==state.shape else (_ for _ in ()).throw(ValueError('unknown upper state')),
    batch_size=lambda sample:sample['query_features'].shape[0],
    batch_item=lambda sample,i:{name:value[i:i+1] for name,value in sample.items()},
    gradient_graph=graph)
