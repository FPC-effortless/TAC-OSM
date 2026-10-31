"""Original099 state/owner probe protocol with the actual100 execution wrapper."""
from dataclasses import replace
import importlib.util,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('langrecomp100_common',HERE/'common.py');C=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=C;_spec.loader.exec_module(C)

def adapter100(policy='recompute'):
    if policy not in C.POLICIES:raise ValueError('registered policy required')
    original=C.load('original_probe',C.C.HERE/'probe_adapter.py').adapter099()
    def forecast(model,state,sample):
        y,_=C.load('recompute').forward(model,sample['tokens'][:,-1:],owners=sample['owners'],state=state,policy=policy)
        return {'nextbyte_logits':y[:,0]}
    def graph(model,state,sample):
        leaves={n:t.detach().clone().requires_grad_(True) for n,t in state.tensors.items()}
        return forecast(model,replace(state,tensors=leaves),sample),leaves
    return replace(original,name='LANGRECOMP100-'+policy+'-original099-native-state',
        advance=lambda model,state,sample,t:C.load('recompute').forward(model,sample['tokens'][:,t:t+1],state=state,owners=sample['owners'],policy=policy)[1],
        forecast=forecast,gradient_graph=graph)
