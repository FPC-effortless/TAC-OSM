"""Full-history trainable linear backbone plus pretrained native innovations."""
import torch
from torch import nn
from tac_osm.me035_residual_forecast import ResidualTACForecaster


class ARResidualTAC(ResidualTACForecaster):
    def __init__(self):
        super().__init__('native_multiscale')
        self.direct=nn.Linear(672,21)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        if x.dim()!=3 or x.size(1)!=96 or x.size(2)!=7:raise ValueError('expected Bx96x7 history')
        mean=x.mean(1,keepdim=True);scale=torch.sqrt((x-mean).square().mean(1,keepdim=True)+.0025)
        z=(x-mean)/scale
        context=torch.cat((z[:,-1],z[:,-6:].mean(1),z[:,-24:].mean(1),
            z[:,-1]-z[:,-6:].mean(1),mean[:,0],torch.log(scale[:,0])),-1)
        a,b=self.mix(torch.cat((context,self.encode(z)),-1)).chunk(2,-1)
        correction=self.output(torch.nn.functional.silu(a)*b).reshape(-1,3,7)*scale
        return x[:,-1:]+self.direct(x.flatten(1)).reshape(-1,3,7)+correction


def initialize(parent_state,coef,seed):
    torch.manual_seed(seed);m=ARResidualTAC()
    state=m.state_dict()
    for k,v in parent_state.items():
        if not k.startswith(('direct.','output.')):state[k]=v
    m.load_state_dict(state,strict=True)
    with torch.no_grad():
        m.direct.weight.copy_(torch.as_tensor(coef[:-1].T,dtype=torch.float32))
        m.direct.bias.copy_(torch.as_tensor(coef[-1],dtype=torch.float32))
    assert sum(p.numel() for p in m.parameters())==32154
    return m
