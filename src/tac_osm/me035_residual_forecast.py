"""Joint trainable residual prediction with native bounded fast/slow memory."""
import math
import torch
from torch import nn

ARMS=('native_joint','native_revin','native_residual','native_multiscale')


class ResidualTACForecaster(nn.Module):
    __constants__=['normalize','residual_slow']
    def __init__(self,arm:str):
        super().__init__()
        if arm not in ARMS:raise ValueError(arm)
        self.normalize=arm!='native_joint'
        self.residual_slow=arm in ('native_residual','native_multiscale')
        self.enc=nn.Linear(7,32);self.transition=nn.Linear(32,32,bias=False)
        self.gate=nn.Linear(39,32);self.slow=nn.Linear(64,32);self.slow_gate=nn.Linear(64,32)
        nn.init.constant_(self.gate.bias,math.log(1/7))
        nn.init.constant_(self.slow_gate.bias,math.log(1/23))
        self.direct=nn.Linear(42,21)
        self.mix=nn.Linear(106,96);self.output=nn.Linear(48,21)
        nn.init.normal_(self.direct.weight,std=.001);nn.init.zeros_(self.direct.bias)
        nn.init.normal_(self.output.weight,std=.001);nn.init.zeros_(self.output.bias)
        if self.residual_slow:
            nn.init.normal_(self.slow.weight,std=.01);nn.init.zeros_(self.slow.bias)
        if arm=='native_multiscale':
            nn.init.orthogonal_(self.transition.weight,gain=.9)
            with torch.no_grad():
                fast=torch.logspace(math.log10(2.),math.log10(32.),32)
                slow=torch.logspace(math.log10(32.),math.log10(256.),32)
                self.gate.bias.copy_(-torch.log(fast-1.))
                self.slow_gate.bias.copy_(-torch.log(slow-1.))

    @torch.jit.export
    def advance(self,u:torch.Tensor,state:torch.Tensor)->torch.Tensor:
        f,s=state[:,:32],state[:,32:]
        q=torch.tanh(self.enc(u)+self.transition(f))
        g=torch.sigmoid(self.gate(torch.cat((u,f),-1)))
        f=f+g*(q-f)
        v=torch.sigmoid(self.slow_gate(torch.cat((f,s),-1)))
        if self.residual_slow:
            innovation=torch.tanh(self.slow(torch.cat((f-s,s),-1)))
            candidate=torch.tanh(s+innovation)
        else:candidate=torch.tanh(self.slow(torch.cat((f,s),-1)))
        return torch.cat((f,s+v*(candidate-s)),-1)

    @torch.jit.export
    def encode(self,z:torch.Tensor)->torch.Tensor:
        state=z.new_zeros((z.size(0),64))
        for t in range(z.size(1)):state=self.advance(z[:,t],state)
        f,s=state[:,:32],state[:,32:]
        return torch.cat((f/torch.sqrt(f.square().mean(-1,keepdim=True)+.01),
                          s/torch.sqrt(s.square().mean(-1,keepdim=True)+.01)),-1)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        if x.dim()!=3 or x.size(1)!=96 or x.size(2)!=7:raise ValueError('expected real Bx96x7 history')
        mean=x.mean(1,keepdim=True)
        scale=torch.sqrt((x-mean).square().mean(1,keepdim=True)+.0025)
        z=(x-mean)/scale if self.normalize else x
        # Level and volatility remain available after local normalization.
        context=torch.cat((z[:,-1],z[:,-6:].mean(1),z[:,-24:].mean(1),
            z[:,-1]-z[:,-6:].mean(1),mean[:,0],torch.log(scale[:,0])),-1)
        h=self.encode(z)
        a,b=self.mix(torch.cat((context,h),-1)).chunk(2,-1)
        residual=(self.direct(context)+self.output(torch.nn.functional.silu(a)*b)).reshape(-1,3,7)
        if self.normalize:residual=residual*scale
        return x[:,-1:].expand(-1,3,-1)+residual


def build_model(arm,seed):
    torch.manual_seed(seed)
    model=ResidualTACForecaster(arm)
    assert sum(p.numel() for p in model.parameters())==18924
    return model
