"""Separate multiscale predictive innovations on native fast/slow memory.

Optional Torch research architecture. Fixed forecast bases are public cadence
priors, not learned future observations. No global novelty or streaming claim.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from tac_osm.me038_innovation_forecast import InnovationForecaster
from tac_osm.me040_dualrate_innovation import pair_observed_patches
from tac_osm.me043_adaptive_compute import ContextSeededMemory
from tac_osm.me043_scheduled_compute import build_model as previous_model
from tac_osm.me043_scheduled_compute import objective as previous_objective

MATCHED=('pyramid_native','pyramid_context','pyramid_gru')
ARMS=MATCHED+('previous_scheduled','original_context')


def forecast_basis(horizon,period):
    if horizon not in (96,720) or period not in (24,144):
        raise ValueError('unregistered horizon/cadence')
    t=torch.arange(horizon,dtype=torch.float64)
    u=t/(horizon-1)
    knots=torch.linspace(0,1,8,dtype=torch.float64)
    hats=(1-7*(u[:,None]-knots[None,:]).abs()).clamp_min(0)
    modes=[hats]
    for k in (1,2,4,8):
        phase=2*math.pi*k*(t+1)/period
        modes += [phase.sin()[:,None],phase.cos()[:,None]]
    matrix=torch.cat(modes,dim=1)
    q,r=torch.linalg.qr(matrix,mode='reduced')
    signs=torch.where(torch.diag(r)<0,-1.,1.)
    return (q*signs[None,:]*math.sqrt(horizon)).T.contiguous().float()


class MultiscaleForecaster(InnovationForecaster):
    def __init__(self,arm,horizon,period):
        super().__init__('context_cross',horizon)
        self.study_arm=arm
        if arm=='pyramid_native':self.branch=ContextSeededMemory()
        elif arm=='pyramid_context':
            self.branch=nn.Sequential(nn.Linear(192,37),nn.SiLU(),nn.Linear(37,64))
        elif arm=='pyramid_gru':
            self.branch=nn.GRU(16,40,batch_first=True)
            self.branch_projection=nn.Linear(40,64)
        else:raise ValueError('unregistered multiscale architecture')
        self.coefficients=nn.Sequential(nn.Linear(208,16),nn.SiLU(),nn.Linear(16,16))
        nn.init.zeros_(self.coefficients[-1].weight)
        nn.init.zeros_(self.coefficients[-1].bias)
        self.trust=nn.Linear(208,1)
        nn.init.zeros_(self.trust.weight);nn.init.zeros_(self.trust.bias)
        self.register_buffer('forecast_modes',forecast_basis(horizon,period))

    def features(self,x):
        if x.ndim!=3 or x.size(1)!=96:raise ValueError('expected B x96 xC observed window')
        b,_,c=x.shape;mean=x.mean(1,keepdim=True)
        scale=torch.sqrt((x-mean).square().mean(1,keepdim=True)+1e-5)
        normalized=(x-mean)/scale
        flat=normalized.transpose(1,2).reshape(b*c,96)
        shared=normalized.mean(2,keepdim=True).expand(b,96,c).transpose(1,2).reshape(b*c,96)
        patches=flat.reshape(-1,12,8)
        tokens=torch.cat((patches,shared.reshape(-1,12,8)),-1)
        context=torch.cat((patches.mean(-1),flat[:,-1:],
            flat[:,-1:]-flat[:,-8:].mean(-1,keepdim=True),
            mean.transpose(1,2).reshape(-1,1),scale.log().transpose(1,2).reshape(-1,1)),-1)
        window=torch.cat((flat,shared),-1);latent=self.encoder(window)
        return self.ar(flat),tokens,context,latent,window,mean,scale

    def unpack(self,flat,mean,scale):
        b,_,c=mean.shape
        return flat.reshape(b,c,self.horizon).transpose(1,2)*scale+mean

    def cheap_from(self,features):
        base,_,ctx,latent,_,mean,scale=features
        a,g=self.mix(torch.cat((ctx,latent),-1)).chunk(2,-1)
        raw=self.out(F.silu(self.basis(F.silu(a)*g)))
        delta=self.horizon_gate(ctx)*torch.tanh(raw)
        return self.unpack(base+delta,mean,scale),delta.reshape(mean.size(0),mean.size(2),self.horizon).transpose(1,2)*scale

    def innovation_from(self,features):
        _,tokens,ctx,latent,window,_,scale=features
        ctx=ctx.detach();latent=latent.detach()
        if self.study_arm=='pyramid_native':state=self.branch(tokens,latent)
        elif self.study_arm=='pyramid_context':state=self.branch(window)
        else:
            _,h=self.branch(pair_observed_patches(tokens),.1*torch.tanh(latent[:,:40]).unsqueeze(0))
            state=self.branch_projection(h[0])
        inputs=torch.cat((ctx,latent,state),-1)
        coefficients=self.coefficients(inputs)
        trust=torch.sigmoid(self.trust(inputs))
        raw=coefficients@self.forecast_modes
        # Separate bounded correction, no forced vanishing at horizon720.
        correction=.5*trust*torch.tanh(raw/.5)
        b,_,c=scale.shape
        delta=correction.reshape(b,c,self.horizon).transpose(1,2)*scale
        return delta,coefficients,trust

    def training_outputs(self,x,expert=True):
        features=self.features(x);cheap,delta=self.cheap_from(features)
        if not expert:return cheap,delta,None,None,None,features[-1]
        residual,coefficients,trust=self.innovation_from(features)
        return cheap,delta,residual,coefficients,trust,features[-1]

    def forward(self,x):
        cheap,_,residual,_,_,_=self.training_outputs(x,True)
        return cheap+residual


def build_model(arm,horizon,seed,period):
    if arm=='previous_scheduled':return previous_model('scheduled_free',horizon,seed)
    torch.manual_seed(seed)
    if arm=='original_context':return InnovationForecaster('context_cross',horizon)
    net=MultiscaleForecaster(arm,horizon,period)
    # Keep every cheap-path initialized coefficient exactly equal to reference.
    torch.manual_seed(seed);old=InnovationForecaster('context_cross',horizon)
    state=net.state_dict();state.update(old.state_dict());net.load_state_dict(state,strict=True)
    return net


def is_expert(name):
    return name.startswith(('branch.','branch_projection.','coefficients.','trust.'))


def objective(net,x,y,step):
    if not isinstance(net,MultiscaleForecaster):return previous_objective(net,x,y,step)
    cheap,delta,residual,coef,_,scale=net.training_outputs(x,step%4==0)
    mse=(cheap-y).square().mean()
    loss=mse+10*net.anchor_loss()+.01*delta.square().mean()
    if residual is not None:
        residual_target=(y-cheap.detach())
        normalized=(residual_target/scale).transpose(1,2).reshape(-1,net.horizon)
        projected=(normalized@net.forecast_modes.T/net.horizon).detach()
        residual_mse=(cheap.detach()+residual-y).square().mean()
        loss=loss+residual_mse+.05*(coef-projected).square().mean()+.01*residual.square().mean()
        mse=residual_mse
    return loss,mse


def clip_gradients(net):
    if isinstance(net,MultiscaleForecaster):
        # Separate clipping preserves cheap-path optimization despite native
        # residual gradients; its target and inputs are detached above.
        base=torch.nn.utils.clip_grad_norm_([p for n,p in net.named_parameters() if not is_expert(n)],1.)
        extra=torch.nn.utils.clip_grad_norm_([p for n,p in net.named_parameters() if is_expert(n)],1.)
        return torch.maximum(base,extra)
    return torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
