"""ME078 native fast/slow/associative world predictor and shared-head controls.

Established gated delta updates, GRU and causal attention are explicitly credited.
All models consume the same available physical history and use identical kinds
of dense multi-horizon quantile/kinematic readouts. No future action inputs.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

HORIZONS=(1,4,16)

class NativeCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.fast_map=nn.Linear(128,64)
        self.slow_map=nn.Linear(64,64)
        self.address=nn.Linear(64,18)
        self.project=nn.Linear(72,64)
        with torch.no_grad():
            self.address.bias[-2]=4.6
            self.address.bias[-1]=-2.
    def forward(self,x):
        f=x.new_zeros(len(x),32);s=f.clone();m=x.new_zeros(len(x),8,8);states=[]
        for t in range(x.shape[1]):
            obs=x[:,t]
            key,value,retention,write=torch.split(self.address(obs),(8,8,1,1),-1)
            key=key/(key.square().sum(-1,keepdim=True)+1e-6).sqrt()
            read=(m*key[:,None,:]).sum(-1)
            c,g=self.fast_map(torch.cat((obs,f,s),-1)).chunk(2,-1)
            f=torch.lerp(f,c.tanh(),g.sigmoid())
            innovation,rate=self.slow_map(torch.cat((f-s,s),-1)).chunk(2,-1)
            s=torch.lerp(s,(s+innovation.tanh()).tanh(),rate.sigmoid())
            decayed=retention.sigmoid()[:,:,None]*m
            error=value.tanh()-(decayed*key[:,None,:]).sum(-1)
            m=decayed+write.sigmoid()[:,:,None]*error[:,:,None]*key[:,None,:]
            m=m/torch.maximum(m.square().sum((1,2),keepdim=True).clamp_min(1e-12).sqrt()/4,torch.ones_like(m[:,:1,:1]))
            states.append(self.project(torch.cat((f,s,read),-1)))
        return torch.stack(states,1)

class GRUCore(nn.Module):
    def __init__(self):
        super().__init__();self.input=nn.Linear(64,168);self.hidden=nn.Linear(56,168);self.project=nn.Linear(56,64)
    def forward(self,x):
        h=x.new_zeros(len(x),56);out=[]
        for t in range(x.shape[1]):
            ir,iz,inc=self.input(x[:,t]).chunk(3,-1)
            hr,hz,hn=self.hidden(h).chunk(3,-1)
            r=(ir+hr).sigmoid();z=(iz+hz).sigmoid()
            n=(inc+r*hn).tanh();h=(1-z)*n+z*h
            out.append(self.project(h))
        return torch.stack(out,1)

class RMSNorm(nn.Module):
    def __init__(self,d):
        super().__init__();self.weight=nn.Parameter(torch.ones(d))
    def forward(self,x):
        return x*(x.square().mean(-1,keepdim=True)+1e-5).rsqrt()*self.weight

class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__();self.n1=RMSNorm(32);self.qkv=nn.Linear(32,96);self.out=nn.Linear(32,32)
        self.n2=RMSNorm(32);self.ff1=nn.Linear(32,64);self.ff2=nn.Linear(64,32)
    def forward(self,x):
        b,t,d=x.shape
        q,k,v=self.qkv(self.n1(x)).reshape(b,t,3,4,8).unbind(2)
        q,k,v=[z.transpose(1,2) for z in (q,k,v)]
        scores=(q@k.transpose(-1,-2))/math.sqrt(8)
        mask=torch.ones(t,t,device=x.device,dtype=torch.bool).triu(1)
        scores=scores.masked_fill(mask,float('-inf'))
        weights=torch.softmax(scores,-1)
        attended=(weights@v).transpose(1,2).reshape(b,t,32)
        x=x+self.out(attended)
        return x+self.ff2(F.silu(self.ff1(self.n2(x))))

class TransformerCore(nn.Module):
    def __init__(self):
        super().__init__();self.input=nn.Linear(64,32);self.blocks=nn.Sequential(AttentionBlock(),AttentionBlock());self.project=nn.Linear(32,64)
        pos=torch.arange(8)[:,None]*torch.exp(torch.arange(16)*(-math.log(10000)/16))[None,:]
        self.register_buffer('position',torch.stack((pos.sin(),pos.cos()),-1).reshape(8,32))
    def forward(self,x):
        return self.project(self.blocks(self.input(x)+self.position[None,:x.shape[1]]))

class KinematicWorld(nn.Module):
    def __init__(self,architecture='native',width=None,identified=None):
        super().__init__();self.architecture=architecture
        widths={'native':133,'gru':123,'transformer':127}
        self.encoder=nn.Sequential(nn.Linear(57,64),nn.SiLU(),nn.Linear(64,64),nn.Tanh())
        self.core={'native':NativeCore,'gru':GRUCore,'transformer':TransformerCore}[architecture]()
        width=widths[architecture] if width is None else width
        self.gated_readout=nn.Linear(170,width*2)
        self.heads=nn.Linear(width,213)
        nn.init.normal_(self.heads.weight,std=.001);nn.init.zeros_(self.heads.bias)
        with torch.no_grad():
            self.heads.bias[126:168].fill_(-2.)
            self.heads.bias[168:210].fill_(-4.)
            self.heads.bias[210:].fill_(-2.)
        defaults={'delta_scale':torch.ones(14)*.01,'servo_gain':torch.zeros(14),'servo_step_limit':torch.ones(14)*.1,
          'action_mean':torch.zeros(14),'action_scale':torch.ones(14),'state_mean':torch.zeros(14),'state_scale':torch.ones(14)}
        for k,v in defaults.items():
            self.register_buffer(k,torch.as_tensor(v if identified is None else identified[k]).float().clone())
        self.register_buffer('horizons',torch.tensor(HORIZONS).float())
    def forward(self,x,actions):
        if x.ndim!=3 or x.shape[-1]!=14 or x.shape[1]>8 or actions.shape!=x.shape:
            raise ValueError('causal <=8-frame 14D histories and prior-known actions required')
        velocity=torch.cat((torch.zeros_like(x[:,:1]),x[:,1:]-x[:,:-1]),1)
        acceleration=torch.cat((torch.zeros_like(x[:,:1]),velocity[:,1:]-velocity[:,:-1]),1)
        command=(actions*self.action_scale+self.action_mean-self.state_mean)/self.state_scale-x
        features=torch.cat((x,command,torch.asinh(velocity/self.delta_scale),torch.asinh(acceleration/self.delta_scale),torch.ones_like(x[:,:,:1])*(1/15)), -1)
        observed=self.encoder(features);state=self.core(observed)
        context=torch.cat((observed,state,features[:,:,14:56]),-1)
        r,g=self.gated_readout(context).chunk(2,-1)
        output=self.heads(F.silu(g)*r)
        raw=output[:,:,:126].reshape(len(x),x.shape[1],3,3,14)
        median,lo,hi=raw.unbind(3)
        vgate=output[:,:,126:168].reshape(len(x),x.shape[1],3,14).sigmoid()
        sgate=output[:,:,168:210].reshape(len(x),x.shape[1],3,14).sigmoid()
        horizon=self.horizons[None,None,:,None]
        servo=torch.clamp(self.servo_gain*command,-self.servo_step_limit,self.servo_step_limit)
        bound=torch.maximum(4*self.servo_step_limit,torch.ones_like(self.delta_scale)*.05)*horizon
        delta=bound*torch.tanh((horizon*(vgate*velocity[:,:,None]+sgate*servo[:,:,None])+bound*median.tanh())/bound)
        center=x[:,:,None]+delta
        unit=self.delta_scale*horizon
        lower=center-unit*(.01+F.softplus(lo))
        upper=center+unit*(.01+F.softplus(hi))
        return {'median':center,'lower':lower,'upper':upper,'motion_logit':output[:,:,210:]}

def dense_loss(prediction,targets,current,valid,importance,physical_scale,normalizer):
    horizons=targets.new_tensor(HORIZONS)[None,None,:,None]
    losses=[]
    for name,tau in (('lower',.05),('median',.5),('upper',.95)):
        error=(targets-prediction[name])*physical_scale/(normalizer*horizons)
        element=torch.maximum(tau*error,(tau-1)*error).mean(-1)
        losses.append((element*valid).sum((1,2))/valid.sum((1,2)).clamp_min(1))
    physical=torch.stack(losses,-1).mean(-1)
    # The global event is a separate honest forecast, not a claimed CDF of the quantiles.
    event=(((targets-current[:,:,None]).abs()*physical_scale).mean(-1)>=.025).float()
    event_loss=F.binary_cross_entropy_with_logits(prediction['motion_logit'],event,reduction='none')
    risk=(event_loss*valid).sum((1,2))/valid.sum((1,2)).clamp_min(1)
    return (importance*(physical+.1*risk)).mean()
