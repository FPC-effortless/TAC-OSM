"""Compute-adaptive TAC: a context exit and conditionally executed native expert.

Optional Torch research module. Six actual observed-token updates preserve
native fast/slow/error-write dynamics; this is not streaming or action control.
"""
import torch
from torch import nn
from torch.nn import functional as F
from tac_osm.me038_innovation_forecast import InnovationForecaster, InnovationMemory
from tac_osm.me040_dualrate_innovation import pair_observed_patches

COSTS = {'adaptive_free': 0., 'adaptive_001': .001, 'adaptive_005': .005}
CONTROLS = ('context_cross', 'innovation_tac', 'gru_cross')
ARMS = CONTROLS + tuple(COSTS)


class ContextSeededMemory(InnovationMemory):
    def forward(self, patches, context):
        tokens = pair_observed_patches(patches)
        # Fixed bounded seeding, no extra parameter padding. Both states remain
        # learned native dynamical states with a separate 8x8 innovation matrix.
        f = .1 * torch.tanh(context[:, :32])
        s = .1 * torch.tanh(context[:, 32:64])
        matrix = patches.new_zeros((patches.size(0), 8, 8))
        for t in range(6):
            f, s, matrix = self.step(tokens[:, t], f, s, matrix)
        state = torch.cat((f, s), -1)
        return state / torch.sqrt(state.square().mean(-1, keepdim=True) + .01)


class AdaptiveForecaster(InnovationForecaster):
    def __init__(self, horizon, cost):
        super().__init__('context_cross', horizon)
        self.cost = cost
        # Context width20 + native weights + router nearly matches width50
        # context-only reference. Every learned coefficient has an active role.
        self.encoder = nn.Sequential(nn.Linear(192,20),nn.SiLU(),nn.Linear(20,128))
        self.memory = ContextSeededMemory()
        self.router = nn.Linear(144,1)
        nn.init.zeros_(self.router.weight);nn.init.zeros_(self.router.bias)
        self.native_gain = nn.Parameter(torch.tensor(-2.))

    def features(self, x):
        if x.ndim != 3 or x.size(1) != 96:
            raise ValueError('expected B x 96 x C observed history')
        b,_,c=x.shape
        mean=x.mean(1,keepdim=True)
        scale=torch.sqrt((x-mean).square().mean(1,keepdim=True)+1e-5)
        normalized=(x-mean)/scale
        flat=normalized.transpose(1,2).reshape(b*c,96)
        shared=normalized.mean(2,keepdim=True).expand(b,96,c).transpose(1,2).reshape(b*c,96)
        patches=flat.reshape(-1,12,8)
        tokens=torch.cat((patches,shared.reshape(-1,12,8)),-1)
        context=torch.cat((patches.mean(-1),flat[:,-1:],
            flat[:,-1:]-flat[:,-8:].mean(-1,keepdim=True),
            mean.transpose(1,2).reshape(-1,1),scale.log().transpose(1,2).reshape(-1,1)),-1)
        latent=self.encoder(torch.cat((flat,shared),-1))
        q=torch.sigmoid(self.router(torch.cat((context,latent),-1)))
        return self.ar(flat),tokens,context,latent,q,mean,scale

    def correction(self, context, latent):
        a,g=self.mix(torch.cat((context,latent),-1)).chunk(2,-1)
        raw=self.out(F.silu(self.basis(F.silu(a)*g)))
        return self.horizon_gate(context)*torch.tanh(raw)

    def refine(self, tokens, latent):
        state=self.memory(tokens,latent)
        return torch.cat((latent[:,:64],latent[:,64:]+torch.sigmoid(self.native_gain)*state),-1)

    def unpack(self, flat, mean, scale):
        b,_,c=mean.shape
        return flat.reshape(b,c,self.horizon).transpose(1,2)*scale+mean

    def training_exits(self, x):
        # Dense TRAIN execution prevents expert starvation. This does NOT
        # claim conditional-training throughput; all training CPU is charged.
        base,tokens,ctx,latent,q,mean,scale=self.features(x)
        cheap=self.correction(ctx,latent)
        full=self.correction(ctx,self.refine(tokens,latent))
        return (self.unpack(base+cheap,mean,scale),
                self.unpack(base+full,mean,scale),q,cheap,full)

    def routed(self, x):
        base,tokens,ctx,latent,q,mean,scale=self.features(x)
        correction=self.correction(ctx,latent)
        selected=torch.nonzero(q[:,0]>.5,as_tuple=False).flatten()
        # Critical compute boundary: no native expert call for unselected
        # channel/windows, rather than multiplying a densely computed output.
        if selected.numel()>0:
            refined=self.refine(tokens.index_select(0,selected),latent.index_select(0,selected))
            native=self.correction(ctx.index_select(0,selected),refined)
            correction=correction.index_copy(0,selected,native)
        return self.unpack(base+correction,mean,scale),q

    def forward(self, x):
        return self.routed(x)[0]


def build_model(arm, horizon, seed):
    torch.manual_seed(seed)
    if arm in COSTS:
        return AdaptiveForecaster(horizon,COSTS[arm])
    if arm in CONTROLS:
        return InnovationForecaster(arm,horizon)
    raise ValueError('unregistered ME043 architecture')


def objective(net, x, y):
    if isinstance(net,AdaptiveForecaster):
        cheap,full,q,dc,df=net.training_exits(x)
        # One observed future per channel, no fabricated alternative outcome.
        ec=(cheap-y).square().mean(1).reshape(-1,1)
        ef=(full-y).square().mean(1).reshape(-1,1)
        target=(ec.detach()-ef.detach()>net.cost).to(q.dtype)
        router=((1-q)*ec.detach()+q*(ef.detach()+net.cost)).mean()
        router=router+.05*F.binary_cross_entropy(q,target)
        mse=.5*(ec.mean()+ef.mean())
        return mse+router+10*net.anchor_loss()+.005*(dc.square().mean()+df.square().mean()),mse
    p,delta,_=net.parts(x)
    mse=(p-y).square().mean()
    return mse+10*net.anchor_loss()+.01*delta.square().mean(),mse
