"""ME061: reliability-conditioned native predictive state and trajectory head."""
from typing import NamedTuple
import torch
from torch import nn
from torch.nn import functional as F
from .real_robot_me060_trajectory import TrajectoryWorldModel, WorldState, Forecast


class VerifiedState(NamedTuple):
    fast: torch.Tensor
    slow: torch.Tensor
    memory: torch.Tensor
    velocity: torch.Tensor
    innovation: torch.Tensor


class ReliableTrajectoryModel(TrajectoryWorldModel):
    """Same TAC identity, with a causal reliability observer and exact null prior."""
    def __init__(self,identified,architecture="native_tac",width=32,decoder_width=64):
        super().__init__(identified,architecture,width,decoder_width)
        features=2*width+8 if architecture=="native_tac" else width
        self.query=nn.Linear(58,8)
        self.decoder=nn.Linear(features+58,2*decoder_width)
        self.output=nn.Linear(decoder_width,74)
        # Zero mean residual and zero signed physical gain give exactly x at
        # every horizon before learning. The final linear map receives a
        # nonzero gradient; earlier maps become active after its first update.
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def initial(self,batch,device=None):
        base=super().initial(batch,device)
        z=self.state_mean.new_zeros(batch,14,device=device)
        return VerifiedState(*base,z,z.clone())

    def observe(self,belief,previous,action,current,authority):
        native=super().observe(WorldState(belief.fast,belief.slow,belief.memory),
            previous,action,current,authority)
        delta=current-previous
        velocity=.75*belief.velocity+.25*torch.tanh(delta/self.delta_scale)
        error=((delta-self.servo_step(previous,action))/self.delta_scale).abs().clamp_max(20.)
        reliability=.9*belief.innovation+.1*error
        return VerifiedState(*native,velocity,reliability)

    def forecast(self,belief,current,actions):
        batch,horizon,_=actions.shape
        if not 1<=horizon<=64:raise ValueError("horizon outside contract")
        state=current;bases=[]
        for t in range(horizon):
            state=state+self.servo_step(state,actions[:,t]);bases.append(state)
        displacement=torch.stack(bases,1)-current[:,None]
        h=torch.arange(1,horizon+1,device=current.device,dtype=current.dtype)
        time=torch.stack((h/16.,torch.log1p(h)/torch.log(h.new_tensor(17.))),-1)
        context=torch.cat((current[:,None].expand(-1,horizon,-1),displacement,
            time[None].expand(batch,-1,-1),belief.velocity[:,None].expand(-1,horizon,-1),
            belief.innovation[:,None].expand(-1,horizon,-1)), -1)
        query=self.address(self.query(context))
        if self.architecture=="native_tac":
            recall=torch.einsum("bij,bhj->bhi",belief.memory,query)
            features=torch.cat((belief.fast[:,None].expand(-1,horizon,-1),
                belief.slow[:,None].expand(-1,horizon,-1),recall),-1)
        else:
            features=belief.fast[:,None].expand(-1,horizon,-1)+F.pad(query,(0,self.width-8))
        left,right=self.decoder(torch.cat((features,context),-1)).chunk(2,-1)
        output=self.output(F.silu(left)*right)
        residual=output[...,:42].reshape(batch,horizon,3,14).tanh()
        weights=output[...,42:46].softmax(-1)
        physical_gain=output[...,46:60].tanh()/(1.+belief.innovation[:,None])
        bound=self.delta_scale[None,None]*h[None,:,None]
        correction=(weights[...,1:,None]*residual).sum(-2)*bound
        mean=current[:,None]+physical_gain*displacement+correction
        scale=(F.softplus(output[...,60:74])+1e-3)*bound.clamp_min(1e-4)
        return Forecast(mean,scale,weights)


def physical_forecast_loss(model,prediction,actual,alive):
    mask=alive.to(actual.dtype)
    physical_error=(prediction.mean-actual).abs()*model.state_scale
    physical_scale=prediction.scale*model.state_scale
    logscore=physical_error/physical_scale+torch.log(2.*physical_scale)
    reference=(model.delta_scale*model.state_scale).mean().detach()
    count=(mask.sum()*14).clamp_min(1.)
    return ((physical_error+.01*reference*logscore)*mask).sum()/count
