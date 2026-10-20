"""ME009: bounded reliability gate on an unchanged native TAC recurrent forecaster.

The gate is fitted to genuinely observed future outcomes from historical calibration
records. Inference uses only past real sensor values/masks and frozen native forecast.
This is not an SSM conversion or autonomous agent safety monitor.
"""
import torch
from torch import nn
from torch.nn import functional as F
from scripts.me004_native_real_forecast import FEATURES,HORIZONS

D=len(FEATURES)
K=len(HORIZONS)

class ReliabilityGate(nn.Module):
    def __init__(self,frozen_native):
        super().__init__()
        self.native=frozen_native
        for p in self.native.parameters():p.requires_grad_(False)
        self.net=nn.Sequential(nn.Linear(8,16),nn.SiLU(),nn.Linear(16,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("horizon_token",torch.tensor([0.0,1.0]))

    def forward(self,x,details=False):
        if x.ndim!=3 or x.shape[-1]!=2*D or x.shape[1]!=32:
            raise ValueError("expected 32 authenticated chronological sensor records")
        mask=x[:,:,D:].clamp(0,1)
        values=torch.where(mask>0,x[:,:,:D],torch.zeros_like(x[:,:,:D]))
        inputs=torch.cat([values,mask],dim=-1)
        native_delta=self.native(inputs)
        observed_count=mask.sum(1).clamp_min(1)
        mean=values.sum(1)/observed_count
        var=(((values-mean[:,None,:])*mask).square()).sum(1)/observed_count
        volatility=var.clamp_min(0).sqrt()
        last_change=torch.where((mask[:,-1,:]*mask[:,-2,:])>0,
                                values[:,-1,:]-values[:,-2,:],
                                torch.zeros_like(values[:,-1,:]))
        def full(z):return z[:,None,:].expand(-1,K,-1)
        h=self.horizon_token.view(1,K,1).expand_as(native_delta)
        f=torch.stack((native_delta,native_delta.abs(),full(volatility),
              full(last_change),full(last_change.abs()),
              full(mask[:,-1,:]),full(mask.mean(1)),h),dim=-1)
        alpha=torch.sigmoid(self.net(f).squeeze(-1))
        prediction=alpha*native_delta
        if not details:return prediction
        return {"prediction":prediction,"original":native_delta,"alpha":alpha,
                "forecast_variance_proxy":full(volatility)}

def loss_on_authentic_outcomes(pred,target,train_only_threshold):
    if pred.shape!=target.shape:
        raise ValueError("different target/output shape")
    pointwise=F.smooth_l1_loss(pred,target,beta=.2,reduction="none")
    changed=target.abs()>train_only_threshold.unsqueeze(0)
    rare=(pointwise*changed).sum()/changed.sum().clamp_min(1)
    return pointwise.mean()+.2*rare
