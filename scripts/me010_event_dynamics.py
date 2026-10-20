"""ME010 native TAC event-conditioned dynamics: trained forecaster plus two regimes.

The same three-output 8->24->3 feature map serves both research arms.
'mixture' learns Bernoulli event probability and two conditional signed forecasts.
'direct' uses the three channels as an ordinary direct residual correction.
No future label enters inference. The native TAC-1000 recurrence is frozen.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from scripts.me004_native_real_forecast import FEATURES, HORIZONS
D=len(FEATURES);K=len(HORIZONS)

class EventDynamics(nn.Module):
    def __init__(self,native,mode="mixture"):
        super().__init__()
        if mode not in ("mixture","direct"):raise ValueError("mode")
        self.mode=mode
        self.native=native
        for p in self.native.parameters():p.requires_grad_(False)
        self.trunk=nn.Sequential(nn.Linear(8,24),nn.SiLU(),nn.Linear(24,3))
        nn.init.zeros_(self.trunk[-1].weight)
        nn.init.zeros_(self.trunk[-1].bias)
        self.register_buffer("horizon_indicator",torch.tensor([0.,1.],dtype=torch.float32))

    def forward(self,x,details=False):
        if x.ndim!=3 or x.shape[1:]!=(32,2*D):raise ValueError("bad causal input shape")
        mask=x[:,:,D:].clamp(0,1)
        values=torch.where(mask>0,x[:,:,:D],torch.zeros_like(x[:,:,:D]))
        native_delta=self.native(torch.cat((values,mask),dim=-1))
        count=mask.sum(1).clamp_min(1)
        mean=values.sum(1)/count
        std=((values-mean[:,None,:]).square()*mask).sum(1).div(count).clamp_min(0).sqrt()
        valid=mask[:,-1,:]*mask[:,-2,:]
        trend=(values[:,-1,:]-values[:,-2,:])*valid
        def rep(v):return v[:,None,:].expand(-1,K,-1)
        horizon=self.horizon_indicator.view(1,K,1).expand_as(native_delta)
        features=torch.stack((native_delta,native_delta.abs(),
             rep(std),rep(trend),rep(trend.abs()),
             rep(mask[:,-1,:]),rep(mask.mean(1)),horizon),dim=-1)
        logits=self.trunk(features)
        if self.mode=="direct":
            pred=native_delta+logits.sum(dim=-1)/3
            p=None;change_pred=None;quiet_pred=None
        else:
            p=torch.sigmoid(logits[...,0])
            # At initialization p=.5, changed-state mean=2*native, quiet mean=0.
            # Thus the initial mixture EXACTLY preserves frozen native output.
            change_pred=2*native_delta+logits[...,1]
            quiet_pred=logits[...,2]
            pred=p*change_pred+(1-p)*quiet_pred
        if not details:return pred
        return {"prediction":pred,"baseline":native_delta,
                "p_change":p,"mean_if_change":change_pred,"mean_if_quiet":quiet_pred}

def train_loss(model,details,target,threshold):
    pred=details["prediction"]
    if pred.shape!=target.shape or tuple(threshold.shape)!=(K,D):
        raise ValueError("forecast/outcome or train-only threshold shape mismatch")
    outcome=(target.abs()>threshold[None,:,:])
    base=F.smooth_l1_loss(pred,target,beta=.2,reduction="none")
    rare=base[outcome].mean() if bool(outcome.any()) else base.mean()*0
    loss=base.mean()+.20*rare
    stats={"forecast_huber":float(base.detach().mean()),
           "observed_event_frequency":float(outcome.float().detach().mean())}
    if model.mode=="mixture":
        p=details["p_change"]
        # Proper classification score: never infer event from future within forward.
        probability_loss=F.binary_cross_entropy(p,outcome.float())
        event_error=F.smooth_l1_loss(details["mean_if_change"],target,beta=.2,reduction="none")
        quiet_error=F.smooth_l1_loss(details["mean_if_quiet"],target,beta=.2,reduction="none")
        event_cond=event_error[outcome].mean() if bool(outcome.any()) else event_error.mean()*0
        quiet_cond=quiet_error[~outcome].mean() if bool((~outcome).any()) else quiet_error.mean()*0
        loss=loss+.15*probability_loss+.10*event_cond+.10*quiet_cond
        stats["event_bce"]=float(probability_loss.detach())
    return loss,stats

def physical_event_metrics(model,details,target,threshold):
    event=target.abs()>threshold[None,:,:]
    quiet=~event
    prediction=details["prediction"]
    metrics={"forecast_mae":float((prediction-target).abs().mean()),
        "event_count":int(event.sum()),"quiet_count":int(quiet.sum()),
        "event_magnitude_mae":float((prediction-target).abs()[event].mean()) if bool(event.any()) else None,
        "quiet_false_alarm_rate":float(((prediction.abs()>threshold[None,:,:])&quiet).sum()/quiet.sum()) if bool(quiet.any()) else None,
        "true_positive_changes":int(((target>threshold[None,:,:])&event).sum()),
        "true_negative_changes":int(((target< -threshold[None,:,:])&event).sum())}
    if model.mode=="mixture":
        p=details["p_change"]
        alarm=(p>=.5)
        tp=(alarm&event).sum()
        metrics.update({"event_brier":float((p-event.float()).square().mean()),
            "event_log_loss":float(F.binary_cross_entropy(p.clamp(1e-7,1-1e-7),event.float())),
            "event_prevalence":float(event.float().mean()),
            "event_precision":float(tp/alarm.sum()) if bool(alarm.any()) else 0.,
            "event_recall":float(tp/event.sum()) if bool(event.any()) else 0.,
            "event_probability_mean":float(p.mean()),
            "event_ece_10_bins":float(sum(
                (mask.float().mean()*abs(p[mask].mean()-event.float()[mask].mean())).item()
                for index in range(10)
                for mask in [((p>=index/10)&(p<(index+1)/10))] if bool(mask.any())))})
    return metrics
