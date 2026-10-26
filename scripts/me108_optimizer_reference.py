"""Executable equations for one ME108 proposal and its two rounding checks.

No optimizer/model method or production guard is called. The exact-real Lean
contract excludes these precision-specific rechecks and fallback branches.
"""
import torch


def step(weights, gradients, point, old_states, groups, blocks):
    next_states=[]
    proposals=[]
    for w,g,s,config in zip(weights,gradients,old_states,groups):
        b1,b2=config['betas'];t=int(s.get('step',0))+1
        m=s.get('exp_avg',torch.zeros_like(w))*b1+g*(1-b1)
        v=s.get('exp_avg_sq',torch.zeros_like(w))*b2+g.square()*(1-b2)
        candidate=w*(1-config['lr']*config['weight_decay'])-(m/(1-b1**t))/((v/(1-b2**t)).sqrt()+config['eps'])*config['lr']
        proposals.append(candidate-w)
        next_states.append(dict(step=torch.tensor(float(t)),exp_avg=m,exp_avg_sq=v))
    delta=[d.clone() for d in proposals]
    for indices in blocks:
        numerator=sum((point[i].double()*proposals[i].double()).sum() for i in indices)
        norm=sum(point[i].double().square().sum() for i in indices)
        if norm>0 and numerator>0:
            for i in indices:
                delta[i]=(proposals[i].double()-numerator/norm*point[i].double()).to(weights[i].dtype)
        residual=sum((point[i].double()*delta[i].double()).sum() for i in indices)
        if residual>0:
            for i in indices:
                delta[i]=torch.where(point[i]==0,proposals[i],torch.zeros_like(proposals[i]))
    candidates=[w+d for w,d in zip(weights,delta)]
    for indices in blocks:
        applied=sum((point[i].double()*(candidates[i].double()-weights[i].double())).sum() for i in indices)
        if applied>0:
            for i in indices:
                candidates[i]=torch.where(point[i]==0,candidates[i],weights[i])
    return candidates,next_states
