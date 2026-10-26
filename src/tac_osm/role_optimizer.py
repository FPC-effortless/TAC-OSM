"""Block-protected AdamW proposals for a prospectively registered TAC study.

Moments follow the clipped combined gradient even when a proposed weight move
is rejected. The complete displacement includes group rates and decay; it is
protected BEFORE any real parameter mutation and checked again after rounding
the candidate weights. This constrains computed point-gradient dots only.
"""
import math
import torch

from .role_update_contract import protect_displacement


class BlockProtectedAdamW(torch.optim.Optimizer):
    def __init__(self, params, *, blocks, lr=3e-4, betas=(.9,.999), eps=1e-8, weight_decay=.001):
        if lr < 0 or eps <= 0 or weight_decay < 0 or not all(0 <= b < 1 for b in betas):
            raise ValueError('invalid AdamW hyperparameters')
        super().__init__(params,dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay))
        self._check_groups()
        self.blocks=tuple(tuple(b) for b in blocks)
        self._point=None
        self.last_guard_receipt=None
        self._params=tuple(p for g in self.param_groups for p in g['params'])
        ids=[i for b in self.blocks for i in b]
        if any(type(i) is not int for i in ids) or sorted(ids)!=list(range(len(self._params))) or any(not b for b in self.blocks):
            raise ValueError('blocks must exhaustively partition flattened optimizer parameters')
        if len({id(p) for p in self._params})!=len(self._params):
            raise ValueError('duplicate optimizer parameters')

    def _check_groups(self):
        for group in self.param_groups:
            values=(group['lr'],group['eps'],group['weight_decay'],*group['betas'])
            if not all(math.isfinite(v) for v in values) or group['lr']<0 or group['eps']<=0 or group['weight_decay']<0 or not all(0<=b<1 for b in group['betas']):
                raise ValueError('invalid group-specific AdamW hyperparameters')

    def set_point_gradients(self, gradients):
        gradients=tuple(gradients)
        if len(gradients)!=len(self._params):
            raise ValueError('point gradients must cover every parameter')
        for p,g in zip(self._params,gradients):
            if g.shape!=p.shape or g.dtype!=p.dtype or g.device!=p.device or not torch.isfinite(g).all():
                raise ValueError('invalid point gradient shape/dtype/device/value')
        self._point=tuple(g.detach().clone() for g in gradients)

    @torch.no_grad()
    def propose(self, point_gradients):
        """Read-only candidates/moments for independent replay; no fit occurs."""
        point=tuple(point_gradients)
        if len(point)!=len(self._params):
            raise ValueError('point gradients must cover every parameter')
        for p,g in zip(self._params,point):
            if g.shape!=p.shape or g.dtype!=p.dtype or g.device!=p.device or not torch.isfinite(g).all():
                raise ValueError('invalid point gradient')
        self._check_groups()
        if tuple(id(p) for group in self.param_groups for p in group['params'])!=tuple(id(p) for p in self._params):
            raise ValueError('optimizer group order changed')
        proposals=[]
        moments=[]
        for group in self.param_groups:
            b1,b2=group['betas']
            for p in group['params']:
                g=p.grad
                if g is None or g.is_sparse or not torch.isfinite(g).all() or not torch.isfinite(p).all():
                    raise ValueError('finite dense combined gradients and parameters required')
                old=self.state.get(p,{})
                t=int(old.get('step',0))+1
                m=old.get('exp_avg',torch.zeros_like(p))*b1+g*(1-b1)
                v=old.get('exp_avg_sq',torch.zeros_like(p))*b2+g.square()*(1-b2)
                denom=(v/(1-b2**t)).sqrt()+group['eps']
                candidate=p*(1-group['lr']*group['weight_decay'])-(m/(1-b1**t))/denom*group['lr']
                if not torch.isfinite(m).all() or not torch.isfinite(v).all() or not torch.isfinite(candidate).all():
                    raise ValueError('nonfinite AdamW moment/proposal')
                proposals.append(candidate-p)
                moments.append(dict(step=torch.tensor(float(t)),exp_avg=m,exp_avg_sq=v))
        safe,receipt=protect_displacement(point,proposals,self.blocks)
        candidates=[p+d for p,d in zip(self._params,safe)]
        for block,row in zip(self.blocks,receipt):
            actual=sum((point[i].double()*(candidates[i].double()-self._params[i].double())).sum() for i in block)
            if not torch.isfinite(actual):
                raise ValueError('nonfinite applied-displacement dot')
            rejected=bool(actual>0)
            if rejected:
                for i in block:
                    candidates[i]=torch.where(point[i]==0,candidates[i],self._params[i])
                actual=sum((point[i].double()*(candidates[i].double()-self._params[i].double())).sum() for i in block)
            if actual>0 or not torch.isfinite(actual):
                raise ValueError('applied displacement contract failed')
            row.update(applied_dot=float(actual),applied_rounding_fallback=rejected)
        if not all(torch.isfinite(c).all() for c in candidates):
            raise ValueError('nonfinite protected candidate')
        return candidates,moments,receipt

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('closures are unsupported; compute frozen-batch roles explicitly')
        if self._point is None:
            raise ValueError('a fresh point-gradient snapshot is required for each step')
        point=self._point
        self._point=None
        candidates,moments,receipt=self.propose(point)
        # All checks precede ANY real weight or moment mutation.
        for p,c,state in zip(self._params,candidates,moments):
            p.copy_(c)
            self.state[p]=state
        self.last_guard_receipt=receipt
        return None
