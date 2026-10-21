"""ME029 isolate useful ordered recurrent state from powerful nonlinear observables.

Nothing in this module changes ME028's actual fast/slow memory equations.
Synthetic tensor checks are only numerical/causal software verification.
"""
from __future__ import annotations
import torch
from torch import nn
from tac_osm.me028_nonlinear_readout import NativeNonlinearObservableReadout

VOCAB=16000


def block_order_sort(items:torch.Tensor, metadata:torch.Tensor):
    """Sort ONLY four fully observed 32-event blocks; NEVER look ahead a block.

    Sorting keys are historical items, not outcome labels; carry outcome and
    time metadata together with original item. Defined at four 32-event
    forecast boundaries; do not use for a mid-block streaming forecast.
    """
    b,t=items.shape
    if t!=128 or metadata.shape!=(b,128,3):
        raise ValueError("original 128 historical items and event metadata required")
    x=items.reshape(b,4,32)
    m=metadata.reshape(b,4,32,3)
    permutation=torch.argsort(x,dim=-1,stable=True)
    moved_items=torch.gather(x,2,permutation).reshape(b,128)
    moved_metadata=torch.gather(
        m,2,permutation.unsqueeze(-1).expand(b,4,32,3)
    ).reshape(b,128,3)
    return moved_items,moved_metadata


class BlockSortedNative(NativeNonlinearObservableReadout):
    """Exactly native ME028 parameters/readout, but memory input block order sorted.

    Observable history is left chronologically ordered and identical across
    arms; no change to the next action, its duration, feedback or prior32.
    """
    def forward(self,items:torch.Tensor,metadata:torch.Tensor,
                action:torch.Tensor,duration:torch.Tensor)->torch.Tensor:
        si,sm=block_order_sort(items,metadata)
        state=self.latent_features(si,sm)
        common=self.observables(items,metadata,action,duration,128)
        prior=(metadata[:,-32:,1].sum(-1,keepdim=True)+1.)/34.
        return self.predict_features(state,common,prior)

    @torch.jit.export
    def prefix_logits(self,items:torch.Tensor,metadata:torch.Tensor,
                      action:torch.Tensor,duration:torch.Tensor)->torch.Tensor:
        si,sm=block_order_sort(items,metadata)
        x=torch.cat((self.emb(si),sm),-1)
        state=x.new_zeros((items.size(0),64))
        logits=torch.jit.annotate(list[torch.Tensor],[])
        for t in range(128):
            state=self.advance(x[:,t],state)
            k=t+1
            if k==32 or k==64 or k==96:
                common=self.observables(items,metadata,items[:,k],
                                        metadata[:,k,0:1],k)
                prior=(metadata[:,k-32:k,1].sum(-1,keepdim=True)+1.)/34.
                logits.append(self.predict_features(self.features(state),common,prior))
        common=self.observables(items,metadata,action,duration,128)
        prior=(metadata[:,-32:,1].sum(-1,keepdim=True)+1.)/34.
        logits.append(self.predict_features(self.features(state),common,prior))
        return torch.cat(logits,-1)


class OrderFreeNonlinearControl(nn.Module):
    """Explicit permutation-invariant, active-parameter near-matched control.

    11->56->56->64 MLP of the *mean* 128 observed event vectors, same
    64-feature split RMS map and nonlinear current-feature readout. All body
    matrices active; no dummy or unused parameter padding.
    """
    def __init__(self,mean:float,scale:float,popularity:torch.Tensor):
        super().__init__()
        self.emb=nn.Embedding(VOCAB,8)
        self.pool=nn.Sequential(nn.Linear(11,56),nn.Tanh(),
                                nn.Linear(56,56),nn.Tanh(),
                                nn.Linear(56,64))
        self.head=nn.Linear(64,1,bias=False)
        self.observable_hidden=nn.Linear(26,12)
        self.observable_output=nn.Linear(12,1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.observable_output.weight)
        nn.init.zeros_(self.observable_output.bias)
        self.register_buffer("duration_mean",torch.tensor(mean))
        self.register_buffer("duration_scale",torch.tensor(scale))
        self.register_buffer("item_popularity",popularity.detach().clone())
        self.register_buffer("halflives",torch.tensor([2.,8.,16.,32.,64.]))
        if popularity.shape != (16000,):raise ValueError("FIT-only popularity 16000")

    @torch.jit.export
    def observables(self,items:torch.Tensor,meta:torch.Tensor,
                    action:torch.Tensor,duration:torch.Tensor,depth:int)->torch.Tensor:
        ages=torch.arange(depth-1,-1,-1,device=meta.device,dtype=meta.dtype)
        weights=torch.exp2(-ages[:,None]/self.halflives[None,:])
        kernels=meta[:,:depth,1]@(weights/weights.sum(0,keepdim=True))
        pooled=torch.cat((self.emb(items[:,:depth]).mean(1),
                           meta[:,:depth].mean(1)),-1)
        return torch.cat((self.emb(action),
                          (duration-self.duration_mean)/self.duration_scale,
                          kernels,
                          self.item_popularity[action].unsqueeze(-1),
                          pooled),-1)

    @torch.jit.export
    def features(self,state:torch.Tensor)->torch.Tensor:
        z,s=state[:,:32],state[:,32:]
        return torch.cat((
            z/torch.sqrt(z.square().mean(-1,keepdim=True)+.01),
            s/torch.sqrt(s.square().mean(-1,keepdim=True)+.01)),dim=-1)

    @torch.jit.export
    def latent_features(self,items:torch.Tensor,meta:torch.Tensor)->torch.Tensor:
        """Exactly permutation invariant for arbitrary past event permutations."""
        values=torch.cat((self.emb(items),meta),-1).mean(1)
        return self.features(self.pool(values))

    @torch.jit.export
    def predict_features(self,latent:torch.Tensor,common:torch.Tensor,
                         prior:torch.Tensor)->torch.Tensor:
        return self.head(latent)+self.observable_output(
            torch.tanh(self.observable_hidden(common)))+torch.logit(prior)

    def forward(self,items:torch.Tensor,meta:torch.Tensor,
                action:torch.Tensor,duration:torch.Tensor)->torch.Tensor:
        state=self.latent_features(items,meta)
        common=self.observables(items,meta,action,duration,128)
        prior=(meta[:,-32:,1].sum(-1,keepdim=True)+1.)/34.
        return self.predict_features(state,common,prior)

    @torch.jit.export
    def prefix_logits(self,items:torch.Tensor,meta:torch.Tensor,
                      action:torch.Tensor,duration:torch.Tensor)->torch.Tensor:
        out=torch.jit.annotate(list[torch.Tensor],[])
        for depth in [32,64,96]:
            state=self.latent_features(items[:,:depth],meta[:,:depth])
            common=self.observables(items,meta,items[:,depth],meta[:,depth,0:1],depth)
            prior=(meta[:,depth-32:depth,1].sum(-1,keepdim=True)+1.)/34.
            out.append(self.predict_features(state,common,prior))
        state=self.latent_features(items,meta)
        common=self.observables(items,meta,action,duration,128)
        prior=(meta[:,-32:,1].sum(-1,keepdim=True)+1.)/34.
        out.append(self.predict_features(state,common,prior))
        return torch.cat(out,-1)


def active_params(net)->int:
    return sum(p.numel() for p in net.parameters())


def preflight():
    torch.set_num_threads(1)
    torch.manual_seed(29029)
    pop=torch.full((16000,),.35)
    a=NativeNonlinearObservableReadout(.5,.2,pop)
    b=BlockSortedNative(.5,.2,pop)
    c=OrderFreeNonlinearControl(.5,.2,pop)
    counts={"chrono_native":active_params(a),
            "block_sorted_native":active_params(b),
            "order_free_pool":active_params(c)}
    if counts != {"chrono_native":135377,"block_sorted_native":135377,"order_free_pool":135913}:
        raise AssertionError(f"pre-registered parameter mismatch: {counts}")
    item=torch.randint(16000,(2,128))
    meta=torch.rand(2,128,3)
    meta[:,:,1]=(meta[:,:,1]>.5).float()
    act=torch.randint(16000,(2,))
    dur=torch.rand(2,1)
    si,sm=block_order_sort(item,meta)
    for k in (32,64,96,128):
        if not torch.equal(torch.sort(item[:,:k].flatten()).values,
                           torch.sort(si[:,:k].flatten()).values):
            raise AssertionError("block sorting leaked or lost original item")
        original=torch.cat((item[:,:k].float().unsqueeze(-1),
                            meta[:,:k]),dim=-1)
        permuted=torch.cat((si[:,:k].float().unsqueeze(-1),
                            sm[:,:k]),dim=-1)
        # Verify exact tuple multiset across genuine complete block boundaries.
        for row in range(2):
            u=original[row].cpu().numpy()
            v=permuted[row].cpu().numpy()
            import numpy as np
            if not np.array_equal(np.sort(u,axis=0),np.sort(v,axis=0)):
                raise AssertionError("sorted native dropped original observed metadata")
    for name,model in (("chrono",a),("sorted",b),("pool",c)):
        model.eval()
        prior=torch.logit((meta[:,-32:,1].sum(-1,keepdim=True)+1)/34)
        out=model(item,meta,act,dur)
        if not torch.allclose(out,prior,atol=1e-6,rtol=1e-6):
            raise AssertionError(name+" wrong analytic initial prior")
        scripted=torch.jit.script(model)
        pp=model.prefix_logits(item,meta,act,dur)
        if pp.shape!=(2,4) or not torch.allclose(
            pp,scripted.prefix_logits(item,meta,act,dur),atol=3e-6,rtol=3e-6):
            raise AssertionError(name+" TorchScript or prefix mismatch")
        # Disable zero-head vacuity.
        with torch.no_grad():
            model.head.weight.normal_(std=.15)
            model.observable_output.weight.normal_(std=.1)
        original=model.prefix_logits(item,meta,act,dur)
        for k,col in ((32,0),(64,1),(96,2)):
            mutate_i=item.clone();mutate_m=meta.clone()
            mutate_i[:,k+1:]=torch.randint(16000,mutate_i[:,k+1:].shape)
            mutate_m[:,k+1:]=torch.rand_like(mutate_m[:,k+1:])
            mutate_m[:,k+1:,1]=(mutate_m[:,k+1:,1]>.5).float()
            mutate_m[:,k,1]=1.-mutate_m[:,k,1] # current unknown response
            test=model.prefix_logits(mutate_i,mutate_m,act,dur)
            if not torch.allclose(original[:,col],test[:,col],atol=6e-6,rtol=6e-6):
                raise AssertionError(name+" nonzero causal violation at "+str(k))
        y=torch.tensor([[1.,0.,1.,0.],[0.,1.,0.,1.]])
        loss=nn.functional.binary_cross_entropy_with_logits(original,y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        for nm,p in model.named_parameters():
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise AssertionError(name+" inactive/nonfinite gradient "+nm)
    # Pool state exact order independence for real observed event **pairs**.
    ci=OrderFreeNonlinearControl(.5,.2,pop)
    perm=torch.randperm(128)
    if not torch.allclose(ci.latent_features(item,meta),
                          ci.latent_features(item[:,perm],meta[:,perm]),
                          atol=3e-6,rtol=3e-6):
        raise AssertionError("order-free hidden contains artificial chronological order")
    return {"status":"ME029_NUMERICAL_ONLY_FALSIFIER_PASS",
            "params":counts,"all_active":True,
            "causal_prefix_noninterference":True,
            "real_training_or_performance_claim":False,
            "reserved_data_read":False}
