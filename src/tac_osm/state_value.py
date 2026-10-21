"""ME027 shared-observable models; native TAC dynamics are unchanged."""
from __future__ import annotations
import math
import torch
from torch import nn


class StateValueModel(nn.Module):
    __constants__=['arm','is_tac','is_gru','is_transformer','is_mlp']

    def __init__(self,arm:str,mean:float,scale:float,popularity:torch.Tensor):
        super().__init__()
        if arm not in ['tac','gru','transformer','mlp']:raise ValueError('unregistered arm')
        self.arm=arm
        self.is_tac=arm=='tac';self.is_gru=arm=='gru';self.is_transformer=arm=='transformer';self.is_mlp=arm=='mlp'
        self.emb=nn.Embedding(16000,8)
        if arm=='tac':
            self.enc=nn.Linear(11,32);self.transition=nn.Linear(32,32,bias=False)
            self.gate=nn.Linear(43,32);self.slow=nn.Linear(64,32);self.slow_gate=nn.Linear(64,32)
            nn.init.constant_(self.gate.bias,math.log(1/7));nn.init.constant_(self.slow_gate.bias,math.log(1/23))
        elif arm=='gru':
            self.cell=nn.GRUCell(11,34);self.to_state=nn.Linear(34,64)
        elif arm=='transformer':
            self.in_proj=nn.Linear(11,16)
            layer=nn.TransformerEncoderLayer(16,2,48,dropout=0.,batch_first=True,norm_first=True,activation='gelu')
            self.blocks=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
            self.final_norm=nn.LayerNorm(16);self.to_state=nn.Linear(16,64)
            pos=torch.arange(128,dtype=torch.float32)[:,None]
            omega=torch.exp(torch.arange(0,16,2)*(-math.log(10000.)/16))
            encoding=torch.zeros(128,16);encoding[:,0::2]=torch.sin(pos*omega);encoding[:,1::2]=torch.cos(pos*omega)
            self.register_buffer('positions',encoding)
        else:self.feature_hidden=nn.Linear(26,252)
        self.head=nn.Linear(252 if arm=='mlp' else 90,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        self.register_buffer('duration_mean',torch.tensor(mean));self.register_buffer('duration_scale',torch.tensor(scale))
        self.register_buffer('item_popularity',popularity.detach().clone())
        self.register_buffer('halflives',torch.tensor([2.,8.,16.,32.,64.]))

    @torch.jit.export
    def observables(self,items:torch.Tensor,meta:torch.Tensor,action:torch.Tensor,duration:torch.Tensor,depth:int)->torch.Tensor:
        ages=torch.arange(depth-1,-1,-1,device=meta.device,dtype=meta.dtype)
        weights=torch.exp2(-ages[:,None]/self.halflives[None,:])
        kernels=meta[:,:depth,1]@(weights/weights.sum(0,keepdim=True))
        pooled=torch.cat((self.emb(items[:,:depth]).mean(1),meta[:,:depth].mean(1)),-1)
        return torch.cat((self.emb(action),(duration-self.duration_mean)/self.duration_scale,
                          kernels,self.item_popularity[action].unsqueeze(-1),pooled),-1)

    @torch.jit.export
    def advance(self,u:torch.Tensor,state:torch.Tensor)->torch.Tensor:
        if self.is_tac:
            z,s=state[:,:32],state[:,32:]
            q=torch.tanh(self.enc(u)+self.transition(z));g=torch.sigmoid(self.gate(torch.cat((u,z),-1)))
            z=(1.-g)*z+g*q;ctx=torch.cat((z,s),-1);v=torch.sigmoid(self.slow_gate(ctx))
            return torch.cat((z,(1.-v)*s+v*torch.tanh(self.slow(ctx))),-1)
        elif self.is_gru:return self.cell(u,state)
        else:return state

    @torch.jit.export
    def features(self,state:torch.Tensor)->torch.Tensor:
        if self.is_gru:state=self.to_state(state)
        z,s=state[:,:32],state[:,32:]
        return torch.cat((z/torch.sqrt(z.square().mean(-1,keepdim=True)+.01),
                          s/torch.sqrt(s.square().mean(-1,keepdim=True)+.01)),-1)

    @torch.jit.export
    def predict_features(self,latent:torch.Tensor,common:torch.Tensor,prior:torch.Tensor)->torch.Tensor:
        if self.is_mlp:return self.head(torch.tanh(self.feature_hidden(common)))+torch.logit(prior)
        else:return self.head(torch.cat((latent,common),-1))+torch.logit(prior)

    @torch.jit.export
    def attention_states(self,items:torch.Tensor,meta:torch.Tensor)->torch.Tensor:
        if self.is_transformer:
            v=self.in_proj(torch.cat((self.emb(items),meta),-1))+self.positions.unsqueeze(0)
            mask=torch.ones(128,128,device=items.device,dtype=torch.bool).triu(1)
            return self.to_state(self.final_norm(self.blocks(v,mask=mask)))
        else:return meta.new_zeros((items.size(0),128,64))

    @torch.jit.export
    def latent_features(self,items:torch.Tensor,meta:torch.Tensor)->torch.Tensor:
        if self.is_mlp:return meta.new_zeros((items.size(0),0))
        elif self.is_transformer:return self.features(self.attention_states(items,meta)[:,-1])
        else:
            x=torch.cat((self.emb(items),meta),-1)
            state=x.new_zeros((items.size(0),34 if self.is_gru else 64))
            for t in range(128):state=self.advance(x[:,t],state)
            return self.features(state)

    def forward(self,items:torch.Tensor,meta:torch.Tensor,action:torch.Tensor,duration:torch.Tensor)->torch.Tensor:
        common=self.observables(items,meta,action,duration,128)
        prior=(meta[:,-32:,1].sum(-1,keepdim=True)+1)/34
        return self.predict_features(self.latent_features(items,meta),common,prior)

    @torch.jit.export
    def prefix_logits(self,items:torch.Tensor,meta:torch.Tensor,action:torch.Tensor,duration:torch.Tensor)->torch.Tensor:
        out=torch.jit.annotate(list[torch.Tensor],[])
        if self.is_tac or self.is_gru:
            x=torch.cat((self.emb(items),meta),-1);state=x.new_zeros((items.size(0),34 if self.is_gru else 64))
            for t in range(128):
                state=self.advance(x[:,t],state);depth=t+1
                if depth==32 or depth==64 or depth==96:
                    prior=(meta[:,depth-32:depth,1].sum(-1,keepdim=True)+1)/34
                    common=self.observables(items,meta,items[:,depth],meta[:,depth,0:1],depth)
                    out.append(self.predict_features(self.features(state),common,prior))
            latent=self.features(state)
        else:
            states=self.attention_states(items,meta)
            for depth in [32,64,96]:
                prior=(meta[:,depth-32:depth,1].sum(-1,keepdim=True)+1)/34
                common=self.observables(items,meta,items[:,depth],meta[:,depth,0:1],depth)
                latent=self.features(states[:,depth-1]) if self.is_transformer else meta.new_zeros((items.size(0),0))
                out.append(self.predict_features(latent,common,prior))
            latent=self.features(states[:,-1]) if self.is_transformer else meta.new_zeros((items.size(0),0))
        prior=(meta[:,-32:,1].sum(-1,keepdim=True)+1)/34
        out.append(self.predict_features(latent,self.observables(items,meta,action,duration,128),prior))
        return torch.cat(out,-1)
