"""Learned per-expert protection of the ENTIRE memory transition."""
from __future__ import annotations

import torch
from torch import Tensor,nn
import torch.nn.functional as F

from .noisy_memory import DeltaMemory,MemoryConfig,NoisyMemoryModel


class ProtectedMemory(DeltaMemory):
    def __init__(self,config: MemoryConfig,mode: str='learned_protection'):
        super().__init__(config)
        if mode not in ('learned_protection','always_update','fixed_protection'):
            raise ValueError('unknown protection arm')
        self.mode = mode
        inputs = config.address_dim+2+config.value_dim+1+config.experts
        self.protection = nn.Sequential(nn.Linear(inputs,32),nn.GELU(),nn.Linear(32,1))
        nn.init.zeros_(self.protection[-1].weight)
        nn.init.constant_(self.protection[-1].bias,2.0)
        self.last_gate: Tensor | None = None
        if mode!='learned_protection':
            for p in self.protection.parameters():
                p.requires_grad_(False)

    def update(self,state: Tensor,address: Tensor,value: Tensor,gate: Tensor|None=None) -> Tensor:
        proposal = super().update(state,address,value)
        if self.mode=='always_update':
            protection = address.new_ones((address.shape[0],self.config.experts))
        elif self.mode=='fixed_protection':
            protection = address.new_full((address.shape[0],self.config.experts),
                                           float(torch.sigmoid(address.new_tensor(2.0))))
        else:
            e = self.config.experts
            read = torch.einsum('bevk,bek->bev',state,self.keys(address))
            identity = torch.eye(e,device=address.device,dtype=address.dtype)[None].expand(address.shape[0],-1,-1)
            current = torch.cat((address,F.one_hot(value.long(),2).to(address.dtype)),-1)
            context = torch.cat((current[:,None].expand(-1,e,-1),read,
                                 torch.linalg.vector_norm(read,dim=-1,keepdim=True),identity),-1)
            protection = self.protection(context).sigmoid().squeeze(-1)
        if gate is not None:
            if gate.shape!=(address.shape[0],) or not torch.isfinite(gate).all() or ((gate<0)|(gate>1)).any():
                raise ValueError('invalid external commit gate')
            protection = protection*gate[:,None]
        self.last_gate = protection
        return state+protection[:,:,None,None]*(proposal-state)


class ProtectionModel(NoisyMemoryModel):
    def __init__(self,config: MemoryConfig=MemoryConfig(),mode: str='learned_protection'):
        super().__init__(config)
        self.memory = ProtectedMemory(config,mode)
        self.mode = mode
        self.last_gate_trace: tuple[tuple[float,float,float],...] = ()
        # This experiment isolates retention; verifier capability is not
        # trained or invoked by either arm.
        for p in self.verifier.parameters():
            p.requires_grad_(False)

    def encode(self,addresses: Tensor,values: Tensor,*,state: Tensor|None=None,trace: bool=True) -> Tensor:
        if addresses.ndim!=3 or values.shape!=addresses.shape[:2] or addresses.shape[1]<1:
            raise ValueError('nonempty matching observation tensors required')
        if state is None:
            state = self.memory.initial(addresses.shape[0],addresses)
        gates = []
        for step in range(addresses.shape[1]):
            state = self.memory.update(state,addresses[:,step],values[:,step])
            if trace:
                gate = self.memory.last_gate.detach()
                # Actual per-write gate summaries across pair/expert members.
                gates.append((float(gate.double().mean()),float(gate.min()),float(gate.max())))
        self.last_gate_trace = tuple(gates)
        return state
