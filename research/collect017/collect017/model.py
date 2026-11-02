"""Reuse native TAC attention over the entire registered context range."""
import hashlib,json
import torch
from collect011.model import CollectiveTAC
from tacround110.model import ResearchTAC110,RoundConfig
from collect016.common import build as phase_initializer
from collect016.model import run_chunks
class AttentionSanity(CollectiveTAC):
    def __init__(self):
        seed=torch.initial_seed();original={n:p.detach().clone() for n,p in phase_initializer('full_bptt',seed).named_parameters()}
        torch.manual_seed(seed)
        config=RoundConfig(depth=4,d_model=16,ffn_dim=38,ssa=True,n_heads=4,key_dim=16,value_dim=16,window=1536,local_rope=True)
        ResearchTAC110.__init__(self,config);self.arm='attention'
        self.collect_signature=hashlib.sha256(json.dumps(dict(arm='COLLECT017-attention',config=self.config_signature),sort_keys=True).encode()).hexdigest()
        with torch.no_grad():
            for n,p in self.named_parameters():
                if n in original and p.shape==original[n].shape:p.copy_(original[n])
                elif n in original and '.ffn_' in n:
                    a=original[n]
                    if p.shape[0]==a.shape[0]:p.copy_(a[:,:p.shape[1]])
                    else:p.copy_(a[:p.shape[0]])
    def balancing(self):
        # No phase bank router exists: this loss is structurally absent.
        return self.core.embedding.weight.new_zeros(()),self.core.embedding.weight.new_zeros(4,4)
