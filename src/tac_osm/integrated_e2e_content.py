"""Typed persistent-state content interface for E2E-004."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .integrated_e2e import IntegratedConfig, IntegratedE2EModel

@dataclass(frozen=True)
class ContentDiagnosisConfig(IntegratedConfig):
    content_mode: str = "learned"

class TypedPersistentState(nn.Module):
    def __init__(self, latent_bits: int, entity_count: int, content_mode: str):
        super().__init__()
        if content_mode not in {"learned", "oracle"}:
            raise ValueError("invalid content_mode")
        self.latent_bits = latent_bits
        self.entity_count = entity_count
        self.content_mode = content_mode

    def initial(self, batch: int, device: torch.device) -> Tensor:
        return torch.zeros(batch, self.entity_count, self.latent_bits, device=device)

    def write(self, memory: Tensor, content: Tensor, target_entity: Tensor):
        probs = F.one_hot(target_entity.long(), num_classes=self.entity_count).to(content.dtype)
        memory = (1.0 - probs.unsqueeze(-1)) * memory + probs.unsqueeze(-1) * content.unsqueeze(1)
        return memory, probs

    def read(self, memory: Tensor, entity: Tensor):
        attention = F.one_hot(entity.long(), num_classes=self.entity_count).to(memory.dtype)
        content = torch.einsum("bn,bnd->bd", attention, memory)
        return content, attention

class TypedContentModel(IntegratedE2EModel):
    """E2E chain with explicit typed state content and exact requested op."""
    def __init__(self, config: ContentDiagnosisConfig = ContentDiagnosisConfig()):
        super().__init__(config)
        self.config = config
        self.state = TypedPersistentState(config.latent_bits, config.entity_count, config.content_mode)

    def observation_write(self, memory: Tensor, z: Tensor, target_entity: Tensor | None = None, payload: Tensor | None = None):
        if target_entity is None:
            raise ValueError("typed state requires target_entity")
        if self.state.content_mode == "oracle":
            if payload is None:
                raise ValueError("oracle typed state requires payload")
            content = payload
        else:
            content = torch.sigmoid(self.rep.bit_head(z))
        return self.state.write(memory, content, target_entity)

    def query(self, memory: Tensor, entity: Tensor, i: Tensor, j: Tensor, op: Tensor):
        content, attention = self.state.read(memory, entity)
        a = content.gather(1, i.view(-1, 1)).squeeze(1)
        b = content.gather(1, j.view(-1, 1)).squeeze(1)
        values = torch.stack(
            (a + b - 2 * a * b, a * b, a + b - a * b, 1 - a - b + 2 * a * b),
            dim=-1,
        )
        action = values.gather(1, op.view(-1, 1)).squeeze(1)
        return {"query":self.query_vector(entity,i,j,op),"entity":entity,"read":content,"attention":attention,"action":action,"logits":torch.stack((-action,action),dim=-1),"content":content}

    def forward_episode(self, observations, observation_entities, payloads, q1, q2):
        memory=self.state.initial(observations["text"].shape[1], observations["text"].device)
        bit_loss=0.0; entity_loss=0.0; write_acc=0.0
        for t in range(observations["text"].shape[0]):
            z,_,_,_=self.encode(observations["text"][t],observations["image"][t],observations["audio"][t])
            pred_bits=torch.sigmoid(self.rep.bit_head(z))
            bit_loss=bit_loss+F.binary_cross_entropy(pred_bits,payloads[t])
            entity_loss=entity_loss+F.cross_entropy(self.rep.entity_head(z),observation_entities[t])
            memory,probs=self.observation_write(memory,z,observation_entities[t],payloads[t])
            write_acc=write_acc+(probs.argmax(-1)==observation_entities[t]).float().mean()
        def run(q):
            entity,i,j,op,target=q
            out=self.query(memory,entity,i,j,op)
            return out,F.cross_entropy(out["logits"],target)
        out1,action1=run(q1); out2,action2=run(q2)
        return {"action1_loss":action1,"action2_loss":action2,"bit_loss":bit_loss/observations["text"].shape[0],"entity_loss":entity_loss/observations["text"].shape[0],"write_target_accuracy":write_acc/observations["text"].shape[0],"q1":out1["logits"],"q2":out2["logits"],"q1_attention":out1["attention"],"q2_attention":out2["attention"],"loss":action1+action2+0.25*bit_loss/observations["text"].shape[0]+0.05*entity_loss/observations["text"].shape[0]}
