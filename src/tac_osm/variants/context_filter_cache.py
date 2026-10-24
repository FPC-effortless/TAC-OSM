"""Versioned context-aware entity state cache; never authoritative world state."""
from dataclasses import dataclass
import hashlib
import torch
from .context_filter_world import ContextFilterState
from .reusable_execution import EntityInferenceCache,_clone


@dataclass(frozen=True)
class _ContextEntry:
    sequence:int
    observed:torch.Tensor
    command:torch.Tensor
    elapsed:torch.Tensor
    image_sha256:bytes
    task_id:int
    clock_context:torch.Tensor
    state:ContextFilterState
    forecast:dict


class ContextFilterCache(EntityInferenceCache):
    @torch.no_grad()
    def step(self,entity,sequence,observed,known_command,*,elapsed,image,task_id,clock_context):
        if self.model.training or self._identity!=self._model_identity():
            self._entries.clear();raise RuntimeError('model changed; create a new revision-bound inference cache')
        if not isinstance(entity,str) or not entity or type(sequence)!=int or sequence<0:raise ValueError('explicit entity and nonnegative integer sequence required')
        if type(task_id)!=int or not -1<=task_id<=4:raise ValueError('known task0..4 or explicit unknown-1 required')
        reference=next(self.model.parameters())
        for value,shape in [(observed,(1,14)),(known_command,(1,14)),(elapsed,(1,1)),(image,(1,1,32,32)),(clock_context,(1,6))]:
            if value.shape!=shape or value.dtype!=reference.dtype or value.device!=reference.device or not torch.isfinite(value).all():raise ValueError('finite input shape/device/dtype matching model required')
        if not (elapsed>0).all() or (image<0).any() or (image>1).any():raise ValueError('positive elapsed and gray32range[0,1] required')
        pixel_hash=hashlib.sha256(image.detach().cpu().contiguous().numpy().tobytes()).digest();previous=self._entries.get(entity)
        if previous and task_id!=previous.task_id:raise ValueError('operation changed; explicitly retire and start a verified new stream')
        if previous and sequence==previous.sequence:
            if pixel_hash!=previous.image_sha256 or not all(torch.equal(a,b) for a,b in [(observed,previous.observed),(known_command,previous.command),(elapsed,previous.elapsed),(clock_context,previous.clock_context)]):raise ValueError('conflicting retry; reset and replay verified prefix')
            return _clone(previous.forecast)
        if sequence!=(previous.sequence+1 if previous else 0):raise ValueError('gap or revision; reset and replay verified prefix')
        if previous is None and len(self._entries)>=self.max_entities:raise RuntimeError('explicitly retire a stream before adding another')
        ids=torch.tensor([task_id],dtype=torch.int64,device=reference.device)
        forecast,state=self.model.step(observed,known_command,None if previous is None else previous.state,elapsed=elapsed,images=image,task_ids=ids,clock_context=clock_context)
        state=ContextFilterState(*(v.detach().clone() for v in state))
        self._entries[entity]=_ContextEntry(sequence,observed.detach().clone(),known_command.detach().clone(),elapsed.detach().clone(),pixel_hash,task_id,clock_context.detach().clone(),state,_clone(forecast))
        return _clone(forecast)
