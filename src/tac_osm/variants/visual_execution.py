"""Explicit versioned visual-stream cache; all model math remains functional."""
from dataclasses import dataclass
import hashlib
import torch
from .clock_value_world import ClockValueState
from .reusable_execution import EntityInferenceCache, _clone


@dataclass(frozen=True)
class _VisualEntry:
    sequence:int
    observed:torch.Tensor
    command:torch.Tensor
    elapsed:torch.Tensor
    image_sha256:bytes
    state:ClockValueState
    forecast:dict


class VisualValueCache(EntityInferenceCache):
    """Reuse an exact latest-event retry only when numeric AND image inputs agree.

Image hash is32 bytes, not a retained camera history. The caller explicitly owns
entity IDs, sequence, preprocessing/model revisions and stream retirement. Normal
weight/buffer mutations invalidate all cached states; `.data` mutation remains
unsupported. This is research inference, not authoritative world-state execution.
"""
    @torch.no_grad()
    def step(self,entity,sequence,observed,known_command,*,elapsed,image):
        if self.model.training or self._identity!=self._model_identity():
            self._entries.clear();raise RuntimeError('model changed; construct a new versioned inference cache')
        if not isinstance(entity,str) or not entity or type(sequence)!=int or sequence<0:raise ValueError('explicit entity and nonnegative integer sequence required')
        reference=next(self.model.parameters())
        for value,shape in [(observed,(1,14)),(known_command,(1,14)),(elapsed,(1,1)),(image,(1,1,32,32))]:
            if value.shape!=shape or value.dtype!=reference.dtype or value.device!=reference.device or not torch.isfinite(value).all():raise ValueError('finite inputs matching model shape/device/dtype required')
        if not (elapsed>0).all() or (image<0).any() or (image>1).any():raise ValueError('positive elapsed and gray image in[0,1] required')
        pixel_hash=hashlib.sha256(image.detach().cpu().contiguous().numpy().tobytes()).digest()
        previous=self._entries.get(entity)
        if previous and sequence==previous.sequence:
            if pixel_hash!=previous.image_sha256 or not all(torch.equal(a,b) for a,b in [(observed,previous.observed),(known_command,previous.command),(elapsed,previous.elapsed)]):raise ValueError('conflicting retry; reset and replay corrected history')
            return _clone(previous.forecast)
        if sequence!=(previous.sequence+1 if previous else 0):raise ValueError('gap or revision; reset and replay verified prefix')
        if previous is None and len(self._entries)>=self.max_entities:raise RuntimeError('explicitly retire a stream before adding another')
        forecast,state=self.model.step(observed,known_command,None if previous is None else previous.state,elapsed=elapsed,images=image)
        state=ClockValueState(*(v.detach().clone() for v in state))
        self._entries[entity]=_VisualEntry(sequence,observed.detach().clone(),known_command.detach().clone(),elapsed.detach().clone(),pixel_hash,state,_clone(forecast))
        return _clone(forecast)


class PhysicalVisualGraph(torch.nn.Module):
    """Pure recorded-units → fixed preprocessing → native graph → recorded units.

Normative original preprocessing is float32, followed by float64 reference
inference. Caller supplies explicit normalized ClockValueState values. This
adds no learned weights, mutable cache, authority or future observations.
"""
    def __init__(self,model):
        super().__init__()
        if next(model.parameters()).dtype!=torch.float64:raise ValueError('normative reference model must be float64')
        self.model=model

    def forward(self,state_qpos,known_reference_qpos,elapsed,pixels,state_values):
        from .reusable_execution import mixture_quantiles
        x=((state_qpos.float()-self.model.state_mean.float())/self.model.state_scale.float()).double()
        a=((known_reference_qpos.float()-self.model.action_mean.float())/self.model.action_scale.float()).double()
        images=(pixels.float()/255).double()
        raw,state=self.model.visual_tensor_graph(x,a,elapsed.float().double(),images,ClockValueState(*state_values),last_only=True)
        q=mixture_quantiles(raw)
        return tuple(v*self.model.state_scale+self.model.state_mean for v in q)+(raw['motion_logit'].sigmoid(),*tuple(state))
