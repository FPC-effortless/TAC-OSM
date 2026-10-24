"""Caller-owned effects for the pure native clock/value tensor graph.

No learned tensors, observations or weights are modified by memoization.
"""
from dataclasses import dataclass
import torch
from .clock_value_world import ClockValueState
from .reusable_execution import EntityInferenceCache, _clone


@dataclass(frozen=True)
class _ClockEntry:
    sequence:int
    observed:torch.Tensor
    command:torch.Tensor
    elapsed:torch.Tensor
    state:ClockValueState
    forecast:dict


class ClockValueCache(EntityInferenceCache):
    """Explicit versioned per-entity state, including recorded elapsed time.

Exact latest-event retries return clones without recomputation. Events must be
contiguous; revised or missing history requires explicit reset and replay. This
is a research execution adapter, not an authorized Woyengi integration.
Inherited revision and tensor-version validation forbids reusing state after
normal model/buffer mutations. Mutation through `.data` is unsupported.
"""
    @torch.no_grad()
    def step(self,entity,sequence,observed,known_command,*,elapsed):
        if self.model.training or self._identity!=self._model_identity():
            self._entries.clear()
            raise RuntimeError('model changed; construct a new versioned inference cache')
        if not isinstance(entity,str) or not entity or type(sequence)!=int or sequence<0:
            raise ValueError('explicit entity and nonnegative integer sequence required')
        reference=next(self.model.parameters())
        for value,shape in [(observed,(1,14)),(known_command,(1,14)),(elapsed,(1,1))]:
            if value.shape!=shape or value.dtype!=reference.dtype or value.device!=reference.device or not torch.isfinite(value).all():
                raise ValueError('finite observations/commands/elapsed matching model device and dtype required')
        if not (elapsed>0).all():raise ValueError('positive observed capture elapsed required')
        previous=self._entries.get(entity)
        if previous and sequence==previous.sequence:
            if not all(torch.equal(a,b) for a,b in [(observed,previous.observed),(known_command,previous.command),(elapsed,previous.elapsed)]):
                raise ValueError('conflicting retry; reset and replay corrected history')
            return _clone(previous.forecast)
        if sequence!=(previous.sequence+1 if previous else 0):raise ValueError('gap or revision; reset and replay verified prefix')
        if previous is None and len(self._entries)>=self.max_entities:raise RuntimeError('explicitly retire a stream before adding another')
        forecast,state=self.model.step(observed,known_command,None if previous is None else previous.state,elapsed=elapsed)
        stored=ClockValueState(*(v.detach().clone() for v in state))
        self._entries[entity]=_ClockEntry(sequence,observed.detach().clone(),known_command.detach().clone(),elapsed.detach().clone(),stored,_clone(forecast))
        return _clone(forecast)
