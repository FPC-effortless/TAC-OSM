"""BYTE100K-001: untrained language-only adapter over unchanged native R2.

Fixed 99,050 unique parameters, tied byte readout, existing FP64 memory option.
Tool/field/accept heads are parameter-free placeholders, not hidden capacity.
No training, checkpoint promotion, learned recall or mathematical qualification.
"""
import torch
from torch import nn

from tac_osm.csl125_r2 import CSL125R2, CSL125R2Config
from tac_osm.csl125_memory_precision import enable_precise_memory_
from tac_osm.causal_byte_corpus import (
    model_input_values,incoming_byte_state,outgoing_byte_state,
)
from tac_osm.state_ingress import validate_restored_stream_state

PARAMETERS=99050


def byte100k_config():
    return CSL125R2Config(vocab_size=256,width=48,layers=4,ff_width=96,
        role_count=4,state_width=8,conv_kernel=3,chunk_size=32,rungs=(4,),
        attention_layers=(2,4),query_heads=4,kv_heads=2,attention_window=128,
        max_parameters=105000)


class ByteLanguage100K(nn.Module):
    """One final language exit; no tool/risk/action API or new recurrence law."""
    def __init__(self):
        super().__init__()
        self.core=CSL125R2(byte100k_config())
        # Preserve the original core forward/state code. Its extra readouts are
        # removed as learned modules and never exposed by this language API.
        self.core.tool_head=nn.Identity()
        self.core.field_head=nn.Identity()
        self.core.accept_head=nn.Identity()
        enable_precise_memory_(self.core)
        self.validate_live_count()

    def validate_live_count(self):
        trained=sum(p.numel() for p in self.parameters() if p.requires_grad)
        deployed=sum(p.numel() for p in self.parameters())
        if (trained,deployed)!=(PARAMETERS,PARAMETERS) or not 95000<=trained<=105000:
            raise ValueError('BYTE100K-001 actual active/deployed parameter gate')
        return {'trainable':trained,'deployed':deployed}

    def forward(self,tokens,*,state=None,stream_ids=None):
        self.validate_live_count()
        if tokens.device.type!='cpu':
            raise ValueError('BYTE100K-001 CPU reference only; GPU/device qualification OPEN')
        outputs,next_state=self.core(tokens,depth=4,state=state,stream_ids=stream_ids)
        return outputs[4]['token_logits'],next_state


def predict_byte_chunk(model,chunk,previous=None):
    """Prediction BEFORE constructing targets; exact #731 ownership handoff.

    Preserves state/AD graph. Explicit boundary validator is read-only, not state
    source authentication. Whole graph correctness still needs source checks.
    """
    if type(model) is not ByteLanguage100K:
        raise TypeError('registered BYTE100K-001 model required')
    values=model_input_values(chunk)
    state=incoming_byte_state(chunk,previous)
    if state is not None:
        validate_restored_stream_state(model.core,state,depth=4,
            stream_ids=values['stream_ids'],observed_tokens=chunk.input_offset)
    tokens=torch.tensor([values['tokens']],dtype=torch.long)
    logits,next_state=model(tokens,state=state,stream_ids=values['stream_ids'])
    return logits,outgoing_byte_state(chunk,next_state)


def byte_language_loss(logits,targets):
    """Identical mean next-byte CE, explicit FP64 objective arithmetic.

    The preregistered amendment addresses value-reduction precision at the fixed
    independent1e-6gate, not historical FP32 memory-derivative failures.
    """
    if logits.ndim!=3 or logits.shape[-1]!=256 or targets.shape!=logits.shape[:-1] or targets.dtype!=torch.long:
        raise ValueError('BYTE100K-001 byte target layout')
    return torch.nn.functional.cross_entropy(logits.double().reshape(-1,256),targets.reshape(-1))
