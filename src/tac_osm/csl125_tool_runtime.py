"""Read-only R4 inference adapter: same weights/state law, restricted output work.

No model architecture claim, trained calibration, external tool call or memory
commit. Validity of a generated call still does not make it the correct call.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from torch.nn import functional as F
from tac_osm.csl125 import CSL125State
from tac_osm.csl125_tool_protocol import ByteFallbackCodec, ToolContractError, ToolProposal, build_tool_prompt
from tac_osm.csl125_tool_schema import CompiledToolSchema


@torch.inference_mode()
def last_features(model, tokens, *, depth, state=None, stream_id='schema-request'):
    """Original R4 layer ordering; omit all auxiliary/vocabulary projections.

    State layout and weights are inherited unchanged. Correspondence to the
    parent full-vocabulary path is tested at all registered exits. Single
    request only: caller owns state; switching depth requires reset/replay.
    """
    cfg = model.config
    if model.training or type(depth) is not int or depth not in cfg.rungs:
        raise ToolContractError('eval mode and registered depth required')
    if tokens.dtype != torch.long or tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 1:
        raise ToolContractError('tokens must be one nonempty long-integer stream')
    if type(stream_id) is not str or not stream_id:
        raise ToolContractError('explicit stream identity required')
    h = model.embedding(tokens)
    ids = (stream_id,)
    if state is None:
        state = model.initial_state(1, tokens.device, h.dtype, depth=depth, stream_ids=ids)
    elif state.stream_ids != ids:
        raise ToolContractError('carried state ownership mismatch')
    if len(state.layers) != depth or len(state.attention) != depth:
        raise ToolContractError('depth change requires reset/replay')
    layers, caches = list(state.layers), list(state.attention)
    for i in range(depth):
        h, layers[i] = model.blocks[i](h, layers[i])
        key = str(i + 1)
        if key in model.attention:
            if caches[i] is None:
                raise ToolContractError('missing attention cache')
            h, caches[i] = model.attention[key](h, caches[i])
    z = model.final_norm(h[:, -1:])
    return z, CSL125State(tuple(layers), tuple(caches), ids)


@torch.inference_mode()
def legal_byte_logits(model, last, ids):
    if last.ndim != 3 or last.shape[:2] != (1, 1):
        raise ToolContractError('projection requires a single final feature vector')
    if ids.ndim != 1 or ids.dtype != torch.long or ids.numel() < 1 or ids.numel() > 256:
        raise ToolContractError('bounded byte candidate IDs required')
    if bool((ids < 0).any()) or bool((ids >= 256).any()):
        raise ToolContractError('candidate IDs must belong to the byte codec')
    rows = model.embedding.weight.index_select(0, ids.to(model.embedding.weight.device))
    return F.linear(last, rows)[0, 0]


@dataclass(frozen=True)
class ToolDecodeReceipt:
    proposal: ToolProposal
    input_bytes: int
    generated_bytes: int
    choice_steps: int
    forced_bytes: int
    elapsed_seconds: float
    grammar_fingerprint: str
    trained_quality: str = 'NOT_MEASURED'


@torch.inference_mode()
def propose_schema_tools(model, user_text, compiled: CompiledToolSchema, *,
                         depth=None, max_new_bytes=512, prefill_chunk=128,
                         prior_observations=()):
    """Actual neural choices plus schema constraints, never oracle answer values.

    The full vocabulary projection for every prefill position is avoided.
    The byte codec is mandatory; this is NOT interchangeable with future BPE.
    Fixed-depth fresh request; observations are explicit caller-provided data.
    """
    if model.training:
        raise ToolContractError('model.eval() is required')
    if model.config.vocab_size < 256:
        raise ToolContractError('256-byte codec requires at least 256 token rows')
    if type(max_new_bytes) is not int or not 1 <= max_new_bytes <= 2048:
        raise ToolContractError('max_new_bytes must be in [1,2048]')
    if type(prefill_chunk) is not int or not 1 <= prefill_chunk <= 512:
        raise ToolContractError('prefill_chunk must be in [1,512]')
    d = model.config.layers if depth is None else depth
    if type(d) is not int or d not in model.config.rungs:
        raise ToolContractError('unregistered depth')
    device = model.embedding.weight.device
    if device.type == 'meta':
        raise ToolContractError('meta weights cannot perform inference')
    tools = compiled.tools
    prompt = ByteFallbackCodec.encode(build_tool_prompt(user_text, tools, prior_observations))
    if len(prompt) + max_new_bytes > 8192:
        raise ToolContractError('bounded byte context exceeded')
    start = time.perf_counter()
    state = None
    for begin in range(0, len(prompt), prefill_chunk):
        token = torch.tensor([prompt[begin:begin + prefill_chunk]], dtype=torch.long, device=device)
        z, state = last_features(model, token, depth=d, state=state)
    session = compiled.session()
    choices = forced = 0
    reason = 'bounded decode ended before a complete schema'
    for _ in range(max_new_bytes):
        allowed = session.allowed()
        if allowed.numel() == 0:
            reason = 'no legal schema continuation'
            break
        if allowed.numel() == 1:
            token_id = int(allowed[0])
            forced += 1
        else:
            logits = legal_byte_logits(model, z, allowed)
            if not torch.isfinite(logits).all():
                reason = 'nonfinite candidate logits'
                break
            token_id = int(allowed[int(logits.argmax())])
            choices += 1
        session.accept(token_id)
        if session.complete:
            break
        token = torch.tensor([[token_id]], dtype=torch.long, device=device)
        z, state = last_features(model, token, depth=d, state=state)
    raw = bytes(session.bytes).decode('utf-8', errors='backslashreplace')
    if session.complete:
        try:
            proposal = session.validate()
        except ToolContractError as exc:
            proposal = ToolProposal((), 'invalid', str(exc), raw)
    else:
        proposal = ToolProposal((), 'incomplete', reason, raw)
    return ToolDecodeReceipt(proposal, len(prompt), len(session.bytes), choices, forced,
                             time.perf_counter() - start, compiled.fingerprint)
