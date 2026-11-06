"""Schema-aware byte decoding for existing CSL125R4, not a new neural model.

Reuses XGrammar. No fabricated tool values, optimizer, corpus or executor.
Caller-owned schemas are snapshotted and final validation stays independent.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Sequence

import torch
from tac_osm.csl125_tool_protocol import (
    ByteFallbackCodec, ToolContractError, ToolSpec, ToolSupervision,
    build_tool_prompt, catalogue, validate_tool_proposal,
)


def snapshot_tools(tools: Sequence[ToolSpec]) -> tuple[ToolSpec, ...]:
    if not isinstance(tools, (tuple, list)) or not tools:
        raise ToolContractError('a nonempty validated tool catalogue is required')
    try:
        data = json.loads(json.dumps([asdict(t) for t in tools], allow_nan=False))
        return catalogue(data)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ToolContractError('invalid or nonfinite tool catalogue') from exc


def _schema(node: dict) -> dict:
    """Materialize R4's implicit bounds and canonical field order."""
    out = {k: v for k, v in node.items() if k != 'description'}
    typ = out['type']
    if typ == 'object':
        out['properties'] = {k: _schema(v) for k, v in sorted(out.get('properties', {}).items())}
        out['required'] = sorted(out.get('required', []))
        out['additionalProperties'] = False
    elif typ == 'array':
        out['items'] = _schema(out['items'])
        out.setdefault('minItems', 0)
        out.setdefault('maxItems', 8)
    elif typ == 'string':
        out.setdefault('minLength', 0)
        out.setdefault('maxLength', 256)
    return out


def tool_response_schema(tools: Sequence[ToolSpec], max_calls: int = 4) -> dict:
    if type(max_calls) is not int or not 0 <= max_calls <= 4:
        raise ToolContractError('max_calls must be an integer in [0,4]')
    specs = snapshot_tools(tools)
    calls = [{
        'type': 'object',
        # Name first fixes the argument schema BEFORE arguments are generated.
        'properties': {'name': {'const': t.name}, 'arguments': _schema(t.parameters)},
        'required': ['name', 'arguments'], 'additionalProperties': False,
    } for t in specs]
    return {'type': 'object', 'properties': {'function_calls': {
        'type': 'array', 'items': {'anyOf': calls}, 'minItems': 0, 'maxItems': max_calls,
    }}, 'required': ['function_calls'], 'additionalProperties': False}


def _ordered(value, schema):
    if schema['type'] == 'object':
        return {k: _ordered(value[k], v) for k, v in sorted(schema.get('properties', {}).items()) if k in value}
    if schema['type'] == 'array':
        return [_ordered(v, schema['items']) for v in value]
    return value


def schema_target(calls, tools: Sequence[ToolSpec], max_calls: int = 4) -> str:
    """Gold serialization aligned with fixed-order decode, never input to inference."""
    specs = snapshot_tools(tools)
    checked = validate_tool_proposal({'function_calls': list(calls)}, specs, max_calls)
    by_name = {t.name: t for t in specs}
    rows = [{'name': c['name'], 'arguments': _ordered(c['arguments'], by_name[c['name']].parameters)}
            for c in checked.function_calls]
    return json.dumps({'function_calls': rows}, ensure_ascii=True, allow_nan=False, separators=(',', ':'))


def prepare_schema_supervision(user_text, tools, gold_calls, *, prior_observations=(), max_context_bytes=8192):
    specs = snapshot_tools(tools)
    prompt = build_tool_prompt(user_text, specs, prior_observations)
    label = schema_target(gold_calls, specs)
    p, y = ByteFallbackCodec.encode(prompt), ByteFallbackCodec.encode(label)
    if type(max_context_bytes) is not int or not 2 <= max_context_bytes <= 8192 or len(p) + len(y) > max_context_bytes:
        raise ToolContractError('example exceeds bounded byte context')
    return ToolSupervision(p + y, (False,) * (len(p)-1) + (True,) * len(y), len(p), len(y))


class CompiledToolSchema:
    """Reusable compiled grammar; each request owns a separate matcher.

    256 byte tokens are the explicit R4 research codec, NOT a fitted BPE.
    Grammar validity does not establish argument truth or execution authority.
    XGrammar constraint coverage is tested; final R4 validation is mandatory.
    """
    def __init__(self, tools: Sequence[ToolSpec], *, max_calls: int = 4):
        try:
            import xgrammar as xgr
        except ImportError as exc:
            raise RuntimeError('schema runtime requires the pinned xgrammar optional dependency') from exc
        self._xgr = xgr
        self._tools = snapshot_tools(tools)
        self.max_calls = max_calls
        self.schema = tool_response_schema(self._tools, max_calls)
        self.fingerprint = hashlib.sha256(json.dumps(self.schema, sort_keys=True).encode()).hexdigest()
        info = xgr.TokenizerInfo([bytes([i]) for i in range(256)], xgr.VocabType.RAW,
                                 vocab_size=256, stop_token_ids=[])
        compiler = xgr.GrammarCompiler(info, max_threads=1)
        # any_order=True may permit omitted/duplicate keys; deliberately disabled.
        self._compiled = compiler.compile_json_schema(
            self.schema, any_whitespace=False, separators=(',', ':'), strict_mode=True,
            any_order=False)

    @property
    def tools(self):
        return snapshot_tools(self._tools)

    def session(self):
        return ToolSchemaSession(self)


class ToolSchemaSession:
    def __init__(self, compiled: CompiledToolSchema):
        self.compiled = compiled
        self.matcher = compiled._xgr.GrammarMatcher(compiled._compiled, terminate_without_stop_token=True)
        self.mask = compiled._xgr.allocate_token_bitmask(1, 256)
        self.bytes = bytearray()

    @property
    def complete(self):
        return self.matcher.is_terminated()

    def allowed(self) -> torch.Tensor:
        if self.complete:
            return torch.empty(0, dtype=torch.long)
        self.matcher.fill_next_token_bitmask(self.mask)
        # Decode the documented packed int32 bitmask, including bit 31.
        ids = torch.arange(256, dtype=torch.long)
        words = self.mask[0].to(torch.int64)[ids // 32]
        return ids[((words >> (ids % 32)) & 1).bool()]

    def accept(self, token: int) -> None:
        if type(token) is not int or not 0 <= token < 256 or self.complete:
            raise ToolContractError('invalid byte or already completed schema')
        if len(self.bytes) >= 8192 or not self.matcher.accept_token(token):
            raise ToolContractError('byte violates schema or output limit')
        self.bytes.append(token)

    def validate(self):
        if not self.complete:
            raise ToolContractError('incomplete schema-constrained output')
        raw = ByteFallbackCodec.decode(self.bytes)
        return validate_tool_proposal(raw, self.compiled.tools, self.compiled.max_calls)
