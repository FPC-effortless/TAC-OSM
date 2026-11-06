"""CSL-125-R4: model-conditioned typed tool-call proposals (NO FIT).

One R3 backbone, unchanged parameter geometry and state law. Its tied token
output learns tool names and arguments from schemas rendered into context.
Byte-level JSON syntax is constrained at decode time and strict schema checks
reject invalid calls. The model is UNTRAINED and cannot execute tools.
"""
from __future__ import annotations

from math import isfinite
from typing import Dict, Optional, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from tac_osm.csl125_r3 import CSL125R3, CSL125R3Config
from tac_osm.csl125_tool_protocol import (
    ByteFallbackCodec, ToolContractError, ToolProposal, ToolSpec,
    build_tool_prompt, validate_tool_proposal,
)
from tac_osm.csl125_tool_grammar import (
    _HEADER, allowed_byte_ids, prefix_state,
)


class CSL125R4(CSL125R3):
    """Tool caller using the existing language head; zero added parameters.

    No fixed tool-index classifier: registered names and argument schemas
    appear in the prompt, allowing a different catalogue per request. This
    is an interface and training objective, NOT evidence that unseen tools,
    argument extraction, abstention or multi-call reasoning have been learned.
    """

    def __init__(self, config: CSL125R3Config = CSL125R3Config(), *, device=None):
        super().__init__(config, device=device)

    def tool_sft_loss(self, tokens: Tensor, target_mask: Tensor, *,
                      weights: Optional[Dict[int, float]] = None,
                      correction_weight: float = 0.0) -> Tensor:
        """Compute causal output-only next-byte tool JSON loss.

        A contiguous suffix of next-token labels must be selected per row;
        prompts, tool schemas and any prior observations receive NO loss.
        Caller must provide source-qualified approved examples and reset
        independent documents. No extra optimizer/training is performed here.
        """
        if tokens.ndim != 2 or tokens.dtype != torch.long or tokens.shape[1] < 2:
            raise ToolContractError("tokens must be [B,T>=2] long integers")
        if target_mask.dtype != torch.bool or target_mask.shape != tokens.shape[:1] + (tokens.shape[1]-1,):
            raise ToolContractError("target_mask must match next-token positions")
        if not isfinite(correction_weight) or not 0 <= correction_weight <= 1:
            raise ToolContractError("invalid correction_weight")
        if weights is None:
            weights = {r: 1.0 for r in self.config.rungs}
        if (set(weights) != set(self.config.rungs) or
            any(type(v) not in (float, int) or not isfinite(v) or v < 0
                for v in weights.values()) or sum(weights.values()) <= 0):
            raise ToolContractError("invalid exit weights")
        # The mask must be one contiguous suffix per example. Forbid hidden
        # zero islands that would silently train across packed documents.
        for row in target_mask:
            selected = torch.nonzero(row, as_tuple=False).flatten()
            if selected.numel() < 1 or not torch.all(row[selected[0]:]):
                raise ToolContractError("each example needs nonempty suffix target span")
            if int(selected[0]) < 1:
                raise ToolContractError("every example needs a nonempty prompt")
        outputs, _ = self.forward(tokens, all_rungs=True,
                                  emit_repair_logits=correction_weight > 0)
        targets = tokens[:, 1:]
        total = tokens.new_zeros((), dtype=torch.float32)
        denom = sum(weights.values())
        for rung, weight in weights.items():
            if weight == 0:
                continue
            base = outputs[rung]["token_logits"][:, :-1]
            loss = F.cross_entropy(base.transpose(1, 2), targets,
                                   reduction="none")[target_mask].mean()
            if correction_weight > 0:
                rev = outputs[rung]["repair_logits"][:, :-1]
                revised = F.cross_entropy(rev.transpose(1, 2), targets,
                                          reduction="none")[target_mask].mean()
                loss = (1-correction_weight)*loss + correction_weight*revised
            total = total + weight/denom*loss
        return total

    @torch.inference_mode()
    def propose_tools(self, user_text: str, tools: Sequence[ToolSpec], *,
                      depth: Optional[int] = None,
                      max_new_bytes: int = 256) -> ToolProposal:
        """Reference byte decoder. NO tool execution and NO calibrated score.

        Force the JSON-envelope prefix and produce the remainder using actual
        model logits restricted to JSON-*syntax*-legal single-byte tokens.
        Validate full callable schema at the end. Output may remain invalid or
        incomplete because this model has never been trained. Not a device-
        optimized grammar engine: parsing allowed ASCII bytes costs CPU work.
        """
        if type(max_new_bytes) is not int or not 1 <= max_new_bytes <= 512:
            raise ToolContractError("max_new_bytes must be 1..512")
        d = self.config.layers if depth is None else depth
        if type(d) is not int or d not in self.config.rungs:
            raise ToolContractError("choose one registered inference exit")
        if self.config.vocab_size < 256:
            raise ToolContractError("byte fallback requires 256 reserved byte IDs")
        prompt = build_tool_prompt(user_text, tools)
        prefix = ByteFallbackCodec.encode(prompt + _HEADER)
        if len(prefix) + max_new_bytes > 8192:
            raise ToolContractError("prompt exceeds reference byte context limit")
        was_training = self.training
        if was_training:
            raise ToolContractError("reference proposal generation requires model.eval()")
        device = next(self.parameters()).device
        if device.type == 'meta':
            raise ToolContractError("meta model cannot produce inference logits")
        tokens = torch.tensor([prefix], dtype=torch.long, device=device)
        raw = _HEADER
        outputs, state = self.forward(tokens, depth=d, stream_ids=("r4-tool-proposal",))
        for _ in range(max_new_bytes):
            grammar = prefix_state(raw)
            if grammar.complete:
                break
            allowed = allowed_byte_ids(raw)
            if not allowed:
                return ToolProposal((), "invalid", "no JSON-syntax continuation", raw)
            logits = outputs[d]["token_logits"][0, -1]
            candidates = torch.as_tensor(allowed, dtype=torch.long, device=device)
            values = logits.index_select(0, candidates)
            if not torch.isfinite(values).all():
                return ToolProposal((), "invalid", "nonfinite model logits", raw)
            token = int(candidates[values.argmax()].item())
            raw += chr(token)
            if prefix_state(raw).complete:
                break
            step = torch.tensor([[token]], dtype=torch.long, device=device)
            outputs, state = self.forward(
                step, depth=d, state=state, stream_ids=("r4-tool-proposal",))
        if not prefix_state(raw).complete:
            return ToolProposal((), "incomplete", "bounded decode ended before valid JSON", raw)
        try:
            return validate_tool_proposal(raw, tools)
        except ToolContractError as exc:
            return ToolProposal((), "invalid", str(exc), raw)
