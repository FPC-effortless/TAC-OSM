"""SENSE-TOK-005: causal learned discrete token identities plus byte fallback.

NO FIT / NO OPTIMIZER. Distinct from SENSE-TOK-004, where a completed TOKEN
always used a continuous pooled vector. Here a completed multi-byte patch
maps to one learned codebook index; that ID's embedding enters the recurrent
model, while a one-byte patch remains an exact byte embedding.

The controller still uses previous observed bytes only, and the local byte
head remains active inside incomplete patches (hierarchical byte decoder).
Hard code selection uses a straight-through soft assignment during backward.
No evidence of learned tokens, useful coding rates, compute savings or skill.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import sqrt, log, isfinite

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .adaptive_byte_token import AdaptiveByteTokenGRU, AdaptiveAudit, RepresentationEvent, EMBED, HIDDEN, Mode, Force


@dataclass(frozen=True)
class TokenCodeEvent:
    end: int
    available_after: int
    code_id: int
    payload: bytes


@dataclass(frozen=True)
class CodebookAudit:
    base: AdaptiveAudit
    codes: tuple[TokenCodeEvent, ...]
    n_codes: int
    intervention_at: int | None = None
    intervention_mode: Force | None = None

    def verify(self, observed: bytes) -> None:
        self.base.verify(observed)
        if self.intervention_at is not None:
            if self.intervention_mode not in ('byte', 'token'):
                raise ValueError('intervention action missing')
            if self.base.decisions[self.intervention_at - 1] != self.intervention_mode.upper():
                raise ValueError('intervention was not applied')
        expected = [e for e in self.base.events if e.mode == 'TOKEN']
        if len(expected) != len(self.codes):
            raise ValueError('missing code identity for committed token')
        for e, c in zip(expected, self.codes):
            if (e.end, e.available_after, e.payload) != (c.end, c.available_after, c.payload):
                raise ValueError('token code committed before its bytes were observed')
            if not 0 <= c.code_id < self.n_codes:
                raise ValueError('out-of-range token identity')

    @property
    def recurrent_updates(self) -> int:
        return self.base.recurrent_updates

    @property
    def active_code_count(self) -> int:
        return len({c.code_id for c in self.codes})

    @property
    def dominant_code_share(self) -> float | None:
        if not self.codes:
            return None
        counts = {}
        for c in self.codes:
            counts[c.code_id] = counts.get(c.code_id, 0) + 1
        return max(counts.values()) / len(self.codes)


class AdaptiveCodebookByteTokenGRU(AdaptiveByteTokenGRU):
    """100752 + code_count * 32 learned weights (8 codes => 101008).

    BYTE: update global recurrent state with the exact individual byte.
    TOKEN: once all bytes are observed, assign a learned discrete code ID
    and update global recurrent state with that code's embedding.

    The byte-local decoder sees observed bytes inside unfinished tokens; this
    is necessary for causal next-byte predictions but is NOT evidence of
    end-to-end compute reduction or byte-cache elimination.
    """
    def __init__(self, *, code_count: int = 8, max_token_bytes: int = 8,
                 threshold: float = .5, temperature: float = 1.,
                 code_temperature: float = 1.) -> None:
        if not isinstance(code_count, int) or isinstance(code_count, bool) or not 2 <= code_count <= 16:
            raise ValueError('code_count must be integer 2..16; learned size parity is checked separately')
        if type(code_temperature) not in (float,int) or not isfinite(code_temperature) or not 0 < code_temperature <= 100:
            raise ValueError('code_temperature must be positive')
        super().__init__(max_token_bytes=max_token_bytes,threshold=threshold,temperature=temperature)
        self.code_count = code_count
        self.code_temperature = code_temperature
        self.token_codes = nn.Embedding(code_count, EMBED)

    def _assignment(self, items: list[Tensor]) -> tuple[Tensor,int]:
        if len(items) < 2:
            raise ValueError('a discrete TOKEN ID requires 2+ observed bytes')
        pooled = self._pool(items)
        # Cosine similarity is bounded and prevents embedding-norm selection.
        z = F.normalize(pooled, p=2, dim=-1)
        keys = F.normalize(self.token_codes.weight, p=2, dim=-1)
        scores = z @ keys.T / (sqrt(EMBED) * self.code_temperature)
        soft = torch.softmax(scores, dim=-1)
        index = int(scores.detach().argmax(dim=-1).item())
        hard = F.one_hot(torch.tensor([index], device=scores.device), num_classes=self.code_count).to(scores.dtype)
        weights = hard + (soft - soft.detach())
        return weights @ self.token_codes.weight, index

    def _code_rep(self, items: list[Tensor]) -> Tensor:
        if len(items) == 1:
            return items[0]
        return self._assignment(items)[0]

    def forward(self, prev_ids: Tensor, *, mode: Mode = 'inference',
                force: Force | None = None, audit: bool = False,
                intervention_at: int | None = None,
                intervention_mode: Force | None = None):
        """Optional ONE-STEP action intervention for source-controlled probes.

        Decision time `intervention_at` is 1-based: after x[t-1] is observed.
        The evaluator, never the model, chooses the intervention schedule.
        No answer, target, gold structure or future byte is passed to the gate.
        """
        if mode not in ('inference', 'surrogate'):
            raise ValueError('unknown execution mode')
        if force not in (None,'byte','token'):
            raise ValueError('invalid forced mode')
        if not isinstance(prev_ids,Tensor) or prev_ids.dtype!=torch.long or prev_ids.ndim!=2:
            raise ValueError('previous IDs must be int64 [batch,time]')
        if prev_ids.shape[1] and not bool(torch.all(prev_ids[:,0]==256)):
            raise ValueError('must have BOS at zero')
        if prev_ids.shape[1]>1 and not bool(torch.all((prev_ids[:,1:]>=0)&(prev_ids[:,1:]<256))):
            raise ValueError('noninitial IDs must be observed bytes')
        if intervention_at is not None:
            if type(intervention_at) is not int or not 1 <= intervention_at < prev_ids.shape[1]:
                raise ValueError('intervention time must be a valid observed-byte decision index')
            if intervention_mode not in ('byte','token') or force is not None:
                raise ValueError('one-step intervention requires mode and unforced policy')
        elif intervention_mode is not None:
            raise ValueError('intervention action supplied without a time')
        logits=[]
        audits=[]
        proxies=[]
        for ids in prev_ids:
            out, receipt, cost=self._one(ids,mode=mode,force=force,
                                         intervention_at=intervention_at,
                                         intervention_mode=intervention_mode)
            logits.append(out)
            proxies.append(cost)
            if audit:
                codes=[]
                for e in receipt.events:
                    if e.mode=='TOKEN':
                        # Debug-only ID replay. Never enters the output logits.
                        with torch.no_grad():
                            seq=self.byte_emb(torch.tensor(list(e.payload),dtype=torch.long,device=out.device))
                            _,idx=self._assignment([row.unsqueeze(0) for row in seq.unbind(0)])
                        codes.append(TokenCodeEvent(e.end,e.available_after,idx,e.payload))
                item=CodebookAudit(receipt,tuple(codes),self.code_count,
                                   intervention_at,intervention_mode)
                item.verify(b''.join(e.payload for e in receipt.events)+receipt.pending)
                audits.append(item)
        output=(torch.stack(logits,dim=0) if logits else
                self.byte_head.weight.new_empty((0,prev_ids.shape[1],256)))
        proxy=(torch.stack(proxies).mean() if proxies else self.byte_head.weight.sum()*0)
        return (output,tuple(audits),proxy) if audit else output

    def _one(self, ids: Tensor, *, mode: Mode, force: Force | None,
             intervention_at: int | None, intervention_mode: Force | None):
        h = self.byte_emb.weight.new_zeros((1, HIDDEN))
        pending: list[Tensor] = []
        pending_bytes: list[int] = []
        predictions: list[Tensor] = []
        choices: list[str] = []
        probabilities: list[float] = []
        events: list[RepresentationEvent] = []
        proxy_cost = h.new_zeros(())
        actual_updates = 0
        candidates = 0
        observed = 0

        def event(payload: bytes, *, forced: bool = False) -> None:
            nonlocal actual_updates
            if not payload:
                raise AssertionError('empty representation')
            end = observed
            # In BYTE branch, a pending token is committed before current byte;
            # the current observed byte may therefore be in a separate event.
            if events:
                start = events[-1].end
            else:
                start = 0
            if start != end - len(payload):
                raise AssertionError('representation endpoints invalid')
            events.append(RepresentationEvent(
                start=start, end=end, available_after=observed,
                mode='BYTE' if len(payload) == 1 else 'TOKEN',
                payload=payload, forced=forced,
            ))
            actual_updates += 1

        for i, idx in enumerate(ids.unbind(0)):
            current = self.byte_emb(idx.view(1))
            if i:
                # x[i-1] has now been observed. x[i] remains unavailable.
                b = int(idx.item())
                observed += 1
                prev_logits = predictions[-1].detach()
                logp = torch.log_softmax(prev_logits, dim=-1)
                entropy = -(logp.exp() * logp).sum() / log(256)
                local = (self._pool(pending) if pending else torch.zeros_like(current))
                features = torch.cat((h, current, local, entropy.reshape(1, 1)), dim=-1)
                prob = torch.sigmoid(self.mode_gate(features) / self.temperature).reshape(())
                decision_force = (intervention_mode if observed == intervention_at else force)
                do_byte = (decision_force == 'byte') or (decision_force is None and bool((prob >= self.threshold).detach().item()))
                probabilities.append(float(prob.detach().item()))
                choices.append('BYTE' if do_byte else 'TOKEN')
                must_finish = (len(pending) + 1) >= self.max_token_bytes
                expected_byte_cost = 1 + int(bool(pending))
                decision_prob = (prob if decision_force is None else
                                 (torch.ones_like(prob) if do_byte else torch.zeros_like(prob)))
                proxy_cost = proxy_cost + decision_prob * expected_byte_cost + (1-decision_prob) * int(must_finish)
                if mode == 'surrogate':
                    # Both alternatives computed for gradient attribution.
                    if pending:
                        prev_rep = self._code_rep(pending)
                        preh = self.gru_cell(prev_rep, h)
                        candidates += 1
                    else:
                        preh = h
                    byte_h = self.gru_cell(current, preh)
                    candidates += 1
                    if must_finish:
                        token_rep = self._code_rep(pending + [current])
                        token_h = self.gru_cell(token_rep, h)
                        candidates += 1
                    else:
                        token_h = h
                    forward_h = byte_h if do_byte else token_h
                    # Exactly equal to selected forward branch. Gate gradients
                    # receive downstream loss via nonzero surrogate derivative.
                    if decision_force is None:
                        h = forward_h + (prob - prob.detach()) * (byte_h - token_h)
                    else:
                        h = forward_h
                if do_byte:
                    if pending:
                        if mode == 'inference':
                            h = self.gru_cell(self._code_rep(pending), h)
                            candidates += 1
                        # Pending token ends BEFORE current byte. At its
                        # reveal time one further byte has been observed.
                        old = bytes(pending_bytes)
                        old_end = observed - 1
                        last = events[-1].end if events else 0
                        if last + len(old) != old_end:
                            raise AssertionError('pending token offset mismatch')
                        events.append(RepresentationEvent(
                            start=last, end=old_end, available_after=observed,
                            mode='BYTE' if len(old) == 1 else 'TOKEN', payload=old))
                        actual_updates += 1
                        pending.clear()
                        pending_bytes.clear()
                    if mode == 'inference':
                        h = self.gru_cell(current, h)
                        candidates += 1
                    event(bytes([b]))
                else:
                    pending.append(current)
                    pending_bytes.append(b)
                    if must_finish:
                        if mode == 'inference':
                            h = self.gru_cell(self._code_rep(pending), h)
                            candidates += 1
                        event(bytes(pending_bytes), forced=True)
                        pending.clear()
                        pending_bytes.clear()
            local = self._pool(pending) if pending else torch.zeros_like(current)
            predictions.append(self.byte_head(torch.cat((h, current, local), dim=-1)).reshape(256))
        result = (torch.stack(predictions, 0) if predictions else
                  self.byte_head.weight.new_empty((0, 256)))
        receipt = AdaptiveAudit(
            observed_bytes=observed, decisions=tuple(choices),
            byte_probabilities=tuple(probabilities), events=tuple(events),
            pending=bytes(pending_bytes), recurrent_updates=actual_updates,
            candidate_recurrent_calls=candidates,
            byte_readouts=len(predictions), eos_used=False,
        )
        receipt.verify(bytes(ids[1:].detach().tolist()))
        return result, receipt, proxy_cost / max(1, observed)


def codebook_parameter_receipt(model: AdaptiveCodebookByteTokenGRU) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    deployed = sum(p.numel() for p in model.parameters())
    reference = 100752
    return {'trainable_parameters': trainable, 'deployed_learned_parameters': deployed,
            'codebook_parameters': model.token_codes.weight.numel(),
            'reference_s004_parameters': reference,
            'relative_gap_s004': abs(trainable-reference)/max(trainable,reference),
            'within_100k_band': 95000 <= trainable <= 105000 and 95000 <= deployed <= 105000,
            'within_1pct_s004': abs(trainable-reference)/max(trainable,reference) <= .01,
            'training_authorized': False, 'trained_codebook': False,
            'subword_vocabulary_learned': False}
