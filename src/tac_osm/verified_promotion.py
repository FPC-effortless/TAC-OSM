"""Fail-closed PNDS/TAC-OSM provisional-to-persistent memory.
Requires a real externally authenticated verifier; these rules do NOT prove
that observed receipts are truthful or that trained decisions are safe.
Companion Lean contract: FPC-effortless/pnds-lean, PNDS/TACOSM.lean.
"""
from dataclasses import dataclass
from copy import deepcopy
from threading import RLock
from typing import Any, Callable

@dataclass(frozen=True)
class AttestedEvidence:
    receipt_id: str
    context_id: str
    authenticated: bool
    independent_context: bool
    utility: bool
    stability: bool
    integrity: bool
    cost: bool
    observed_success: bool

    def eligible(self) -> bool:
        return all(type(value) is bool and value for value in (
            self.authenticated, self.independent_context, self.utility,
            self.stability, self.integrity, self.cost, self.observed_success))

class VerifiedPromotion:
    """In-memory E/P/Q state transitions. Not a durable transaction store.

    Trusted verifier takes (candidate key, opaque receipt) and returns an
    AttestedEvidence. Caller-provided Boolean flags must never be trusted.
    Context strings alone do not prove genuine independence.
    """
    def __init__(self, verifier: Callable[[str, Any], AttestedEvidence] | None,
                 minimum_contexts: int = 2):
        if type(minimum_contexts) is not int or minimum_contexts < 2:
            raise ValueError("at least two distinct contexts required")
        self._verifier = verifier
        self.minimum_contexts = minimum_contexts
        self._provisional = {}
        self._persistent = {}
        self._quarantined = set()
        self._used_receipts = set()
        self._lock = RLock()

    def propose(self, key: str, value: Any) -> bool:
        if not isinstance(key,str) or not key: raise ValueError("nonempty key required")
        with self._lock:
            if key in self._persistent or key in self._quarantined or key in self._provisional:
                return False
            self._provisional[key] = {"payload":deepcopy(value),"contexts":set()}
            return True

    def observe(self, key: str, opaque_receipt: Any) -> bool:
        with self._lock:
            if key not in self._provisional:
                raise ValueError("no provisional item (or item already finalized)")
            if self._verifier is None:
                raise PermissionError("missing independent receipt verifier")
            evidence = self._verifier(key,opaque_receipt)
            if not isinstance(evidence,AttestedEvidence):
                raise PermissionError("verifier did not return a typed attestation")
            if not isinstance(evidence.receipt_id,str) or not evidence.receipt_id.strip():
                raise PermissionError("invalid receipt ID")
            if not isinstance(evidence.context_id,str) or not evidence.context_id.strip():
                raise PermissionError("invalid context ID")
            if evidence.receipt_id in self._used_receipts: raise ValueError("replayed receipt")
            if evidence.authenticated is not True:
                raise PermissionError("unauthenticated receipt")
            if evidence.context_id in self._provisional[key]["contexts"]:
                raise ValueError("repeated evidence context")
            self._used_receipts.add(evidence.receipt_id)
            if evidence.observed_success is not True:
                self._quarantined.add(key)
                del self._provisional[key]
                return False
            if not evidence.eligible():
                return False
            record=self._provisional[key]
            record["contexts"].add(evidence.context_id)
            if len(record["contexts"]) < self.minimum_contexts: return False
            self._persistent[key]=deepcopy(record["payload"])
            del self._provisional[key]
            return True

    def tier(self,key:str) -> str:
        with self._lock:
            if key in self._persistent: return "persistent"
            if key in self._quarantined: return "quarantined"
            if key in self._provisional: return "provisional"
            return "absent"

    def inspect_persistent(self,key:str)->Any:
        with self._lock:
            if key not in self._persistent: raise KeyError(key)
            return deepcopy(self._persistent[key])

    def snapshot(self)->dict:
        with self._lock:
            return {"pending":len(self._provisional),"persistent":len(self._persistent),
                    "quarantined":len(self._quarantined),
                    "consumed_receipts":len(self._used_receipts)}
