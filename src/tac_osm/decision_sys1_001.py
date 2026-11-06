"""DECISION-SYS1-001: optional, untrained residual-correction candidate.

Inherits the exact native observation/prediction graph. Historical source and
checkpoint keys are unchanged. A zero residual is a no-op for latent correction;
confirmatory evidence logging/calibration is a separate caller responsibility.
No novelty, calibration, long-term safety or trained-utility claim is made.
"""
import torch
from .predictive_state_agent import Belief, PredictiveStateAgent


class SurpriseGatedPredictiveStateAgent(PredictiveStateAgent):
    def verify(self, state: Belief, executed_action: torch.Tensor,
               outcome: torch.Tensor, predicted_outcome: torch.Tensor,
               available: torch.Tensor) -> Belief:
        """Correct only on finite, available, nonzero residuals.

        Entire outcome vectors are available or unavailable in this interface.
        Callers must authenticate arrival time, entity/checkpoint identity and
        exactly-once processing. This method cannot establish those facts.
        """
        b = self._validate(state)
        if (executed_action.shape != (b, self.action_size)
                or outcome.shape != (b, self.outcome_size)
                or predicted_outcome.shape != outcome.shape
                or available.shape != (b, 1)):
            raise ValueError("verification shape mismatch")
        if available.dtype != torch.bool:
            raise ValueError("availability must be boolean")
        tensors = (*state, executed_action, outcome, predicted_outcome)
        if any(t.device != state.fast.device for t in (*tensors, available)):
            raise ValueError("all tensors must be on the state device")
        if any(t.dtype != state.fast.dtype for t in tensors):
            raise ValueError("all numeric tensors must share state dtype")
        if any(not bool(torch.isfinite(t).all()) for t in state):
            raise ValueError("state must be finite")
        for tensor in (executed_action, outcome, predicted_outcome):
            if not bool(torch.isfinite(tensor[available.expand_as(tensor)]).all()):
                raise ValueError("available feedback must be finite")
        # Sanitize before learned operations: masking only their output can
        # preserve the forward value while poisoning backward with 0 * NaN.
        safe_action = torch.where(available, executed_action, torch.zeros_like(executed_action))
        safe_y = torch.where(available, outcome, torch.zeros_like(outcome))
        safe_p = torch.where(available, predicted_outcome.detach(), torch.zeros_like(predicted_outcome))
        residual = torch.tanh(safe_y - safe_p)
        surprise = residual.square().mean(-1, keepdim=True)
        active = available & (surprise > 0)
        action = torch.tanh(self.action(safe_action))
        joint = torch.cat((state.fast, state.slow), -1)
        proposal, gate = self.verify_map(torch.cat((joint, action, residual), -1)).chunk(2, -1)
        delta = .25 * surprise * torch.sigmoid(gate) * torch.tanh(proposal)
        corrected = torch.where(active, joint + delta, joint)
        key = self.address(self.verify_key(action))
        value = torch.tanh(self.verify_value(residual))
        error = value - (state.memory @ key.unsqueeze(-1)).squeeze(-1)
        update = .25 * surprise.unsqueeze(-1) * error.unsqueeze(-1) * key.unsqueeze(-2)
        matrix = torch.where(active.unsqueeze(-1), state.memory + update, state.memory)
        return Belief(corrected[:, :32], corrected[:, 32:], matrix)


class NativeRiskStateModel(torch.nn.Module):
    """Exact no-fit risk graph: native recurrence + four interval hazards.

    No tokenizer, action-policy head, feedback update, CASM, learned entity router
    or attention over a history. The caller supplies 212 causal features.
    Four interval hazards define five ordered classes, not calibrated confidence.
    With context_size=212 and head_width=301: 99,963 learned parameters.
    """
    initial = PredictiveStateAgent.initial
    _validate = PredictiveStateAgent._validate
    address = staticmethod(PredictiveStateAgent.address)
    observe = PredictiveStateAgent.observe

    def __init__(self, context_size: int = 212, head_width: int = 301):
        super().__init__()
        if context_size < 1 or head_width < 1:
            raise ValueError("positive dimensions required")
        # Reuse the actual inherited operators, not an imitation of their math.
        parent = PredictiveStateAgent(context_size, 1, 1)
        self.context_size = context_size
        for name in ('observation', 'enc', 'transition', 'read_fast', 'gate',
                     'slow', 'slow_gate', 'query', 'key', 'value', 'write', 'retention'):
            setattr(self, name, getattr(parent, name))
        self.risk_head = torch.nn.Sequential(
            torch.nn.Linear(context_size + 72, head_width), torch.nn.SiLU(),
            torch.nn.Linear(head_width, 4))

    def predict(self, state: Belief, context: torch.Tensor):
        """Return four hazard logits and five normalized log probabilities.

        Class order: farthest interval/tail first, then progressively nearer
        intervals. Exact publisher cutoff semantics require source qualification.
        No action-effect or counterfactual quantity is predicted.
        """
        b = self._validate(state)
        if context.shape != (b, self.context_size):
            raise ValueError("context shape mismatch")
        joint = torch.cat((state.fast, state.slow), -1)
        q = self.address(self.query(joint))
        recalled = (state.memory @ q.unsqueeze(-1)).squeeze(-1)
        logits = self.risk_head(torch.cat((context, joint, recalled), -1))
        return logits, self.log_probabilities(logits)

    @staticmethod
    def log_probabilities(logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[-1] != 4:
            raise ValueError("four interval hazard logits required")
        log_survive = torch.nn.functional.logsigmoid(-logits)
        prefix = torch.cat((torch.zeros_like(log_survive[:, :1]),
                            log_survive.cumsum(-1)[:, :-1]), -1)
        near_first = prefix + torch.nn.functional.logsigmoid(logits)
        tail = log_survive.sum(-1, keepdim=True)
        return torch.cat((tail, near_first.flip(-1)), -1)
