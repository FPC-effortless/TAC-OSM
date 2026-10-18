"""Training-only numerical correction; deployed observation/control is unchanged.

Original DEV061 fitting source remains frozen for reproducibility. This class
makes the STE forward gate exact, preserving public timestamp identities used
by counterfactual targets. No additional policy, stop rule or input is added.
"""
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from .structured_memory import MemoryAgent


class ExactRetentionAgent(MemoryAgent):
    def training_state(self, data):
        d = self.config.dimension
        if not self.entity:
            raw = data['bank']
            occ = data['occupied']
            memory = torch.cat((self.encoder(raw) * occ[:, :, None], raw), -1)
            return (memory, occ)
        sequences = data['routed']
        counts = data['route_counts']
        b, m, t, _ = sequences.shape
        packed = pack_padded_sequence(sequences.reshape(b * m, t, 22), counts.reshape(-1).clamp_min(1).cpu(), batch_first=True, enforce_sorted=False)
        encoded, hidden = self.local(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=t)
        hidden = hidden[0].reshape(b, m, d)
        occ = (counts > 0).to(hidden.dtype)
        last = sequences.gather(2, (counts - 1).clamp_min(0)[:, :, None, None].expand(-1, -1, 1, 22)).squeeze(2)
        memory = torch.cat((hidden * occ[:, :, None], last), -1)
        memory = torch.cat((memory, torch.zeros(b, 16, d + 22, device=memory.device)), 1)
        used = torch.cat((occ, torch.zeros(b, 17, device=memory.device)), 1)
        events = data['signals']
        valid = data['signal_present']
        route = data['signal_route']
        rank = data['signal_rank']
        outputs = encoded.reshape(b, m, t, d)
        prefix = outputs[torch.arange(b)[:, None], route, rank]
        keep_logits = self.keep(torch.cat((events, prefix), -1)).squeeze(-1) if self.retention else torch.ones_like(valid, dtype=hidden.dtype) * 10
        if self.retention and bool(self.retention_ready):
            probability = keep_logits.sigmoid()
            hard = (keep_logits >= 0).to(probability.dtype) * valid
            gate = hard + (probability - probability.detach()) if self.training else hard
            gate = gate * valid
        else:
            gate = valid.to(hidden.dtype)
            hard = gate
        values = torch.cat((self.encoder(events), events), -1)
        counter = torch.zeros(b, dtype=torch.long)
        for index in range(events.shape[1]):
            slot = 16 + counter % 16
            select = F.one_hot(slot, 32).to(hidden.dtype) * gate[:, index, None]
            memory = memory * (1 - select[:, :, None]) + select[:, :, None] * values[:, index, None]
            used = torch.cat((torch.maximum(used[:, :32], select.detach()), used[:, 32:]), -1)
            counter = counter + (hard[:, index] > 0).long()
        used = torch.cat((used[:, :32], counter[:, None].to(hidden.dtype)), 1)
        self.training_diagnostics = {'keep_logits': keep_logits, 'events': events, 'valid': valid}
        return (memory, used)
