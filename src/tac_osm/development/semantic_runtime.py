"""Inference-only batching of the unchanged learned admission gate and FIFO."""
import torch


@torch.no_grad()
def observe_many(model, state, events, present=None):
    if events.ndim != 3 or events.shape[0] != len(state['keys']) or events.shape[-1] != 46:
        raise ValueError('Expected batch x records x46 public events')
    if present is None:
        present = torch.ones(events.shape[:2], dtype=torch.bool, device=events.device)
    if present.shape != events.shape[:2] or present.dtype != torch.bool:
        raise ValueError('Boolean batch x records mask required')
    if not torch.isfinite(events[present]).all():
        raise ValueError('Finite public records required')
    source = events[:, :, :6]
    typed = ((source == 0) | (source == 1)).all(-1) & (source.sum(-1) == 1)
    if not (typed | ~present).all():
        raise ValueError('The optimized adapter requires one-hot source6 types')
    if events.shape[1] == 0:
        return state
    # This is the actual trained gate, not a hand-authored source lookup.
    take = (model.gate(source).squeeze(-1) >= 0) & present
    if not take.any():
        return state
    stored, valid = state['events'].clone(), state['present'].clone()
    for row in take.any(-1).nonzero(as_tuple=True)[0].tolist():
        combined = torch.cat((state['events'][row, state['present'][row]], events[row, take[row]]))[-64:]
        stored[row].zero_(); valid[row].zero_()
        stored[row, :len(combined)] = combined; valid[row, :len(combined)] = True
    return dict(state, events=stored, present=valid)
