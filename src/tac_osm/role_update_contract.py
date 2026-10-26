"""Pure proposed-displacement guard. Does not run or replace an optimizer.

Use AFTER forming the complete candidate displacement (including rates,
preconditioning, momentum, clipping and decay), BEFORE any parameter mutation.
This is a candidate for future registered training, not already-trained ME106.
Only a computed first-order point-gradient dot is constrained; finite-step loss,
other objectives, generalization and exact IEEE-to-real refinement stay OPEN.
"""
import torch


def protect_displacement(point_gradients, proposed, blocks):
    if not point_gradients or len(point_gradients) != len(proposed):
        raise ValueError('nonempty matching gradient/displacement lists required')
    blocks = tuple(tuple(block) for block in blocks)
    indices = [i for block in blocks for i in block]
    if any(type(i) is not int for i in indices) or sorted(indices) != list(range(len(proposed))) or any(not b for b in blocks):
        raise ValueError('blocks must partition every parameter exactly once')
    for g, d in zip(point_gradients, proposed):
        if g.shape != d.shape or g.dtype != d.dtype or g.device != d.device or not g.is_floating_point():
            raise ValueError('matching floating tensor shape/dtype/device required')
        if not torch.isfinite(g).all() or not torch.isfinite(d).all():
            raise ValueError('nonfinite gradient or displacement')
    result, receipt = [d.detach().clone() for d in proposed], []
    with torch.no_grad():
        for block in blocks:
            norm = sum(point_gradients[i].double().square().sum() for i in block)
            before = sum((point_gradients[i].double()*proposed[i].double()).sum() for i in block)
            if not torch.isfinite(norm) or not torch.isfinite(before):
                raise ValueError('nonfinite FP64 dot/norm')
            active = bool(norm > 0 and before > 0)
            if active:
                for i in block:
                    result[i] = (proposed[i].double() - before/norm*point_gradients[i].double()).to(proposed[i].dtype)
            after = sum((point_gradients[i].double()*result[i].double()).sum() for i in block)
            fallback = bool(after > 0)
            if fallback:
                # Quantization may reintroduce opposition. Reject this block's
                # proposed move on protected coordinates instead of silently
                # claiming a bound. Keep truly zero-gradient coordinates free.
                for i in block:
                    result[i] = torch.where(point_gradients[i] == 0, proposed[i], torch.zeros_like(proposed[i]))
                after = torch.zeros_like(after)
            if not torch.isfinite(after) or not all(torch.isfinite(result[i]).all() for i in block):
                raise ValueError('nonfinite protected displacement')
            receipt.append(dict(before=float(before), after=float(after),
                                projected=active, rounding_fallback=fallback))
    return result, receipt
