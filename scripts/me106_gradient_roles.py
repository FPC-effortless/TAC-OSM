"""Explicit point-priority training graph and asymmetric gradient projection.

This applies established multi-task gradient surgery to native TAC, not a new
probability law. The guarantee is for an exact-real raw gradient direction;
AdamW, clipping, decay and finite steps do not inherit a loss-descent theorem.
"""
import torch


def objective_parts(model, forecast, trace, batch, rawscale, coefficients):
    mask, truth = batch[7], batch[6]
    normalized = (torch.log1p(truth) - model.target_mean) / model.target_scale

    def avg(values):
        return torch.where(mask, values, torch.zeros_like(values)).sum() / mask.sum().clamp_min(1)

    uncertainty = model.uncertainty_forecast(forecast, trace)
    nll = model.nll(uncertainty, truth, mask)
    location_mse = avg((forecast.log_location - normalized).square())
    reconstruction_nll = model.reconstruction_loss(trace, batch[8], batch[9])
    point = avg((model.raw_quantiles(forecast)[..., 1] - truth).abs() / rawscale)
    lower, _, upper = model.raw_quantiles(uncertainty).unbind(-1)
    interval = avg((upper-lower + 20*(lower-truth).clamp_min(0)
                    + 20*(truth-upper).clamp_min(0)) / rawscale)
    total = (coefficients['point'] * point + nll + 0.1*location_mse
             + 0.3*reconstruction_nll + coefficients['interval']*interval)
    return total, dict(nll=nll, point=point, location_mse=location_mse,
                       reconstruction_nll=reconstruction_nll, interval=interval)


def project_gradient_lists(point, auxiliary):
    """Project only an opposing auxiliary component; preserve point direction.

    Inputs are dense same-shaped tensors (including explicit zeros for unused
    paths). FP64 aggregate dot products reduce cancellation; result tensors keep
    model dtype. No gradients are differentiated through this optimizer rule.
    """
    if len(point) != len(auxiliary) or not point:
        raise ValueError('nonempty matching gradient lists required')
    if any(p.shape != u.shape or p.device != u.device or p.dtype != u.dtype
           for p, u in zip(point, auxiliary)):
        raise ValueError('gradient shape/device/dtype mismatch')
    with torch.no_grad():
        dot = sum((p.double()*u.double()).sum() for p, u in zip(point, auxiliary))
        norm = sum(p.double().square().sum() for p in point)
        denominator = torch.where(norm > 0, norm, torch.ones_like(norm))
        coefficient = torch.minimum(dot, torch.zeros_like(dot)) / denominator
        projected = [(u.double() - coefficient * p.double()).to(p.dtype)
                     for p, u in zip(point, auxiliary)]
        combined = [p + u for p, u in zip(point, projected)]
        after = sum((p.double()*u.double()).sum() for p, u in zip(point, projected))
    return combined, dict(auxiliary_dot_before=dot, auxiliary_dot_after=after,
                          point_gradient_norm_squared=norm, projection_coefficient=coefficient)


def assign_priority_gradients(model, parts, point_weight, interval_weight):
    params = tuple(model.parameters())
    point_loss = point_weight * parts['point']
    auxiliary_loss = (parts['nll'] + 0.1*parts['location_mse']
                      + 0.3*parts['reconstruction_nll'] + interval_weight*parts['interval'])
    point = torch.autograd.grad(point_loss, params, retain_graph=True, allow_unused=True)
    auxiliary = torch.autograd.grad(auxiliary_loss, params, allow_unused=True)
    point = [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, point)]
    auxiliary = [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, auxiliary)]
    gradients, info = project_gradient_lists(point, auxiliary)
    for parameter, gradient in zip(params, gradients):
        parameter.grad = gradient
    return {name: float(value) for name, value in info.items()}
