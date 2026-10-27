"""PAR-MEM-013 full-network source-event discrimination objective.

Historical targets are genuinely previously observed source measurements but
never passed to query keys, encoder or readout as additional inputs.
The matched context-only control is end-to-end trained and never reads
the EXTRA aligned associative matrix; original TAC backbone memory remains.
Both models execute identical operator shapes and use the same 98326 params.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F
from tac_osm.variants.source_aligned_parallel_world import SourceAlignedParallelWorld


def context_only_source_read(model:SourceAlignedParallelWorld, embedding,
                             source_current_ordinal, source_query_ordinal):
    if (embedding.ndim!=2 or embedding.shape[-1]!=48 or
        source_current_ordinal.shape!=embedding.shape[:1] or
        source_query_ordinal.shape!=embedding.shape[:1]):
        raise ValueError("source contextual historical query geometry changed")
    key_now,_=model._keys(source_current_ordinal,embedding.dtype)
    key_then,phi=model._keys(source_query_ordinal,embedding.dtype)
    similarity=(key_now*key_then).sum(-1)
    head_logit=(model.read_router(phi) +
                model.write_router(embedding) +
                model.retention(embedding) +
                model.write_strength(embedding))
    weights=head_logit.softmax(-1)
    known_context=(weights*similarity).sum(-1)
    value=model.source_content(embedding).tanh()
    return value*known_context.unsqueeze(-1)


def paired_real_source_predictions(model, batch, lag:int, *,
                                   associative_read:bool):
    if not isinstance(model,SourceAlignedParallelWorld):
        raise ValueError("wrong original source trained model")
    ordinals=batch["indexes"]
    if ordinals.ndim!=2 or ordinals.shape[1]!=129 or lag not in (16,64,128):
        raise ValueError("source-observed history / lag unregistered")
    if not (ordinals[:,1:]>ordinals[:,:-1]).all().item():
        raise ValueError("source clock not strictly increasing")
    current=ordinals[:,-1]
    older=ordinals[:,-1-lag]
    neighboring=ordinals[:,-lag]
    if not ((older<neighboring).all().item() and
            (neighboring<current).all().item()):
        raise ValueError("legitimate prior source event identity absent")
    query_base=(ordinals-1).clamp_min(0)
    query_base[:,-1]=older
    no_future_mask=torch.ones_like(batch["dt"],dtype=torch.bool)
    original_forecast,trace=model(
        batch["obs"],batch["action"],elapsed_dt=batch["dt"],
        write_mask=no_future_mask,source_ids=batch["owners"],
        event_index=ordinals,query_event_index=query_base,return_trace=True)
    memory=trace["aligned_before_write"][:,-1]
    embedding=trace["embedding"][:,-1]
    read_a,read_route_a=model._read(memory,older)
    read_b,read_route_b=model._read(memory,neighboring)
    context_a=context_only_source_read(model,embedding,current,older)
    context_b=context_only_source_read(model,embedding,current,neighboring)
    mix=float(bool(associative_read))
    selected_a=read_a*mix+context_a*(1.0-mix)
    selected_b=read_b*mix+context_b*(1.0-mix)
    physical=model.backbone(
        batch["obs"],batch["action"],
        elapsed_dt=batch["dt"],write_mask=no_future_mask)[:,-1]
    forecast=model._predictive_correction(physical,selected_a)
    prediction_a=model.historical_content(selected_a)[:,0]
    prediction_b=model.historical_content(selected_b)[:,0]
    return dict(prediction_a=prediction_a,prediction_b=prediction_b,
                physical_prediction=forecast,
                read_difference_l2=(read_a-read_b).norm(dim=-1),
                context_difference_l2=(context_a-context_b).norm(dim=-1),
                trace_original_forecast=original_forecast[:,-1],
                read_route_a=read_route_a,read_route_b=read_route_b)


def loss_and_measures(out,batch):
    target_a=batch["past"]
    target_b=batch["paired_past"]
    pa=out["prediction_a"]
    pb=out["prediction_b"]
    if not (target_a.shape==target_b.shape==pa.shape==pb.shape):
        raise RuntimeError("true and paired historic source target shapes mismatch")
    huber=(F.smooth_l1_loss(pa,target_a)+
           F.smooth_l1_loss(pb,target_b))/2
    gap=target_a-target_b
    pred_gap=pa-pb
    weights=gap.abs().clamp(max=1.0)
    discriminative=(F.softplus(0.1-pred_gap*gap/0.25)*weights).mean() / (
        weights.mean()+1e-6)
    from scripts.partime002_real_source import score
    physical=score(out["physical_prediction"],batch["next"],
                   batch["next_valid"])
    total=huber+0.35*discriminative+0.15*physical
    if not torch.isfinite(total).item():
        raise RuntimeError("nonfinite genuine source discrimination training objective")
    return total,dict(huber=huber,discrimination=discriminative,
                      future_source_physical=physical)


def recorded_source_pair_metrics(pa,pb,ya,yb):
    abs_true=(pa-ya).abs()+(pb-yb).abs()
    abs_swapped=(pa-yb).abs()+(pb-ya).abs()
    valid=(ya!=yb)
    correct=((pa-pb)*(ya-yb)>0)
    if valid.any().item():
        accuracy=correct[valid].float().mean()
    else:accuracy=pa.new_tensor(float("nan"))
    return dict(
        absolute_pair_MAE=float(abs_true.mean()/2),
        source_target_A_MAE=float((pa-ya).abs().mean()),
        swapped_pairs_error_minus_true_pairs_error=float(
            (abs_swapped-abs_true).mean()/2),
        source_observed_unequal_target_pairs=int(valid.sum()),
        valid_pair_target_order_accuracy=float(accuracy),
        historical_source_target_gap_abs_mean=float((ya-yb).abs().mean()),
        output_query_difference_abs_mean=float((pa-pb).abs().mean()))
