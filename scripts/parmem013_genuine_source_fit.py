"""PAR-MEM-013 genuine source historic-pair content discrimination, with
FULLY TRAINED identical-size context-only associative-READ-BLIND control.

Trained on already registered genuine 2014 FIT physical data only; original
May-Jun 2014 DEV is previously seen and not an independent capability test.
No 2015-17 publisher observations, no synthetic performance claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch

from scripts.partime002_real_source import SHA256
from scripts.partime005_continuous_real_training import (
    source_and_times, frozen, verify_originals, sha as sha_file
)
from scripts.parmem011_genuine_source_train import (
    neural, source_batch, LAGS, ANCHORS, DEV_ANCHORS,
    SEEDS, BATCH, ROOT
)
from scripts.parmem013_source_event_objective import (
    paired_real_source_predictions, loss_and_measures,
    recorded_source_pair_metrics,
)

STUDY="PAR-MEM-013-REAL-EVENT-SPECIFIC-CONTENT-DISCRIMINATION"
PROTO=ROOT/"research/parallel-six/PAR-MEM-013-PROTOCOL.json"
PLAN=ROOT/"research/parallel-six/PAR-MEM-013-FROZEN-PLAN.json"
OUT=ROOT/"research/replay-bundles/PAR-MEM-013"
ARMS=("aligned_associative_read","trained_context_only_read")
UPDATES=36
LR=.0005


def digest(value):
    return hashlib.sha256(
        json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def source(seed,archive):
    p=json.loads(PROTO.read_text())
    checks=dict(study_id=STUDY,source_original_SHA256=SHA256,
                arms=list(ARMS),seeds=list(SEEDS),
                source_training_steps_per_seed_arm=UPDATES,
                batch_source_stations=BATCH,
                training_lags=list(LAGS),
                source_anchor_options=list(ANCHORS),
                development_anchors=list(DEV_ANCHORS),
                full_source_history_length=129,
                parameters_each=98326,
                persistent_fp32_scalars_each=410)
    for key,val in checks.items():
        if p.get(key)!=val:raise RuntimeError("new source discrimination preregistration changed: "+key)
    data,fit,dev=source_and_times(archive)
    original=frozen(archive)
    receipt=verify_originals(original)
    if len(fit)!=12 or len(dev)!=12 or any(
        len(z)<=max(ANCHORS) for z in fit+dev):
        raise RuntimeError("12 real source station histories unavailable")
    return data,fit,dev,original,receipt


def training_order(seed,fit):
    rng=random.Random(seed+13013)
    out=[]
    for update in range(UPDATES):
        lag=LAGS[update%len(LAGS)]
        sites=rng.sample(range(12),BATCH)
        anchors=[rng.choice(ANCHORS) for _ in sites]
        out.append(dict(lag=lag,sites=sites,anchors=anchors,
            true_recorded_old_source_hours=[
                int(fit[s][a-lag]) for s,a in zip(sites,anchors)],
            next_recorded_wrong_source_hours=[
                int(fit[s][a-lag+1]) for s,a in zip(sites,anchors)],
            source_current_hours=[
                int(fit[s][a]) for s,a in zip(sites,anchors)]))
    return out


def make_plan(archive):
    if PLAN.exists() or OUT.exists():
        raise RuntimeError("original historical pair plan already exists; original research is immutable")
    _,fit,dev,old,receipt=source(0,archive)
    plan=dict(study_id=STUDY,source_SHA256=SHA256,
              old_real_pretrained_train_clock_SHA256=old["train_source_clock_SHA256"],
              old_real_pretrained_DEV_clock_SHA256=old["exposed_DEV_source_clock_SHA256"],
              training_schedule_by_seed={str(seed):training_order(seed,fit) for seed in SEEDS},
              published_seed_ids=list(SEEDS),registered_arms=list(ARMS),
              published_DEV_anchor_site_hours=[
                  [int(z[a]) for a in DEV_ANCHORS] for z in dev],
              registered_before_ANY_neural_training=True,
              target_label_difference_was_not_used_to_select_query_cases=True,
              original_source_DEV_previously_exposed=True,
              independent_real_source_TEST_unavailable=True)
    PLAN.write_text(json.dumps(plan,indent=2,sort_keys=True)+"\n")
    print("PAR_MEM_013_SOURCE_ORDER_BEFORE_FIT="+json.dumps({
        "original_source_SHA":SHA256,"seed_count":len(SEEDS),
        "real_fit_examples_per_seed_arm":UPDATES*BATCH,
        "source_orders_SHA":{str(s):digest(plan["training_schedule_by_seed"][str(s)])
                             for s in SEEDS},
        "optimizer_updates_completed":0},sort_keys=True),flush=True)


def fixed(archive):
    if not PLAN.exists():raise RuntimeError("genuine source pair plan must be committed before fit")
    data,fit,dev,old,receipt=source(0,archive)
    p=json.loads(PLAN.read_text())
    if (p["study_id"]!=STUDY or p["source_SHA256"]!=SHA256
        or p["old_real_pretrained_train_clock_SHA256"]!=old["train_source_clock_SHA256"]
        or p["old_real_pretrained_DEV_clock_SHA256"]!=old["exposed_DEV_source_clock_SHA256"]
        or p["training_schedule_by_seed"]!={str(seed):training_order(seed,fit) for seed in SEEDS}
        or p["published_seed_ids"]!=list(SEEDS)
        or p["registered_arms"]!=list(ARMS)
        or p["published_DEV_anchor_site_hours"]!=[
            [int(z[a]) for a in DEV_ANCHORS] for z in dev]):
        raise RuntimeError("original authentic source case sample order or clock changed")
    return p,data,fit,dev,receipt


def paired_source_batch(data,rows,sites,anchors,lag):
    b=source_batch(data,rows,sites,anchors,lag,
                   "correct_shared_source_key")
    # The negative target is a REAL PREVIOUSLY OBSERVED sensor value:
    # the directly adjacent original source event inside the same history.
    # This field is ONLY passed to the loss AFTER the historic read.
    b["paired_past"]=b["obs"][:,-lag,0].clone()
    if not torch.equal(b["past"],b["obs"][:,-1-lag,0]):
        raise RuntimeError("genuine source event A measurement/clock mismatch")
    if not (b["indexes"][:,-1-lag]<b["indexes"][:,-lag]).all().item():
        raise RuntimeError("unregistered genuine source previous event B chronology")
    return b


def make_model(seed,receipt):
    model=neural(seed,receipt)
    contract=model.contract()
    if (contract["learned_trainable_parameters"]!=98326
        or contract["total_fp32_per_stream"]!=410):
        raise RuntimeError("registered same-bounded-memory model configuration drift")
    return model


def fit(archive):
    if OUT.exists():raise RuntimeError("original trained model directory cannot be overwritten")
    p,data,fit,_,receipt=fixed(archive)
    OUT.mkdir(parents=True,exist_ok=False)
    result=[]
    for seed in SEEDS:
        for arm in ARMS:
            random.seed(seed+13013)
            model=make_model(seed,receipt)
            model.train()
            optimizer=torch.optim.AdamW(
                model.parameters(),lr=LR,weight_decay=.001,
                foreach=False,fused=False)
            initial={name:param.detach().clone()
                     for name,param in model.named_parameters()}
            loss_trace=[]; last_good=None
            t0=time.monotonic()
            for case in p["training_schedule_by_seed"][str(seed)]:
                b=paired_source_batch(
                    data,fit,case["sites"],case["anchors"],case["lag"])
                optimizer.zero_grad(set_to_none=True)
                output=paired_real_source_predictions(
                    model,b,case["lag"],
                    associative_read=(arm==ARMS[0]))
                loss,details=loss_and_measures(output,b)
                loss.backward()
                if any(v.grad is None or not torch.isfinite(v.grad).all().item()
                       for v in model.parameters()):
                    raise RuntimeError("trained source model has disconnected/dummy/nonfinite weights")
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                loss_trace.append(dict(
                    total=float(loss.detach()),
                    pair_content=float(details["huber"].detach()),
                    factual_discrimination=float(details["discrimination"].detach()),
                    next_hour=float(details["future_source_physical"].detach())))
            if not all(not torch.equal(initial[name],param.detach())
                       for name,param in model.named_parameters()):
                raise RuntimeError("one or more model weights were not updated by original fit")
            path=OUT/f"seed-{seed}-{arm}-original.pt"
            torch.save(dict(
                study_id=STUDY,seed=seed,arm=arm,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                torch_rng_state=torch.get_rng_state(),
                python_rng_state=random.getstate(),
                source_SHA256=SHA256,
                registered_fit_case_order_SHA256=digest(
                    p["training_schedule_by_seed"][str(seed)]),
                trained_updates=UPDATES,
                active_learned_parameters=98326,
                source_stream_FP32_scalars=410,
                model_code_SHA256=sha_file(
                    ROOT/"src/tac_osm/variants/source_aligned_parallel_world.py"),
                objective_code_SHA256=sha_file(
                    ROOT/"scripts/parmem013_source_event_objective.py"),
                runner_code_SHA256=sha_file(
                    ROOT/"scripts/parmem013_genuine_source_fit.py"),
                exposed_DEV_unopened_before_training=True,
                initial_seed_original_backbone_preserved=True,
                train_loss_trace=loss_trace),path)
            item=dict(seed=seed,arm=arm,
                original_checkpoint_SHA256=sha_file(path),
                training_steps=UPDATES,
                true_observed_source_examples=UPDATES*BATCH,
                first_loss=loss_trace[0]["total"],
                final_loss=loss_trace[-1]["total"],
                final_rank_loss=loss_trace[-1]["factual_discrimination"],
                CPU_wall_seconds_NOT_budget=time.monotonic()-t0)
            result.append(item)
            print("PAR_MEM_013_REAL_PAIRED_FIT="+json.dumps(item,sort_keys=True),flush=True)
    (OUT/"TRAIN_RECEIPT.json").write_text(json.dumps(dict(
        study_id=STUDY,seed_count=len(SEEDS),original_model_count=len(result),
        source_SHA256=SHA256,
        frozen_original_case_order_SHA256=digest(p["training_schedule_by_seed"]),
        original_10_full_checkpoint_and_optimizer_weights_frozen_before_DEV=True,
        source_DEV_previously_exposed_not_independent=True,
        full_backward_optimizer_mathematical_FLOP_not_certified=True,
        original_model_receipts=result),indent=2,sort_keys=True)+"\n")


def originals(plan):
    receipt=json.loads((OUT/"TRAIN_RECEIPT.json").read_text())
    if (receipt["study_id"]!=STUDY or receipt["original_model_count"]!=10
        or receipt["frozen_original_case_order_SHA256"]!=digest(
            plan["training_schedule_by_seed"])
        or sorted((x["seed"],x["arm"]) for x in receipt["original_model_receipts"])!=
            sorted((seed,arm) for seed in SEEDS for arm in ARMS)):
        raise RuntimeError("ten original source-trained paired models not frozen")
    for row in receipt["original_model_receipts"]:
        path=OUT/f"seed-{row['seed']}-{row['arm']}-original.pt"
        if sha_file(path)!=row["original_checkpoint_SHA256"]:
            raise RuntimeError("source trained checkpoint byte hash drift")
        saved=torch.load(path,map_location="cpu",weights_only=False)
        if (saved["source_SHA256"]!=SHA256 or saved["trained_updates"]!=UPDATES
            or saved["source_stream_FP32_scalars"]!=410
            or saved["active_learned_parameters"]!=98326
            or saved["registered_fit_case_order_SHA256"]!=digest(
                plan["training_schedule_by_seed"][str(row["seed"])])
            or saved["model_code_SHA256"]!=sha_file(
                ROOT/"src/tac_osm/variants/source_aligned_parallel_world.py")
            or saved["objective_code_SHA256"]!=sha_file(
                ROOT/"scripts/parmem013_source_event_objective.py")
            or saved["runner_code_SHA256"]!=sha_file(
                ROOT/"scripts/parmem013_genuine_source_fit.py")):
            raise RuntimeError("saved paired neural model and optimizer provenance violated")
    return receipt


def evaluate(archive,*,replay):
    p,data,_,dev,old=fixed(archive)
    receipt=originals(p)
    outpath=OUT/"DEV_RESULTS.json"
    if outpath.exists()!=replay:
        raise RuntimeError("original DEV results may be scored ONCE, replay thereafter")
    all_rows=[]
    for seed in SEEDS:
        for arm in ARMS:
            checkpoint=OUT/f"seed-{seed}-{arm}-original.pt"
            saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
            model=make_model(seed,old)
            model.load_state_dict(saved["model_state"],strict=True)
            model.eval()
            collected={key:[] for key in (
                "pa","pb","ya","yb","source_current","future_pred",
                "future_true","future_valid","read_diff","context_diff")}
            source_ids=[];source_lags=[]
            with torch.no_grad():
                for anchor in DEV_ANCHORS:
                    for lag in LAGS:
                        b=paired_source_batch(
                            data,dev,list(range(12)),[anchor]*12,lag)
                        y=paired_real_source_predictions(
                            model,b,lag,associative_read=(arm==ARMS[0]))
                        parts=dict(pa=y["prediction_a"],pb=y["prediction_b"],
                          ya=b["past"],yb=b["paired_past"],
                          source_current=b["obs"][:,-1,0],
                          future_pred=y["physical_prediction"][:,0],
                          future_true=b["next"][:,0],
                          future_valid=b["next_valid"][:,0],
                          read_diff=y["read_difference_l2"],
                          context_diff=y["context_difference_l2"])
                        for key,val in parts.items():
                            collected[key].append(val.detach())
                        source_ids.extend((site,anchor,lag) for site in range(12))
                        source_lags.extend([lag]*12)
            tensors={key:torch.cat(val) for key,val in collected.items()}
            tensors["source_case_identity"]=torch.tensor(source_ids,dtype=torch.long)
            tensors["lag"]=torch.tensor(source_lags,dtype=torch.long)
            archive_file=OUT/f"seed-{seed}-{arm}-DEV_original.pt"
            if replay:
                reference=torch.load(archive_file,map_location="cpu",weights_only=True)
                if set(reference)!=set(tensors):
                    raise RuntimeError("original paired real source prediction names changed")
                for key in tensors:
                    torch.testing.assert_close(
                        reference[key],tensors[key],rtol=2e-5,atol=2e-6)
            else:torch.save(tensors,archive_file)
            scores={}
            for lag in LAGS:
                ix=tensors["lag"]==lag
                metric=recorded_source_pair_metrics(
                    tensors["pa"][ix],tensors["pb"][ix],
                    tensors["ya"][ix],tensors["yb"][ix])
                future=ix & tensors["future_valid"]
                metric.update(dict(
                    source_observed_pair_count=int(ix.sum()),
                    weak_current_value_copy_old_A_MAE=float(
                        (tensors["source_current"][ix]-tensors["ya"][ix]).abs().mean()),
                    original_true_next_PM25_cases=int(future.sum()),
                    next_source_PM25_normalized_MAE=float((
                        tensors["future_pred"][future]-
                        tensors["future_true"][future]).abs().mean()),
                    original_query_read_vector_separation_L2=float(
                        tensors["read_diff"][ix].mean()),
                    source_context_only_query_vector_separation_L2=float(
                        tensors["context_diff"][ix].mean())))
                if metric["source_observed_pair_count"]!=120:
                    raise RuntimeError("precommitted source observed DEV query count changed")
                scores[str(lag)]=metric
            item=dict(seed=seed,arm=arm,
                original_checkpoint_SHA256=sha_file(checkpoint),
                original_true_source_lag_results=scores)
            all_rows.append(item)
            print("PAR_MEM_013_ORIGINAL_REAL_DEV="+json.dumps(item,sort_keys=True),flush=True)
    means={}
    for arm in ARMS:
        subset=[r for r in all_rows if r["arm"]==arm]
        means[arm]={}
        for lag in LAGS:
            means[arm][str(lag)]={}
            for key in (
                "absolute_pair_MAE","source_target_A_MAE",
                "swapped_pairs_error_minus_true_pairs_error",
                "valid_pair_target_order_accuracy",
                "historical_source_target_gap_abs_mean",
                "output_query_difference_abs_mean",
                "next_source_PM25_normalized_MAE"):
                vals=[r["original_true_source_lag_results"][str(lag)][key]
                      for r in subset]
                means[arm][str(lag)][key]=sum(vals)/len(vals)
    result=dict(
        study_id=STUDY,
        original_publisher_source_SHA256=SHA256,
        genuine_source_full_neural_models_trained=10,
        previous_2014_DEV_is_exposed_not_independent=True,
        no_zero_source_label_leakage_by_query_argument=True,
        matched_parameters_and_training_schedule=True,
        full_mathematical_FLOP_parity_not_certified=True,
        no_qualified_model_superiority_or_event_specific_recall=True,
        all_original_five_seed_true_source_records=all_rows,
        five_seed_average_by_arm=means)
    if replay:
        original=json.loads(outpath.read_text())
        def close(x,y,path="root"):
            if type(x)!=type(y):raise RuntimeError("source comparison schema mismatch "+path)
            if isinstance(x,dict):
                if set(x)!=set(y):raise RuntimeError("source comparison keys drift "+path)
                for key in x:close(x[key],y[key],path+"."+str(key))
            elif isinstance(x,list):
                if len(x)!=len(y):raise RuntimeError("source seed count changed")
                for i,(a,b) in enumerate(zip(x,y)):close(a,b,path+f".{i}")
            elif isinstance(x,float):
                if abs(x-y)>1e-5*max(1,abs(x)):
                    raise RuntimeError("original physical source numeric score changed "+path)
            elif x!=y:raise RuntimeError("original source SHA / source case identity changed "+path)
        close(original,result)
        print("PAR_MEM_013_INDEPENDENT_ORIGINAL_FULL_NEURAL_PAIR_REPLAY_PASS")
    else:outpath.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--mode",choices=("plan","fit","eval","replay"),required=True)
    args=ap.parse_args()
    torch.set_num_threads(1)
    if args.mode=="plan":make_plan(args.source)
    elif args.mode=="fit":fit(args.source)
    else:evaluate(args.source,replay=(args.mode=="replay"))
