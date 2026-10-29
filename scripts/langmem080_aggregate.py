#!/usr/bin/env python3
"""LANGMEM080 exact publisher source original historical Q-write trace aggregation.

No model training. Strict original owners, SHA, same registered anchor/lag,
both native lanes, three actual original checkpoint seeds and non-zero sham.
"""
from __future__ import annotations
import argparse,json,math,statistics
from pathlib import Path
SEEDS=(1701,1702,1703)
ANCHORS=(32,64,96,128,192)
DELAYS=(1,8,32,64,128,192)
PAIRS=tuple((a,d) for a in ANCHORS for d in DELAYS if a+d<320)
LANES=("fast","slow")
EXPECTED_ROWS=4*len(PAIRS)*len(LANES)

def collect(root:Path):
    files=list(root.rglob("EVENT_RESULTS.json"))
    if len(files)!=3:raise ValueError("exactly three source checkpoint source-event original receipts required")
    records={}
    signatures={}
    for path in files:
        r=json.loads(path.read_text())
        seed=r.get("seed")
        if seed not in SEEDS or seed in records:raise ValueError("duplicate/unregistered trained original source seed")
        if (r.get("study")!="LANGMEM080" or r.get("no_new_fits") is not True
            or r.get("gpu_minutes")!=0 or r.get("protected_TEST")!="NOT_ACCESSED"
            or r.get("parameters")!=499627 or r.get("state_fp32_bytes_per_owner")!=4160
            or r.get("position_mode")!="cap_all_POST_EXPOSED_DEV"
            or tuple(r.get("source_anchors",[]))!=ANCHORS or tuple(r.get("source_delays",[]))!=DELAYS
            or r.get("registered_source_pairs_per_owner")!=len(PAIRS)
            or r.get("source_target_pairs_with_both_lanes")!=EXPECTED_ROWS
            or r.get("original_stream_steps_verified")!=1280
            or r.get("zero_negative_control_bitwise_count")!=8):
            raise ValueError("source registered experimental protocol changed")
        for kind in ("original_checkpoint_sha256","original_fit_sample_order_sha256",
                     "source_target_fingerprint_sha256","source_observed_anchor_fingerprint_sha256"):
            v=r.get(kind)
            if not isinstance(v,str) or len(v)!=64 or not all(c in "0123456789abcdef" for c in v):
                raise ValueError("actual original source/checkpoint fingerprint missing")
        for f in ("max_original_source_affine_update_abs_error",
                  "max_original_source_event_transport_abs_error",
                  "max_original_source_read_projection_abs_error"):
            v=r.get(f)
            if not isinstance(v,(int,float)) or not math.isfinite(v) or v>1e-5 or v<0:
                raise ValueError("trained source one-event affine or read correspondence not passed")
        raw=r.get("rows")
        if not isinstance(raw,list) or len(raw)!=EXPECTED_ROWS:
            raise ValueError("incomplete original genuine source event observations")
        ids=[(x["owner"],x["anchor_clock"],x["lag"],x["lane"]) for x in raw]
        owners=set(x[0] for x in ids)
        if len(owners)!=4 or set(r.get("original_owners",[]))!=owners:
            raise ValueError("owner identity mismatch")
        needed={(o,a,d,lane) for o in owners for a,d in PAIRS for lane in LANES}
        if set(ids)!=needed or len(ids)!=len(set(ids)):
            raise ValueError("wrong original source anchor/lag/lane or duplicate")
        for x in raw:
            if (x["query_clock"]!=x["anchor_clock"]+x["lag"] or
                type(x["historical_observed_source_byte"])!=int or
                not 0<=x["historical_observed_source_byte"]<=255 or
                not 0<=x["source_next_byte_target"]<=255 or
                not math.isfinite(x["without_event_minus_full_nll"]) or
                not math.isfinite(x["direct_old_event_read_l2"]) or
                not 0<=x["surviving_write_frobenius_ratio"]<=x["survival_source_alpha_bound"]+1e-4):
                raise ValueError("real original source observed event or measured event contribution illegal")
        sign=tuple((x["owner"],x["anchor_clock"],x["query_clock"],x["lane"],
                    x["historical_observed_source_byte"],x["historical_predecessor_source_byte"],
                    x["source_next_byte_target"]) for x in raw)
        signatures[seed]=sign
        records[seed]=r
    if set(records)!=set(SEEDS):
        raise ValueError("original trained three seed matrix incomplete")
    if len(set(x["source_target_fingerprint_sha256"] for x in records.values()))!=1:
        raise ValueError("source real next-byte target/anchor time fingerprint changed")
    if len(set(x["source_observed_anchor_fingerprint_sha256"] for x in records.values()))!=1:
        raise ValueError("source historical observed byte/write anchors changed")
    if not (signatures[1701]==signatures[1702]==signatures[1703]):
        raise ValueError("source original actual byte/event/label order differs across models")
    group={}
    for lane in LANES:
        by_lag={}
        for lag in DELAYS:
            rows=[x for r in records.values() for x in r["rows"]
                  if x["lane"]==lane and x["lag"]==lag]
            count=4*3*sum((a,lag) in PAIRS for a in ANCHORS)
            if len(rows)!=count:raise ValueError("source per-lag original cohort changed")
            def avg(field):return statistics.mean(x[field] for x in rows)
            by_lag[str(lag)]={
               "source_event_count":count,
               "mean_write_survival_norm_ratio":avg("surviving_write_frobenius_ratio"),
               "mean_alpha_product_upper_bound":avg("survival_source_alpha_bound"),
               "mean_direct_historical_read_l2":avg("direct_old_event_read_l2"),
               "mean_without_event_minus_full_nll":avg("without_event_minus_full_nll"),
               "event_read_helpful_fraction":sum(x["without_event_minus_full_nll"]>1e-9 for x in rows)/count,
               "event_read_harmful_fraction":sum(x["without_event_minus_full_nll"]< -1e-9 for x in rows)/count,
               "numerically_resolved_fraction":sum(x["old_event_matrix_numerically_resolved"] is True for x in rows)/count,
               "mean_full_nextbyte_nll":avg("full_nll"),
               "max_source_event_matrix_error":max(x["event_matrix_source_diff_max_abs_error"] for x in rows),
            }
        group[lane]={"by_lag":by_lag,"all_pairs":4*len(PAIRS)*3}
    return {"study":"LANGMEM080",
      "status":"ORIGINAL_REAL_SOURCE_THREE_CHECKPOINT_EXACT_HISTORICAL_WRITE_TRACE",
      "model_seed_count":3,"source_owners_count":4,
      "source_pairs_per_owner_per_lane":len(PAIRS),
      "total_source_pair_event_lane_records":EXPECTED_ROWS*3,
      "source_observed_anchor_fingerprint_sha256":next(iter(
        x["source_observed_anchor_fingerprint_sha256"] for x in records.values())),
      "source_target_fingerprint_sha256":next(iter(
        x["source_target_fingerprint_sha256"] for x in records.values())),
      "all_original_source_trace_and_sham_checks_pass":True,
      "new_trained_checkpoints":0,"gpu_minutes":0,
      "protected_TEST":"NOT_ACCESSED","independently_qualified":False,
      "metrics":group,
      "original_seed_sha256":{str(seed):records[seed]["original_checkpoint_sha256"] for seed in SEEDS},
      "limits":["licensed genuine original publisher DEV previously EXPOSED",
        "one latent associative write, NOT byte/factual content identity retrieval or learned historical query choice",
        "one-step event-only matrix intervention holds other downstream recurrent states unchanged",
        "some old fast write contributions underflow or are swallowed by float32 cancellation",
        "training FLOPs of original dual versus fast-only model not matched",
        "no new source FIT training, no protected TEST, no independent qualification, no device benchmark"]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("original source result preserved, overwrite forbidden")
    result=collect(a.results)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("LANGMEM080_EVENT_AGGREGATE "+json.dumps({
      "source":result["status"],"original_records":result["total_source_pair_event_lane_records"],
      "metrics":{lane:{lag:{
        "survival":v["mean_write_survival_norm_ratio"],
        "read_l2":v["mean_direct_historical_read_l2"],
        "nll_effect":v["mean_without_event_minus_full_nll"],
        "help_fraction":v["event_read_helpful_fraction"],
        "resolved_fraction":v["numerically_resolved_fraction"]}
      for lag,v in d["by_lag"].items()} for lane,d in result["metrics"].items()}
    },sort_keys=True))

if __name__=="__main__":main()
