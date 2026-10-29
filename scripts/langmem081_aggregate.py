#!/usr/bin/env python3
"""LANGMEM081 aggregate original true source address specificity, not new learning.

Fail-closed original three seed checkpoints and exactly same publisher
observed contexts/outcomes. 6912 repeated L/H query CELLS per seed is
768 unique real next byte labels, not 6912 independent examples.
"""
from __future__ import annotations
import argparse,json,math,statistics
from pathlib import Path
SEEDS=(1701,1702,1703)
LENGTHS=(2,3,4)
AGES=(32,128,256)
START=128
STOP=320
OWNERS=4
CELLS=OWNERS*(STOP-START)*len(LENGTHS)*len(AGES)
ARMS=("model_writer_observed_continuation_correct",
      "hardest_wrong_observed_continuation_correct",
      "exact_recent_match_observed_continuation_correct",
      "uniform_random_observed_continuation_correct",
      "most_recent_any_observed_continuation_correct")


def summarize(root:Path):
    files=list(root.rglob("CONTEXT_RESULTS.json"))
    if len(files)!=3:raise ValueError("requires three unmodified original checkpoint seed source reports")
    runs={}
    signatures=[]
    for p in files:
        r=json.loads(p.read_text())
        seed=r.get("seed")
        if seed not in SEEDS or seed in runs:
            raise ValueError("original source seed duplicated/unknown")
        if (r.get("study")!="LANGMEM081" or r.get("trained_original_parameters")!=499627
            or r.get("native_state_fp32_bytes_per_owner")!=4160
            or r.get("history_side_table")!="NOT_PART_OF_4160BYTE_PERSISTENT_MODEL_STATE"
            or r.get("original_query_clock_begin")!=START
            or r.get("original_query_clock_end_exclusive")!=STOP
            or tuple(r.get("source_context_lengths",[]))!=LENGTHS
            or tuple(r.get("source_history_caps",[]))!=AGES
            or r.get("source_observed_nextbyte_targets_per_seed")!=768
            or r.get("source_observed_cohort_cells")!=CELLS
            or r.get("original_model_no_new_fit") is not True
            or r.get("optimizer_updates")!=0 or r.get("gpu_minutes")!=0
            or r.get("protected_TEST")!="NOT_ACCESSED"
            or not math.isfinite(r.get("original_graph_128_prefix_max_abs_logit_error",math.inf))
            or r["original_graph_128_prefix_max_abs_logit_error"]>3e-5):
            raise ValueError("registered original nofit genuine source protocol compromised")
        for field in ("original_model_checkpoint_sha256","original_fit_sample_order_sha256",
                      "actual_source_nextbyte_sha256","original_observed_context_sha256"):
            x=r.get(field)
            if not isinstance(x,str) or len(x)!=64 or not all(c in "0123456789abcdef" for c in x):
                raise ValueError("missing original trained checkpoint or true source SHA")
        rows=r.get("rows")
        if not isinstance(rows,list) or len(rows)!=CELLS:
            raise ValueError("missing original source context/window cells")
        owner_names=tuple(r.get("source_document_owners",[]))
        if len(owner_names)!=4 or len(set(owner_names))!=4:
            raise ValueError("actual original publisher owner cohort differs")
        indices=set()
        source_identity=[]
        for row in rows:
            name=row["owner"]
            t=row["clock"];L=row["context_len"];H=row["age_cap"]
            if name not in owner_names or not START<=t<STOP or L not in LENGTHS or H not in AGES:
                raise ValueError("source original owner, clock or selector changed")
            key=(name,t,L,H)
            if key in indices:raise ValueError("duplicate source query cell")
            indices.add(key)
            if not 0<=row["actual_original_next_byte_target"]<=255:
                raise ValueError("unobserved or illegal actual publisher target")
            if (row["candidate_count"]<=0 or row["candidate_count"]>H
                or not 0<=row["positive_match_count"]<=row["candidate_count"]
                or row["positive_exists"]!=(row["positive_match_count"]>0)
                or row["model_writer_context_hit"] and not row["positive_exists"]):
                raise ValueError("historical matching/candidate source count inconsistent")
            for k in ARMS:
                if row[k] is not None and row[k] not in (0,1):
                    raise ValueError("source next byte accuracy not 0/1/None")
            if row["positive_exists"]:
                if not 0<row["model_best_matching_reciprocal_rank"]<=1:
                    raise ValueError("history positive source rank missing")
                if row["exact_recent_match_observed_continuation_correct"] not in (0,1):
                    raise ValueError("exact actual past match continuation absent")
            else:
                if (row["model_best_matching_rank"] is not None
                    or row["model_best_matching_reciprocal_rank"] is not None
                    or row["exact_recent_match_observed_continuation_correct"] is not None):
                    raise ValueError("false credit for nonexisting history match")
            for f in ("original_trained_TAC_nextbyte_nll","exact_history_smoothed_source_nll"):
                if not math.isfinite(row[f]) or row[f]<0:raise ValueError("nonfinite authentic next byte source loss")
            source_identity.append((name,t,L,H,tuple(row["query_context_observed"]),
                                    row["positive_match_count"],
                                    row["positive_exists"],
                                    row["actual_original_next_byte_target"]))
        expected={(o,t,L,H) for o in owner_names for t in range(START,STOP)
                   for L in LENGTHS for H in AGES}
        if indices!=expected:raise ValueError("some real source query contexts omitted")
        signatures.append(tuple(source_identity))
        runs[seed]=r
    if set(runs)!=set(SEEDS):raise ValueError("original all three trained seed receipts missing")
    if not all(signature==signatures[0] for signature in signatures):
        raise ValueError("model-specific original source context, historical matches or future truth differ")
    for field in ("actual_source_nextbyte_sha256","original_observed_context_sha256"):
        if len({r[field] for r in runs.values()})!=1:
            raise ValueError("original frozen source nextbyte/context SHA differed across seeds")
    metrics={}
    for L in LENGTHS:
        for H in AGES:
            cells=[r for x in runs.values() for r in x["rows"] if r["context_len"]==L and r["age_cap"]==H]
            all_count=len(cells)
            if all_count!=768*3:raise ValueError("incomplete original context/window cohort")
            positive=[r for r in cells if r["positive_exists"]]
            n_pos=len(positive)
            negatives=[r for r in cells if r["hardest_wrong_observed_continuation_correct"] is not None]
            def avg(rows,key):
                return statistics.mean(r[key] for r in rows) if rows else None
            item={
              "original_cell_count":all_count,"real_individual_nextbyte_source_outcomes_per_seed":768,
              "source_positive_coverage":n_pos/all_count,
              "source_positive_cells":n_pos,
              "source_mean_available_positive_writers_when_match":avg(positive,"positive_match_count"),
              "mean_candidate_writer_count":avg(cells,"candidate_count"),
              "learned_query_top1_correct_context_when_positive":avg(positive,"model_writer_context_hit"),
              "learned_query_context_match_mrr_when_positive":avg(positive,"model_best_matching_reciprocal_rank"),
              "learned_query_top1_correct_context_overall":avg(cells,"model_writer_context_hit"),
              "learned_query_source_nextbyte_top_writer_accuracy":avg(cells,"model_writer_observed_continuation_correct"),
              "hardest_wrong_context_source_nextbyte_accuracy":avg(negatives,"hardest_wrong_observed_continuation_correct"),
              "exact_history_recent_matching_continuation_accuracy_when_match":avg(positive,"exact_recent_match_observed_continuation_correct"),
              "uniform_source_writer_random_accuracy":avg(cells,"uniform_random_observed_continuation_correct"),
              "source_recent_any_writer_continuation_accuracy":avg(cells,"most_recent_any_observed_continuation_correct"),
              "exact_history_smoothed_8_posterior_nll_all":avg(cells,"exact_history_smoothed_source_nll"),
              "exact_history_smoothed_8_posterior_nll_when_match":avg(positive,"exact_history_smoothed_source_nll"),
              "original_trained_model_nll_all":avg(cells,"original_trained_TAC_nextbyte_nll"),
              "mean_learned_best_positive_cosine_minus_hard_wrong_cosine":(
                 avg([r for r in positive if r["model_writer_cosine_margin_to_hard_wrong"] is not None],
                     "model_writer_cosine_margin_to_hard_wrong")),
            }
            metrics[f"L{L}_H{H}"]=item
    return {
      "study":"LANGMEM081",
      "status":"ORIGINAL_SOURCE_THREE_SEED_CAUSAL_CONTEXT_MATCH_VS_NATIVE_LEARNED_ADDRESS",
      "source_matched_three_original_model_seed_checkpoints":3,
      "original_real_publisher_dev_owners":4,
      "source_unique_actual_nextbyte_targets_per_seed":768,
      "source_repeated_LH_query_cells_per_seed":CELLS,
      "all_real_source_context_cohorts_and_target_SHA_identical":True,
      "original_source_target_sha256":runs[1701]["actual_source_nextbyte_sha256"],
      "original_observed_context_sha256":runs[1701]["original_observed_context_sha256"],
      "history_side_table_in_model_persistent_bytes":False,
      "learned_model_state_fp32_bytes_per_stream":4160,
      "trained_model_weights_changed":False,
      "gpu_minutes":0,"protected_TEST":"NOT_ACCESSED",
      "independently_qualified":False,
      "metrics":metrics,
      "model_original_sha256_by_seed":{str(seed):r["original_model_checkpoint_sha256"] for seed,r in runs.items()},
      "limitations":[
        "genuine PSF source but original DEV previously EXPOSED, repeated true labels across L/H/seeds",
        "exact-history matching uses entire recent observed plaintext and all learned source writer keys as SIDE TABLE, not native model 4160B constant memory",
        "context-match target is historic preceding-context equality, NOT correct future continuation or natural delayed-fact recall",
        "original learned native writer rank alone does not certify what downstream learned network reads",
        "frozen tau8 history count posterior a precommitted non-neural inference baseline, not newly trained TAC model",
        "different L/H original source cohorts have different positive coverage; no fairness claim with independently matched SSM/Transformer",
        "original source cap_all position mode chosen after DEV; no TEST, new trained model, GPU, phone device or generalization"]
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("preserve original source aggregation without overwrite")
    r=summarize(a.results)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("LANGMEM081_CONTEXT_SUMMARY "+json.dumps({
      "study":r["study"],"targets_per_seed":r["source_unique_actual_nextbyte_targets_per_seed"],
      "source_cells_per_seed":r["source_repeated_LH_query_cells_per_seed"],
      "metrics":{k:{"coverage":v["source_positive_coverage"],
          "learned_context_top1_when_present":v["learned_query_top1_correct_context_when_positive"],
          "learned_mrr":v["learned_query_context_match_mrr_when_positive"],
          "trained_native_pred_byte_accuracy":v["learned_query_source_nextbyte_top_writer_accuracy"],
          "exact_recent_byte_accuracy_when_match":v["exact_history_recent_matching_continuation_accuracy_when_match"],
          "random_byte_accuracy":v["uniform_source_writer_random_accuracy"],
          "history_smoothed_nll":v["exact_history_smoothed_8_posterior_nll_all"],
          "original_TAC_nll":v["original_trained_model_nll_all"]} for k,v in r["metrics"].items()}
    },sort_keys=True),flush=True)
if __name__=="__main__":main()
