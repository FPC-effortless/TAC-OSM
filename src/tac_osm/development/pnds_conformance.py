"""PNDS-inspired trace measurements. These are not formal implementation proofs.

Ground truth is evaluator-owned. Missing labels yield unavailable rates, never
an assumption that a public receipt proves semantic correctness.
"""
import math


def routing_metrics(scores,relevant,selected,temperature=1.):
    """Unmasked softmax mass and actual selected-index recall, measured separately."""
    scores=tuple(scores);relevant=set(relevant);selected=tuple(selected)
    if not scores or not all(math.isfinite(s) for s in scores):raise ValueError('finite scores required')
    if not math.isfinite(temperature) or temperature<=0:raise ValueError('positive temperature required')
    indices=set(range(len(scores)))
    if not relevant<=indices or not set(selected)<=indices or len(set(selected))!=len(selected):raise ValueError('invalid index set')
    scaled=[s/temperature for s in scores]
    if not all(math.isfinite(s) for s in scaled):raise ValueError('scaled score overflow')
    shift=max(scaled);weights=[math.exp(s-shift) for s in scaled];denominator=sum(weights)
    distractors=indices-relevant;hits=len(relevant&set(selected))
    margin=min(scores[i] for i in relevant)-max(scores[i] for i in distractors) if relevant and distractors else None
    return dict(candidate_count=len(scores),selected_count=len(selected),relevant_count=len(relevant),
        retrieved_relevant=hits,recall=hits/len(relevant) if relevant else None,
        distractors_selected=len(set(selected)-relevant),distractor_mass=sum(weights[i] for i in distractors)/denominator,
        minimum_relevant_margin=margin,total_indexing_cost_measured=False)


def admission_rates(admitted,invalid):
    """Both Bayes directions, plus coverage; None is unknown ground truth."""
    if len(admitted)!=len(invalid):raise ValueError('unaligned trace labels')
    if any(type(a) is not bool for a in admitted):raise TypeError('explicit admission decisions required')
    if any(i is not None and type(i) is not bool for i in invalid):raise TypeError('invalidity must be boolean or unknown')
    pairs=[(a,i) for a,i in zip(admitted,invalid) if i is not None]
    bad=sum(i for _,i in pairs);accepted=sum(a for a,_ in pairs);both=sum(a and i for a,i in pairs)
    # Incompletely labeled populations do not identify the population rates.
    complete=len(pairs)==len(admitted)
    return dict(total=len(admitted),labeled=len(pairs),label_coverage=len(pairs)/len(admitted) if admitted else None,
        false_admit_rate=both/bad if complete and bad else None,
        contamination_rate=both/accepted if complete and accepted else None,
        invalid_count=bad,admitted_labeled_count=accepted,false_admissions=both,
        population_rates_identified=complete and bool(pairs))


def four_gate_decision(utility,stability,integrity,cost):
    """Shadow conjunction only; callers must supply independently measured evidence.

    None means unmeasured. This function does not certify gate measurements,
    enforce PersistentStore writes, or estimate utility from a verifier score.
    """
    gates=dict(utility=utility,stability=stability,integrity=integrity,cost=cost)
    if any(x is not None and type(x) is not bool for x in gates.values()):raise TypeError('gates require booleans or unknown')
    return dict(gates=gates,all_measured=all(x is not None for x in gates.values()),
        would_admit=all(x is True for x in gates.values()),online_enforced=False)


def path_calibration(probabilities,correct,lengths,threshold=.95,bins=10):
    """Descriptive held-out diagnostics by length; no fitted threshold or guarantee."""
    if not (len(probabilities)==len(correct)==len(lengths)):raise ValueError('unaligned paths')
    if not 0<=threshold<=1 or type(bins) is not int or bins<1:raise ValueError('invalid calibration policy')
    if any(not math.isfinite(p) or not 0<=p<=1 for p in probabilities):raise ValueError('probabilities required')
    if any(type(y) is not bool for y in correct):raise TypeError('semantic path correctness labels required')
    if any(type(n) is not int or n<1 for n in lengths):raise ValueError('positive path lengths required')
    result=[]
    for length in sorted(set(lengths)):
        rows=[(p,y) for p,y,n in zip(probabilities,correct,lengths) if n==length]
        admitted=[(p,y) for p,y in rows if p>=threshold];ece=0.
        for b in range(bins):
            bucket=[(p,y) for p,y in rows if min(int(p*bins),bins-1)==b]
            if bucket:ece+=abs(sum(p-y for p,y in bucket))/len(rows)
        result.append(dict(length=length,count=len(rows),brier=sum((p-y)**2 for p,y in rows)/len(rows),
            ece=ece,admitted=len(admitted),false_commit_fraction=sum(not y for _,y in admitted)/len(admitted) if admitted else None))
    return dict(by_length=result,threshold=threshold,descriptive_only=True,calibration_guaranteed=False)


def inspect_engineering_report(report):
    """Expose DEV-032's absent correctness/gate evidence without inventing labels."""
    trace=[step for row in report['rows'] for step in row['trace']]
    return dict(report_source=report['source'],ordinary_rollouts=len(report['rows']),ordinary_actions=len(trace),
        public_receipts_accepted=sum(t['receipt_acceptance'] for t in trace),
        commit_decisions=admission_rates([t['commit_accepted'] for t in trace],[None]*len(trace)),
        four_gate_shadow=four_gate_decision(None,None,None,None),
        relevance_ground_truth_available=False,learned_causal_forecast_evaluated=False,
        formal_correspondence_proved=False,
        finding='Receipt acceptance is not semantic path correctness; registration and calibration remain unmeasured')
