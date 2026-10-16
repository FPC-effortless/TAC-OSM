#!/usr/bin/env python3
"""Descriptive DEV-029 summary; preserve both registered and legacy metrics.

Only run after the independent pilot audit. Episode distributions below are
not seed uncertainty estimates. Historical cancellation criteria are reported
separately from the prospectively registered all-role selection metric.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def summarize(folder):
    folder=Path(folder);audit=json.loads((folder/'summary.json').read_text())
    if not audit['audit_passed'] or audit['smoke']:
        raise ValueError('requires completed non-smoke independent audit')
    result=dict(experiment='DEV-029 validation-only pilot',source=audit['source'],
        metric_definitions={
            'registered_selection_accuracy':'Target entity wait prediction, H512, all four roles, at/after deadlines (96 episodes)',
            'legacy_cancellation_accuracy':'Target entity wait prediction, H512, pending_cancel/reissued_cancel, at/after deadlines (48 episodes)',
            'cost_difference':'Selected minus model-only cost per entity, all 144 validation episodes',
            'episode_distribution':'Descriptive correlated episode distribution, not seed uncertainty'},
        initialization_seeds=1,training_recipe_count=1,test_accessed=False,
        full_tuning_complete=False,fresh_twenty_seed_reliability_complete=False,
        original_reliability_gates_assessed=False,architectural_advantage_established=False,arms={})
    for arm in audit['arms']:
        raw=folder/f'{arm}.json';r=json.loads(raw.read_text());rows=r['validation']['rows']
        selected=[x for x in rows if x['deadline']!='before']
        legacy=[x for x in selected if 'cancel' in x['role']]
        accuracy=lambda xs:sum(x['predictions'][x['target_entity']][0]==x['labels'][x['target_entity']][0] for x in xs)/len(xs)
        # Independently recompute all-query CE from archived logits.
        losses=[]
        for row in rows:
            for entity in range(8):
                for action in range(3):
                    logits=row['logits'][entity][action];maximum=max(logits)
                    losses.append(maximum-logits[row['labels'][entity][action]]+math.log(sum(math.exp(x-maximum) for x in logits)))
        assert abs(statistics.mean(losses)-r['validation']['cross_entropy'])<1e-6
        assert accuracy(selected)==r['validation']['primary_accuracy']
        policies={}
        for policy in ('none','selected','matched_random'):
            outcomes=[row[policy] for row in rows]
            costs=[x['cost_per_entity'] for x in outcomes]
            policies[policy]=dict(total_failures=sum(x['failures'] for x in outcomes),
                failure_rate=sum(x['failures'] for x in outcomes)/(len(rows)*8),
                total_services=sum(len(x['services']) for x in outcomes),
                unnecessary_services=sum(x['unnecessary_services'] for x in outcomes),
                unnecessary_rate=sum(x['unnecessary_services'] for x in outcomes)/(len(rows)*8),
                inspections=sum(x['inspections'] for x in outcomes),
                total_cost=sum(x['total_cost'] for x in outcomes),
                cost_per_entity_mean=statistics.mean(costs),cost_per_entity_median=statistics.median(costs),
                cost_per_entity_min=min(costs),cost_per_entity_max=max(costs),
                cost_per_entity_quartiles=statistics.quantiles(costs,n=4,method='inclusive'))
        differences=[x['selected']['cost_per_entity']-x['none']['cost_per_entity'] for x in rows]
        delta=statistics.mean(differences)
        result['arms'][arm]=dict(registered_selection_accuracy=accuracy(selected),
            legacy_cancellation_accuracy=accuracy(legacy),registered_n=len(selected),legacy_n=len(legacy),
            cross_entropy=statistics.mean(losses),parameters=r['parameters'],
            training_seconds=r['training_seconds'],training_cpu_seconds=r['training_cpu_seconds'],
            state_bytes=r['state_bytes'],validation_seconds=r['validation']['seconds'],
            checkpoint_bytes=r['checkpoint_bytes'],policies=policies,selected_minus_none_cost_mean=delta,
            selected_minus_none_cost_min=min(differences),selected_minus_none_cost_max=max(differences),
            strict_mean_cost_component_pass=delta<=0,
            raw_report_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();s=summarize(a.input);Path(a.output).write_text(json.dumps(s,indent=2)+'\n')
    print(json.dumps({a:dict(selection=x['registered_selection_accuracy'],cancellation=x['legacy_cancellation_accuracy'],cost_delta=x['selected_minus_none_cost_mean']) for a,x in s['arms'].items()}))
