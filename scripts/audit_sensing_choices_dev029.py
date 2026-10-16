#!/usr/bin/env python3
"""Supplemental independent replay of ALL hypothetical public receipt choices.

Added after the frozen run as an integrity check. No metric, decision threshold
or outcome is changed. Does not import the experiment's sensing implementation.
"""
import argparse
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from tac_osm.development.optimization_program import ARMS,EventTablePredictor
from tac_osm.development.memory_baselines import MemoryBaseline
from tac_osm.development.entity_aging import EntityAgingControl
from tac_osm.development.schema_maintenance import make_pair
from tac_osm.development.delayed_maintenance import PublicObservation
from run_learned_schema_dev import public_stream_tensors,public_queries


@torch.no_grad()
def audit(folder):
    torch.set_num_threads(1);folder=Path(folder)
    summary=json.loads((folder/'summary.json').read_text());assert summary['audit_passed'] and not summary['smoke']
    counts={};max_error=0.
    cost=lambda probabilities:sum(min(4*p[0],1+4*p[1]) for p in probabilities)
    for arm in ARMS:
        report=json.loads((folder/f'{arm}.json').read_text())
        saved=torch.load(folder/f'{arm}.pt',map_location='cpu',weights_only=True)
        model=(EventTablePredictor() if arm=='event_table' else EntityAgingControl('masked_local')
               if arm=='masked_local' else MemoryBaseline('tac_joint' if arm=='native' else arm))
        model.load_state_dict(saved['model']);model.eval();rows=report['validation']['rows']
        for offset in range(0,len(rows),2):
            pair=rows[offset:offset+2];r=pair[0]
            cases=make_pair(900000+r['repetition'],512,r['due_delay'],r['role'],r['deadline'],domain='dev029-v1-validation')
            assert [x['fingerprint'] for x in pair]==[c.fingerprint for c in cases]
            state=model.encode(*public_stream_tensors([c.observations for c in cases]))
            queries=public_queries([c.tick for c in cases],[c.horizon for c in cases])
            pre=model.decode(state,queries).softmax(-1)[...,1].tolist()
            streams=[(PublicObservation(c.tick,e,'inspect',bit),) for c in cases for e in range(8) for bit in (0,1)]
            inputs=public_stream_tensors(streams)
            hypothetical=model.update(state.repeat_interleave(16,0),*(x[:,0] for x in inputs))
            future=model.decode(hypothetical,queries.repeat_interleave(16,0)).softmax(-1)[...,1].reshape(2,8,2,8,3).tolist()
            for b,row in enumerate(pair):
                costs=[[cost(future[b][e][bit]) for bit in (0,1)] for e in range(8)]
                gains=[cost(pre[b])-(1-pre[b][e][2])*costs[e][0]-pre[b][e][2]*costs[e][1]-.1 for e in range(8)]
                recorded=row['selected']
                torch.testing.assert_close(torch.tensor(costs),torch.tensor(recorded['hypothetical_costs']),rtol=1e-5,atol=1e-6)
                torch.testing.assert_close(torch.tensor(gains),torch.tensor(recorded['estimated_gains']),rtol=1e-5,atol=1e-6)
                max_error=max(max_error,max(abs(x-y) for x,y in zip(gains,recorded['estimated_gains'])))
                chosen=max(range(8),key=lambda e:(gains[e],-e))
                assert recorded['entity']==(chosen if gains[chosen]>0 else None)
        counts[arm]=len(rows)*16
        print(json.dumps(dict(arm=arm,hypothetical_receipts_replayed=counts[arm])),flush=True)
    return dict(experiment='DEV-029 supplemental sensing-choice integrity audit',audit_passed=True,
        frozen_training_source=summary['source'],hypothetical_receipts_replayed=counts,
        total_hypothetical_receipts=sum(counts.values()),maximum_gain_error=max_error,
        thresholds_unchanged=True,actual_selected_entities_exact=True,
        limitation='Post-run integrity check; does not add initialization seeds or generalization evidence')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=audit(a.input);Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
