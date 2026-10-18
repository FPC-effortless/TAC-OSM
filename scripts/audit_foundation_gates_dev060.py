"""Separate reproducible engineering from automatically evaluated claim gates."""
import argparse
import json
from pathlib import Path


def derive(memory,compatibility):
    arms=memory['arms'];ten=all(len(rows)==10 for rows in arms.values())
    zero_stale=all(cell['stale_same_identity']==0 for cells in arms['temporal'].values() for cell in cells.values())
    temporal_effect=ten and all(c['holm_p']<.05 and c['paired_seed95_t'][0]>0 for c in memory['comparisons'].values())
    return {'zero_stale_retrieval_target_met':zero_stale,
            'temporal_advantage_established':temporal_effect,
            'learned_uncertain_policy_trained':False,'new_default_weights_promoted':False,
            'sparse_original_decisions_preserved':compatibility['status']=='PASS',
            'physical_dram_traffic_measured':False,'references_are_learned':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',default='results');p.add_argument('--index',default='research/dev060/evidence-index.json');a=p.parse_args()
    root=Path(a.results)
    gates=derive(json.loads((root/'dev060-memory/summary.json').read_text()),json.loads((root/'dev060-sparse-compatibility.json').read_text()))
    expected=json.loads(Path(a.index).read_text())['acceptance'];assert gates==expected,'Scientific gate labels disagree with recorded evidence'
    print(json.dumps({'engineering_reproducibility':'PASS','claim_gates':gates,'scope':'No whole-policy training in foundations; scientific failure is preserved despite passing replay CI.'}),flush=True)
