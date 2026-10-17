"""Inspect DEV-042 archived action probabilities without refitting or selecting models.

Post-hoc diagnostic. No p-value, generalization claim or inference-time access to
ground-truth labels is supplied to the models; this script inspects archived rows.
Usage: python scripts/inspect_order_pair_posthoc.py DEV-042-complete-audited-evidence.zip
"""
from __future__ import annotations
import argparse
import json
import statistics
import zipfile


def summarize(archive):
    output={}
    for seed in (4101,4102,4103):
        name=f"DEV042-RAW-seed{seed}-not-confirmatory/report.json"
        source=json.loads(archive.read(name))
        result={}
        for arm in ("bounded_trainable","bounded_random_frozen","memoryless"):
            rows=source["arms"][arm]["rows"]
            contrasts=[];hard_flips=0
            for i in range(0,len(rows),2):
                a,b=rows[i:i+2]
                assert a["partner"]==0 and b["partner"]==1
                assert a["target_entity"]==b["target_entity"]
                assert a["deadline"]==b["deadline"]
                if a["deadline"]=="before":
                    continue
                target=a["target_entity"]
                assert a["labels"][target][0]==1 and b["labels"][target][0]==0
                pa=a["probabilities"][target][0]
                pb=b["probabilities"][target][0]
                contrasts.append(pa-pb)
                hard_flips+=int((pa>.5)!=(pb>.5))
            assert len(contrasts)==96
            result[arm]=dict(mature_pairs=len(contrasts),
                             correct_direction=sum(d>0 for d in contrasts),
                             wrong_direction=sum(d<0 for d in contrasts),
                             ties=sum(d==0 for d in contrasts),
                             mean_signed_difference=statistics.mean(contrasts),
                             mean_absolute_difference=statistics.mean(map(abs,contrasts)),
                             hard_prediction_flips=hard_flips)
        output[str(seed)]=result
    return dict(study="DEV-042-posthoc-probability-inspection",
                status="EXPLORATORY_INSPECTED_COHORT_NO_NEW_TRAINING",
                by_seed=output)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("archive")
    args=p.parse_args()
    with zipfile.ZipFile(args.archive) as f:
        assert f.testzip() is None
        print(json.dumps(summarize(f),indent=2,allow_nan=False))
