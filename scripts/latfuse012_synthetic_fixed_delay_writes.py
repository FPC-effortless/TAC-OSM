"""SYNTHETIC ONLY; original pretrained positive-control checkpoints, no further fits.

Freeze synthetic task elapsed delay at512 and source/observations. Change
only number of intervening associative-memory updates after the labeled
past event, leaving fast/slow/source state and all observations unchanged.
NOT real-source skill, NOT architecture deployment, OOD memory intervention.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from scripts import latfuse002_real_fit as cal
from scripts import latfuse011_synthetic_trainability as prior
from tac_osm.latent_fusion_007_historic_query import SourceTimedEventQuery

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/"research/mathematical-engineering/latfuse012/SYNTHETIC_DISTRACTOR_REGISTRATION.json"
OUT=ROOT/"research/replay-bundles/LATFUSE011-SYNTHETIC"
COUNTS=(0,8,32,128,512)
DELAY=512

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def synthetic_batch(seed):
    # Identical 32 heldout synthetic generator groups to original source-free
    # LATFUSE011 diagnostic. Original observed synthetic control is count512.
    b=[];sites=[];labels=[];times=[]
    for i in range(3000,3032):
        t,x,y,s=prior.synthetic_case(seed,i,DELAY)
        b.append(x)
        labels.append(y)
        sites.append(s)
        times.extend([t+23]*4)
    return torch.cat(b,dim=1),torch.cat(labels),torch.cat(sites),times

def for_count(m,xx,labels,site,times,count):
    state=m.initial(len(labels))
    done=0
    with torch.no_grad():
        for t in range(xx.shape[0]):
            pre=state
            zero=torch.zeros_like(xx[t])
            mask=torch.ones(xx[t].shape,dtype=torch.bool)
            state=m.step(state,xx[t],mask,zero,zero,site)
            # After past event step23, retain ONLY N evenly spaced suffix
            # updates. Every non-associative state still processes full input.
            if t>=24:
                i=t-23
                write=(i*count//DELAY) > ((i-1)*count//DELAY)
                if not write:
                    state=state._replace(associative=pre.associative)
                else:
                    done+=1
        if done!=count:raise RuntimeError("fixed-horizon write cardinality failed")
        pastclock=cal.calendar(times)[:,None,:]
        logits=m.historic_event_logits(state,site,pastclock)[:,0]
        probs=logits.sigmoid()
        ablated=state._replace(associative=torch.zeros_like(state.associative))
        null= m.historic_event_logits(ablated,site,pastclock)[:,0].sigmoid()
        p1=probs[::2]
        p0=probs[1::2]
        ranking=float(((p1>p0).float()+.5*(p1==p0).float()).mean())
        balanced=.5*float(((probs[labels==1]>=.5).float().mean()+
            (probs[labels==0]<.5).float().mean()))
        brier=float((probs-labels).square().mean())
        return dict(synthetic_only=True,horizon_steps=DELAY,
             eligible_intervening_associative_writes=count,
             total_intervening_source_steps=DELAY,
             synthetic_pairs=len(labels)//2,
             balanced_accuracy=balanced,paired_event_ranking=ranking,
             event_Brier=brier,
             whole_read_lesion_mean_probability_difference=float((probs-null).abs().mean()),
             original_label_event_value_read_only_for_scoring=True,
             zero_added_optimizer_updates=True),state

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    x=ap.parse_args()
    if x.output.exists():raise RuntimeError("original synthetic-only empirical follow-up exists")
    torch.set_num_threads(1)
    r=json.loads(REG.read_text())
    if (r["schema_version"]!="tac-osm.synthetic-diagnostic.v1" or
        r["synthetic_only"] is not True or r["no_real_world_claim"] is not True or
        r["intervening_write_counts"]!=list(COUNTS) or r["fixed_delay_hours"]!=DELAY or
        r["no_new_model_fits"] is not True):
        raise RuntimeError("synthetic-only registered count-sweep boundary invalid")
    pri=json.loads((OUT/"RESULTS.json").read_text())
    result=[]
    for rec in pri["records"]:
        seed=rec["seed"]
        cp=OUT/f"seed-{seed}"/"final.pt"
        dg=sha(cp)
        if dg!=rec["original_checkpoint_sha256"]:
            raise RuntimeError("original synthetic-only trained model checkpoint changed")
        old=torch.load(cp,map_location="cpu",weights_only=False)
        torch.manual_seed(seed)
        m=SourceTimedEventQuery()
        m.load_state_dict(old["model"],strict=True)
        m.eval()
        weights=[p.detach().clone() for p in m.parameters()]
        xx,label,site,times=synthetic_batch(seed)
        last_state=None
        run={}
        for count in COUNTS:
            val,state=for_count(m,xx,label,site,times,count)
            if last_state is not None:
                # Verify SAME nonassociative state, source owner flags,
                # current source values and read-clock for all counted arms.
                for a,b in zip(
                  (last_state.fast,last_state.slow,last_state.entities,last_state.active),
                  (state.fast,state.slow,state.entities,state.active)):
                    if not torch.equal(a,b):
                        raise RuntimeError("synthetic fixed-delay count intervention changed non-assoc state")
            last_state=state
            run[str(count)]=val
        base=old["synthetic_heldout_results"][str(DELAY)]
        full=run[str(DELAY)]
        if (abs(base["synthetic_Brier"]-full["event_Brier"])>5e-5 or
            abs(base["synthetic_pair_ranking"]-full["paired_event_ranking"])>1e-5 or
            abs(base["synthetic_balanced_accuracy"]-full["balanced_accuracy"])>1e-5):
            raise RuntimeError("original synthetic positive-control all-writes results cannot replay")
        if dg!=sha(cp) or any(not torch.equal(p,z) for p,z in zip(m.parameters(),weights)):
            raise RuntimeError("original synthetic-trained source-free model altered")
        result.append(dict(seed=seed,original_synthetic_checkpoint_sha256=dg,
                           intervention_results=run))
        print("LATFUSE012_SYNTHETIC_ONLY_FROZEN_SEED="+json.dumps({
            "seed":seed,"pairs":len(label)//2,
            "write_counts":{k:{"paired":v["paired_event_ranking"],
               "brier":v["event_Brier"]} for k,v in run.items()}},sort_keys=True),flush=True)
    output=dict(study_id=r["study_id"],
        status="SYNTHETIC DIAGNOSTIC ONLY — NEVER REAL SOURCE OR QUALIFIED ABILITY",
        original_model_param_count=99390,original_recurrent_nominal_bytes=6800,
        original_synthetic_train_seed_count=len(result),
        no_new_fit=True,
        fixed_delay_steps=DELAY,source_archive_ever_used=False,
        original_paired_inputs_held_constant_across_all_memory_write_counts=True,
        exact_nonassociative_state_held_constant_across_arms=True,
        suppressing_memory_write_also_suppresses_alpha_decay_under_original_update=True,
        learned_original_model_trained_with_full_intervening_source_writes=True,
        changes_memory_dynamics_out_of_distribution=True,
        original_checkpoint_shas_verified=True,
        findings_not_valid_real_world_capability_claims=True,
        records=result)
    x.output.parent.mkdir(parents=True,exist_ok=True)
    x.output.write_text(json.dumps(output,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("LATFUSE012_SYNTHETIC_WRITE_COUNT_PASS")
if __name__=="__main__":main()
