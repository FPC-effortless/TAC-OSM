"""LATFUSE011 synthetic TRAINABILITY diagnostic ONLY on unchanged original TAC model.

Not source data. Does not establish real-world, architecture superiority,
causal policy or qualified historical-event recall. Unlike authentic-data
preflight, this *intentionally trains* two synthetic-only checkpoints for
one clearly bounded algorithmic delayed recall task. Saves ALL original
model optimizer and RNG artifacts; no UCI source or real protected labels.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import torch
import torch.nn.functional as F

from scripts import latfuse002_real_fit as source
from tac_osm.latent_fusion_007_historic_query import SourceTimedEventQuery

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/"research/mathematical-engineering/latfuse011/SYNTHETIC_DIAGNOSTIC_REGISTRATION.json"
OUT=ROOT/"research/replay-bundles/LATFUSE011-SYNTHETIC"
SEEDS=(11801,11802)
DELAYS=(16,64,128,512)
STEPS=160

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prior_registration():
    r=json.loads(REG.read_text())
    if (r["schema_version"]!="tac-osm.synthetic-diagnostic.v1" or
        r["study_type"]!="synthetic_only_diagnostic_training" or
        not r["synthetic_only"] or r["no_real_world_claim"] is not True or
        r["original_source_archive_used"] is not False or
        r["protected_source_used"] is not False):
        raise RuntimeError("explicit synthetic-only research boundaries changed")
    if [(x["seed"],x["trainable_parameters"]) for x in r["models"]]!=[(11801,99390),(11802,99390)]:
        raise RuntimeError("fake positive-control seed or architecture mismatch")
    return r

def synthetic_case(seed,step,lag):
    """Paired 4 synthetic rows with exactly same nuisance; only one past event varies.

    For every synthetic pair row 0 high(+3), row1 ordinary(-3);
    random distractors and any decoy are *identical* within each pair.
    Query historic clock source known time, not the event label. This is
    deliberately simple and diagnostic-only (not a real-world task).
    """
    g=torch.Generator().manual_seed(seed*100000+step*997+lag*13)
    start=source.hour(__import__("datetime").datetime(2013,6,1))
    start=start+int(torch.randint(0,1200,(1,),generator=g))
    b=4
    pair_site=torch.tensor([0,0,1,1],dtype=torch.long)
    # Readable label produced explicitly by synthetic event channel.
    labels=torch.tensor([1.,0.,1.,0.])
    n=24+lag
    draws=torch.randn(n,2,13,generator=g)*.25
    xx=draws.repeat_interleave(2,dim=1)
    xx[23,:,0]=torch.tensor([3.,-3.,3.,-3.])
    # Different fake intervening high-frequency distractors; no label cue.
    for k in range(29,n,37):
        xx[k,:,0]=torch.tensor([2.5,2.5,-2.5,-2.5])
    return start,xx,labels,pair_site

def predict(m,seed,case_idx,lag,*,need_grad):
    t0,vals,label,site=synthetic_case(seed,case_idx,lag)
    s=m.initial(4)
    ages=torch.zeros((4,13))
    seen=torch.ones((4,13),dtype=torch.bool)
    last=torch.zeros((4,13))
    for i in range(vals.shape[0]):
        x=vals[i]
        s=m.step(s,x,seen,ages,last,site)
        last=x
    source_known_time=source.calendar([t0+23]*4)[:,None,:]
    logit=m.historic_event_logits(s,site,source_known_time)[:,0]
    return logit,label

def measure(m,seed,lag):
    records=[]
    with torch.no_grad():
        for case in range(3000,3032):
            logits,label=predict(m,seed,case,lag,need_grad=False)
            p=torch.sigmoid(logits)
            records.append((p,label))
    p=torch.cat([x[0] for x in records])
    label=torch.cat([x[1] for x in records])
    assert torch.equal(label[::2],torch.ones_like(label[::2]))
    score=(p[::2]>p[1::2]).float()+.5*(p[::2]==p[1::2]).float()
    balanced=.5*((p[label==1]>=.5).float().mean()+(p[label==0]<.5).float().mean())
    return dict(synthetic_test_pairs=64,synthetic_Brier=float((p-label).square().mean()),
          synthetic_balanced_accuracy=float(balanced),
          synthetic_pair_ranking=float(score.mean()),
          neutral_prior_Brier=.25,
          no_real_source_or_generalization_claim=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=("fit","replay"),required=True)
    x=ap.parse_args()
    torch.set_num_threads(1)
    prior_registration()
    if x.mode=="fit":
        if OUT.exists():raise RuntimeError("synthetic diagnostic original trained artifacts already saved; NO overwrite")
        OUT.mkdir(parents=True,exist_ok=False)
        report=[]
        for seed in SEEDS:
            torch.manual_seed(seed)
            m=SourceTimedEventQuery()
            if m.original_state_and_weight_contract()["trainable_parameters"]!=99390:
                raise RuntimeError("original 100k synthetic positive-control contract invalid")
            opt=torch.optim.AdamW(m.parameters(),lr=.0005,weight_decay=.001)
            losses=[]
            began=time.monotonic()
            for step in range(STEPS):
                lag=DELAYS[step%len(DELAYS)]
                opt.zero_grad(set_to_none=True)
                p,label=predict(m,seed,step,lag,need_grad=True)
                loss=F.binary_cross_entropy_with_logits(p,label)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("nonfinite synthetic diagnostic training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
                opt.step()
                losses.append(float(loss.detach()))
                if step%32==0:
                    print("LATFUSE011_SYNTHETIC_PROGRESS="+json.dumps({
                        "seed":seed,"step":step,"lag":lag,"loss":losses[-1],
                        "synthetic_only":True}),flush=True)
            m.eval()
            results={str(h):measure(m,seed,h) for h in DELAYS}
            folder=OUT/f"seed-{seed}"
            folder.mkdir()
            cp=folder/"final.pt"
            torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),
                torch_rng=torch.get_rng_state(),seed=seed,synthetic_only=True,
                original_model_parameter_count=99390,original_recurrent_bytes=6800,
                optimizer_steps=STEPS,synthetic_training_losses=losses,
                original_code_sha256=sha(ROOT/"scripts/latfuse011_synthetic_trainability.py"),
                all_synthetic_random_pair_cases_deterministic=True,
                synthetic_heldout_results=results),cp)
            z=dict(seed=seed,original_checkpoint_sha256=sha(cp),optimizer_steps=STEPS,
                synthetic_training_loss_first=losses[0],
                synthetic_training_loss_last=losses[-1],
                synthetic_results=results,CPU_wall_seconds=time.monotonic()-began)
            report.append(z)
            print("LATFUSE011_SYNTHETIC_TRAINED="+json.dumps(z,sort_keys=True),flush=True)
        result=dict(study="LATFUSE011-SYNTHETIC-ALGORITHMIC-POSITIVE-CONTROL",
           scientific_status="SYNTHETIC DIAGNOSTIC ONLY, NEVER REAL SOURCE EVIDENCE",
           model_architecture_unchanged=True,
           source_archive_used=False,protected_numeric_ever_opened=False,
           seed_count=2,total_optimizer_updates=STEPS*2,
           no_original_UCI_checkpoint_or_fit_mutated=True,
           original_6800byte_fp32_state=True,records=report)
        (OUT/"RESULTS.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
        print("LATFUSE011_SYNTHETIC_FINISHED",flush=True)
    else:
        result=json.loads((OUT/"RESULTS.json").read_text())
        if result["seed_count"]!=2 or result["total_optimizer_updates"]!=STEPS*2:
            raise RuntimeError("synthetic diagnostic training incomplete")
        for r in result["records"]:
            cp=OUT/f"seed-{r['seed']}"/"final.pt"
            if sha(cp)!=r["original_checkpoint_sha256"]:
                raise RuntimeError("original synthetic-only trained checkpoint mutated")
            saved=torch.load(cp,map_location="cpu",weights_only=False)
            torch.manual_seed(r["seed"])
            m=SourceTimedEventQuery()
            m.load_state_dict(saved["model"],strict=True)
            m.eval()
            for h in DELAYS:
                now=measure(m,r["seed"],h)
                prior=saved["synthetic_heldout_results"][str(h)]
                for k in now:
                    if isinstance(now[k],float):
                        if abs(now[k]-prior[k])>1e-9:raise RuntimeError(
                            "synthetic diagnostic deterministic replay mismatch")
                    elif now[k]!=prior[k]:raise RuntimeError("synthetic replay metadata changed")
        print("LATFUSE011_SYNTHETIC_ORIGINAL_REPLAY_PASS",flush=True)
if __name__=="__main__":
    main()
