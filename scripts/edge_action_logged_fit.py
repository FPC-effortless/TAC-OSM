"""EDGEACTION001: one preregistered real FIT-only logged-setpoint pilot.

First stage FITS/FREEZES five original models but does NOT open the reserved
FIT subholdout for scoring. Second stage SCORES only from committed originals.
No real-world control, counterfactual, robot safety, or independent test claim.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time,random,subprocess
import numpy as np
import torch
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from scripts import edge_action_fit_source_audit as source
from scripts.validate_research_parity import check_manifest
from tac_osm.variants.action_query_edge_world import ActionQueryEdgeWorld
from tac_osm.parameter_budget import budgeted_optimizer,checked_optimizer_step,save_training_checkpoint,validate_runtime_evidence

OUT=ROOT/"research/replay-bundles/EDGEACTION001"
REG=ROOT/"research/mathematical-engineering/edgeaction001/FAIR_COMPARISON.json"
SEEDS=(91001,91002,91003,91004,91005)
SOURCE_FILES=[
 "scripts/edge_action_logged_fit.py",
 "scripts/edge_action_fit_source_audit.py",
 "src/tac_osm/variants/action_query_edge_world.py",
 "src/tac_osm/variants/evidence_retrieval_world.py",
 "src/tac_osm/variants/parallel_continuous_world.py",
 "src/tac_osm/variants/retention_controlled_world.py",
 "src/tac_osm/variants/parallel_affine_memory.py",
 "src/tac_osm/parameter_budget.py",
 "research/mathematical-engineering/edgeaction001/FAIR_COMPARISON.json",
]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_committed(paths):
    for path in paths:
        path=Path(path)
        rel=str(path.relative_to(ROOT))
        try:data=subprocess.check_output(["git","show","HEAD:"+rel],cwd=ROOT,stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:raise RuntimeError(f"required frozen original absent from Git: {rel}") from e
        if data!=path.read_bytes():raise RuntimeError(f"original source bytes changed: {rel}")

def read_fit(path):
    source.audit(Path(path))  # Verify pinned source and all 59 FIT episodes, before use.
    table=pq.read_table(path,columns=list(source.FIELDS),
                        filters=[("episode_index","<",source.FIT_EPISODE_STOP)]).to_pydict()
    rows={}
    for ep,frame,t,x,a in zip(*(table[k] for k in source.FIELDS)):
        if not 0<=ep<59:raise RuntimeError("out-of-FIT access")
        rows.setdefault(ep,[]).append((int(frame),float(t),np.asarray(x,dtype=np.float32),np.asarray(a,dtype=np.float32)))
    if len(rows)!=59 or set(rows)!=set(range(59)):raise RuntimeError("source episode partition changed")
    ordered=[]
    for ep in range(59):
        r=sorted(rows[ep],key=lambda x:x[0])
        if len(r)<35:raise RuntimeError("insufficient genuine episode history")
        if any(r[i+1][0]!=r[i][0]+1 or not 0<r[i+1][1]-r[i][1]<.2 for i in range(len(r)-1)):
            raise RuntimeError("nonchronological sensor/command timestamps")
        ordered.append(r)
    if len(set(len(r) for r in ordered))!=1:
        raise RuntimeError("variable source episode length needs an explicitly audited loader")
    xs=np.stack([[row[2] for row in ep] for ep in ordered])
    actions=np.stack([[row[3] for row in ep] for ep in ordered])
    n=xs.shape[1]
    if n<40:raise RuntimeError("empty history")
    # FIT 49 episodes only, never fit preprocess on scored ten source episodes.
    xm=xs[:49].mean((0,1));xsd=xs[:49].std((0,1)).clip(.0001)
    am=actions[:49].mean((0,1));asd=actions[:49].std((0,1)).clip(.0001)
    xz=((xs-xm)/xsd).astype(np.float32)
    uz=((actions-am)/asd).astype(np.float32)
    ds=np.sqrt(np.mean(np.square(xz[:49,1:]-xz[:49,:-1]),axis=(0,1))).clip(.00001)
    raw_gain=(actions[:49,32:-1]-xs[:49,32:-1])
    raw_delta=xs[:49,33:]-xs[:49,32:-1]
    servo=np.clip((raw_gain*raw_delta).sum((0,1))/(raw_gain.square().sum((0,1))+1e-12),0,1).astype(np.float32)
    threshold=float(np.quantile(np.sqrt(np.mean(np.square(xz[:49,1:]-xz[:49,:-1]),-1)),.90))
    prep=dict(state_mean=xm,state_scale=xsd,action_mean=am,action_scale=asd,delta_scale=ds,servo_gain=servo,moving_threshold=threshold)
    return dict(xraw=xs,uraw=actions,xz=xz,uz=uz,episode_length=n,prep=prep)

def batch(data,indices):
    sid=np.asarray(indices[:,0],dtype=np.int64)
    t=np.asarray(indices[:,1],dtype=np.int64)
    look=t[:,None]-np.arange(31,-1,-1)[None,:]
    obs=torch.from_numpy(data["xz"][sid[:,None],look].copy())
    prev=torch.from_numpy(data["uz"][sid[:,None],look-1].copy())
    candidate=torch.from_numpy(data["uz"][sid,t].copy())[:,None,:]
    outcome=torch.from_numpy(data["xz"][sid,t+1].copy())
    clock=torch.from_numpy(look.copy()).to(torch.int64)
    owner=torch.from_numpy(sid.copy())
    raw_obs=data["xraw"][sid,t].copy()
    raw_prev=data["xraw"][sid,t-1].copy()
    raw_action=data["uraw"][sid,t].copy()
    raw_outcome=data["xraw"][sid,t+1].copy()
    return (obs,prev,candidate,clock,owner,outcome), (raw_obs,raw_prev,raw_action,raw_outcome)

def fit(path):
    check_manifest(json.loads(REG.read_text()),"preflight")
    verify_committed([ROOT/p for p in SOURCE_FILES]+[ROOT/".github/workflows/edge-action-fit-originals.yml"])
    if OUT.exists():raise RuntimeError("fit originals already exist; no resume/overwrite")
    data=read_fit(path)
    OUT.mkdir(parents=True)
    reg=json.loads(REG.read_text())
    src_hashes={p:sha(ROOT/p) for p in SOURCE_FILES}
    receipts=[]
    for seed in SEEDS:
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        m=ActionQueryEdgeWorld()
        if m.active_parameter_count!=99760:raise RuntimeError("frozen actual parameter count differs")
        m.core.set_train_delta_scale(torch.from_numpy(data["prep"]["delta_scale"]))
        folder=OUT/f"seed-{seed}";folder.mkdir()
        opt=budgeted_optimizer(m,torch.optim.AdamW,lr=.0006,weight_decay=.001,
            budget_manifest=REG,budget_receipt=folder/"live-budget.json",
            budget_seed=seed,budget_architecture="edge_action_query")
        rng=np.random.default_rng(seed)
        schedule=np.stack((rng.integers(0,49,size=(256,8)),rng.integers(32,data["episode_length"]-1,size=(256,8))),axis=-1).astype(np.int64)
        start=time.process_time();losses=[]
        for pairs in schedule:
            (x,u,a,h,owner,target),_=batch(data,pairs)
            opt.zero_grad(set_to_none=True)
            pred,_=m(x,u,a,h,owner)
            scaled=(pred.mean[:,0]-target)/m.core.delta_scale
            loss=scaled.square().mean()
            if not torch.isfinite(loss):raise RuntimeError("nonfinite original loss")
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()):
                raise RuntimeError("invalid full active gradient")
            torch.nn.utils.clip_grad_norm_(m.parameters(),1)
            checked_optimizer_step(opt)
            losses.append(float(loss.detach()))
        sav=folder/"final.pt"
        save_training_checkpoint(dict(
            model=m.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(),python_rng=random.getstate(),
            schedule=torch.from_numpy(schedule),preprocessing=data["prep"],
            source_sha256=source.SHA256,source_files=src_hashes,seed=seed,steps=256,losses=losses),sav)
        receipt=dict(seed=seed,architecture="edge_action_query",
          trainable_parameters=m.active_parameter_count,deployed_parameters=m.active_parameter_count,
          steps=256,training_examples=256*8,
          checkpoint_sha256=sha(sav),optimizer_cpu_seconds=time.process_time()-start)
        (folder/"FIT_RECEIPT.json").write_text(json.dumps(receipt,indent=2)+"\n")
        receipts.append(receipt)
        print(json.dumps(receipt),flush=True)
    index=dict(study_id=reg["study_id"],models=receipts,source_sha256=source.SHA256,
        source_files=src_hashes,fit_episodes=49,
        sealed_internal_report_episodes=10,
        historical_DEV_scored=False,original_TEST_scored=False)
    (OUT/"FIT_INDEX.json").write_text(json.dumps(index,indent=2)+"\n")
    print("ORIGINAL_FITS_READY_TO_COMMIT_BEFORE_SCORING",flush=True)

def score(path,replay=False):
    idx=json.loads((OUT/"FIT_INDEX.json").read_text())
    check_manifest(json.loads(REG.read_text()),"preflight")
    original_files=[OUT/"FIT_INDEX.json"]
    for seed in SEEDS:original_files.extend((OUT/f"seed-{seed}").glob("*"))
    verify_committed(original_files)
    verify_committed([ROOT/p for p in SOURCE_FILES])
    for name,hashv in idx["source_files"].items():
        if sha(ROOT/name)!=hashv:raise RuntimeError("original trained-source hash mismatch")
    data=read_fit(path)
    originals=idx["models"]
    samples=np.array([(ep,t) for ep in range(49,59) for t in range(32,data["episode_length"]-1,8)],dtype=np.int64)
    ids=[];truth=[];all_ref={k:[] for k in ("persistence","logged_target","FIT_servo","velocity")};outputs=[]
    raw_prep=data["prep"]
    with torch.no_grad():
        for seed,row in zip(SEEDS,originals):
            file=OUT/f"seed-{seed}"/"final.pt"
            if sha(file)!=row["checkpoint_sha256"]:raise RuntimeError("frozen checkpoint altered")
            ckpt=torch.load(file,map_location="cpu",weights_only=False)
            m=ActionQueryEdgeWorld();m.load_state_dict(ckpt["model"]);m.eval()
            for name in ("state_mean","state_scale","action_mean","action_scale","delta_scale","servo_gain"):
                np.testing.assert_allclose(np.asarray(ckpt["preprocessing"][name]),np.asarray(raw_prep[name]),rtol=0,atol=0)
            out=[]
            for start in range(0,len(samples),64):
                (x,u,a,h,owner,_),_=batch(data,samples[start:start+64])
                pred,_=m(x,u,a,h,owner)
                y=pred.mean[:,0].numpy()*raw_prep["state_scale"]+raw_prep["state_mean"]
                out.append(y)
            outputs.append(np.concatenate(out))
    for i in range(0,len(samples),64):
        _,(xprev,xx,cmd,obs)=None,(None,None,None,None) # overwritten below
        (_,_,_,_,_,_),raw=batch(data,samples[i:i+64])
        x,prev,cmd,y=raw
        ids.extend(samples[i:i+64].tolist());truth.append(y)
        all_ref["persistence"].append(x)
        all_ref["logged_target"].append(cmd)
        all_ref["FIT_servo"].append(x+raw_prep["servo_gain"]*(cmd-x))
        all_ref["velocity"].append(x+(x-prev))
    target=np.concatenate(truth)
    refs={name:np.concatenate(parts) for name,parts in all_ref.items()}
    scores=lambda values:dict(mse=float(np.mean((values-target)**2)),mae=float(np.mean(np.abs(values-target))),
         moving_mse=float(np.mean((values-target)[np.sqrt(np.mean(((target-refs["persistence"])/raw_prep["state_scale"])**2,-1))>=raw_prep["moving_threshold"]]**2)))
    ref_scores={name:scores(pred) for name,pred in refs.items()}
    strongest=min(ref_scores,key=lambda key:ref_scores[key]["mse"])
    rows=[]
    for seed,pred in zip(SEEDS,outputs):
        m=scores(pred)
        passed=m["mse"]<=.95*ref_scores[strongest]["mse"] and m["moving_mse"]<=ref_scores[strongest]["moving_mse"]
        rows.append(dict(seed=seed,**m,capability_gate=passed))
    result=dict(status="REAL_INTERNAL_FIT_SUBHOLDOUT_EXPOSED_NOT_INDEPENDENT",
       source_sha256=source.SHA256,train_episodes=list(range(49)),
       score_episodes=list(range(49,59)),queries=len(samples),source_cluster_count=10,
       same_source_fitted_references=True,reference_scores=ref_scores,strongest_reference=strongest,
       models=rows,full_gate_passed=all(x["capability_gate"] for x in rows),
       actual_new_model_fits=5,qualifies_real_world_superiority=False,
       qualifies_causal_control=False,verified_executed_motor_timing=False,
       independent_test_scored=False,dev_test_scored=False,
       note="Logged joint target means observational conditional-on-action only; no source independent outcome or causal effect")
    if replay:
        frozen=json.loads((OUT/"RESULTS.json").read_text())
        if result!=frozen:raise RuntimeError("original score replay mismatch")
        np.testing.assert_allclose(np.stack(outputs),
            np.load(OUT/"FORECASTS.npz")["models"],rtol=1e-5,atol=1e-6)
        print("ORIGINAL_ACTION_QUERY_SCORE_REPLAY=PASS")
    else:
        if (OUT/"RESULTS.json").exists():raise RuntimeError("original result exists, no overwrite")
        np.savez_compressed(OUT/"FORECASTS.npz",models=np.stack(outputs),target=target,
            ids=np.asarray(ids,dtype=np.int64),persistence=refs["persistence"],
            logged_target=refs["logged_target"],servo=refs["FIT_servo"])
        (OUT/"RESULTS.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("mode",choices=("fit","score","replay"))
    a.add_argument("--data",default="/tmp/aloha-action-fit-only.parquet");args=a.parse_args()
    torch.set_num_threads(1)
    if args.mode=="fit":fit(Path(args.data))
    else:score(Path(args.data),replay=args.mode=="replay")
