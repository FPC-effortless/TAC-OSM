#!/usr/bin/env python3
"""ME047 original real-world cross-domain goal forecasting and frozen transfer.

Hard boundary: NO unseen ETTm2 bytes fetched before tokenizer/teacher and ALL
eight candidate models trained and SHA-checkpointed. ETTm2 scored just once,
no gradient / retuning / model selection. Source datasets real, no synthetic
evaluation, no fabricated logged actions.
"""
from __future__ import annotations
import argparse,csv,datetime,hashlib,io,json,math,platform,resource,subprocess,time,urllib.request
from pathlib import Path
import numpy as np
import torch
from scripts.me043_train_adaptive import DOMAINS, save,sha
from tac_osm.me047_hybrid_capabilities import (ARMS,ROLES,GOALS,
  TemporalTokenizer,MacroTeacher,CapabilityForecaster,complete_loss,
  normalized_observation,goal_targets,preflight)

SEEDS=(47001,47002)
HORIZON=96;BATCH=32;PRETEXT_UPDATES=600;MACRO_UPDATES=600;FIT_UPDATES=1000
ETTM2_COMMIT="1d16c8f4f943005d613b5bc962e9eeb06058cf07"
ETTM2_BLOB="f9d46027c987d442d2d8f81308d1d8c4d1d35db8"
ETTM2_BYTES=9677236
ETTM2_URI=f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/{ETTM2_COMMIT}/ETT-small/ETTm2.csv"
STUDY="ME047_REAL_TRANSFER_INTENT_MACRO_HYBRID"
torch.set_num_threads(1)


def original_blob_hash(content):
    return hashlib.sha1(b"blob "+str(len(content)).encode()+b"\0"+content).hexdigest()


def load_authentic_source(root):
    sources={}
    for domain,mod in DOMAINS.items():
        dest=root/"source"/domain
        dest.mkdir(parents=True,exist_ok=True)
        z,record=mod.get_source(dest)
        mean=torch.tensor(record["mean_training_only"],dtype=torch.float32)
        std=torch.tensor(record["std_training_only"],dtype=torch.float32)
        original=(z*std+mean).contiguous()
        if not torch.isfinite(original).all():raise ValueError("genuine source contains NaN")
        labels=record["data_columns"]
        roles=[]
        for col in labels:
            s=col.lower()
            if "zone" in s or "power consumption" in s:roles.append(4)
            elif "temperature" in s or s=="ot":roles.append(2)
            elif s in ("hufl","hull","mufl","mull","lufl","lull"):roles.append(1)
            else:roles.append(3)
        sources[domain]={"values":original,"FIT":mod.TR,"DEV":mod.VA,
                         "roles":roles,"provenance":record}
    return sources


def sample_batch(sources,gen,batch=BATCH,fit_only=True):
    domains=tuple(sources)
    inputs=[];targets=[];roles=[]
    # Use exactly one shared seeded sampler stream for all four candidate arms.
    for _ in range(batch):
        d=domains[int(torch.randint(len(domains),(1,),generator=gen))]
        record=sources[d]
        upper=record["FIT"] if fit_only else record["DEV"]
        origin=int(torch.randint(96,upper-HORIZON+1,(1,),generator=gen))
        c=int(torch.randint(record["values"].shape[1],(1,),generator=gen))
        inputs.append(record["values"][origin-96:origin,c])
        targets.append(record["values"][origin:origin+HORIZON,c])
        roles.append(record["roles"][c])
    return torch.stack(inputs),torch.stack(targets),torch.tensor(roles,dtype=torch.long)


def train_shared_sources(root,sources):
    folder=root/"pretrained";folder.mkdir()
    torch.manual_seed(47000)
    tok=TemporalTokenizer()
    opt=torch.optim.AdamW(tok.parameters(),lr=.001,weight_decay=.0001)
    rng=torch.Generator().manual_seed(47000)
    start=time.process_time()
    trace=[]
    for step in range(1,PRETEXT_UPDATES+1):
        x,_,_=sample_batch(sources,rng)
        prefix=x[:,:88]
        mu=prefix.mean(-1,keepdim=True)
        sd=torch.sqrt((prefix-mu).square().mean(-1,keepdim=True)+1e-5)
        opt.zero_grad(set_to_none=True)
        pred=tok.masked_next8((prefix-mu)/sd)
        target=(x[:,88:]-mu)/sd
        loss=(pred-target).square().mean()
        if not torch.isfinite(loss):raise ArithmeticError("invalid real pretext loss")
        loss.backward();torch.nn.utils.clip_grad_norm_(tok.parameters(),1.);opt.step()
        if step%100==0:trace.append({"step":step,"true_past_reconstruction_MSE":float(loss.detach())})
    tokenizer_cpu=time.process_time()-start
    tok_cp=folder/"frozen-tokenizer.pt"
    torch.save({"weights":tok.state_dict(),"optimizer":opt.state_dict(),
       "rng":rng.get_state(),"steps":PRETEXT_UPDATES,"real_source_only":True},tok_cp)
    for par in tok.parameters():par.requires_grad_(False)
    torch.manual_seed(47000)
    teacher=MacroTeacher()
    macroopt=torch.optim.AdamW(teacher.parameters(),lr=.001,weight_decay=.0001)
    rng=torch.Generator().manual_seed(47000)
    start=time.process_time()
    macro_trace=[]
    for step in range(1,MACRO_UPDATES+1):
        x,y,_=sample_batch(sources,rng)
        norm,actual=normalized_observation(x,y)
        bins=actual.reshape(-1,8,12).mean(-1)
        macroopt.zero_grad(set_to_none=True)
        pred=teacher(norm)
        loss=(pred-bins).square().mean()
        if not torch.isfinite(loss):raise ArithmeticError("invalid real macro pretraining")
        loss.backward();torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.);macroopt.step()
        if step%100==0:macro_trace.append({"step":step,"observed_future_macro_MSE":float(loss.detach())})
    macro_cpu=time.process_time()-start
    macro_cp=folder/"frozen-macro-teacher.pt"
    torch.save({"weights":teacher.state_dict(),"optimizer":macroopt.state_dict(),
       "rng":rng.get_state(),"steps":MACRO_UPDATES,"real_source_only":True},macro_cp)
    for par in teacher.parameters():par.requires_grad_(False)
    save(folder/"traces.json",{"tokenizer":trace,"macro":macro_trace})
    record={"tokenizer_file":str(tok_cp.relative_to(root)),
      "macro_file":str(macro_cp.relative_to(root)),
      "tokenizer_sha256":sha(tok_cp),"macro_sha256":sha(macro_cp),
      "original_tokenizer_CPU":tokenizer_cpu,"original_macro_CPU":macro_cpu,
      "tokenizer_steps":PRETEXT_UPDATES,"macro_steps":MACRO_UPDATES,
      "both_trained_exclusively_source_FIT":True,
      "target_domain_not_fetched":True}
    save(folder/"receipt.json",record)
    return tok,teacher,record


def fit_arm(root,sources,tok,teacher,shared,arm,seed):
    folder=root/"models"/f"{arm}-s{seed}";folder.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed)
    net=CapabilityForecaster(arm,tok,teacher)
    optimizer=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                                lr=.001,weight_decay=.0001)
    rng=torch.Generator().manual_seed(seed+2000)
    start=time.process_time();cpu=0.;trace=[]
    for step in range(1,FIT_UPDATES+1):
        x,y,role=sample_batch(sources,rng)
        goal=torch.randint(len(GOALS),(BATCH,),generator=rng)
        t=time.process_time()
        net.train();optimizer.zero_grad(set_to_none=True)
        loss=complete_loss(net,x,y,role,goal)
        if not torch.isfinite(loss):raise ArithmeticError("nonfinite genuine FIT loss")
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(
            [p for p in net.parameters() if p.requires_grad],1.)
        if not torch.isfinite(grad):raise ArithmeticError("nonfinite FIT gradient")
        optimizer.step();cpu+=time.process_time()-t
        if step%100==0:
            trace.append({"step":step,"FIT_true_future_loss":float(loss.detach()),
                          "optimizer_CPU_seconds":cpu})
    cp=folder/"original-model.pt"
    torch.save({"study":STUDY,"arm":arm,"seed":seed,"steps":FIT_UPDATES,
        "weights":net.state_dict(),"optimizer":optimizer.state_dict(),
        "rng":rng.get_state(),"torch_rng":torch.get_rng_state(),
        "shared_tok_sha256":shared["tokenizer_sha256"],
        "shared_macro_sha256":shared["macro_sha256"]},cp)
    row={"arm":arm,"seed":seed,"original_fit_update_count":FIT_UPDATES,
         "FIT_true_observation_windows":FIT_UPDATES*BATCH,
         "FIT_target_future_scalar_values":FIT_UPDATES*BATCH*HORIZON,
         "active_trainable_parameters":net.count_active(),
         "total_parameters_including_frozen":sum(p.numel() for p in net.parameters()),
         "optimizer_CPU_seconds":cpu,"full_fit_cpu_seconds":time.process_time()-start,
         "shared_pretraining_CPU_seconds":
             shared["original_tokenizer_CPU"]+shared["original_macro_CPU"],
         "checkpoint":str(cp.relative_to(root)),"checkpoint_sha256":sha(cp),
         "never_trained_on_target_ETTm2":True}
    save(folder/"FIT-trace.json",trace);save(folder/"summary.json",row)
    print(json.dumps({"event":"ME047_SOURCE_FIT_COMPLETE","arm":arm,"seed":seed,
          "CPU":round(cpu,2),"params":row["active_trainable_parameters"]}),flush=True)
    return row


def windows_for_source(sources,domain):
    s=sources[domain]
    values=s["values"];roles=s["roles"]
    for origin in range(s["FIT"],s["DEV"]-HORIZON+1,96):
        for c in range(values.shape[1]):
            yield values[origin-96:origin,c],values[origin:origin+HORIZON,c],roles[c],origin,c


def download_untouched_ETTm2(root):
    """Called ONLY after source-trained models/checkpoints are all immutable."""
    folder=root/"independent-site";folder.mkdir(exist_ok=True)
    dest=folder/"ETTm2-original.csv"
    request=urllib.request.Request(ETTM2_URI,
                      headers={"User-Agent":"TAC-OSM-ME047-fixed-first-look/1.0"})
    if dest.exists():
        raw=dest.read_bytes()
    else:
        with urllib.request.urlopen(request,timeout=180) as response:
            raw=response.read()
    if len(raw)!=ETTM2_BYTES or original_blob_hash(raw)!=ETTM2_BLOB:
        raise ValueError("third source changed; no ETTm2 scoring performed")
    dest.write_bytes(raw)
    reader=list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    if len(reader)!=69680:raise ValueError("unexpected ETTm2 source size")
    names=list(reader[0])
    if names!=["date","HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"]:
        raise ValueError("unrecognized real second-station schema")
    dates=[datetime.datetime.strptime(v["date"],"%Y-%m-%d %H:%M:%S")
           for v in reader]
    if any((b-a).total_seconds()!=900 for a,b in zip(dates,dates[1:])):
        raise ValueError("unexpected sampling cadence")
    # Parse only the preregistered test origin region, plus its observed history:
    # All other target rows remain unused in all models.
    local=reader[48000:52096+HORIZON]
    original=torch.tensor([[float(r[c]) for c in names[1:]] for r in local],dtype=torch.float32)
    if original.shape!=(4192,7) or not torch.isfinite(original).all():
        raise ValueError("ETTm2 selected actual observation region invalid")
    receipt={"source_git_blob_sha1":ETTM2_BLOB,
        "source_git_commit":ETTM2_COMMIT,"original_bytes":ETTM2_BYTES,
        "original_file_sha256":hashlib.sha256(raw).hexdigest(),
        "observations_for_transfer_only":len(local),
        "full_source_rows":len(reader),"cadence_seconds":900,
        "target_site_FIT_gradients":0,"transfer_origins":[48096,52096,96],
        "same_model_weights_as_source":True}
    save(folder/"original-receipt.json",receipt)
    return original,receipt


def windows_for_heldout(site):
    # site[0] corresponds to original ETTm2 row48000, preserving causal inputs.
    for origin in range(48096,52096,96):
        local=origin-48000
        for c in range(7):
            yield site[local-96:local,c],site[local:local+HORIZON,c],(2 if c==6 else 1),origin,c


@torch.no_grad()
def evaluate_model(net,windows,save_path,seed,domain):
    net.eval()
    rows=list(windows)
    forecasts=[];targets=[];conditional=[];correct_goals=[];persist=[];macro_scores=[]
    labels=[]
    for start in range(0,len(rows),32):
        group=rows[start:start+32]
        x=torch.stack([r[0] for r in group]);y=torch.stack([r[1] for r in group])
        role=torch.tensor([r[2] for r in group],dtype=torch.long)
        norm,actual=normalized_observation(x,y)
        goals=[]
        for g in range(3):
            intent=torch.full((len(x),),g,dtype=torch.long)
            p,guess,macro,mu,sd=net(x,role,intent)
            if g==0:
                forecasts.append(p.numpy())
                targets.append(actual.numpy())
                persist.append(norm[:,-1:].expand_as(actual).numpy())
                macro_scores.append((macro-actual.reshape(-1,8,12).mean(-1)).numpy())
            goals.append(guess.numpy())
        conditional.append(np.stack(goals,axis=1))
        correct_goals.append(np.stack([actual.mean(-1).numpy(),
                                      actual.amax(-1).numpy(),actual[:,-1].numpy()],axis=1))
        labels.extend((int(r[3]),int(r[4])) for r in group)
    pred=np.concatenate(forecasts)
    truth=np.concatenate(targets)
    cond=np.concatenate(conditional);gt=np.concatenate(correct_goals)
    p0=np.concatenate(persist);macro=np.concatenate(macro_scores)
    goalerr=cond-gt
    mse=float(np.mean((pred.astype(np.float64)-truth.astype(np.float64))**2))
    pmse=float(np.mean((p0.astype(np.float64)-truth.astype(np.float64))**2))
    goalMSE=np.mean(goalerr.astype(np.float64)**2,axis=0)
    goalMAE=np.mean(abs(goalerr.astype(np.float64)),axis=0)
    np.savez_compressed(save_path,pred=pred,actual=truth,goal_predictions=cond,
        goal_real=gt,persistence=p0,macro_delta=macro,origin_channel=np.asarray(labels))
    return {"domain":domain,"seed":seed,"real_windows":len(rows),
         "trajectory_MSE":mse,"trajectory_MAE":float(np.mean(np.abs(pred-truth))),
         "persistence_trajectory_MSE":pmse,
         "macro_teacher_bin_MSE":float(np.mean(macro.astype(np.float64)**2)),
         "per_goal_MSE":dict(zip(GOALS,goalMSE.tolist())),
         "per_goal_MAE":dict(zip(GOALS,goalMAE.tolist())),
         "actual_target_values_derived_from_real_measurements":True,
         "saved_predictions":save_path.name,"predictions_SHA256":sha(save_path)}


def score(root,sources,site,records):
    models=[]
    for item in records:
        ck=torch.load(root/item["checkpoint"],weights_only=True,map_location="cpu")
        t=TemporalTokenizer();m=MacroTeacher()
        t.load_state_dict(torch.load(root/"pretrained"/"frozen-tokenizer.pt",weights_only=True)["weights"])
        m.load_state_dict(torch.load(root/"pretrained"/"frozen-macro-teacher.pt",weights_only=True)["weights"])
        net=CapabilityForecaster(item["arm"],t,m)
        net.load_state_dict(ck["weights"],strict=True)
        folder=root/"models"/f'{item["arm"]}-s{item["seed"]}'
        scores={}
        for d in sources:
            path=folder/f"DEV-{d}.npz"
            scores[d]=evaluate_model(net,windows_for_source(sources,d),path,item["seed"],d)
        path=folder/"UNSEEN-ETTm2.npz"
        scores["unseen_ETTm2"]=evaluate_model(net,windows_for_heldout(site),
                 path,item["seed"],"unseen_ETTm2")
        item["scores"]=scores
        save(folder/"summary.json",item)
        models.append(item)
    ranking=sorted(ARMS,key=lambda arm:np.mean([r["scores"]["unseen_ETTm2"]["trajectory_MSE"]
                                       for r in models if r["arm"]==arm]))
    result={}
    for arm in ARMS:
        subset=[r["scores"]["unseen_ETTm2"] for r in models if r["arm"]==arm]
        result[arm]={"trajectory_MSE":float(np.mean([r["trajectory_MSE"] for r in subset])),
              "goal_MSE":{g:float(np.mean([r["per_goal_MSE"][g] for r in subset])) for g in GOALS}}
    baseline=float(np.mean([r["scores"]["unseen_ETTm2"]["persistence_trajectory_MSE"]
                           for r in models]))
    tac=result["hybrid_tac"]
    gate=(tac["trajectory_MSE"]<baseline
          and tac["trajectory_MSE"]<result["hybrid_gru"]["trajectory_MSE"]
          and tac["trajectory_MSE"]<result["hybrid_context"]["trajectory_MSE"]
          and all(tac["goal_MSE"][g]<min(result["hybrid_context"]["goal_MSE"][g],
                                          result["hybrid_gru"]["goal_MSE"][g])
                  for g in GOALS))
    return {"ranking_unseen_ETTm2":ranking,"unseen_site_scores":result,
         "persistence_MSE":baseline,
         "native_better_than_persistence_GRU_context_on_trajectory_and_all_goals":gate,
         "zero_gradient_unseen_physical_station":True,"semantic_free_text_understanding":False,
         "trained_autonomous_action_policy":False,"general_cross_domain_universality":False,
         "parameter_compute_resource_parity_unqualified":True,
         "no_promotion_or_merge":True}


def run(output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    manifest={"study":STUDY,"completed":False,"target_site_opened":False,
      "authentic_sources":["ETTh1","Tetouan"],"independent_site":"ETTm2",
      "no_ETTm2_gradient_updates":True,"source_training_models":[],
      "environment":{"torch":torch.__version__,"python":platform.python_version(),
                     "cpu_capability":torch.backends.cpu.get_cpu_capability()}}
    save(root/"results.json",manifest)
    sources=load_authentic_source(root)
    manifest["source_hashes"]={d:x["provenance"].get("official_csv_sha256") for d,x in sources.items()}
    tok,macro,record=train_shared_sources(root,sources)
    manifest["frozen_shared"]=record
    save(root/"results.json",manifest)
    for seed in SEEDS:
        for arm in ARMS:
            rec=fit_arm(root,sources,tok,macro,record,arm,seed)
            manifest["source_training_models"].append(rec)
            save(root/"results.json",manifest)
    if len(manifest["source_training_models"])!=8:
        raise ValueError("target cannot open with incomplete source training")
    for rec in manifest["source_training_models"]:
        if sha(root/rec["checkpoint"])!=rec["checkpoint_sha256"]:
            raise ArithmeticError("source model changed before transfer target opened")
    if sha(root/record["tokenizer_file"])!=record["tokenizer_sha256"] or \
       sha(root/record["macro_file"])!=record["macro_sha256"]:
        raise ArithmeticError("frozen shared source weights changed")
    manifest["all_training_checkpoints_frozen_before_target_open"]=True
    save(root/"results.json",manifest)
    site,receipt=download_untouched_ETTm2(root)
    manifest["target_site_opened"]=True
    manifest["target_original_provenance"]=receipt
    manifest["selection"]=score(root,sources,site,manifest["source_training_models"])
    manifest["completed"]=True
    save(root/"results.json",manifest)
    print(json.dumps({"event":"ME047_AUTHENTIC_TRANSFER_COMPLETE",
         "unseen_ranking":manifest["selection"]["ranking_unseen_ETTm2"],
         "scores":manifest["selection"]["unseen_site_scores"],
         "gate_pass":manifest["selection"]["native_better_than_persistence_GRU_context_on_trajectory_and_all_goals"],
         "action_policy_trained":False}),flush=True)


def replay(root):
    root=Path(root)
    rec=json.loads((root/"results.json").read_text())
    assert rec["completed"] and rec["all_training_checkpoints_frozen_before_target_open"]
    sources=load_authentic_source(root)
    site,_=download_untouched_ETTm2(root)
    for item in rec["source_training_models"]:
        assert sha(root/item["checkpoint"])==item["checkpoint_sha256"]
        ck=torch.load(root/item["checkpoint"],weights_only=True)
        t=TemporalTokenizer();m=MacroTeacher()
        t.load_state_dict(torch.load(root/"pretrained"/"frozen-tokenizer.pt",weights_only=True)["weights"])
        m.load_state_dict(torch.load(root/"pretrained"/"frozen-macro-teacher.pt",weights_only=True)["weights"])
        model=CapabilityForecaster(item["arm"],t,m)
        model.load_state_dict(ck["weights"],strict=True)
        folder=root/"models"/f'{item["arm"]}-s{item["seed"]}'
        for domain in (*sources,"unseen_ETTm2"):
            with np.load(folder/("UNSEEN-ETTm2.npz" if domain=="unseen_ETTm2"
                                else f"DEV-{domain}.npz")) as old:
                windows=windows_for_heldout(site) if domain=="unseen_ETTm2" else windows_for_source(sources,domain)
                # Verify actual model forecasts and every goal head against original bytes
                temporary=folder/f"replay-{domain}.npz"
                try:
                    result=evaluate_model(model,windows,temporary,item["seed"],domain)
                    with np.load(temporary) as check:
                        for key in ("pred","actual","goal_predictions","goal_real",
                                    "persistence","macro_delta","origin_channel"):
                            assert np.array_equal(check[key],old[key]),(item["arm"],domain,key)
                finally:temporary.unlink(missing_ok=True)
    print(json.dumps({"status":"ME047_ALL_ORIGINAL_REAL_PREDICTIONS_REPLAY_EXACTLY",
         "model_checkpoints":len(rec["source_training_models"]),
         "heldout_gradient_updates":0}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="results/me047")
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--replay",action="store_true")
    args=parser.parse_args()
    if args.self_test: print(json.dumps(preflight(),sort_keys=True),flush=True)
    elif args.replay:replay(args.out)
    else:run(args.out)
