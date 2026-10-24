#!/usr/bin/env python3
"""Source-pinned real ALOHA TRAIN+EXPOSED-DEV gate-retention experiment, no TEST.

Contrasts original continuous TAC vs controlled-gate TAC, identical 93,324
parameter weights at initialization and matched genuine observation/action
windows and training steps. Dynamic, writable 8x8 memory retains long history
longer, without new weights. End-to-end real prediction, old-memory-use
ablation, actual gate statistics and loss are scored on previously exposed
13 development episodes only. Independent test and full FLOP parity gates
remain CLOSED regardless of numbers.
"""
from __future__ import annotations
import copy,hashlib,json,os,random,sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/"src"))
from tac_osm.variants.parallel_continuous_world import ParallelTACWorld
from tac_osm.variants.retention_controlled_world import RetentionControlledTACWorld
from train_aloha_real_pilot import load_authentic_records,splits,windows,normalization,pack
from train_aloha_timescale_dev import DATA_REV,DATA_SHA256,metrics,gate_trace_on_real_development


def dev_score_without_address_memory(model,tensors):
    """Ablation of the entire key-dependent associative read/write channel.

    A forward hook zeros k and thus all memory reads; it is removed before
    the next forward call. Input e/action and learned fast/slow maps are
    unchanged. This is only an observed-source DEV diagnostic.
    """
    h=model.address.register_forward_hook(
       lambda _module,_args,out:torch.zeros_like(out))
    try:
        return metrics(model,tensors,early=tensors[0].shape[1]//2)
    finally:h.remove()


def run(length=64,steps=64,seeds=(20261025,20261026)):
    torch.set_num_threads(1)
    dataset,prov=load_authentic_records()
    if prov["dataset_revision_sha"]!=DATA_REV or prov["parquet_sha256"]!=DATA_SHA256:
        raise RuntimeError("original publisher real tabular data changed")
    eps=splits(dataset)
    tr=windows(dataset,eps["train"],length=length,max_windows_per_episode=8)
    dv=windows(dataset,eps["validation"],length=length,max_windows_per_episode=8)
    norm=normalization(tr)
    train,dev=pack(tr,norm),pack(dv,norm)
    scale=(train[2]-train[0]).square().mean(dim=(0,1)).sqrt().clamp_min(1e-5)
    out=[]
    for seed in seeds:
        torch.manual_seed(seed);random.seed(seed)
        baseline=ParallelTACWorld()
        retained=RetentionControlledTACWorld()
        initial=copy.deepcopy(baseline.state_dict())
        retained.load_original_world_weights(baseline)
        assert sum(x.numel() for x in baseline.parameters())==sum(
            x.numel() for x in retained.parameters())==93324
        schedule=torch.randint(0,len(tr),(steps,8),
               generator=torch.Generator().manual_seed(seed))
        digest=hashlib.sha256(json.dumps(schedule.tolist(),separators=(',',':')).encode()).hexdigest()
        metrics_out={}
        for tag,model in (("source_gate",baseline),("controlled_gate",retained)):
            model.set_train_delta_scale(scale)
            opt=torch.optim.AdamW(model.parameters(),lr=.0006,weight_decay=.001)
            last=None
            for idx in schedule:
                obs,act,y=(v[idx] for v in train)
                model.train()
                opt.zero_grad(set_to_none=True)
                loss=(model(obs,act)-y).square().mean()
                if not torch.isfinite(loss).item():raise RuntimeError("nonfinite real training")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step()
                last=float(loss.item())
            result={
              "real_dev_prediction":metrics(model,dev,early=length//2),
              "real_dev_no_address_memory":dev_score_without_address_memory(model,dev),
              "actual_trained_gates":gate_trace_on_real_development(model,dev),
              "training_terminal_loss":last,
              "parameters":sum(p.numel() for p in model.parameters())}
            savedir=Path("tac_retention_checkpoints")
            savedir.mkdir(exist_ok=True)
            checkpoint=savedir/f"{tag}-{seed}.pt"
            torch.save({
                "model":model.state_dict(),"optimizer":opt.state_dict(),
                "source_sha256":DATA_SHA256,
                "schedule":schedule,
                "training_episode_ids":eps["train"],
                "test_episodes_not_used":True,
                "seed":seed,"train_steps":steps,"source":tag},checkpoint)
            result["checkpoint_sha256"]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            metrics_out[tag]=result
        out.append({
          "seed":seed,
          "source_schedule_sha256":digest,
          "source_gate":metrics_out["source_gate"],
          "controlled_gate":metrics_out["controlled_gate"]
        })
    print(json.dumps({
      "status":"REAL_ROBOT_DEV_ONLY_RETENTION_GATE_PILOT_NO_INDEPENDENT_TEST",
      "source_dataset":prov,"original_source_revision_fixed":True,
      "train_episodes":len(eps["train"]),"dev_episodes":len(eps["validation"]),
      "TEST_reaccess_count":0,"train_windows":len(tr),"dev_windows":len(dv),
      "history":length,"same_parameters_per_arm":93324,
      "same_real_data_and_updates":True,
      "complete_training_FLOPs_matched":False,
      "architecture_superiority_qualified":False,
      "world_memory_utility_qualified":False,
      "seed_results":out},sort_keys=True))


if __name__=="__main__":
    run(length=int(os.environ.get("TAC_RETENTION_LENGTH","64")),
        steps=int(os.environ.get("TAC_RETENTION_STEPS","64")))
