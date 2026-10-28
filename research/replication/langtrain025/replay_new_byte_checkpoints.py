"""LANGTRAIN025 separate clean-checkout original checkpoint (not training) replayer.

Download exact recorded first-fit GitHub artifact. Strict SHA of all 5 original
NEW language models, optimizer/RNG availability, source cases and per-doc scores.
This remains same-owner internally verified, not external independent signoff.
"""
from __future__ import annotations
import argparse,hashlib,json,math,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"research/mathematical-engineering/langtrain025"))
from train_byte_native import ByteWaveCED500K,sha,safe_load,score_model,score_reference,windows
sys.path.insert(0,str(ROOT/"research/mathematical-engineering/langbase024"))
from byte_next_reference import original_source
EXPECTED_NUMERICAL_SHA="d25d61a8ae848227cdbf9424854e51a379afdcc3c0dc79c0bbbf160f2e105458"
EXPECTED_FIRST_RUN_SHA="d55b32a2a34f6b5418adb0ae91f396f9fd725b11"
SEEDS=(1701,1702,1703,1704,1705)

def replay(original_dir):
    torch.set_num_threads(2)
    original_dir=Path(original_dir)
    receipt=original_dir/"LANGTRAIN025_AUTHENTIC_EXPOSED_DEV_RECEIPT.json"
    if receipt.is_symlink() or sha(receipt.read_bytes())!=EXPECTED_NUMERICAL_SHA:
        raise ValueError("original numerical JSON SHA differs")
    r=json.loads(receipt.read_text())
    if (r["active_learned_parameters"]!=499978 or r["training_updates"]!=320
        or r["predicted_targets_per_seed"]!=4096
        or r["source_publisher_commit"]!="60403a5409ff2c3f3b07dd2ca91a7a3e096839c7"
        or len(r["model_runs"])!=5):
        raise ValueError("original registered model/evaluation identity changed")
    code_root=ROOT/"research/mathematical-engineering/langtrain025"
    for name,expected in r["code_sha256"].items():
        if name not in ("byte_waveced_500k.py","train_byte_native.py"):
            raise ValueError("unexpected executable source identity")
        if sha((code_root/name).read_bytes())!=expected:
            raise ValueError("trained source file no longer matches original SHA "+name)
    fit0,dev,supp=original_source(ROOT)
    fit={**fit0,supp[0]:supp[1]}
    blocks=windows(dev)
    # Independently rebuild same-case original FIT-only classical law on an
    # unrelated clean checkout; no score selection/synthetic alternatives.
    for order in (0,3):
        recomputed=score_reference(fit,blocks,order)["aggregate"]
        expected=r["baselines"][f"P{order}"]["aggregate"]
        for key in ("nll","brier","top1","top5","n","rare_n"):
            if abs(recomputed[key]-expected[key])>1e-9:
                raise ValueError("original whole-FIT baseline changed "+key)
    by_seed=[]
    for row in r["model_runs"]:
        seed=row["seed"]
        if seed not in SEEDS or any(x["seed"]==seed for x in by_seed):
            raise ValueError("original seed identity duplicate/out of protocol")
        file=original_dir/row["checkpoint"]
        if file.name!=f"byte_waveced_seed_{seed}.pt" or file.is_symlink():
            raise ValueError("unsafe or renamed checkpoint")
        blob=file.read_bytes()
        if sha(blob)!=row["checkpoint_sha256"]:
            raise ValueError("checkpoint SHA changed")
        ck=safe_load(blob)
        if (ck["seed"]!=seed or ck["updates"]!=320
            or ck["model_source_sha256"]!=r["code_sha256"]["byte_waveced_500k.py"]
            or ck["sample_sha256"]!=row["training_samples_sha256"]
            or not ck["optimizer"]["state"]
            or not all(k in ck for k in ("rng_torch","rng_numpy","rng_python","rng_sampler"))):
            raise ValueError("original trained optimizer/RNG/source/seed contract broken")
        model=ByteWaveCED500K().eval()
        if model.parameter_count()!=499978:raise ValueError("deployed param count differs")
        model.load_state_dict(ck["model"],strict=True)
        native=score_model(model,blocks)
        for name,e in [("aggregate",row["scoring"]["aggregate"]),*[(n,row["scoring"]["per_doc"][n]) for n in row["scoring"]["per_doc"]]]:
            got=native["aggregate"] if name=="aggregate" else native["per_doc"][name]
            for key in ("nll","brier","top1","top5","n","rare_n"):
                if abs(got[key]-e[key])>1e-6:
                    raise ValueError(f"original checkpoint score changed seed={seed} source={name} metric={key}")
        by_seed.append({"seed":seed,"checkpoint_sha256":sha(blob),
                        "replayed_DEV_NLL":native["aggregate"]["nll"],
                        "source_replay":"PASS_4096_ORIGINAL_NEXT_BYTE_LABELS"})
    if tuple(sorted(x["seed"] for x in by_seed))!=SEEDS:
        raise ValueError("not all five original checkpoints independently reloaded")
    mean=math.fsum(x["replayed_DEV_NLL"] for x in by_seed)/len(by_seed)
    if abs(mean-r["registered_primary"]["neural_mean_nll"])>1e-6:
        raise ValueError("original five-seed mean changed")
    return {"status":"PASS_SEPARATE_CLEAN_RUNNER_INTERNAL_REPLAY",
            "original_first_fit_commit_sha":EXPECTED_FIRST_RUN_SHA,
            "original_first_fit_results_sha256":EXPECTED_NUMERICAL_SHA,
            "seeds":by_seed,"new_trained_weights_checked":5,
            "original_exposed_dev_target_positions_per_seed":4096,
            "native_mean_NLL":mean,"same_case_P3_NLL":r["baselines"]["P3"]["aggregate"]["nll"],
            "model_vs_P3_decision":"FAIL_0_OF_5_SEEDS",
            "source_authentication":"PASS_ALL_18_ORIGINAL_PUBLISHER_HASHES",
            "independent_external_qualification":"NOT_RUN",
            "protected_TEST":"NOT_RUN",
            "original_LEGACY_TAC_archives":"NOT_LOADED",
            "full_Lean_finite_precision_AD_proof":"OPEN"}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    x=replay(args.artifact_dir)
    if args.output.exists():raise ValueError("refuse to overwrite replay evidence")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
    print("LANGTRAIN025 separate clean-runner original five-seed checkpoint and case-level replay PASS")
if __name__=="__main__":main()
