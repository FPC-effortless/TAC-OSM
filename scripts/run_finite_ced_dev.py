"""DEV-010: matched dense/sparse inference on learned 768-scalar PLM memory.

Development-only; native comparator and CED reader share initial PLM writer,
conditional executor and task-head parameters but are NOT parameter/FLOP matched.
All answers and evaluator-only task roles remain outside forward().

The *same CED checkpoint* is evaluated twice, dense and sparse: no retraining,
no differing RNG initialization, no changing supervision between CED modes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tac_osm.development.protected_memory import ProtectionModel  # noqa: E402
from tac_osm.development.finite_ced_memory import (  # noqa: E402
    FiniteMemoryCEDModel, copy_common_initialization,
)
from tac_osm.development.noisy_memory import parameter_hash  # noqa: E402
from tac_osm.development.protection_benchmark import (  # noqa: E402
    make_pair, fingerprint, seed_for, tensors,
)

TRAIN_SEEDS = (43, 59, 71)
EVAL_SEEDS = (241, 257, 269)


def train_batch(seed: int, step: int):
    h = (32, 128)[step % 2]
    d = (4, 16)[(step // 2) % 2]
    fraction = (0.0, 1.0)[(step // 4) % 2]
    noise = (.08, .25)[(step // 8) % 2]
    episodes = []
    for pair_idx in range(2):
        role = ("retention", "acquisition")[pair_idx]
        pair = make_pair(seed_for("finite-ced-train", seed, step, pair_idx),
                         h, d, fraction, noise, role)
        episodes.extend((pair.left, pair.right))
    addresses, values, query, current = tensors(episodes)
    target = torch.tensor([e.target for e in episodes], dtype=torch.long)
    return (addresses, values, query, current, target,
            tuple(fingerprint(e) for e in episodes))


def make_models(seed: int):
    torch.manual_seed(seed)
    native = ProtectionModel(mode="always_update")
    ced = FiniteMemoryCEDModel()
    copy_common_initialization(native, ced)
    for key, value in native.state_dict().items():
        if key in ced.state_dict() and not torch.equal(value, ced.state_dict()[key]):
            raise AssertionError(f"incorrect initial parity on {key}")
    return {"native":native,"ced":ced}


def train(seed: int, steps: int):
    models = make_models(seed)
    initial = {name:parameter_hash(model) for name,model in models.items()}
    opts = {name:torch.optim.AdamW((p for p in model.parameters()
                                    if p.requires_grad),lr=.003)
            for name,model in models.items()}
    last_ce = {}
    gradient_checks = {name:False for name in models}
    fingerprints = []
    for step in range(steps):
        a, v, q, c, labels, fps = train_batch(seed,step)
        fingerprints.extend(fps)
        for name, model in models.items():
            model.train()
            state = model.encode(a, v, trace=False)
            logits = model.predict(state,q,c)
            loss = F.cross_entropy(logits,labels)
            opts[name].zero_grad(set_to_none=True)
            loss.backward()
            grad = model.memory.write.weight.grad
            gradient_checks[name] |= bool(
                grad is not None and torch.isfinite(grad).all()
                and grad.abs().sum().item() > 0)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opts[name].step()
            last_ce[name] = float(loss.detach())
    if not all(gradient_checks.values()):
        raise AssertionError("missing gradient to inherited finite memory writer")
    for name, model in models.items():
        model.eval()
        if parameter_hash(model) == initial[name]:
            raise AssertionError("unchanged trained model")
    return models,{"initial_hashes":initial, "final_hashes":{
        name:parameter_hash(model) for name,model in models.items()},
        "writer_gradients_present":gradient_checks,
        "final_train_ce":last_ce,
        "train_visible_fingerprints":fingerprints,
        "total_train_examples":len(fingerprints)}


def eval_conditions(smoke: bool):
    if smoke:
        return [(32,4,0.0,.08,"retention",0)]
    base = [(h,d,f,noise,role,0)
            for h in (32,128) for d in (4,16)
            for f in (0.0,1.0) for noise in (.08,.25)
            for role in ("retention","acquisition")]
    long = [(512,32,f,noise,role,0)
            for f in (0.0,1.0) for noise in (.08,.25)
            for role in ("retention","acquisition")]
    return base + long


@torch.no_grad()
def evaluate(model, eval_seed: int, mode: str, smoke: bool=False):
    """Rows always use *same examples* across modes. Never provide oracle slot."""
    rows=[]
    for h,d,near,noise,role,rep in eval_conditions(smoke):
        seed=seed_for("finite-ced-eval",eval_seed,h,d,near,noise,role,rep)
        pair=make_pair(seed,h,d,near,noise,role)
        episodes=(pair.left,pair.right)
        a,v,q,c=tensors(list(episodes))
        target=torch.tensor([e.target for e in episodes],dtype=torch.long)
        state=model.encode(a,v,trace=False)
        if state.shape != (2,4,8,24):
            raise AssertionError("finite memory capacity changed")
        if mode == "native":
            logits=model.predict(state,q,c)
            logical_work=None
        elif mode in ("ced_dense","ced_sparse"):
            logits=model.predict(state,q,c,second_top_k=(
                2 if mode=="ced_sparse" else None))
            logical_work = vars(model.ced_read.last_work)
        else:
            raise ValueError("bad evaluation mode")
        if logits.shape != (2,2):
            raise AssertionError("task-head shape")
        chosen=logits.argmax(-1)
        # Reset memory control: same public present inputs, no learned past.
        zeros=torch.zeros_like(state)
        if mode=="native":
            reset=model.predict(zeros,q,c)
            swapped=model.predict(state.flip(0),q,c)
        else:
            kw={"second_top_k":2 if mode=="ced_sparse" else None}
            reset=model.predict(zeros,q,c,**kw)
            swapped=model.predict(state.flip(0),q,c,**kw)
        rows.append({
            "condition":{"h":h,"d":d,"near":near,"noise":noise,"role":role,"rep":rep},
            "fingerprints":[fingerprint(e) for e in episodes],
            "target":target.tolist(),
            "task_logits":logits.tolist(),
            "task_prediction":chosen.tolist(),
            "reset_prediction":reset.argmax(-1).tolist(),
            "swapped_prediction":swapped.argmax(-1).tolist(),
            "state_pair_l2":float((state[0]-state[1]).norm()),
            "finite_memory_scalars_per_episode":state[0].numel(),
            "processed_write_events":2*(h+d),
            "ced_logical_work":logical_work,
        })
    return rows


def summarize(rows):
    total=sum(len(r["target"]) for r in rows)
    success=sum(sum(int(x==y) for x,y in zip(r["target"],r["task_prediction"])) for r in rows)
    reset=sum(sum(int(x==y) for x,y in zip(r["target"],r["reset_prediction"])) for r in rows)
    both=sum(int(r["target"]==r["task_prediction"]) for r in rows)
    counts={}
    for role in ("retention","acquisition"):
        cells=[r for r in rows if r["condition"]["role"]==role]
        counts[role]=sum(sum(int(x==y) for x,y in zip(r["target"],r["task_prediction"]))
                         for r in cells)/(2*len(cells))
    return {"episodes":total,"accuracy":success/total,
            "reset_accuracy":reset/total,"both_pair_correct_rate":both/len(rows),
            "role_accuracy":counts,
            "mean_state_pair_l2":sum(r["state_pair_l2"] for r in rows)/len(rows)}


def run(out: Path, *, smoke: bool=False):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    steps=2 if smoke else 72
    seeds=TRAIN_SEEDS[:1] if smoke else TRAIN_SEEDS
    sources=("src/tac_osm/development/finite_ced_memory.py",
             "src/tac_osm/development/noisy_memory.py",
             "src/tac_osm/development/protected_memory.py",
             "src/tac_osm/development/protection_benchmark.py",
             "scripts/run_finite_ced_dev.py",
             "scripts/validate_finite_ced_dev.py")
    report={
        "study":"DEV-010","status":"DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
        "smoke":smoke,"steps":steps,"seeds":list(seeds),
        "encoder":"finite state 768 scalar PLM; query-only CED over four experts",
        "ced_mode_control":"same checkpoint evaluated dense and sparse",
        "source_hashes":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                         for p in sources},
        "records":[],
    }
    for seed,evseed in zip(seeds,EVAL_SEEDS):
        models,training=train(seed,steps)
        record={"seed":seed,"eval_seed":evseed,"training":training,"arms":{}}
        for name,model in models.items():
            file=out/f"{name}-seed{seed}.pt"
            torch.save(model.state_dict(),file)
            record["arms"][name]={"file":file.name,"sha256":hashlib.sha256(file.read_bytes()).hexdigest(),
                                  "parameter_count":sum(p.numel() for p in model.parameters()),
                                  "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad)}
        rows={}
        for mode in ("native","ced_dense","ced_sparse"):
            model=models["native" if mode=="native" else "ced"]
            rows[mode]=evaluate(model,evseed,mode,smoke)
        fps=[[tuple(row["fingerprints"]) for row in rows[mode]]
              for mode in ("native","ced_dense","ced_sparse")]
        if not (fps[0]==fps[1]==fps[2]):
            raise AssertionError("different evaluation examples across arms")
        record["rows"]=rows
        record["summaries"]={mode:summarize(r) for mode,r in rows.items()}
        print("seed",seed, " ".join(
            f"{mode}={record['summaries'][mode]['accuracy']:.2%}"
            for mode in ("native","ced_dense","ced_sparse")),flush=True)
        report["records"].append(record)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--output",type=Path,default=Path("artifacts/DEV-010"))
    args=parser.parse_args()
    run(args.output,smoke=args.smoke)
