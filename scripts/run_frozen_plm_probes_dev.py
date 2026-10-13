"""DEV-011: frozen TAC-OSM learned-memory decoding probes (development-only).

Loads immutable DEV-010 checkpoints by SHA, freezes *all* writer parameters,
then trains small decoder probes on disjoint opposite-past synthetic examples.
Zero-state, random-initialization, memory-swap, and full-history Euclidean
controls distinguish task information from state sensitivity. This is NOT
a proof of information absence, and probe training is not world-model training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import torch
from torch import Tensor, nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tac_osm.development.protected_memory import ProtectionModel  # noqa: E402
from tac_osm.development.finite_ced_memory import FiniteMemoryCEDModel  # noqa: E402
from tac_osm.development.noisy_memory import parameter_hash  # noqa: E402
from tac_osm.development.protection_benchmark import (  # noqa: E402
    make_pair, seed_for, fingerprint, tensors, euclidean,
)
from run_finite_ced_dev import eval_conditions  # noqa: E402

WRITERS = ("native_trained", "ced_trained", "native_untrained")
PROBES = ("keyed_linear", "keyed_mlp", "state_mlp")
SEEDS = (43, 59, 71)
EVAL_SEEDS = (241, 257, 269)
# Prior GitHub Actions run 38362662234, artifact 11795975699.
PARENT_SHA256 = "a15685af62122a876c9d990fd877c060c22e41f2d81d80bd47f2426ab558fc56"


def prepare_parent(archive: Path) -> dict:
    if hashlib.sha256(archive.read_bytes()).hexdigest() != PARENT_SHA256:
        raise AssertionError("DEV-010 parent archive hash mismatch")
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        expected = {"report.json"} | {f"{mode}-seed{seed}.pt"
                                      for mode in ("native", "ced") for seed in SEEDS}
        if names != expected:
            raise AssertionError("DEV-010 archive has missing or unexpected members")
        report = json.loads(z.read("report.json"))
        if (report.get("study") != "DEV-010" or report.get("steps") != 72
                or report.get("seeds") != list(SEEDS)
                or len(report["records"]) != len(SEEDS)):
            raise AssertionError("unrecognized parent report")
        for record in report["records"]:
            seed = record["seed"]
            for mode in ("native","ced"):
                arm = record["arms"][mode]
                data = z.read(f"{mode}-seed{seed}.pt")
                if arm["file"] != f"{mode}-seed{seed}.pt" or (
                        hashlib.sha256(data).hexdigest() != arm["sha256"]):
                    raise AssertionError("changed parent checkpoint content")
    return report


def open_writer(archive: Path, report: dict, seed: int, origin: str):
    if origin not in WRITERS:
        raise ValueError("unknown writer origin")
    if origin == "native_untrained":
        torch.manual_seed(seed)
        model = ProtectionModel(mode="always_update")
        record = next(r for r in report["records"] if r["seed"] == seed)
        if parameter_hash(model) != record["training"]["initial_hashes"]["native"]:
            raise AssertionError("untrained writer is not DEV-010's starting state")
    else:
        mode = "native" if origin == "native_trained" else "ced"
        model = (ProtectionModel(mode="always_update")
                 if mode == "native" else FiniteMemoryCEDModel())
        record = next(r for r in report["records"] if r["seed"] == seed)
        with zipfile.ZipFile(archive) as z:
            import io
            checkpoint = torch.load(
                io.BytesIO(z.read(f"{mode}-seed{seed}.pt")),
                map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint, strict=True)
        if parameter_hash(model) != record["training"]["final_hashes"][mode]:
            raise AssertionError("parent tensor provenance mismatch")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


class DecoderProbe(nn.Module):
    """Uses only fixed-size frozen writer state, public query and current bit."""
    def __init__(self, kind: str):
        super().__init__()
        if kind not in PROBES:
            raise ValueError("bad probe")
        self.kind = kind
        input_dim = (768 if kind == "state_mlp" else 32) + 12 + 2
        if kind == "keyed_linear":
            self.head = nn.Linear(input_dim, 2)
        else:
            self.head = nn.Sequential(nn.Linear(input_dim, 64), nn.GELU(),
                                      nn.Linear(64, 2))

    def forward(self, feature: Tensor) -> Tensor:
        expected = (768 if self.kind == "state_mlp" else 32) + 14
        if feature.ndim != 2 or feature.shape[1] != expected:
            raise ValueError("invalid frozen feature shape")
        return self.head(feature)


@torch.no_grad()
def frozen_features(writer, states: Tensor, address: Tensor,
                    current: Tensor) -> dict[str, Tensor]:
    if states.ndim != 4 or states.shape[1:] != (4, 8, 24):
        raise AssertionError("persistent memory unexpectedly changed capacity")
    if address.shape != (states.shape[0], 12) or current.shape != (states.shape[0],):
        raise AssertionError("bad public query shape")
    keys = writer.memory.keys(address)
    reads = torch.einsum("bevk,bek->bev", states, keys).reshape(-1, 32)
    present = torch.cat((address, F.one_hot(current, 2).float()), dim=-1)
    return {"keyed": torch.cat((reads, present), dim=-1),
            "state": torch.cat((states.flatten(1), present), dim=-1)}


def training_cases(seed: int, smoke: bool):
    count = 4 if smoke else 128
    for i in range(count):
        h = (32,128)[i % 2]
        d = (4,16)[(i//2)%2]
        near = (0.0,1.0)[(i//4)%2]
        noise = (.08,.25)[(i//8)%2]
        role = ("retention","acquisition")[(i//16)%2]
        yield ("train", i, h,d,near,noise,role,
               seed_for("dev011-probe-train",seed,i,h,d,near,noise,role))


def evaluation_cases(seed: int, smoke: bool):
    conditions = eval_conditions(smoke)
    repeats = 1 if smoke else 2
    for repeat in range(repeats):
        for i, (h,d,near,noise,role,_) in enumerate(conditions):
            yield ("eval", repeat*len(conditions)+i, h,d,near,noise,role,
                   seed_for("dev011-probe-eval",seed,repeat,i,h,d,near,noise,role))


@torch.no_grad()
def encode_examples(writer, cases):
    rows = []
    by_kind = {"keyed":[], "state":[]}
    for partition, idx, h,d,near,noise,role,task_seed in cases:
        pair = make_pair(task_seed,h,d,near,noise,role)
        eps=(pair.left,pair.right)
        addresses, values, query, current = tensors(list(eps))
        state = writer.encode(addresses, values, trace=False)
        feats = frozen_features(writer,state,query,current)
        for kind in by_kind:
            by_kind[kind].append(feats[kind].detach().cpu())
        predictions = [euclidean(ep) for ep in eps]
        rows.append({
            "partition":partition, "index":idx, "h":h, "d":d,
            "near":near, "noise":noise, "role":role,
            "fingerprints":[fingerprint(ep) for ep in eps],
            "target":[int(ep.target) for ep in eps],
            "oracle_predictions":[int(pred[0]) for pred in predictions],
            "oracle_inspected":[h+d]*2,
            "state_pair_l2":float((state[0]-state[1]).norm()),
        })
    features={k:torch.cat(v,dim=0) for k,v in by_kind.items()}
    if any(x.shape[0]!=2*len(rows) for x in features.values()):
        raise AssertionError("incomplete example extraction")
    return rows,features


def feature_for(kind: str, features: dict[str,Tensor]) -> Tensor:
    return features["state" if kind=="state_mlp" else "keyed"]


def fit_probe(kind: str, source: dict[str,Tensor], rows: list[dict], seed: int,
              smoke: bool):
    # Paired examples remain paired in the data; no pair member is split
    # between training and held-out evaluation.
    torch.manual_seed(seed)
    model = DecoderProbe(kind)
    initial = {k:v.detach().clone() for k,v in model.state_dict().items()}
    x = feature_for(kind, source)
    y = torch.tensor([v for row in rows for v in row["target"]],dtype=torch.long)
    if x.shape[0]!=y.numel():
        raise AssertionError("training samples mismatch")
    opt=torch.optim.AdamW(model.parameters(),lr=0.002,weight_decay=.0001)
    steps=2 if smoke else 240
    final_loss=0.
    for step in range(steps):
        # Fixed seeded sampler; all probe architectures see exactly the same
        # cases for given seed. The writer is never on the optimizer.
        torch.manual_seed(seed_for("dev011-probe-batch",seed,step)%(2**31))
        take=torch.randint(0,y.numel(),(min(64,y.numel()),))
        model.train()
        logits=model(x[take])
        loss=F.cross_entropy(logits,y[take])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        final_loss=float(loss.detach())
    model.eval()
    if not any(not torch.equal(initial[k],v) for k,v in model.state_dict().items()):
        raise AssertionError("probe weights never changed")
    return model,final_loss


@torch.no_grad()
def evaluate_probe(model: DecoderProbe, features: dict[str,Tensor],
                   rows: list[dict], zero_memory: bool=False):
    x=feature_for(model.kind,features).clone()
    # The query and current bit stay unchanged; only the state-dependent
    # features are cleared in the zero-memory control.
    if zero_memory:
        x[:, :-14] = 0.
    logits = model(x)
    pred = logits.argmax(-1).tolist()
    # Pair-wise state swap, keeping query/present the same. Because each
    # opposite-past pair shares present, swapping all feature dims *except*
    # final 14 present dims must harm a genuine memory-derived predictor.
    flipped=x.clone()
    flipped[:, :-14]=x.reshape(-1,2,x.shape[1]).flip(1).reshape_as(x)[:, :-14]
    swapped=model(flipped).argmax(-1).tolist()
    raw=[]
    for i,row in enumerate(rows):
        raw.append({**row,"prediction":pred[2*i:2*i+2],
                    "swapped_prediction":swapped[2*i:2*i+2]})
    total=2*len(raw)
    successes=sum(sum(int(a==b) for a,b in zip(r["target"],r["prediction"])) for r in raw)
    swap_success=sum(sum(int(a==b) for a,b in zip(r["target"],r["swapped_prediction"])) for r in raw)
    return {"episodes":total,"accuracy":successes/total,
            "swapped_accuracy":swap_success/total,
            "both_correct_pairs":sum(int(r["prediction"]==r["target"]) for r in raw),
            "rows":raw}


def run(parent_archive: Path, output: Path, smoke: bool=False):
    torch.set_num_threads(2)
    parent=prepare_parent(parent_archive)
    output.mkdir(parents=True,exist_ok=True)
    source_files=("scripts/run_frozen_plm_probes_dev.py",
                  "scripts/validate_frozen_plm_probes_dev.py",
                  "src/tac_osm/development/noisy_memory.py",
                  "src/tac_osm/development/protection_benchmark.py")
    report={
        "study":"DEV-011", "status":"DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
        "parent_artifact_sha256":PARENT_SHA256, "smoke":smoke,
        "seeds":list(SEEDS[:1] if smoke else SEEDS),
        "origins":list(WRITERS),"probe_types":list(PROBES),
        "source_sha256":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                         for p in source_files},
        "records":[],
    }
    for seed in report["seeds"]:
        record={"seed":seed,"origins":{}}
        for origin in WRITERS:
            writer=open_writer(parent_archive,parent,seed,origin)
            state_before=parameter_hash(writer)
            train_rows,train_feat=encode_examples(writer,training_cases(seed,smoke))
            eval_rows,eval_feat=encode_examples(writer,evaluation_cases(
                EVAL_SEEDS[SEEDS.index(seed)],smoke))
            tfps={fp for r in train_rows for fp in r["fingerprints"]}
            efps={fp for r in eval_rows for fp in r["fingerprints"]}
            if tfps & efps or len(tfps)!=2*len(train_rows) or len(efps)!=2*len(eval_rows):
                raise AssertionError("train/test duplicate visible observations")
            if any(r["target"][0]==r["target"][1] for r in train_rows+eval_rows):
                raise AssertionError("opposite-past pair target mismatch")
            origin_result={"frozen_writer_parameter_hash":state_before,
                           "train_rows":train_rows,"eval_rows":eval_rows,
                           "probes":{}}
            for kind in PROBES:
                probe,training_ce=fit_probe(kind,train_feat,train_rows,seed,smoke)
                file=f"{origin}-{kind}-seed{seed}.pt"
                torch.save(probe.state_dict(),output/file)
                ordinary=evaluate_probe(probe,eval_feat,eval_rows)
                reset=evaluate_probe(probe,eval_feat,eval_rows,zero_memory=True)
                origin_result["probes"][kind]={
                    "file":file,
                    "sha256":hashlib.sha256((output/file).read_bytes()).hexdigest(),
                    "parameters":sum(p.numel() for p in probe.parameters()),
                    "last_train_ce":training_ce,
                    "heldout":ordinary,
                    "reset_accuracy":reset["accuracy"]}
                print(f"seed={seed} writer={origin} probe={kind}"
                      f" accuracy={ordinary['accuracy']:.3f}"
                      f" swapped={ordinary['swapped_accuracy']:.3f}"
                      f" reset={reset['accuracy']:.3f}",flush=True)
            if parameter_hash(writer)!=state_before:
                raise AssertionError("frozen writer was changed by fitting probes")
            record["origins"][origin]=origin_result
        report["records"].append(record)
    (output/"report.json").write_text(json.dumps(report,indent=2))
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--parent",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path("artifacts/DEV-011"))
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    run(args.parent,args.output,args.smoke)
