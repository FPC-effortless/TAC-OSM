"""TEMP021: original 500K TAC authentic, previously exposed Vim DEV calibration.

Read-only frozen checkpoints; no model training and no protected TEST.
Run with original researcher-owned long003 archive and matching publisher Vim docs.
This is independent of the TEMP020 scalar source but implements the exact
same inverse-temperature CE/gradient and 64-iteration fixed search contract.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,io,json,math,tempfile
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import torch

ARCHIVE_SHA="cde9a91be40a4e613975c0bd8caa4c19ac24e3d1d0e993db5cacfbd002fecc32"
MODEL_SHA="acf80cb9051b7f2d2c3900d1676eed212890208cd5213088c6751f9cfd24392c"
CAL=("usr_45.txt","usr_50.txt","usr_51.txt")
SCORE=("usr_52.txt","usr_90.txt","usr_toc.txt")
SEEDS=(1701,1702,1703,1704,1705)
def digest(raw):return hashlib.sha256(raw).hexdigest()

def load_original_checkpoint(raw):
    with torch.serialization.safe_globals([np._core.multiarray._reconstruct,
            np._core.multiarray.scalar,np.ndarray,np.dtype,
            np.dtypes.UInt32DType,np.dtypes.Float64DType]):
        return torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)

def validate_moments(z,y,a):
    if type(a) not in (float,int) or not math.isfinite(a) or a<=0:
        raise ValueError("positive finite inverse temperature required")
    if (z.ndim!=2 or not z.shape[0] or z.shape[1]<2 or y.ndim!=1
        or y.shape[0]!=z.shape[0] or z.dtype!=torch.float64 or y.dtype!=torch.long
        or z.device.type!="cpu" or y.device!=z.device):
        raise ValueError("aligned CPU FP64 logits/int64 labels required")
    if not torch.isfinite(z).all() or not ((y>=0)&(y<z.shape[1])).all():
        raise ValueError("nonfinite logits or out-of-vocabulary labels")

def moments(z,y,a):
    validate_moments(z,y,a)
    z=z-z.max(-1,keepdim=True).values
    lp=a*z-torch.logsumexp(a*z,-1,keepdim=True)
    p=lp.exp()
    mu=(p*z).sum(-1)
    gold=z.gather(-1,y[:,None]).squeeze(-1)
    result=(-(lp.gather(-1,y[:,None]).squeeze(-1)).mean().item(),
            (mu-gold).mean().item(),
            (p*(z-mu[:,None]).square()).sum(-1).mean().item())
    if not all(math.isfinite(x) for x in result):raise ValueError("nonfinite scalar moments")
    return result

def fit(z,y):
    low,high=.25,4.
    for _ in range(64):
        mid=(low+high)/2
        if moments(z,y,mid)[1]>0:high=mid
        else:low=mid
    choices=(1.,.25,4.,(low+high)/2)
    return min(((moments(z,y,a)[0],a) for a in choices),
               key=lambda v:(v[0],v[1]!=1.,v[1]))[1]

def score(z,y,a):
    result={}
    for tag,t in (("identity",1.),("calibrated",a)):
        nll=moments(z,y,t)[0]
        prob=torch.softmax(t*z,-1)
        gold=prob.gather(-1,y[:,None]).squeeze(-1)
        result[tag]={
            "nll":nll,"ppl":math.exp(nll),"bits_per_mapped_symbol":nll/math.log(2),
            "raw_bits_per_byte":None,"top1":(prob.argmax(-1)==y).double().mean().item(),
            "top5":(prob.topk(5,-1).indices==y[:,None]).any(-1).double().mean().item(),
            "brier":(1-2*gold+prob.square().sum(-1)).mean().item(),
            "targets":int(y.numel())}
    if any(result["identity"][k]!=result["calibrated"][k] for k in ("top1","top5")):
        raise ValueError("positive scalar changed ranking")
    return result

def require_replay_nll(actual,expected):
    if not math.isfinite(actual) or not math.isfinite(expected) or abs(actual-expected)>1e-6:
        raise ValueError("original 6144-character native score replay FAILED")

def replay(archive:Path,source_root:Path):
    torch.set_num_threads(2)
    if digest(archive.read_bytes())!=ARCHIVE_SHA:
        raise ValueError("original archive SHA256 mismatch")
    with ZipFile(archive) as z:
        if len(z.namelist())!=len(set(z.namelist())):raise ValueError("duplicate ZIP members")
        source_bytes=z.read("tac_language_500k/tac_wave_ced_500k.py")
        if digest(source_bytes)!=MODEL_SHA:raise ValueError("original model-source SHA mismatch")
        protocol_raw=z.read("tac_language_500k/SOURCE_PARTITION_MANIFEST.json")
        original=json.loads(protocol_raw)["source"]
        if tuple(sorted(original["doc_partitions"]["dev"]))!=CAL+SCORE:
            raise ValueError("original six DEV document membership mismatch")
        names=original["doc_partitions"]["fit"]+original["doc_partitions"]["dev"]
        if len(names)!=36:raise ValueError("original 36-document contract missing")
        docs={}
        for name in names:
            raw=(source_root/name).read_bytes()
            if digest(raw)!=original["byte_sha256"][name]:
                raise ValueError("real publisher source SHA drift: "+name)
            if name in CAL+SCORE:
                x=np.frombuffer(raw,dtype=np.uint8).astype(np.int64).copy()
                x[x>=128]=63
                starts=np.linspace(0,len(x)-129,8,dtype=np.int64)
                docs[name]=torch.stack([torch.tensor(x[s:s+129]) for s in starts])
        hashes=json.loads(z.read("tac_language_500k/history_long_003/BUNDLE_FILE_HASHES.json"))["paths"]
        # Only execute the independently byte-pinned original architecture.
        with tempfile.TemporaryDirectory() as directory:
            model_file=Path(directory)/"tac_wave_ced_500k.py"
            model_file.write_bytes(source_bytes)
            spec=importlib.util.spec_from_file_location("original_waveced500k_temp021",model_file)
            mod=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            results=[]
            for seed in SEEDS:
                member=f"tac_language_500k/seed_{seed}.pt"
                blob=z.read(member)
                if digest(blob)!=hashes[member]["sha256"] or len(blob)!=hashes[member]["bytes"]:
                    raise ValueError("original trained checkpoint SHA mismatch")
                old=load_original_checkpoint(blob)
                if old["seed"]!=seed or old["updates"]!=120 or not old["optimizer"]["state"]:
                    raise ValueError("original optimizer, seed or training-step receipt missing")
                required={"model","optimizer","torch_rng","numpy_rng","python_rng",
                          "sampler_rng","seed","samples_sha256","dev","updates"}
                if not required.issubset(old) or not isinstance(old["torch_rng"],torch.Tensor) or not old["torch_rng"].numel():
                    raise ValueError("original checkpoint RNG/replay fields missing")
                if not all(torch.isfinite(v).all() for v in old["model"].values()):
                    raise ValueError("original checkpoint has nonfinite model weights")
                model=mod.WaveCED500K().eval()
                if model.parameter_count()!=499290:raise ValueError("original learned parameter count drift")
                model.load_state_dict(old["model"],strict=True)
                frozen={k:v.detach().clone() for k,v in model.state_dict().items()}
                batches={}
                with torch.no_grad():
                    for name in CAL+SCORE:
                        b=docs[name]
                        logits=model(b[:,:-1]).to(torch.float64).reshape(-1,128).contiguous()
                        label=b[:,1:].reshape(-1).contiguous()
                        if logits.shape!=(1024,128):raise ValueError("unexpected row count")
                        batches[name]=(logits,label)
                def collect(names):
                    return (torch.cat([batches[n][0] for n in names]),
                            torch.cat([batches[n][1] for n in names]))
                all_z,all_y=collect(CAL+SCORE)
                full=moments(all_z,all_y,1.)[0]
                require_replay_nll(full,float(old["dev"]["nll"]))
                cz,cy=collect(CAL)
                sz,sy=collect(SCORE)
                if cy.numel()!=3072 or sy.numel()!=3072:raise ValueError("disjoint role totals changed")
                scalar=fit(cz,cy)
                c=score(cz,cy,scalar);s=score(sz,sy,scalar)
                if c["calibrated"]["nll"]>c["identity"]["nll"]+1e-12:
                    raise ValueError("calibration identity fallback failed")
                docs_score={name:score(*batches[name],scalar) for name in CAL+SCORE}
                if any(p.grad is not None for p in model.parameters()) or any(
                    not torch.equal(v,frozen[k]) for k,v in model.state_dict().items()):
                    raise ValueError("frozen checkpoint model changed during replay")
                results.append({"seed":seed,"checkpoint_sha256":digest(blob),
                    "original_native_dev_nll":float(old["dev"]["nll"]),
                    "replayed_full_dev_nll":full,"inverse_temperature":scalar,
                    "temperature":1/scalar,"calibration":c,"score":s,
                    "per_document":docs_score})
    def avg(role,tag,key):return math.fsum(r[role][tag][key] for r in results)/len(results)
    return {
        "study":"TEMP021_REAL_ORIGINAL_500K_EXPOSED_DEV",
        "status":"LOCAL_ORIGINAL_REPLAY_PASS_EXPOSED_DEV_CALIBRATION_ONLY",
        "source_manifest_sha256":digest(protocol_raw),"original_archive_sha256":ARCHIVE_SHA,
        "original_model_sha256":MODEL_SHA,
        "original_corpus_hashes_verified":36,
        "calibration_documents":list(CAL),"score_documents":list(SCORE),
        "source_sha256":{n:original["byte_sha256"][n] for n in CAL+SCORE},
        "targets_per_seed_per_role":3072,
        "seeds":results,
        "aggregate":{role:{tag:{key:avg(role,tag,key) for key in ("nll","brier","top1","top5")}
                   for tag in ("identity","calibrated")} for role in ("calibration","score")},
        "gates":{"original_local_frozen_replay":"PASS","scoring_document_disjointness":"PASS",
                 "Git_LFS_original_checkpoint_replay":"BLOCKED",
                 "untouched_test":"NOT_RUN","native_predictive_intelligence_gain":"NOT_ESTABLISHED",
                 "full_finite_precision_AD_Lean_proof":"OPEN"}}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive",required=True,type=Path)
    p.add_argument("--source-root",type=Path,default=Path("/usr/share/vim/vim91/doc"))
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    receipt=replay(args.archive,args.source_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    print("TEMP021 original frozen model calibration completed: EXPOSED DEV ONLY")
if __name__=="__main__":main()
