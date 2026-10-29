#!/usr/bin/env python3
"""LANGSPEED044: original-trained, same-host cached Transformer vs streaming TAC.

No new fit/DEV/TEST. Never substitute untrained comparator or silently report
BDH/SSM as cached when their exact checkpoint cache APIs remain unverified.
"""
import argparse,hashlib,importlib.util,io,json,math,platform,statistics,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"research/mathematical-engineering/langstream043"),str(ROOT/"research/mathematical-engineering/langmemory042")]
from incremental import IncrementalTAC043
from candidate import controlled_position
from scripts.langlayer041_real_context_probe import model_for
from scripts.langlayer035_sampling import fit_sources,schedule_receipt

SOURCE_BLOB="935decb17dfac48419a82d84dc03aaf4b0327ecd"
TRANSFORMER_SHA="ab36031debf5b68abc25dfc3c8ca362110854b49877e2b2f19e3df806e7786ae"
SOURCE_COMMIT="60403a5409ff2c3f3b07dd2ca91a7a3e096839c7"

def sha(x):return hashlib.sha256(x).hexdigest()

def unique(folder,pattern):
    files=list(folder.rglob(pattern))
    if len(files)!=1 or files[0].is_symlink():raise ValueError("missing/duplicated frozen artifact: "+pattern)
    return files[0]

class KV:
    """KV-cache adapter for original ByteTransformer 4x96 pre-norm attention."""
    def __init__(self,m):
        if m.training or len(m.layers.layers)!=4 or m.embedding.embedding_dim!=96:raise ValueError("unsupported Transformer graph")
        if any(layer.training or not layer.norm_first or layer.self_attn.num_heads!=4 for layer in m.layers.layers):raise ValueError("unregistered Transformer layout")
        self.m=m
    def pe(self,start,n):
        w=self.m.embedding.weight
        idx=torch.arange(start,start+n,device=w.device,dtype=w.dtype)
        inv=torch.exp(-math.log(10000.)*torch.arange(0,96,2,device=w.device,dtype=w.dtype)/96)
        a=idx[:,None]*inv[None,:]
        return torch.stack((a.sin(),a.cos()),-1).reshape(n,96)
    @staticmethod
    def project(layer,z):
        q,k,v=F.linear(z,layer.self_attn.in_proj_weight,layer.self_attn.in_proj_bias).chunk(3,-1)
        b,t,_=z.shape
        def reshape(a):return a.reshape(b,t,4,24).permute(0,2,1,3).contiguous()
        return reshape(q),reshape(k),reshape(v)
    @torch.no_grad()
    def prefill(self,tokens):
        if tokens.ndim!=2 or tokens.shape[0]!=1 or tokens.shape[1]<1 or tokens.dtype!=torch.long or tokens.min()<0 or tokens.max()>255:raise ValueError("invalid byte prompt")
        n=tokens.shape[1];x=self.m.embedding(tokens)+self.pe(0,n)[None]
        mask=torch.ones(n,n,dtype=torch.bool,device=tokens.device).triu(1)
        cache=[]
        for layer in self.m.layers.layers:
            _,k,v=self.project(layer,layer.norm1(x));cache.append((k,v))
            x=layer(x,src_mask=mask)
        return self.m.head(self.m.norm(x[:,-1])),cache,n
    @torch.no_grad()
    def step(self,token,cache,clock):
        if token.shape!=(1,) or token.dtype!=torch.long or token.min()<0 or token.max()>255:raise ValueError("bad byte")
        if len(cache)!=4 or any(k.shape!=(1,4,clock,24) or v.shape!=k.shape for k,v in cache):raise ValueError("invalid cache")
        x=self.m.embedding(token[:,None])+self.pe(clock,1)[None];updated=[]
        for layer,(previous_k,previous_v) in zip(self.m.layers.layers,cache):
            q,k,v=self.project(layer,layer.norm1(x))
            k=torch.cat((previous_k,k),2);v=torch.cat((previous_v,v),2)
            attn=F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False)
            x=x+layer.dropout1(layer.self_attn.out_proj(attn.transpose(1,2).contiguous().reshape(x.shape)))
            x=x+layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(x))))))
            updated.append((k,v))
        return self.m.head(self.m.norm(x[:,-1])),updated,clock+1
    @staticmethod
    def cache_bytes(cache):
        return sum(k.numel()*k.element_size()+v.numel()*v.element_size() for k,v in cache)

def load(tac_folder,transformer_folder,source_file,fit):
    raw=source_file.read_bytes()
    blob=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
    if blob!=SOURCE_BLOB:raise ValueError("changed original Transformer source")
    spec=importlib.util.spec_from_file_location("original_transformer044",source_file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sample=schedule_receipt(fit,1701,steps=320)["sample_order_sha256"]
    transformer_file=unique(transformer_folder,"causal_transformer_seed1701_fit320.pt")
    traw=transformer_file.read_bytes()
    if sha(traw)!=TRANSFORMER_SHA:raise ValueError("changed Transformer checkpoint")
    t=torch.load(io.BytesIO(traw),map_location="cpu",weights_only=True)
    if (t["study"],t["arm"],t["seed"],t["updates"],t["source_commit"],t["sample_order_sha256"])!=("LANGCOMPARE031","causal_transformer",1701,320,SOURCE_COMMIT,sample):raise ValueError("Transformer training provenance mismatch")
    tm=module.ByteTransformer().eval();tm.load_state_dict(t["model"],strict=True)
    cp=unique(tac_folder,"*.pt");craw=cp.read_bytes()
    receipt=json.loads(unique(tac_folder,"FIT_AND_EXPOSED_DEV_RECEIPT.json").read_text())
    if sha(craw)!=receipt["checkpoint_sha256"]:raise ValueError("TAC checksum differs")
    c=torch.load(io.BytesIO(craw),map_location="cpu",weights_only=True)
    if (c["study"],c["arm"],c["seed"],c["updates"],c["source_commit"],c["sample_order_sha256"])!=("LANGLAYER040","state48_no_graph",1701,320,SOURCE_COMMIT,sample):raise ValueError("TAC training provenance mismatch")
    if any(not v["optimizer"]["state"] or any(k not in v for k in ("rng_torch","rng_numpy","rng_python")) for v in (t,c)):raise ValueError("missing optimizer/RNG")
    cm=model_for("h48_plain").eval();cm.load_state_dict(c["model"],strict=True)
    counts=(sum(p.numel() for p in cm.parameters()),sum(p.numel() for p in tm.parameters()))
    if counts!=(499623,496960) or (max(counts)-min(counts))/max(counts)>.01:raise ValueError("parameter parity failed")
    return {"tac":cm,"transformer":tm},{"tac":sha(craw),"transformer":sha(traw)},sample

def compare(a,b):
    if a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError("invalid logits")
    if not torch.allclose(a,b,atol=3e-5,rtol=1e-4):raise ValueError("full-prefix parity failure")
    return float((a-b).abs().max())

@torch.no_grad()
def predict(m,arm,prompt,owner,parity=False):
    x=torch.tensor([list(prompt)],dtype=torch.long)
    tick=time.perf_counter_ns()
    engine=IncrementalTAC043(m,"cap_all") if arm=="tac" else KV(m)
    setup_ms=(time.perf_counter_ns()-tick)/1e6
    tick=time.perf_counter_ns()
    if arm=="tac":
        y,state=engine.prefill(x,owners=(owner,));size=state.tensor_bytes()
    else:
        y,cache,clock=engine.prefill(x);size=engine.cache_bytes(cache)
    prefill=(time.perf_counter_ns()-tick)/1e6
    err=0.
    if parity:
        if arm=="tac":
            with controlled_position(m,"cap_all"):expected=m(x)[:,-1]
        else:expected=m(x)[:,-1]
        err=compare(y,expected)
    times=[];generated=[]
    for _ in range(16):
        token=y.argmax(-1);generated.append(int(token[0]))
        tick=time.perf_counter_ns()
        if arm=="tac":y,state=engine.step(token,state,owners=(owner,))
        else:y,cache,clock=engine.step(token,cache,clock)
        times.append((time.perf_counter_ns()-tick)/1e6)
        if parity:
            x=torch.cat((x,token[:,None]),1)
            if arm=="tac":
                with controlled_position(m,"cap_all"):expected=m(x)[:,-1]
            else:expected=m(x)[:,-1]
            err=max(err,compare(y,expected))
    final=state.tensor_bytes() if arm=="tac" else engine.cache_bytes(cache)
    return {"parity_abs":err,"setup_ms":setup_ms,"prefill_ms":prefill,
            "step_ms":times,"total_ms":prefill+sum(times),
            "cache_before_bytes":size,"cache_after_bytes":final,"generated_sha":sha(bytes(generated))}

def dist(v):
    v=sorted(v)
    return {"median_ms":statistics.median(v),"p95_ms":v[min(len(v)-1,math.ceil(len(v)*.95)-1)],"raw_ms":v}

def main():
    p=argparse.ArgumentParser()
    for k in ("tac_dir","transformer_dir","transformer_source","output"):p.add_argument("--"+k.replace("_","-"),type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError("refuse to overwrite evidence")
    if torch.cuda.is_initialized():raise ValueError("CPU-only study")
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    fit=fit_sources();models,hashes,order=load(args.tac_dir,args.transformer_dir,args.transformer_source,fit)
    owner=next(k for k in sorted(fit) if len(fit[k])>=544)
    results=[]
    with torch.inference_mode():
        for n in (128,512):
            prompt=fit[owner][:n]
            parity={arm:predict(m,arm,prompt,owner,parity=True)["parity_abs"] for arm,m in models.items()}
            rows={arm:[] for arm in models}
            for i in range(9):
                order_arms=("tac","transformer") if i%2==0 else ("transformer","tac")
                for arm in order_arms:
                    r=predict(models[arm],arm,prompt,owner)
                    if i>=2:rows[arm].append(r)
            for arm,cohort in rows.items():
                if len({r["generated_sha"] for r in cohort})!=1:raise ValueError("greedy nondeterminism across trials")
                steps=[s for r in cohort for s in r["step_ms"]]
                results.append({"arm":arm,"prompt_bytes":n,"parity_max_abs":parity[arm],
                    "decode":dist(steps),"prefill":dist([r["prefill_ms"] for r in cohort]),
                    "total_prefill_decode":dist([r["total_ms"] for r in cohort]),
                    "engine_setup":dist([r["setup_ms"] for r in cohort]),
                    "state_bytes_before":cohort[0]["cache_before_bytes"],
                    "state_bytes_after":cohort[0]["cache_after_bytes"],
                    "generated_sha256":cohort[0]["generated_sha"],
                    "timed_repeats":7,"generated_updates_per_repeat":16})
    result={"study":"LANGSPEED044","status":"INTERNAL_ORIGINAL_CHECKPOINT_CACHE_MEASURED",
        "source":"genuine CPython FIT only; no DEV/TEST","original_sha256":hashes,"sample_order_sha256":order,
        "model_parameters":{"tac":499623,"transformer":496960},"seed":1701,
        "settings":"same ubuntu CPU, eager FP32, batch1, two PyTorch threads",
        "torch":torch.__version__,"platform":platform.platform(),
        "semantics":"growing original prefix; no crop; model-specific greedy continuation",
        "bdh_cached":"NOT_RUN_UNVERIFIED_INTERFACE","ssm_cached":"NOT_RUN_CPU_UNQUALIFIED",
        "no_fit":True,"no_architecture_superiority_claim":True,"protected_TEST":"NOT_ACCESSED","gpu_minutes":0,
        "limits":["single hosted CPU and seed","training FLOPs unmatched; DEV previously exposed",
                  "not mobile energy/production kernel benchmark","no new memory-retrieval intelligence"],
        "results":results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"study":result["study"],"status":result["status"],"latency":[{"arm":r["arm"],"context":r["prompt_bytes"],"step_median_ms":r["decode"]["median_ms"],"prefill_median_ms":r["prefill"]["median_ms"]} for r in results]}),flush=True)

if __name__=="__main__":main()
