"""GRAPH010 genuine Vim FIT-only learner; source- and sample-hash gated.

No use of GRAPH009's six previously evaluated Python-source files or
independent protected TEST. Requires caller-supplied original Vim9.1
PROTOCOL.json and exact genuine chapter files. All source hashes checked.
Fitted checkpoints must be cold-replayed and preserved before promotion.

Example:
 python fit_learned_query_v6.py --original-protocol /path/PROTOCOL.json \
   --vim-root /usr/share/vim/vim91/doc --output /tmp/graph010
"""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from learned_context_query_v6 import LearnedContextQuery
from bayesian_cache_scan_v5 import ScannedCountCache

SEEDS=(1701,1702,1703,1704,1705)
WINDOW=129
STEPS=256
BATCH=4

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ascii_tokens(path):
    return np.frombuffer(bytes(v if v<128 else 63 for v in path.read_bytes()),
                         dtype=np.uint8).astype(np.int64)

def get_fit(protocol,root):
    manifest=protocol["source"]["byte_sha256"]
    parts=protocol["source"]["doc_partitions"]
    fit=parts["fit"];dev=parts["dev"]
    if len(fit)!=30 or len(dev)!=6 or set(fit)&set(dev) or len(manifest)!=36:
        raise ValueError("expected original 30/6 source partitions")
    for name,h in manifest.items():
        if Path(name).name!=name or sha(root/name)!=h:
            raise ValueError("real Vim chapter digest mismatch: "+name)
    docs={name:ascii_tokens(root/name) for name in fit}
    if any(len(d)<=WINDOW+1 for d in docs.values()):
        raise ValueError("authentic FIT document too short")
    return fit[:16],fit[16:],docs

def batch_sample(keys,docs,rng,batch=BATCH):
    weight=np.asarray([len(docs[k])-WINDOW-1 for k in keys],dtype=np.float64)
    weight/=weight.sum()
    chosen=rng.choice(len(keys),size=batch,p=weight)
    xs=np.empty((batch,WINDOW),dtype=np.int64)
    receipt=hashlib.sha256()
    for i,which in enumerate(chosen):
        d=docs[keys[which]]
        start=int(rng.integers(0,len(d)-WINDOW))
        xs[i]=d[start:start+WINDOW]
        receipt.update(keys[which].encode()+b"\0")
        receipt.update(start.to_bytes(8,"little"))
    receipt.update(xs.tobytes())
    return torch.from_numpy(xs),receipt.digest()

def fixed_validation(keys,docs):
    batches=[]
    for name in keys:
        d=docs[name]
        for start in np.linspace(0,len(d)-WINDOW-1,3,dtype=np.int64):
            batches.append(torch.from_numpy(d[start:start+WINDOW]))
    return [torch.stack(batches[i:i+BATCH]) for i in range(0,len(batches),BATCH)]

def evaluate(query,windows,prior):
    exact_cache=ScannedCountCache(512)
    total=dict(nll_exact=0.,nll_learned=0.,nll_no_train=0.,
               nll_prior=0.,num=0,exact_hits=0,approx_hits=0,
               exact_miss_n=0,exact_miss_prior=0.,exact_miss_learned=0.)
    with torch.no_grad():
      for batch in windows:
        if batch.shape[0]<1:continue
        ids=tuple("FIT-VALID-"+str(i) for i in range(batch.shape[0]))
        source,_=exact_cache.scan_counts(batch,source_ids=ids)
        c=source["counts"][:,:,1].to(prior.dtype)
        n=source["totals"][:,:,1].to(prior.dtype)
        copy=(c+.05*prior[None,None,:])/(n[...,None]+.05)
        exact_gate=torch.where(n>0,.6,0.)
        base=(1-exact_gate[...,None])*prior+.6*(n>0)[...,None]*copy
        learned=query.predict_window(batch)
        add=learned["hit"]& (n==0)
        second=(1-.6*add[...,None])*base + .6*add[...,None]*learned["posterior"]
        target=batch[:,1:,None]
        valid=slice(None,-1)
        scores_prior=prior[batch[:,1:]]
        lp=base[:,valid].gather(-1,target).squeeze(-1).clamp_min(1e-30)
        ln=second[:,valid].gather(-1,target).squeeze(-1).clamp_min(1e-30)
        total["nll_prior"]+=float((-scores_prior.clamp_min(1e-30).log()).sum())
        total["nll_exact"]+=float((-lp.log()).sum())
        total["nll_learned"]+=float((-ln.log()).sum())
        total["num"]+=target.numel()
        total["exact_hits"]+=int((n[:,valid]>0).sum())
        total["approx_hits"]+=int(add[:,valid].sum())
        misses=n[:,valid]==0
        total["exact_miss_n"]+=int(misses.sum())
        total["exact_miss_prior"]+=float((-lp.log())[misses].sum())
        total["exact_miss_learned"]+=float((-ln.log())[misses].sum())
    n=total["num"];nm=total["exact_miss_n"]
    if not n or not nm:raise ValueError("no validation labels or misses")
    return {"n":n,"exact_only_nll":total["nll_exact"]/n,
            "query_on_miss_nll":total["nll_learned"]/n,
            "fit_unigram_nll":total["nll_prior"]/n,
            "exact_hit_rate":total["exact_hits"]/n,
            "approx_hit_rate":total["approx_hits"]/n,
            "exact_miss_n":nm,
            "miss_only_exact_nll":total["exact_miss_prior"]/nm,
            "miss_only_query_nll":total["exact_miss_learned"]/nm}

def main(args):
    torch.set_num_threads(1)
    root=Path(args.vim_root);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    proto=json.loads(Path(args.original_protocol).read_text())
    train_names,valid_names,docs=get_fit(proto,root)
    counts=np.full(128,.5,dtype=np.float64)
    for name in train_names:
        counts+=np.bincount(docs[name],minlength=128)
    prior=torch.tensor(counts/counts.sum(),dtype=torch.float32)
    if not bool(torch.isfinite(prior).all()):raise ValueError("FIT prior not finite")
    validation=fixed_validation(valid_names,docs)
    specification={"experiment":"GRAPH010","status":"FIT-only exploratory",
       "original_protocol_sha256":sha(args.original_protocol),
       "source_sha256":proto["source"]["byte_sha256"],
       "fit_query_train":train_names,"fit_query_validation":valid_names,
       "seeds":SEEDS,"steps":STEPS,"batch":BATCH,"window":WINDOW,
       "query_params":2816,"base_unchanged_params":499290,
       "cache_events_per_owner":512,"topK":8,"lambda":.6,
       "tau":.05,"contrastive":"FIT positive-continuation InfoNCE",
       "untouched_TEST":"no independent TEST used"}
    (out/"PROTOCOL.json").write_text(json.dumps(specification,indent=2,sort_keys=True))
    results={"protocol":specification,"runs":[]}
    for seed in SEEDS:
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        q=LearnedContextQuery(prior,capacity=512)
        if q.learned_parameters()!=2816:raise ValueError("unexpected parameter count")
        random_untrained=LearnedContextQuery(prior,capacity=512)
        # Same architecture and seed's initial weights, but no optimizer.
        random_untrained.load_state_dict(q.state_dict())
        untrained=evaluate(random_untrained,validation,prior)
        opt=torch.optim.AdamW(q.parameters(),lr=.001,weight_decay=.01)
        input_sha=hashlib.sha256();positive=miss=attempts=0
        t0=time.perf_counter()
        for step in range(STEPS):
            x,receipt=batch_sample(train_names,docs,rng)
            input_sha.update(receipt)
            loss,info=q.contrastive_window_loss(x)
            attempts+=info["training_queries"]
            miss+=info["nonexact_miss_queries"]
            positive+=info["positive_evidence_queries"]
            opt.zero_grad(set_to_none=True);loss.backward()
            if not bool(torch.isfinite(loss)):raise ValueError("nonfinite training objective")
            torch.nn.utils.clip_grad_norm_(q.parameters(),1.)
            opt.step()
        val=evaluate(q.eval(),validation,prior)
        ck={"source":"GRAPH010 FIT-only query","seed":seed,
            "model":q.state_dict(),"optimizer":opt.state_dict(),
            "sampler_sha256":input_sha.hexdigest(),
            "torch_rng":torch.get_rng_state(),
            "numpy_state_json":json.dumps(rng.bit_generator.state),
            "protocol":specification}
        path=out/f"graph010_query_{seed}.pt";torch.save(ck,path)
        loaded=torch.load(path,map_location="cpu",weights_only=True)
        replay=LearnedContextQuery(prior,capacity=512).eval()
        replay.load_state_dict(loaded["model"],strict=True)
        repeat=evaluate(replay,validation,prior)
        if any(abs(float(val[k])-float(repeat[k]))>1e-8 for k in val):
            raise ValueError("cold query checkpoint replay differs")
        result={"seed":seed,"fitted":val,"initial_untrained_query":untrained,
                "sampler_sha256":input_sha.hexdigest(),
                "checkpoint":path.name,"checkpoint_sha256":sha(path),
                "training_positive_evidence_rate":positive/max(attempts,1),
                "training_nonexact_miss_rate":miss/max(attempts,1),
                "train_wall_s":time.perf_counter()-t0,
                "cold_replay":"PASS"}
        results["runs"].append(result)
        (out/"RESULTS.json").write_text(json.dumps(results,indent=2,sort_keys=True))
        print("seed",seed,"exact",val["exact_only_nll"],
              "learned",val["query_on_miss_nll"],
              "untrained",untrained["query_on_miss_nll"],
              "positive_evidence_rate",result["training_positive_evidence_rate"],flush=True)
    results["summary"]={"n_seeds":len(results["runs"]),
       "mean_exact_only_nll":float(np.mean([r["fitted"]["exact_only_nll"] for r in results["runs"]])),
       "mean_fitted_nll":float(np.mean([r["fitted"]["query_on_miss_nll"] for r in results["runs"]])),
       "mean_untrained_query_nll":float(np.mean([r["initial_untrained_query"]["query_on_miss_nll"] for r in results["runs"]])),
       "learned_beats_untrained":sum(r["fitted"]["query_on_miss_nll"]<
                                    r["initial_untrained_query"]["query_on_miss_nll"]
                                    for r in results["runs"])}
    (out/"RESULTS.json").write_text(json.dumps(results,indent=2,sort_keys=True))
    print(json.dumps(results["summary"],sort_keys=True,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--original-protocol",required=True)
    p.add_argument("--vim-root",required=True)
    p.add_argument("--output",required=True)
    main(p.parse_args())
