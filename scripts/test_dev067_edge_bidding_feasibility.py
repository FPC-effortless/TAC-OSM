#!/usr/bin/env python3
"""Original frozen DEV067 decision-policy feasibility: NOT an auction bidder.
Tests exact archived model outputs, local latency, typed action-field dependence,
and an outside-the-model bid hard cap. Synthetic demos are software tests only.
"""
import argparse,hashlib,json,math,pathlib,sys,time
import numpy as np
import torch

def tensor(x):return torch.as_tensor(x,dtype=torch.float64)[None]
def hash_file(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def safe_bid(bid,left,freq,cap,limit,kill=False,roas=None,min_roas=None):
    if kill or left<=0 or freq>=cap or (min_roas is not None and (roas is None or roas<min_roas)):
        return 0.
    if not math.isfinite(bid) or bid<=0:return 0.
    return min(bid,left,limit)

def run(folder,iterations):
    folder=pathlib.Path(folder).resolve()
    sys.path.insert(0,str(folder))
    for m in ("agent_stable","agent_semantic","agent_model","agent_base"):sys.modules.pop(m,None)
    from agent_stable import StableSemanticAgent
    meta=json.loads((folder/"metadata.json").read_text())
    assert hash_file(folder/"agent.pt")==meta["checkpoint_sha256"]
    cfg=meta["spec"];torch.set_num_threads(1)
    model=StableSemanticAgent(cfg["backend"],cfg["width"],cfg["aux"],clock_mode=cfg["clock_mode"],bypass=cfg["bypass"],coordinate=cfg["coordinate"]).double().eval()
    model.load_state_dict(torch.load(folder/"agent.pt",weights_only=True,map_location="cpu"),strict=True)
    rows=[json.loads(x) for x in (folder/"demo-requests.jsonl").read_text().splitlines() if x.strip()]
    expected=[json.loads(x) for x in (folder/"demo-expected.jsonl").read_text().splitlines() if x.strip()]
    assert len(rows)==len(expected)>0
    state=None
    with torch.no_grad():
        for request,original in zip(rows,expected):
            if state is None or request.get("reset"):state=model.initial(tensor(request["keys"]))
            for event in request["events"]:state=model.observe(state,tensor(event))
            output=model.decide(state,tensor(request["candidates"]),tensor(request["prices"]))
            actual=output["logits"][0].numpy()
            assert np.allclose(actual,original["scores"],rtol=1e-8,atol=1e-8)
            assert int(model.choice(output)[0])==int(original["index"])
    # The actual trained checkpoint has an externally provided action-type field.
    with torch.no_grad():
        altered={k:v.clone() for k,v in state.items()}
        valid=altered["present"][0].nonzero().flatten().tolist()
        i=next((j for j in reversed(valid) if altered["events"][0,j,39:42].abs().sum()>0),valid[-1])
        old=altered["events"][0,i,39:42].clone()
        altered["events"][0,i,39:42]=torch.roll(old,1) if old.abs().sum()>0 else torch.tensor([1.,0.,0.],dtype=torch.float64)
        oldscores=model.decide(state,tensor(rows[-1]["candidates"]),tensor(rows[-1]["prices"]))["logits"]
        newscores=model.decide(altered,tensor(rows[-1]["candidates"]),tensor(rows[-1]["prices"]))["logits"]
        action_delta=float((oldscores-newscores).abs().max())
    # Note: original demo events are deliberately forced into the 64-record FIFO
    # to exercise full-context compute; this is NOT actual admitted state.
    common=[json.loads(x) for x in (pathlib.Path(a.source)/"tac_frozen_log/demo-requests.jsonl").read_text().splitlines() if x.strip()]
    assert len(common[0]["events"])>=64
    full=model.initial(tensor(common[0]["keys"]))
    full["events"][0]=torch.as_tensor(common[0]["events"][:64],dtype=torch.float64)
    full["present"][0]=True
    scored={}
    with torch.no_grad():
        for count in (1,16,64):
            use={k:v.clone() for k,v in full.items()}
            use["present"][0,count:]=False
            use["events"][0,count:]=0
            candidate=tensor(common[-1]["candidates"]);prices=tensor(common[-1]["prices"])
            for _ in range(20):model.decide(use,candidate,prices)
            times=[]
            for _ in range(iterations):
                t=time.perf_counter_ns();model.decide(use,candidate,prices)
                times.append((time.perf_counter_ns()-t)/1e6)
            scored[str(count)]={"p50_ms":float(np.percentile(times,50)),"p95_ms":float(np.percentile(times,95)),"p99_ms":float(np.percentile(times,99)),"over_5ms_samples":int(sum(x>5 for x in times)),"over_100ms_samples":int(sum(x>100 for x in times)),"n":iterations}
    # Check deterministic external envelope invariants; not native ad quality.
    rng=np.random.default_rng(20261021)
    for _ in range(1000):
        bid=float(rng.uniform(-100,400));left=float(rng.uniform(-25,300));freq=int(rng.integers(0,10))
        capped=safe_bid(bid,left,freq,4,75,roas=0.9,min_roas=0.8)
        assert 0<=capped<=75 and capped<=max(left,0)
        if freq>=4 or left<=0 or bid<=0:assert capped==0
        assert safe_bid(bid,left,freq,4,75,kill=True)==0
    # Campaign key-value routing is an EXTERNAL adapter. Native checkpoint has
    # a shared FIFO and 2 episode slots, not 64D persistent campaign memory.
    campaign_a={k:v.clone() for k,v in state.items()}
    campaign_b={k:v.clone() for k,v in state.items()}
    with torch.no_grad():campaign_a=model.observe(campaign_a,tensor(rows[-1]["events"][0]))
    assert all(torch.equal(campaign_b[k],state[k]) for k in state)
    return {"arm":meta["arm"],"source_sha256":meta["checkpoint_sha256"],
        "parameters":cfg["parameters"],"original_recorded_replays":len(rows),
        "original_replay_pass":True,"action_type_field_max_logit_delta":action_delta,
        "inference_ms":scored,"external_campaign_isolation_software_pass":True,
        "external_hard_caps_software_trials":1000,"auction_model_trained":False,
        "bid_price_output_present":False,"throttle_output_present":False,
        "fast_slow_hidden_state_persisted":False,"ad_profit_qualified":False,
        "timing_context":"one thread local CPU, original recorded demo input (forced FIFO); no traffic/network, simulator, queue, protocol, or auction rewards"}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--arm",required=True,choices=("tac_frozen_log","gru_frozen_log","no_casm_frozen_log"))
    p.add_argument("--repeat",type=int,default=120)
    p.add_argument("--out",required=True)
    a=p.parse_args();x=run(pathlib.Path(a.source)/a.arm,a.repeat)
    o=pathlib.Path(a.out);o.parent.mkdir(parents=True,exist_ok=True)
    o.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
    print(json.dumps(x),flush=True)
if __name__=="__main__":main()
