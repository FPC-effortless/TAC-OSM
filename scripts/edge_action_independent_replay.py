"""Independent original EDGEACTION001 real FIT forecast replay; no training.

Original training/scoring source and checkpoints remain completely immutable.
The pre-registered raw-forecast numerical tolerance is rtol1e-5/atol1e-6;
aggregate comparisons are tighter (rtol1e-5/atol1e-9).
A model-only replay may round aggregates differently across CPU kernels;
metadata, original source, checkpoint SHA and every gate stay exact.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

from scripts import edge_action_logged_fit as original
from tac_osm.variants.action_query_edge_world import ActionQueryEdgeWorld


def check_numbers(saved,found,path="result"):
    if isinstance(saved,dict):
        assert isinstance(found,dict) and saved.keys()==found.keys(),path
        for k in saved:check_numbers(saved[k],found[k],path+"."+str(k))
    elif isinstance(saved,list):
        assert isinstance(found,list) and len(saved)==len(found),path
        for i,(a,b) in enumerate(zip(saved,found)):check_numbers(a,b,path+f"[{i}]")
    elif type(saved) is float:
        assert type(found) is float,path
        np.testing.assert_allclose(found,saved,rtol=1e-5,atol=1e-9,
            err_msg=f"original frozen statistic differs: {path}")
    else:
        assert saved==found,(path,saved,found)


def independent_replay():
    root=original.OUT
    originals=json.loads((root/"FIT_INDEX.json").read_text())
    scores=json.loads((root/"RESULTS.json").read_text())
    saved=np.load(root/"FORECASTS.npz")
    original.verify_committed([root/"FIT_INDEX.json",root/"RESULTS.json"]+
        [root/f"seed-{s}"/"final.pt" for s in original.SEEDS])
    for rel,sha in originals["source_files"].items():
        assert original.sha(original.ROOT/rel)==sha,"fitted training/source hash changed: "+rel
    assert scores["train_episodes"]==list(range(49))
    assert scores["score_episodes"]==list(range(49,59))
    assert scores["dev_test_scored"] is False
    assert scores["independent_test_scored"] is False
    assert scores["full_gate_passed"] is False
    data=original.read_fit(Path("/tmp/aloha-action-fit-only.parquet"))
    ids=np.array([(ep,t) for ep in range(49,59)
         for t in range(32,data["episode_length"]-1,8)],dtype=np.int64)
    assert ids.shape==(1840,2)
    np.testing.assert_array_equal(saved["ids"],ids)
    original_models=originals["models"]
    raw_true=[];raw_ref={n:[] for n in ("persistence","logged_target","FIT_servo","velocity")}
    for k in range(0,len(ids),64):
        _,(x,prev,cmd,y)=original.batch(data,ids[k:k+64])
        raw_true.append(y)
        raw_ref["persistence"].append(x)
        raw_ref["logged_target"].append(cmd)
        raw_ref["FIT_servo"].append(x+data["prep"]["servo_gain"]*(cmd-x))
        raw_ref["velocity"].append(x+(x-prev))
    target=np.concatenate(raw_true)
    refs={n:np.concatenate(a) for n,a in raw_ref.items()}
    np.testing.assert_array_equal(saved["target"],target)
    def score(y):
        e=y-target
        mask=np.sqrt(np.mean(((target-refs["persistence"])/data["prep"]["state_scale"])**2,axis=-1))>=data["prep"]["moving_threshold"]
        return {"mse":float(np.mean(e**2)),
            "mae":float(np.mean(np.abs(e))),
            "moving_mse":float(np.mean(e[mask]**2))}
    calculated_refs={name:score(refs[name]) for name in refs}
    strongest=min(calculated_refs,key=lambda x:calculated_refs[x]["mse"])
    assert strongest==scores["strongest_reference"]=="velocity"
    check_numbers(scores["reference_scores"],calculated_refs,"references")
    results=[]
    for row in original_models:
        seed=row["seed"];checkpoint=root/f"seed-{seed}"/"final.pt"
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==row["checkpoint_sha256"]
        state=torch.load(checkpoint,map_location="cpu",weights_only=False)
        assert state["steps"]==256 and seed==state["seed"]
        m=ActionQueryEdgeWorld();m.load_state_dict(state["model"]);m.eval()
        assert m.active_parameter_count==99760
        for key in ("state_mean","state_scale","action_mean","action_scale","delta_scale","servo_gain"):
            np.testing.assert_allclose(state["preprocessing"][key],data["prep"][key],rtol=0,atol=0)
        outputs=[]
        with torch.no_grad():
            for k in range(0,len(ids),64):
                (x,u,a,h,owner,_),_=original.batch(data,ids[k:k+64])
                pred,_=m(x,u,a,h,owner)
                outputs.append(pred.mean[:,0].numpy()*data["prep"]["state_scale"]+data["prep"]["state_mean"])
        out=np.concatenate(outputs)
        raw_saved=saved["models"][len(results)]
        np.testing.assert_allclose(out,raw_saved,rtol=1e-5,atol=1e-6,
            err_msg=f"original output drift in seed {seed}")
        z=score(out)
        passed=z["mse"]<=.95*calculated_refs[strongest]["mse"] and z["moving_mse"]<=calculated_refs[strongest]["moving_mse"]
        results.append({"seed":seed,**z,"capability_gate":passed})
    check_numbers(scores["models"],results,"models")
    assert all(x["capability_gate"] is False for x in results)
    print(json.dumps({
      "status":"ORIGINAL_SOURCE_CHECKPOINT_RAW_FORECAST_AND_GATE_COLD_REPLAY_PASS",
      "seeds":[r["seed"] for r in results],
      "original_heldout_real_episodes":10,
      "query_count":len(ids),
      "full_gate_passed":False,
      "training_updates_in_replay":0,
      "old_DEV_or_TEST_scored":False,
      "numerical_contract":{"raw_forecast_rtol":1e-5,"raw_forecast_atol":1e-6,
          "aggregate_rtol":1e-5,"aggregate_atol":1e-9}},indent=2),flush=True)


if __name__=="__main__":
    torch.set_num_threads(1)
    independent_replay()
