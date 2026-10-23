"""Supporting real-record streaming/missing-input checks; no memory-advantage claim."""
import argparse
from pathlib import Path
import numpy as np
import torch

from scripts.me071_persistent_workflow import ROOT,source,original_models
from scripts.me069_workflow_source import encode_cases,write_json
from scripts.me070_ngram_real_workflow import STATE,predict
from tac_osm.workflow_me071 import EntityMemory
import json


@torch.no_grad()
def run(cache):
    torch.set_num_threads(1);cohorts,pre,_=source(cache);prior=json.loads(STATE.read_text())
    cases=[c for c in cohorts["REPORT"] if c["n"]>16][:16]
    inv_a={v:k for k,v in pre["activities"].items()};inv_r={v:k for k,v in pre["resources"].items()}
    outputs={}
    for record,model in original_models(pre):
        serial=EntityMemory(model);mixed=EntityMemory(model);encodings=[]
        for c in cases:
            encodings.append(model.event_encoding(torch.tensor(c["activity"][:16][None]),
                torch.tensor(c["resource"][:16][None]),torch.tensor(c["features"][:16][None])))
        for i,c in enumerate(cases):
            for t in range(16):serial.observe(c["id"],encodings[i][:,t],float(c["timestamps"][t]))
        for t in range(16):
            for i,c in enumerate(cases):mixed.observe(c["id"],encodings[i][:,t],float(c["timestamps"][t]))
        error=max(float((a-b).abs().max()) for c in cases for a,b in zip(serial.states[c["id"]],mixed.states[c["id"]]))
        if error>1e-6:raise ValueError("entity interleaving changed independent state")
        original_loss=[];masked_loss=[]
        for c in cases:
            losses=[]
            for indices in (list(range(16)),list(range(11))+[15]):
                # Recompute gap/progress features and prior using only visible
                # events: hidden observations cannot leak through either path.
                raw={"id":c["id"],"events":[[inv_a.get(int(c["activity"][i]),"UNK"),
                    inv_r.get(int(c["resource"][i]),"UNK"),float(c["timestamps"][i])] for i in indices]}
                visible=encode_cases([raw],pre)[0]
                bank=EntityMemory(model)
                encoded=model.event_encoding(torch.tensor(visible["activity"][None]),torch.tensor(visible["resource"][None]),torch.tensor(visible["features"][None]))
                for t in range(len(indices)):state=bank.observe(c["id"],encoded[:,t],float(visible["timestamps"][t]))
                history=visible["activity"].tolist()
                history_prior=torch.tensor(np.log(predict(prior,history,5))[None],dtype=torch.float32)
                p=model.forecast(state,encoded[:,-1],torch.tensor([history[-1]]),history_prior)
                losses.append(float(-p.activity_logits.log_softmax(-1)[0,int(c["activity"][16])]))
            original_loss.append(losses[0]);masked_loss.append(losses[1])
        outputs[str(record["seed"])]={"original_real_cases":len(cases),"observed_prefix_events":16,
            "max_interleaved_vs_isolated_state_difference":error,"unmasked_event_NLL":float(np.mean(original_loss)),
            "four_hidden_events_NLL":float(np.mean(masked_loss))}
    write_json(ROOT/"STREAMING_PROBES.json",{"scope":"supporting execution on already exposed real DEV; posthoc missing-input intervention, no architecture/memory superiority or independent capability qualification", "seeds":outputs})
    print(json.dumps(outputs),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--cache",type=Path,required=True)
    run(parser.parse_args().cache)
