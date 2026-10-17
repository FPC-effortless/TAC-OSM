"""Restore MTSK and execute a fresh public-history/native-action loop.

This is the DEV-054 component model. Joint updates/planning are explicit;
CDL/CASM and full TAC-OSM are not present.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from run_joint_repair_dev054 import setup
from run_learned_belief_dev052b import write
from tac_osm.development import joint_repair as j
from tac_osm.development.adaptive_repair import rollout_probe


def demo(directory, output, veritas_root):
    setup(veritas_root)
    out=Path(directory); common=json.loads((out/'common.json').read_text())
    seed=7401; arm='mtsk'; cell=json.loads((out/f'seed{seed}/report.json').read_text())['arms'][arm]
    model=j.JointModel(seed,arm);model.load_state_dict(torch.load(out/f'seed{seed}/{arm}.pt',weights_only=True));model.eval()
    stream=j.collect({'seed':5542001,'worlds':1,'kind':'switching'})[0]
    tokens=torch.tensor(stream['tokens'],dtype=torch.float32)
    state=model.initial(tokens[:1]); events=[]
    with torch.no_grad():
        for n in range(33):
            if n: state=model.observe(state,tokens[n-1:n])
            w=torch.softmax(model.predict(state)/cell['temperature'],-1).numpy()[0]
            event={'observations':n,'regime_weights':w.tolist()}
            if n in j.LEVELS:
                q=w@np.asarray(common['parameters']['endpoints'])
                event['native_probe']=rollout_probe(stream,n,q,'hmm_mixture_dp',common['parameters'],w)
            events.append(event)
        batch=j.weights(model,j.data([stream],common['parameters'],.08),cell['temperature'])[0]
        assert np.allclose(batch,np.array([e['regime_weights'] for e in events]),atol=1e-7)
    write(Path(output),{'study':'DEV-054 live incremental component demo','seed':5542001,'checkpoint_seed':seed,
                        'scope':'MTSK belief encoder, explicit joint planner and native Veritas; not full TAC-OSM',
                        'public_observations':stream['observations'],'public_tokens':stream['tokens'],'events':events})
    print(json.dumps([{'observations':e['observations'],'weights':e['regime_weights'],'cost':e['native_probe']['cost']/2,'faults':e['native_probe']['failures']} for e in events if 'native_probe' in e],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);p.add_argument('--output',required=True);p.add_argument('--veritas-root',required=True)
    a=p.parse_args();demo(a.directory,a.output,a.veritas_root)
