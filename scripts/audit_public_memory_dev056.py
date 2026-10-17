"""Restore every DEV056 checkpoint and replay public-only/native trajectories."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from run_public_memory_dev056 import PROTOCOL,write,digest,git,evaluate
from audit_learned_stack_dev055 import native_replay,PIN
from tac_osm.development.public_memory_deadline import DeadlineModel,PublicHazardWorld,HAZARDS,oracle_transitions,collect
from tac_osm.development.learned_stack import FittedFilter,independent_transitions


def run(directory,output,veritas_root):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    started=time.monotonic();out=Path(directory);p=json.loads(PROTOCOL.read_text())
    summary=json.loads((out/'summary.json').read_text())
    assert not summary['smoke'] and summary['protocol_sha256']==digest(PROTOCOL)
    for name,sha in json.loads((out/'manifest.json').read_text()).items():assert digest(out/name)==sha
    vr=Path(veritas_root).resolve()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=vr,text=True).strip()==PIN
    sys.path.insert(0,str(vr/'src'))
    import investigation_world
    assert Path(investigation_world.__file__).resolve().is_relative_to(vr)
    tables=np.load(out/'filters.npz')
    data,labels=collect(*p['splits']['train'])
    np.testing.assert_array_equal(tables['fitted_hmm'],FittedFilter.fit(labels,data[1]))
    np.testing.assert_array_equal(tables['misspecified_independent'],independent_transitions(labels,data[1]))
    np.testing.assert_array_equal(tables['public_oracle'],oracle_transitions())
    controls=json.loads((out/'controls.json').read_text());reports=json.loads((out/'reports.json').read_text())
    result={'status':'RUNNING','restored_checkpoints':0,'native_worlds':0,'native_actions':0,'unrepaired_worlds':0,
            'source':summary['source'],'audit_source':git('rev-parse','HEAD'),'mask_backend_skips':{},
            'veritas_commit':PIN,'limitation':'Same author/shared physics generator; recorded native effect tables; no independent laboratory replication.'}
    def verify(policy,old,family):
        restored=evaluate(policy,p['splits'][family],family,tables)
        assert restored['rows']==old['rows']
        if isinstance(policy,DeadlineModel):
            assert restored['actual_operator_calls']==old['actual_operator_calls']
            assert restored['memory_backend_sample_updates']==old['memory_backend_sample_updates']
        for row in old['rows']:
            replay=native_replay(row,world_factory=PublicHazardWorld)
            assert replay['residual_faults']==row['final_faults']
            result['native_worlds']+=1;result['native_actions']+=replay['actions']
            result['unrepaired_worlds']+=int(not replay['state_verified'])
    for arm,by_family in controls.items():
        for family,old in by_family.items():verify(arm,old,family)
    for seed in p['model_seeds']:
        for arm in p['arms']:
            old=reports[str(seed)][arm]
            assert old['status']=='COMPLETE'
            checkpoint=out/f'{seed}-{arm}.pt';assert digest(checkpoint)==old['checkpoint_sha256']
            model=DeadlineModel(seed,arm).eval()
            model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu'))
            for family in HAZARDS:verify(model,old['evaluation'][family],family)
            if arm=='masked_mtsk':
                result['mask_backend_skips'][str(seed)]=sum(x['potential_memory_updates']-x['memory_backend_sample_updates'] for x in old['evaluation'].values())
            result['restored_checkpoints']+=1;print('audited',seed,arm,flush=True)
    result.update(status='PASS',seconds=time.monotonic()-started)
    write(output,result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--veritas-root',required=True)
    args=parser.parse_args();run(args.directory,args.output,args.veritas_root)
