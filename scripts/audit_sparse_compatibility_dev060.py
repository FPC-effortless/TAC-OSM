"""Verify sparse-only execution preserves both original learned agents."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from tac_osm.development.learned_agent import AgentConfig,LearnedAgent
from tac_osm.development.temporal_agent import SparseTemporalAgent
from run_fully_learned_dev058 import evaluate
from audit_fully_learned_dev058 import native_replay,PIN,export,standalone
from run_learned_stack_dev055 import write,digest,git
import subprocess
import shutil
import tempfile


def run(output,veritas_root=None,export_root=None):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if veritas_root:
        vr=Path(veritas_root).resolve();assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=vr,text=True).strip()==PIN
        sys.path.insert(0,str(vr/'src'))
    result={'status':'RUNNING','source':git('rev-parse','HEAD'),'temporal_bias':False,'new_weights_trained':False,
            'limitation':'Execution optimization of two existing agents, not ten-seed recipe validation or improved model reliability. Original parameter and state layouts retained when temporal=False.',
            'checkpoints':{},'exports':{},'cells':0,'trajectories':0,'actions':0,'native_actions':0}
    for seed in (7581,7582):
        root=ROOT/f'results/dev058-main/r0-{seed}';report=json.loads((root/'report.json').read_text())
        model=SparseTemporalAgent(AgentConfig(**report['config']),sparse=True,temporal=False).eval()
        loaded=model.load_state_dict(torch.load(root/'agent.pt',weights_only=True),strict=False)
        assert not loaded.missing_keys and not loaded.unexpected_keys
        result['checkpoints'][str(seed)]=digest(root/'agent.pt')
        for recorded in json.loads((root/'test.json').read_text())['cells']:
            assert recorded['lesion']=='none'
            actual=evaluate(model,recorded['seed'],recorded['entities'],recorded['history'],recorded['worlds'])
            assert actual==recorded,'Sparse execution changed recorded decisions, receipts, costs or dispatch counts'
            result['cells']+=1;result['trajectories']+=len(actual['rows'])
            for row in actual['rows']:
                result['actions']+=len(row['actions'])
                if veritas_root:result['native_actions']+=native_replay(row)
        destination=Path(export_root)/str(seed) if export_root else Path(tempfile.mkdtemp(prefix='sparse-dev060-'))
        export(root,report,destination)
        shutil.copy(ROOT/'src/tac_osm/development/temporal_agent.py',destination/'temporal_model.py')
        shutil.copy(ROOT/'scripts/learned_sparse_runtime_dev060.py',destination/'runtime.py')
        original=LearnedAgent(AgentConfig(**report['config'])).eval()
        original.load_state_dict(torch.load(root/'agent.pt',weights_only=True))
        result['exports'][str(seed)]=standalone(destination,original,5589001)
        with (destination/'README.txt').open('a') as note:note.write('Sparse execution enabled; temporal bias disabled. Original weights/state layout retained.\n')
        print('sparse compatibility',seed,'PASS',flush=True)
    result['status']='PASS';write(output,result);print(json.dumps(result),flush=True);return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',required=True);ap.add_argument('--veritas-root');ap.add_argument('--export-root');args=ap.parse_args();run(args.output,args.veritas_root,args.export_root)
