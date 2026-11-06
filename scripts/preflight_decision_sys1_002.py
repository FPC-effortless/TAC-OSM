#!/usr/bin/env python3
"""No-fit parameter/source/runtime receipt; never grants training admission."""
import hashlib
import json
import platform
from pathlib import Path
import torch
from tac_osm.decision_sys1_001 import NativeRiskStateModel
from tac_osm.decision_sys1_baselines_002 import GRURiskModel, TransformerRiskModel, mamba2_runtime_status


def receipt():
    torch.set_num_threads(1)
    torch.manual_seed(2102)
    counts={C.__name__:sum(p.numel() for p in C().parameters())
            for C in [NativeRiskStateModel,GRURiskModel,TransformerRiskModel]}
    root=Path(__file__).resolve().parents[1]
    paths=['src/tac_osm/decision_sys1_001.py','src/tac_osm/predictive_state_agent.py',
           'src/tac_osm/decision_sys1_data_002.py','src/tac_osm/decision_sys1_baselines_002.py']
    return {'study':'DECISION-SYS1-002','status':'NO_FIT_ENGINEERING',
            'python':platform.python_version(),'torch':str(torch.__version__),
            'torch_git_version':torch.version.git_version,'cuda':torch.cuda.is_available(),
            'parameter_counts':counts,'max_relative_parameter_gap':max(counts.values())/min(counts.values())-1,
            'count_only_gate':max(counts.values())/min(counts.values())-1<=.01,
            'ssm':mamba2_runtime_status(),'optimizer_updates':0,'real_data_predictions':0,
            'training_admitted':False,'independent_review':'OPEN',
            'source_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}}

if __name__=='__main__':print(json.dumps(receipt(),indent=2,sort_keys=True))
