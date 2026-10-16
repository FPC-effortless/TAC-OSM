"""DEV-027 same-runtime initialization/stream factorial: train all four corners."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import torch  # noqa: E402
from tac_osm.development.entity_aging import EntityAgingControl  # noqa: E402
from tac_osm.development.noisy_memory import parameter_hash  # noqa: E402
from run_memory_controls_dev import train,evaluate,component_hash  # noqa: E402
from audit_dev027_init_stream import config,specification,source_hashes,reference,block_order,metrics,digest  # noqa: E402


def initial_model(seed,init):
    m=EntityAgingControl('masked_local',seed+config()['initialization_offsets'][init])
    ref=reference(seed,init)
    assert parameter_hash(m)==ref['initial_hash']
    assert {p:component_hash(m,p) for p in ('event_encoder.','memory.')}==ref['before_components']
    return m


def run(seed,output,replay=False):
    c=config();assert seed in c['seeds']
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    spec=specification();sources=source_hashes();order=block_order(seed)
    cpu=next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),'unknown')
    runtime=dict(torch=str(torch.__version__),python=platform.python_version(),cpu_model=cpu,torch_threads=2,block_id=f'{seed}:{os.getpid()}')
    for position,arm in enumerate(order):
        init,stream=arm.split('_');folder=output/f'{seed}-{arm}';folder.mkdir(parents=True,exist_ok=True)
        path=folder/'report.json';checkpoint=folder/'model.pt';model=initial_model(seed,init)
        if replay:
            r=json.loads(path.read_text())
            assert r['source_sha256']==sources and r['seed']==seed and r['arm']==arm
            assert digest(checkpoint)==r['model']['sha256']
            model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
            assert parameter_hash(model)==r['training']['final_hash']
        else:
            common={p:component_hash(model,p) for p in ('event_encoder.','query_encoder.','head.')}
            training=train(model,seed+c['training_offsets'][stream],'long',spec,False)
            torch.save(model.state_dict(),checkpoint)
            r=dict(study='DEV-027',seed=seed,arm=arm,initialization_offset=c['initialization_offsets'][init],training_seed_offset=c['training_offsets'][stream],source_sha256=sources,block_order=order,block_position=position,runtime=runtime,training=training,initial_common_components=common,
                   model=dict(sha256=digest(checkpoint),state_scalars=model.state_scalars,parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)),grids={})
        for grid,offset in c['evaluation_offsets'].items():
            rows,timing=evaluate(model,seed+offset,spec,False)
            assert not set(r['training']['fingerprints']) & {x['fingerprint'] for x in rows}
            entry=dict(rows=rows,summary=metrics(rows))
            if replay:assert entry==r['grids'][grid], 'exact checkpoint prediction/action/cost replay differs'
            else:r['grids'][grid]=entry
        if replay:
            (folder/'replay.json').write_text(json.dumps(dict(verified=True,checkpoint_sha256=digest(checkpoint),report_sha256=digest(path),source_sha256=sources))+'\n')
        else:path.write_text(json.dumps(r)+'\n')
        print('DEV027_REPLAY' if replay else 'DEV027_TRAINED',seed,arm,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--replay',action='store_true');a=p.parse_args();run(a.seed,a.output,a.replay)
