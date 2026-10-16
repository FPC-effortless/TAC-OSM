"""Independent DEV-027 initialization/stream factorial and world-cost audit."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random

from audit_dev024_writer_attribution import (ROOT, validate_rows, metrics, interval, mean, training_fingerprints)

PROTOCOL='development_protocols/DEV-027-INIT-STREAM.json'
INITIALIZATION='research/autoresearch/dev027-initialization.json'
ARMS=('A_A','A_B','B_A','B_B')
SOURCES=(PROTOCOL,INITIALIZATION,'research/autoresearch/dev025-inputs.json',
         'docs/DEV-027-INIT-STREAM-PROTOCOL.md','scripts/run_dev027_init_stream.py','scripts/audit_dev027_init_stream.py',
         'scripts/audit_dev024_writer_attribution.py','contracts/DEV-024.json',
         'scripts/run_memory_controls_dev.py','scripts/run_learned_schema_dev.py','scripts/audit_memory_controls_dev.py','contracts/DEV-018.json',
         'src/tac_osm/development/memory_baselines.py','src/tac_osm/development/noisy_memory.py',
         'src/tac_osm/development/schema_maintenance.py','src/tac_osm/development/delayed_maintenance.py',
         'src/tac_osm/development/retention_controls.py','src/tac_osm/development/entity_aging.py')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def config():
    c=json.loads((ROOT/PROTOCOL).read_text())
    old=json.loads((ROOT/'contracts/DEV-024.json').read_text())
    assert c['steps']==old['steps']==216 and c['regime']==old['regimes']['long']=={'history':512,'batch_size':2}
    assert c['seeds']==old['seeds'] and c['eval_steps']==4 and c['h_levels']==[512]
    assert c['arms']==list(ARMS)
    assert c['optimizer']=={'learning_rate':.002,'weight_decay':.0001,'clip':1.}
    return c


def source_hashes():return {p:digest(ROOT/p) for p in SOURCES}


def block_order(seed):
    order=list(ARMS)
    random.Random(f'dev027-fixed-order:{seed}').shuffle(order)
    return order


def reference(seed,init):
    r=json.loads((ROOT/INITIALIZATION).read_text())[f'{seed}-{init}']
    m=json.loads((ROOT/'research/autoresearch/dev025-inputs.json').read_text())
    cohort='DEV023' if init=='A' else 'DEV024'
    assert r['original_report_sha256']==m[f'{cohort}-long-{seed}']['report_sha256']
    assert r['initialization_offset']==config()['initialization_offsets'][init]
    return r


def specification():
    c=config()
    s=json.loads((ROOT/'contracts/DEV-018.json').read_text())
    assert s['optimizer']['learning_rate']==c['optimizer']['learning_rate']
    s.update(steps=c['steps'],eval_steps=c['eval_steps'],h_levels=c['h_levels'],seeds=c['seeds'],regimes={'long':c['regime']})
    return s


def audit(folder):
    path=folder/'report.json'
    r=json.loads(path.read_text());c=config()
    seed,arm=r['seed'],r['arm']
    assert r['study']=='DEV-027' and seed in c['seeds'] and arm in ARMS
    init,stream=arm.split('_')
    assert r['initialization_offset']==c['initialization_offsets'][init]
    assert r['training_seed_offset']==c['training_offsets'][stream]
    assert r['source_sha256']==source_hashes()
    assert r['block_order']==block_order(seed) and r['block_position']==block_order(seed).index(arm)
    assert r['runtime']['torch_threads']==2
    tr=r['training'];ref=reference(seed,init)
    assert tr['initial_hash']==ref['initial_hash'] and tr['before_components']==ref['before_components']
    assert r['initial_common_components']==ref['initial_common_components']
    assert (tr['steps'],tr['history'],tr['batch_size'],tr['observed_events'],tr['training_receipts'])==(216,512,2,221184,432)
    assert tr['fingerprints']==training_fingerprints(seed+c['training_offsets'][stream]-6000,'long')
    for prefix in ('event_encoder.','memory.'):
        assert tr['before_components'][prefix]!=tr['after_components'][prefix] and tr['gradients'][prefix] is True
    assert tr['initial_hash']!=tr['final_hash'] and math.isfinite(tr['final_loss'])
    assert r['model']['state_scalars']==768 and r['model']['parameters']==r['model']['trainable_parameters']==8674
    assert digest(folder/'model.pt')==r['model']['sha256']
    assert json.loads((folder/'replay.json').read_text())==dict(verified=True,checkpoint_sha256=r['model']['sha256'],report_sha256=digest(path),source_sha256=r['source_sha256'])
    assert set(r['grids'])==set(c['evaluation_offsets'])
    for grid,entry in r['grids'].items():
        rows=entry['rows']
        validate_rows(rows,seed+c['evaluation_offsets'][grid]-24000,'joint',specification())
        assert not set(tr['fingerprints']) & {x['fingerprint'] for x in rows}
        assert metrics(rows)==entry['summary'] and entry['summary']['primary_n']==96 and entry['summary']['reset_primary']==.5
    return r


def effect(values):
    r=interval(values);lo,hi=r['exploratory_95_t_df4']
    r['meaningful_separation']=abs(r['mean'])>=.05 and (lo>0 or hi<0)
    return r


def summarize(records):
    c=config();by={(r['seed'],r['arm']):r for r in records}
    assert len(by)==len(records)==20 and set(by)=={(s,a) for s in c['seeds'] for a in ARMS}
    assert all(r['source_sha256']==records[0]['source_sha256'] for r in records)
    for seed in c['seeds']:
        rs=[by[seed,a] for a in ARMS]
        assert all(r['runtime']==rs[0]['runtime'] and r['block_order']==rs[0]['block_order'] for r in rs)
        for x in ('A','B'):
            assert by[seed,x+'_A']['training']['initial_hash']==by[seed,x+'_B']['training']['initial_hash']
            assert by[seed,'A_'+x]['training']['fingerprints']==by[seed,'B_'+x]['training']['fingerprints']
        for grid in c['evaluation_offsets']:
            assert all([x['fingerprint'] for x in r['grids'][grid]['rows']]==[x['fingerprint'] for x in rs[0]['grids'][grid]['rows']] for r in rs)
    result=dict(study='DEV-027',models=20,evaluation_cells=40,training_seeds=c['seeds'],official_TAC_overall_success=False,arms={},effects={},adaptive_not_confirmatory=True)
    pooled={}
    for arm in ARMS:
        pooled[arm]=[mean([by[s,arm]['grids'][g]['summary']['primary'] for g in c['evaluation_offsets']]) for s in c['seeds']]
        out={'pooled_primary_per_seed':pooled[arm],'pooled_primary_mean':mean(pooled[arm]),'by_grid':{},'training_seconds':[by[s,arm]['training']['training_seconds'] for s in c['seeds']]}
        for grid in c['evaluation_offsets']:
            ms=[by[s,arm]['grids'][grid]['summary'] for s in c['seeds']]
            prediction=all(m['primary']>=.8 and m['primary']-m['swap_primary']>=.1 for m in ms)
            physical=all(m['selected_failures']<=.05 and m['selected_unnecessary_services']<=.1 and m['selected_total_cost']<=.8 for m in ms)
            physical &= mean([m['selected_total_cost'] for m in ms])<=mean([m['none_total_cost'] for m in ms])
            out['by_grid'][grid]=dict(per_seed=ms,means={k:mean([m[k] for m in ms]) for k in ms[0]},prediction_and_swap_gate=prediction,selected_physical_gate=physical,combined_selected_gate=prediction and physical)
        result['arms'][arm]=out
    aa,ab,ba,bb=(pooled[a] for a in ARMS)
    result['effects']['initialization_B_minus_A']=effect([((z+w)-(x+y))/2 for x,y,z,w in zip(aa,ab,ba,bb)])
    result['effects']['training_stream_B_minus_A']=effect([((y+w)-(x+z))/2 for x,y,z,w in zip(aa,ab,ba,bb)])
    result['effects']['initialization_by_stream_interaction']=effect([w-z-y+x for x,y,z,w in zip(aa,ab,ba,bb)])
    for treatment,control in [('B_A','A_A'),('B_B','A_B'),('A_B','A_A'),('B_B','B_A')]:
        result['effects'][treatment+'_minus_'+control]=effect([x-y for x,y in zip(pooled[treatment],pooled[control])])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folders',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rs=[audit(folder) for folder in a.folders];summary=summarize(rs)
    a.output.write_text(json.dumps(summary,indent=2)+'\n');print('DEV027_INDEPENDENT_AUDIT_PASSED',len(rs),flush=True)
