"""Normalize already frozen ME073 size evidence to the subsequently fetched policy.

No model fit, parameter resize, budget change or performance selection occurs.
Original prefit registration is retained; conversion timing is explicit.
"""
import argparse,json,subprocess
from pathlib import Path
import torch
from scripts.me069_workflow_source import write_json,digest
from scripts.validate_research_parity import check_manifest,check_model_parameter_target
from scripts.me073_autoregressive_world import ROOT,BUNDLE,sha,SEEDS
from tac_osm.workflow_me073 import TrajectoryModel


def audit(final=False):
    pre=json.loads((ROOT/'FIT_PREPROCESSING.json').read_text());model=TrajectoryModel(pre)
    trained=sum(p.numel() for p in model.parameters() if p.requires_grad);deployed=sum(p.numel() for p in model.parameters())
    original=json.loads(subprocess.check_output(['git','show','4afb9131:research/mathematical-engineering/me073/FAIR_COMPARISON.json'],text=True))
    assert original['trainable_parameters']==trained==deployed
    check_model_parameter_target(trained,deployed,label='ME073 live implementation')
    rows=[{'seed':s,'trainable_parameters':trained,'deployed_parameters':deployed} for s in SEEDS]
    if final:
        records=json.loads((BUNDLE/'CHECKPOINTS.json').read_text())
        for row,record in zip(rows,records,strict=True):
            assert row['seed']==record['seed'] and sha(record['path'])==record['checkpoint_sha256']
            saved=torch.load(record['path'],map_location='cpu',weights_only=False)
            restored=TrajectoryModel(saved['preprocessing']);restored.load_state_dict(saved['model'],strict=True)
            row['measured']={'trainable_parameters':sum(p.numel() for p in restored.parameters() if p.requires_grad),
                             'deployed_parameters':sum(p.numel() for p in restored.parameters()),'checkpoint_sha256':sha(record['path'])}
    receipt={'schema_version':'tac-osm.parity.v1','study_id':'ME073_SCALE_NORMALIZATION','study_type':'single_model_training',
             'registered_before_first_fit':original['registered_before_first_fit'],'real_world_source':original['real_world_source'],'models':rows,
             'original_prefit_commit':'4afb9131','original_prefit_registration_sha256':digest(original),
             'original_registered_live_total_parameters':original['trainable_parameters'],
             'no_model_resize_or_refit':True,'original_training_protocol_unchanged':True,
             'master_policy_commit':'6cbbe10ab1f23d183b1fe82d0ea95d857051c467'}
    receipt['provenance']='Supplemental schema normalization after the master100k rule was fetched DURING already registered FIT execution. This derived file did not exist before fitting. The measured99,781 total parameters were frozen in original prefit commit4afb9131; all original parameters require gradients, so trainable and deployed counts coincide. No model, budget or scoring rule changed.'
    print(check_manifest(receipt,'final' if final else 'preflight'))
    write_json(BUNDLE/'FINAL_SCALE_AUDIT.json' if final else ROOT/'SCALE_REGISTRATION.json',receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args();audit(a.final)
