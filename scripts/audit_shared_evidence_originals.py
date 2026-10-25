"""Restore actual original tensors and pre-update guard receipts; never fit."""
from pathlib import Path
import argparse, json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from tac_osm.parameter_budget import file_sha,validate_runtime_evidence

def audit(study):
    if study not in ('me096','me097'):raise ValueError('unregistered original portfolio')
    reg=ROOT/f'research/mathematical-engineering/{study}/FAIR_COMPARISON.json'
    manifest=json.loads(reg.read_text());out=ROOT/f'research/replay-bundles/{study.upper()}'
    ids=[]
    for model in manifest['models']:
        folder=out/f"seed-{model['seed']}";receipt=folder/'live-budget.json';checkpoint=folder/'final.pt'
        row=dict(model,_prefit_registration_sha256=file_sha(reg),_expected_optimizer_steps=256,
           runtime_evidence=dict(receipt_path=str(receipt),receipt_sha256=file_sha(receipt),
                checkpoint_path=str(checkpoint),checkpoint_sha256=file_sha(checkpoint)))
        ids.append(validate_runtime_evidence(row))
    print(json.dumps(dict(original_model_live_receipt_audit='PASS',study=study,original_models=len(ids),runtime_run_ids=ids,model_weight_updates=0,archive_order_exception=(study=='me096'))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study');audit(p.parse_args().study)
