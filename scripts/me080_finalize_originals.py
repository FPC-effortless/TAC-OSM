"""Postfit metadata repair only; no model, fit, original score or registration edits.

The frozen ME080 runner's final document omitted the root preregistration hash
required by the schema. All receipts/checkpoints bind the correct original hash.
Rebuild the delivery document with that reference and independently verify it.
"""
import copy,json
import torch
from scripts.me080_train_world import integrity,frozen,ROOT,OUT,MANIFEST,sha,write,COUNT,STEPS
from scripts.validate_research_parity import check_manifest
from tac_osm.parameter_budget import validate_runtime_evidence
from tac_osm.variants.streaming_mixture_world import StreamingMixtureWorld


def main():
    reg=integrity();frozen();final=copy.deepcopy(reg)
    final['prefit_registration_sha256']=sha(MANIFEST)
    for row in final['models']:
        folder=OUT/f"seed-{row['seed']}";checkpoint=folder/'final.pt'
        receipt=json.loads((folder/'FIT_RECEIPT.json').read_text())
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if state['prefit_registration_sha256']!=sha(MANIFEST) or state['updates']!=STEPS:
            raise RuntimeError('Original model registration/update count differs')
        model=StreamingMixtureWorld(state['FIT_only_identified']);model.load_state_dict(state['weights'],strict=True)
        n=sum(p.numel() for p in model.parameters())
        if n!=COUNT or receipt['actual_steps']!=STEPS:raise RuntimeError('Actual whole model budget differs')
        evidence={'receipt_path':str(folder/'live-budget.json'),'receipt_sha256':sha(folder/'live-budget.json'),
                  'checkpoint_path':str(checkpoint),'checkpoint_sha256':sha(checkpoint)}
        validate_runtime_evidence({**row,'checkpoint_sha256':sha(checkpoint),'runtime_evidence':evidence,
            '_prefit_registration_sha256':sha(MANIFEST),'_expected_optimizer_steps':STEPS})
        row['measured']={'trainable_parameters':n,'deployed_parameters':n,'checkpoint_sha256':sha(checkpoint),
                         'actual_steps':STEPS,'training_receipt':receipt,'runtime_evidence':evidence}
    result=check_manifest(final,'final');write(ROOT/'FINAL_COMPARISON.json',final)
    write(ROOT/'POSTFIT_METADATA_REPAIR.json',{'scope':'final-delivery registration reference only; original frozen runner unchanged',
        'original_runner_finalize':'FAIL: omitted root prefit_registration_sha256',
        'original_source_weights_scores_registration_unchanged':True,'no_refit':True,
        'original_registration_sha256':sha(MANIFEST),'final_status':result})
    print(json.dumps(result))

if __name__=='__main__':main()
