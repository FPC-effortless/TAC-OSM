"""Compare original paired continuation receipts, exact gradient/update hashes."""
import argparse,importlib.util,sys
from pathlib import Path
import numpy as np
_spec=importlib.util.spec_from_file_location('langrecomp100_common',Path(__file__).with_name('common.py'));C=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=C;_spec.loader.exec_module(C)

def compare_pair(a,b):
    for field in ('seed','steps','absolute_optimizer_steps','trainable_parameters','deployed_parameters','parent_checkpoint_sha256','initial_model_sha256','initial_optimizer_sha256','initial_torch_rng_sha256','sample_sha256','source_commit','source_hashes'):
        if a[field]!=b[field]:raise ValueError('pair source/start/data/count mismatch: '+field)
    if len(a['history'])!=len(b['history']) or len(a['history'])!=16:raise ValueError('complete16 paired updates required')
    result=[]
    for x,y in zip(a['history'],b['history']):
        fields=('step','absolute_step','mode','loss','outputs_state_sha256','raw_all_parameter_gradient_sha256','parameter_gradient_norms','gradient_norm_before_clip','model_sha256','optimizer_sha256','torch_rng_sha256','exposure')
        for field in fields:
            if x[field]!=y[field]:raise ValueError('actual full paired gradient/optimizer mismatch at step'+str(x['step'])+': '+field)
        if x['forward_calls_including_recompute']!=1 or y['forward_calls_including_recompute']<2:raise ValueError('backward did not exercise actual rematerialization')
        result.append({'step':x['step'],'all10M_gradient_bytes_equal':True,'all_parameter_and_AdamW_moment_bytes_equal':True,'logits_full_native_state_loss_equal':True,'RNG_equal':True})
    return result

def run(args):
    if args.output.exists():raise FileExistsError('retain original comparison')
    reg=C.read(C.HERE/'REGISTER.json');source=C.source_hashes(args.source_commit);matrix={};summaries=[];manifest=C.read(C.HERE/'FAIR_COMPARISON.json')
    files=list(args.fits.rglob('FIT_RECEIPT.json'))
    if len(files)!=6:raise ValueError('all six genuine continuation archives required')
    for p in files:
        r=C.read(p);key=(r['seed'],r['policy'])
        if key in matrix or r['status']!='COMPLETE_FIT_ONLY_CONTINUATION' or r['steps']!=16 or r['absolute_optimizer_steps']!=2064 or r['source_hashes']!=source or r['source_commit']!=args.source_commit:
            raise ValueError('incomplete exact source-bound continuation matrix')
        if C.sha(p.parent/'MODEL.pt')!=r['checkpoint_sha256'] or not r['cold_replay_bitwise']:raise ValueError('original continuation model/replay differs')
        matrix[key]=r
    args.output.mkdir(parents=True)
    for seed in C.SEEDS:
        native,recompute=matrix[(seed,'native')],matrix[(seed,'recompute')]
        paired=compare_pair(native,recompute)
        for mode in ('language','episodic'):
            a=[r for r in native['history'] if r['mode']==mode];b=[r for r in recompute['history'] if r['mode']==mode]
            na=float(np.median([r['forward_backward_optimizer_seconds'] for r in a]));re=float(np.median([r['forward_backward_optimizer_seconds'] for r in b]))
            logical_a=float(native['activation_profiles'][mode]['logical_requested_bytes']);logical_b=float(recompute['activation_profiles'][mode]['logical_requested_bytes'])
            peaks_a=[r['memory']['whole_process_step_peak_rss_bytes'] for r in a];peaks_b=[r['memory']['whole_process_step_peak_rss_bytes'] for r in b]
            physical=all(v is not None for v in peaks_a+peaks_b)
            summaries.append({'seed':seed,'mode':mode,'measured_steps_each':len(a),'native_median_step_seconds':na,'recompute_median_step_seconds':re,'recompute_time_ratio':re/na,
                'native_profile_logical_saved_bytes':logical_a,'recompute_profile_logical_saved_bytes':logical_b,'logical_save_reduction_fraction':1-logical_b/logical_a,
                'physical_peak_measurement':'MEASURED_LINUX_WHOLE_PROCESS_RSS' if physical else 'INCONCLUSIVE_HWM_RESET_UNAVAILABLE',
                'native_max_step_peak_rss_bytes':max(peaks_a) if physical else None,'recompute_max_step_peak_rss_bytes':max(peaks_b) if physical else None,
                'whole_process_peak_reduction_fraction':1-max(peaks_b)/max(peaks_a) if physical else None})
        C.save(args.output/f'PAIRED_{seed}.json',{'seed':seed,'status':'PASS_EXACT_BYTEWISE_GRADIENT_AND_OPTIMIZER_PARITY','steps':paired})
    for row in manifest['models']:row['measured']=matrix[(row['seed'],row['name'])]
    C.save(args.output/'FINAL_RESOURCE_MANIFEST.json',manifest)
    result={'status':'COMPLETE_REAL_FIT_ONLY_RECOMPUTATION_TRIAL','source_commit':args.source_commit,'seeds':list(C.SEEDS),'policies':list(C.POLICIES),
        'parameters_each':10000000,'new_optimizer_steps_each':16,'total_new_optimizer_steps':96,'gradient_and_optimizer_parity':'PASS_EXACT_BYTEWISE_ALL17_TENSORS_EVERY_STEP',
        'summaries':summaries,'warmups_per_worker':4,'GPU_minutes':0,'actual_peak_GPU_VRAM':'NOT_MEASURED_CPU_ONLY','new_capability_evaluation':'NOT_RUN_NO_DEV_OR_TEST',
        'no_intelligence_gain_inferred':True,'scope':'Internal paired execution engineering; CPU RSS includes model, optimizer, gradients and runtime; logical tensor requests are separate.'}
    C.save(args.output/'RESULTS.json',result);print(result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fits',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-commit',required=True);run(p.parse_args())
