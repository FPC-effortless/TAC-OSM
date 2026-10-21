"""Publish measured architecture-development progress, including all limits."""
import json,shutil,time
import numpy as np
import torch
from scripts.me035_architecture_research import ROOT,source,write_json


def run():
    torch.set_num_threads(1)
    a=ROOT/'results/me035';b=ROOT/'results/me035b';report=ROOT/'research/mathematical-engineering/me035';rb=ROOT/'research/mathematical-engineering/me035b'
    r=json.loads((a/'results.json').read_text());s=json.loads((b/'results.json').read_text())
    v=json.loads((a/'replay.json').read_text());w=json.loads((b/'replay.json').read_text())
    assert v['status']=='PASS' and w['forecast_replay_status']=='PASS'
    for root,dest in ((a,report),(b,rb)):
        for name in ('results.json','replay.json','frozen-models.json'):
            shutil.copyfile(root/name,dest/name)
    shutil.copyfile(b/'raw-runtime-exports.json',rb/'raw-runtime-exports.json')
    stages=[]
    for arm in ('native_joint','native_revin','native_residual','native_multiscale'):
        steps=r['evaluation']['mean_metrics'][arm]
        selected=min(steps,key=lambda x:(steps[x]['MSE'],int(x)))
        stages.append({'arm':arm,'selected_step':int(selected),'active_parameters':18924,**steps[selected]})
    selected=s['evaluation']['selected_step']
    stages.append({'arm':'AR + native multiscale','selected_step':selected,'active_parameters':32154,**s['evaluation']['means'][str(selected)]})
    ridge=r['evaluation']['references']['ridge'];base=r['evaluation']['mean_metrics']['native_joint']['250']['MSE']
    summary={'stages':stages,'ridge':ridge,'persistence':r['evaluation']['references']['persistence'],
        'native_250_engineered_relative_gain':1-r['evaluation']['mean_metrics']['native_multiscale']['250']['MSE']/base,
        'composite_relative_gain_vs_selected_native_reference':1-s['evaluation']['selected_MSE']/base,
        'composite_relative_MSE_vs_ridge':s['evaluation']['relative_MSE_vs_ridge'],
        'neural_fits':15,'ridge_fits':1,'total_new_training_CPU_seconds':r['physical_CPU_seconds']+s['new_physical_CPU_seconds'],
        'development_windows':r['development_windows'],'original_training_windows':r['train_windows'],
        'scalar_checkpoint_forecasts_replayed':v['scalar_forecasts']+w['scalar_forecasts'],
        'all_scalar_checkpoint_forecasts_bit_exact':v['scalar_forecasts']+w['scalar_forecasts']==v['bit_exact_checkpoint_forecasts']+w['bit_exact_checkpoint_forecasts'],
        'initial_equivalence_registered_tolerance_pass':w['initial_ridge_equivalence'],
        'initial_equivalence_checks':w['initial_ridge_checks'],'independent_qualification':False,'architecture_superiority_claim':False}
    write_json(report/'summary.json',summary)
    _,dev,meta=source(a);raw=dev['x']*torch.tensor(meta['scale'])+torch.tensor(meta['mean']);resources=[]
    for model in json.loads((b/'raw-runtime-exports.json').read_text())['models']:
        m=torch.jit.load(str(b/model['name'])).eval();times=[]
        with torch.no_grad():
            for _ in range(3):m(raw[:64])
            for _ in range(9):
                t=time.perf_counter();m(raw[:64]);times.append(time.perf_counter()-t)
        resources.append({'seed':model['seed'],'B64_raw_inference_median_seconds':float(np.median(times)),
            'weight_bytes':sum(p.numel()*p.element_size() for p in m.parameters()),
            'buffer_bytes':sum(p.numel()*p.element_size() for p in m.buffers()),'core_state_bytes':256,
            'observed_window_bytes':2688,'full_window_replay_required':True})
    write_json(rb/'deployment-resources.json',resources)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Native\njoint','+ causal\nnormalization','+ residual\nslow state','+ multiscale\ninitialization','AR backbone\n+ native TAC']
    values=[row['MSE'] for row in stages]
    fig,ax=plt.subplots(figsize=(9,4.5));ax.plot(range(5),values,'o-',color='#2766b0',lw=2,ms=8)
    ax.axhline(ridge['MSE'],ls='--',color='#19866c',label=f"Ridge reference: {ridge['MSE']:.4f}")
    for i,val in enumerate(values):ax.annotate(f'{val:.4f}',(i,val),xytext=(0,10),textcoords='offset points',ha='center')
    ax.set_xticks(range(5),labels);ax.set_ylabel('Development MSE in training-standardized units');ax.set_ylim(.35,.53)
    ax.set_title('ME035: build → train → retain better forecasting stages');ax.spines[['top','right']].set_visible(False);ax.legend()
    fig.text(.015,.015,'Selected checkpoints on reused ETTh1 development data. Final composite adds capacity and warm starts; no independent superiority claim.',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(report/'architecture-progress.png',dpi=180);fig.savefig(report/'architecture-progress.svg');plt.close(fig)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':run()
