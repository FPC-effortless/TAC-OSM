"""Measured serving costs and a supported-operator FLOP subtotal; no fit."""
import json,time
import numpy as np
import torch
import scripts.me038_research as b
from tac_osm.me038_innovation_forecast import build_model


def run():
    torch.set_num_threads(1);frozen=b.verify();rows=[];profiles=[]
    for r in frozen['fits']:
        if r['status']!='PASS':continue
        z,meta=b.data(r['dataset']);x,_=b.batch(z,[96],r['horizon'])
        m=build_model(r['arm'],r['horizon'],r['seed']).eval()
        m.load_state_dict(torch.load(b.ROOT/r['selected']['checkpoint'],weights_only=False)['state_dict'])
        durations=[]
        with torch.no_grad():
            for _ in range(5):m(x)
            for _ in range(20):
                start=time.perf_counter();m(x);durations.append((time.perf_counter()-start)*1000)
        state_per_channel={'innovation_tac':128,'gru_cross':44,'context_cross':0,'ar_adapt':0}[r['arm']]
        rows.append({'key':r['key'],'lane':r['lane'],'dataset':r['dataset'],'horizon':r['horizon'],'arm':r['arm'],
            'B1_full_window_median_ms':float(np.median(durations)),'B1_full_window_p95_ms':float(np.quantile(durations,.95)),
            'weight_bytes':sum(p.numel()*p.element_size() for p in m.parameters()),
            'recurrent_state_bytes':state_per_channel*meta['channels']*4,'observed_history_bytes':96*meta['channels']*4,
            'output_bytes':r['horizon']*meta['channels']*4,'history_replay_required':True,
            'state_memory_is_not_total_activation_memory':True})
        if r['lane']=='exposure' and r['dataset']=='ETTh1' and r['horizon']==96 and r['seed']==38001:
            x,y=b.batch(z,[96],96);m.zero_grad(set_to_none=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as prof:
                pred=m(x);loss=(pred-y).square().mean();loss.backward()
            events=[{'operator':e.key,'reported_FLOPs':e.flops,'calls':e.count} for e in prof.key_averages() if e.flops]
            profiles.append({'arm':r['arm'],'B':1,'H':96,'C':meta['channels'],
                'supported_forward_backward_FLOP_subtotal':sum(e['reported_FLOPs'] for e in events),'operators':events,
                'complete_pipeline_FLOPs':False,'optimizer_parent_preprocessing_selection_included':False})
    receipt={'models':rows,'profiling':profiles,'same_CPU_environment':True,'threads':1,
        'serving_measurement':'Eager complete standardized-input forecast; B1,5warmups,20samples; no concurrent fitting/scoring.',
        'FLOP_scope':'Supported Torch profiler operators only; omitted unsupported work, optimizer, normalization/data acquisition costs outside forward and parent training. No full FLOP/energy parity.'}
    b.dump(b.OUT/'resources.json',receipt);print('PASS',len(rows),'model latency measurements;',len(profiles),'FLOP subtotals',flush=True)


if __name__=='__main__':run()
