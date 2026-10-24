"""No-fit real-input output-equivalence and measured final-only execution cost."""
import argparse,json,time,statistics
from pathlib import Path
import torch
import scripts.clock_value_train as R
from tac_osm.variants.clock_value_world import ClockValueWorld
from tac_osm.variants.reusable_execution import mixture_quantiles


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',choices=tuple(R.CONFIG),default='me085');args=parser.parse_args()
    R.configure(args.study);R.integrity();torch.set_num_threads(1)
    _,fit,_,identified,_=R.load_original(Path('/tmp/me077-original-source'))
    entries=R.schedule(fit,identified,R.SEEDS[0])[0][0]
    x,a,_,_,dt=R.batch(fit,entries)
    m=ClockValueWorld(identified);path=R.OUT/f'seed-{R.SEEDS[0]}'/'final.pt'
    m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['weights'],strict=True);m.double().eval()
    x,a,dt=x.double(),a.double(),dt.double()
    def full():
        p=m(x,a,elapsed=dt);return mixture_quantiles({k:v[:,-1] for k,v in p.items()})
    def last():return mixture_quantiles(m.forecast_last(x,a,elapsed=dt))
    with torch.no_grad():
        expected=full();actual=last()
        for u,v in zip(expected,actual):torch.testing.assert_close(u,v,rtol=1e-5,atol=1e-6)
        for _ in range(5):full();last()
        timings={'full_historical_decoder':[],'last_only_decoder':[]}
        for i in range(31):
            for name,fn in ([('full_historical_decoder',full),('last_only_decoder',last)] if i%2==0 else [('last_only_decoder',last),('full_historical_decoder',full)]):
                start=time.perf_counter_ns();fn();timings[name].append((time.perf_counter_ns()-start)/1e6)
    med={k:statistics.median(v) for k,v in timings.items()}
    receipt={'no_fit':True,'source':'real FIT only; no new REPORT scoring','study':R.STUDY,'checkpoint_sha256':R.sha(path),
      'actual_batch':len(x),'history':32,'threads':1,'torch':torch.__version__,'dtype':'float64 complete qualified reference',
      'warmups_each':5,'alternating_repetitions_each':31,'median_ms':med,'raw_ms':timings,
      'same_final_quantiles_original_tolerance':'PASS','measured_speedup':med['full_historical_decoder']/med['last_only_decoder'],
      'scope':'within SAME model; skips unrequested historical forecast heads only. No architecture, CPU parity, streaming latency, GPU or asymptotic scan claim.'}
    R.write(R.ROOT/'EFFICIENCY.json',receipt);print(json.dumps({k:receipt[k] for k in ['median_ms','measured_speedup','same_final_quantiles_original_tolerance']}))

if __name__=='__main__':main()
