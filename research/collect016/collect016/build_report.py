"""Standalone, source-backed scientific report; requires complete nine-model collection."""
import argparse,base64,html,json,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT,ARMS,SEEDS,require
LABEL={'full_bptt':'Connected BPTT','detach_128':'Detach at 128','current_only':'Current-value phase lane'}
COLOR={'full_bptt':'#007f78','detach_128':'#c57523','current_only':'#4668b2'}
def table(headers,rows):
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def main():
    p=argparse.ArgumentParser();p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    s=json.loads((ROOT/'results/SUMMARY.json').read_text());require(s['status']=='COMPLETE_INTERNAL_FIXED_BUDGET' and len(s['records'])==9,'study collection incomplete')
    a.output_dir.mkdir(parents=True,exist_ok=True);records=s['records'];ag=s['aggregates'];figs=[]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    fig,axes=plt.subplots(1,2,figsize=(11,4.0))
    for arm in ARMS:
        journals=[[json.loads(x) for x in (a.runs_root/f'{arm}_{seed}'/'journal.jsonl').read_text().splitlines()] for seed in SEEDS]
        for ax,metric,title in zip(axes,['natural_nll','answer_nll'],['Natural TRAIN next-byte loss','Four answer-byte TRAIN loss']):
            values=np.array([[r[metric] for r in j] for j in journals]);smooth=np.array([np.convolve(v,np.ones(32)/32,mode='valid') for v in values]);x=np.arange(32,513)
            ax.plot(x,smooth.mean(0),color=COLOR[arm],label=LABEL[arm]);ax.fill_between(x,smooth.min(0),smooth.max(0),color=COLOR[arm],alpha=.12)
            ax.set(xlabel='Optimizer update',ylabel='Nats per target byte',title=title)
    axes[0].legend(frameon=False,fontsize=9);fig.tight_layout();path=a.output_dir/'training_curves.png';fig.savefig(path,dpi=160);plt.close(fig);figs.append((path,'32-update rolling means; shaded ranges span three seeds. Loss reduction alone is not query-dependent recall.'))
    fig,axes=plt.subplots(1,2,figsize=(11,3.9))
    for ax,n in zip(axes,[4,8]):
        for arm in ARMS:
            z=[r for r in records if r['arm']==arm];means=[];lo=[];hi=[]
            for gap in [64,256,768]:
                v=[next(q['first_accuracy'] for q in r['retrieval_summary'] if q['keys']==n and q['gap']==gap and q['condition']=='normal')*100 for r in z]
                means.append(statistics.mean(v));lo.append(min(v));hi.append(max(v))
            ax.plot([64,256,768],means,marker={'full_bptt':'o','detach_128':'s','current_only':'^'}[arm],color=COLOR[arm],label=LABEL[arm]);ax.fill_between([64,256,768],lo,hi,color=COLOR[arm],alpha=.1)
        ax.axhline(12.5,color='#777',ls='--',lw=1,label='Constant-class baseline');ax.set(xlabel='Distractor bytes after records',ylabel='Full-vocabulary first-byte accuracy (%)',title=f'{n} active keys',ylim=(-2,102))
    axes[0].legend(frameon=False,fontsize=8);fig.tight_layout();path=a.output_dir/'retrieval_by_delay.png';fig.savefig(path,dpi=160);plt.close(fig);figs.append((path,'Registered source-copy assay. Shading is the seed range, not a confidence interval. Delays reuse base record identities; all conditions share histories.'))
    fig,ax=plt.subplots(figsize=(10,3.8))
    for arm in ARMS:
        values=[]
        for r in records:
            if r['arm']==arm:
                v=np.array(r['first_answer_gradient_profile']['position_gradient_l1']);values.append(v/(v.sum() or 1))
        ax.plot(np.arange(384),np.mean(values,0),color=COLOR[arm],label=LABEL[arm])
    ax.axvspan(0,56,color='#b8c8c0',alpha=.28,label='Record region at final TRAIN step');ax.axvline(128,color='#aaa',ls=':');ax.axvline(256,color='#aaa',ls=':')
    ax.set(xlabel='Input byte position',ylabel='Fraction of activation-gradient L1',title='First-answer sensitivity at final weights');ax.legend(frameon=False,fontsize=8);fig.tight_layout();path=a.output_dir/'first_answer_credit.png';fig.savefig(path,dpi=160);plt.close(fig);figs.append((path,'Answer-only local sensitivity on each paired final TRAIN episode. Nonzero credit is not proof that the stored facts are used correctly.'))
    def figure(path,caption):return '<figure><img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'" alt="'+html.escape(caption)+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>'
    pct=lambda x:f'{100*x:.2f}%'
    primary=table(['Condition','Parameters','Ordinary DEV NLL','First byte','Allowed-character first byte','Exact four bytes','Complete maps','Absent first byte'],[[LABEL[r['arm']],r['parameters'],f"{r['dev_nll_mean']:.4f}",pct(r['first_accuracy']),pct(r['closed_first_accuracy']),pct(r['exact4']),pct(r['complete_maps']),pct(r['absent_first'])] for r in ag])
    comparisons=table(['Condition','Normal history','Wrong history','Reset all state','Permuted query','Seeds passing registered gates'],[[LABEL[r['arm']],pct(r['first_accuracy']),pct(r['wrong_first']),pct(r['reset_first']),pct(r['swap_first']),f"{r['seeds_passing']}/3"] for r in ag])
    sup=[]
    for name in records[0]['supplemental']:
        row=[name.replace('_',' ')]
        for arm in ARMS:
            z=[next(q for q in r['supplemental'][name] if q['condition']=='normal') for r in records if r['arm']==arm]
            row.append(pct(statistics.mean(v['first_accuracy'] for v in z)))
        sup.append(row)
    supplemental=table(['Supplemental stratum']+[LABEL[a] for a in ARMS],sup)
    seed_table=table(['Condition','Seed','DEV NLL','First byte','Exact four','Absent','First/last credit ratio','Optimizer seconds'],[[LABEL[r['arm']],r['seed'],f"{r['dev_nll']:.5f}",pct(r['primary_normal']['first_accuracy']),pct(r['primary_normal']['exact4']),pct(r['primary_normal']['absent_first']),f"{r['first_answer_gradient_profile']['first_to_last_gradient_ratio']:.5g}" if r['first_answer_gradient_profile']['first_to_last_gradient_ratio'] is not None else 'undefined',f"{r['optimizer_seconds']:.1f}"] for r in records])
    address=[]
    for arm in ['full_bptt','detach_128']:
        allrows=[json.loads((ROOT/f"results/{arm}_{seed}/ADDRESS_PROBE.json").read_text()) for seed in SEEDS]
        for layer in range(1,5):
            z=[next(t for t in q['summary'] if t['layer']==layer) for q in allrows]
            address.append([LABEL[arm],layer,f"{statistics.mean(v['mean_query_pair_cosine'] for v in z):.6f}",f"{max(v['max_query_pair_difference'] for v in z):.6g}",f"{statistics.mean(v['mean_quote_admission'] for v in z):.4f}",f"{statistics.mean(v['mean_filler_admission'] for v in z):.4f}"])
    address_table=table(['Condition','Layer','Mean query cosine','Largest query difference','Quote admission','Filler admission'],address)
    title='Supervised carried-state retrieval remains unestablished' if not s['full_condition_narrow_gate'] else 'Registered narrow supervised retrieval gates passed'
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>COLLECT016 research results</title><style>
body{{margin:0;background:#f4f5f2;color:#182c2b;font:16px/1.62 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:48px 28px 70px}}h1{{font-size:38px;line-height:1.15;max-width:900px}}h2{{margin-top:40px;font-size:25px}}h3{{font-size:19px}}.eyebrow{{font-size:12px;letter-spacing:.12em;color:#526c69}}.lead{{font-size:20px;max-width:930px}}.callout{{border-left:4px solid #007f78;background:white;padding:18px 22px}}.table{{overflow:auto;background:white;margin:20px 0;border:1px solid #dce3df;border-radius:8px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:11px 12px;text-align:left;vertical-align:top;border-bottom:1px solid #e6ebe7}}th{{background:#e8efeb}}figure{{margin:28px 0;background:white;padding:16px;border-radius:10px}}img{{width:100%;height:auto}}figcaption,.small,footer{{font-size:13px;color:#526c69}}a{{color:#006c65}}code{{font-size:13px;overflow-wrap:anywhere}}details{{background:white;border:1px solid #dce3df;padding:12px 16px;margin:12px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}footer{{margin-top:40px;border-top:1px solid #cbd7d0;padding-top:20px}}</style><main>
<p class="eyebrow">WOYENGI INTELLIGENCE LABS · COLLECT016 · 3 NOVEMBER 2026</p><h1>{title}</h1>
<p class="lead">Nine fresh, fixed-budget fits tested whether connected multi-window credit and explicit source-copy supervision teach the existing integrated model to retrieve distant facts. A trained current-value-only phase lane and a detached-state condition provide the registered controls.</p>
<div class="callout"><strong>Scope:</strong> {s['updates']:,} full-fit updates, {s['smoke_updates']} separately charged smoke updates, {s['trained_checkpoints']} full-fit checkpoints, 0 GPU minutes. All seeds retained. Internal exposed-DEV research; no independent qualification, converged-model claim, global-attention replacement or architecture winner.</div>
<h2>1. Primary results</h2><p>The registered primary retrieval case has four active identifiers, overwrites, and 256 distractor bytes after the records. Every seed sees the same 24 histories, 96 present queries and 48 absent queries. First answer characters are exactly balanced across a–h; a constant allowed character scores 12.5%. Exact four-byte and complete-map results are stricter. Ordinary-text loss uses the same 8,192 original DEV byte targets for all new models.</p>{primary}{comparisons}
<p>The “allowed-character” column restricts the decision to a–h and ?. Full-vocabulary decisions allow all 256 byte IDs. A model can score at the restricted constant baseline while emitting a different byte in unrestricted generation. No restricted score is substituted for full-vocabulary accuracy.</p>
{figure(*figs[1])}
<h2>2. What was trained</h2><p>Each model completed 512 FP32 AdamW updates, batch two/context 384, learning rate 0.0003. One sequence is an original TRAIN text window and one is a constructed temporary-key copy episode containing verbatim TRAIN quotations. The four copied answer bytes are traced to publisher-text offsets. The loss is natural-byte CE + 0.25 task-sequence CE + answer-byte CE + 0.01 router balancing. TRAIN and DEV text and identifier first-character families are separate.</p>
<p>Connected BPTT carries the actual state through three 128-byte calls without detaching. The detached condition preserves the same forward state but stops cross-boundary gradients. The current-value condition removes historical phase reads and unused address weights, reallocating capacity to active FFNs: 99,907 versus 99,971 parameters. It retains native fast/slow/rotary/SSA history. It is not a history-free baseline or a shape-identical intervention.</p>
<p>Measured smoke CPU cost caused the original 768-update plan to be reduced to 512 before any full fit. The original plan and all 24 smoke updates are retained. This is a finite early learnability budget, not convergence. The joint curriculum changes training exposure and objective together; there is no separately fitted long-context language-only arm here. Equal samples/updates/parameters do not establish full FLOPs parity.</p>{figure(*figs[0])}
<h2>3. Local binding and controlled history</h2><p>These supplemental diagnostics were declared after launch, before inspecting any full-fit results. They retain original DEV quotations while varying identifier family and delay. Nearby four-key episodes fit inside 128 bytes; the one-key episode is a simpler but untrained load. “Familiar” denotes the TRAIN identifier family, not a guarantee every sampled string is new. The registered primary protocol remains unchanged.</p>{supplemental}
<p>The frozen primary wrong-history generator changes filler in many cases because sampling different quotations consumes RNG differently. Its original results remain visible. The supplemental cases hold the entire post-record filler identical while shifting the stored quotations for the same keys, providing the cleaner counterfactual. An exact-history parser achieves 100% four-byte recall; a last-128-byte parser recovers none of the present answers at the primary long delay. These parsers use supplied syntax and exact storage and are unmatched engineering references.</p>
<h2>4. Credit and address diagnostics</h2>{figure(*figs[2])}
<p>Actual-source fixtures and trained-checkpoint audits test earlier-activation gradients, not only shared weight gradients. Connected credit exists in the finite fixtures; the explicit detached condition blocks earlier activation credit. This removes an accidental detach explanation. It does not establish useful optimization, correct binding, or a physical capacity bound.</p>{address_table}
<p>The first phase layer constructs its address from the normalized current byte embedding before mixing its own recurrent outputs. Its final query-space address is therefore context-free. Later layers and query-driven state writes can still produce query-dependent behavior; first-layer invariance does not prove the whole model ignores identifiers. Query similarity and gate rates are diagnostic geometry, not evidence that a particular fact has been stored or retrieved.</p>
<h2>5. Research implications</h2><p>A failed trained copy assay does not identify a fixed-state capacity ceiling. Local binding, query/write alignment, retention, output-mode learning and insufficient optimization remain distinct explanations. A passing narrow assay would likewise not establish arbitrary global retrieval. The fixed matrices still compress history and SSA retains bounded query/key weighting.</p>
<p>The next candidates are a compositional identifier frontend with aligned read/write addresses, a local curriculum that establishes binding before delay, and supervision that separately measures the first uncertain answer byte. A current-value path can improve local prediction while sharing admission pressure with persistent storage; decoupling that local computation from memory-write decisions is a specific hypothesis to test. None of these proposed repairs is claimed implemented by COLLECT016.</p>
<p>The user’s coverage maps, sketches, landmarks and Bayesian beliefs are useful when the required questions admit compact statistics. They cannot reconstruct arbitrary discarded text. Under finite-state exact-recall assumptions, N independent C-valued facts require at least N log₂(C) bits. That bound does not identify this model’s failure mechanism or force one particular slot implementation.</p>
<p>Primary research supports complementary memory systems and interference reduction, while human hippocampal studies also document false contextual recall. Biology motivates hypotheses; it does not establish an exact cache, a universally optimal 5% admission rate, or an untrainable neural router. See the source-linked research notes for the reviewed evidence and explicit limits.</p>
<h2>6. All seeds and verification</h2>{seed_table}
<p>Original artifact digests/CRC, all checkpoint hashes, 512-step source-offset exposure reconstruction, journal chains, AdamW counters, immutable initializers, paired exposure identities, serialization replay, actual-source causality/owner isolation, and trained FP32/FP64 numerical/surrogate-gradient fixtures are retained. Cross-runtime replay uses the predeclared 2×10⁻⁵ tolerance. These are finite internal engineering audits, not independent source authentication or a full formal proof.</p>
<p>All final internal single-model gates retain architecture-superiority eligibility false. Full Lean graph/source/IEEE/AD correspondence, protected TEST, independent qualification and promotion remain open. The partial mathematical record and strict peer registry checks are separate from empirical capability.</p>
<h2>Source and evidence</h2><ul><li><a href="https://github.com/FPC-effortless/TAC-OSM/pull/700">Draft source PR #700</a> · <a href="https://github.com/FPC-effortless/TAC-OSM/actions/runs/41544326038">Original nine-fit CPU run</a></li><li>Frozen fit source <code>2209e1328e3b42db4dd48cd702a21a55ffad24af</code>; pre-results analysis protocols <code>28ad49d1a29007ea3affa6b601e8fa2d92c963b6</code>.</li><li><a href="https://github.com/FPC-effortless/pnds-lean/pull/107">PNDS mathematical companion #107</a>; original 17 historical records and negative findings preserved.</li><li><a href="https://github.com/FPC-effortless/TAC-OSM/blob/research/collect016-carried-supervision-2026-11-03/research/collect016/RESEARCH_NOTES.md">Primary-source notes, biological corrections and candidate repairs</a>.</li></ul>
<footer>Fixed final checkpoints · All attempts retained · Companion model and evidence bundles contain generated originals, not copies of the Git-backed source/dataset. Constant state tensor payload does not measure peak training memory or total runtime overhead.</footer></main></html>'''
    (a.output_dir/'COLLECT016_RESEARCH_REPORT.html').write_text(body);print(json.dumps(dict(report=str(a.output_dir/'COLLECT016_RESEARCH_REPORT.html'),figures=[str(p) for p,_ in figs])))
if __name__=='__main__':main()
