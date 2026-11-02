"""Self-contained scientific report from the closed repair/falsification cohort."""
import argparse,base64,html,io,json,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT

LABEL={'control':'Frozen control','balance':'Balance only','write_first':'Write first','combined':'Combined'}
COLOR={'control':'#475569','balance':'#087f8c','write_first':'#a05c16','combined':'#713f8f'}

def table(headers,rows):
    esc=lambda x:html.escape(str(x))
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'

def pic(fig,out,name):
    buf=io.BytesIO();fig.savefig(buf,dpi=160,format='png',bbox_inches='tight');plt.close(fig)
    (out/name).write_bytes(buf.getvalue())
    return 'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()

def render(downloads,old_downloads,out,analysis_commit):
    s=json.loads((ROOT/'results/SUMMARY.json').read_text());rows=s['rows'];agg=s['aggregates']
    c=json.loads((ROOT/'residual_tiered/SUMMARY.json').read_text());out.mkdir(exist_ok=True,parents=True)
    results=table(['Setting','Parameters','DEV NLL mean ± seed SD','Paired Δ vs control','Mean largest-bank share'],[
        [LABEL[a['variant']],'99,971',f"{a['common_dev_nll']['mean']:.5f} ± {a['common_dev_nll']['sample_sd']:.5f}",
         f"{a['paired_dev_delta_vs_control']['mean']:+.5f}",f"{100*a['largest_bank_mean']['mean']:.1f}%"] for a in agg])
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
    for i,a in enumerate(agg):
        rr=[r for r in rows if r['variant']==a['variant']]
        yy=[r['common_dev_nll'] for r in rr];axes[0].scatter([i-.09,i,i+.09],yy,color=COLOR[a['variant']],s=38)
        axes[0].plot([i-.22,i+.22],[statistics.mean(yy)]*2,color=COLOR[a['variant']],lw=2)
        for j,r in enumerate(rr):
            for layer in r['routing']:
                frac=max(layer['counts'])/sum(layer['counts'])
                axes[1].scatter(i+(j-1)*.07,frac,color=COLOR[a['variant']],s=22,alpha=.8)
    axes[0].set(ylabel='DEV NLL (nats / byte)',title='Fixed final checkpoint · three seeds')
    axes[1].set(ylabel='Largest bank fraction per layer',title='Utilization: every seed and layer');axes[1].axhline(.25,color='#888',ls=':',lw=1)
    for ax in axes:
        ax.set_xticks(range(4),['Control','Balance','Write first','Combined']);ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False)
    overview=pic(fig,out,'figure_1_language_and_routing.png')
    fig,ax=plt.subplots(figsize=(11,3.7),layout='constrained')
    for a in agg:
        curves=[]
        for r in rows:
            if r['variant']!=a['variant']:continue
            folder=old_downloads/f"hybrid_hard_p_{r['seed']}" if r['reused_control'] else downloads/f"{r['variant']}_{r['seed']}"
            loss=np.array([json.loads(x)['nll'] for x in (folder/'journal.jsonl').read_text().splitlines()]).reshape(32,32).mean(1);curves.append(loss)
        yy=np.array(curves);xx=np.arange(32)*32+16
        ax.plot(xx,yy.mean(0),color=COLOR[a['variant']],label=LABEL[a['variant']]);ax.fill_between(xx,yy.min(0),yy.max(0),color=COLOR[a['variant']],alpha=.10)
    ax.set(xlabel='Optimizer update',ylabel='TRAIN NLL · 32-update means',title='Language loss only; auxiliary loss is charged separately')
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15);ax.legend(frameon=False)
    learning=pic(fig,out,'figure_2_training.png')
    split_rows=[]
    for variant in ('write_first','combined'):
        rr=[r for r in rows if r['variant']==variant]
        vals={mode:[next(x['nll'] for x in r['read_split']['rows'] if x['mode']==mode) for r in rr] for mode in ('normal','current_component_only','historical_component_only')}
        split_rows.append([LABEL[variant]]+[f"{statistics.mean(vals[mode]):.5f}" for mode in vals])
    split=table(['Setting','Normal DEV','Current component only','Historical component only'],split_rows)
    retrieval_rows=[];primary_rows=[]
    for a in agg:
        def entry(cohort,mode):return next(x for x in a['retrieval'] if x['cohort']==cohort and x['delay']==1024 and x['mode']==mode)
        n=entry('balanced_retrieval','normal');z=entry('balanced_retrieval','reset');w=entry('balanced_retrieval','wrong_history')
        retrieval_rows.append([LABEL[a['variant']],f"{100*n['present_accuracy']['mean']:.2f}%",f"{100*z['present_accuracy']['mean']:.2f}%",f"{100*w['present_accuracy']['mean']:.2f}%",f"{100*n['complete_present_map']['mean']:.2f}%",f"{100*n['absent_marker_accuracy']['mean']:.2f}%",f"{100*n['full_vocab_present_accuracy']['mean']:.2f}%"])
        p=entry('primary_retrieval','normal')
        primary_rows.append([LABEL[a['variant']],f"{100*p['present_accuracy']['mean']:.2f}%",'25%',f"{100*n['present_accuracy']['mean']:.2f}%",'12.5%'])
    retrieved=table(['Balanced cohort · 1,024-byte delay','Present normal','History reset','Wrong history','Complete present maps','Absent marker correct','Full-vocabulary present top-1'],retrieval_rows)
    probe_disclosure=table(['Setting','Original present accuracy','Original best constant','Balanced present accuracy','Balanced best constant'],primary_rows)
    carry=table(['Setting','Carry gain over reset, nats/byte','Wrong-history cost, nats/byte'],[
        [LABEL[a['variant']],f"{a['continuation_carry_gain']['mean']:+.5f}",f"{a['continuation_wrong_history_cost']['mean']:+.5f}"] for a in agg])
    fig,ax=plt.subplots(figsize=(10.5,3.6),layout='constrained')
    heat=[];labels=[]
    for cap in (8,16):
        for regime in c['protocol']['regimes']:
            line=[]
            for n in (8,16,32,64):
                get=lambda policy:next(x for x in c['summaries'] if x['N']==n and x['capacity']==cap and x['regime']==regime and x['policy']==policy)
                line.append(100*(get('residual')['class_accuracy']-get('random_count_matched')['class_accuracy']))
            heat.append(line);labels.append(f"{cap} slots · {regime.replace('_',' ')}")
    heat=np.array(heat);limit=max(abs(heat.min()),abs(heat.max()),1)
    im=ax.imshow(heat,cmap='BrBG',vmin=-limit,vmax=limit,aspect='auto')
    for i in range(6):
        for j in range(4):ax.text(j,i,f'{heat[i,j]:+.1f}',ha='center',va='center',color='white' if abs(heat[i,j])>.6*limit else '#182733',fontsize=11)
    ax.set(xticks=range(4),xticklabels=[8,16,32,64],yticks=range(6),yticklabels=labels,xlabel='Active fact keys',title='Residual gate minus matched random · class-accuracy percentage points')
    fig.colorbar(im,ax=ax,label='Percentage-point difference')
    residual_plot=pic(fig,out,'figure_3_residual_gate.png')
    residual_rows=[]
    for n in (16,32,64):
        for regime in c['protocol']['regimes']:
            get=lambda p:next(x for x in c['summaries'] if x['N']==n and x['capacity']==16 and x['regime']==regime and x['policy']==p)
            r=get('residual');q=get('random_count_matched');all_=get('admit_all')
            residual_rows.append([n,regime.replace('_',' '),f"{100*r['class_accuracy']:.2f}%",f"{100*q['class_accuracy']:.2f}%",f"{100*all_['class_accuracy']:.2f}%",f"{100*r['complete_class_map']:.2f}%"])
    residual_table=table(['Keys','Context regime','Residual gate','Matched random','Admit all','Residual complete maps'],residual_rows)
    details=table(['Setting','Seed','DEV NLL','Δ vs paired control','Largest bank, worst layer','Step seconds','Carry gain'],[
        [LABEL[r['variant']],r['seed'],f"{r['common_dev_nll']:.6f}",f"{r['common_dev_nll']-next(x['common_dev_nll'] for x in rows if x['variant']=='control' and x['seed']==r['seed']):+.6f}",f"{100*r['largest_bank_maximum']:.2f}%",f"{r['mean_step_seconds']:.4f}",f"{r['continuation_carry_gain']:+.6f}"] for r in rows])
    source=f'https://github.com/FPC-effortless/TAC-OSM/tree/{analysis_commit}/research/collect014'
    report=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>COLLECT014–015: trained memory repairs and residual-tiered falsification</title><style>
body{{font:17px/1.58 system-ui,sans-serif;background:#f7faf9;color:#182733;margin:0}}main{{max-width:1140px;margin:auto;padding:48px 28px 80px}}h1{{font-size:40px;line-height:1.14;margin:12px 0 22px}}h2{{font-size:26px;margin:42px 0 12px}}h3{{font-size:20px;margin:28px 0 10px}}p{{max-width:980px}}.eyebrow{{font-size:12px;font-weight:750;letter-spacing:.12em;color:#087f8c}}.lead{{font-size:22px;line-height:1.45}}.box{{background:#e8f2ef;border-left:4px solid #087f8c;padding:18px 23px}}.note,figcaption{{font-size:13px;color:#52636d}}.table{{overflow-x:auto;margin:22px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:11px 12px;border-bottom:1px solid #d5e0da;vertical-align:top}}th{{background:#e6eeea;font-size:12px}}figure{{margin:28px 0}}img{{max-width:100%;height:auto}}a{{color:#006e74}}code{{font:13px ui-monospace,monospace;overflow-wrap:anywhere}}li{{margin:10px 0}}footer{{margin-top:48px;border-top:1px solid #cbd8d2;padding-top:16px;color:#52636d;font-size:13px}}@media(max-width:650px){{main{{padding:26px 15px}}h1{{font-size:31px}}.lead{{font-size:19px}}}}
</style></head><body><main><div class="eyebrow">WOYENGI INTELLIGENCE LABS · COLLECT014–015 · 2 NOVEMBER 2026 · INTERNAL RESEARCH</div>
<h1>Trained routing and read-order repairs, tested separately from exact retrieval.</h1>
<p class="lead">Nine new 99,971-parameter models completed their registered training budgets. Three immutable controls supplied the fourth cell of a routing-balance × read/write-order experiment. A separate residual-tiered storage test examined when current compression error is useful for deciding what to preserve exactly.</p>
<div class="box"><strong>The distinction that matters:</strong> improved next-byte prediction, balanced bank utilization and reliable delayed fact retrieval are separate outcomes. This report measures each and preserves unfavorable results. It does not qualify a global-attention replacement or an architecture winner.</div>
<p>The combined repair has the lowest mean DEV NLL in this cohort, but is not best in every seed. Its balanced delayed-retrieval score at 1,024 bytes is 13.89%, versus a 12.5% best-constant baseline and 14.58% with wrong history. Complete-map and absent-marker accuracy remain zero. These interventions support useful language features; they do not demonstrate correct key-to-fact retrieval.</p>
<h2>1. The trained factorial repair experiment</h2><p>The control reads prior phase state before writing. “Balance only” adds 0.01 times the mean-layer Switch-style auxiliary loss. “Write first” updates phase state before reading the current key. “Combined” applies both changes. The fast/slow native banks, rotating field, SSA and all parameter dimensions remain intact.</p>
<p>SSA still performs query/key weighting over bounded slots. This campaign neither removes every attention-like operation nor compares against full global softmax attention. The language model's phase path and the separate ideal-address residual-cache diagnostic must not be conflated.</p>
<p>Every setting has 99,971 parameters. Initial tensors and all 1,024 sampled minibatch identities match the paired control for each seed. Each fit uses batch 4, 128-byte windows, FP32 AdamW, learning rate 0.0003, betas 0.9/0.95, decay 0.1 and clipping 1.0. The final update was fixed before training, with no best-DEV selection. Generated retrieval episodes never enter training.</p>
<p>Forward routing selects one bank; backward propagation uses the declared straight-through softmax surrogate. The reference checks validate that implemented surrogate, not an exact derivative of hard argmax. Balancing supplies an additional learning signal without removing discrete routing or guaranteeing read/write semantic consistency.</p>
{results}<figure><img src="{overview}" alt="Every seed's final language loss and every layer's largest bank share"><figcaption>Points represent seeds or seed/layer observations; mean bars summarize language loss. A 25% largest-bank share is perfectly even four-bank usage. Even usage does not establish semantic specialization or useful retrieval.</figcaption></figure>
<p>Scores use the same 8,192 original exposed-DEV byte targets for every model, recomputed in one local runtime. Seeds do not multiply the unique evaluation cohort. Original remote scores and exact same-runtime replay evidence are also retained. Full training FLOPs and controlled hardware throughput are unmeasured; these are internal same-graph repairs, not a compute-qualified architecture ranking.</p>
<figure><img src="{learning}" alt="Language-only training loss across all four settings"><figcaption>32-update language-loss means and three-seed ranges. Auxiliary loss, routing counts and actual objectives are retained separately in every new per-step journal. Shading is a seed range, not a confidence interval.</figcaption></figure>
<h2>2. Current-value credit versus historical phase output</h2><p>Write-first restores a direct gradient to the current value emission. With the shared unit address, the normalized write gives an output consisting of an attenuated prior-state read plus the routed current value. A supplemental operator fixture checked that decomposition to 2.98×10<sup>−8</sup> in FP32 and 8.33×10<sup>−17</sup> in FP64.</p>
{split}<p>These are frozen-checkpoint output interventions on all four phase modules. Actual writes and the other native memory paths remain. “Current component only” still receives hidden features containing information from native banks and lower layers, so it is not a history-free model. “Historical component only” is the retained prior-state part of the phase output, not every historical path in the network. Neither intervention is a separately trained alternative.</p>
<p class="note">The read-split protocol was declared after launch and before inspecting new full-fit outcomes. It is an additional mechanism diagnostic, not the original primary comparison or an exhaustive attribution of trained gains. A separately trained current-value-only control is still needed to attribute training gains specifically to persistent phase state.</p>
<h2>3. Delayed retrieval remains a distinct task</h2><p>The frozen primary probe contains 12 generated histories, four active keys, two overwrites and two absent queries per history. Queries follow 64, 256 or 1,024 original-DEV distractor bytes. It uses nine fixed answer candidates and no tuned rejection threshold. Prefill state is immutable across query branches; answers are excluded from query inputs.</p>
<p>The primary latest-value schedule unintentionally covers only four of the eight possible letters, giving a best-constant present-answer baseline of 25%. Its results are retained unchanged. After inspecting the reused controls, before inspecting new full-fit results, a separately recorded supplemental cohort made all eight latest values equally frequent (six occurrences each). Its best-constant baseline is 12.5%. The primary probe was not silently replaced.</p>
{probe_disclosure}<h3>Balanced cohort at a 1,024-byte delay</h3>{retrieved}
<p>History reset removes every carried path before the query. Wrong history permutes carried tensors between owners while preserving owner metadata for the explicit intervention. Normal performance must exceed these controls to support useful context-dependent retrieval. Twelve histories are reused across delays, seeds and variants; repeated scoring does not create additional independent examples.</p>
<p><strong>Interpretation limit:</strong> these small language models were not trained on this instruction grammar. Weak results establish a failure to demonstrate zero-shot retrieval utility here; they do not locate an intrinsic storage-capacity limit. The absent-marker score is an untrained nine-way decision, not production rejection accuracy. Both the original and balanced raw logits, predictions, targets and full-vocabulary top-1 predictions are preserved.</p>
<h2>4. Original-text continuity beyond the training window</h2><p>Eight original 512-byte DEV windows are scored in 128-byte chunks with carried, reset and wrong-history state. The table summarizes chunks 2–4; a positive carry gain means carrying state reduced NLL relative to resetting. A positive wrong-history cost means substituting another stream's state increased NLL.</p>{carry}
<p>This tests effects of longer state continuation on ordinary text. It does not establish exact fact recall or coherent long-horizon conversation. Registered 512-byte generation at temperatures 0.8 and 1.2 and all tensor-state observations are retained separately.</p>
<h2>5. Residual-tiered memory: what was actually tested</h2><p>The user's design combines local exact mixing, phase-addressed state, multiple decay timescales, a residual admission gate and a bounded routed exact store. COLLECT015 isolates storage behavior with supplied stable addresses. It is not a language-model integration or learned-routing result.</p>
<p>The diagnostic uses 16-dimensional unit phase keys, eight one-hot value classes, two delta states with decays 0.98/0.999, a common eight-record exact local buffer, and an eight- or sixteen-slot FIFO exact cache. The gate admits when pre-write residual norm exceeds 0.75. Query matching accepts only an identical phase address; this is a best-case address test. It does not establish paraphrase matching. Cache entries are invalidated on matching observed writes to prevent stale overrides.</p>
<p>Three seeds, 16 histories per seed, four fact loads, three distractor regimes, two capacities and four policies produce 4,608 policy evaluations from 576 unique stream histories. Controls include no exact cache, admission of every event and random admission at exactly the residual policy's count. Count matching is retrospective and is not a deployable causal gate with knowledge of future admission counts. No query answers are used by the gate.</p>
<figure><img src="{residual_plot}" alt="Residual gate versus matched random at every tested load and capacity"><figcaption>Every tested load/regime/capacity is shown. Cells are descriptive means across 48 histories. Near-zero differences should not be read as established advantages. The component uses supplied keys/values and no neural training.</figcaption></figure>
<h3>Sixteen-slot cache: load and distractor sensitivity</h3>{residual_table}
<p>The residual gate helps under some favorable repeated-distractor conditions. It does not reliably outperform matched random across the sweep, and large-load complete-map failures remain. At 64 facts with novel distractors it admits about 248.6 of 256 events: surprise does not filter much of this irrelevant background. Fewer cache admissions are not lower total compute; every event still updates compressed state and the local buffer, and this prototype also checks cache invalidation. Numeric retained payload includes both state matrices, local vectors and resident cache vectors; it excludes Python objects, search metadata and temporary workspaces. A bounded cache still saturates, while an expanding archive gives up constant-memory claims.</p>
<p>A four-event counterexample exposes the conceptual weakness: an already predictable fact has zero write error and is not cached; a later correlated-address write changes its compressed read from the correct [1,0] to [0.0975,0.6534], producing the wrong class. Conversely, unit-gain delta writes can recall the current unit-key association immediately even when pre-write error is large. Pre-write surprise therefore measures neither guaranteed future relevance nor post-write impossibility.</p>
<p>A separate constructed scale check also changed the absolute-threshold admission decision while preserving decoded output to floating-point tolerance: scaling state/value by 0.1 and inversely scaling the readout reduced residual from 0.9434 to 0.0943. A relative residual removes that particular scale dependence, but has not been trained or tested as a retrieval repair here.</p>
<p>Close prior work exists: <a href="https://arxiv.org/abs/2603.06642v2">SR-TTT v2 contains a correction and negative surprise-gated-cache results</a>. Broader relatives include <a href="https://arxiv.org/abs/2501.00663">Titans</a> and <a href="https://research.google/pubs/memorizing-transformers/">Memorizing Transformers</a>. The detailed source review and limits are preserved in the research notes; no broad novelty claim is made.</p>
<h2>6. What these findings justify next</h2><ul>
<li>Require retrieval gains alongside language loss before promoting routing balance or write-first as a memory repair. Direct value credit and useful historical storage must remain separate ablations.</li>
<li>Train a source-grounded delayed-use interface and expose the model to carried state beyond 128 bytes before drawing capacity conclusions from an unfamiliar instruction grammar. Every current training window starts fresh, so long-state use is evaluated beyond its trained state-age distribution. Keep ordinary-text performance, query dependence, overwrite accuracy and absent rejection as separate metrics.</li>
<li>For residual-tiered memory, test a protected age/position quota or a small uniform reservoir alongside residual admission. These are untested candidate hedges against predictable-but-later-useful facts, with matched storage and search cost required.</li>
<li>Profile and optimize an equivalent recurrent implementation independently of algorithmic memory changes. No GPU HBM diagnosis, Triton speedup or global-attention equivalence was established here.</li></ul>
<h2>7. Complete seed results and execution evidence</h2>{details}
<p>The new cohort contains <strong>9,216 full-fit optimizer updates</strong> plus 24 separately charged short-smoke updates. The three reused controls account for 3,072 earlier updates, not new training. The archive preserves 45 new and 15 reused full-fit model/AdamW/RNG checkpoints, all originals and journals, 24 generation samples for this cohort, raw retrieval probes, reference audits and residual-tiered predictions. New fits recorded {s['new_optimizer_seconds']/60:.2f} summed CPU optimizer minutes, excluding setup, evaluation and local audits. No GPU quota was used.</p>
<p>The saved-source audits and collection checks verify fixed-budget checkpoint closure, journal/AdamW consistency, actual original minibatch exposure, initial identity, future-byte causality, owner isolation and independent-expression whole-graph logits/state/parameter gradients in FP32/FP64 on finite fixtures. Original remote replay is exact under Torch 2.5.1 CPU. Common-runtime evaluation uses Torch {html.escape(rows[0]['common_torch'])} and explicit cross-runtime replay tolerance 2×10<sup>−5</sup>; all original identities remain unchanged.</p>
<p>An initial read-only audit adapter failed on an obsolete replay-report field. The old source and failure record are retained; the corrected adapter reran the same numerical checks without retraining or changing tolerances. The original label imbalance, supplemental-cohort declaration, all negative model/component outcomes and generation outputs remain visible.</p>
<p>The pre-existing operator amendment permits public licensed FIT/DEV internal research without external signers. Every new final single-model gate passes while architecture-superiority eligibility stays false. Protected TEST, independent qualification, full FLOPs parity, promotion and full formal/IEEE/AD release remain open. These ~100K, 128-byte-window, early-budget experiments are not frontier-scale or converged language-model evidence.</p>
<h2>Source and reproducibility</h2><ul><li><a href="https://github.com/FPC-effortless/TAC-OSM/tree/6e19362ab0028b7e857e4506657f2703403b451f/research/collect014">Frozen fit code and registrations</a> · <code>6e19362ab0028b7e857e4506657f2703403b451f</code></li><li><a href="{source}">Read-only analysis, protocols and research notes</a> · <code>{analysis_commit}</code></li><li><a href="https://github.com/FPC-effortless/TAC-OSM/actions/runs/41532283413">Original training run</a> · <a href="https://github.com/FPC-effortless/TAC-OSM/pull/699">Draft PR #699</a></li><li><a href="https://github.com/FPC-effortless/TAC-OSM/actions/runs/41513912016">Original immutable COLLECT012 controls</a> · fit commit <code>036d778255a416d3aae886acdbfeef8fb1b6a1ab</code></li></ul>
<footer>Internal research · Fixed final checkpoints · All seeds retained · Companion bundles: COLLECT014_NEW_MODELS.zip and COLLECT014_CONTROLS_AND_EVIDENCE.zip. Source/data remain in the pinned Git repository; original generated model/evaluation evidence is preserved separately.</footer></main></body></html>'''
    target=out/'COLLECT014_015_RESEARCH_REPORT.html';target.write_text(report)
    print(json.dumps(dict(report=str(target),bytes=target.stat().st_size,figures=3,settings=4,models=12)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--downloads',type=Path,required=True);p.add_argument('--old-downloads',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--analysis-commit',required=True);a=p.parse_args();render(a.downloads,a.old_downloads,a.output,a.analysis_commit)

if __name__=='__main__':main()
