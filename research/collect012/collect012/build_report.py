"""Render the completed internal pilot from original receipts and audit results."""
import argparse,base64,html,io,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .common import ROOT

LABELS={'attention':'TAC + local attention','phase_p':'TAC + wide phase/P','hybrid_hard_p':'TAC + hard hybrid/P'}
COLORS={'attention':'#44546a','phase_p':'#007f73','hybrid_hard_p':'#a65416'}

def picture(fig):
    b=io.BytesIO();fig.savefig(b,format='png',dpi=160,bbox_inches='tight');plt.close(fig)
    return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()

def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'

def render(downloads,output,analysis_commit):
    s=json.loads((ROOT/'results/SUMMARY.json').read_text());rows=s['rows'];agg=s['aggregates']
    fig,ax=plt.subplots(1,2,figsize=(10.5,3.7),layout='constrained')
    for i,a in enumerate(agg):
        group=[r for r in rows if r['arm']==a['arm']];ys=[r['final_dev_nll'] for r in group]
        ax[0].scatter([i-.09,i,i+.09],ys,color=COLORS[a['arm']],s=40,zorder=3)
        ax[0].plot([i-.23,i+.23],[statistics.mean(ys)]*2,color=COLORS[a['arm']],lw=2)
        if a['arm']!='attention':
            for offset,key in ((-.13,'memory_zero_delta'),(.13,'memory_shuffle_delta')):
                ax[1].scatter([i-1+offset]*3,[r[key] for r in group],color=COLORS[a['arm']],marker='o' if offset<0 else '^',s=38)
    ax[0].set(xticks=range(3),xticklabels=['Local attention','Wide phase/P','Hard hybrid/P'],ylabel='Exposed-DEV NLL (nats/byte)',title='Fixed final checkpoint · three seeds')
    ax[1].set(xticks=[0,1],xticklabels=['Wide phase/P','Hard hybrid/P'],ylabel='NLL increase after intervention',title='Memory reads: zero (●), shuffled (▲)')
    for x in ax:x.grid(axis='y',alpha=.18);x.spines[['top','right']].set_visible(False)
    overview=picture(fig)
    fig,ax=plt.subplots(figsize=(10.5,3.7),layout='constrained')
    for a in agg:
        curves=[]
        for r in rows:
            if r['arm']!=a['arm']:continue
            lines=[json.loads(x)['nll'] for x in (downloads/f"{r['arm']}_{r['seed']}"/'journal.jsonl').read_text().splitlines()]
            curves.append(np.array(lines).reshape(32,32).mean(1))
        y=np.array(curves);x=np.arange(32)*32+16
        ax.plot(x,y.mean(0),label=LABELS[a['arm']],color=COLORS[a['arm']]);ax.fill_between(x,y.min(0),y.max(0),color=COLORS[a['arm']],alpha=.13)
    ax.set(xlabel='Optimizer update',ylabel='TRAIN NLL · 32-update means',title='Learnability pilot: all optimizer losses retained')
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15);ax.legend(frameon=False)
    curves=picture(fig)
    first=table(['Actual graph','Parameters','Initial DEV mean','Final DEV mean [seed range]','Zero-read Δ mean','Shuffled-read Δ mean'],[
        [LABELS[a['arm']],f"{a['parameters']:,}",f"{a['initial_dev_nll']['mean']:.4f}",f"{a['final_dev_nll']['mean']:.4f} [{a['final_dev_nll']['minimum']:.4f}, {a['final_dev_nll']['maximum']:.4f}]",
         'N/A' if a['memory_zero_delta'] is None else f"+{a['memory_zero_delta']['mean']:.4f}",
         'N/A' if a['memory_shuffle_delta'] is None else f"+{a['memory_shuffle_delta']['mean']:.4f}"] for a in agg])
    per_seed=table(['Graph','Seed','Final DEV','Zero Δ','Shuffle Δ','Mean optimizer step (s)','State tensor payload (KiB)'],[
        [r['arm'],r['seed'],f"{r['final_dev_nll']:.6f}",'N/A' if r['memory_zero_delta'] is None else f"{r['memory_zero_delta']:+.6f}",
         'N/A' if r['memory_shuffle_delta'] is None else f"{r['memory_shuffle_delta']:+.6f}",f"{r['mean_optimizer_step_seconds']:.4f}",f"{r['generation_state_tensor_bytes']/1024:.3f}"] for r in rows])
    c=s['curvature_diagnostic_highest_load']
    curvature=table(['Frozen C10 operator control','Curvature bytes / owner','Other state bytes / owner','Present-key accuracy','Complete-map recall'],[
        [r['mode'],f"{r['curvature_state_bytes_per_owner']:,}",f"{r['other_state_bytes_per_owner']:,}",f"{100*r['present_query_accuracy_mean']:.2f}%",f"{100*r['complete_map_mean']:.2f}%"] for r in c])
    spectral=table(['Fresh-history spectral control','Curvature bytes / owner','Present-key accuracy','Complete-map recall','Observed operator seconds'],[
        [r['mode'],f"{r['curvature_state_bytes_per_owner']:,}",f"{100*r['present_query_accuracy_mean']:.2f}%",f"{100*r['complete_map_mean']:.2f}%",f"{r['operator_seconds_mean']:.3f}"] for r in s['spectral_diagnostic_highest_load']])
    bank_rows=[]
    for r in rows:
        for b in r['hard_bank_usage']:
            bank_rows.append([r['seed'],b['layer']+1,', '.join(str(x) for x in b['observed_counts']),b['active_banks'],f"{100*max(b['observed_counts'])/b['total']:.1f}%"])
    banks=table(['Seed','Layer','Hard selections: banks 1–4','Active banks','Largest bank share'],bank_rows)
    samples=''
    for r in rows:
        for temp in (.8,1.2):
            text=(downloads/f"{r['arm']}_{r['seed']}"/f'generation_temperature_{temp}.bin').read_bytes()[:200].decode('utf-8',errors='replace')
            samples+=f'<details><summary>{html.escape(r["arm"])} · seed {r["seed"]} · T={temp}</summary><pre>{html.escape(text)}</pre></details>'
    link='https://github.com/FPC-effortless/TAC-OSM/tree/'
    source=link+s['frozen_fit_commit']+'/research/collect012'
    analysis=link+analysis_commit+'/research/collect012'
    text=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>COLLECT012 — TAC-OSM language pilot</title>
<style>body{{font:17px/1.58 system-ui,sans-serif;color:#182733;background:#f8faf9;margin:0}}main{{max-width:1120px;margin:auto;padding:48px 28px 80px}}h1{{font-size:42px;line-height:1.14;margin:8px 0 22px}}h2{{margin:46px 0 12px;font-size:26px}}h3{{margin:28px 0 8px;font-size:20px}}p{{max-width:950px}}.eyebrow{{letter-spacing:.12em;font-size:12px;font-weight:750;color:#007f73}}.lead{{font-size:22px;line-height:1.45}}.box{{padding:18px 23px;border-left:4px solid #007f73;background:#eaf4f1}}.note{{color:#4a5b65;font-size:14px}}table{{width:100%;border-collapse:collapse;font-size:14px;margin:20px 0;display:block;overflow-x:auto}}td,th{{text-align:left;padding:11px 12px;border-bottom:1px solid #d9e2de;vertical-align:top}}th{{background:#e7eeeb;font-size:12px}}figure{{margin:28px 0}}img{{max-width:100%;height:auto}}figcaption{{font-size:13px;color:#4a5b65}}code,pre{{font-family:ui-monospace,monospace;font-size:13px}}pre{{white-space:pre-wrap;background:#edf1ef;padding:14px;overflow-wrap:anywhere}}a{{color:#006b62}}li{{margin:9px 0}}details{{border-bottom:1px solid #d9e2de;padding:9px 0}}footer{{margin-top:48px;border-top:1px solid #ccd8d2;padding-top:16px;font-size:13px;color:#4a5b65}}@media(max-width:650px){{main{{padding:26px 15px}}h1{{font-size:32px}}.lead{{font-size:19px}}}}</style></head><body><main>
<div class="eyebrow">TAC-OSM · COLLECT012 · 2 NOVEMBER 2026 · INTERNAL RESEARCH</div>
<h1>Phase memory now learns from ordinary text. The bottlenecks remain measurable.</h1>
<p class="lead">Nine genuine 100K-parameter TAC-OSM language fits completed their fixed budgets. Both phase-addressed variants learned next-byte prediction without a parser, key labels or synthetic write supervision. Removing their added memory reads increased DEV loss.</p>
<div class="box"><strong>What this establishes:</strong> a working, trainable language integration and a useful contribution from its added memory path on the exposed DEV cohort. <strong>What remains open:</strong> exact semantic retrieval in language, robust long-horizon memory, convergence, full training FLOPs, controlled speed comparisons and independent qualification.</div>
<h2>1. Trained results</h2><p>Each graph used three fixed seeds and 1,024 FP32 AdamW updates: batch 4, sequence length 128, learning rate 0.0003, betas 0.9/0.95, weight decay 0.1 and gradient clipping 1.0. We evaluated the fixed final update; no best-DEV checkpoint selection occurred.</p>
{first}<figure><img src="{overview}" alt="Final DEV NLL for every seed and memory intervention loss increases"><figcaption>Points are individual initialization seeds, bars are means. These are descriptive internal measurements; parameter/token alignment does not establish matched FLOPs.</figcaption></figure>
<p>The same 64 original DEV windows supply <strong>8,192 next-byte targets</strong> for every model. Reusing them across seeds does not create additional independent evaluation data. TRAIN and DEV come from pinned Tiny Shakespeare; protected TEST bytes were not read.</p>
<p>Zero-read and cross-example shuffled-read interventions act on the added phase-memory output in all four layers while preserving each checkpoint’s weights. Native TAC fast/slow banks and SSA remain active. These interventions support the added path’s contribution; they do not isolate preconditioning from phase encoding, routing or FFN allocation. The attention reference has no such phase path, so these interventions are not applicable to it.</p>
<figure><img src="{curves}" alt="Training loss curves with all three seeds per graph"><figcaption>32-update loss means; shading shows the three-seed range, not a confidence interval. Each fit presents 524,288 TRAIN bytes with replacement, about 0.522 times the TRAIN byte count. This is an early learnability pilot.</figcaption></figure>
<h2>2. What was actually integrated</h2><p>The code retains TAC’s fast/slow associative banks, rotating field, perspective mixer, feed-forward transformations and mid-stack sparse structure access. The reference keeps TAC’s local attention lane with a bounded 320-token cache. The other arms replace that lane with separated phase-address and value projections. This reference is <strong>TAC plus local attention</strong>, not a standalone Transformer.</p>
<p>The wide phase arm uses one 64-dimensional address and 64-dimensional value memory per layer, with a full 64×64 inverse-curvature state. The hard hybrid uses four 16-dimensional banks and a straight-through one-bank routing decision. Its projections and state updates still run densely; this code has no sparse-kernel speed claim. The learned projection named “role” receives ordinary hidden features, not semantic role labels.</p>
<p>The actual frontend is new and trained from bytes. It is not the frozen three-character key encoder that achieved COLLECT010’s synthetic matching score. COLLECT010’s exact-memory results therefore do not transfer automatically to this language frontend.</p>
<h2>3. Runtime and state costs</h2>{per_seed}<p class="note">Step times are observed CPU optimizer executions on separate GitHub workers, excluding some sampling/setup costs; they are not controlled throughput benchmarks. State payload is the sum of retained tensors for one generation owner, measured after 256 and 512 generated bytes. It excludes weights, allocator overhead, Python objects, optimizer state and BPTT activations. Constant carried payload does not imply constant training memory.</p>
<p>All 18 registered generation executions produced 512 bytes with finite logits at temperatures 0.8 and 1.2, with unchanged checkpoint weights and equal tensor payload at the two observation points. This proves bounded execution over that horizon, not coherent 512-token language or long-context semantic retention.</p>
<h2>4. Cheaper curvature did not preserve exact recall</h2><p>A separate, no-fit diagnostic used frozen COLLECT010 encoders, the validated COLLECT011 syntax commits and the same support lane. Its predeclared grid covered three seeds, four loads and three operator controls: 36 cells, 16 histories per cell. At the highest load, 256 records and 32 active keys:</p>{curvature}
<p>Every retained prediction/target was recomputed to verify query and complete-map summaries. Full geometry passed these fresh diagnostic histories; this is a component result, not a production reliability estimate. Block and diagonal approximations lost cross-key geometry. Their state savings were real, and their complete-map failures were also real. “Other state” includes 4,352 value-memory bytes and a 512-byte support lane; curvature savings are not whole-model memory savings.</p>
<p>This diagnostic is a simple engineering approximation, not a reproduction of <a href="https://arxiv.org/abs/2604.21100">Preconditioned DeltaNet (2026)</a>. Curvature-aware DeltaNet and diagonal approximations are existing work; preconditioning itself is not a new contribution here.</p>
<h3>A partial state-cost repair: spectral sketch + residual diagonal</h3><p>After the negative diagonal/block result, a second protocol froze a fresh namespace and three ranks, 8/16/32, before evaluating 16 new histories per seed at 256 records and 32 keys. A streaming truncated-SVD Gram sketch retains cross-key directions; a residual diagonal captures remaining diagonal mass. The prior inverse action uses the Woodbury identity, followed by the same normalized memory write. All 12 cells and their predictions are preserved:</p>{spectral}
<p>Rank 32 retained 100% query and complete-map recall on all three seeds (48 maps total). Curvature state fell from 16 KiB to 8.25 KiB, a 48.4% reduction; including the unchanged 4.75 KiB memory/support state, the component payload fell from 20.75 to 13 KiB. Ranks 8 and 16 failed the same strict map criterion. The error is not monotonic in rank for this streaming approximation, so those failures remain visible.</p>
<p>This is a partial repair under a bounded active-key span. It uses a standard matrix-approximation strategy, not a claimed novel general algorithm. The SVD/solve implementation took about 17 times the full operator’s time in this fixture and incurs temporary workspaces. The stated savings concern retained algorithmic state, not measured process peak memory. No language-model fit used this sketch; natural-language keys may have substantially higher effective rank than this 32-key component task.</p>
<h2>5. A concrete credit-assignment weakness</h2><p>The frozen implementation reads the prior memory before writing the current value. In actual one-token and eight-token TAC graphs, the current value emission has zero immediate gradient through that memory read in all four phase/hybrid layers. Earlier values receive gradients across longer sequences. Attention’s current value path has a nonzero immediate gradient in the same diagnostic.</p>
<p>For the normalized write S′ = S + g(v − Sk)uᵀ with uᵀk = 1, reading the updated state at the same key gives S′k = (1 − g)Sk + gv. A constructed FP64 diagnostic verified this identity to 4.44×10<sup>−16</sup> and recovered the expected value gradient g = 0.3. This restores a local learning signal, but it may also encourage current-token copying. No trained improvement has been measured.</p>
<p>Write-before-read is established practice, including the <a href="https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/naive.py">official FLA Gated DeltaNet reference</a>. Applying and ablating that ordering in TAC-OSM is the proposed repair; it is not claimed as a new general algorithm.</p>
<h2>6. Hard-routing utilization</h2>{banks}<p class="note">Counts use the same 64 exposed-DEV windows, 8,192 selections per layer. Active banks and the largest share describe observed utilization, not semantic specialization, routing correctness or effective memory capacity. No bank-balancing objective was added to the frozen fits.</p>
<h2>7. Repairs worth testing next</h2>
<ol><li><strong>Restore direct value credit.</strong> Preregister read-before-write versus write-before-read with identical phase geometry and FFN allocation. Include a current-value residual control so a gain cannot be attributed only to an easier shortcut. Measure next-byte loss and a delayed-use probe together. Status: structural diagnostic passed; trained repair untested.</li>
<li><strong>Make the spectral repair affordable.</strong> The rank-32 component result justifies testing incremental QR/SVD updates or a bounded active-key basis with a residual diagonal. Charge basis storage and update work, stress loads above 32 keys and retain strict complete-map criteria. A separate block-plus-low-rank cross-block correction is another candidate. Status: one compact spectral form passed this bounded component diagnostic but is slow; LM translation and cheaper updates remain untested.</li>
<li><strong>Separate read selection from write admission.</strong> Compare the current shared hard route against separate read and write routes, with soft-routing warm-up and a fixed annealing schedule. Record per-bank usage, overwrite collisions and absent-key false accepts. Keep any load-balancing cost in the training budget. Status: untrained routing proposal, not an observed solution.</li>
<li><strong>Test semantics before scaling.</strong> Use an ordinary-language state-update task with a frozen generator/source policy, novel keys, overwrites, absent queries and distant scopes. Require all-seed whole-map reliability alongside text DEV loss. Synthetic key success and next-byte learnability currently leave a measurable gap between them.</li></ol>
<p>Stress-sharing or cooperative plasticity was not implemented in this pilot. It should remain a separate hypothesis until the memory frontend and routing controls have direct task evidence.</p>
<h2>8. Reproducibility and limits</h2><p>The full cohort contains <strong>9,216 full-fit optimizer updates</strong>, separately from 32 local smoke/replay updates; 4,718,592 TRAIN byte presentations; 45 immutable full-fit model/AdamW/RNG checkpoints; hash-chained per-step journals; all original receipts; all 18 generation payloads; and nine trained-source audits. Total recorded optimizer time is {s['summed_optimizer_seconds']/60:.2f} CPU minutes, summed over jobs, excluding setup/evaluation and local diagnostics.</p>
<p>All nine audits verify actual saved counts and checkpoint closure, AdamW counters, source/protocol identities, future-byte poison invariance, owner isolation and independent-expression whole-graph logits/state/parameter gradients in FP32 and FP64 on finite fixtures. Original fresh-process replay is exact in the training runtime (Torch 2.5.1 CPU). Local Torch 2.15 replay is tolerance-based (absolute/relative 2×10<sup>−5</sup>), not bitwise across versions.</p>
<p>Mid-checkpoint continuation code exists, but only an eight-update initialization replay was exercised. This report does not claim full mid-training trajectory replay. The manifest lists generation sample seeds 12011–12013; the frozen evaluator actually uses RNG 120100 for every model and temperature. All samples are retained, and generation is reported as fixed execution diagnostics, not independent sampling-seed coverage.</p>
<p>The pre-existing <a href="https://github.com/FPC-effortless/TAC-OSM/blob/3449041ec63c7efe843422086b6213226b94167a/research/RESEARCH-EXECUTION-LEVELS-2026-10-27.md">operator amendment</a> permits ordinary internal public FIT/DEV research without external signers. All three final single-graph gates pass with <code>eligible_for_architecture_superiority: false</code>. Full training FLOPs and compute parity remain unmeasured; independent qualification, protected TEST, promotion and full mathematical/IEEE/AD release remain open. These results neither certify safety nor establish a faster attention replacement.</p>
<h2>9. Fixed generation excerpts</h2><p class="note">The first 200 output bytes from every model/temperature are shown below with no sample selection. Each full 512-byte binary and decoded text is retained in the evidence archive. The common prompt is the frozen TRAIN slice [20000:20064].</p>{samples}
<h2>Evidence and primary sources</h2><ul><li><a href="{source}">Frozen fit code, registrations and permitted source package</a> · commit <code>{s['frozen_fit_commit']}</code></li><li><a href="{analysis}">Read-only audit and diagnostic code/results</a> · commit <code>{analysis_commit}</code></li><li><a href="{s['workflow_run']}">Original nine-fit workflow</a> · <a href="https://github.com/FPC-effortless/TAC-OSM/pull/698">draft research PR #698</a></li><li><a href="https://github.com/karpathy/char-rnn/blob/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/Readme.md">Pinned publisher README/license</a> · source and permitted TRAIN/DEV hashes are retained in the frozen repository package.</li><li><a href="https://arxiv.org/abs/2604.21100">Tumma, Loo & Rus: Preconditioned DeltaNet</a> · <a href="https://arxiv.org/abs/2412.06464">Yang, Kautz & Hatamizadeh: Gated Delta Networks</a> · <a href="https://github.com/fla-org/flash-linear-attention">Official FLA implementations</a></li></ul>
<footer>COLLECT012 · Internally measured research · Report generated from preserved originals. Numerical tables are descriptive, not a compute-qualified architecture ranking. No sealed TEST access, checkpoint promotion or merge occurred.</footer></main></body></html>'''
    output.write_text(text)
    print(json.dumps(dict(report=str(output),bytes=output.stat().st_size,embedded_figures=2,models=len(rows))))

def main():
    p=argparse.ArgumentParser();p.add_argument('--downloads',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--analysis-commit',required=True);a=p.parse_args()
    render(a.downloads,a.output,a.analysis_commit)

if __name__=='__main__':main()
