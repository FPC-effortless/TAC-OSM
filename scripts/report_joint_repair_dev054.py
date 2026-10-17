"""Write reviewable DEV-054 tables directly from raw-derived summaries."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def report():
    r=json.loads((ROOT/'research/dev054/summary.json').read_text())
    common=json.loads((ROOT/'results/dev054/common.json').read_text())
    audit=json.loads((ROOT/'results/dev054/audit.json').read_text())
    assert audit['status']=='PASSED_DEVELOPMENT_REPLAY' and audit['checkpoint_reconstructions']==15
    assert '1321 passed' in Path('/workspace/scratch/dev054-final-regression.log').read_text()
    profile=json.loads((ROOT/'research/dev054/profile.json').read_text())
    rows=[]
    for arm,v in r['pooled'].items():
        ratio=f"{100*v['fraction_gap_closed']:.2f}%" if v['ratio_eligible'] else 'Suppressed: floor/stability rule'
        lo,hi=v['delta_95pct'];fs=v['paired_faults_per_world']
        rows.append(f"| {arm} | {v['cost']:.6f} | {v['delta_vs_beta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {ratio} | {fs['difference']:+.5f} |")
    comparisons=[]
    for arm,v in r['contrasts'].items():
        lo,hi=v['bonferroni_four_95pct']
        comparisons.append(f"| {arm} | {v['mtsk_minus_reference']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {v['cost_reliability_guard']} |")
    endpoints=[]
    for family,arms in r['by_family'].items():
        for arm,levels in arms.items():
            v=levels['32'];endpoints.append(f"| {family} | {arm} | {v['cost']:.6f} | {v['delta_vs_beta']:+.6f} | {v['faults_mean_per_seed']:.2f} | {v['oracle_headroom']:.6f} | {v['model_break_even_decisions'] if v['model_break_even_decisions'] is not None else 'No positive saving'} |")
    training=[]
    for arm in r['training']['7401']:
        cells=[arms[arm] for arms in r['training'].values()]
        training.append(f"| {arm} | {cells[0]['parameters']} | {cells[0]['parameter_bytes']} | {cells[0]['retained_state_bytes']} | {sum(c['plateau'] for c in cells)}/{len(cells)} | "+', '.join(f"{100*c['plateau_relative_improvement']:+.2f}%" for c in cells)+' |')
    timing=[]
    for v in profile['rows']:
        timing.append(f"| {v['arm']} | {v['stored_parameters']} | {v['parameters_with_gradient_path']} | {1000*v['median_32_event_encoder_seconds']:.3f} |")
    compute=(common['elapsed_seconds']+audit['elapsed_seconds'])/60
    text='''# DEV-054 — Common joint planning with matched belief encoders

**Completed registered development matrix and deterministic replay audit. No full TAC-OSM or commercial superiority claim.** The experiment tests learned regime beliefs with explicit joint regime–fault updates, addressing and entity-local Bellman planning. CDL, CASM, selective computation and learned physical transition rules are absent.

The preceding [DEV-053 null review](DEV-053-NULL-REVIEW.md) remains the interpretation of that experiment. Its reset contrast does not survive multiplicity adjustment. The original excluded split and immutable archives are preserved.

## Registered result

The primary measure is absolute cost/entity difference versus stationary Beta, pooled equally over switching/drift and 0/8/32 public observations. Lower is better. Intervals resample whole stream clusters and initialization seeds, paired with controls. There are 256 distinct evaluation streams, reused across three model seeds and observation budgets; repeated seeds are not additional independent worlds. Intervals are descriptive development evidence, not confirmation.

| Controller | Native cost/entity | Cost minus Beta | Paired 95% interval | Oracle gap closed if eligible | Fault delta/world |
| --- | ---: | ---: | --- | --- | ---: |
'''+ '\n'.join(rows)+'''

Fractional normalization is secondary and is omitted unless observed headroom is at least 0.05/entity and fewer than 1% of denominator draws are nonpositive. All absolute results remain included. Full proper scores, paired fault intervals, bin counts, denominator diagnostics and each budget are in [summary.json](../research/dev054/summary.json) and [curves.csv](../research/dev054/curves.csv).

![Native cost against public observations](../research/dev054/cost-curves.png)

## Matched mechanism comparisons

Negative differences favor MTSK. These Bonferroni 95% familywise intervals adjust for the four registered comparisons. The reliability guard requires no additional faults or constraint violations in every initialization/family, as well as a negative adjusted upper cost bound. The registered full superiority condition also requires a measured deployment stream; that was not tested here.

| MTSK minus reference | Cost difference/entity | Four-comparison adjusted interval | Cost/reliability guard |
| --- | ---: | --- | --- |
'''+ '\n'.join(comparisons)+'''

## Endpoint reliability and economics

Each 32-observation prefix costs 19.2/entity, on top of productive-probe costs below. Break-even is a hypothetical ceiling(acquisition/observed saving) assuming that saving persists for subsequent decisions. Uncertainty is not removed by this division. No finite count is given for nonpositive savings. Probe outcomes do not feed later prefixes; this is not a measured adaptive production stream or a commercial value demonstration.

| Family | Controller | Cost/entity | Cost minus Beta | Faults, seed mean | Oracle headroom | Hypothetical break-even decisions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
'''+ '\n'.join(endpoints)+'''

## Training and forecast diagnostics

All 15 fits received 1,024 updates, four times DEV-053B's 256 per fit. The same 512 training streams and privileged next-rate labels are available to neural training and baseline fitting. Training varies Markov hazards (.02/.08/.16/.32), sine/triangle paths and five periods. Evaluation uses fresh streams and interpolated hazards/periods within those numeric ranges. This is broader temporal training on a fixed one-dimensional endpoint manifold, not arbitrary task meta-learning.

A plateau flag requires absolute relative validation cross-entropy change ≤1% from update 768 to 1,024. It is a diagnostic, not a stopping or selection rule. A failure to plateau limits any architectural comparison; passing that check also does not prove global optimization.

| Encoder | Stored parameters | Parameter bytes | Ideal retained-state bytes | Plateau seeds | Validation improvement 768→1024 by seed |
| --- | ---: | ---: | ---: | ---: | --- |
'''+ '\n'.join(training)+'''

![Training and validation learning curves](../research/dev054/training-curves.png)

The small GRU and hybrid match MTSK's stored parameter count within 1%. MTSK's inherited `write_proj` is unused by the `_step` interface; this is disclosed below rather than padding other models. Matching stored parameters does not match active parameters, FLOPs or all tuning effort. GRU-128 is retained as a larger state-matched reference. The hybrid's HMM is additional to its neural parameters; its two-value NumPy posterior is 16 bytes in this implementation. Training metadata's 8-byte addend describes an ideal FP32 posterior, not actual NumPy storage. Python allocations, catalogue tables, planning workspace and logs are excluded from ideal recurrent-state counts.

| Encoder | Stored parameters | Parameters with gradient path | Median 32-event encoder ms |
| --- | ---: | ---: | ---: |
'''+ '\n'.join(timing)+'''

These one-thread timings include hybrid filtering but exclude token conversion, planner/native execution and logging. They use the first seed and eight validation streams; they do not establish production speed, power or FLOP efficiency. Raw [profiling scope and unused tensors](../research/dev054/profile.json) are preserved.

![Expected repair calibration](../research/dev054/calibration.png)

Calibration bins compare predicted repair probabilities with simulator true probabilities. The plot illustrates the first seed; all seeds' bins/proper scores are retained. Expected log loss against current latent rates includes irreducible uncertainty and does not prove that an oracle-matching posterior is attainable from finite public evidence. Binned calibration can hide conditional errors. The continuous-drift family deliberately misspecifies the two-endpoint mixture class for all joint models.

## What is learned and what executes

MTSK/GRUs learn the next endpoint-mixture weight from public attempt/success history using soft regime cross entropy at all 33 positions. Endpoint rates are fitted once from privileged training labels and shared across neural/HMM arms. The GRU-plus-HMM also receives a posterior computed solely from the same public history. All use exactly the same joint planner and analytic service/inspection belief updates; the joint fault-state dependence is maintained by that explicit machinery. Beta and the privileged oracle use identical duplicate components to represent point-rate forecasts. This removes the old point-versus-mixture planning confound without claiming that neural memory discovered joint physics.

A restored checkpoint runs incrementally on a fresh stream through native Veritas:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/demo_joint_repair_dev054.py \\
  --directory results/dev054 --output /tmp/dev054-live.json \\
  --veritas-root /workspace/veritas
```

After downloading evidence, use its raw run as `--directory` if not extracted to `results/dev054`. The demo checks that incremental public-state updates reproduce batch forecasts and records native actions, value margins, costs and verification. It is the complete execution loop for this component model, not the complete TAC-OSM architecture. The unselected fresh demo leaves 1/2/0 residual faults at 0/8/32 observations: passing the checkpoint/incremental engineering check does not mean fault-free task success. These failed outcomes remain in the raw evidence.

## Integrity and replay

Registration commit `86b1cbc` precedes implementation `0940063` and training. Reporting additions do not alter the frozen protocol/model. Smoke seeds are offset 10,000,000; unit tests use 8541xxx; accepted streams use 5541001–5541005. There is no dropped seed or training cell. The [prior excluded-split record](../research/dev054/ORIGINAL-EXCLUDED-SPLIT-RECORD.md) and [provenance](../research/dev054/provenance.json) remain intact.

'''+f"The same-author separate-process audit reconstructed **{audit['checkpoint_reconstructions']} checkpoints**, replayed **{audit['separate_primitive_probe_executions']:,} probe executions / {audit['separate_probe_actions']:,} primitive actions**, plus **{audit['separate_coupon_actions']:,} fixture actions**. It shares model/generator code, so it establishes deterministic development reconstruction and separate native accounting, not independent scientific replication. Main runtime **{common['elapsed_seconds']:.2f}s**, audit **{audit['elapsed_seconds']:.2f}s**, combined **{compute:.2f} process-minutes**; smoke, reporting, demo, regression and profiling are additional separately logged engineering work. Main+audit execute 30,720 optimizer updates. The complete 1,321-test regression suite passed. Native Veritas remains pinned and clean. Draft [PR #214](https://github.com/FPC-effortless/TAC-OSM/pull/214) remains unmerged.\n"
    text += ('\nThe main/reconstruction tranche exceeded the approximate 30-minute target by '+f'{compute-30:.2f}'+' minutes; the full fixed matrix was retained without outcome-based stopping.\n') if compute>30 else '\nThe main/reconstruction tranche remained within the approximate 30-minute compute target.\n'
    m=r['pooled']['mtsk'];lo,hi=m['delta_95pct'];fs=m['paired_faults_per_world']
    conclusion=f"MTSK's pooled cost difference versus Beta is {m['delta_vs_beta']:+.6f}/entity, with paired 95% interval [{lo:+.6f},{hi:+.6f}]. Its residual-fault difference is {fs['difference']:+.6f}/world, interval [{fs['interval_95pct'][0]:+.6f},{fs['interval_95pct'][1]:+.6f}]. Every adjusted mechanism comparison fails. This is a null on MTSK superiority, with a reliability regression against stationary Beta; the positive point estimate of normalized headroom closure is not an advantage claim.\n\n"
    text=text.replace('## Registered result\n\n','## Registered result\n\n'+conclusion)
    o=r['pooled']['oracle']['paired_faults_per_world']
    text += f"\nThe privileged monetary oracle also has {o['difference']:+.6f} additional faults/world versus Beta. Its cost headroom is therefore not a demonstrated reliability-feasible saving. The planner optimizes the stated monetary penalty, while the separate acceptance guard requires no additional faults. This objective/guard distinction must remain visible; it does not imply that true parameter knowledge is brittle.\n"
    text += '\nChanging from the earlier study also changed data, exposure and evaluation budgets. Cross-study differences do not isolate the causal effect of joint planning. Here, continuous drift can violate the shared endpoint-mixture representation, and the monetary Bellman objective permits fault/cost trade-offs that fail the separate reliability guard. These are model/objective limitations to test, not post-hoc explanations established by this result.\n'
    (ROOT/'docs/DEV-054-RESULTS.md').write_text(text)


if __name__=='__main__':report()
