#!/usr/bin/env python3
"""Archive an already completed, hash-bound source audit; no model execution."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--archive-sha256',required=True)
    p.add_argument('--run-id',type=int,required=True)
    p.add_argument('--artifact-id',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    payload=a.archive.read_bytes()
    if sha(payload)!=a.archive_sha256:raise ValueError('downloaded source artifact digest differs')
    with zipfile.ZipFile(a.archive) as z:
        first=z.read('first/RESULTS.json')
        repeated=z.read('replay/RESULTS.json')
        receipt=json.loads(z.read('REPLAY_RECEIPT.json'))
    if first!=repeated or sha(first)!=receipt['results_sha256']:
        raise ValueError('source audit replay bytes differ')
    report=json.loads(first)
    spec=json.loads((a.output/'FAIR_COMPARISON.json').read_text())
    if (not report['all_exercised_end_to_end_invariance_checks_pass'] or report['training_steps']!=0
        or report['protected_TEST'] is not False or report['windows']!=258 or len(report['groups'])!=22
        or report['source_sha256']!=spec['real_world_source']['sha256']
        or report['cohort_sha256']!=spec['split_policy']['cohort_sha256']
        or not report['checkpoint_and_parameter_immutability']):
        raise ValueError('incomplete/failing or mismatched original source audit')
    model_names={str(r['seed'])+':'+r['arm'] for r in spec['models']}
    if {g['cutoff_hour'] for g in report['groups']}!=set(spec['cutoff_hours']):
        raise ValueError('frozen source cutoffs differ')
    all_geometry={n:[] for n in model_names}
    unused_sensitive=0
    comparisons=0
    changed_rows=0
    for g in report['groups']:
        if set(g['baseline_determinism'])!=model_names or any(v is not None for v in g['baseline_determinism'].values()):
            raise ValueError('unstable original baseline')
        if {c['mode'] for c in g['controls']}!={'values','missingness'} or len(g['controls'])!=2:
            raise ValueError('missing preregistered source control')
        for n in model_names:all_geometry[n]+=g['baseline_geometry'][n]
        for c in g['controls']:
            if (not c['raw_future_changed'] or not all(c['consumed_prefix_invariant'].values())
                or not all(c['consumed_normalizer_invariant'].values()) or set(c['models'])!=model_names
                or any(v['first_divergence'] is not None for v in c['models'].values())):
                raise ValueError('actual source-to-prediction invariant failed')
            comparisons+=len(g['source_cases'])*len(model_names)
            changed_rows+=c['archive']['changed_FIT_rows']
            unused_sensitive+=not c['unused_intermediate_source_frames_normalizer_invariant']
    if set(report['positive_control']['models'])!=model_names or not all(v['actual_encoder_activation_changed'] for v in report['positive_control']['models'].values()):
        raise ValueError('inert source-visible encoder positive control')
    geometry={}
    for n,windows in sorted(all_geometry.items()):
        queries=[q for w in windows for q in w['queries']]
        eligible=[q for q in queries if q['actual_target_write']]
        geometry[n]=dict(windows=len(windows),queried_source_identities=len(queries),
            eligible_actual_target_writes=len(eligible),inconclusive_no_actual_write=len(queries)-len(eligible),
            mean_native_query_different_clock_cosine=statistics.mean(w['mean_different_clock_query_key_cosine'] for w in windows),
            mean_target_key_cosine=statistics.mean(q['target_cosine'] for q in eligible) if eligible else None,
            mean_strict_rank_lower_bound=statistics.mean(1+q['strictly_higher_cosine_legal_writes'] for q in eligible) if eligible else None,
            mean_tied_writer_count=statistics.mean(q['tied_legal_writes'] for q in eligible) if eligible else None,
            unique_best_target_key_fraction=statistics.mean(q['strictly_higher_cosine_legal_writes']==0 and q['tied_legal_writes']==1 for q in eligible) if eligible else None)
    summary=dict(study_id=report['study_id'],status='PASS_EXERCISED_SOURCE_TO_PREDICTION_INVARIANCE',
        evidence_level='internal_mechanism_diagnostic_not_capability_improvement',
        windows=258,cutoffs=22,models=4,future_source_controls=44,
        model_window_future_control_comparisons=comparisons,
        changed_FIT_rows_across_separate_controls=changed_rows,
        unused_future_sensitive_temporary_normalizers=unused_sensitive,
        unused_transform_status='FUTURE_SENSITIVE_UNUSED_INTERMEDIATE' if unused_sensitive else 'NO_SENSITIVITY_OBSERVED',
        geometry=geometry,source_control_replay=receipt['status'],
        source_code_commit='b472476dabd7bb7648ec8e0af653896d1382a0f0',
        source_run_id=a.run_id,source_artifact_id=a.artifact_id,source_artifact_zip_sha256=a.archive_sha256,
        original_results_sha256=sha(first),models_and_history_unchanged=True,training_steps=0,GPU_minutes=0,protected_TEST=False,
        local_AD='INCONCLUSIVE_NOT_IMPLEMENTED',whole_model_IEEE_AD='OPEN',
        prior_memory_usefulness='HISTORY_QUERY016_BOTH_SEEDS_FAIL_PRESERVED',
        prior_arbitrary_runner_bitwise_portability='OPEN_PRIOR_FAILURE_PRESERVED')
    packed=gzip.compress(first,mtime=0)
    out=a.output/'ORIGINAL_RESULTS.json.gz'
    if out.exists() and out.read_bytes()!=packed:raise ValueError('refuse replacing existing source result bytes')
    out.write_bytes(packed)
    summary['compressed_original_results_sha256']=sha(packed)
    save(a.output/'SUMMARY.json',summary)
    save(a.output/'REPLAY_RECEIPT.json',receipt)
    lines=['# HISTORY-SOURCE017: original-source reload and write/query observability','',
        '**PASS for the exercised source-to-prediction invariance checks. No capability improvement claimed.**','',
        'All 258 frozen primary/secondary windows, 22 cutoffs and four original HISTORY-QUERY016 checkpoints were audited. Both future value and missingness controls reached the actual reloaded source suffix. Every consumed raw/value/mask/age/last prefix and outer fitted statistic stayed exactly unchanged. Actual encoder/write-module calls, full state including active owner flags, and native historical/forecast outputs stayed unchanged. Baseline repeats were exact; the source-visible positive edit changed all four actual encoder activations. Checkpoints/parameters remained unchanged and gradients absent.','',
        'The fresh-process full audit reproduced the complete receipt byte-for-byte. There were '+str(comparisons)+' model-window future-control comparisons, not independent observations. Controls are derived synthetic perturbations of authorized real FIT records, never authentic alternative outcomes. They supply no prediction improvement evidence. No new training, GPU, protected TEST, promotion or merge.','',
        '## Boundary that must not be hidden','',
        str(unused_sensitive)+' of 44 controls changed the discarded temporary source_frames normalizer. That intermediate is future-sensitive if used for early predictions. The actual original training wrapper consumes only raw from it and recomputes statistics from preMarch2014, so the consumed pipeline passed. This distinction does not certify every use of the generic reader.','',
        '## Trained source-key geometry, descriptive only','',
        'The source-pinned writer encoder takes physical observations, observation masks, elapsed-missing duration and causal last values; absolute calendar and owner ID are absent from its input. Owner isolation is supplied by separate stream state. The historical read query uses the requested past calendar plus current fast/slow belief. Source identities below are real owner/hour writer identities, not oracle keys supplied to predictions. Missing actual writes remain INCONCLUSIVE.','',
        '| Seed/model | Windows | Native different-clock key cosine | Target/write cosine | Mean rank lower bound | Unique-best target fraction |',
        '|---|---:|---:|---:|---:|---:|']
    for n,s in sorted(geometry.items()):
        fmt=lambda v:'INCONCLUSIVE' if v is None else f'{v:.6f}'
        lines.append('|'+n+'|'+str(s['windows'])+'|'+fmt(s['mean_native_query_different_clock_cosine'])+'|'+fmt(s['mean_target_key_cosine'])+'|'+fmt(s['mean_strict_rank_lower_bound'])+'|'+fmt(s['unique_best_target_key_fraction'])+'|')
    lines+=['','For the fastslow control, the historical head reads a transient fast/slow feature product rather than persistent associative memory; its query/write cosine is only a geometric reference. Cosine and rank are mechanism observations, not ground-truth reconstruction or a usefulness gate. Temporal signal could be inferred indirectly from physical context; missing an explicit writer clock does not prove impossibility. No geometry value rescues the prior failed real-data usefulness gates.','',
        '## Durable source and limits','',
        'Source run: https://github.com/FPC-effortless/TAC-OSM/actions/runs/'+str(a.run_id)+'. Code head b472476dabd7bb7648ec8e0af653896d1382a0f0; original inference/reader source fd201207cb76b30470f64e1fc1baecf9de44560d. UCI501 original publisher archive/license receipt and frozen source identities/checkpoint hashes remain in FAIR_COMPARISON.json and history_audit014.','',
        'ORIGINAL_RESULTS.json.gz decompresses to the exact full original RESULT JSON; SHA256 '+sha(first)+'. It retains per-window real key geometry, all derived source hashes, mutation counts, prefix receipts, positive control and actual trace fingerprints. The publisher parent hash is distinct from every declared derived hash; the default original checksum guard rejected undeclared derivatives.','',
        'The first implementation run 40501595685 stopped on quoted CSV headers before model inference; proper CSV handling and regression fixtures fixed it. That failure is preserved. Prior HISTORY-QUERY016 negative scores and arbitrary-runner replay failure remain unchanged. Whole-model PNDS IEEE/AD and local probe AD are OPEN/INCONCLUSIVE. These scoped synthetic source tests cannot establish absence of every hidden information channel, reward certification, safety or real-world capability.','',
        'Source-reloaded future-poison is now exercised for this exact frozen cohort and pipeline; historical INCONCLUSIVE records remain immutable. No subsequent model fit is authorized by this receipt alone.']
    if not (a.output/'RESULTS.md').exists():
        (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))


if __name__=='__main__':main()
