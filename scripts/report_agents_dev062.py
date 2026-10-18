"""Render measured outcomes without promoting unknown scientific/commercial gates."""
import argparse
import json
from pathlib import Path
import torch
from run_learned_stack_dev055 import write


def run(study,directory):
    root=Path('research')/study;result=json.loads((root/'statistics.json').read_text());arms=result['arms'];identity=study=='dev0615';lines=[f'# {study.upper()} — registered learned-agent research results','', '**Status: research evidence; no promotion or commercial-readiness claim.**','',f"Completed {result['run']['fits']} fresh fits and {result['run']['pilot_fits']} validation-only pilot fits. Fitting/collection/validation CPU: {result['run']['fitting_cpu_seconds']/60:.2f} minutes. All twenty training seeds retained per arm.",'', '| Arm | Completion | Mean economic cost | Residual faults | Passing seeds |','|---|---:|---:|---:|---:|']
    for a,v in arms.items():lines.append(f"| {a} | {(v['completion_mean'] if identity else v['mean_completion']):.2%} | {(v['cost_mean'] if identity else v['mean_cost']):.4f} | {v['faults']} | {(v['passing_combined_target'] if identity else v['passing_seeds'])}/20 |")
    lines+=['','Passing-seed criteria differ by study; consult the registration. Completion and physical repair are reported separately in the per-seed evidence. Averages do not replace the reliability gate.','', '## Acceptance gates','']
    for k,v in result['gates'].items():lines.append(f'- {k}: **{v}**')
    if identity:
        lines+=['',f"The explicit public-history reference achieves {result['reference']['completion']:.2%} completion at cost {result['reference']['cost']:.4f}. This benchmark remains public-rule solvable. The identity treatment bundles separated content and identity augmentation; the robustness treatment bundles shared multi-policy DAgger, EMA and curriculum. Neither factorial isolates each ingredient.",'','The public manifest, hard cosine routing and relationship-key stream are structural priors supplied to every control. Recurrent content, CDL strengths, graph/operator computation, action predictions and control are trained. This is not entity discovery, symbolic CASM or learned indexing.']
    else:
        lines+=['','The exact public reference is an information-matched, reliability-constrained depth2 planner; it is not globally optimal. The privileged control sees hidden state and is a different information ceiling. Latest-observation shortcuts are reported separately.','', 'This is posterior-supervised StageA. Public action-start records identify transitions; repair receipts have a state-independent two-entity schema. No filter, teacher, physical transition law, analytical planner or stop override enters exported inference. The latent state vocabulary and identity binding remain supplied priors. StageB learning without posterior targets has not been demonstrated.']
    if not identity:
        rows=list(arms['tac']['per_seed'].values());kl=[r['fixed']['posterior_KL'] for r in rows];secondary=[{'cell_index':i,'worst_seed_completion':min(r['secondary'][i]['completion'] for r in rows),'residual_faults_all_seeds':sum(r['secondary'][i]['faults'] for r in rows)} for i in range(len(rows[0]['secondary']))]
        lines+=['','## Distribution and transfer qualification','',f"The registered KL gate is a median across 20 seeds, not a per-seed guarantee. {sum(v<=.02 for v in kl)}/20 TAC seeds individually meet KL<=.02; fixed-history KL range[{min(kl):.6f},{max(kl):.6f}]. Completion reliability and posterior accuracy are different gates.",'', 'Secondary cells remain separate from primary acceptance; in protocol order:']
        for v in secondary:lines.append(f"- Cell{v['cell_index']}: worst-seed completion {v['worst_seed_completion']:.2%}, total residual faults across shared-seed evaluations {v['residual_faults_all_seeds']}.")
        lines+=['','Twenty out of twenty successful training seeds has a Wilson95% seed-pass interval approximately[83.9%,100%], conditional on this training corpus and shared worlds. It does not prove>=90% population training reliability or a real-world fault bound.']
    lines+=['','## Artifacts and limits','','- [Protocol](PROTOCOL.md) and [per-seed table](statistics.csv).','- [Complete statistics](statistics.json), [local replay audit](audit.json), [measured hardware costs](hardware.json), and [SHA256 evidence index](evidence-index.json).','- Exported seed is preregistered, not chosen for favorable held-out performance. `learned-agent.tar.gz` includes model, weights, public JSON-lines runtime and runnable examples.','- Restore evidence by concatenating indexed archive parts in order, then extracting the gzip tar archive into `results/`. Raw, EMA deployment and failed outcomes remain available.','- Native Veritas evidence uses the same recorded synthetic effects. It is execution reproducibility, not independently validated physics.','- Remote action matching remains UNKNOWN until the corresponding CI audit actually passes. Float64 alone does not prove hardware invariance.','- Synthetic intervention units are not money; CPU measurements are not production SLAs. No measured DRAM-traffic/FLOPs reduction is claimed.','', 'Commercial validation still requires a real task, permissioned outcome-labelled traces, actual prices and an authorized shadow trial. See [commercial acceptance contract](../commercial/READINESS.md).']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    # Acquisition accounting retains both actual charged prefixes and unfinished-work penalties.
    data=Path(directory)/'_datasets';acquisition={}
    if identity:
        for name in ('train','validation'):
            rows=torch.load(data/f'{name}.pt',weights_only=True);acquisition[name]={'snapshots':len(rows),'prefix_collection_synthetic_units':sum(float(r['collection_cost']) for r in rows)}
    groups=json.loads((Path(directory)/'groups.json').read_text());acquisition['student_union_collection_units']=sum(g['collection_cost'] for g in groups);acquisition['scope']='Shared corpus/union cost once per seed group; synthetic prefix collection costs, not amortized commercial ROI. Full raw trajectories preserve terminal penalties separately. Exact simulator targets are training assistance, not free real labels.';write(root/'acquisition-cost.json',acquisition)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',choices=['dev0615','dev062','dev062-r1'],required=True);p.add_argument('--directory',required=True);a=p.parse_args();run(a.study,a.directory)
