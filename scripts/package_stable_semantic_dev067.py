"""Preserve all registered fits and decisions, including qualification failures."""
import gzip
import json
from pathlib import Path
import subprocess
from package_research_dev062 import package
from run_learned_stack_dev055 import digest, write


def run():
    root = Path('research/dev067-stable-interface')
    statistics = json.loads((root/'statistics.json').read_text())
    audit = json.loads((root/'audit.json').read_text())
    assert statistics['status'] == audit['status'] == 'COMPLETE'
    interventions = json.loads((root/'interventions.json').read_text())
    assert interventions['status'] == 'COMPLETE' and interventions['checkpoints'] == 120
    assert audit['checkpoints'] == 132 and audit['semantic_interface_cases'] == 132
    assert audit['clock_invariance_cases'] == 264 and audit['batched_state_parity_cases'] == 132
    assert audit['portable_batched_decisions'] == audit['portable_decisions']
    assert audit['telemetry']['max_abs'] <= 1e-8
    assert digest(root/'intervention-trajectories.jsonl.gz') == interventions['details_sha256']
    files = (
        'src/tac_osm/development/belief_agent.py',
        'src/tac_osm/development/belief_bottleneck.py',
        'src/tac_osm/development/semantic_state.py',
        'src/tac_osm/development/stable_semantic.py',
        'src/tac_osm/development/uncertain_world.py',
        'scripts/data_belief_dev062.py', 'scripts/data_belief_dev063.py',
        'scripts/data_semantic_dev065.py', 'scripts/run_semantic_dev065.py',
        'scripts/data_stable_semantic_dev067.py', 'scripts/run_stable_semantic_dev067.py',
        'research/dev067-stable-interface/PROTOCOL.md',
        'research/dev067-stable-interface/protocol.json')
    frozen = 'c39fc6c2110e63e01314d6409ae52f5eb245eb04'
    for name in files:
        assert subprocess.check_output(['git', 'show', frozen+':'+name]) == Path(name).read_bytes(), name
    write(root/'source-provenance.json', dict(frozen_fit_commit=frozen,
        files_sha256={name: digest(name) for name in files}))
    qualifications = {}
    for label in ('primary', 'preselected_candidate'):
        assessment = statistics[label]
        rows = [v for key, v in interventions['per_seed'].items() if key.startswith(assessment['arm']+'/')]
        assert len(rows) == 20
        passing = sum(all(cell['completion'] >= .95 for mode in ('predicted', 'exact')
                          for cell in row[mode]['strata']) for row in rows)
        faults = sum(row[mode]['overall']['faults'] for row in rows for mode in ('predicted', 'exact'))
        gates = {key: value for key, value in assessment['gates'].items() if isinstance(value, bool)}
        gates.update(replay_complete=True, intervention_reliability18=passing >= 18,
                     intervention_zero_faults=faults == 0)
        qualifications[label] = dict(arm=assessment['arm'], gates=gates,
            passing_intervention_seeds=passing, intervention_faults=faults,
            local_synthetic_qualified=all(gates.values()))
    write(root/'qualification.json', dict(status='COMPLETE', assessments=qualifications,
        mechanism_superiority=statistics['mechanism_completion_gain']['practical_5pp_gain'],
        commercial_qualified=False, remote_replay='VERIFY_EXACT_FINAL_HEAD_SEPARATELY'))
    package('dev067-stable-interface', 'results/dev067-main', 'results/dev067-agents', 'results/dev067-smoke-final')
    reference = root/'public-reference.json'
    with gzip.open(root/'public-reference.json.gz', 'wb') as stream:
        stream.write(reference.read_bytes())
    reference.unlink()
    index = json.loads((root/'evidence-index.json').read_text())
    index['decision'] = ('LOCAL_SYNTHETIC_CANDIDATE_QUALIFIED_REMOTE_PENDING'
        if qualifications['preselected_candidate']['local_synthetic_qualified']
        else 'NO_PROMOTION_REGISTERED_GATES_FAILED')
    index['qualification'] = qualifications
    index['commercial_qualification'] = 'UNMET_REAL_FORECAST_AND_REPAIR_OUTCOME_GATES'
    index['derived_evidence_sha256'] = {name: digest(root/name) for name in (
        'statistics.json', 'statistics.csv', 'audit.json', 'interventions.json',
        'intervention-trajectories.jsonl.gz', 'public-reference.json.gz',
        'hardware.json', 'diagnostics.json', 'shuffle.json', 'learning-curves.png',
        'actor-curves.png', 'per-seed.png', 'local-verification.json',
        'source-provenance.json', 'qualification.json')}
    write(root/'evidence-index.json', index)


if __name__ == '__main__':
    run()
