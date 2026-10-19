"""Preserve the complete physical-forecast matrix and explicit unit correction."""
import json
from pathlib import Path
import subprocess
from package_research_dev062 import package
from data_real_maintenance_003 import write,digest


def run():
    root=Path('research/real-maint-003')
    stats=json.loads((root/'statistics.json').read_text());audit=json.loads((root/'audit.json').read_text())
    hardware=json.loads((root/'hardware.json').read_text());parity=json.loads((root/'unit-correction-parity.json').read_text())
    assert stats['status']==audit['status']=='COMPLETE' and audit['checkpoints']==88 and audit['references']==8
    assert audit['telemetry']['max_abs']<=1e-8 and audit['portable_predictions']==8
    assert parity['status']=='COMPLETE' and not parity['numerical_predictions_changed']
    frozen='7fedb402ef1ea04c4ab2f471e79494ec9d0d27e2'
    files=('src/tac_osm/development/thermal_forecast.py','scripts/data_real_maintenance_003.py',
        'scripts/run_real_maintenance_003.py','scripts/analyze_real_maintenance_003.py',
        'research/real-maint-003/PROTOCOL.md','research/real-maint-003/protocol.json')
    for name in files:assert subprocess.check_output(['git','show',frozen+':'+name])==Path(name).read_bytes(),name
    write(root/'source-provenance.json',dict(frozen_fit_source=frozen,files_sha256={n:digest(n) for n in files},
        erratum='Frozen protocol incorrectly saysC; rawtraining/evaluationF unchanged. Corrected labels only in portable adapter, with preserved pre-correction export.'))
    gates={key:value for key,value in stats['gates'].items() if isinstance(value,bool)}
    gates.update(source_checkpoint_standalone_replay=True,cpu10ms=hardware['candidate_batch1_cpu10ms_pass'],
        corrected_unit_adapter_weight_prediction_parity=True)
    qualification=dict(status='COMPLETE',gates=gates,numerical_forecast_gates_pass=all(gates.values()),
        candidate=stats['candidate'],scope='RAW_F_ROOM_FORECAST_COMPONENT_ONLY_WITH_DECLARED_UNIT_ERRATUM',
        commercial_qualified=False,remote='VERIFY_NEW_FINAL_HEAD_SEPARATELY')
    write(root/'qualification.json',qualification)
    package('real-maint-003','results/real003-main','results/real003-agents','results/real003-smoke-frozen')
    index=json.loads((root/'evidence-index.json').read_text())
    (root/'learned-agent.tar.gz').rename(root/'learned-forecaster.tar.gz')
    index['archives']['learned-forecaster.tar.gz']=index['archives'].pop('learned-agent.tar.gz')
    index.update(decision='LOCAL_NUMERICAL_GATES_PASS_UNIT_ERRATUM_REMOTE_PENDING' if all(gates.values()) else 'NO_PROMOTION_NUMERICAL_GATE_FAILED',
        qualification=qualification,commercial_qualification='UNMET_REAL_REPAIR_POLICY_COST_OUTCOME_GATES',
        scope='All80fresh+8pilot neuralraw/finalEMA checkpoints,37train15validation11testCSVdays,allreferencefits,windows,curvesanderrors preserved. Onebuilding, measured fault-injection data. Demo packet unitsF; original protocol unit label erroneous and retained. No actual repair policy/ROI or allanalogchannel claim.')
    names=('statistics.json','statistics.csv','audit.json','hardware.json','qualification.json',
        'unit-erratum.json','unit-correction-parity.json','UNIT-ERRATUM.md',
        'pre-unit-correction-audit.json','pre-unit-correction-export.tar.gz',
        'learning-curves.png','per-channel.png','first-day-forecasts.png',
        'local-verification.json','regression.log','source-provenance.json')
    index['derived_evidence_sha256']={name:digest(root/name) for name in names}
    write(root/'evidence-index.json',index)


if __name__=='__main__':run()
