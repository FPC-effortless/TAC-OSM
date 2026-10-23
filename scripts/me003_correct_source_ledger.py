#!/usr/bin/env python3
"""ME003 controlled correction, from archived original real-source counter ledger.
No publisher re-download, model fitting or reserved future-time inspection.
"""
import hashlib, json
from pathlib import Path

SOURCE_ARCHIVE_SHA = "aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a"
BAD_INDEX_COLUMNS = {"", "Unnamed: 0", "index"}

def correct(original,output):
    original=Path(original);output=Path(output)
    data=json.loads(original.read_text())
    assert data["study"]=="ME003" and data["source"]["archive_sha256"]==SOURCE_ARCHIVE_SHA
    assert data["model_fits"]==0 and data["future_reserved_labels_read"] is False
    assert data["rows_inspected"]==222113 and data["eligible_short_interval_pairs"]==222076
    dropped=[f for f in data["features"] if f in BAD_INDEX_COLUMNS]
    real=[f for f in data["features"] if f not in BAD_INDEX_COLUMNS]
    counts={f:data["per_feature_transitions"][f] for f in real}
    both=[f for f in real if counts[f]["pos"]>=100 and counts[f]["neg"]>=100]
    assert dropped == [""], f"unexpected index columns: {dropped}"
    assert len(real)==15, f"expected real physical signals, got {len(real)}"
    assert len(both)==12
    corrected={
        "status":"CORRECTED_ANNOTATION_FROM_ARCHIVED_REAL_SOURCE_ONLY",
        "original_workflow_run_id":39338118254,
        "original_artifact_id":12227571362,
        "original_evidence_file_sha256":hashlib.sha256(original.read_bytes()).hexdigest(),
        "publisher_zip_sha256":SOURCE_ARCHIVE_SHA,
        "discarded_nonphysical_columns":dropped,
        "real_sensor_count":len(real),"real_sensors":real,
        "bidirectional_sensor_count":len(both),"bidirectional_sensors":both,
        "per_sensor_transition_counts":counts,
        "rows_inspected_from_original":data["rows_inspected"],
        "short_interval_pairs_from_original":data["eligible_short_interval_pairs"],
        "model_fits":0,"new_real_source_reads":0,"future_reserved_labels_read":False,
        "promoted":False,
        "claim":"SOURCE ELIGIBILITY ONLY: no world-model prediction advantage"
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(corrected,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:corrected[k] for k in ("status","real_sensor_count","bidirectional_sensor_count","new_real_source_reads")},sort_keys=True))
    return corrected

if __name__=="__main__":
    correct("results/me003/source-audit.json","results/me003-correction/corrected-source-audit.json")
