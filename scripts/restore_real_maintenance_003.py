"""Restore recorded physical forecasts without refitting or new sensor evaluation."""
import json
from pathlib import Path
from restore_research_evidence_dev062 import restore
from data_real_maintenance_003 import digest


if __name__=='__main__':
    root=Path('research/real-maint-003');index=json.loads((root/'evidence-index.json').read_text())
    for name,sha in index['derived_evidence_sha256'].items():assert digest(root/name)==sha,name
    restore('real-maint-003')
