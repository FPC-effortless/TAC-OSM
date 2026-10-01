"""CI-only full execution of the registered integrated research runner."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.run_integrated_execution_feedback_axon_001 import main


def test_registered_integrated_runner(capsys):
    main()
    artifact = Path("artifacts/TACOSM-INTEGRATED-001.json")
    assert artifact.exists()
    data = json.loads(artifact.read_text())
    with capsys.disabled():
        print("\n=== TACOSM-INTEGRATED-001 ===")
        print(json.dumps(data["c5"], sort_keys=True))
        print(json.dumps(data["procedural"], sort_keys=True))
