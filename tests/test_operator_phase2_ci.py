import json
from pathlib import Path

from scripts.run_axon_structmeans_seca_002 import main


def test_registered_axon_structmeans_seca_runner(capsys):
    result = main()
    artifact = Path("artifacts/TACOSM-AXON-STRUCTMEANS-SECA-002.json")
    artifact.parent.mkdir(exist_ok=True)
    artifact.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with capsys.disabled():
        print("\n=== TACOSM-AXON-STRUCTMEANS-SECA-002 ===")
        print(json.dumps(result, indent=2, sort_keys=True))
    assert result["pst"]["heldout_transition_accuracy"] >= 0.95
    assert result["structmeans"]["signature_purity"] >= 0.85
    assert result["axon"]["single_operator_route_success"] >= 0.90
    assert result["seca"]["verified"] >= 1
