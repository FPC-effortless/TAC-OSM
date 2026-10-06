#!/usr/bin/env python3
"""Single fixed functional-completion runway for the signed-CASM learner.

This is development-only. It evaluates fresh episodes from TRAIN_COMBOS and
never uses the retired E2E-005 holdout. The 1200-step horizon is a fixed
completion runway, not a parameter sweep and not a scientific endpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_plm_functional_repair_dev import run

if __name__ == "__main__":
    result = run(
        steps=1200,
        artifact_name="PLM-FUNCTIONAL-COMPLETION-SIGNED-CASM.json",
    )
    assert result["not_scientific_evidence"] is True
    assert result["retired_e2e005_holdout_used"] is False
