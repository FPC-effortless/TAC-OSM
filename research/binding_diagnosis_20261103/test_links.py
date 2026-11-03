"""Scalar provenance regressions; no Torch, weights, model or training imports."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"research/scripts"))
from check_pnds_math_links import validate


class Links(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT/"research/pnds-variant-math-links.json").read_text())

    def check(self, change=None):
        data = copy.deepcopy(self.data)
        if change:
            change(next(x for x in data["variants"] if x["id"] == "FLOCK013-PHASE-ADJOINT"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"links.json"
            path.write_text(json.dumps(data))
            return validate(path)

    def test_full_29_partial_records_validate(self):
        self.assertEqual(self.check()["source_pinned_versions"], 29)

    def test_model_source_cannot_be_replaced(self):
        with self.assertRaises(ValueError):
            self.check(lambda row: row["tac"].update(commit="f"*40))

    def test_math_source_cannot_be_replaced(self):
        with self.assertRaises(ValueError):
            self.check(lambda row: row["pnds"].update(commit="f"*40))

    def test_shared_kernel_cannot_replace_specific_formulation(self):
        with self.assertRaises(ValueError):
            self.check(lambda row: row["pnds"].update(lean_path="PNDS/TACOSMVariantContracts.lean"))

    def test_partial_math_cannot_be_promoted(self):
        with self.assertRaises(ValueError):
            self.check(lambda row: row.update(lean_full_graph_proved=True))


if __name__ == "__main__":
    unittest.main()
