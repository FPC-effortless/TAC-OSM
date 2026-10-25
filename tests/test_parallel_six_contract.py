"""Fail-closed six-family portfolio and immutable PNDS history crosswalk."""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "research/parallel-six/active-variants.json"
LINKS = ROOT / "research/pnds-variant-math-links.json"


class ParallelSixContractTests(unittest.TestCase):
    def test_exact_six_shared_parallel_engine(self):
        from tac_osm.variants.parallel_six import PARALLEL_FAMILIES, parallel_family
        from tac_osm.variants.registry import Variant
        data = json.loads(MANIFEST.read_text())
        self.assertEqual(len(PARALLEL_FAMILIES), 6)
        self.assertEqual(set(v.value for v in Variant),
                         set(m["id"] for m in data["active_models"]))
        self.assertEqual({m["core_id"] for m in data["active_models"]},
                         {"TAC-PARALLEL-DUAL-MODE-V1.7"})
        self.assertEqual(data["shared_core"]["stream_fp32_scalar_count"], 152)
        for variant in Variant:
            self.assertFalse(parallel_family(variant).qualified_real_skill)

    def test_all_historical_pinned_versions_preserved(self):
        active = json.loads(MANIFEST.read_text())
        historical = json.loads(LINKS.read_text())
        self.assertEqual(active["governance"]["legacy_variants_indexed"], 17)
        by_id = {v["id"]: v for v in historical["variants"]}
        rows = active["historical_crosswalk"]
        self.assertEqual(len(rows), len(by_id))
        self.assertEqual({r["historical_id"] for r in rows}, set(by_id))
        for r in rows:
            original = by_id[r["historical_id"]]
            self.assertEqual(r["source_commit"], original["tac"]["commit"])
            self.assertTrue(r["active_families"])
        self.assertFalse(active["governance"]["real_world_superiority_established"])

    def test_fail_closed_unqualified_task_heads(self):
        from tac_osm.variants.parallel_six import (
            require_qualified_domain_head, parallel_family,
        )
        for name in ("system1", "physical", "enterprise", "temporal",
                     "edge_lite", "counterfactual"):
            self.assertFalse(parallel_family(name).authorized_to_act)
            with self.assertRaises(RuntimeError):
                require_qualified_domain_head(name)
        with self.assertRaises(ValueError):
            parallel_family("eswam")


if __name__ == "__main__":
    unittest.main()
