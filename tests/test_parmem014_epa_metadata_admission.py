"""Unit diagnostics of EPA publisher metadata admission. These local constructed
owner/time fixtures never support real-world efficacy or source performance.
"""
from __future__ import annotations
import unittest

from scripts.parmem014_epa_metadata_admission import (
    source_owner,source_phase,clean_source_hour,checked_protocol
)

class SourceMetadataNotPhysicalPerformanceTests(unittest.TestCase):
    def test_registered_protocol_covers_protected_fields(self):
        p=checked_protocol()
        self.assertIn("Sample Measurement",p["forbidden_numeric_fields_during_intake"])
        self.assertTrue(p["seal_policy"].startswith("no numeric"))
        self.assertFalse(p["new_neural_training_or_test_eval"])
        self.assertEqual(p["minimum_source_hour_metadata_count_per_monitor_in_its_assigned_phase"],160)

    def test_source_owner_uses_all_admin_identity_fields(self):
        raw={"State Code":"1","County Code":"89","Site Num":"14",
             "Parameter Code":"88101","POC":"2"}
        self.assertEqual(source_owner(raw),"01-089-0014-88101-2")
        changed=dict(raw,POC="3")
        self.assertNotEqual(source_owner(changed),source_owner(raw))
        changed=dict(raw,**{"Parameter Code":"88502"})
        with self.assertRaises(ValueError):source_owner(changed)

    def test_split_source_owner_uses_hash_not_pollutant_value(self):
        p=checked_protocol()
        for owner in (
            "01-089-0014-88101-2","04-013-0001-88101-1",
            "16-077-0006-88101-1","49-053-0007-88101-2"):
            matched=[phase for phase,date in (
              ("train","2024-03-01"),("dev","2024-09-20"),
              ("sealed_test","2024-12-01")) if
              source_phase(owner,date,p)==phase]
            self.assertEqual(len(matched),1)
            role=matched[0]
            if role=="train":
                self.assertIsNone(source_phase(owner,"2024-09-20",p))
            if role=="sealed_test":
                self.assertIsNone(source_phase(owner,"2024-07-01",p))

    def test_recorded_utc_hour_validation_and_rejection(self):
        self.assertEqual(
            clean_source_hour({"Date GMT":"2024-02-29","Time GMT":"17:00"}),
            "2024-02-29T17:00:00Z")
        with self.assertRaises(ValueError):
            clean_source_hour({"Date GMT":"2024-12-01","Time GMT":"17:30"})
        with self.assertRaises(ValueError):
            clean_source_hour({"Date GMT":"2024-12-01","Time GMT":"25:00"})
        with self.assertRaises(ValueError):
            clean_source_hour({"Date GMT":"2025-01-01","Time GMT":"00:00"})

if __name__=="__main__":
    unittest.main()
