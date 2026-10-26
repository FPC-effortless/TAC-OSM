"""S1RET008 deterministic source-entity trust and 7-day censoring tests.

Constructed dates and values ONLY used to verify the source reader; they are
not real world model performance, not manufactured training/evaluation data.
"""
import unittest
from datetime import date,timedelta
import numpy as np
from tac_osm.variants.s1ret008_backblaze_source import (
    SMART_FEATURES,PRIORITY_COUNTERS,outcome_status,
    authorized_observation,validate_source_row,
    known_source_entity_slots,
)

class DriveTruthSourceTests(unittest.TestCase):
    def setUp(self):
        self.day=date(2013,6,1)
        self.dates={self.day+timedelta(days=i) for i in range(10)}
        self.x={f:"0" for f in SMART_FEATURES}
        self.x["smart_5_raw"]="5"

    def test_future_7_day_true_failure_does_not_use_current_label(self):
        failure={self.day+timedelta(days=5)}
        self.assertEqual(outcome_status(self.day,self.dates,failure),1)
        self.assertEqual(outcome_status(self.day+timedelta(days=5),self.dates,failure),None)
        self.assertEqual(outcome_status(self.day+timedelta(days=6),self.dates,failure),None)
        self.assertEqual(outcome_status(self.day+timedelta(days=1),self.dates,failure),1)

    def test_nonfailure_requires_complete_7_future_calendar_days(self):
        self.assertEqual(outcome_status(self.day,self.dates,set()),0)
        g=set(self.dates);g.remove(self.day+timedelta(days=3))
        self.assertEqual(outcome_status(self.day,g,set()),None)
        self.assertEqual(outcome_status(self.day+timedelta(days=9),
                                        self.dates,set()),None)

    def test_earlier_failure_same_entity_is_not_future_label(self):
        prior={self.day}
        future_day=self.day+timedelta(days=1)
        self.assertEqual(outcome_status(future_day,self.dates,prior),0)

    def test_failed_source_record_must_not_enter_model_input(self):
        row={**self.x,"date":self.day.isoformat(),
              "serial_number":"REAL-DEVICE-IDENTITY","failure":"1"}
        serial,x,p,priority,failed=validate_source_row(row,archive_day=self.day)
        self.assertEqual(serial,"REAL-DEVICE-IDENTITY")
        self.assertTrue(failed)
        self.assertEqual(x.shape,(14,))
        self.assertEqual(p.shape,(14,))
        self.assertEqual(priority,1.)
        with self.assertRaisesRegex(ValueError,"unauthorized"):
            authorized_observation(row)
        a,b,c=authorized_observation(self.x)
        spoof=dict(self.x)
        spoof["failure"]="0"
        with self.assertRaisesRegex(ValueError,"unauthorized"):
            authorized_observation(spoof)

    def test_missing_smart_and_actual_signed_negative_cannot_imply_event(self):
        x={k:"" for k in SMART_FEATURES}
        v,p,priority=authorized_observation(x)
        self.assertTrue(np.array_equal(v,np.zeros(14)))
        self.assertEqual(float(p.sum()),0.)
        self.assertEqual(priority,0.)
        x[PRIORITY_COUNTERS[0]]="-5"
        v,p,priority=authorized_observation(x)
        self.assertEqual(priority,0.)

    def test_identity_collisions_or_same_drive_twice_are_rejected(self):
        ids=[f"dr{i}" for i in range(8)]
        slots=known_source_entity_slots(ids)
        self.assertEqual(set(slots.values()),set(range(8)))
        with self.assertRaises(ValueError):
            known_source_entity_slots(["same"]*8)
        with self.assertRaises(ValueError):
            known_source_entity_slots(ids[:7])

    def test_raw_future_label_poison_cannot_change_legal_source_tensor(self):
        row={**self.x,"date":self.day.isoformat(),
              "serial_number":"SERIAL","failure":"0"}
        p0=validate_source_row(row,archive_day=self.day)
        bad={**row,"failure":"1"}
        p1=validate_source_row(bad,archive_day=self.day)
        np.testing.assert_array_equal(p0[1],p1[1])
        np.testing.assert_array_equal(p0[2],p1[2])
        self.assertEqual(p0[3],p1[3])
        self.assertNotEqual(p0[4],p1[4])

if __name__=="__main__":
    unittest.main()
