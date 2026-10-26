"""Process case partition and as-of offer IDs -- no private XES data."""
import unittest
from scripts.s1temp008_bpi_source_partition_audit import (
    original_split,source_prefix_slots)
class PureSourceCaseChecks(unittest.TestCase):
    def test_disjoint_deterministic_case_assignment_independent_of_event_labels(self):
        a=[original_split("case_123") for _ in range(50)]
        self.assertEqual(len(set(a)),1)
        self.assertIn(a[0],{"FIT","DEV","TEST"})
        # The original case ID may not be used as a global model slot.
        self.assertEqual(original_split("case_123"),original_split("case_123"))
    def test_offer_first_disclosure_not_future(self):
        seen=[
            {"activity":"A_Create Application"},
            {"OfferID":"offer-a","activity":"O_Created"},
            {"OfferID":"offer-b","activity":"O_Created"},
            {"OfferID":"offer-a","activity":"O_Sent"},
            {"OfferID":"offer-c","activity":"O_Created"},
            {"activity":"A_Finalized"},
        ]
        full,n=source_prefix_slots(seen)
        self.assertEqual(full,[0,1,2,1,3,0])
        for t in range(len(seen)):
            prior,_=source_prefix_slots(seen[:t+1])
            self.assertEqual(full[:t+1],prior)
    def test_seventh_offer_authorized_and_eighth_denied(self):
        seven=[{"OfferID":f"distinct_{i}"} for i in range(7)]
        slots,n=source_prefix_slots(seven)
        self.assertEqual(slots,[1,2,3,4,5,6,7])
        with self.assertRaises(ValueError):
            source_prefix_slots(seven+[{"OfferID":"eighth"}])
    def test_identical_offer_string_allowed_only_within_case_scope(self):
        rows=[{"OfferID":"A"},{"OfferID":"B"}]
        a,_=source_prefix_slots(rows)
        b,_=source_prefix_slots([{"OfferID":"B"},{"OfferID":"A"}])
        self.assertEqual(a,[1,2]);self.assertEqual(b,[1,2])
    def test_case_partition_is_not_label_based(self):
        ids=[f"Application_{i}" for i in range(1000)]
        counts={g:sum(original_split(i)==g for i in ids)
                for g in ("FIT","DEV","TEST")}
        self.assertTrue(all(x>0 for x in counts.values()))
        self.assertEqual(sum(counts.values()),len(ids))
if __name__=="__main__":unittest.main()
