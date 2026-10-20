"""ME006 software checks only — no external outcome authenticity established."""
import unittest
from dataclasses import replace
from tac_osm.verified_promotion import VerifiedPromotion, AttestedEvidence

def att(receipt,context,**changes):
    return replace(AttestedEvidence(receipt,context,True,True,True,True,True,True,True),**changes)

class VerifiedPromotionContracts(unittest.TestCase):
    def setUp(self):
        self.ledger=VerifiedPromotion(lambda _key,event:event)
        assert self.ledger.propose("entity",{"facts":[1]})

    def test_no_verifier_fails_closed(self):
        l=VerifiedPromotion(None);l.propose("key",123)
        with self.assertRaises(PermissionError):l.observe("key",att("r","c"))
        self.assertEqual(l.tier("key"),"provisional")

    def test_each_gate_blocks_promotion(self):
        for field in ("authenticated","independent_context","utility",
                      "stability","integrity","cost"):
            with self.subTest(field=field):
                l=VerifiedPromotion(lambda _,event:event);l.propose("k",{"x":1})
                bad=att("bad-"+field,"one",**{field:False})
                if field=="authenticated":
                    with self.assertRaises(PermissionError):l.observe("k",bad)
                else:self.assertFalse(l.observe("k",bad))
                self.assertFalse(l.observe("k",att("ok-"+field,"two")))
                self.assertEqual(l.tier("k"),"provisional")

    def test_duplicate_receipt_and_context_no_double_count(self):
        self.assertFalse(self.ledger.observe("entity",att("r1","a")))
        with self.assertRaises(ValueError):self.ledger.observe("entity",att("r1","b"))
        with self.assertRaises(ValueError):self.ledger.observe("entity",att("r2","a"))
        self.assertTrue(self.ledger.observe("entity",att("r3","b")))
        self.assertEqual(self.ledger.tier("entity"),"persistent")
        with self.assertRaises(ValueError):self.ledger.observe("entity",att("r4","c"))

    def test_negative_verified_outcome_quarantines(self):
        self.assertFalse(self.ledger.observe("entity",att("r1","a",observed_success=False)))
        self.assertEqual(self.ledger.tier("entity"),"quarantined")
        self.assertFalse(self.ledger.propose("entity","tamper"))

    def test_unverified_utility_does_not_accumulate(self):
        self.assertFalse(self.ledger.observe("entity",att("r1","a",utility=False)))
        self.assertFalse(self.ledger.observe("entity",att("r2","b")))
        self.assertEqual(self.ledger.tier("entity"),"provisional")

    def test_payload_is_copied_before_and_after_promotion(self):
        payload={"numbers":[1]};l=VerifiedPromotion(lambda _,event:event)
        l.propose("k",payload);payload["numbers"].append(2)
        l.observe("k",att("r1","one"));l.observe("k",att("r2","two"))
        a=l.inspect_persistent("k");a["numbers"].append(3)
        self.assertEqual(l.inspect_persistent("k"),{"numbers":[1]})

    def test_caller_object_does_not_count_as_attestation(self):
        with self.assertRaises(PermissionError):self.ledger.observe("entity",object())
        self.assertEqual(self.ledger.snapshot()["persistent"],0)

    def test_minimum_contexts_rejects_insecure_values(self):
        for n in (0,1,-1,True):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):VerifiedPromotion(lambda _,e:e,n)

if __name__=="__main__":unittest.main(verbosity=2)
