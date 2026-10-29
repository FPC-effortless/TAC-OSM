import unittest
from scripts.audit_langmem084_controls import target_in_decoy_context

class HistoricalContextAuditTests(unittest.TestCase):
    def test_rule_matches_actual_four_byte_context_membership(self):
        observed=bytes(range(200));anchor=52;target=observed[anchor]
        for b in range(4,len(observed)):
            self.assertEqual(target_in_decoy_context(anchor,b),target in observed[b-4:b])

if __name__=='__main__':unittest.main()
