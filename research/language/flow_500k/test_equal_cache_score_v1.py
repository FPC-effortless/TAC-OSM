"""Constructed math/validation diagnostics only; no real-data claims."""
import copy
import hashlib
import json
import unittest
from equal_cache_score_v1 import score_document

D = "a" * 64

def dist(y, confidence):
    return [confidence if i == y else (1-confidence)/127 for i in range(128)]

def fixture():
    prior = dist(1,0.125)
    prior_digest = hashlib.sha256(json.dumps(prior,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    return {
      "schema":"graph012-equal-cache-v1",
      "policy":{"exact_weight":0.6,"approx_weight":0.1,"cache_capacity":512,"cache_order":3,
                "tau":0.05,"query_topk":8,"query_temperature":0.25,
                "query_checkpoint_sha256":D,"fit_prior_sha256":prior_digest},
      "data":{"source_manifest_sha256":D,"scoring_protocol_sha256":D,"split":"DEV", "prior_test_exposures": True},
      "fit_prior":prior,
      "models":{
          "tac":{"architecture":"WaveCED","active_parameters":499290,"training_updates":120,
                 "training_tokens":122880,"training_flops":1000000,"fit_sample_sha256":D,"checkpoint_sha256":D},
          "ssm":{"architecture":"selective diagonal","active_parameters":498784,"training_updates":120,
                 "training_tokens":122880,"training_flops":1000000,"fit_sample_sha256":D,"checkpoint_sha256":D}},
      "rows":[
          {"owner_id":"document-1","source_sha256":D,"ordinal":10,"target":1,
           "exact_hit":True,"approx_hit":False,"exact_posterior":dist(1,.9),
           "native":{"tac":dist(1,.5),"ssm":dist(1,.7)}},
          {"owner_id":"document-1","source_sha256":D,"ordinal":11,"target":1,
           "exact_hit":False,"approx_hit":True,"approx_posterior":dist(1,.9),
           "native":{"tac":dist(1,.5),"ssm":dist(1,.7)}},
          {"owner_id":"document-1","source_sha256":D,"ordinal":12,"target":1,
           "exact_hit":False,"approx_hit":False,
           "native":{"tac":dist(1,.5),"ssm":dist(1,.7)}}]}

class MathContracts(unittest.TestCase):
    def test_exact_miss_and_no_hit_laws(self):
        d=fixture();r=score_document(d)
        self.assertEqual(r["routing"],{"exact":1,"approx":1,"no_retrieval":1})
        self.assertEqual(r["evidence_status"],"SCORING_ONLY_NOT_ARCHITECTURE_QUALIFIED")
        import math
        expected=-(math.log(.4*.5+.6*.9)+math.log(.9*.5+.1*.9)+math.log(.5))/3
        self.assertAlmostEqual(r["nll"]["tac"]["same_cache"],expected)
        self.assertGreater(r["paired_differences_nats_per_char"]["tac_minus_ssm_equal_cache_nll"],0)
        self.assertEqual(len(r["fairness_and_promotion_gaps"]),2)
    def test_fail_closed_on_same_owner_ordinal_duplicate_and_gate(self):
        d=fixture();d["rows"].append(copy.deepcopy(d["rows"][0]))
        with self.assertRaisesRegex(ValueError,"duplicate"):score_document(d)
        d=fixture();d["rows"][0]["approx_hit"]=True
        with self.assertRaisesRegex(ValueError,"exact priority"):score_document(d)
        d=fixture();d["rows"][1]["exact_posterior"]=dist(1,.9)
        with self.assertRaisesRegex(ValueError,"unexpected exact"):score_document(d)
    def test_wrong_budget_source_and_distributions(self):
        d=fixture();d["models"]["ssm"]["active_parameters"]=300000
        d["models"]["ssm"]["training_flops"]=900000
        gaps=score_document(d)["fairness_and_promotion_gaps"]
        self.assertTrue(any("parameters" in g for g in gaps))
        self.assertTrue(any("FLOP" in g for g in gaps))
        d=fixture();d["rows"][0]["native"]["tac"][0]=-.2
        with self.assertRaisesRegex(ValueError,"strictly-positive"):score_document(d)
        d=fixture();d["rows"][0]["source_sha256"]="wrong"
        with self.assertRaisesRegex(ValueError,"owner/source"):score_document(d)
        d=fixture();d["fit_prior"][2]=0.02
        with self.assertRaisesRegex(ValueError,"FIT prior"):score_document(d)
        d=fixture();d["policy"]["approx_weight"]=.6
        with self.assertRaisesRegex(ValueError,"policy"):score_document(d)
    def test_native_ordering_consistent_with_equal_cache(self):
        d=fixture();r=score_document(d)
        self.assertLess(r["nll"]["ssm"]["same_cache"],r["nll"]["tac"]["same_cache"])
        self.assertAlmostEqual(r["paired_differences_nats_per_char"]["tac_minus_ssm_native_nll"],-__import__("math").log(.5)+__import__("math").log(.7))

if __name__=="__main__": unittest.main()
