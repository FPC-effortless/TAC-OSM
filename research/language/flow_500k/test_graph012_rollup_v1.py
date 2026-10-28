"""Constructed numerical diagnostics: never call these real-world results."""
import unittest
from test_equal_cache_score_v1 import fixture, dist
from graph012_rollup_v1 import aggregate


def five(*,sources=8):
    docs=[]
    for j in range(5):
        doc=fixture();doc["seed"]=1701+j
        doc["models"]["tac"]["checkpoint_sha256"]=(f"{j+1:064x}")
        doc["models"]["ssm"]["checkpoint_sha256"]=(f"{j+11:064x}")
        doc["rows"]=[]
        for i in range(sources):
            source=f"{i+100:064x}"
            doc["rows"].append({"owner_id":f"doc-{i}","source_sha256":source,
                "ordinal":10,"target":1,"exact_hit":True,"approx_hit":False,
                "exact_posterior":dist(1,.9),
                "native":{"tac":dist(1,.5 + .005*j),"ssm":dist(1,.7)}})
        docs.append(doc)
    return docs


class RollupContracts(unittest.TestCase):
    def test_five_seed_source_cluster_bootstrap_and_unqualified_status(self):
        r=aggregate(five(),resamples=120)
        self.assertEqual(r["distinct_seed_count"],5)
        self.assertEqual(r["distinct_source_count"],8)
        self.assertEqual(r["evidence_status"],"SCORING_ONLY_NOT_ARCHITECTURE_QUALIFIED")
        self.assertEqual(len(r["source_clustered_two_level_bootstrap_95pct_interval"]),2)
        self.assertGreater(r["pooled_tac_minus_ssm_equal_cache_nll"],0)
        self.assertEqual(r["source_clustered_two_level_bootstrap_95pct_interval"],
                         aggregate(five(),resamples=120)["source_clustered_two_level_bootstrap_95pct_interval"])
    def test_missing_or_duplicate_seeds_and_different_labels_fail(self):
        with self.assertRaisesRegex(ValueError,"exactly five"):aggregate(five()[:4])
        docs=five();docs[3]["seed"]=docs[0]["seed"]
        with self.assertRaisesRegex(ValueError,"distinct integer seed"):aggregate(docs)
        docs=five();docs[2]["rows"][0]["target"]=2
        with self.assertRaisesRegex(ValueError,"source/target/owner/ordinal"):aggregate(docs)
    def test_split_manifest_and_sample_order_fail(self):
        docs=five();docs[1]["data"]["source_manifest_sha256"]="b"*64
        with self.assertRaisesRegex(ValueError,"split/manifest"):aggregate(docs)
        docs=five();docs[1]["models"]["ssm"]["fit_sample_sha256"]="b"*64
        with self.assertRaisesRegex(ValueError,"fit sample schedules"):aggregate(docs)
    def test_fewer_sources_interval_suppressed(self):
        r=aggregate(five(sources=3),resamples=100)
        self.assertIsNone(r["source_clustered_two_level_bootstrap_95pct_interval"])
        self.assertEqual(r["bootstrap_resamples"],0)


if __name__=="__main__":unittest.main()
