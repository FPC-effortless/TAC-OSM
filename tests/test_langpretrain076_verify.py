"""Constructed-only negative-control tests for Kaggle receipts; NEVER GPU data."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.langpretrain076_verify_t4 import verify,ReceiptError,SHA,EXPECTED_RUN_ID,WORKLOAD_PATH


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        runtime={
          "schema":"tac-osm.kaggle-t4x2-runtime.v1",
          "requested_accelerator":"NvidiaTeslaT4",
          "run_id": EXPECTED_RUN_ID,
          "workload_sha256": hashlib.sha256(WORKLOAD_PATH.read_bytes()).hexdigest(),
          "status":"COMPLETED_UNQUALIFIED_REQUIRES_INDEPENDENT_EVALUATION",
          "seeds":[1701,1702],"strategy":"seed_shard",
          "observed":{"count":2,"accelerator":"NvidiaTeslaT4","names":["Tesla T4","Tesla T4"]},
          "results":[{"worker_index":0,"exit_code":0},{"worker_index":1,"exit_code":0}],
          "elapsed_seconds":180,
        }
        (self.folder/"phone-research-runtime.json").write_text(json.dumps(runtime))
        for slot,seed in enumerate((1701,1702)):
            cp=self.folder/f"seed-{seed}.pt"
            cp.write_bytes(f"CONSTRUCTED_NOT_REAL_GPU_WEIGHTS_{seed}".encode())
            data={
              "schema":"tac-osm.langpretrain076.t4-throughput.v1",
              "study":"LANGPRETRAIN076","seed":seed,"physical_slot":slot,
              "device_name":"Tesla T4","source_sha256":SHA,"trainable_parameters":499623,
              "deployed_parameters":499623,"model_tier":"h48",
              "warmup_updates":8,"measured_updates":32,
              "training_target_bytes":40960,"measured_target_bytes":32768,
              "batch":8,"context_bytes":128,
              "checkpoint_cold_reload_logit_parity":"PASS_ATOL1E-6_RTOL1E-6_FIT_ONLY",
              "DEV":"NOT_ACCESSED","protected_TEST":"NOT_ACCESSED",
              "paid_cloud_spend_authorized_usd":0,
              "checkpoint_sha256":hashlib.sha256(cp.read_bytes()).hexdigest(),
              "measured_synced_wall_seconds":4.0,
              "measured_byte_tokens_per_second":8192.0,
              "peak_cuda_tensor_allocated_bytes":1048576,
            }
            (self.folder/f"seed-{seed}.json").write_text(json.dumps(data))

    def test_constructed_receipt_contract_is_not_real_gpu_result(self):
        r=verify(self.folder)
        self.assertEqual(r["status"],"SELF_REPORTED_DUAL_T4_TRAIN_RECEIPTS_CHECKED")
        self.assertEqual(len(r["workers"]),2)
        self.assertEqual(r["aggregate_timed_tokens_per_second"],
                         "NOT_COMPARABLE_UNTIL_ALIGNED_WORKER_CLOCKS")
        self.assertEqual(r["scientific_status"],"INTERNAL_EFFICIENCY_ONLY_NOT_INDEPENDENT")

    def test_missing_real_checkpoint_must_fail(self):
        (self.folder/"seed-1702.pt").unlink()
        with self.assertRaises(ReceiptError):
            verify(self.folder)

    def test_modified_checkpoint_must_fail(self):
        with (self.folder/"seed-1701.pt").open("ab") as file:
            file.write(b"tampered")
        with self.assertRaises(ReceiptError):
            verify(self.folder)

    def test_not_a_completed_saved_kaggle_version_must_fail(self):
        p=self.folder/"phone-research-runtime.json"
        v=json.loads(p.read_text())
        v["status"]="RUNNING_NOT_SCIENTIFICALLY_QUALIFIED"
        p.write_text(json.dumps(v))
        with self.assertRaises(ReceiptError):
            verify(self.folder)

    def test_wrong_raw_byte_rate_fails(self):
        p=self.folder/"seed-1701.json"
        v=json.loads(p.read_text())
        v["measured_byte_tokens_per_second"]=99999
        p.write_text(json.dumps(v))
        with self.assertRaises(ReceiptError):
            verify(self.folder)

    def test_not_actual_t4_pair_fails(self):
        p=self.folder/"phone-research-runtime.json"
        v=json.loads(p.read_text())
        v["observed"]["names"][1]="P100"
        p.write_text(json.dumps(v))
        with self.assertRaises(ReceiptError):
            verify(self.folder)


if __name__=="__main__":
    unittest.main()
