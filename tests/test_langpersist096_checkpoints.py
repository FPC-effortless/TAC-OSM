"""Fixture validation for immutable checkpoint preparation; no model scoring."""
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "research/mathematical-engineering/langpersist096/checkpoints.py"
SPEC = importlib.util.spec_from_file_location("langpersist096_checkpoints", PATH)
cp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cp)


def records():
    rows = []
    seal = {"study": "CODEVALUE003", "status": "SEALED_BEFORE_HELDOUT_SCORING",
            "all45_checkpoints_complete": True, "independently_qualified": False,
            "source_cohort_sha256": cp.COHORT_SHA256, "checkpoints": {}}
    for arm in cp.ARMS:
        for seed in cp.SEEDS:
            count = cp.PARAMETERS[arm]
            measured = {"study": "CODEVALUE003", "arm": arm, "seed": seed,
                        "steps": cp.STEPS, "targets": cp.TARGETS, "trainable_parameters": count,
                        "deployed_parameters": count, "source_cohort_sha256": cp.COHORT_SHA256,
                        "protected_TEST": "NOT_OPENED", "training_flops": cp.TRAINING_FLOPS,
                        "cold_replay_max_abs_difference": 0., "sample_sha256": f"{seed:064x}",
                        "checkpoint_sha256": cp.CHECKPOINT_SHA256[(arm, seed)]}
            rows.append({"name": arm, "seed": seed, "trainable_parameters": count,
                         "deployed_parameters": count, "measured": measured})
            seal["checkpoints"][f"{arm}-{seed}/MODEL-010240.pt"] = measured["checkpoint_sha256"]
    manifest = {"study_id": "CODEVALUE003", "models": rows,
                "real_world_source": {"sha256": cp.COHORT_SHA256},
                "split_policy": {"cohort_sha256": cp.COHORT_SHA256},
                "budget": {"training_updates": cp.STEPS, "targets_per_model": cp.TARGETS,
                           "batch_size": 8, "context_bytes": 128, "training_FLOPs": cp.TRAINING_FLOPS}}
    return manifest, seal, {"study": "CODEVALUE003", "cohort_sha256": cp.COHORT_SHA256}


class Response(io.BytesIO):
    def __init__(self, payload, content_length=None):
        super().__init__(payload)
        self.headers = {} if content_length is None else {"Content-Length": str(content_length)}


class CheckpointTests(unittest.TestCase):
    def test_fixed_matrix_and_original_unmatched_compute_preserved(self):
        rows = cp.validate_original_records(*records())
        self.assertEqual(len(rows), 9)
        self.assertEqual({(r["arm"], r["seed"]) for r in rows}, set(cp.CHECKPOINT_SHA256))
        self.assertTrue(all(r["steps"] == 10240 and r["targets"] == 10485760 for r in rows))

    def test_mismatched_schedule_seal_cohort_or_parameter_count_fails(self):
        mutations = (
            lambda m, s, r: m["models"][3]["measured"].update(sample_sha256="f" * 64),
            lambda m, s, r: s["checkpoints"].update({"tac-1701/MODEL-010240.pt": "f" * 64}),
            lambda m, s, r: r.update(cohort_sha256="f" * 64),
            lambda m, s, r: m["models"][0].update(trainable_parameters=510000),
            lambda m, s, r: m["budget"].update(training_FLOPs="MATCHED"),
            lambda m, s, r: m["models"][0]["measured"].update(cold_replay_max_abs_difference=float("nan")),
            lambda m, s, r: m["models"].pop(),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                args = records()
                mutate(*args)
                with self.assertRaises(ValueError):
                    cp.validate_original_records(*args)

    def test_download_reuse_never_networks_or_overwrites(self):
        payload = b"fixture checkpoint bytes"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "MODEL.pt"
            with patch.object(cp, "urlopen", return_value=Response(payload)) as fetch:
                self.assertEqual(cp.ensure_verified_download("safe/model.pt", target, digest), len(payload))
                self.assertIn(cp.SOURCE_COMMIT, fetch.call_args.args[0])
            before = target.stat().st_mtime_ns
            with patch.object(cp, "urlopen", side_effect=AssertionError("must reuse verified bytes")):
                cp.ensure_verified_download("safe/model.pt", target, digest)
            self.assertEqual(target.stat().st_mtime_ns, before)
            target.write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                cp.ensure_verified_download("safe/model.pt", target, digest)
            self.assertEqual(target.read_bytes(), b"corrupted")

    def test_wrong_hash_and_oversized_stream_leave_no_checkpoint(self):
        for payload, header, limit, digest in (
                (b"bad", None, 100, "f" * 64),
                (b"abcde", None, 4, hashlib.sha256(b"abcde").hexdigest()),
                (b"x", 500, 4, hashlib.sha256(b"x").hexdigest())):
            with self.subTest(header=header, limit=limit), tempfile.TemporaryDirectory() as temporary:
                target = Path(temporary) / "MODEL.pt"
                with patch.object(cp, "urlopen", return_value=Response(payload, header)):
                    with self.assertRaises(ValueError):
                        cp.ensure_verified_download("safe/model.pt", target, digest, byte_limit=limit)
                self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_symlink_and_path_traversal_are_refused(self):
        with self.assertRaises(ValueError):
            cp._url("../private")
        with tempfile.TemporaryDirectory() as temporary:
            original = Path(temporary) / "original"
            original.write_bytes(b"x")
            target = Path(temporary) / "linked"
            target.symlink_to(original)
            with self.assertRaises(ValueError):
                cp.ensure_verified_download("safe/model.pt", target, hashlib.sha256(b"x").hexdigest())

    def test_manifest_is_immutable_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "CHECKPOINTS.json"
            cp._immutable_json(path, {"verified_checkpoint_count": 9})
            raw = path.read_bytes()
            cp._immutable_json(path, {"verified_checkpoint_count": 9})
            with self.assertRaises(ValueError):
                cp._immutable_json(path, {"verified_checkpoint_count": 8})
            self.assertEqual(path.read_bytes(), raw)

    def test_pinned_local_source_mismatch_fails_without_fetch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.py"
            path.write_bytes(b"modified source")
            with patch.object(cp, "ROOT", Path(temporary)), patch.object(cp, "urlopen") as fetch:
                with self.assertRaises(ValueError):
                    cp._source_bytes("source.py", "f" * 64)
                fetch.assert_not_called()

    def test_loaded_metadata_requires_complete_original_state(self):
        row = cp.validate_original_records(*records())[0]
        payload = {"study": "CODEVALUE003", "arm": row["arm"], "seed": row["seed"],
                   "steps": cp.STEPS, "next_update": cp.STEPS, "params": row["trainable_parameters"],
                   "sample_sha256": row["sample_sha256"], "source_cohort_sha256": cp.COHORT_SHA256,
                   "model": {"weights": [1.]}, "optimizer": {"state": {0: {"step": cp.STEPS}}},
                   "torch_rng": [1], "cuda_rng": [[1]], "python_rng": [1], "numpy_rng": [1],
                   "sampler_state": {"state": 7}, "loss_history": [1.] * cp.STEPS,
                   "register_sha256": "a" * 64}
        self.assertEqual(cp.verify_checkpoint_metadata(payload, row)["optimizer_updates_performed"], 0)
        for field in ("optimizer", "torch_rng", "sampler_state"):
            invalid = deepcopy(payload)
            del invalid[field]
            with self.assertRaises(ValueError):
                cp.verify_checkpoint_metadata(invalid, row)
        invalid = deepcopy(payload)
        invalid["loss_history"][-1] = float("nan")
        with self.assertRaises(ValueError):
            cp.verify_checkpoint_metadata(invalid, row)


if __name__ == "__main__":
    unittest.main()
