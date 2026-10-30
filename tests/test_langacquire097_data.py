"""Constructed data-boundary checks; fixtures supply no capability evidence."""
import copy
import hashlib
import importlib.util
import io
import json
import re
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "research/mathematical-engineering/langacquire097"
spec = importlib.util.spec_from_file_location("langacquire097_data_test", HERE / "data.py")
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


def document(name):
    return b"".join((b"filler_value = 0\n" * 8 + name + b" = 1\n" + b"\n" * 45
                     + name + b" = 2\n" + b"\n" * 100) for _ in range(12))


def row(path, partition, raw):
    return {"path": path, "partition": partition, "repo": "fixture", "commit": "fixture",
            "sha256": data.digest(raw)}


def archive(records):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as result:
        for identifier, record in records.items():
            result.writestr("fixture/data/training/" + identifier + ".json", json.dumps(record))
        result.writestr("fixture/data/evaluation/protected.json", "not readable JSON")
    return stream.getvalue()


def arc_record():
    return {"train": [{"input": [[0, 1]], "output": [[1, 0]]},
                      {"input": [[2, 3]], "output": [[3, 2]]}],
            "test": [{"input": [[4, 5]], "output": [[5, 4]]}]}


class DataContractTests(unittest.TestCase):
    def test_source_contract_partitions_are_fixed_and_exclude_prior_arc(self):
        value = data.contract()
        rows = value["cpython_files"]
        self.assertEqual(sum(r["partition"] == "FIT" for r in rows), 10)
        self.assertEqual(sum(r["partition"] == "DEV" for r in rows), 4)
        self.assertEqual(len({r["sha256"] for r in rows}), 14)
        self.assertEqual({r["repo"] for r in rows}, {"python/cpython"})
        self.assertEqual(len(value["arc_policy"]["exclude_langpersist096_tasks"]), 8)
        self.assertFalse(value["arc_policy"]["primary_evidence"])

    def test_create_once_identity_and_symlink_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.json"
            data.immutable(path, b"original")
            data.immutable(path, b"original")
            with self.assertRaises(ValueError):
                data.immutable(path, b"changed")
            linked = Path(folder) / "linked.json"
            linked.symlink_to(path)
            with self.assertRaises(ValueError):
                data.immutable(linked, b"original")

    def test_original_source_hash_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            original = Path(folder) / "original"
            original.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                data._read_source(Path(folder), "copy", {"sha256": data.digest(b"authentic")}, original)
            self.assertFalse((Path(folder) / "sources/copy").exists())

    def test_complete_identifier_reuse_and_fit_inventory_exclusion(self):
        raw = document(b"dev_memory_token")
        candidates = data.lexical_candidates(raw)
        self.assertTrue(candidates["positive"])
        for offset, name, earlier in candidates["positive"]:
            self.assertEqual(raw[offset:offset + len(name)], name)
            self.assertEqual(raw[earlier:earlier + len(name)], name)
            self.assertGreaterEqual(earlier, offset - 96)
            self.assertLessEqual(earlier + len(name), offset - 32)
            self.assertNotIn(name, raw[offset - 32:offset])
        excluded = data.lexical_candidates(raw, {b"dev_memory_token"})
        self.assertTrue(all(name != b"dev_memory_token" for group in excluded.values()
                            for _, name, _ in group))

    def test_lexical_cohort_deterministic_disjoint_and_raw_byte_exact(self):
        documents = {"fit.py": document(b"fit_memory_token"), "dev.py": document(b"dev_memory_token")}
        rows = [row("fit.py", "FIT", documents["fit.py"]), row("dev.py", "DEV", documents["dev.py"])]
        policy = {"fit_counts": {g: 1 for g in data.GROUPS}, "dev_counts": {g: 1 for g in data.GROUPS}}
        episodes, receipt = data.build_lexical(documents, rows, policy)
        repeated, again = data.build_lexical(documents, rows, policy)
        self.assertEqual((episodes, receipt), (repeated, again))
        self.assertEqual(receipt["status"], "COHORT_READY")
        self.assertEqual(len(episodes), 6)
        occupied = {}
        for episode in episodes:
            metadata = episode["metadata"]
            self.assertRegex(episode["id"], r"^097-[0-9a-f]{64}$")
            self.assertNotIn(metadata["source_path"], episode["id"])
            self.assertEqual(episode["id"], data.opaque_episode_id(metadata["original_episode_id"]))
            raw = documents[metadata["source_path"]]
            start, query, target, end = (metadata[k] for k in
                ("acquisition_start", "query_start", "target_offset", "target_end"))
            self.assertEqual(bytes(episode["acquisition_bytes"]), raw[start:query])
            self.assertEqual(bytes(episode["query_bytes"]), raw[query:target])
            self.assertEqual(bytes(episode["target_bytes"]), raw[target:end])
            for old_start, old_end in occupied.setdefault(metadata["source_path"], []):
                self.assertTrue(end <= old_start or start >= old_end)
            occupied[metadata["source_path"]].append((start, end))

    def test_insufficient_cohort_is_not_silently_reduced(self):
        raw = b"x = 1\n" * 30
        documents = {"fit.py": raw, "dev.py": raw}
        rows = [row("fit.py", "FIT", raw), row("dev.py", "DEV", raw)]
        policy = {"fit_counts": {g: 1 for g in data.GROUPS}, "dev_counts": {g: 1 for g in data.GROUPS}}
        _, receipt = data.build_lexical(documents, rows, policy)
        self.assertEqual(receipt["status"], "INSUFFICIENT_COHORT")
        self.assertTrue(receipt["failures"])

    def test_compact_grid_lossless_and_strict(self):
        grid = [[i % 10 for i in range(6)] for _ in range(6)]
        self.assertEqual(data.decode_grid(data.encode_grid(grid)), grid)
        self.assertEqual(data.decode_arc_output(data.encode_grid(grid) + b";ignored"), grid)
        for invalid in ([[True]], [[0], [0, 1]], [], [[10]], [[0] * 7]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                data.encode_grid(invalid)
        for invalid in (b"1100", b"11x", b"110", b"11"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                data.decode_arc_output(invalid)

    def test_arc_only_training_members_prior_tasks_excluded_and_task_split(self):
        policy = copy.deepcopy(data.contract()["arc_policy"])
        policy["fit_tasks"], policy["dev_tasks"] = 1, 1
        prior = policy["exclude_langpersist096_tasks"][0]
        records = {prior: arc_record(), "fixture1": arc_record(), "fixture2": arc_record()}
        episodes, receipt = data.build_arc(archive(records), policy)
        self.assertEqual(receipt["status"], "COHORT_READY")
        self.assertEqual(receipt["excluded"]["prior096task"], 1)
        self.assertEqual(len(episodes), 2)
        self.assertFalse(set(receipt["task_partitions"]["FIT"]) & set(receipt["task_partitions"]["DEV"]))
        for episode in episodes:
            self.assertNotIn(prior, episode["id"])
            self.assertRegex(episode["id"], r"^097-[0-9a-f]{64}$")
            self.assertNotIn(episode["metadata"]["cluster_id"], episode["id"])
            self.assertEqual(data.decode_arc_output(episode["target_bytes"]), [[5, 4]])
        insufficient = {**policy, "fit_tasks": 2}
        empty, failure = data.build_arc(archive(records), insufficient)
        self.assertEqual(empty, [])
        self.assertEqual(failure["status"], "INSUFFICIENT_COHORT")

    def test_binding_shuffle_uses_past_only_preserves_clock_and_tracks_null(self):
        acquisition = b"R;110>1212;1223>113;"
        changed, available = data.shuffled_demonstration_outputs(acquisition)
        self.assertTrue(available)
        self.assertEqual(len(changed), len(acquisition))
        self.assertEqual(changed, b"R;110>113;1223>1212;")
        repeated = b"R;110>111;112>111;"
        self.assertEqual(data.shuffled_demonstration_outputs(repeated), (repeated, False))
        # No query, target or source/stratum fields exist in this actor input.
        controls = data.control_actor_records([{"id": "fixture", "domain": "arc", "acquisition_bytes": list(acquisition)}])
        self.assertEqual(set(controls[0]), {"id", "acquisition_bytes", "available", "reason"})

    def test_donors_depend_only_on_observed_clock_domain_and_owner(self):
        actor = [{"id": str(i), "domain": "arc", "acquisition_bytes": [0] * length}
                 for i, length in enumerate([4, 4, 5])]
        judge = {str(i): {"source_owner": str(i), "target_bytes": [i], "group": "first"} for i in range(3)}
        before = data.donor_records(actor, judge)
        for record in judge.values():
            record.update(target_bytes=[255], group="changed")
        self.assertEqual(before, data.donor_records(actor, judge))
        self.assertEqual(before[0]["donor_id"], "1")
        self.assertEqual(before[1]["donor_id"], "0")
        self.assertFalse(before[2]["available"])
        self.assertIsNone(before[2]["donor_id"])

    def test_fixed_future_source_poison_leaves_actor_inputs_and_originals_unchanged(self):
        raw = document(b"dev_memory_token")
        offset, _, _ = data.lexical_candidates(raw)["positive"][0]
        lexical = {"id": "lexical-fixture", "domain": "lexical",
                   "acquisition_bytes": list(raw[offset - 96:offset - 32]),
                   "query_bytes": list(raw[offset - 32:offset]), "generation_limit": 4}
        arc_raw = archive({"fixture-task": arc_record()})
        policy = {**data.contract()["arc_policy"], "fit_tasks": 0, "dev_tasks": 1}
        cases, _ = data.build_arc(arc_raw, policy)
        arc = {k: cases[0][k] for k in ("id", "domain", "acquisition_bytes", "query_bytes", "generation_limit")}
        actor = [lexical, arc]
        judge = {"lexical-fixture": {"source_path": "fixture.py", "source_file_sha256": data.digest(raw),
                 "acquisition_start": offset - 96, "query_start": offset - 32,
                 "target_offset": offset, "target_end": offset + 4}, arc["id"]: cases[0]["metadata"]}
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            data.immutable(cache / "sources/cpython/fixture.py", raw)
            data.immutable(cache / "sources/arc.zip", arc_raw)
            data.immutable(cache / "DEV_ACTOR.json", data.json_bytes(actor))
            data.immutable(cache / "DEV_JUDGE.json", data.json_bytes(judge))
            with mock.patch.object(data, "contract", return_value={"arc_assets": {"arc.zip": {"sha256": data.digest(arc_raw)}}}):
                alternate, audit = data.future_poison_audit(cache)
            self.assertEqual(alternate, actor)
            self.assertEqual(audit["status"], "PASS_FIXED_SOURCE_OBSERVATION_NONINTERFERENCE")
            self.assertFalse(audit["eligibility_recomputed"])
            self.assertTrue(all(audit["original_hashes"][key] != value
                                for key, value in audit["poisoned_hashes"].items()))
            self.assertEqual((cache / "sources/cpython/fixture.py").read_bytes(), raw)
            self.assertEqual((cache / "sources/arc.zip").read_bytes(), arc_raw)


if __name__ == "__main__":
    unittest.main()
