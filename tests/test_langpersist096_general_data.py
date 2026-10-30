"""Boundary/selection fixtures for public development tasks; no model results."""
from copy import deepcopy
import gzip
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

PATH = Path(__file__).resolve().parents[1] / "research/mathematical-engineering/langpersist096/general_data.py"
SPEC = importlib.util.spec_from_file_location("langpersist096_general_data", PATH)
data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data)


def arc_archive():
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for i in range(9):
            record = {"train": [{"input": [[1]], "output": [[2]]}] * 2,
                      "test": [{"input": [[3]], "output": [[9]]}]}
            archive.writestr(f"archive/data/training/{i:08x}.json", json.dumps(record))
        archive.writestr("archive/data/evaluation/forbidden.json", b"this is not parsed")
    return output.getvalue()


def math_lines():
    return b"\n".join(json.dumps({"question": f"What is {i} plus one?", "answer": f"reasoning #### {i+1}"}).encode()
                       for i in range(9))


def code_lines():
    return gzip.compress(b"\n".join(json.dumps({"task_id": f"HumanEval/{i}",
        "prompt": f"def solution_{i}():\n    '''Return an integer.'''\n",
        "canonical_solution": "    return 123456789\n", "test": "def check(candidate):\n    assert candidate()==123456789\n",
        "entry_point": f"solution_{i}"}).encode() for i in range(9)))


class GeneralDataTests(unittest.TestCase):
    def cases(self):
        return [data.build_arc_cases(arc_archive()), data.build_gsm8k_cases(math_lines()),
                data.build_humaneval_cases(code_lines())]

    def test_fixed_selection_is_deterministic_and_metric_free(self):
        first, second = self.cases(), self.cases()
        self.assertEqual(first, second)
        for cases, eligible in first:
            self.assertEqual(eligible, 9)
            self.assertEqual(len(cases), 8)
            ranks = [data.rank_task(actor["domain"], actor["task_id"]) for actor, _ in cases]
            self.assertEqual(ranks, sorted(ranks))

    def test_arc_test_output_is_not_actor_input(self):
        cases, _ = data.build_arc_cases(arc_archive())
        for actor, judge in cases:
            observations = bytes(actor["acquisition_bytes"] + actor["query_bytes"]).decode()
            self.assertNotIn("9", observations)
            self.assertEqual(judge["expected"], [[9]])
            self.assertIn("2", bytes(actor["acquisition_bytes"]).decode())
            self.assertIn("3", bytes(actor["query_bytes"]).decode())

    def test_code_reference_and_tests_never_enter_actor_records(self):
        cases, _ = data.build_humaneval_cases(code_lines())
        for actor, judge in cases:
            serialized = json.dumps(actor)
            self.assertEqual(set(actor), {"domain", "task_id", "acquisition_bytes", "query_bytes", "generation_limit"})
            self.assertNotIn("canonical_solution", serialized)
            self.assertNotIn("123456789", bytes(actor["query_bytes"]).decode())
            self.assertIn("123456789", judge["canonical_solution"])
            self.assertIn("test", judge)

    def test_math_gold_comes_only_from_final_separator(self):
        cases, _ = data.build_gsm8k_cases(math_lines())
        for actor, judge in cases:
            self.assertNotIn("reasoning", bytes(actor["query_bytes"]).decode())
            self.assertNotIn("answer", actor)
            self.assertNotIn("####", bytes(actor["query_bytes"]).decode())
            self.assertTrue(judge["answer"].isdigit())

    def test_complete_actor_schema_rejects_labels_or_changed_budgets(self):
        actors = [actor for cases, _ in self.cases() for actor, _ in cases]
        self.assertEqual(data.validate_actor_records(actors), {d: 8 for d in data.DOMAINS})
        for change in (lambda r: r[0].update(answer="secret"), lambda r: r[0].update(generation_limit=129),
                       lambda r: r[0].update(query_bytes=[True]), lambda r: r.pop()):
            invalid = deepcopy(actors)
            change(invalid)
            with self.assertRaises(ValueError):
                data.validate_actor_records(invalid)

    def test_grid_limits_and_small_domain_fail_explicitly(self):
        for grid in ([[True]], [[10]], [[1], [1, 2]], [[0] * 7], []):
            with self.assertRaises(ValueError):
                data.grid_text(grid)
        with self.assertRaises(ValueError):
            data.build_gsm8k_cases(b'{"question":"one?","answer":"#### 1"}')

    def test_asset_hash_failure_and_existing_artifact_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            target = cache / "source" / "arc.zip"
            target.parent.mkdir()
            target.write_bytes(b"corrupted")
            with patch.object(data, "urlopen") as fetch:
                with self.assertRaises(ValueError):
                    data._asset(cache, "arc.zip")
                fetch.assert_not_called()
            self.assertEqual(target.read_bytes(), b"corrupted")
            path = cache / "fixed.json"
            data._immutable(path, b"first")
            data._immutable(path, b"first")
            with self.assertRaises(ValueError):
                data._immutable(path, b"second")
            self.assertEqual(path.read_bytes(), b"first")


if __name__ == "__main__":
    unittest.main()
