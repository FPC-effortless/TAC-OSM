"""SENSE-TOK-001 engineering-only falsifiers: not capability evaluation."""
import random
import unittest
from hashlib import sha256

from tac_osm.sensory_tokenizers import VARIANTS, Patch, Tokenization, Relation, tokenize


class TokenizerMechanismTests(unittest.TestCase):
    @staticmethod
    def encode(data, name, q=b"what happened to motor 123"):
        return tokenize(data, variant=name, query=q if name in ("relevance", "hybrid") else None)

    def test_all_variants_lossless_on_adversarial_input(self):
        samples = [b"", b"x", b"\x00" * 1024, bytes(range(256)),
                   "你好 👋 وَالسَّلَامُ".encode(), b"abc123 abc123\n" * 60,
                   b"\xff\xfe\x00\xc0\xaf", b"\r\n\r\n\t\\u0000"]
        rng = random.Random(20261031)
        for n in (0, 1, 2, 3, 4, 7, 8, 9, 24, 32, 65, 255, 1024):
            samples.append(bytes(rng.randrange(256) for _ in range(n)))
        for name in VARIANTS:
            for raw in samples:
                with self.subTest(name=name, n=len(raw)):
                    item = self.encode(raw, name)
                    self.assertEqual(item.reconstruct(), raw)
                    self.assertEqual(item.source_sha256, sha256(raw).hexdigest())
                    self.assertEqual(sum(len(p.payload) for p in item.patches), len(raw))
                    self.assertGreaterEqual(item.scan_work, len(raw))

    def test_deterministic_and_no_shared_state_between_entities(self):
        a = b"engine temperature: 42; brake: 99\n"
        b = b"engine temperature: 18; brake: 12\n"
        for name in VARIANTS:
            first = self.encode(a, name)
            self.encode(b, name)
            self.assertEqual(self.encode(a, name), first)

    def test_offline_query_required_only_for_query_arms(self):
        for name in VARIANTS:
            if name in ("relevance", "hybrid"):
                with self.assertRaises(ValueError):
                    tokenize(b"abc", variant=name)
            else:
                with self.assertRaises(ValueError):
                    tokenize(b"abc", variant=name, query=b"answer")

    def test_strict_types_and_limits(self):
        for bad in ("abc", bytearray(b"abc"), memoryview(b"abc")):
            with self.assertRaises(TypeError):
                tokenize(bad, variant="fixed")
        with self.assertRaises(TypeError):
            tokenize(b"abc", variant="hybrid", query="abc")
        with self.assertRaises(ValueError):
            tokenize(b"abc", variant="fixed", min_patch=0)
        with self.assertRaises(ValueError):
            tokenize(b"abc", variant="fixed", max_patch=1, min_patch=2)
        with self.assertRaises(ValueError):
            tokenize(b"abc", variant="unknown")

    def test_query_can_change_resolution_not_contents(self):
        data = b"motor 123 noise. temperature 456 noise. motor 123 again."
        results = [tokenize(data, variant="relevance", query=q) for q in (b"motor", b"temperature")]
        self.assertNotEqual(results[0].patches, results[1].patches)
        self.assertEqual([r.reconstruct() for r in results], [data, data])
        self.assertNotEqual(results[0].query_sha256, results[1].query_sha256)

    def test_structural_graph_has_no_oracle_edges(self):
        data = b"motor motor motor | error 1 error 2"
        for name in ("structure", "hybrid"):
            graph = self.encode(data, name)
            self.assertTrue(graph.edges)
            self.assertTrue(all(0 <= e.src < e.dst < len(graph.patches) for e in graph.edges))
            self.assertLessEqual(len(graph.edges), 3 * len(graph.patches))
            self.assertFalse(any(e.kind in ("gold", "label", "true_edge") for e in graph.edges))

    def test_hierarchy_leaves_and_roots_each_lossless(self):
        data = b"test=17, other=32!" * 13
        for name in ("hierarchy", "hybrid"):
            t = self.encode(data, name)
            self.assertEqual(b"".join(p.payload for p in t.leaves), data)
            self.assertEqual(b"".join(p.payload for p in t.patches), data)
            self.assertGreaterEqual(t.total_representation_nodes, len(t.patches))

    def test_causal_stream_prefix_stability_for_base_segmenters_only(self):
        data = b"temperature=42\n" * 9
        for name in ("fixed", "entropy", "sensory"):
            for n in range(1, len(data)):
                before = self.encode(data[:n], name)
                after = self.encode(data[:n + 1], name)
                self.assertEqual([(p.start, p.end) for p in before.patches[:-1]],
                                 [(p.start, p.end) for p in after.patches[:max(0, len(before.patches) - 1)]],
                                 (name, n))

    def test_mutant_missing_payload_rejected(self):
        data = b"abcd"
        t = self.encode(data, "fixed")
        bad = Tokenization(variant="fixed", length=4, source_sha256=t.source_sha256,
                           patches=(Patch(0, 3, b"abc"),))
        with self.assertRaises(ValueError):
            bad.validate()
        bad2 = Tokenization(variant="fixed", length=4, source_sha256=t.source_sha256,
                            patches=(Patch(0, 4, b"abcd"),), edges=(Relation(0, 0, "next"),))
        with self.assertRaises(ValueError):
            bad2.validate()

    def test_query_does_not_change_other_variants(self):
        data = b"aaaaaa bb c d e f 012345"
        for name in ("fixed", "entropy", "sensory", "hierarchy", "structure"):
            self.assertIsNone(self.encode(data, name).query_sha256)


if __name__ == "__main__":
    unittest.main()
