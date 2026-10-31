"""SENSE-TOK-002 no-fit causal stream test matrix."""
import itertools
import random
import unittest
from hashlib import sha256

from tac_osm.causal_patch_stream import CausalPatchStream, PatchCommit, PrefixView, STREAM_VARIANTS, tokenize_stream
from tac_osm.sensory_tokenizers import Patch, _classes, tokenize


def reference_v2(data, minimum=2, maximum=24):
    if not data:
        return ()
    result = []
    start = 0
    for pos in range(1, len(data)):
        if pos - start >= maximum or (
            pos - start >= minimum and _classes(data[pos]) != _classes(data[pos - 1])
        ):
            result.append(data[start:pos])
            start = pos
    result.append(data[start:])
    return tuple(result)


class TestCausalPatchStream(unittest.TestCase):
    def test_original_variants_reproduce_existing_patches(self):
        rng = random.Random(330401)
        samples = [b"", b"abc123", bytes(range(256)), b"ab \xff\x00\x14" * 5]
        samples += [bytes(rng.randrange(256) for _ in range(n))
                    for n in (1, 2, 3, 4, 5, 7, 8, 9, 13, 24, 25, 47, 1024)]
        for raw in samples:
            for v in ("fixed", "entropy", "sensory"):
                with self.subTest(v=v, n=len(raw)):
                    stream = tokenize_stream(raw, variant=v)
                    self.assertEqual(tuple(e.patch for e in stream.committed),
                                     tokenize(raw, variant=v).patches)
                    self.assertEqual(stream.complete_byte_payload, raw)

    def test_v2_matches_independent_direct_reference(self):
        rng = random.Random(40081)
        samples = [b"", b"abc123", bytes(range(256))]
        samples += [bytes(rng.randrange(256) for _ in range(n))
                    for n in (1, 2, 3, 4, 7, 11, 24, 25, 39, 100, 333)]
        for low, high in ((1, 1), (1, 8), (2, 24), (3, 6), (5, 9)):
            for raw in samples:
                with self.subTest(n=len(raw), lo=low, hi=high):
                    result = tokenize_stream(raw, variant="sensory_v2", min_patch=low, max_patch=high)
                    self.assertEqual(tuple(e.patch.payload for e in result.committed),
                                     reference_v2(raw, low, high))
                    self.assertEqual(result.complete_byte_payload, raw)

    def test_t2b_boundary_requires_trigger_byte(self):
        stream = CausalPatchStream(variant="sensory_v2")
        stream.observe(b"abc")
        before = stream.snapshot()
        self.assertEqual(before.pending, b"abc")
        self.assertFalse(before.committed)
        stream.observe(b"1")
        after = stream.snapshot()
        self.assertEqual(after.pending, b"1")
        self.assertEqual((after.committed[0].patch.start, after.committed[0].patch.end), (0, 3))
        self.assertEqual(after.committed[0].available_after, 4)
        self.assertEqual(stream.delayed_commits, 1)

    def test_eos_is_explicit_and_post_eos_blocked(self):
        for v in STREAM_VARIANTS:
            stream = CausalPatchStream(variant=v)
            stream.observe(b"ab")
            self.assertFalse(stream.snapshot().finalized)
            self.assertFalse(any(e.requires_eos for e in stream.snapshot().committed))
            events = stream.finish()
            self.assertTrue(all(e.requires_eos for e in events))
            with self.assertRaises(ValueError):
                stream.observe(b"c")
            with self.assertRaises(ValueError):
                stream.finish()

    def test_chunking_independent(self):
        raw = bytes(random.Random(1044 + i).randrange(256) for i in range(200)) + b"abc123"
        for v in STREAM_VARIANTS:
            gold = tokenize_stream(raw, variant=v)
            for seed in range(10):
                rr = random.Random(seed)
                stream = CausalPatchStream(variant=v)
                offset = 0
                while offset < len(raw):
                    n = rr.randint(1, 21)
                    stream.observe(raw[offset:offset+n])
                    stream.snapshot().validate()
                    offset += n
                stream.finish()
                self.assertEqual(stream.snapshot(), gold)

    def test_exhaustive_short_sequences(self):
        alphabet = (ord("a"), ord("1"), ord(" "), 0xff)
        for n in range(6):
            for values in itertools.product(alphabet, repeat=n):
                raw = bytes(values)
                result = tokenize_stream(raw, variant="sensory_v2", min_patch=2, max_patch=4)
                self.assertEqual(tuple(x.patch.payload for x in result.committed),
                                 reference_v2(raw, 2, 4))

    def test_future_poison_cannot_change_predecision_view(self):
        rng = random.Random(3219)
        for _ in range(40):
            prefix = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 70)))
            a = bytes(rng.randrange(256) for _ in range(28))
            b = bytes(rng.randrange(256) for _ in range(28))
            for v in STREAM_VARIANTS:
                x = CausalPatchStream(variant=v)
                y = CausalPatchStream(variant=v)
                x.observe(prefix)
                y.observe(prefix)
                view = x.snapshot()
                self.assertEqual(view, y.snapshot())
                x.observe(a)
                y.observe(b)
                self.assertEqual(view.complete_byte_payload, prefix)

    def test_committed_patches_never_change(self):
        raw = bytes(random.Random(88+i).randrange(256) for i in range(140))
        for v in STREAM_VARIANTS:
            stream = CausalPatchStream(variant=v)
            previous = ()
            for b in raw:
                stream.observe(bytes([b]))
                current = stream.snapshot().committed
                self.assertEqual(current[:len(previous)], previous)
                self.assertTrue(all(e.available_after <= stream.observed_count for e in current))
                previous = current

    def test_mutants_rejected(self):
        invalid = PrefixView(3, (PatchCommit(Patch(0, 3, b"abc"), 4),),
                             b"", False, sha256(b"abc").hexdigest())
        with self.assertRaises(ValueError):
            invalid.validate()
        invalid2 = PrefixView(3, (PatchCommit(Patch(0, 3, b"abc"), 3, True),),
                              b"", False, sha256(b"abc").hexdigest())
        with self.assertRaises(ValueError):
            invalid2.validate()

    def test_types_bounds_and_empty_inputs(self):
        with self.assertRaises(TypeError):
            CausalPatchStream(variant="fixed").observe("abc")
        for opts in ({"variant":"unknown"}, {"variant":"fixed","min_patch":0},
                     {"variant":"fixed","min_patch":5,"max_patch":4},
                     {"variant":"entropy","entropy_window":0}):
            with self.assertRaises(ValueError):
                CausalPatchStream(**opts)
        for v in STREAM_VARIANTS:
            stream = CausalPatchStream(variant=v)
            self.assertEqual(stream.snapshot().complete_byte_payload, b"")
            self.assertEqual(stream.observe(b""), ())
            self.assertEqual(stream.finish(), ())
