"""Constructed exact-store integrity checks; no learned-recall evaluation."""

import copy
import json
import unittest

from research.langcache104.exact_store import BoundedExactStore


SCOPE = {"owner_id": "alice", "entity_id": "stream-1"}


def make_store(capacity=3, **kwargs):
    return BoundedExactStore(**SCOPE, capacity=capacity, **kwargs)


def put(store, key, value, tick):
    store.put(key, value, observed_at=tick, **SCOPE)


def get(store, key, tick):
    return store.get(key, as_of=tick, **SCOPE)


class ExactStoreTests(unittest.TestCase):
    def test_exact_bytes_and_causal_cutoff(self):
        store = make_store()
        put(store, b"\x00key\xff", b"\x00value\xff", 4)
        self.assertIsNone(get(store, b"\x00key\xff", 3))
        self.assertIsNone(get(store, b"\x00key\xfe", 4))
        self.assertIsNone(get(store, b"key", 4))
        self.assertEqual(get(store, b"\x00key\xff", 4), b"\x00value\xff")
        put(store, b"", b"", 4)
        self.assertEqual(get(store, b"", 4), b"")

    def test_versioned_latest_write_and_duplicate_timestamp(self):
        store = make_store(4)
        put(store, b"k", b"old", 2)
        put(store, b"k", b"new", 5)
        put(store, b"k", b"same-time-newer", 5)
        self.assertIsNone(get(store, b"k", 1))
        self.assertEqual(get(store, b"k", 2), b"old")
        self.assertEqual(get(store, b"k", 4), b"old")
        self.assertEqual(get(store, b"k", 5), b"same-time-newer")
        self.assertEqual(get(store, b"k", 100), b"same-time-newer")

    def test_fifo_counts_versions_and_reads_do_not_refresh(self):
        store = make_store(2)
        put(store, b"k", b"old", 1)
        put(store, b"x", b"other", 2)
        self.assertEqual(get(store, b"k", 1), b"old")
        put(store, b"k", b"new", 3)
        self.assertIsNone(get(store, b"k", 1))  # old version was evicted
        self.assertEqual(get(store, b"x", 3), b"other")
        put(store, b"k", b"newer", 3)
        self.assertIsNone(get(store, b"x", 3))
        self.assertEqual(get(store, b"k", 3), b"newer")
        self.assertEqual(store.accounting(**SCOPE)["retained_versions"], 2)

    def test_scope_mismatch_fails_closed(self):
        for wrong_scope in [
            {"owner_id": "bob", "entity_id": "stream-1"},
            {"owner_id": "alice", "entity_id": "stream-2"},
        ]:
            with self.subTest(scope=wrong_scope):
                store = make_store()
                put(store, b"k", b"secret", 1)
                before = store.to_dict(**SCOPE)
                with self.assertRaises(PermissionError):
                    store.get(b"k", as_of=1, **wrong_scope)
                with self.assertRaises(PermissionError):
                    store.put(b"k", b"overwrite", observed_at=2, **wrong_scope)
                with self.assertRaises(PermissionError):
                    store.to_dict(**wrong_scope)
                with self.assertRaises(PermissionError):
                    store.accounting(**wrong_scope)
                with self.assertRaises(PermissionError):
                    BoundedExactStore.from_dict(before, **wrong_scope)
                self.assertEqual(store.to_dict(**SCOPE), before)

    def test_identical_keys_in_distinct_scopes_are_independent(self):
        a = make_store()
        b = BoundedExactStore("bob", "stream-1", 3)
        put(a, b"k", b"a", 1)
        b.put(b"k", b"b", observed_at=1, owner_id="bob", entity_id="stream-1")
        self.assertEqual(get(a, b"k", 1), b"a")
        self.assertEqual(b.get(b"k", as_of=1, owner_id="bob", entity_id="stream-1"), b"b")

    def test_roundtrip_continuation_and_accounting(self):
        store = make_store(2, max_key_bytes=3, max_value_bytes=7)
        for tick in range(5):
            put(store, bytes([tick]), bytes([tick, 0, 255]), tick)
        snapshot = json.loads(json.dumps(store.to_dict(**SCOPE)))
        restored = BoundedExactStore.from_dict(snapshot, **SCOPE)
        self.assertEqual(restored.to_dict(**SCOPE), snapshot)
        for tick in range(6):
            for key in range(5):
                self.assertEqual(get(restored, bytes([key]), tick), get(store, bytes([key]), tick))
        put(store, b"end", b"final", 8)
        put(restored, b"end", b"final", 8)
        self.assertEqual(restored.to_dict(**SCOPE), store.to_dict(**SCOPE))
        account = restored.accounting(**SCOPE)
        self.assertEqual(account["key_payload_bytes"], 4)
        self.assertEqual(account["value_payload_bytes"], 8)
        self.assertEqual(account["scope_payload_bytes"], 13)
        self.assertEqual(account["total_payload_bytes"], 25)
        self.assertEqual(account["maximum_record_payload_bytes"], 20)
        self.assertFalse(account["resident_memory_measured"])
        self.assertFalse(account["resident_overhead_included"])
        encoded = json.dumps(restored.to_dict(**SCOPE), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.assertEqual(account["serialized_utf8_bytes"], len(encoded))

    def test_empty_roundtrip_and_failed_put_does_not_mutate(self):
        store = make_store(max_key_bytes=2, max_value_bytes=3)
        self.assertEqual(BoundedExactStore.from_dict(store.to_dict(**SCOPE), **SCOPE).to_dict(**SCOPE), store.to_dict(**SCOPE))
        put(store, b"ok", b"yes", 4)
        before = store.to_dict(**SCOPE)
        for key, value, tick in [(b"long", b"yes", 5), (b"ok", b"long", 5), (b"ok", b"yes", 3), (b"ok", b"yes", True), (b"ok", b"yes", -1)]:
            with self.subTest(key=key, value=value, tick=tick):
                with self.assertRaises(ValueError):
                    put(store, key, value, tick)
                self.assertEqual(store.to_dict(**SCOPE), before)
        with self.assertRaises(TypeError):
            put(store, bytearray(b"ok"), b"yes", 4)
        with self.assertRaises(ValueError):
            get(store, b"ok", -1)

    def test_snapshot_corruption_rejected(self):
        corruptions = [
            lambda d: d.update(capacity=0),
            lambda d: d.update(capacity=True),
            lambda d: d.update(next_sequence=1),
            lambda d: d.update(clock=20),
            lambda d: d.update(extra="unexpected"),
            lambda d: d["records"][1].update(sequence=d["records"][0]["sequence"]),
            lambda d: d["records"][1].update(observed_at=0),
            lambda d: d["records"][0].update(key_hex="abc"),
            lambda d: d["records"][0].update(value_hex="zz"),
            lambda d: d["records"][0].update(key_hex="FF"),
            lambda d: d["records"][0].update(key_hex="00" * 257),
            lambda d: d["records"][0].update(value_hex="00" * 4097),
            lambda d: d["records"][0].update(observed_at=True),
            lambda d: d["records"].pop(),
        ]
        store = make_store()
        put(store, b"k", b"v", 1)
        put(store, b"x", b"y", 2)
        for index, mutate in enumerate(corruptions):
            with self.subTest(corruption=index):
                snapshot = copy.deepcopy(store.to_dict(**SCOPE))
                mutate(snapshot)
                with self.assertRaises(ValueError):
                    BoundedExactStore.from_dict(snapshot, **SCOPE)

    def test_invalid_configuration_and_empty_clock_rejected(self):
        for kwargs in [
            {"capacity": 0}, {"capacity": True}, {"max_key_bytes": -1},
            {"max_value_bytes": 0}, {"owner_id": ""}, {"entity_id": 1},
        ]:
            with self.subTest(config=kwargs):
                config = {**SCOPE, "capacity": 2, **kwargs}
                with self.assertRaises(ValueError):
                    BoundedExactStore(**config)
        snapshot = make_store().to_dict(**SCOPE)
        snapshot["clock"] = 0
        with self.assertRaises(ValueError):
            BoundedExactStore.from_dict(snapshot, **SCOPE)


if __name__ == "__main__":
    unittest.main()
