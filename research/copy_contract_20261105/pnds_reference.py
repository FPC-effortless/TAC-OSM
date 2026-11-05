"""Independent finite-record specification; no TAC/parser/neural imports."""
from collections import Counter, defaultdict


def lookup(records, key):
    latest = {}
    for observed_key, observed_value in records:
        latest[observed_key] = observed_value
    return latest.get(key)


def last_value(records):
    return records[-1][1] if records else None


def query_blind_ceiling(cases):
    """Cases are (observed_prefix, key, gold_value), including absence None.

    Finite empirical exact-answer ceiling only, not a generalization bound.
    """
    groups = defaultdict(Counter)
    for prefix, _key, gold in cases:
        groups[prefix][gold] += 1
    return sum(max(c.values()) for c in groups.values())


def self_test():
    assert lookup([], b"aa") is None
    assert lookup([(b"aa", b"abcd")], b"aa") == last_value([(b"aa", b"abcd")])
    rows = [(b"aa", b"abcd"), (b"bb", b"efgh"), (b"aa", b"hgfe")]
    assert lookup(rows, b"aa") == b"hgfe" and lookup(rows, b"bb") == b"efgh"
    assert lookup(rows, b"cc") is None
    assert query_blind_ceiling([(b"same", b"aa", b"abcd"), (b"same", b"bb", b"efgh")]) == 1
    assert query_blind_ceiling([(b"one", b"aa", b"abcd"), (b"two", b"aa", b"efgh")]) == 2
    return {"status": "PASS_FINITE_RECORD_REFERENCE", "neural_proof": False}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test()))
