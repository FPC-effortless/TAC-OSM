"""Necessary task controls, not a model or evidence of learned retrieval.

The literal reference reads only observed wire bytes. Labels/IDs/records metadata
are checked afterward. Fixed-key singleton copying cannot identify key selection.
"""
from collections import Counter, defaultdict
from itertools import combinations
import re

RECORD = re.compile(rb"@([a-z]{2}): ([a-h]{4})xxxx\n")
QUERY = re.compile(rb"Copy ([a-z]{2}): \Z")
ABSENT = b"????"


def records_from_prefix(prefix: bytes) -> tuple:
    if not isinstance(prefix, bytes):
        raise ValueError("observed prefix must be bytes")
    if b"@" in RECORD.sub(b"", prefix):
        raise ValueError("malformed observed record")
    return tuple(RECORD.findall(prefix))


def literal_answer(prefix: bytes, query: bytes) -> bytes:
    """Latest value for the observed query key; no target/suffix arguments."""
    if not isinstance(query, bytes) or (match := QUERY.fullmatch(query)) is None:
        raise ValueError("malformed query")
    for key, value in reversed(records_from_prefix(prefix)):
        if key == match.group(1):
            return value
    return ABSENT


def validate_case(case: dict) -> tuple:
    try:
        prefix, query, target, slab = (bytes.fromhex(case[k]) for k in
                                       ("prefix_hex", "query_hex", "target_hex", "slab_hex"))
        position = case["first_position"]
        declared = tuple((k.encode("ascii"), bytes.fromhex(v)) for k, v in case["records"])
    except (KeyError, TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("malformed case encoding") from exc
    answer = literal_answer(prefix, query)  # BEFORE using gold/metadata.
    if (type(position) is not int or position != len(prefix + query) - 1
            or len(slab) != 129 or slab[:position + 1] != prefix + query):
        raise ValueError("prediction-time boundary mismatch")
    if declared != records_from_prefix(prefix):
        raise ValueError("record metadata differs from observed wire")
    if case.get("queried_key") != query[5:7].decode("ascii"):
        raise ValueError("query metadata differs from observed wire")
    if target != answer or slab[position + 1:position + 5] != answer:
        raise ValueError("target differs from observed latest-value semantics")
    if not isinstance(case.get("id"), str) or not case["id"]:
        raise ValueError("case identity required for accounting")
    return prefix, query, answer


def audit_cases(cases: list, training_values=()) -> dict:
    if not cases:
        raise ValueError("empty fixture cohort is not a passed check")
    by_prefix = defaultdict(list)
    targets = Counter()
    first_byte_groups = defaultdict(Counter)
    query_keys = set()
    counts = []
    last_correct = 0
    seen_ids = set()
    present_targets = set()
    absent = 0
    for case in cases:
        prefix, query, answer = validate_case(case)
        if case["id"] in seen_ids:
            raise ValueError("duplicate case identity")
        seen_ids.add(case["id"])
        records = records_from_prefix(prefix)
        by_prefix[prefix].append((query, answer))
        targets[answer] += 1
        if answer != ABSENT:
            first_byte_groups[answer[:1]][answer] += 1
        query_keys.add(query)
        counts.append(len(records))
        last_correct += int((records[-1][1] if records else ABSENT) == answer)
        if answer == ABSENT:
            absent += 1
        else:
            present_targets.add(answer)
    witnesses = sum(qa != qb and ya != yb for group in by_prefix.values()
                    for (qa, ya), (qb, yb) in combinations(group, 2))
    # Exact empirical ceiling for ANY deterministic function of this finite
    # prefix alone. It is not a model score or an unseen-distribution bound.
    blind_ceiling = sum(max(Counter(y for _, y in group).values()) for group in by_prefix.values())
    overlap = present_targets.intersection(training_values)
    return {"cases": len(cases), "reference_consistency": "PASS_WIRE_ONLY_REFERENCE",
            "distinct_query_keys": len(query_keys), "records_per_history_min": min(counts),
            "records_per_history_max": max(counts), "same_prefix_query_selection_witnesses": witnesses,
            "key_blind_empirical_max_correct": blind_ceiling,
            "literal_last_record_correct": last_correct,
            "best_constant_exact_answer_correct": max(targets.values()),
            "present_cases": len(cases) - absent,
            "best_complete_answer_given_gold_first_byte_correct":
                sum(max(group.values()) for group in first_byte_groups.values()),
            "present_value_overlap_with_training": len(overlap), "absence_cases": absent,
            "classification": ("ACQUISITION_ONLY_NO_QUERY_SELECTION_WITNESS" if not witnesses
                               else "NECESSARY_QUERY_SELECTION_WITNESS_ONLY"),
            "binding_or_capacity_claim_authorized": False,
            "limits": "Finite engineered fixture identifiability only; no model success, memory attribution, real-world qualification or training authority"}


def _case(identity, records, key, family, gap=16):
    history = b"".join(b"@" + k + b": " + v + b"xxxx\n" for k, v in records) + b"\n"
    query = b"Copy " + key + b": "
    lead = 113 - len(query) - len(history) - gap
    if lead < 0:
        raise ValueError("fixture too long for declared observation budget")
    prefix = b"." * lead + history + b"." * gap
    target = literal_answer(prefix, query)
    observation = prefix + query
    return {"id": identity, "family": family, "records": [[k.decode(), v.hex()] for k, v in records],
            "queried_key": key.decode(), "first_position": len(observation) - 1, "gap": gap,
            "prefix_hex": prefix.hex(), "query_hex": query.hex(), "target_hex": target.hex(),
            "slab_hex": (observation + target + b"." * (129 - len(observation) - len(target))).hex()}


def prospective_controls(training_values: set) -> list:
    """Exposed unit fixtures for a FUTURE separately registered experiment.

    Never append these to the frozen GRU512 allocation or evaluate a model here.
    """
    # No single value byte identifies a complete four-byte answer. This is
    # still a fixed, exposed engineered codebook, not independent random TEST.
    values = [bytes((97 + (i & 1), 99 + ((i >> 1) & 1), 101 + ((i >> 2) & 1),
                     103 + ((i & 1) ^ ((i >> 1) & 1) ^ ((i >> 2) & 1)))) for i in range(8)]
    if set(values).intersection(training_values):
        raise ValueError("declared prospective value templates overlap training")
    rows = []
    for i, value in enumerate(values):
        other, newer = values[(i + 1) % 8], values[(i + 2) % 8]
        rows.append(_case(f"copy-{i}", [(b"aa", value)], b"aa", "unseen_value_copy", 0))
        for reverse in (False, True):
            records = [(b"aa", value), (b"bb", other)]
            if reverse:
                records.reverse()
            for key in (b"aa", b"bb"):
                rows.append(_case(f"select-{i}-{int(reverse)}-{key.decode()}", records, key, "query_selection"))
        records = [(b"aa", value), (b"bb", other), (b"aa", newer)]
        for key in (b"aa", b"bb"):
            rows.append(_case(f"overwrite-{i}-{key.decode()}", records, key, "latest_write"))
        rows.append(_case(f"absent-{i}", [(b"aa", value), (b"bb", other)], b"cc", "absence"))
    return rows
