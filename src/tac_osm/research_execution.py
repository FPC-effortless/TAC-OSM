"""Execution levels: ordinary internal FIT/DEV work versus qualification.

Metadata validation is not proof a dataset or metric is honest. Runners still
verify original bytes, causal inputs, frozen splits and checkpoint replay.
"""
from __future__ import annotations

import re

INTERNAL = "internal_research"
INDEPENDENT = "independent_qualification"
LEVELS = (INTERNAL, INDEPENDENT)


def execution_level(manifest):
    level = manifest.get("execution_level", INDEPENDENT)
    if level not in LEVELS:
        raise ValueError("unknown research execution level")
    return level


def validate_internal_research(manifest):
    if execution_level(manifest) != INTERNAL:
        raise ValueError("explicit internal_research preregistration required")
    scope = manifest.get("research_scope", {})
    if (scope.get("publicly_licensed") is not True or not scope.get("partitions")
            or not isinstance(scope["partitions"], list)
            or any(p not in ("FIT", "DEV") for p in scope["partitions"])):
        raise ValueError("internal research requires publicly licensed authorized FIT/DEV scope")
    for flag in ("protected_evaluation", "reward_integrity_certification",
                 "safety_claim", "checkpoint_promotion"):
        if scope.get(flag) is not False:
            raise ValueError(flag + " requires independent qualification")
    source = manifest.get("real_world_source", {})
    if not re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256", ""))):
        raise ValueError("original source hash required")
    for key in ("publisher_url", "license", "license_evidence_url", "source_receipt"):
        if not isinstance(source.get(key), str) or not source[key].strip():
            raise ValueError("recorded publisher/license/source receipt required: " + key)
    split = manifest.get("split_policy", {})
    if (split.get("frozen") is not True or
            not re.fullmatch(r"[0-9a-f]{64}", str(split.get("cohort_sha256", "")))):
        raise ValueError("frozen split and original cohort hash required")
    for key in ("causal_protocol", "checkpoint_policy"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError("reproducible causal/checkpoint protocol required")
    return {"execution_level": INTERNAL, "status": "PASS_INTERNAL_RESEARCH_METADATA",
            "evidence_level": "internally_measured_not_independently_qualified",
            "independently_qualified": False, "protected_TEST_permitted": False,
            "checkpoint_promotion_permitted": False}
