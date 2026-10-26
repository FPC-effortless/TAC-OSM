#!/usr/bin/env python3
"""Read only Woyengi PredictiveReadExport v1; retain explicit UNATTESTED status.

This adapter validates a serialized *claimed* Woyengi export. Only the trusted
Woyengi host can actually authenticate source principal/rights/READ/EXPORT.
No local receipt can establish independent source or reward attestation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from validate_ecosystem_boundary import (
    BoundaryError, _obj, _keys, _text, _sha, _when, _safe_features,
)

EXPORT_KEYS = {
    "kind", "schemaVersion", "status", "subject", "predicate", "validAt",
    "recordedAt", "predictionCutoff", "selectedClaimId",
    "conflictingClaimIds", "observations", "qualification",
    "publisherAuthenticity", "independentRewardVerification", "actionAuthority",
}
OBS_KEYS = {
    "claimId", "validFrom", "knownFrom", "sourceAvailableAt",
    "featureHash", "features",
}


def adapt_predictive_read_export(payload: object) -> dict:
    d = _obj(payload, "Woyengi export")
    _keys(d, EXPORT_KEYS, "Woyengi export")
    if (d.get("kind") != "PREDICTIVE_READ_EXPORT"
            or type(d.get("schemaVersion")) is not int or d["schemaVersion"] != 1
            or d.get("status") != "ADVISORY_ONLY"
            or d.get("qualification") != "STRUCTURAL_EXPORT_ONLY"
            or d.get("publisherAuthenticity") != "NOT_ATTESTED_BY_THIS_EXPORTER"
            or d.get("independentRewardVerification") != "NOT_PERFORMED"
            or d.get("actionAuthority") != "NOT_GRANTED"):
        raise BoundaryError("Woyengi export does not meet non-authoritative v1 contract")
    for name in ("subject", "predicate", "selectedClaimId"):
        _text(d[name], name)
    cutoff = _when(d["predictionCutoff"], "predictionCutoff")
    valid = _when(d["validAt"], "validAt")
    known = _when(d["recordedAt"], "recordedAt")
    if valid > cutoff or known > cutoff:
        raise BoundaryError("future source projection beyond prediction cutoff")
    if not isinstance(d["observations"], list) or not d["observations"]:
        raise BoundaryError("missing source observations")
    if not isinstance(d["conflictingClaimIds"], list):
        raise BoundaryError("missing source conflicts")
    ids = set()
    records = []
    for raw in d["observations"]:
        o = _obj(raw, "Woyengi source observation")
        _keys(o, OBS_KEYS, "Woyengi observation")
        cid = _text(o["claimId"], "claimId")
        if cid in ids:
            raise BoundaryError("duplicate Woyengi claim ID")
        ids.add(cid)
        if _when(o["validFrom"], "validFrom") > valid:
            raise BoundaryError("claim not valid at requested historical time")
        if _when(o["knownFrom"], "knownFrom") > known:
            raise BoundaryError("claim not yet recorded at historical as-known-at cutoff")
        if _when(o["sourceAvailableAt"], "sourceAvailableAt") > cutoff:
            raise BoundaryError("source available only after inference cutoff")
        _sha(o["featureHash"], "featureHash (declared derived hash only)")
        feats = _obj(o["features"], "allowlisted features")
        if not feats:
            raise BoundaryError("empty model features")
        _safe_features(feats, "features")
        records.append({
            "id": "observation:" + cid,
            "entity_id": d["subject"],
            "valid_at": o["validFrom"],
            "recorded_at": o["knownFrom"],
            "available_at": o["sourceAvailableAt"],
            "source_record_id": cid,
            # This is a derived host-declared feature SHA, NOT original publisher hash.
            "record_sha256": o["featureHash"],
            "conflicts_count": len(d["conflictingClaimIds"]),
            "conflicts_preserved": True,
            "features": feats,
        })
    selected = d["selectedClaimId"]
    conflicts = d["conflictingClaimIds"]
    if selected not in ids or selected in conflicts or len(set(conflicts)) != len(conflicts):
        raise BoundaryError("invalid selected or conflicting claim identity")
    if any(not isinstance(cid, str) or cid not in ids for cid in conflicts):
        raise BoundaryError("conflicting claim not present in candidate list")
    # Unlike the native Woyengi ClaimLedger, this restricted export intentionally
    # lacks raw claim.object, so *cannot* independently recompute all conflicts.
    # Require the complete signed/trusted source projection for that assurance.
    return {
        "adapter": "woyengi-predictive-read-export-v1",
        "status": "STRUCTURAL_ONLY",
        "cutoff": d["predictionCutoff"],
        "observation_count": len(records),
        "selected_claim_id": selected,
        "conflicting_claim_ids": sorted(conflicts),
        "observations": records,
        "authorization_provenance": "HOST_ASSERTED_NOT_INDEPENDENTLY_AUTHENTICATED",
        "source_rights": "UNVERIFIED_BY_THIS_CONSUMER",
        "derived_feature_hashes": "FORMAT_VALID_ONLY_NOT_RECOMPUTED",
        "source_original_hashes": "NOT_PROVIDED",
        "veritas_reward_qualification": "NOT_EVALUATED",
        "model_scientific_qualification": "NOT_EVALUATED",
        "woyengi_action_authority": "NOT_GRANTED",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("woyengi_export_json", type=Path)
    args = parser.parse_args(argv)
    try:
        def reject_nonfinite(x):
            raise BoundaryError("nonfinite JSON constant")
        blob = json.loads(args.woyengi_export_json.read_text(encoding="utf-8"),
                          parse_constant=reject_nonfinite)
        print(json.dumps(adapt_predictive_read_export(blob), sort_keys=True))
        return 0
    except (BoundaryError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "REJECTED", "reason": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
