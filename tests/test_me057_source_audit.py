"""ME057 software fixture contracts. Fake tensors here never imply model skill."""
import io
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.me057_audit_original_fit import (
    canonical_hash, verify_registration, verify_parquet)
from scripts import me055_reserve_independent_robot_task as source


def metadata_fixture():
    rows=[{"episode_index":i,"length":300} for i in range(12)]
    episodes=b"".join(json.dumps(r).encode()+b"\n" for r in rows)
    info=json.dumps({"robot_type":"r1_lite","fps":15,
        "total_episodes":len(rows),"total_frames":3600,
        "features":{"observations.state.qpos":{"shape":[14]},
                    "action.qpos":{"shape":[14]}}}).encode()
    frozen=source.build(info,episodes)
    frozen["publisher_info_sha256"]=source.sha(info)
    frozen["publisher_episodes_metadata_sha256"]=source.sha(episodes)
    return info,episodes,frozen


def test_identical_original_metadata_passes_and_changed_hash_fails():
    info,episodes,frozen=metadata_fixture()
    assert verify_registration(info,episodes,frozen)==source.reserve(
        source.metadata_rows(episodes))
    changed=dict(frozen)
    changed["publisher_info_sha256"]="0"*64
    with pytest.raises(RuntimeError,match="publisher info SHA"):
        verify_registration(info,episodes,changed)


def test_frozen_split_integrity_is_checked():
    info,episodes,frozen=metadata_fixture()
    changed=json.loads(json.dumps(frozen))
    changed["cohorts"]["FIT"]["episode_id_sha256"]="0"*64
    with pytest.raises(RuntimeError,match="split FIT"):
        verify_registration(info,episodes,changed)


def parquet_fixture(n=275,dimension=14):
    table=pa.table({
        "observations.state.qpos":pa.array(
            [[0.1]*dimension for _ in range(n)]),
        "action.qpos":pa.array([[0.2]*dimension for _ in range(n)]),
        "intervention":pa.array([False]*(n-1)+[True])})
    out=io.BytesIO()
    pq.write_table(table,out)
    return out.getvalue()


def test_real_source_shape_contract_on_software_parquet_fixture():
    row=verify_parquet(parquet_fixture(),275)
    assert row["frames"]==275
    assert row["actual_takeover_onsets"]==1
    assert row["robot_controlled_sample_origins"]>=1
    with pytest.raises(RuntimeError,match="dimension"):
        verify_parquet(parquet_fixture(dimension=13),275)
    with pytest.raises(RuntimeError,match="length"):
        verify_parquet(parquet_fixture(),276)


def test_canonical_receipt_hash_is_order_independent_but_value_sensitive():
    assert canonical_hash({"a":1,"b":2})==canonical_hash({"b":2,"a":1})
    assert canonical_hash({"a":1})!=canonical_hash({"a":2})
