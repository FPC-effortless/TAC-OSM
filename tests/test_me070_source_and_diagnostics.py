"""Software-only fixtures. These are NOT authentic-data or prediction claims."""
import gzip
import json
from pathlib import Path
import pytest

from scripts.me070_audit_me069 import exact_mcnemar_p, audited_pair, audit
from scripts.me070_qualify_long_xes import measure


def score(case, correct, width=3600., loss=.5):
    return {"case_id":case,"prefix_last_index":3,"activity_correct":correct,
            "activity_NLL":loss,"interval_width_seconds":width,
            "interval_90_covered":True,"positive_wait_observed":True}


def test_exact_mcnemar_is_paired_and_two_sided():
    assert exact_mcnemar_p(0, 0) == 1.
    assert exact_mcnemar_p(3, 0) == pytest.approx(.25)
    assert exact_mcnemar_p(2, 2) == 1.
    assert exact_mcnemar_p(0, 6) == pytest.approx(.03125)


def test_case_overlap_even_when_aggregate_accuracies_match():
    native=[score("a",True,3600),score("b",False,7200),
            score("c",True,3600),score("d",False,7200)]
    gru=[score("c",False,3600),score("a",True,7200),
         score("d",True,3600),score("b",False,7200)]
    r=audited_pair(native,gru)
    assert (r["both_correct"],r["both_wrong"],r["native_only_correct"],r["gru_only_correct"]) == (1,1,1,1)
    assert r["accuracy_native"]==r["accuracy_gru"]==.5
    assert r["exact_mcnemar_p_two_sided"] == 1.
    assert r["interval_widths_native"]["median_hours"]==1.5


def test_unpaired_or_duplicate_case_is_fail_closed():
    x=[score("same",True),score("same",False)]
    with pytest.raises(ValueError,match="duplicates"):
        audited_pair(x,[score("same",True),score("other",False)])
    with pytest.raises(ValueError,match="unpaired"):
        audited_pair([score("a",True)],[score("b",True)])


def test_test_and_dev_all_five_seeds_mandatory():
    row=[score("a",True)]
    pairs={f"{a}-s{seed}":row for seed in range(69001,69006)
           for a in ("native_tac","gru_world")}
    report=audit({"scores":{"DEV":pairs,"TEST":pairs}})
    assert len(report["cohorts"]["TEST"])==5
    assert report["lifecycle_label_prevalence"].startswith("NOT AVAILABLE")
    del pairs["gru_world-s69003"]
    with pytest.raises(ValueError,match="missing original pair"):
        audit({"scores":{"DEV":pairs,"TEST":pairs}})


def write_source(path,case_lengths):
    lines=['<?xml version="1.0" encoding="UTF-8"?><log>']
    for n,length in enumerate(case_lengths):
        lines.append(f'<trace><string key="concept:name" value="c{n}"/>')
        for event in range(length):
            lines.append(f'<event><string key="concept:name" value="A"/>'
                         f'<string key="lifecycle:transition" value="complete"/>'
                         f'<date key="time:timestamp" value="2020-01-{1+n:02}T00:{event%60:02}:00+00:00"/>'
                         '</event>')
        lines.append('</trace>')
    lines.append('</log>')
    path.write_text("".join(lines))


def test_source_qualification_counts_real_input_shape_only_as_fixture(tmp_path):
    original=tmp_path/"software_fixture.xes"
    write_source(original,[3,4,5,6,7])
    r=measure(original,(2,4),.4,.2)
    assert r["case_count"]==5 and r["event_count"]==25
    assert r["cohorts"]["TEST"]["cases"]==2
    assert r["cohorts"]["TEST"]["history_support"]["4"]["distinct_cases_with_future_after_h"]==2
    assert not r["long_256_support_gate"]
    assert r["original_source_sha256"]


def test_source_qualification_rejects_nonchronological(tmp_path):
    original=tmp_path/"bad_source.xes"
    write_source(original,[61,3,3,3,3])
    with pytest.raises(ValueError,match="negative observed source event time"):
        measure(original)


def test_source_qualification_can_read_compressed_xes(tmp_path):
    original=tmp_path/"software_fixture.xes"
    write_source(original,[3,4,5,6,7])
    gz=tmp_path/"software_fixture.xes.gz"
    gz.write_bytes(gzip.compress(original.read_bytes()))
    assert measure(gz)["case_count"]==5

def test_lifecycle_targets_are_separated_from_base_activity():
    from scripts.me070_audit_lifecycle import lifecycle, audit
    assert lifecycle("W_Behandelen|START") == ("W_Behandelen","start")
    assert lifecycle("W_Behandelen|COMPLETE") == ("W_Behandelen","complete")
    cases=[{"n":4,"activity":[1,2,1,2]},{"n":3,"activity":[1,2,3]}]
    r=audit(cases,{"A|START":1,"A|COMPLETE":2,"B|COMPLETE":3})
    assert r["eligible_midpoint_cases"] == 2
    assert r["start_to_same_activity_complete_fraction"] == 1
    assert r["target_lifecycle_counts"]["complete"] == 2
