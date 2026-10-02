from tac_osm.representation import (
    FixedTransitionCase,
    RepresentationAnchor,
    RepresentationFeature,
    RepresentationProposal,
    RepresentationValidator,
)


def test_proposal_digest_is_stable():
    p = RepresentationProposal(
        proposal_id="p1",
        proposer="test",
        prompt_hash="abc",
        features=(
            RepresentationFeature(
                "x",
                "first coordinate",
                (RepresentationAnchor("zero", 0.0, "s0"),),
            ),
        ),
    )
    assert p.digest() == p.digest()
    assert p.feature_names == ("x",)


def test_dynamic_validation_uses_fixed_cases_and_emits_counterexample():
    vectors = {"a": (0.0,), "b": (0.1,), "a2": (0.0,), "b2": (1.0,)}
    enc = lambda sid: vectors[sid]
    validator = RepresentationValidator(enc)
    cases = (
        FixedTransitionCase(
            pair_id="pair-1",
            state_a="a",
            state_b="b",
            actions=("act",),
            horizon=1,
        ),
    )

    result = validator.validate_dynamic(
        cases,
        transition=lambda sid, action: {
            ("a", "act"): "a2",
            ("b", "act"): "b2",
        }[(sid, action)],
        outcome=lambda sid, action: 1,
        near_threshold=0.2,
        successor_tolerance=0.2,
    )

    assert result.considered_pairs == 1
    assert result.near_pairs == 1
    assert result.checked_transitions == 1
    assert result.passed is False
    assert len(result.violations) == 1
    assert result.violations[0].pair_id == "pair-1"


def test_non_near_fixed_pair_is_not_silently_treated_as_equivalent():
    vectors = {"a": (0.0,), "b": (2.0,)}
    validator = RepresentationValidator(lambda sid: vectors[sid])
    cases = (
        FixedTransitionCase("pair-2", "a", "b", ("act",), 1),
    )
    result = validator.validate_dynamic(
        cases,
        transition=lambda sid, action: sid,
        outcome=lambda sid, action: 0,
        near_threshold=0.5,
        successor_tolerance=0.0,
    )
    assert result.near_pairs == 0
    assert result.checked_transitions == 0
    assert result.passed is True


def test_anchor_mismatch_is_rejected():
    proposal = RepresentationProposal(
        "p2",
        "test",
        "hash",
        (
            RepresentationFeature(
                "x",
                "feature",
                (RepresentationAnchor("mid", 0.5, "s"),),
            ),
        ),
    )
    validator = RepresentationValidator(lambda sid: (0.7,))
    try:
        validator.check_anchors(proposal)
    except ValueError:
        pass
    else:
        raise AssertionError("anchor mismatch must be rejected")
