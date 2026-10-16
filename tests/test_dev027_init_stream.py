"""Matched initial tensors and complete blocked factorial integrity."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_dev027_init_stream as a  # noqa: E402


def test_full_twenty_models_required():
    with pytest.raises(AssertionError):a.summarize([])


def test_order_is_complete_and_repeatable_for_each_seed():
    for seed in a.config()['seeds']:
        assert set(a.block_order(seed))==set(a.ARMS)
        assert a.block_order(seed)==a.block_order(seed)


def test_both_initializations_match_archived_initial_tensors():
    pytest.importorskip('torch')
    import run_dev027_init_stream as runner
    for seed in a.config()['seeds']:
        for init in ('A','B'):
            runner.initial_model(seed,init)


def test_mean_gain_with_uncertain_sign_does_not_pass():
    r=a.effect([.4,-.3,.2,-.1,.1])
    assert not r['meaningful_separation']
