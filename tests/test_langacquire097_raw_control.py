"""Constructed control checks, not trained capability evidence."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


raw = load(ROOT/'research/mathematical-engineering/langacquire097/raw_control.py', 'raw097')
old = load(ROOT/'research/mathematical-engineering/langpersist096/raw_history.py', 'raw096')


def test_fixed_budget_wraparound_and_established_distribution():
    ring = raw.RawRing()
    observed = list(range(256)) * 9
    ring.admit(observed)
    assert len(ring.storage) == 1928
    assert ring.observed() == bytes(observed[-raw.CAPACITY:])
    np.testing.assert_array_equal(raw.distribution(b'alpha = alpha\nalpha'),
                                  old.distribution(b'alpha = alpha\nalpha'))


def test_greedy_feedback_uses_only_observations_and_own_outputs():
    acquisition, query = list(b'alpha alpha alpha\n'), list(b'alpha ')
    result = raw.predict(acquisition, query)
    history = bytes(acquisition + query); expected = []
    np.testing.assert_array_equal(result['log_probs'], np.log(old.distribution(history)))
    for _ in range(4):
        value = int(old.distribution(history).argmax()); expected.append(value)
        history += bytes([value])
    np.testing.assert_array_equal(result['generated'], expected)
    assert result['cost']['state_payload_bytes'] == 1928


def test_last_demo_reference_copies_only_past_output_with_fixed_budget():
    acquisition = list(b'R;110>111;112>113;')
    output = raw.copy_last_demonstration(acquisition)
    assert output.dtype == np.uint8 and output.shape == (40,)
    assert output.tobytes().rstrip() == b'113;'
