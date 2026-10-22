import numpy as np
import pytest

from scripts.me056_profile_paired_execution import percentile_interval


def test_bootstrap_interval_is_deterministic_and_contains_constant_effect():
    assert percentile_interval([.1]*11)==pytest.approx([.1,.1])


def test_bootstrap_rejects_too_few_or_nonfinite_pairs():
    with pytest.raises(ValueError):
        percentile_interval([.1]*4)
    with pytest.raises(ValueError):
        percentile_interval([.1]*10+[np.nan])


def test_dense_auxiliary_snapshots_precede_actual_next_state(monkeypatch):
    # Synthetic tensors are software-causality checks, not performance data.
    import torch
    import scripts.me056_profile_paired_execution as profile
    from tac_osm.real_robot_me055_rollout import CompactBelief

    monkeypatch.setattr(profile,"HISTORY",2)

    class MinimalModel:
        def initial_from_observation(self,observed):
            row=observed[:,:1].clone()
            return CompactBelief(row,row,row)

    def advance(belief,current_state,action,human,next_state,next_human):
        # Deliberately copies the later physical measurement into memory.
        later=next_state[:,:1].clone()
        return CompactBelief(later,later,later)

    states=torch.zeros(1,3,14)
    states[:,0,:]=1.
    states[:,1,:]=2.
    states[:,2,:]=3.
    actions=torch.zeros(1,2,14)
    human=torch.zeros(1,3,1,dtype=torch.bool)

    def run(source):
        return profile.burnin_with_causal_snapshots(
            MinimalModel(),source,actions,human,advance,advance)

    final,snapshots=run(states)
    assert [float(s.fast[0,0]) for s in snapshots]==[1.,2.]
    assert float(final.fast[0,0])==3.

    changed_last=states.clone()
    changed_last[:,2,:]=999.
    final_changed,old_history=run(changed_last)
    assert all(torch.equal(a.fast,b.fast) for a,b in zip(snapshots,old_history))
    assert float(final_changed.fast[0,0])==999.
