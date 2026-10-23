"""An adapter-only CI lane must not hide changes to model mathematics."""

import pytest
from scripts.audit_guard_migration import normalized


def test_only_supported_optimizer_adapters_are_erased_for_source_comparison():
    original = 'import torch\ndef fit(model):\n opt=torch.optim.AdamW(model.parameters(),lr=.01)\n opt.step()\n torch.save(model.state_dict(),"original.pt")\n'
    guarded = 'import torch\nfrom tac_osm.parameter_budget import budgeted_optimizer, checked_optimizer_step, save_training_checkpoint\ndef fit(model):\n opt=budgeted_optimizer(model,torch.optim.AdamW,model.parameters(),lr=.01)\n checked_optimizer_step(opt)\n save_training_checkpoint(model.state_dict(),"original.pt")\n'
    assert normalized(original) == normalized(guarded)
    assert normalized(original) != normalized(guarded.replace("lr=.01", "lr=.1"))
    assert normalized("def forward(x):\n return x+1\n") != normalized(
        "def forward(x):\n return x+2\n"
    )


def test_migration_lane_rejects_scientific_result_or_registration_changes(
    tmp_path, monkeypatch
):
    from scripts.audit_guard_migration import verify_migration

    monkeypatch.setattr(
        "scripts.audit_guard_migration.subprocess.check_output",
        lambda *a, **kw: (
            "research/mathematical-engineering/me073/FAIR_COMPARISON.json\n"
        ),
    )
    with pytest.raises(ValueError, match="scientific evidence"):
        verify_migration("base...HEAD", tmp_path)
