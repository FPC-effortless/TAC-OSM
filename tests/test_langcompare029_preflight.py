"""LANGCOMPARE029 no-fit safety checks, not trained benchmark evidence."""
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"scripts/langcompare029_preflight.py"
spec=importlib.util.spec_from_file_location("langcompare029_preflight_test",SCRIPT)
p=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)

def test_parameter_budget_formula_and_spread():
    assert p.BDH_ANALYTICAL==501504
    assert p.MAMBA_ANALYTICAL==501884
    assert p.TRANSFORMER==496960
    assert (max(p.EXPECTED)-min(p.EXPECTED))/max(p.EXPECTED)<0.01

def test_transformer_actual_and_no_future_leakage():
    model=p.ByteTransformer()
    assert p.parameter_count(model)=={"trainable":496960,"deployed":496960}
    assert p.compare_prefix(model,"cpu",1e-5)<1e-5

def test_exact_existing_native_graph_not_mutated_and_causal():
    m=p.load_tac()
    assert p.parameter_count(m)=={"trainable":500122,"deployed":500122}
    assert p.compare_prefix(m,"cpu",1e-5)<1e-5

def test_preflight_never_claims_full_four_arm_or_training_without_mamba():
    r=p.preflight(bdh_path=None,attempt_mamba=False,check_causality=False)
    assert r["status"]=="NO_FIT_PARTIAL_PREFLIGHT_NOT_READY_TO_TRAIN"
    assert r["optimizer_steps"]==0 and r["gpu_minutes"]==0
    assert r["DEV_NLL"]=="NOT_RUN" and not r["training_authorized"]
    assert r["source"]["fit_docs"]==14 and r["source"]["exposed_dev_docs"]==4

def test_complete_gate_fails_closed():
    with pytest.raises(RuntimeError,match="INCOMPLETE"):
        p.preflight(bdh_path=None,attempt_mamba=False,
                    require_complete=True,check_causality=False)


def test_mamba3_t4_is_rejected_before_any_kernel_or_gpu_run(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda,"is_available",lambda: True)
    monkeypatch.setattr(torch.cuda,"current_device",lambda: 0)
    monkeypatch.setattr(torch.cuda,"get_device_capability",lambda *args: (7,5))
    with pytest.raises(RuntimeError,match="SM75.*minimum SM80"):
        p.load_mamba3()

def test_frozen_training_schedule_reproduces_same_sources_and_offsets():
    import importlib.util
    spec=importlib.util.spec_from_file_location(
        "langcompare029_sampling_test",ROOT/"scripts/langcompare029_sampling.py")
    sampler=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sampler)
    fit=sampler.fit_sources()
    a=sampler.frozen_plan()
    b=sampler.frozen_plan()
    assert a==b
    assert a["optimizer_steps_performed"]==0
    assert len(a["by_seed"])==5
    assert len({x["sample_order_sha256"] for x in a["by_seed"]})==5
    assert all(x["examples_seen"]==2560 and
               x["next_byte_targets_seen"]==327680 and
               x["distinct_fit_owners_seen"]==14 for x in a["by_seed"])
    for step,batch in sampler.iter_batches(fit,1701,steps=2):
        assert step in (0,1) and len(batch)==8
        for name,start,raw in batch:
            assert raw==fit[name][start:start+129]
            assert raw[1:]==fit[name][start+1:start+129]

def test_true_training_loss_backward_without_optimizer_or_metrics():
    import importlib.util
    import torch
    import torch.nn.functional as F
    spec=importlib.util.spec_from_file_location(
        "langcompare029_sampling_test_grad",ROOT/"scripts/langcompare029_sampling.py")
    sampler=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sampler)
    fit=sampler.fit_sources()
    _,entries=next(sampler.iter_batches(fit,1701,steps=1))
    token=torch.tensor([list(entries[0][2][:13])],dtype=torch.long)
    for model in (p.load_tac(),p.ByteTransformer()):
        model.train()
        logits=model(token[:,:-1])
        loss=F.cross_entropy(logits.reshape(-1,256),token[:,1:].reshape(-1))
        loss.backward()
        assert torch.isfinite(loss).item()
        assert all(par.grad is not None and torch.isfinite(par.grad).all()
                   for par in model.parameters() if par.requires_grad)
        assert sum(float(par.grad.abs().sum()) for par in model.parameters())>0


def test_500k_exception_exact_study_marker_and_scope():
    import importlib.util
    spec=importlib.util.spec_from_file_location(
        "langcompare029_parity_test",ROOT/"scripts/validate_research_parity.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for sid in ("LANGTRAIN025","LANGCOMPARE029"):
        assert module.checked_new_scale({
            "study_id":sid,"parameter_budget_contract":"owner-2026-10-28-500k"})
    for sid in ("LANGCOMPARE030","ME075","OTHER"):
        assert not module.checked_new_scale({
            "study_id":sid,"parameter_budget_contract":"owner-2026-10-28-500k"})
    assert not module.checked_new_scale({"study_id":"LANGCOMPARE029"})
    module.check_500k_model_size(500122,500122,"matched_TAC")
    module.check_500k_model_size(496960,496960,"matched_Transformer")
    with pytest.raises(module.ParityError):
        module.check_500k_model_size(100000,100000,"invalid_100k_new_scale")
    with pytest.raises(module.ParityError):
        module.check_model_parameter_target(500122,500122,"other_study")
