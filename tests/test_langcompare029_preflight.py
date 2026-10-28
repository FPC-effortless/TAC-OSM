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
        module.check_model_parameter_target(500122,500122,label="other_study")


def test_frozen_source_schedule_matches_archived_original_all_five_seeds():
    import importlib.util
    import json
    spec=importlib.util.spec_from_file_location(
        "langcompare029_sampling_source_match",ROOT/"scripts/langcompare029_sampling.py")
    sampler=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sampler)
    plan=sampler.frozen_plan()
    original=json.loads((ROOT/"research/replay-bundles/LANGTRAIN025/"
          "LANGTRAIN025_AUTHENTIC_EXPOSED_DEV_RECEIPT.json").read_text())
    expected={x["seed"]:x["training_samples_sha256"] for x in original["model_runs"]}
    actual={x["seed"]:x["sample_order_sha256"] for x in plan["by_seed"]}
    assert actual==expected and len(actual)==5
    assert original["training_updates"]==plan["training_updates"]==320
    assert original["predicted_targets_per_seed"]==4096 # DEV exposures stay separated


def test_official_siso_mamba3_independent_source_parameter_ledger():
    from scripts.langcompare029_mamba_ledger import ledger
    result=ledger()
    assert result["individual_siso_mixer_parameters"]["in_proj.weight [407,80]"]==32560
    assert result["per_layer_parameters"]==76794
    assert result["analytical_total"]==501884
    assert result["actual_cuda_model_count"]=="PENDING"

def test_safe_untrained_model_optimizer_rng_checkpoint_replay():
    from scripts.langcompare029_checkpoint import (
        create_bytes,verify_bytes,compare_no_fit_logits,sha)
    from scripts.langcompare029_sampling import fit_sources,iter_batches,schedule_receipt
    import torch,copy
    torch.manual_seed(1701)
    model=p.load_tac().eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    fit=fit_sources()
    receipt=schedule_receipt(fit,1701)
    raw=create_bytes(model=model,optimizer=optimizer,model_name="tac_context026",
        seed=1701,source_order_sha=receipt["sample_order_sha256"],
        updates=0,diagnostic_only=True)
    replay=p.load_tac().eval()
    opt2=torch.optim.AdamW(replay.parameters(),lr=.001)
    x=verify_bytes(raw,fresh_model=replay,fresh_optimizer=opt2,
        expected_name="tac_context026",seed=1701,
        sample_order_sha=receipt["sample_order_sha256"],
        expected_digest=sha(raw),diagnostic_only=True)
    assert x["status"]=="PASS_SAME_SOURCE_COLD_LOAD_UNTRAINED"
    _,batch=next(iter_batches(fit,1701,steps=1))
    prefix=torch.tensor([list(batch[0][2][:9])],dtype=torch.long)
    assert compare_no_fit_logits(model,replay,prefix,atol=0.)==0.
    with pytest.raises(ValueError,match="SHA256 mismatch"):
        verify_bytes(raw,fresh_model=p.load_tac(),
            fresh_optimizer=torch.optim.AdamW(p.load_tac().parameters()),
            expected_name="tac_context026",seed=1701,
            sample_order_sha=receipt["sample_order_sha256"],expected_digest="0"*64,
            diagnostic_only=True)
    with pytest.raises(ValueError,match="metadata"):
        verify_bytes(raw,fresh_model=p.load_tac(),
            fresh_optimizer=torch.optim.AdamW(p.load_tac().parameters()),
            expected_name="tac_context026",seed=1702,
            sample_order_sha=receipt["sample_order_sha256"],
            diagnostic_only=True)
    with pytest.raises(ValueError,match="trained record"):
        create_bytes(model=model,optimizer=optimizer,model_name="tac_context026",
            seed=1701,source_order_sha=receipt["sample_order_sha256"],
            updates=0,diagnostic_only=False)


def test_guarded_training_entrypoint_denies_all_unapproved_models_on_cpu():
    from scripts.langcompare029_train import run,blockers
    assert "NO_APPROVED_SM80_PLUS_BF16_TWO_GPU_COMPARISON_VENUE" in blockers()
    assert "FOUR_ARM_PARITY_MANIFEST_MISSING" in blockers()
    for arm in p.ARM_ORDER:
        r=run(mode="dry-run",arm=arm,seed=1701)
        assert r["status"]=="BLOCKED_NO_FIT"
        assert r["optimizer_steps"]==0 and r["gpu_minutes"]==0
        assert r["DEV_NLL"]=="NOT_RUN" and r["protected_TEST"]=="NOT_ACCESSED"
        with pytest.raises(PermissionError,match="no fit permitted"):
            run(mode="fit",arm=arm,seed=1701)


def test_full_four_arm_scoring_is_blocked_before_all_20_matched_checkpoints(monkeypatch):
    from scripts.langcompare029_eval import cohort_readiness,evaluate
    monkeypatch.delenv("PHONE_LAB_GPU_EXECUTION_ENABLED",raising=False)
    r=cohort_readiness()
    assert r["status"]=="BLOCKED_NO_SCORING"
    assert "FINAL_RESOURCE_PARITY_MANIFEST" in r["missing"]
    assert r["checkpoint_count"]==0 and r["DEV_NLL"]=="NOT_RUN"
    with pytest.raises(PermissionError,match="operator GPU quota approval"):
        evaluate(bdh_source=None)

def test_next_byte_scoring_geometry_on_four_authentic_fit_documents_only():
    from scripts.langcompare029_eval import authentic_dev_metrics
    from scripts.langcompare029_sampling import fit_sources
    import sys
    from pathlib import Path
    sys.path.insert(0,str(ROOT/"research/mathematical-engineering/langtrain025"))
    from train_byte_native import windows
    fit=fit_sources()
    # No DEV: same scorer must work on original FIT-only windows as a diagnostic.
    cases=windows({k:fit[k] for k in sorted(fit)[:4]})
    m=p.ByteTransformer().eval()
    output=authentic_dev_metrics(m,cases)
    assert output["n"]==4096 and len(output["per_document"])==4
    assert 0.<output["nll"]<15.
    assert 0.<=output["top1"]<=1.
    assert 0.<=output["top5"]<=1.
    assert output["brier"]>0.
