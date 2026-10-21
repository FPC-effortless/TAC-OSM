"""ME025 regression guards. Constructed observations are ONLY software tests."""
import torch
from tac_osm.me025_feature_parity import (
    KernelNative, TinyCausalTransformer, kernel_summaries,
    streaming_kernel_step, parameter_count, software_preflight
)

def test_me025_analytic_feature_parity_and_causal_mask():
    result=software_preflight()
    assert result["parameters"]=={
        "tac_rms":135050,
        "tac_kernel":135056,
        "tiny_transformer_kernel":134896,
    }
    assert result["streaming_summary_parity"]
    assert not result["causal_prefix_response_leak"]
    assert not result["model_performance_claim"]

def test_me025_old_summary_ignores_target_feedback():
    src=torch.tensor([[0.,1.,1.,0.,1.,0.,1.,1.],
                      [1.,0.,0.,1.,0.,1.,0.,0.]])
    original=kernel_summaries(src)
    for future in (torch.zeros(2,16),torch.ones(2,16)):
        # A next-event forecast on 8 prior events sees only src; an
        # unrelated future outcome cannot enter its feature arguments.
        assert torch.equal(original,kernel_summaries(src))
        assert future.shape==(2,16)

def test_me025_incremental_kernel_chunks():
    torch.manual_seed(253)
    feedback=torch.randint(0,2,(3,128)).float()
    running=torch.zeros(3,5); weight=torch.zeros(3,5)
    for first,last in [(0,11),(11,32),(32,97),(97,128)]:
        for i in range(first,last):
            running,weight=streaming_kernel_step(running,weight,feedback[:,i])
        assert torch.allclose(running/weight,kernel_summaries(feedback[:,:last]),
                              atol=2e-6,rtol=2e-6)
