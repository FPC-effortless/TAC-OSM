"""Local mathematical guards ONLY; all claims derive from original physical data."""
import torch
import torch.nn.functional as F
from scripts.latfuse012_original_source_oracle import (
    headview,read_value_head,read_value,normalized_cos,station_block_ci
)

def test_head_partition_capacity_and_operator_order():
    torch.manual_seed(1)
    M=torch.randn(4,16,16)
    k=F.normalize(torch.randn(4,2,8),dim=-1)
    read=read_value_head(M,k)
    native=torch.einsum("bhvk,bhk->bhv",headview(M),k)
    torch.testing.assert_close(read,native,atol=0,rtol=0)

def test_oracle_read_equals_weighted_per_head():
    torch.manual_seed(2)
    M=torch.randn(4,16,16)
    k=F.normalize(torch.randn(4,2,8),dim=-1)
    weights=F.softmax(torch.randn(4,2),dim=-1)
    got=read_value(M,k,weights)
    expected=(read_value_head(M,k)*weights[:,:,None]).sum(1)
    torch.testing.assert_close(got,expected,atol=0,rtol=0)

def test_cosine_has_no_fictitious_nonzero_when_source_absent():
    x=torch.randn(4,16)
    assert torch.equal(normalized_cos(torch.zeros_like(x),x),torch.zeros(4))

def test_deterministic_station_episode_bootstrap_not_independent_case_ci():
    # Artificial numbers verify BOOTSTRAP ONLY; NOT performance evidence.
    sites=torch.tensor([0,0,1,1,2,2])
    hits=torch.tensor([1.,1.,0.,0.,1.,1.])
    a=station_block_ci(sites,hits)
    b=station_block_ci(sites,hits)
    assert a==b and 0<=a[0]<=a[1]<=1
