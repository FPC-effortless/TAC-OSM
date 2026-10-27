"""LATFUSE011 small mathematical tests ONLY, not synthetic capability claims."""
import math
import torch
from scripts.latfuse011_real_memory_falsifiers import (
    wilson,exact_two_sided_sign_p,paired_summary,projected_read)

def test_random_unit_sphere_cosine_exact_formula_d8():
    d=8
    exact=math.gamma(d/2)/(math.sqrt(math.pi)*math.gamma((d+1)/2))
    assert .29<exact<.293
    assert abs(exact-math.sqrt(2/(math.pi*d)))<.012

def test_strict_source_paired_flip_sign_test_and_wilson():
    values=torch.tensor([.2,.8]*18+[.9,.1]*6)
    s=paired_summary(values)
    assert s["pairs"]==24 and s["winning_pairs"]==6
    assert s["rank"]==.25
    assert s["exact_two_sided_sign_p"]<.03
    assert s["rank_Wilson_95_interval"][1]<.5
    reversed_=torch.stack([values[1::2],values[::2]],dim=-1).flatten()
    assert paired_summary(reversed_)["rank"]==.75
    assert exact_two_sided_sign_p(18,24)==s["exact_two_sided_sign_p"]

def test_interference_projection_head_partition_dimensions():
    torch.manual_seed(92)
    M=torch.randn(4,16,16)
    q=torch.randn(4,2,8)
    route=torch.softmax(torch.randn(4,2),dim=-1)
    got=projected_read(M,q,route)
    native=M.reshape(4,16,2,8).permute(0,2,1,3)
    expected=torch.einsum("bhvk,bhk->bhv",native,q)
    expected=(expected*route[:,:,None]).sum(1)
    torch.testing.assert_close(got,expected,atol=0,rtol=0)

def test_zero_prefix_perturbation_projects_to_zero():
    torch.manual_seed(13)
    q=torch.randn(5,2,8)
    w=torch.softmax(torch.randn(5,2),dim=-1)
    assert torch.equal(projected_read(torch.zeros(5,16,16),q,w),
                       torch.zeros(5,16))
