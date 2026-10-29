"""Constructed correctness/causality fixtures, not capability evidence."""
import sys
from pathlib import Path
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research/mathematical-engineering/langmem084'))
from operators import ARMS,arm_parts,context_key,byte_book,update_matrix,decode,select_controls

def test_full_byte_identity_is_injective_and_single_write_decodes_all_values():
    book=byte_book();assert len(torch.unique(book,dim=0))==256
    key=context_key(b'abcd')
    for b in range(256):
        M=update_matrix(torch.zeros(8,8),key,book[b],mode='source',rho=.99,beta=.02,epsilon=.005)
        assert decode(M,key,book)[0]==b

def test_unit_residual_overwrite_exact_current_value_and_orthogonal_old_value():
    g=torch.Generator().manual_seed(84)
    M=torch.randn(8,8,generator=g,dtype=torch.float64)
    k=torch.nn.functional.normalize(torch.randn(8,generator=g,dtype=torch.float64),dim=0)
    v=torch.randn(8,generator=g,dtype=torch.float64)
    new=update_matrix(M,k,v,mode='overwrite',rho=.99,beta=.02,epsilon=.005)
    assert torch.allclose(new@k,v,atol=1e-12,rtol=1e-12)
    orth=torch.randn(8,generator=g,dtype=torch.float64);orth=orth-(orth@k)*k
    assert torch.allclose(new@orth,.99*M@orth,atol=1e-12,rtol=1e-12)

def test_native_update_is_affine_formula_and_exact_payload():
    k=context_key(b'abc1');v=byte_book()[241];M=torch.arange(64).float().view(8,8)/64
    new=update_matrix(M,k,v,mode='source',rho=.98,beta=.03,epsilon=.002)
    expected=.98*M-.98*.03*(M@k)[:,None]*k[None,:]+.002*v[:,None]*k[None,:]
    assert torch.allclose(new,expected,atol=1e-6)
    assert new.numel()*new.element_size()==256

def test_context_hash_is_deterministic_prewrite_only_and_target_blind():
    prefix=b'abcd1abcd2'*14;t=128;a=32
    keys={b:context_key(prefix[b-4:b]) for b in range(4,t+1)}
    out=select_controls(prefix[:t+1],keys,a,t,'owner')
    for b in out[:2]:assert b<t and prefix[b-4:b]!=prefix[a-4:a]
    # Actual code path uses only past/current prefix; appending future cannot enter it.
    poisoned=prefix[:t+1]+b'Z'*100
    assert select_controls(poisoned[:t+1],keys,a,t,'owner')==out
    assert torch.equal(context_key(b'abcd'),context_key(b'abcd'))
    with pytest.raises(ValueError):select_controls(poisoned,keys,a,t,'owner')

def test_registered_factorial_and_no_unknown_arm():
    assert len(ARMS)==8
    for arm in ARMS:assert len(arm_parts(arm))==3
    with pytest.raises(ValueError):arm_parts('unregistered')
