"""Constructed engineering falsifiers; no optimizer or capability evidence."""
import sys
from dataclasses import replace
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'research/mathematical-engineering/langmem086'),str(ROOT/'research/mathematical-engineering/langmem087')]
from compact_tac import CompactDualClippedTAC086
from diagnostic_tac087 import DiagnosticTAC087

def test_default087_is_original086_and_oldslow_split_has_independent_current_write():
    torch.set_num_threads(2);torch.manual_seed(87)
    original=CompactDualClippedTAC086().double().eval();diagnostic=DiagnosticTAC087().double().eval()
    diagnostic.load_state_dict(original.state_dict());x=torch.tensor([[31,65,200,33,90,65]])
    with torch.no_grad():
        a,sa=original(x,owners=('one',));b,sb=diagnostic(x,owners=('one',))
        assert torch.equal(a,b) and all(torch.equal(sa.tensors[k],sb.tensors[k]) for k in sa.tensors)
        _,nxt,tr=diagnostic(torch.tensor([[42]]),state=sb,owners=('one',),return_trace=True)
        assert torch.allclose(tr[0]['old_slow_component']+tr[0]['current_slow_component'],nxt.tensors['slow_memory'],atol=1e-14,rtol=1e-14)
        blank=replace(sb,tensors={**sb.tensors,'slow_memory':torch.zeros_like(sb.tensors['slow_memory'])})
        currentonly,cs=diagnostic(torch.tensor([[42]]),state=blank,owners=('one',))
        attributed,atr=diagnostic(torch.tensor([[42]]),state=sb,owners=('one',),lesion='old_slow_zero')
        assert torch.equal(currentonly,attributed)
        assert torch.equal(atr.tensors['slow_memory'],nxt.tensors['slow_memory'])
        assert not torch.equal(cs.tensors['slow_memory'],atr.tensors['slow_memory'])

def test_oldslow_counterfactuals_preserve_original_writes_and_all_nonupper_state():
    torch.manual_seed(87);m=DiagnosticTAC087().eval()
    with torch.no_grad():
        _,state=m(torch.arange(16).view(1,-1),owners=('one',))
        before={k:v.clone() for k,v in state.tensors.items()}
        _,normal=m(torch.tensor([[21]]),state=state,owners=('one',))
        for lesion in ('old_slow_zero','old_slow_permuted','current_slow_write_zero'):
            _,other=m(torch.tensor([[21]]),state=state,owners=('one',),lesion=lesion)
            assert all(torch.equal(other.tensors[k],normal.tensors[k]) for k in normal.tensors if k!='upper')
        assert all(torch.equal(state.tensors[k],before[k]) for k in before)

def test_sampler_uses_exact_original086_prefix_and_every_checkpoint_nested():
    from scripts.langmem087_run import plan,schedule,batches
    from scripts.langmem086_run import fit_sources086,schedule as schedule086
    docs,_=fit_sources086()
    for seed in (1701,1702,1703):
        assert schedule(docs,seed,160)==schedule086(docs,seed)
        it=iter(batches(docs,seed,640));prefix=[next(it) for _ in range(160)]
        assert prefix==list(batches(docs,seed,160))
