"""LANGMEM088 constructed engineering tests only, no scientific success implied."""
import sys,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'research/mathematical-engineering/langmem086'),str(ROOT/'research/mathematical-engineering/langmem087'),str(ROOT)]
from compact_tac import CompactDualClippedTAC086
from diagnostic_tac087 import DiagnosticTAC087

def test_bias_plus1_has_same_100521_weights_and_only_one_initial_delta():
    torch.manual_seed(1701);original=CompactDualClippedTAC086().eval()
    torch.manual_seed(1701);biased=CompactDualClippedTAC086().eval()
    with torch.no_grad():biased.read_mix.bias[1].add_(1.)
    assert original.counts()['trainable']==biased.counts()['trainable']==100521
    assert original.start(('test',)).tensor_bytes()==biased.start(('test',)).tensor_bytes()==1920
    diffs={name for name in original.state_dict() if not torch.equal(original.state_dict()[name],biased.state_dict()[name])}
    assert diffs=={'read_mix.bias'}
    assert torch.equal(original.read_mix.bias[:1],biased.read_mix.bias[:1])
    assert abs(float(biased.read_mix.bias[1]-original.read_mix.bias[1])-1)<1e-6

def test_biased_fresh_graph_matches_diagnostic_default_and_late_state():
    torch.manual_seed(8801)
    model=CompactDualClippedTAC086().eval()
    with torch.no_grad():model.read_mix.bias[1].add_(1.)
    diagnostic=DiagnosticTAC087().eval()
    diagnostic.load_state_dict(model.state_dict(),strict=True)
    tokens=torch.tensor([[32,55,121,67,77,101,100,34]],dtype=torch.long)
    with torch.no_grad():
        p,a=model(tokens,owners=('same',))
        q,b=diagnostic(tokens,owners=('same',))
    assert torch.equal(p,q)
    assert all(torch.equal(a.tensors[k],b.tensors[k]) for k in a.tensors)

def test_softmax_bias_monotonic_at_equal_base_logits():
    logits=torch.tensor([[0.,0.]])
    fractions=[]
    for shift in (0.,1.,2.):
        v=logits.clone();v[:,1]+=shift
        fractions.append(float(v.softmax(-1)[0,1]))
    assert fractions[0]<fractions[1]<fractions[2]
