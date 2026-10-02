import random
import pytest
torch = pytest.importorskip("torch")
import torch as th
from tac_osm.integrated_e2e_benchmark import HELDOUT, sample_episode, validate_episode
from tac_osm.integrated_e2e_content import ContentDiagnosisConfig, TypedContentModel
from scripts.run_integrated_e2e_004 import evaluate_oracle

def test_temporal_entities_and_heldout_compositions_are_valid():
    rng=random.Random(90210)
    for _ in range(50):
        ep=sample_episode(rng,combo1=HELDOUT[0],combo2=HELDOUT[1]); validate_episode(ep)
        assert ep[1][0]!=ep[2][0]
        assert (ep[1][3],ep[1][1],ep[1][2]) in HELDOUT
        assert (ep[2][3],ep[2][1],ep[2][2]) in HELDOUT

def test_typed_state_exact_operator_is_correct():
    model=TypedContentModel(ContentDiagnosisConfig(content_mode="learned"))
    memory=model.state.initial(1,th.device("cpu")); content=th.tensor([[1.0,1.0]+[0.0]*10]); entity=th.tensor([2])
    memory,_=model.state.write(memory,content,entity)
    for idx,expected in [(0,0.0),(1,1.0),(2,1.0),(3,1.0)]:
        out=model.query(memory,entity,th.tensor([0]),th.tensor([1]),th.tensor([idx]))
        assert abs(float(out["action"].item())-expected)<1e-6

def test_typed_learned_backward_reaches_representation_bit_head():
    model=TypedContentModel(ContentDiagnosisConfig(content_mode="learned")); batch=4; steps=3
    text=th.randint(0,40,(steps,batch,7)); image=th.rand(steps,batch,1,16,16); audio=th.randn(steps,batch,1,96)
    entities=th.randint(0,16,(steps,batch)); payloads=th.randint(0,2,(steps,batch,12)).float()
    q=(th.randint(0,16,(batch,)),th.randint(0,12,(batch,)),th.randint(0,12,(batch,)),th.randint(0,4,(batch,)),th.randint(0,2,(batch,)))
    out=model.forward_episode({"text":text,"image":image,"audio":audio},entities,payloads,q,q)
    out["loss"].backward()
    g=dict(model.named_parameters())["rep.bit_head.weight"].grad
    assert g is not None and th.isfinite(g).all()
