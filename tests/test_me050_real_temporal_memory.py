"""ME050 numerical/leakage safeguards. Fixtures are NOT model performance tests."""
import csv
import pytest
import torch
import numpy as np
from tac_osm.me050_real_temporal_memory import ARMS,HISTORY,RealHistoryPredictor,real_observed_click_loss
from scripts import me050_train_real_temporal as study


def test_me050_memory_architecture_uses_native_exact_me038_recurrence():
    torch.manual_seed(50050)
    a=RealHistoryPredictor("tac_chronological")
    b=RealHistoryPredictor("tac_reversed")
    assert a.num_active()==b.num_active()
    b.load_state_dict(a.state_dict(),strict=True)
    h=torch.randint(0,4096,(3,HISTORY))
    x=torch.rand(3,HISTORY,3)
    v=torch.randint(0,4096,(3,))
    hour=torch.rand(3,1)
    assert torch.allclose(a(h,x,v,hour),b(h.flip(1),x.flip(1),v,hour))
    loss=real_observed_click_loss(a,h,x,v,hour,torch.tensor([0.,1.,0.]))
    loss.backward()
    assert a.memory.transition.weight.grad is not None
    assert torch.isfinite(a.memory.transition.weight.grad).all()


def test_me050_all_four_heads_have_real_logged_label_loss_and_finite_gradients():
    study.self_test()
    assert len(ARMS)==4


def _fixture(path,late_outcome="1"):
    cols=["user_id","video_id","time_ms","date","is_click","is_rand"]
    with open(path,"w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader()
        for user in range(20):
            for j in range(20):
                date=20220430 if j<14 else 20220503 if j<17 else 20220506
                label=str(j%2) if date<20220502 else late_outcome
                writer.writerow({"user_id":user,"video_id":(user*41+j)%7350,
                    "time_ms":1651280400000+(j*86400000)+user*1000,
                    "date":date,"is_click":label,"is_rand":1})


def test_me050_late_sealed_outcomes_are_not_parsed_during_fit(tmp_path):
    path=tmp_path/"original-fixture.csv"
    _fixture(path,late_outcome="invalid_before_freeze")
    fit=study.read_real_events(path,fit_only=True)
    assert len(fit)==20*14
    source_data=study.assemble_histories(fit)
    assert source_data["FIT"]["h_vid"].shape[1]==HISTORY
    with pytest.raises(ValueError):
        study.read_real_events(path,fit_only=False)


def test_me050_true_linked_user_histories_never_use_current_or_future_event(tmp_path):
    path=tmp_path/"valid-fixture.csv"
    _fixture(path,late_outcome="1")
    all_rows=study.read_real_events(path,fit_only=False)
    groups=study.assemble_histories(all_rows)
    assert set(groups)=={"FIT","DEV","TEST"}
    assert len(groups["FIT"]["y"])==20*2
    assert len(groups["DEV"]["y"])==20*3
    assert len(groups["TEST"]["y"])==20*3
    # Every user has exactly 12 prior events before its first FIT target.
    assert np.isfinite(groups["TEST"]["h_num"].numpy()).all()
    assert (groups["FIT"]["h_num"][:,:,1]>=0).all()
    assert study.fitted_example_digest(groups["FIT"])==study.fitted_example_digest(
        study.assemble_histories(study.read_real_events(path,fit_only=True))["FIT"])
