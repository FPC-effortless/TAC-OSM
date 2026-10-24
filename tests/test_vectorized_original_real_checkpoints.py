"""No-fit parity of all FIVE previously trained original real-data checkpoints.

Input probes below are source-free diagnostics only, not evidence of real
predictive improvement. Checkpoint weights are authentic ESWAM003 models.
"""
from pathlib import Path
import torch

from tac_osm.variants.selective_exchange_world import SelectiveExchangeWorldModel
from tac_osm.variants.vectorized_exchange_world import VectorizedSelectiveExchangeWorldModel

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "research" / "replay-bundles" / "ESWAM003"


def test_all_five_original_trained_checkpoint_weights_and_inference_equivalence():
    torch.set_num_threads(1)
    for seed in range(76001, 76006):
        path = FROZEN / f"seed-{seed}" / "final.pt"
        assert path.is_file(), f"missing original real trained checkpoint: {path}"
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        a = SelectiveExchangeWorldModel(14, 14)
        b = VectorizedSelectiveExchangeWorldModel(14, 14)
        a.load_state_dict(bundle["weights"],strict=True)
        b.load_state_dict(bundle["weights"],strict=True)
        torch.manual_seed(seed)
        state_a, state_b = a.initial_selective(4), b.initial_selective(4)
        with torch.no_grad():
            for t in range(5):
                measured = torch.randn(4, 14)
                issued = torch.randn(4, 14)
                mask = torch.rand(4, 14) > .2
                mask[0] = t != 3
                elapsed = torch.full((4,), .1)
                ids = torch.tensor([0, 1, 2, t // 2], dtype=torch.long)
                p = a(measured, issued, elapsed, ids, mask, state=state_a)
                q = b(measured, issued, elapsed, ids, mask, state=state_b)
                for field in ("mean_next", "scale_next", "proposed_action",
                              "movement_logit", "movement_probability"):
                    torch.testing.assert_close(getattr(p,field),getattr(q,field),
                                               atol=5e-6,rtol=2e-5)
                for field in ("ids","keys","values","occupied"):
                    left=getattr(p.state.world.memory,field)
                    right=getattr(q.state.world.memory,field)
                    if left.dtype.is_floating_point:
                        torch.testing.assert_close(left,right,atol=5e-6,rtol=2e-5)
                    else: assert torch.equal(left,right)
                state_a, state_b = p.state, q.state
