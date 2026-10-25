"""The no-step trace must expose direct objective-to-head gradient paths."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import me100_sensor_data as data
from scripts import me104_split_panel
from scripts import me105_flow_trace


def test_point_loss_trains_location_and_median_heads_but_not_width_head():
    fit = data.load("fit")
    rawscale = me104_split_panel.torch.tensor(
        me104_split_panel.np.nanstd(
            fit["raw"][..., :6].astype(me104_split_panel.np.float64), axis=(0, 1)
        ).clip(1e-4),
        dtype=me104_split_panel.torch.float32,
    )
    row = me105_flow_trace._trace_one(fit, rawscale, 0)
    old = row["models"]["ME104"]["gradient_norms"]["point"]
    new = row["models"]["ME105"]["gradient_norms"]["point"]

    assert old["core"]["norm"] > 0
    assert old["width_head"]["norm"] == 0
    assert new["core"]["norm"] > 0
    assert new["median_head"]["norm"] > 0
    assert new["width_head"]["norm"] == 0
    assert row["models"]["ME104"]["gradient_norms"]["nll"]["width_head"]["norm"] > 0
    assert row["models"]["ME105"]["gradient_norms"]["nll"]["width_head"]["norm"] > 0
    for model_result in row["models"].values():
        for component in model_result["gradient_norms"].values():
            assert all(group["finite"] for group in component.values())
