"""DEV-039 head boundary is exactly the existing model's readout."""
import torch
from scripts.run_predictive_head_dev039 import build
from scripts.run_learned_schema_dev import public_stream_tensors,public_queries
from tac_osm.development.schema_maintenance import make_pair


def test_frozen_head_features_reproduce_original_predictions():
    torch.set_num_threads(1);model=build('bounded_local_multi_fp64').eval()
    cases=make_pair(399,32,32,'reissued_cancel','at',domain='dev039-contract')
    events=public_stream_tensors([c.observations for c in cases]);queries=public_queries([c.tick for c in cases],[c.horizon for c in cases])
    with torch.no_grad():
        state=model.encode(*events);q=model.query_encoder(queries)
        features=torch.cat((model.adapter(model.memory.read(state,q)),q),-1)
        assert features.shape==(2,8,3,44)
        assert torch.equal(model.head(features),model.decode(state,queries))
