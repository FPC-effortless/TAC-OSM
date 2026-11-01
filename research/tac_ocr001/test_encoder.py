import pytest
import torch

from .encoder import FeatureCache,FrozenTrOCREncoder,spatial_patch_features


def test_immutable_encoder_revision_required_before_import():
    with pytest.raises(ValueError,match="immutable"):
        FrozenTrOCREncoder("microsoft/trocr-small-printed","main")


def test_image_only_content_cache(tmp_path):
    cache=FeatureCache(tmp_path)
    key=cache.key(b"actual image bytes","frozen-encoder-revision")
    other=cache.key(b"different image bytes","frozen-encoder-revision")
    assert key!=other and cache.get(key) is None
    features,boxes=torch.randn(1,8,4),torch.rand(1,8,4)
    cache.put(key,features,boxes)
    original=features.clone()
    features.add_(10)
    saved,_=cache.get(key)
    torch.testing.assert_close(saved,original)
    assert cache.get(other) is None
    with pytest.raises(ValueError):
        cache.get("../escape")


@pytest.mark.parametrize("prefix",[1,2])
def test_vit_and_deit_special_tokens_have_no_spatial_boxes(prefix):
    features=torch.randn(2,576+prefix,384)
    patches,boxes=spatial_patch_features(features,384,16)
    assert patches.shape==(2,576,384) and boxes.shape==(2,576,4)
    torch.testing.assert_close(patches,features[:,prefix:])
    torch.testing.assert_close(boxes[0,0],torch.tensor([0.,0.,1/24,1/24]))
    torch.testing.assert_close(boxes[0,-1],torch.tensor([23/24,23/24,1.,1.]))
