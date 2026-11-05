import numpy as np
import torch
from tac_osm.csl125 import CSL125Config,LocalGQA
from tac_osm.csl125_math_reference import grouped_local_attention,legal_keys,serial_ema


def test_causal_legal_windows():
    ids=np.arange(12)
    assert legal_keys(9,ids,4).nonzero()[0].tolist()==[6,7,8,9]


def test_bounded_gqa_matches_independent_numpy():
    torch.manual_seed(44)
    cfg=CSL125Config(vocab_size=256,width=24,layers=4,ff_width=48,
        role_count=3,state_width=8,rungs=(2,4),attention_layers=(2,4),
        query_heads=4,kv_heads=2,attention_window=5)
    m=LocalGQA(cfg).eval()
    x=torch.randn(1,12,24)
    s=m.initial_state(1,torch.device('cpu'),x.dtype)
    with torch.no_grad():
        out,_=m(x,s)
        h=m.pre(x);d=m.head_dim
        q=m._rope(m.q(h).reshape(1,12,4,d).transpose(1,2),torch.arange(12))
        kv=m.kv(h).reshape(1,12,2,2,d)
        k=m._rope(kv[:,:,0].transpose(1,2),torch.arange(12))
        v=kv[:,:,1].transpose(1,2)
        ref=grouped_local_attention(q.numpy(),k.numpy(),v.numpy(),np.arange(12),np.arange(12),5)
        ref=torch.from_numpy(ref.astype('float32')).transpose(1,2).reshape(1,12,24)
        ref=x+m.gain*m.out(ref)
    torch.testing.assert_close(out,ref,atol=3e-6,rtol=3e-6)


def test_numpy_ema_constant_decay():
    writes=np.array([[[[1.0]],[[2.0]],[[3.0]]]])
    out=serial_ema(writes,np.zeros((1,1,1)),np.array([[0.5]]))
    np.testing.assert_allclose(out.ravel(),[0.5,1.25,2.125],atol=1e-12)