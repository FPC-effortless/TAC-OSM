"""No-fit KV mathematical correspondence; not a trained-language score."""
import math,pytest,torch
from torch import nn
from scripts.langspeed044_cached import KV

class RegisteredShape(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(256,96)
        layer=nn.TransformerEncoderLayer(d_model=96,nhead=4,dim_feedforward=384,
            dropout=0.,batch_first=True,norm_first=True)
        self.layers=nn.TransformerEncoder(layer,num_layers=4,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(96);self.head=nn.Linear(96,256)
    def forward(self,tokens):
        t=tokens.shape[1]
        pos=torch.arange(t,dtype=self.embedding.weight.dtype)
        inv=torch.exp(-math.log(10000.)*torch.arange(0,96,2,dtype=pos.dtype)/96)
        a=pos[:,None]*inv[None,:]
        pe=torch.stack((a.sin(),a.cos()),-1).reshape(t,96)
        mask=torch.ones(t,t,dtype=torch.bool).triu(1)
        return self.head(self.norm(self.layers(self.embedding(tokens)+pe[None],mask=mask)))

@pytest.mark.parametrize("length",(1,8,32,128))
def test_cache_is_numerically_equal_to_original_structure(length):
    torch.manual_seed(44044);torch.set_num_threads(2)
    model=RegisteredShape().eval();kv=KV(model)
    tokens=torch.randint(0,256,(1,length))
    with torch.no_grad():
        y,cache,n=kv.prefill(tokens)
        assert torch.allclose(y,model(tokens)[:,-1],atol=3e-5,rtol=1e-4)
        for step in range(12):
            token=y.argmax(-1)
            prior=cache[0][0].clone()
            y,cache,n=kv.step(token,cache,n)
            tokens=torch.cat((tokens,token[:,None]),dim=1)
            assert torch.allclose(y,model(tokens)[:,-1],atol=3e-5,rtol=1e-4)
            assert torch.equal(cache[0][0][:,:,:-1],prior)
            assert kv.cache_bytes(cache)==4*2*4*n*24*4

def test_cache_rejects_bad_input():
    model=RegisteredShape().eval();kv=KV(model)
    with torch.no_grad():
        _,cache,n=kv.prefill(torch.tensor([[15,16]],dtype=torch.long))
        with pytest.raises(ValueError):kv.step(torch.tensor([256]),cache,n)
        with pytest.raises(ValueError):kv.step(torch.tensor([1]),cache,n+1)
