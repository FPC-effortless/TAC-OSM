import torch
from vectorized_memory import retrieve, admission

def test_retrieval():
    torch.manual_seed(31)
    for b in (1,3,17):
        ids=torch.randint(-1,6,(b,8)); keys=torch.randn(b,8,16,requires_grad=True)
        values=torch.randn(b,8,3,16,requires_grad=True)
        occupied=torch.rand(b,8,3)>.4; query=torch.randn(b,16,requires_grad=True)
        req=torch.randint(0,6,(b,))
        for role in range(3):
            for requested in (None,req):
                got,hits=retrieve(ids,keys,values,occupied,query,role,requested)
                expected=[]; expected_hit=[]
                for i in range(b):
                    mask=occupied[i,:,role]&(ids[i]>=0)
                    if requested is not None: mask=mask&(ids[i]==requested[i])
                    expected_hit.append(bool(mask.any()))
                    if mask.any():
                        w=torch.softmax(keys[i,mask]@torch.nn.functional.normalize(query[i],dim=0,eps=1e-6),0)
                        expected.append((w[:,None]*values[i,mask,role]).sum(0))
                    else: expected.append(torch.zeros_like(query[i]))
                assert torch.equal(hits,torch.tensor(expected_hit))
                assert torch.allclose(got,torch.stack(expected),atol=1e-6,rtol=1e-6)
                if got.requires_grad: torch.autograd.grad(got.sum(),(keys,values,query),allow_unused=True)

def test_admission():
    ids=torch.tensor([[1,2,-1,-1],[1,2,3,4],[-1,-1,-1,-1]])
    req=torch.tensor([2,5,7]);enabled=torch.tensor([True,True,False])
    new,mask,ok=admission(ids,req,enabled)
    assert ok.tolist()==[True,False,False]
    assert torch.equal(new,ids)
    req=torch.tensor([9,5,7]);enabled=torch.tensor([True,True,True])
    new,mask,ok=admission(ids,req,enabled)
    assert ok.tolist()==[True,False,True]
    assert new.tolist()==[[1,2,9,-1],[1,2,3,4],[7,-1,-1,-1]]

if __name__=='__main__':
    test_retrieval();test_admission();print('PASS')
