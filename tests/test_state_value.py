"""ME027 constructed mathematical guards; no capability evidence."""
import types
import torch
from tac_osm.state_value import StateValueModel
from tac_osm.coupled_native import CoupledNative

ARMS=['tac','gru','transformer','mlp']
COUNTS={'tac':135067,'gru':135125,'transformer':134907,'mlp':135057}


def args():
    torch.manual_seed(27027);m=torch.rand(3,128,3);m[:,:,1]=(m[:,:,1]>.5).float()
    return torch.randint(16000,(3,128)),m,torch.randint(16000,(3,)),torch.rand(3,1)


def model(arm):
    torch.manual_seed(27);return StateValueModel(arm,.7,.05,torch.linspace(.2,.8,16000))


def test_counts_active_capacity_and_identical_initial_forecasts():
    torch.set_num_threads(1);a=args()
    for arm in ARMS:
        net=model(arm);assert sum(p.numel() for p in net.parameters())==COUNTS[arm]
        assert abs(COUNTS[arm]-135050)/135050<.01
        pred=net(*a);prior=(a[1][:,-32:,1].sum(-1,keepdim=True)+1)/34
        torch.testing.assert_close(pred,torch.logit(prior),atol=0,rtol=0)
        assert net.observables(*a,128).shape==(3,26)


def test_native_equations_weights_and_embedded_original_head_identity():
    torch.manual_seed(27);old=CoupledNative(0.,True,.7,.05);new=model('tac')
    for name,v in old.state_dict().items():
        if not name.startswith('head.'):assert torch.equal(v,new.state_dict()[name])
    with torch.no_grad():
        old.head.weight.normal_(std=.1);new.head.weight[:,:73].copy_(old.head.weight);new.head.bias.copy_(old.head.bias)
    a=args();torch.testing.assert_close(old(*a),new(*a),atol=1e-6,rtol=0)


def test_gru_cell_sequence_matches_standard_torch_gru():
    net=model('gru');seq=torch.nn.GRU(11,34,batch_first=True)
    with torch.no_grad():
        for name,p in net.cell.named_parameters():getattr(seq,name+'_l0').copy_(p)
    x=torch.randn(3,128,11);state=torch.zeros(3,34)
    for t in range(128):state=net.advance(x[:,t],state)
    expected=seq(x)[1][0];torch.testing.assert_close(state,expected,atol=1e-6,rtol=0)


def test_shared_observables_equal_for_identical_embedding_tables():
    a=args();models=[model(x) for x in ARMS]
    expected=models[0].observables(*a,128)
    for m in models[1:]:torch.testing.assert_close(m.observables(*a,128),expected,atol=0,rtol=0)
    pooled=torch.cat((models[0].emb(a[0]).mean(1),a[1].mean(1)),-1)
    torch.testing.assert_close(expected[:,-11:],pooled,atol=0,rtol=0)


def causal_error(net,a):
    before=net.prefix_logits(*a);errors=[]
    for col,k in enumerate([32,64,96]):
        meta=a[1].clone();meta[:,k:,1]=1-meta[:,k:,1]
        items=a[0].clone();items[:,k+1:]=(items[:,k+1:]+101)%16000
        after=net.prefix_logits(items,meta,a[2],a[3])
        errors.append(float((before[:,col]-after[:,col]).abs().max().detach()))
    return max(errors)


def test_active_head_prefix_causality_and_attention_negative_control():
    a=args()
    for arm in ARMS:
        net=model(arm).eval()
        with torch.no_grad():net.head.weight.normal_(std=.1)
        assert causal_error(net,a)<=3e-6
        if arm=='transformer':
            def unmasked(self,items,meta):
                v=self.in_proj(torch.cat((self.emb(items),meta),-1))+self.positions.unsqueeze(0)
                return self.to_state(self.final_norm(self.blocks(v)))
            net.attention_states=types.MethodType(unmasked,net)
            assert causal_error(net,a)>1e-4


def test_finite_active_gradients_and_jit_persistence(tmp_path):
    a=args()
    for arm in ARMS:
        net=model(arm)
        with torch.no_grad():net.head.weight.normal_(std=.1)
        scripted=torch.jit.script(net);logits=scripted.prefix_logits(*a)
        loss=logits.square().mean();loss.backward()
        for name,p in scripted.named_parameters():assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,name
        with torch.no_grad():torch.testing.assert_close(logits[:,-1:],scripted(*a),atol=1e-6,rtol=0)
        path=tmp_path/f'{arm}.pt';torch.jit.save(scripted,str(path))
        with torch.no_grad():torch.testing.assert_close(torch.jit.load(str(path))(*a),scripted(*a),atol=0,rtol=0)


def test_post_normalization_feature_mean_preserves_latent_logit_average():
    a=args()
    for arm in ['tac','gru','transformer']:
        net=model(arm).eval()
        with torch.no_grad():
            net.head.weight.normal_(std=.1);latent=net.latent_features(a[0],a[1]);common=net.observables(*a,128)
            prior=(a[1][:,-32:,1].sum(-1,keepdim=True)+1)/34
            original=net.predict_features(latent,common,prior)
            replaced=net.predict_features(latent.mean(0,keepdim=True).expand_as(latent),common,prior)
            torch.testing.assert_close(original.mean(),replaced.mean(),atol=1e-6,rtol=0)
