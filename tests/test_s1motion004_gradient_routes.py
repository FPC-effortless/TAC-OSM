"""S1MOTION004 synthetic numeric fixtures ONLY; do not cite as real skill."""
import unittest
import torch

from tac_osm.variants.s1motion004_gradient_routes import (
    IsolatedVelocitySystem1,VelocityGRUControl,losses,outputs
)
from tac_osm.variants.system1_velocity_anchored import VelocityAnchoredSystem1


class PointGradientIsolationContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42001)
        self.scale=torch.full((14,),.17)
        self.x=torch.randn(3,9,14)*.3
        self.u=torch.randn(3,9,14)*.2
        self.prev=torch.randn(3,14)*.3
        self.y=self.x+torch.randn_like(self.x)*.08
        self.event=torch.randint(0,2,(3,9)).float()
        self.tac=VelocityAnchoredSystem1();self.tac.set_train_delta_scale(self.scale)
        self.iso=IsolatedVelocitySystem1()
        self.iso.load_state_dict(self.tac.state_dict())
        self.gru=VelocityGRUControl(self.scale,False)
        self.gru_iso=VelocityGRUControl(self.scale,True)
        self.gru_iso.load_state_dict(self.gru.state_dict())

    def batch(self):
        return self.x,self.u,self.y,self.event,self.prev

    def test_pairwise_same_initial_weights_and_prediction_values(self):
        for a,b in ((self.tac,self.iso),(self.gru,self.gru_iso)):
            self.assertEqual(sum(p.numel() for p in a.parameters()),
                             sum(p.numel() for p in b.parameters()))
            for k,v in a.state_dict().items():
                torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
            got=outputs(a,self.x,self.u,self.prev)
            guarded=outputs(b,self.x,self.u,self.prev)
            for v,w in zip(got,guarded):
                torch.testing.assert_close(v,w,rtol=0,atol=1e-6)
            ll,_=losses(a,*self.batch(),isolated=False)
            rr,_=losses(b,*self.batch(),isolated=True)
            torch.testing.assert_close(ll,rr,rtol=1e-6,atol=1e-6)

    def test_tac_risk_and_uncertainty_gradients_blocked_from_memory(self):
        for model,isolated in ((self.tac,False),(self.iso,True)):
            total,terms=losses(model,*self.batch(),isolated=isolated)
            auxiliary=.1*terms["proxy_bce"]+.01*terms["dispersion_nll"]
            core=model.core.core.memory_gates.weight
            aux_head=model.core.reactive_head[0].weight
            core_aux=torch.autograd.grad(auxiliary,core,allow_unused=True,
                                         retain_graph=True)[0]
            aux_grad=torch.autograd.grad(auxiliary,aux_head,
                                         retain_graph=True)[0]
            point_grad=torch.autograd.grad(terms["point"],core,
                                          allow_unused=True,retain_graph=True)[0]
            self.assertIsNotNone(aux_grad)
            self.assertGreater(aux_grad.abs().sum().item(),0)
            self.assertIsNotNone(point_grad)
            self.assertGreater(point_grad.abs().sum().item(),0)
            if isolated:
                self.assertIsNone(core_aux)
            else:
                self.assertIsNotNone(core_aux)
                self.assertGreater(core_aux.abs().sum().item(),0)

    def test_gru_recurrent_auxiliary_gradient_isolation(self):
        for model,isolated in ((self.gru,False),(self.gru_iso,True)):
            _,terms=losses(model,*self.batch(),isolated=isolated)
            auxiliary=.1*terms["proxy_bce"]+.01*terms["dispersion_nll"]
            recurrent=model.recurrent.weight_ih_l0
            routed=torch.autograd.grad(auxiliary,recurrent,
                                       retain_graph=True,allow_unused=True)[0]
            point=torch.autograd.grad(terms["point"],recurrent,
                                     retain_graph=True)[0]
            self.assertGreater(point.abs().sum().item(),0)
            if isolated:
                self.assertIsNone(routed)
            else:
                self.assertGreater(routed.abs().sum().item(),0)

    def test_no_future_observations_can_change_original_prefix(self):
        with torch.no_grad():
            for model in (self.tac,self.iso,self.gru,self.gru_iso):
                original=outputs(model,self.x,self.u,self.prev)
                changed_x=self.x.clone();changed_u=self.u.clone()
                changed_x[:,5:]+=123;changed_u[:,5:]-=88
                changed=outputs(model,changed_x,changed_u,self.prev)
                for before,after in zip(original,changed):
                    torch.testing.assert_close(before[:,:5],after[:,:5],
                                               rtol=0,atol=1e-6)

    def test_exact_velocity_bound_and_honest_state(self):
        with torch.no_grad():
            for model in (self.tac,self.iso,self.gru,self.gru_iso):
                pred,_,_=outputs(model,self.x,self.u,self.prev)
                v=2*self.x-torch.cat((self.prev[:,None],self.x[:,:-1]),1)
                self.assertTrue(((pred-v).abs()<=.1*self.scale+1e-6).all().item())
        self.assertEqual(self.tac.carried_fp32_bytes,664)
        self.assertEqual(self.iso.carried_fp32_bytes,664)
        self.assertEqual(sum(p.numel() for p in self.iso.parameters()),95907)
        self.assertEqual(sum(p.numel() for p in self.gru.parameters()),95869)

    def test_isolated_stream_and_batch_agree(self):
        with torch.no_grad():
            forecast=self.iso(self.x,self.u,
                 initial_previous_observation=self.prev)
            state=self.iso.initial_state(3,previous_observation=self.prev)
            recorded=[]
            for t in range(self.x.shape[1]):
                o,state=self.iso.step(state,self.x[:,t],self.u[:,t])
                recorded.append(o)
            for name in ("next_observation","adverse_event_logit","residual_scale"):
                sequence=torch.stack([getattr(v,name) for v in recorded],1)
                torch.testing.assert_close(sequence,getattr(forecast,name),
                                           rtol=1e-5,atol=1e-5)


if __name__=="__main__":
    unittest.main()
