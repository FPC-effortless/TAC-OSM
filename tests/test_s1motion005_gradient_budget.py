"""S1MOTION005 engineering fixtures only; do NOT claim empirical competence."""
import unittest
import torch

from tac_osm.variants.s1motion005_gradient_budget import (
    project_auxiliary, compute_gradients, step_model, RHO, MAX_BETA
)
from tac_osm.variants.system1_velocity_anchored import VelocityAnchoredSystem1

class GradientBudgetContract(unittest.TestCase):
    def test_exact_projection_opposition_zero_and_budget(self):
        point=[torch.tensor([3.,4.],dtype=torch.float64)]
        aux=[torch.tensor([-6.,-8.],dtype=torch.float64)]
        applied,meta=project_auxiliary(point,aux)
        torch.testing.assert_close(applied[0],point[0],atol=0,rtol=0)
        self.assertTrue(meta["conflict"])
        self.assertGreaterEqual(meta["projected_point_dot"],0)
        self.assertLessEqual(meta["beta"],MAX_BETA)

    def test_noncollinear_opposite_gradients_keep_orthogonal_part(self):
        p=[torch.tensor([1.,0.],dtype=torch.float64)]
        a=[torch.tensor([-2.,10.],dtype=torch.float64)]
        applied,meta=project_auxiliary(p,a)
        self.assertAlmostEqual(applied[0][0].item(),1.)
        self.assertLessEqual(abs(applied[0][1].item()), RHO+1e-10)
        self.assertGreater(applied[0][1].item(),0)
        self.assertTrue(meta["local_first_order_sgd_nonharm"])

    def test_zero_point_and_nonfinite_fail_closed(self):
        p=[torch.zeros(2,dtype=torch.float64)]
        a=[torch.tensor([10.,2.],dtype=torch.float64)]
        applied,meta=project_auxiliary(p,a)
        self.assertTrue(torch.equal(applied[0],p[0]))
        self.assertEqual(meta["beta"],0)
        with self.assertRaises(ValueError):
            project_auxiliary([p[0]], [torch.tensor([float("nan"),1.],dtype=torch.float64)])
        with self.assertRaises(ValueError):
            project_auxiliary(p,a,rho=-.2)

    def test_budget_no_added_weights_gradient_route_and_adamw_diagnostics(self):
        torch.manual_seed(29001)
        base=VelocityAnchoredSystem1()
        self.assertEqual(sum(p.numel() for p in base.parameters()),95907)
        self.assertEqual(base.carried_fp32_bytes,664)
        x=torch.randn(2,8,14)*.4
        action=torch.randn(2,8,14)*.3
        previous=torch.randn(2,14)*.2
        y=x+torch.randn_like(x)*.02
        event=torch.randint(0,2,(2,8)).float()
        batch=(x,action,y,event,previous)
        state=base.state_dict()
        models={}
        for arm in ("tac_shared","tac_isolated","tac_projected"):
            m=VelocityAnchoredSystem1()
            m.load_state_dict(state,strict=True)
            models[arm]=m
        for arm,m in models.items():
            point,aux,grad,info,point_grad=compute_gradients(m,batch,arm=arm)
            self.assertTrue(torch.isfinite(point+aux).item())
            self.assertTrue(torch.isfinite(torch.stack([t.abs().max() for t in grad.values()])).all())
            if arm=="tac_isolated":
                gp=grad["core.core.encoder.weight"].clone()
                self.assertGreater(gp.abs().sum().item(),0)
            opt=torch.optim.AdamW(m.parameters(),lr=.001)
            _,out=step_model(m,opt,batch,arm=arm)
            self.assertTrue(out["optimizer_preconditioned_step_verified_finite"])
            self.assertTrue(torch.isfinite(torch.tensor(out["post_point_fit"])).item())
            self.assertIn("actual_point_gradient_dot_parameter_delta",out)
        self.assertEqual(sum(p.numel() for p in models["tac_projected"].parameters()),95907)
        # Same loss forward, same starting weights, different trained gradient only.
        self.assertLessEqual(models["tac_projected"].carried_fp32_bytes,664)


if __name__=="__main__":
    unittest.main()
