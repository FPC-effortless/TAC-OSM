"""NO FIT: original coffee lock, model parity, original five-field step."""
import json,unittest
import torch
from scripts.s1motion006_real_coffee_fit import (
    _load_lock,make_models,ARMS,SEEDS,FIT,DEV,TEST,episode
)
from tac_osm.variants.s1motion005_gradient_budget import compute_gradients,step_model

class SourceAndOriginalPreflight(unittest.TestCase):
    def test_original_25_FIT_only_source_frozen_before_optimizer(self):
        lock=_load_lock()
        self.assertEqual(lock["source_eligibility_run"],40289531876)
        self.assertEqual(len(lock["FIT_receipts"]),25)
        self.assertTrue(all(len(x["sha256"])==64 and x["frames"]==1100
                            for x in lock["FIT_receipts"]))
        self.assertFalse(set(FIT)&set(DEV))
        self.assertFalse(set(TEST)&(set(FIT)|set(DEV)))
        with self.assertRaises(ValueError):
            episode(25,lock,__import__("pathlib").Path("/tmp/no-DEV"),dev_released=False)
        with self.assertRaises(ValueError):
            episode(35,lock,__import__("pathlib").Path("/tmp/no-TEST"),dev_released=True)
    def test_equal_init_and_genuine_point_grad_motion_head_separation(self):
        models=make_models(SEEDS[0],torch.full((14,),.02))
        self.assertEqual(set(models),set(ARMS))
        left,right=(models[a] for a in ARMS)
        for k,v in left.state_dict().items():
            torch.testing.assert_close(v,right.state_dict()[k],atol=0,rtol=0)
        torch.manual_seed(77)
        x=torch.randn(2,5,14)*.2;u=torch.randn_like(x)*.2
        previous=torch.randn(2,14)*.2
        y=x+torch.randn_like(x)*.02
        event=torch.randint(0,2,(2,5)).float()
        batch=(x,u,y,event,previous)
        for model in models.values():
            p,a,combined,meta,point_grads=compute_gradients(
                model,batch,arm="tac_projected")
            self.assertTrue(torch.isfinite(p+a).item())
            self.assertTrue(meta["local_first_order_sgd_nonharm"])
            self.assertEqual(sum(v.numel() for v in model.parameters()),95907)
            self.assertTrue(all(torch.isfinite(z).all().item()
                                for z in combined.values()))
            opt=torch.optim.AdamW(model.parameters(),lr=.001)
            _,details=step_model(model,opt,batch,arm="tac_projected")
            self.assertTrue(details["optimizer_preconditioned_step_verified_finite"])

if __name__=="__main__":unittest.main()
