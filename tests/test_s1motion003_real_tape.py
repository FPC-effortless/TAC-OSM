"""S1MOTION003 constructed engineering fixtures ONLY, not model capability."""
import unittest
import numpy as np
import torch

from scripts.s1motion003_real_tape_fit import (
    FIT, DEV, SEEDS, MatchedGRU, VelocityAnchoredSystem1,
    prepare, windows, batch, objective, outputs, sha256, REV, SOURCE_SHA256,
)


def recordings():
    rows={}
    for ep in FIT+DEV:
        rows[ep]=[]
        for t in range(257):
            obs=np.arange(14,dtype=float)+ep*.001 + .1*np.sin(t/19+np.arange(14)/5)
            command=np.arange(14,dtype=float)+t*.001+ep*.1
            rows[ep].append((t,t/50,obs,command))
    return rows


class ProspectiveRealTapeContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data=recordings()
        cls.fit,cls.dev,cls.norm=prepare(
            {ep:data[ep] for ep in FIT},{ep:data[ep] for ep in DEV})
        cls.records=data

    def test_source_snapshot_pretraining_frozen(self):
        self.assertEqual(REV,"92955878beb0da15eee5d4d869841577f87e903e")
        self.assertEqual(SOURCE_SHA256,"ff4e1640f3ab7586863332f333dc9341e3cf86b919664c6be90110d37e0ad6d4")
        self.assertEqual(SEEDS,(27001,27002,27003))
        self.assertTrue(set(FIT).isdisjoint(DEV))
        self.assertTrue(set(DEV).isdisjoint(range(35,50)))

    def test_no_future_information_in_logged_t_minus_one_or_velocity(self):
        x=np.stack([r[2] for r in self.records[0]])
        u=np.stack([r[3] for r in self.records[0]])
        xm=np.asarray(self.norm["state_mean"])
        xs=np.asarray(self.norm["state_sd"])
        um=np.asarray(self.norm["action_mean"])
        us=np.asarray(self.norm["action_sd"])
        z=self.fit[0]
        np.testing.assert_allclose(z["previous_x"][0],(x[0]-xm)/xs,rtol=1e-5,atol=1e-5)
        np.testing.assert_allclose(z["x"][0],(x[1]-xm)/xs,rtol=1e-5,atol=1e-5)
        np.testing.assert_allclose(z["prior_u"][0],(u[0]-um)/us,rtol=1e-5,atol=1e-5)
        np.testing.assert_allclose(z["y"][0],(x[2]-xm)/xs,rtol=1e-5,atol=1e-5)

    def test_fit_statistics_not_affected_by_dev_future_edits(self):
        changed=recordings()
        for ep in DEV:
            for i,(fr,t,x,a) in enumerate(changed[ep]):
                changed[ep][i]=(fr,t,x*1e5,a*1e4)
        _,_,new_norm=prepare(
            {ep:changed[ep] for ep in FIT},{ep:changed[ep] for ep in DEV})
        self.assertEqual(new_norm,self.norm)

    def test_same_real_exposure_sampling_and_matched_velocity_priors(self):
        first=windows(self.fit,27001)
        self.assertEqual(first,windows(self.fit,27001))
        self.assertEqual(len(first),192)
        data=batch(self.fit,*first[0])
        self.assertEqual([list(x.shape) for x in data],
                         [[8,32,14],[8,32,14],[8,32,14],[8,32],[8,14]])
        torch.manual_seed(27001)
        scale=torch.tensor(self.norm["train_delta_scale_normalized"],dtype=torch.float32)
        tac=VelocityAnchoredSystem1(); tac.set_train_delta_scale(scale)
        gru=MatchedGRU(scale)
        self.assertEqual(sum(p.numel() for p in tac.parameters()),95907)
        self.assertEqual(sum(p.numel() for p in gru.parameters()),95869)
        self.assertLess(abs(95907-95869)/95907,.01)
        torch.manual_seed(27002)
        for m in (tac,gru):
            pred, _, _=outputs(m,data[0],data[1],data[4])
            velocity=2*data[0]-torch.cat((data[4][:,None],data[0][:,:-1]),1)
            gain_bound=.1*scale
            self.assertTrue(((pred-velocity).abs() <= gain_bound+.000001).all().item())
            loss=objective(m,*data)
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            self.assertTrue(any(p.grad is not None and torch.any(p.grad != 0).item()
                                for p in m.parameters()))


if __name__=="__main__":
    unittest.main()
