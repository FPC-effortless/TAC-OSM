"""Construction-only fixtures. No synthetic results count toward capability."""
import inspect
import unittest

import numpy as np
import torch
from scripts.s1motion004_real_towel_fit import (
    REV,BLOB_SHA,ARMS,SEEDS,FIT,DEV,WINDOW,BATCH,UPDATES,
    encode_source_episode,prepare_fit_only,initialization,real_source
)
from tac_osm.variants.s1motion004_gradient_routes import (
    losses,outputs
)


def fixtures():
    records={}
    for ep in FIT:
        rows=[]
        for t in range(257):
            state=np.arange(14,dtype=float)*.05+ep*.01+ .2*np.sin(.023*t+np.arange(14)*.1)
            action=np.arange(14,dtype=float)*.04+ep*.03+.02*np.cos(.02*t)
            rows.append((t,t/50,state,action))
        records[ep]=rows
    return records


class RealSourceDesignContracts(unittest.TestCase):
    def test_source_immutability_and_dev_reveal_opt_in(self):
        self.assertEqual(REV,"13ad96f5ed0e219e48c72471626a2e3d0eb5aeff")
        self.assertEqual(BLOB_SHA,"aa61a929598903401e10d9b677bd2f2df480da0c912b6a85a27541028155fa1b")
        self.assertFalse(inspect.signature(real_source).parameters["release_dev"].default)
        self.assertEqual(len(FIT),25)
        self.assertEqual(len(DEV),10)
        self.assertTrue(set(FIT).isdisjoint(DEV))
        self.assertTrue(set(DEV).isdisjoint(range(35,50)))
        self.assertEqual(len(ARMS),4)
        self.assertEqual(SEEDS,(28001,28002,28003))
        self.assertEqual((WINDOW,BATCH,UPDATES),(32,8,192))

    def test_only_fit_stats_and_prior_logged_action_are_causal(self):
        real=fixtures()
        train,norm=prepare_fit_only(real)
        self.assertEqual(set(train),set(FIT))
        ep=real[0]
        x=np.stack([r[2] for r in ep])
        u=np.stack([r[3] for r in ep])
        xm=np.asarray(norm["state_mean"]);xs=np.asarray(norm["state_sd"])
        um=np.asarray(norm["action_mean"]);us=np.asarray(norm["action_sd"])
        z=train[0]
        np.testing.assert_allclose(z["previous_x"][0],(x[0]-xm)/xs,atol=1e-6)
        np.testing.assert_allclose(z["x"][0],(x[1]-xm)/xs,atol=1e-6)
        np.testing.assert_allclose(z["prior_u"][0],(u[0]-um)/us,atol=1e-6)
        np.testing.assert_allclose(z["y"][0],(x[2]-xm)/xs,atol=1e-6)
        self.assertEqual(z["episode_frames"],255)
        self.assertGreater(norm["high_motion_threshold_raw_squared_units"],0)

    def test_pair_initialization_exact_weights_scalar_equal(self):
        scale=torch.full((14,),.1)
        models=initialization(SEEDS[0],scale)
        for prefix in ("tac","gru"):
            a=models[prefix+"_shared"];b=models[prefix+"_isolated"]
            for k,v in a.state_dict().items():
                torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)
        train,norm=prepare_fit_only(fixtures())
        x=train[0]["x"][:32][None]
        u=train[0]["prior_u"][:32][None]
        y=train[0]["y"][:32][None]
        e=train[0]["event"][:32][None]
        old=train[0]["previous_x"][:1]
        for prefix in ("tac","gru"):
            a=models[prefix+"_shared"];b=models[prefix+"_isolated"]
            aa,_=losses(a,x,u,y,e,old,isolated=False)
            bb,_=losses(b,x,u,y,e,old,isolated=True)
            torch.testing.assert_close(aa,bb,atol=1e-6,rtol=1e-6)


if __name__=="__main__":
    unittest.main()
