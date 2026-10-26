"""S1MOTION005 source and model engineering-only constructed fixtures."""
import json
import unittest
from pathlib import Path
import torch
from scripts.s1motion005_real_screwdriver_fit import (
    FIT,DEV,SEEDS,ARMS,ROOT,LOCK,original_models,episode_data,
    approved_source_metadata
)
from tac_osm.variants.s1motion005_gradient_budget import (
    compute_gradients,RHO,MAX_BETA
)
class OriginalSourceAndBudgets(unittest.TestCase):
    def test_real_source_FIT_SHA_frozen_and_PROTECTED_sealed(self):
        lock=json.loads(LOCK.read_text())
        approved_source_metadata(lock)
        self.assertEqual((len(FIT),len(DEV)),(25,10))
        self.assertEqual(lock["revision_sha"],"0b873be59c9702f8d3de89022617121970db112f")
        self.assertEqual(len({r["sha256"] for r in lock["fit_files"]}),25)
        self.assertEqual(SEEDS,(29001,29002,29003))
        self.assertEqual(len(ARMS),5)
        self.assertEqual(set(FIT)&set(DEV),set())
        with self.assertRaises(ValueError):
            episode_data(35,lock,Path("/tmp/do-not-use"),expected_sha=None)

    def test_original_pair_initial_weights_and_deployed_footprint(self):
        models=original_models(29001,torch.ones(14)*.2)
        for other in ("tac_isolated","tac_projected"):
            for k,v in models["tac_shared"].state_dict().items():
                torch.testing.assert_close(v,models[other].state_dict()[k],
                                           rtol=0,atol=0)
        for k,v in models["gru_shared"].state_dict().items():
            torch.testing.assert_close(v,models["gru_isolated"].state_dict()[k],
                                       rtol=0,atol=0)
        for arm in ("tac_shared","tac_isolated","tac_projected"):
            m=models[arm]
            self.assertEqual(m.carried_fp32_bytes,664)
            self.assertEqual(sum(x.numel() for x in m.parameters()),95907)
        for arm in ("gru_shared","gru_isolated"):
            self.assertEqual(sum(x.numel() for x in models[arm].parameters()),95869)
        self.assertEqual((RHO,MAX_BETA),(.1,.25))

    def test_all_TAC_gradient_modes_same_forward_point_and_aux_pre_fit(self):
        torch.manual_seed(29001)
        state=original_models(29001,torch.ones(14)*.2)
        x=torch.randn(2,6,14)*.2
        u=torch.randn(2,6,14)*.1
        old=torch.randn(2,14)*.1
        y=x+torch.randn_like(x)*.01
        event=torch.randint(0,2,(2,6)).float()
        observation=[]
        for arm in ("tac_shared","tac_isolated","tac_projected"):
            p,a,grad,meta,pg=compute_gradients(state[arm],
                    (x,u,y,event,old),arm=arm)
            observation.append((float(p.detach()),float(a.detach())))
            self.assertTrue(all(torch.isfinite(v).all().item() for v in grad.values()))
            self.assertTrue(all(torch.isfinite(v).all().item() for v in pg.values()))
        self.assertEqual(observation[0],observation[1])
        self.assertEqual(observation[0],observation[2])


if __name__=="__main__":
    unittest.main()
