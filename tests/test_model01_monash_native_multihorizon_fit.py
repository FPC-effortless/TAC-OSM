"""Constructed mechanics only; performance must use genuine Monash source CI."""
import datetime as dt
import os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]/"specialist_models"
sys.path.insert(0,str(ROOT))
import model01_monash_native_multihorizon_fit as m

def fixture():
    names=[f"MT_{i:03d}" for i in range(1,20)]
    t0=dt.datetime(2012,1,1)
    values=[3.+.02*(i%24) for i in range(800)]
    entries={name:(t0,list(values)) for name in names}
    return {"FIT":entries,"CAL":{},"DEV":{}}

class NativeCustomerTransferMechanics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
    def test_param_budget_and_native_two_horizon_graph(self):
        net=m.MultiHorizonTAC()
        self.assertEqual(m.ancestral.params(net),103381)
        self.assertTrue(95000<=m.ancestral.params(net)<=105000)
        x=torch.randn(3,48,3)
        y=net(x)
        self.assertEqual(y.shape,(3,2))
        y.square().mean().backward()
        self.assertIsNotNone(net.key.weight.grad)
        self.assertIsNotNone(net.sgate.weight.grad)
        self.assertIsNotNone(net.head[-1].weight.grad)
    def test_only_prior_hourly_context_in_input(self):
        series=list(range(350))
        x=m.input_window(series,175,4.,2.)
        self.assertEqual(x.shape,(48,3))
        self.assertTrue(np.all(np.isfinite(x)))
        changed=list(series);changed[199]=999999
        self.assertTrue(np.array_equal(x,m.input_window(changed,175,4.,2.)))
    def test_hourly_client_eligibility_never_accepts_future(self):
        c=fixture()["FIT"][0] if isinstance(fixture()["FIT"],list) else list(fixture()["FIT"].values())[0]
        t0,values=c
        self.assertTrue(m.eligible(t0,values,300,1))
        self.assertTrue(m.eligible(t0,values,300,24))
        self.assertFalse(m.eligible(t0,values,795,24))
    def test_sampler_reads_only_fit_named_clients(self):
        clients=fixture()
        clients["CAL"]={"SENTINEL":(dt.datetime(2012,1,1),[99.]*800)}
        clients["DEV"]={"HIDDEN":(dt.datetime(2012,1,1),[1000.]*800)}
        gen=np.random.default_rng(1801)
        x,y,receipts=m.training_batch(clients,gen,1.,1.)
        self.assertEqual(x.shape,(64,48,3))
        self.assertEqual(y.shape,(64,2))
        self.assertTrue(all(s.startswith("MT_") for s in receipts))
        self.assertFalse(any("SENTINEL" in s or "HIDDEN" in s for s in receipts))
    def test_scaler_uses_only_fit_clients(self):
        clients=fixture()
        before=m.fit_scaling(clients)
        clients["CAL"]={"FUTURE":(dt.datetime(2012,1,1),[1e6]*800)}
        clients["DEV"]={"UNSEEN":(dt.datetime(2012,1,1),[1e7]*800)}
        self.assertEqual(m.fit_scaling(clients),before)
    def test_invalid_short_history_blocked(self):
        with self.assertRaises(m.CapabilityError):m.input_window([1]*10,4,2.,1.)
    def test_fit_gate_precedes_zenodo_fetch(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.dict(os.environ,{},clear=True):
                with patch.object(m,"source_data",side_effect=AssertionError("must not fetch")):
                    self.assertEqual(m.main(["--out",str(Path(d)/"out.json")]),2)
    def test_no_sealed_target_path_in_sample_contract(self):
        clients=fixture();clients["SEALED_TEST"]={"HIGHLY_SECRET":(dt.datetime(2012,1,1),[1e99]*800)}
        x,y,receipts=m.training_batch(clients,np.random.default_rng(1),1.,1.)
        self.assertEqual(len(x),64)
        self.assertTrue(all("HIGHLY_SECRET" not in r for r in receipts))
    def test_source_baseline_fair_eligibility_per_horizon(self):
        c=fixture()["FIT"];name,tup=next(iter(c.items()))
        orig={"per_client_CAL_DEV":{"DEV":[m.source.baseline_report(name,*tup)]}}
        data=m.eval_batches({"DEV":{name:tup}},"DEV",*m.fit_scaling({"FIT":c}),orig)
        self.assertEqual(len(data),1)
        obj=data[name]
        self.assertEqual(obj["x"].shape,(336,48,3))
        for i,h in enumerate(m.HORIZONS):
            self.assertEqual(int(obj["mask"][:,i].sum()),
                             obj["source_baselines"][str(h)]["last_observation"]["n"])

if __name__=="__main__":unittest.main()
