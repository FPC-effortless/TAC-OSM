"""Constructed-fixture engineering checks, NOT SCANIA observations or ML results."""
import math
import unittest
import numpy as np
import torch
from tac_osm.decision_sys1_data_002 import (
    FEATURES, CUTOFFS, Readout, FitOnlyStatistics, CausalFeatureStream,
    survival_target, censor_aware_loss, split_of)
from tac_osm.decision_sys1_001 import NativeRiskStateModel


def fit_ids():
    return [str(i) for i in range(100) if split_of(str(i)) == 'fit'][:4]


def row(vid=None, time=0., val=1.):
    return Readout(fit_ids()[0] if vid is None else vid, time, (val,)*105)


def stats():
    s = FitOnlyStatistics(fit_ids())
    s.update(row(val=1.));s.update(row(time=1.,val=3.));s.freeze()
    return s


class AdapterTests(unittest.TestCase):
    def test_01_exact_schema(self):
        self.assertEqual(len(FEATURES),105);self.assertEqual(len(set(FEATURES)),105)
    def test_02_schema_order_independence(self):
        d={k:i for i,k in enumerate(FEATURES)};d.update(vehicle_id='1',time_step=2.)
        self.assertEqual(Readout.from_mapping(d),Readout.from_mapping(dict(reversed(list(d.items())))))
    def test_03_future_label_rejected(self):
        d={k:1. for k in FEATURES};d.update(vehicle_id='1',time_step=1.,in_study_repair=1)
        with self.assertRaises(ValueError):Readout.from_mapping(d)
    def test_04_future_endpoint_rejected(self):
        d={k:1. for k in FEATURES};d.update(vehicle_id='1',time_step=1.,length_of_study_time_step=999.)
        with self.assertRaises(ValueError):Readout.from_mapping(d)
    def test_05_missing_column_rejected(self):
        d={k:1. for k in FEATURES[:-1]};d.update(vehicle_id='1',time_step=1.)
        with self.assertRaises(ValueError):Readout.from_mapping(d)
    def test_06_frozen_split_determinism(self):
        self.assertEqual([split_of(i) for i in range(100)],[split_of(str(i)) for i in range(100)])
    def test_07_dev_transform_fit_rejected(self):
        dev=next(str(i) for i in range(100) if split_of(i)=='dev');s=FitOnlyStatistics(fit_ids())
        with self.assertRaises(ValueError):s.update(row(vid=dev))
        self.assertEqual(s.rows,0)
    def test_08_frozen_transform_rejected(self):
        with self.assertRaises(ValueError):stats().update(row())
    def test_09_empty_transform_rejected(self):
        with self.assertRaises(ValueError):FitOnlyStatistics(fit_ids()).freeze()
    def test_10_all_missing_feature_rejected(self):
        s=FitOnlyStatistics(fit_ids());s.update(Readout(fit_ids()[0],0.,(float('nan'),)+(1.,)*104))
        with self.assertRaises(ValueError):s.freeze()
    def test_11_missing_mean_imputation(self):
        x=stats().transform(Readout('99',2.,(float('nan'),)+(2.,)*104),None)
        self.assertEqual(x.shape,(212,));self.assertEqual(x[0],0.);self.assertEqual(x[105],1.)
        self.assertTrue(np.isfinite(x).all())
    def test_12_elapsed_and_age(self):
        x=stats().transform(row(time=8.),2.)
        self.assertEqual(x[-2],math.log1p(6.));self.assertEqual(x[-1],math.log1p(8.))
    def test_13_first_elapsed_zero(self):
        self.assertEqual(stats().transform(row(time=100.),None)[-2],0.)
    def test_14_negative_or_infinite_time_rejected(self):
        for t in [-1.,float('nan'),float('inf')]:
            with self.subTest(t=t),self.assertRaises(ValueError):row(time=t)
    def test_15_infinite_measurement_rejected(self):
        with self.assertRaises(ValueError):row(val=float('inf'))
    def test_16_bad_identifiers_rejected(self):
        for v in ['01','a','1.0','-1']:
            with self.subTest(v=v),self.assertRaises(ValueError):row(vid=v)
    def test_17_duplicate_or_reverse_clock_rejected(self):
        stream=CausalFeatureStream(stats());stream.transform(row(time=2.))
        for t in [2.,1.]:
            with self.subTest(t=t),self.assertRaises(ValueError):stream.transform(row(time=t))
        self.assertEqual(stream.last_time[fit_ids()[0]],2.)
    def test_18_entity_interleaving(self):
        a,b=CausalFeatureStream(stats()),CausalFeatureStream(stats())
        a.transform(row(time=1.));a.transform(row(vid='99',time=400.))
        b.transform(row(time=1.))
        np.testing.assert_array_equal(a.transform(row(time=2.)),b.transform(row(time=2.)))
    def test_19_future_does_not_change_prefix(self):
        stream=CausalFeatureStream(stats());first=stream.transform(row(time=1.)).copy()
        stream.transform(row(time=200.,val=1e10))
        other=CausalFeatureStream(stats()).transform(row(time=1.))
        np.testing.assert_array_equal(first,other)
    def test_20_receipt_deterministic(self):
        self.assertEqual(stats().receipt()['sha256'],stats().receipt()['sha256'])
    def test_21_moments_independent_numpy_reference(self):
        s=FitOnlyStatistics(fit_ids());xs=[]
        rng=np.random.default_rng(910)
        for t in range(20):
            x=rng.normal(size=105);x[rng.random(105)<.05]=np.nan;xs.append(x)
            s.update(Readout(fit_ids()[0],float(t),tuple(x)))
        s.freeze();xs=np.array(xs)
        np.testing.assert_allclose(s.mean,np.nanmean(xs,0),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(s.scale,np.nanstd(xs,0),rtol=1e-12,atol=1e-12)
    def test_22_actual_native_input_integration(self):
        torch.manual_seed(12);model=NativeRiskStateModel().double()
        state=model.initial(1);stream=CausalFeatureStream(stats())
        for t in range(4):
            x=torch.from_numpy(stream.transform(row(time=float(t)))).unsqueeze(0)
            state=model.observe(state,x);_,lp=model.predict(state,x)
        self.assertTrue(torch.isfinite(lp).all())
        torch.testing.assert_close(lp.exp().sum(-1),torch.ones(1,dtype=torch.float64))


class TargetTests(unittest.TestCase):
    def test_23_event_classes(self):
        for r,c in [(1.,4),(8.,3),(18.,2),(36.,1),(49.,0)]:
            with self.subTest(r=r):self.assertEqual(survival_target(0.,r,1).known_class,c)
    def test_24_exact_cutoffs_not_invented(self):
        for r in CUTOFFS:
            t=survival_target(0.,r,1);self.assertFalse(any(t.observed));self.assertEqual(t.status,'unqualified_exact_boundary')
    def test_25_censor_partial_is_unknown(self):
        t=survival_target(0.,8.,0)
        self.assertEqual(t.observed,(True,False,False,False));self.assertIsNone(t.known_class)
    def test_26_censor_completed_horizons(self):
        for r,n in [(0.,0),(5.,0),(6.,1),(12.,2),(24.,3),(48.,4),(50.,4)]:
            with self.subTest(r=r):self.assertEqual(sum(survival_target(0.,r,0).observed),n)
    def test_27_post_repair_excluded(self):
        for t,end,repair in [(5.,5.,1),(6.,5.,1),(6.,5.,0)]:
            with self.subTest(t=t,end=end),self.assertRaises(ValueError):survival_target(t,end,repair)
    def test_28_invalid_indicators(self):
        for r in [2,-1,float('nan'),1.0,True]:
            with self.subTest(r=r),self.assertRaises(ValueError):survival_target(0.,10.,r)
    def test_29_empty_information_batch_not_silently_fitted(self):
        with self.assertRaises(ValueError):censor_aware_loss(torch.zeros(1,4),[survival_target(0.,1.,0)])
    def test_30_loss_matches_independent_numpy(self):
        z=torch.tensor([[.4,-1.,2.,-.5],[1.,2.,3.,4.]],dtype=torch.float64,requires_grad=True)
        ts=[survival_target(0.,18.,1),survival_target(0.,8.,0)]
        loss=censor_aware_loss(z,ts)
        expected=(np.logaddexp(0.,z.detach().numpy())-z.detach().numpy()*np.array([t.values for t in ts]))
        expected=(expected*np.array([t.observed for t in ts])).sum(-1).mean()
        self.assertAlmostEqual(loss.item(),expected,places=12)
    def test_31_unknown_hazards_have_zero_gradient(self):
        z=torch.zeros(1,4,dtype=torch.float64,requires_grad=True)
        censor_aware_loss(z,[survival_target(0.,8.,0)]).backward()
        self.assertEqual(z.grad.tolist(),[[.5,0.,0.,0.]])
    def test_32_event_loss_matches_class_mass(self):
        z=torch.tensor([[.4,-1.,2.,-.5]],dtype=torch.float64)
        for remaining in [1.,8.,18.,36.,49.]:
            t=survival_target(0.,remaining,1)
            torch.testing.assert_close(censor_aware_loss(z,[t]),-NativeRiskStateModel.log_probabilities(z)[0,t.known_class])
    def test_33_loss_gradcheck(self):
        z=torch.randn(2,4,dtype=torch.float64,requires_grad=True)
        ts=[survival_target(0.,18.,1),survival_target(0.,30.,0)]
        self.assertTrue(torch.autograd.gradcheck(lambda z:censor_aware_loss(z,ts),(z,)))
    def test_34_target_changes_cannot_change_features(self):
        s=stats();r=row(time=1.);before=s.transform(r,None)
        for repaired in (0,1):survival_target(1.,100.,repaired)
        np.testing.assert_array_equal(before,s.transform(r,None))
    def test_35_no_promotion_without_ssm(self):
        from tac_osm.decision_sys1_baselines_002 import mamba2_runtime_status
        self.assertFalse(mamba2_runtime_status()['qualified'])

if __name__=='__main__':unittest.main()
