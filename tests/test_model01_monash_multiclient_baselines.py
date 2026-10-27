"""Constructed tests are mechanics only, not evidence of model skill."""
from datetime import datetime, timedelta
import hashlib,io,json,sys,unittest,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"specialist_models"))
import model01_monash_multiclient_baselines as m

def example_tsf(n=321,bad_holdout=False):
    s=["# Publisher-like simulated FILE (not genuine energy source)",
       "@relation electricity","@attribute series_name string",
       "@attribute start_timestamp date","@frequency hourly",
       "@horizon 48","@missing false","@equallength true","@data"]
    for i in range(n):
        name=f"MT_{i+1:03d}"
        if m.split(name)=="SEALED_TEST" and bad_holdout:
            xs="BAD FUTURE LABEL NEVER ALLOWED TO BE PARSED"
        else:
            xs=",".join(str(4.0+((k+i)%24)*.1) for k in range(700))
        s.append(f"{name}:2012-01-01 00-00-00:{xs}")
    buff=io.BytesIO()
    with zipfile.ZipFile(buff,"w",compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("electricity_hourly_dataset.tsf","\n".join(s)+"\n")
    return buff.getvalue()

class RealSourceContractFixture(unittest.TestCase):
    def test_deterministic_physical_client_disjoint_group_assignment(self):
        names=[f"MT_{i:03d}" for i in range(1,322)]
        self.assertEqual(len(names),len(set(names)))
        a=[m.split(n) for n in names]
        b=[m.split(n) for n in names]
        self.assertEqual(a,b)
        for cat in ("FIT","CAL","DEV","SEALED_TEST"):
            self.assertGreaterEqual(a.count(cat),15)

    def test_only_real_source_frequency_contract_admitted(self):
        meta=["@relation Electricity","@frequency hourly","@missing false",
              "@equallength true","@data"]
        self.assertEqual(m.parse_header(meta)["@frequency"],"hourly")
        with self.assertRaises(m.SourceError):
            m.parse_header([x.replace("hourly","weekly") for x in meta])

    def test_numeric_test_target_never_parsed_in_causal_source_protocol(self):
        data=example_tsf(bad_holdout=True)
        found={"FIT":0,"CAL":0,"DEV":0,"SEALED_TEST":0}
        for arm,item in m.iter_source(data):
            if arm=="HEADER":continue
            found[arm]+=1
            if arm=="SEALED_TEST":
                self.assertIsNone(item[2])
            else:
                self.assertEqual(len(item[2]),700)
        self.assertEqual(sum(found.values()),321)

    def test_fixed_client_disjoint_source_splits_have_references(self):
        result=m.run(example_tsf())
        self.assertEqual(sum(result["source_arm_counts"].values()),321)
        self.assertFalse(result["TEST_customer_numeric_targets_parsed"])
        self.assertFalse(result["model_trained"])
        self.assertEqual(set(result["macro_references"]),{"CAL","DEV"})
        self.assertEqual(set(result["macro_references"]["DEV"]),{"1","24"})
        self.assertTrue(result["macro_references"]["DEV"]["1"]["last_observation"]["mean_client_mae_kw"]>=0)

    def test_reject_duplicate_publisher_client_identity(self):
        v=example_tsf(1)
        # source zip duplicate names not acceptable
        s=["@relation Electricity","@frequency hourly","@missing false","@equallength true","@data"]
        row="MT_111:2012-01-01 00-00-00:"+",".join(["2"]*700)
        s.extend([row,row])
        buff=io.BytesIO()
        with zipfile.ZipFile(buff,"w") as z:z.writestr("s.tsf","\n".join(s))
        with self.assertRaises(m.SourceError):m.run(buff.getvalue())

    def test_missing_source_hourly_value_rejected_when_exposed(self):
        with self.assertRaises(m.SourceError):
            m.parse_line("MT_101:2012-01-01 00-00-00:1,?,2",True)
        n,d,values=m.parse_line("MT_101:2012-01-01 00-00-00:1,?,2",False)
        self.assertEqual(n,"MT_101")
        self.assertIsNone(values)

    def test_reject_negative_nan_and_time_origin_corruption(self):
        for val in ("-1","NaN","inf"):
            with self.assertRaises(m.SourceError):
                m.parse_line("MT_101:2012-01-01 00-00-00:"+",".join([val]*700),True)
        with self.assertRaises(m.SourceError):
            m.parse_line("MT_101:2012-01-XX 00-00-00:"+",".join(["2"]*700),True)

    def test_portugal_dst_source_provenance_veto(self):
        self.assertTrue(m.dst_ambiguous(datetime(2014,3,30,3)))
        self.assertTrue(m.dst_ambiguous(datetime(2014,10,26,0)))
        self.assertFalse(m.dst_ambiguous(datetime(2014,7,1,3)))

    def test_pre_activation_history_not_scored_as_true_zero_usage(self):
        v=[0.0]*300+[5.0]*400
        o=m.baseline_report("MT_TEST",datetime(2012,1,1),v)
        self.assertGreater(o["eligible"]["scored_1"],0)
        self.assertEqual(o["source_baselines"]["1"]["same_hour_previous_week"]["n"],
                         o["source_baselines"]["24"]["same_hour_previous_week"]["n"])

    def test_aggregated_kwh_not_falsely_claimed(self):
        # The target is kW, no conversion to appliance Wh.
        source=m.run(example_tsf())
        self.assertIn("load kW",source["prediction_target"])
        self.assertIn("NOT UCI374 appliance Wh",source["prediction_target"])
        self.assertFalse(source["original_source_available_at_verified"])

    def test_MD5_publisher_version_is_original_not_model_output(self):
        self.assertEqual(m.ZIP_MD5,"18096614662b02640d265ad2a6a416bd")
        self.assertEqual(m.DOI,"10.5281/zenodo.4656140")
        self.assertNotEqual(hashlib.md5(example_tsf()).hexdigest(),m.ZIP_MD5)

if __name__=="__main__":unittest.main()
