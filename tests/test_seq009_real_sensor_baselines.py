"""SEQ009-S source/matrix preflight on toy fixtures ONLY (not research evidence)."""
import tempfile
from pathlib import Path
import unittest

from scripts.run_seq009_real_sensor_baselines import (
    FILES, ARCHIVE_URL, fit_and_evaluate,
)

class TestRealSensorContract(unittest.TestCase):
    def test_only_train_validation_sensor_files_requested(self):
        self.assertIn("/2024-34/3/file/zip",ARCHIVE_URL)
        self.assertEqual(set(FILES),
                         {"train_operational_readouts.csv",
                          "validation_operational_readouts.csv"})
        self.assertNotIn("test_operational_readouts.csv",FILES)

    def test_code_math_and_vehicle_separation(self):
        cols=["vehicle_id","time_step"]+[f"s{i}" for i in range(22)]
        # Synthetic miniature fixtures validate the reader; this has NO
        # standing as evidence of a learned-world-model capability.
        def row(vid,t,offset):
            return ",".join([vid,str(t)]+[str(offset+j) for j in range(22)])+"\n"
        with tempfile.TemporaryDirectory() as td:
            a,b=(Path(td)/x for x in FILES)
            a.write_text(",".join(cols)+"\n"+
                "".join(row(f"a{i}",t,float(i+t))
                        for i in range(120) for t in range(3)))
            b.write_text(",".join(cols)+"\n"+
                "".join(row(f"v{i}",t,float(i+t))
                        for i in range(5) for t in range(3)))
            d=fit_and_evaluate(a,b)
            self.assertEqual(d["feature_count"],22)
            self.assertEqual(d["train_vehicle_count"],120)
            self.assertEqual(d["validation_last_readout_pairs"],5)
            self.assertFalse(d["test_set_opened"])
            self.assertEqual(set(d["metrics"]),{"persistence","affine_autoreg"})
            self.assertTrue(all(x["train_change_scaled_mae_macro"]>=0
                                for x in d["metrics"].values()))

if __name__=="__main__":
    unittest.main()
