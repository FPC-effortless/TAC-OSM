import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from tac_osm.decision_sys1_data_002 import FEATURES

spec=importlib.util.spec_from_file_location('source_audit',Path(__file__).resolve().parents[1]/'scripts/audit_decision_sys1_train_source_002.py')
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)

class SourceAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.r=self.root/'train_operational_readouts.csv';self.y=self.root/'train_tte.csv'
        with self.r.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['vehicle_id','time_step',*FEATURES]);w.writeheader()
            for vid,t in [('1',0.),('1',2.),('2',0.),('2',3.)]:
                w.writerow(dict(vehicle_id=vid,time_step=t,**{k:1. for k in FEATURES}))
        with self.y.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['vehicle_id','length_of_study_time_step','in_study_repair'])
            w.writerows([['1',20.,1],['2',5.,0]])
    def tearDown(self):self.tmp.cleanup()
    def test_01_fixture_audit_not_training_admission(self):
        result=source.audit(self.r,self.y,self.root/'out',source.sha256(self.r))
        self.assertEqual(result['rows'],4);self.assertFalse(result['training_admitted'])
    def test_02_hash_failure_precedes_labels(self):
        with self.assertRaises(ValueError):source.audit(self.r,self.y,self.root/'out')
        self.assertFalse((self.root/'out/split_before_labels.json').exists())
    def test_03_split_written_even_if_label_missing(self):
        self.y.unlink()
        with self.assertRaises(FileNotFoundError):source.audit(self.r,self.y,self.root/'out',source.sha256(self.r))
        self.assertTrue((self.root/'out/split_before_labels.json').exists())
    def test_04_no_test_names(self):
        with self.assertRaises(ValueError):source.audit(self.root/'test_operational_readouts.csv',self.y,self.root/'out')
    def test_05_outcome_join_guard(self):
        s=self.y.read_text().replace('2,5.0,0','999,5.0,0');self.y.write_text(s)
        with self.assertRaises(ValueError):source.audit(self.r,self.y,self.root/'out',source.sha256(self.r))
    def test_06_no_empty_data(self):
        self.r.write_text(','.join(['vehicle_id','time_step',*FEATURES])+'\n')
        with self.assertRaises(ValueError):source.audit(self.r,self.y,self.root/'out',source.sha256(self.r))

if __name__=='__main__':unittest.main()
