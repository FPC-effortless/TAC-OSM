import importlib.util
import io
import hashlib
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('acquire003', Path(__file__).resolve().parents[1]/'scripts/acquire_decision_sys1_train_003.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class AcquisitionGuards(unittest.TestCase):
    def test_allowed_urls(self):
        for name in m.ALLOWED:
            m.check_url(f'https://api.researchdata.se/dataset/2024-34/3/file/data/{name}', name)
    def test_publisher_redirect_exact_host_path_pair(self):
        for name in m.ALLOWED:
            m.check_url(f'https://scania.open.care.snd.se/downloads/public/2024-34/3/data/{name}',name)
        for url in ('https://scania.open.care.snd.se/dataset/2024-34/3/file/data/train_tte.csv',
                    'https://api.researchdata.se/downloads/public/2024-34/3/data/train_tte.csv',
                    'https://scania.open.care.snd.se/downloads/public/2024-34/3/data/test_labels.csv'):
            with self.assertRaises(ValueError):m.check_url(url,'train_tte.csv')
    def test_wrong_resource_rejected(self):
        base='https://api.researchdata.se/dataset/2024-34/3/file/data/'
        for name in ['test_labels.csv','validation_operational_readouts.csv',
                     'train_specifications.csv','../train_tte.csv']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                m.check_url(base+name,name)
    def test_unsafe_redirects_rejected(self):
        urls=['http://api.researchdata.se/dataset/2024-34/3/file/data?filePath=train_tte.csv',
              'https://evil.test/api/file/2024-34/3/data?filePath=train_tte.csv',
              'https://doris.snd.se/api/file/2024-34/3/data?filePath=test_labels.csv',
              'https://api.researchdata.se/dataset/2024-34/3/file/zip',
              'https://doris.snd.se/api/file/2024-34/3/data?filePath=train_tte.csv&extra=1',
              'https://user:pass@doris.snd.se/api/file/2024-34/3/data?filePath=train_tte.csv']
        for url in urls:
            with self.subTest(url=url), self.assertRaises(ValueError):
                m.check_url(url,'train_tte.csv')
    def test_other_file_fails_before_network(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            m.acquire('test_labels.csv',Path(d),[])
    def test_old_query_routes_rejected(self):
        with self.assertRaises(ValueError):
            m.check_url('https://api.researchdata.se/dataset/2024-34/3/file/data?filePath=train_tte.csv','train_tte.csv')
    def test_fake_html_not_accepted(self):
        class Response(io.BytesIO):
            status=200
            headers={'Content-Type':'text/html'}
            def geturl(self):return 'https://api.researchdata.se/dataset/2024-34/3/file/data/train_tte.csv'
        class Opener:
            def open(self,*a,**k):return Response(b'<html>no data</html>')
        with tempfile.TemporaryDirectory() as d:
            attempts=[]
            with self.assertRaises(RuntimeError):m.acquire('train_tte.csv',Path(d),attempts,lambda _:Opener())
            self.assertEqual(len(attempts),1)
            self.assertFalse((Path(d)/'train_tte.csv').exists())
    def test_valid_fixture_hash_not_independent_attestation(self):
        class Response(io.BytesIO):
            status=200
            headers={'Content-Type':'text/csv'}
            def geturl(self):return 'https://api.researchdata.se/dataset/2024-34/3/file/data/train_tte.csv'
        class Opener:
            def open(self,*a,**k):return Response(b'vehicle_id,length_of_study_time_step,in_study_repair\n1,10,0\n')
        fixture=b'vehicle_id,length_of_study_time_step,in_study_repair\n1,10,0\n'
        with tempfile.TemporaryDirectory() as d, patch.dict(m.ALLOWED, {'train_tte.csv': (1000,hashlib.sha256(fixture).hexdigest())}):
            result=m.acquire('train_tte.csv',Path(d),[],lambda _:Opener())
            self.assertEqual(result['status'],'ACQUIRED')
            self.assertIn('not_independent',result['hash_attestation'])

if __name__=='__main__':unittest.main()
