"""Constructed-data integration/security tests for local guarded ingress only.

Not a production security assessment, cloud deployment or real-data score.
"""
import hashlib
import http.client
import importlib.util
import json
from pathlib import Path
import secrets
import sys
from threading import Thread
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"specialist_models"))
spec=importlib.util.spec_from_file_location("model01_guarded_ingress",
                                            ROOT/"specialist_models"/"model01_guarded_ingress.py")
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)

class GuardedServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.secret=secrets.token_urlsafe(36)
        cls.sha=hashlib.sha256(cls.secret.encode()).hexdigest()
        cls.server=m.create_server(cls.sha,rate=200,concurrency=4)
        cls.worker=Thread(target=cls.server.serve_forever,daemon=True)
        cls.worker.start()
        cls.request={"schema":"model01-energy-source-aligned-v1",
                     "source_aligned_step_index":555,
                     "last_145_energy_wh":[float(80+(j%9)) for j in range(145)]}

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join(timeout=3)

    def send(self,auth=None,body=None,method="POST",path="/v1/predict",ct="application/json"):
        c=http.client.HTTPConnection("127.0.0.1",self.server.server_port,timeout=4)
        headers={"Content-Type":ct}
        if auth is not None:
            headers["Authorization"]=auth
        try:
            c.request(method,path,body=m.reference.canonical(self.request).encode() if body is None else body,headers=headers)
            response=c.getresponse()
            data=json.loads(response.read())
            return response.status,data,response
        finally:
            c.close()

    def test_fail_closed_missing_credential(self):
        code,out,_=self.send()
        self.assertEqual(code,401)
        self.assertEqual(out["error"],"UNAUTHORIZED")

    def test_wrong_credentials(self):
        for auth in ("Bearer wrong", "Basic abc", "Bearer "+"x"*45):
            with self.subTest(auth=auth[:8]):
                self.assertEqual(self.send(auth=auth)[0],401)

    def test_original_checkpoint_token_authorized(self):
        code,out,resp=self.send(auth="Bearer "+self.secret)
        self.assertEqual(code,200)
        self.assertEqual(out["model_checkpoint_sha256"],m.reference.ORIGINAL_MODEL_SHA256)
        self.assertEqual(out["deployment_status"],"LOCAL_REFERENCE_ONLY")
        self.assertEqual(out["independent_qualification"],"NOT_QUALIFIED")
        self.assertEqual(resp.getheader("Cache-Control"),"no-store")

    def test_authenticated_health_and_no_anonymous_health(self):
        self.assertEqual(self.send(method="GET",path="/health",body=None)[0],401)
        code,data,_=self.send(method="GET",path="/health",body=None,auth="Bearer "+self.secret)
        self.assertEqual(code,200)
        self.assertEqual(data["independent_qualification"],"NOT_QUALIFIED")

    def test_hidden_labels_rejected(self):
        bad=dict(self.request);bad["future_truth_wh"]=999999
        self.assertEqual(self.send(auth="Bearer "+self.secret,body=json.dumps(bad).encode())[0],400)

    def test_nonfinite_and_duplicate_json_rejected(self):
        s=m.reference.canonical(self.request)
        self.assertEqual(self.send(auth="Bearer "+self.secret,body=s.replace("80.0","NaN",1).encode())[0],400)
        duplicate=s.replace('"schema":','"schema":"model01-energy-source-aligned-v1","schema":',1)
        self.assertEqual(self.send(auth="Bearer "+self.secret,body=duplicate.encode())[0],400)

    def test_invalid_mime(self):
        self.assertEqual(self.send(auth="Bearer "+self.secret,ct="text/plain")[0],415)

    def test_oversized_request(self):
        self.assertEqual(self.send(auth="Bearer "+self.secret,body=b"X"*6000)[0],413)

    def test_unknown_route(self):
        self.assertEqual(self.send(auth="Bearer "+self.secret,path="/admin")[0],404)

    def test_rate_limit_is_monotonic_and_deterministic(self):
        t=[200.]
        limiter=m.MinuteLimiter(2,clock=lambda:t[0])
        self.assertTrue(limiter.admit())
        self.assertTrue(limiter.admit())
        self.assertFalse(limiter.admit())
        t[0]=261.
        self.assertTrue(limiter.admit())

    def test_separate_rate_limited_server(self):
        srv=m.create_server(self.sha,rate=1,concurrency=1)
        worker=Thread(target=srv.serve_forever,daemon=True);worker.start()
        def get():
            c=http.client.HTTPConnection("127.0.0.1",srv.server_port,timeout=3)
            try:
                c.request("GET","/health",headers={"Authorization":"Bearer "+self.secret})
                r=c.getresponse();status=r.status;r.read();return status
            finally:c.close()
        try:
            self.assertEqual(get(),200)
            self.assertEqual(get(),429)
        finally:
            srv.shutdown();srv.server_close();worker.join(timeout=3)

    def test_invalid_configuration_denied(self):
        for secret in ("", "0"*64,hashlib.sha256(b"").hexdigest()):
            with self.subTest():
                with self.assertRaises(m.PolicyError):
                    m.create_server(secret)
        with self.assertRaises(m.PolicyError):
            m.create_server(self.sha,rate=0)
        with self.assertRaises(m.PolicyError):
            m.create_server(self.sha,concurrency=1000)

    def test_localhost_bind_only(self):
        self.assertEqual(self.server.server_address[0],"127.0.0.1")

if __name__ == "__main__":
    unittest.main()
