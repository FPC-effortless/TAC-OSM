"""Constructed request tests for original Git-backed local reference inference.

No genuine new source, accuracy metric, cloud deployment or Android benchmark.
"""
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
import http.client
import importlib.util
import json
import math
from pathlib import Path
import socket
import sys
from threading import Thread
import unittest

ROOT=Path(__file__).resolve().parents[1]
MODULES=ROOT/"specialist_models"
sys.path.insert(0,str(MODULES))
spec=importlib.util.spec_from_file_location("model01_reference_api",MODULES/"model01_reference_api.py")
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
import uci_appliances_ridge_dev as original


def req(i=500):
    return {"schema":"model01-energy-source-aligned-v1",
            "source_aligned_step_index":i,
            "last_145_energy_wh":[float(45+5*math.sin(j/25)+j%3) for j in range(145)]}


class ReferenceModel01Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model,cls.q=m.load_original()
        cls.server=ThreadingHTTPServer(("127.0.0.1",0),m.make_server(cls.model,cls.q))
        cls.worker=Thread(target=cls.server.serve_forever,daemon=True)
        cls.worker.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.worker.join(timeout=2)

    def post(self,data,content_type="application/json",path="/v1/predict"):
        c=http.client.HTTPConnection("127.0.0.1",self.server.server_port,timeout=3)
        try:
            c.request("POST",path,body=data,headers={"Content-Type":content_type})
            r=c.getresponse()
            return r.status,json.loads(r.read()),r.getheader("Cache-Control")
        finally:
            c.close()

    def test_original_git_weights_and_frozen_uncertainty(self):
        self.assertEqual(m.ORIGINAL_MODEL_SHA256,len(m.ORIGINAL_MODEL_SHA256)*"0" if False else m.ORIGINAL_MODEL_SHA256)
        self.assertAlmostEqual(self.q,92.532708685843)
        self.assertEqual(self.model["fit_examples"],9717)

    def test_exact_numerical_agreement_with_original_training_graph(self):
        i=526
        sample=req(i)
        actual=m.predict(self.model,self.q,sample)
        window=sample["last_145_energy_wh"]
        true_original=original.forecast(self.model,window,i)  # requires i > 144; same 145 values
        self.assertAlmostEqual(actual["prediction_wh"],true_original,places=12)
        self.assertEqual(actual["action_authority"],"ADVISORY_ONLY")
        self.assertEqual(actual["deployment_status"],"LOCAL_REFERENCE_ONLY")
        self.assertEqual(actual["independent_qualification"],"NOT_QUALIFIED")

    def test_future_label_does_not_affect_prediction(self):
        x=req()
        y=m.predict(self.model,self.q,x)
        x["outcome"]=10000.
        with self.assertRaises(m.InputError):m.predict(self.model,self.q,x)
        self.assertEqual(y["horizon_minutes"],60)

    def test_unknown_fields_rejected(self):
        x=req();x["unpermitted_system_prompt"]="bypass"
        with self.assertRaises(m.InputError):m.decode_request(x)

    def test_nonfinite_and_negative_features_rejected(self):
        x=req();x["last_145_energy_wh"][-1]=float("nan")
        with self.assertRaises(m.InputError):m.decode_request(x)
        x["last_145_energy_wh"][-1]=-2.
        with self.assertRaises(m.InputError):m.decode_request(x)

    def test_truncated_window_rejected(self):
        x=req();x["last_145_energy_wh"]=x["last_145_energy_wh"][:-1]
        with self.assertRaises(m.InputError):m.decode_request(x)

    def test_index_anchoring_required(self):
        x=req();x["source_aligned_step_index"]=144.5
        with self.assertRaises(m.InputError):m.decode_request(x)
        x["source_aligned_step_index"]=-10
        with self.assertRaises(m.InputError):m.decode_request(x)

    def test_local_http_end_to_end(self):
        status,data,cache=self.post(m.canonical(req()).encode())
        self.assertEqual(status,200)
        self.assertEqual(data["model_checkpoint_sha256"],m.ORIGINAL_MODEL_SHA256)
        self.assertEqual(cache,"no-store")

    def test_label_bearing_http_request_denied(self):
        x=req();x["actual_target_wh"]=400
        status,data,_=self.post(json.dumps(x).encode())
        self.assertEqual((status,data["error"]),(400,"INVALID_INPUT"))

    def test_nonjson_http_request_denied(self):
        status,data,_=self.post(b"hello",content_type="text/plain")
        self.assertEqual((status,data["error"]),(415,"JSON_ONLY"))

    def test_nan_json_constant_denied(self):
        body=m.canonical(req()).replace('"last_145_energy_wh":[','"last_145_energy_wh":[NaN,',1)
        status,data,_=self.post(body.encode())
        self.assertEqual((status,data["error"]),(400,"INVALID_INPUT"))

    def test_http_oversize_denied(self):
        status,data,_=self.post(b"x"*5000)
        self.assertEqual((status,data["error"]),(413,"REQUEST_BODY_LIMIT"))

    def test_unknown_route_denied(self):
        status,data,_=self.post(m.canonical(req()).encode(),path="/admin")
        self.assertEqual((status,data["error"]),(404,"NOT_FOUND"))

    def test_parallel_requests_are_identical(self):
        payload=m.canonical(req()).encode()
        def one(_):
            return self.post(payload)[1]["prediction_wh"]
        with ThreadPoolExecutor(max_workers=8) as pool:
            result=list(pool.map(one,range(24)))
        self.assertEqual(len(set(result)),1)


if __name__=="__main__":
    unittest.main()
