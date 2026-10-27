"""Constructed request security checks. No production hosting or trained-model gain."""
from http.server import ThreadingHTTPServer
from threading import Thread
import http.client
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]/"specialist_models"
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location("model01_authenticated_gateway",ROOT/"model01_authenticated_gateway.py")
g=importlib.util.module_from_spec(spec);sys.modules[spec.name]=g;spec.loader.exec_module(g)

TOKEN="fixture-only-long-opaque-token-abcdefghijklmnopqrstuvwxyz123456789"
BAD_TOKEN="fixture-only-long-opaque-token-abcdefghijklmnopqrstuvwxyz123456780"


def request():
    return {"schema":"model01-energy-source-aligned-v1",
            "source_aligned_step_index":10500,
            "last_145_energy_wh":[float(65+i%9) for i in range(145)]}


class AuthGatewayContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model,cls.q=g.load_original()
        cls.server=g.create_server(cls.model,cls.q,TOKEN,capacity=250,
                                   refill_per_second=250,concurrent=8)
        cls.worker=Thread(target=cls.server.serve_forever,daemon=True);cls.worker.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.worker.join(timeout=3)

    def call(self,token=TOKEN,body=None,method="POST",path="/v1/predict",
             auth_headers=None,extra_headers=None):
        if body is None:
            body=g.canonical(request()).encode()
        client=http.client.HTTPConnection("127.0.0.1",self.server.server_port,timeout=3)
        try:
            client.putrequest(method,path)
            if auth_headers is None:
                if token is not None:
                    client.putheader("Authorization","Bearer "+token)
            else:
                for item in auth_headers:
                    client.putheader("Authorization",item)
            if method=="POST":
                client.putheader("Content-Type","application/json")
                client.putheader("Content-Length",str(len(body)))
            for k,v in (extra_headers or {}).items():
                client.putheader(k,v)
            client.endheaders()
            if method=="POST":client.send(body)
            res=client.getresponse()
            return res.status,res.read().decode()
        finally:
            client.close()

    def test_valid_bearer_and_checkpoint_identity(self):
        status,data=self.call()
        self.assertEqual(status,200)
        res=json.loads(data)
        self.assertEqual(res["model_checkpoint_sha256"],g.load_original()[0].get("sha256",res["model_checkpoint_sha256"]))
        self.assertEqual(res["independent_qualification"],"NOT_QUALIFIED")
        self.assertEqual(res["deployment_status"],"LOCAL_REFERENCE_ONLY")

    def test_missing_auth_denied_even_when_body_valid(self):
        self.assertEqual(self.call(token=None)[0],401)

    def test_wrong_key_denied_without_token_reflection(self):
        status,body=self.call(token=BAD_TOKEN)
        self.assertEqual(status,401)
        self.assertNotIn(BAD_TOKEN,body)
        self.assertNotIn(TOKEN,body)

    def test_duplicate_authorization_rejected(self):
        self.assertEqual(self.call(auth_headers=["Bearer "+TOKEN,"Bearer "+TOKEN])[0],401)

    def test_get_health_requires_auth(self):
        self.assertEqual(self.call(token=None,method="GET",path="/health")[0],401)
        status,body=self.call(method="GET",path="/health")
        self.assertEqual(status,200)
        self.assertNotIn(TOKEN,body)

    def test_transfer_encoding_with_content_length_rejected(self):
        status,_=self.call(extra_headers={"Transfer-Encoding":"chunked"})
        self.assertEqual(status,400)

    def test_malformed_future_field_is_not_overridden_by_authentication(self):
        x=request();x["future_outcome"]=999
        status,_=self.call(body=g.canonical(x).encode())
        self.assertEqual(status,400)

    def test_limited_local_server_never_binds_public_network(self):
        with self.assertRaises(g.GatewayError):
            g.create_server(self.model,self.q,TOKEN,host="0.0.0.0")
        with self.assertRaises(g.GatewayError):
            g.create_server(self.model,self.q,TOKEN,host="::")
    
    def test_fail_closed_without_env_secret(self):
        with patch.dict("os.environ",{},clear=True):
            with self.assertRaises(g.GatewayError):g.key_from_environment()

    def test_reject_short_secret(self):
        with patch.dict("os.environ",{"MODEL01_API_KEY":"weak"}):
            with self.assertRaises(g.GatewayError):g.key_from_environment()

    def test_frozen_original_graph_does_not_change_under_authenticated_requests(self):
        m,q=g.load_original()
        status,body=self.call()
        self.assertEqual(status,200)
        output=json.loads(body)
        import model01_reference_api as ref
        expected=ref.predict(m,q,request())
        self.assertEqual(output,expected)

    def test_rate_limit_fail_closed(self):
        server=g.create_server(self.model,self.q,TOKEN,capacity=1,refill_per_second=.00000001)
        t=Thread(target=server.serve_forever,daemon=True);t.start()
        try:
            def send():
                client=http.client.HTTPConnection("127.0.0.1",server.server_port,timeout=3)
                try:
                    client.request("POST","/v1/predict",g.canonical(request()).encode(),
                                   headers={"Authorization":"Bearer "+TOKEN,
                                            "Content-Type":"application/json"})
                    resp=client.getresponse();resp.read()
                    return resp.status
                finally:client.close()
            self.assertEqual(send(),200)
            self.assertEqual(send(),429)
        finally:server.shutdown();server.server_close();t.join(timeout=3)

    def test_health_rejects_unknown_resource(self):
        status,_=self.call(method="GET",path="/private/model")
        self.assertEqual(status,404)


if __name__=="__main__":
    unittest.main()
