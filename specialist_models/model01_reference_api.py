#!/usr/bin/env python3
"""MODEL01 original Git-backed UCI ridge, typed no-fit inference and loopback HTTP.

Research/developer prototype ONLY: this checkpoint was trained on one original
household and evaluated on exposed DEV. This is not TAC-OSM, a cloud deployment,
production security approval, live source-time qualification or calibrated
cross-site coverage. The source-aligned sample index is NOT a universal clock.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "research/replay-bundles/PRODUCT-MODEL01-UCI374-CLASSICAL/ORIGINAL_CHECKPOINT.json"
RESULTS = ROOT / "research/replay-bundles/PRODUCT-MODEL01-UCI374-CLASSICAL/ORIGINAL_DEV_RESULTS.json"
ORIGINAL_MODEL_SHA256 = "cb756a734b026602bb13f6af11b08711e2029f78aad1e95bb469d55131bc6c82"
ORIGINAL_SOURCE_SHA256 = "2fccf354445d886e7917620b0195db1f3e3e34d5a067a93b844694a4c561255a"
ALLOWED = {"schema", "source_aligned_step_index", "last_145_energy_wh"}
MAX_HTTP_BODY = 4096

class InputError(ValueError):
    pass

def canonical(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), allow_nan=False)

def load_original():
    raw = MODEL.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ORIGINAL_MODEL_SHA256:
        raise InputError("original Git checkpoint bytes have changed")
    m = json.loads(raw)
    r = json.loads(RESULTS.read_bytes())
    if (m.get("model_schema") != "model01-uci374-lag-ridge-original-v1"
        or m.get("source_zip_sha256") != ORIGINAL_SOURCE_SHA256
        or r.get("source_zip_sha256") != ORIGINAL_SOURCE_SHA256
        or m.get("fit_examples") != 9717
        or r.get("fit_examples") != 9717
        or r.get("scientific_promotion") != "NOT_QUALIFIED"):
        raise InputError("checkpoint/receipt contract mismatch")
    w, means, stds = (m.get(k) for k in ("coefficients","normalizer_fit_means","normalizer_fit_scales"))
    if (not all(isinstance(v, list) for v in (w,means,stds))
        or [len(w),len(means),len(stds)] != [7,6,6]
        or any(type(v) not in (float,int) or not math.isfinite(v) for v in w+means+stds)
        or any(v<=0 for v in stds)):
        raise InputError("invalid original numerical checkpoint")
    q = r.get("calibration_q_90_wh")
    if type(q) not in (float,int) or not math.isfinite(q) or q < 0:
        raise InputError("invalid frozen original CAL interval")
    return m, float(q)

def decode_request(value):
    if not isinstance(value, dict) or set(value) != ALLOWED:
        raise InputError("exact typed request schema required; labels and other fields forbidden")
    if value["schema"] != "model01-energy-source-aligned-v1":
        raise InputError("unknown request schema")
    index = value["source_aligned_step_index"]
    if type(index) is not int or not 144 <= index <= 20_000_000:
        raise InputError("source-aligned index must be integer >=144")
    seq = value["last_145_energy_wh"]
    if not isinstance(seq,list) or len(seq) != 145:
        raise InputError("exactly 145 original 10-minute historical energy readings required")
    if any(type(v) not in (int,float) or not math.isfinite(v) or not 0 <= v <= 1_000_000
           for v in seq):
        raise InputError("energy readings must be finite and nonnegative")
    return index, seq

def predict(model, calibration_q, request):
    i, seq = decode_request(request)
    # Only observation-time values can enter the model (no labels, no future
    # data). The exact training-time 144-step periodic phase is reconstructed
    # from the caller-asserted source-origin-aligned index; not authenticated.
    angle = 2 * math.pi * (i % 144) / 144
    x = [seq[-1],seq[-2],seq[-7],seq[0],math.sin(angle),math.cos(angle)]
    w=model["coefficients"]; means=model["normalizer_fit_means"]; stds=model["normalizer_fit_scales"]
    p=max(0.,w[0]+sum(w[j+1]*(x[j]-means[j])/stds[j] for j in range(6)))
    if not math.isfinite(p):
        raise InputError("nonfinite model response")
    return {"schema":"model01-energy-forecast-response-v1",
            "type":"one-hour-ahead-energy-point-and-interval",
            "horizon_minutes":60,"prediction_wh":p,
            "interval_90_cal_only_wh":[p-calibration_q,p+calibration_q],
            "model_checkpoint_sha256":ORIGINAL_MODEL_SHA256,
            "reference_source_sha256":ORIGINAL_SOURCE_SHA256,
            "interval_status":"EMPIRICAL_HISTORICAL_CAL_NOT_GUARANTEED",
            "independent_qualification":"NOT_QUALIFIED",
            "deployment_status":"LOCAL_REFERENCE_ONLY",
            "action_authority":"ADVISORY_ONLY"}

def make_server(model, q):
    class Handler(BaseHTTPRequestHandler):
        server_version = "TACModel01Prototype"
        sys_version = ""

        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path != "/health":
                self.respond(404, {"error":"NOT_FOUND"})
                return
            self.respond(200,{"status":"LOCAL_ONLY", "model_sha256":ORIGINAL_MODEL_SHA256,
                              "scientific_qualification":"NOT_QUALIFIED"})

        def do_POST(self):
            if self.path != "/v1/predict":
                self.respond(404,{"error":"NOT_FOUND"});return
            length=self.headers.get("Content-Length","")
            if not length.isdecimal() or not 0 < int(length) <= MAX_HTTP_BODY:
                self.respond(413,{"error":"REQUEST_BODY_LIMIT"});return
            if self.headers.get("Content-Type","").split(";")[0].strip()!="application/json":
                self.respond(415,{"error":"JSON_ONLY"});return
            try:
                data=self.rfile.read(int(length))
                req=json.loads(data,parse_constant=lambda _: (_ for _ in ()).throw(InputError("NaN denied")))
                response=predict(model,q,req)
            except (ValueError,InputError,UnicodeError,TypeError):
                self.respond(400,{"error":"INVALID_INPUT"});return
            self.respond(200,response)

        def respond(self, status, value):
            raw=canonical(value).encode()
            self.send_response(status)
            self.send_header("Content-Type","application/json; charset=utf-8")
            self.send_header("Content-Length",str(len(raw)))
            self.send_header("Cache-Control","no-store")
            self.send_header("X-Content-Type-Options","nosniff")
            self.end_headers()
            self.wfile.write(raw)
    return Handler

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("predict")
    a.add_argument("--request",type=Path,required=True)
    s=sub.add_parser("serve")
    s.add_argument("--port",type=int,default=8765)
    args=p.parse_args(argv)
    try:
        model,q=load_original()
        if args.command=="predict":
            print(canonical(predict(model,q,json.loads(args.request.read_text()))))
        else:
            if not 1 <= args.port <= 65535:
                raise InputError("invalid loopback port")
            # This reference intentionally supports loopback only.
            with ThreadingHTTPServer(("127.0.0.1",args.port),make_server(model,q)) as server:
                print(canonical({"status":"LOCAL_DEV_SERVER_ONLY","host":"127.0.0.1",
                                 "port":args.port, "no_auth_tls":"DO_NOT_EXPOSE"}),flush=True)
                server.serve_forever(poll_interval=.1)
        return 0
    except (InputError,OSError,ValueError,TypeError) as err:
        print(canonical({"status":"BLOCKED_OR_FAILED","reason":str(err)}),file=sys.stderr)
        return 2

if __name__=="__main__":
    raise SystemExit(main())
