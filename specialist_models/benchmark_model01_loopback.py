#!/usr/bin/env python3
"""Loopback-only Model01 HTTP reference benchmark on constructed input.

Reports actual CPU/OS, process-local end-to-end p50/p95 incl JSON and HTTP.
NOT production cloud, user-path latency, genuine model accuracy or Android.
"""
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
import http.client
import json
import math
import os
import platform
from pathlib import Path
import statistics
import sys
from threading import Thread
from time import perf_counter_ns

sys.path.insert(0,str(Path(__file__).resolve().parent))
from model01_reference_api import load_original, make_server, canonical, ORIGINAL_MODEL_SHA256


def sample():
    return {"schema":"model01-energy-source-aligned-v1","source_aligned_step_index":10500,
            "last_145_energy_wh":[85.0+13.0*math.sin(i/20) for i in range(145)]}


def main():
    m, q = load_original()
    server = ThreadingHTTPServer(("127.0.0.1",0),make_server(m,q))
    t=Thread(target=server.serve_forever,daemon=True)
    t.start()
    port=server.server_port
    encoded=canonical(sample()).encode("utf-8")
    def once(_):
        client=http.client.HTTPConnection("127.0.0.1",port,timeout=4)
        start=perf_counter_ns()
        try:
            client.request("POST","/v1/predict",body=encoded,
                           headers={"Content-Type":"application/json"})
            response=client.getresponse()
            raw=response.read()
            if response.status!=200:
                raise RuntimeError("HTTP response != 200")
            body=json.loads(raw)
            if body["model_checkpoint_sha256"]!=ORIGINAL_MODEL_SHA256 or body["independent_qualification"]!="NOT_QUALIFIED":
                raise RuntimeError("invalid model or status from reference endpoint")
        finally:
            client.close()
        return (perf_counter_ns()-start)/1e6

    def measure(count, concurrency):
        for _ in range(20):
            once(0)
        if concurrency==1:
            lat=[once(i) for i in range(count)]
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                lat=list(ex.map(once, range(count)))
        lat.sort()
        return {"requests":count,"clients":concurrency,"p50_ms":lat[(count-1)//2],
                "p95_ms":lat[math.ceil(.95*count)-1],
                "p99_ms":lat[math.ceil(.99*count)-1],
                "average_ms":statistics.fmean(lat),
                "max_ms":lat[-1],
                "failures":0}
    try:
        s=measure(200,1)
        c=measure(300,4)
        result={"schema":"model01-loopback-benchmark-v1",
                "checkpoint_sha256":ORIGINAL_MODEL_SHA256,
                "measurement":"local GitHub Actions loopback HTTP+JSON+model (socket per request)",
                "host":platform.platform(),
                "machine":platform.machine(),
                "python":platform.python_version(),
                "cpu_count_reported":os.cpu_count(),
                "sequential":s,
                "concurrent":c,
                "deployment":"LOCAL_REFERENCE_NOT_CLOUD_DEPLOYED",
                "source_data":"CONSTRUCTED_REQUEST_ONLY",
                "independent_science":"NOT_EVALUATED",
                "warnings":["in-run GitHub VM CPU may vary and is not priced server hardware",
                            "loopback excludes Internet/TLS/auth/network and real production load",
                            "latency alone cannot qualify a model's scientific accuracy or utility"]}
        print(canonical(result))
        return 0
    finally:
        server.shutdown()
        server.server_close()
        t.join(timeout=3)

if __name__=="__main__":
    raise SystemExit(main())
