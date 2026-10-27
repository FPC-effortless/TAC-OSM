#!/usr/bin/env python3
"""Measure actual private guarded HTTP loopback throughput/latency (NO cloud).

Constructed 145-step request, Git original immutable checkpoint, fresh secret
in process only, no source data or key in receipts. No network/TLS/real cloud.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import math
import os
import platform
import secrets
import statistics
from threading import Thread
from time import perf_counter_ns

import model01_guarded_ingress as guarded
import model01_reference_api as api


def main():
    token=secrets.token_urlsafe(36)
    key_sha=hashlib.sha256(token.encode()).hexdigest()
    server=guarded.create_server(key_sha,rate=1500,concurrency=8,port=0)
    worker=Thread(target=server.serve_forever,daemon=True)
    worker.start()
    payload=api.canonical({"schema":"model01-energy-source-aligned-v1",
                           "source_aligned_step_index":11200,
                           "last_145_energy_wh":[80+6*math.sin(i/30) for i in range(145)]}).encode()
    def once(_):
        c=http.client.HTTPConnection("127.0.0.1",server.server_port,timeout=5)
        start=perf_counter_ns()
        try:
            c.request("POST","/v1/predict",body=payload,headers={
                "Content-Type":"application/json",
                "Authorization":"Bearer "+token})
            response=c.getresponse()
            raw=response.read()
            if response.status!=200:
                raise RuntimeError("guarded local benchmark must have no denied request")
            import json
            prediction=json.loads(raw)
            if prediction["model_checkpoint_sha256"] != api.ORIGINAL_MODEL_SHA256:
                raise RuntimeError("wrong checkpoint")
            return (perf_counter_ns()-start)/1e6
        finally:
            c.close()
    def measure(n, clients):
        for i in range(12):
            once(i)
        if clients==1:
            times=[once(i) for i in range(n)]
        else:
            with ThreadPoolExecutor(max_workers=clients) as pool:
                times=list(pool.map(once,range(n)))
        times.sort()
        return {"completed":n,"clients":clients,"errors":0,
                "p50_ms":times[(n-1)//2],
                "p95_ms":times[math.ceil(.95*n)-1],
                "p99_ms":times[math.ceil(.99*n)-1],
                "avg_ms":statistics.fmean(times),
                "max_ms":times[-1]}
    try:
        a=measure(200,1)
        b=measure(300,4)
        print(api.canonical({
            "schema":"model01-authenticated-private-ingress-loopback-v1",
            "environment":"GitHub Actions runner, local-only HTTP, synthetic request",
            "model_checkpoint_sha256":api.ORIGINAL_MODEL_SHA256,
            "host_machine":platform.machine(),
            "host_os":platform.platform(),
            "reported_cpu_count":os.cpu_count(),
            "python":platform.python_version(),
            "single_client":a,"four_clients":b,
            "auth":"bearer key SHA256 comparison; key NOT emitted",
            "cloud_deployment":False,"real_internet_tls":False,
            "price_per_thousand_predictions":"NOT_MEASURED",
            "independent_accuracy_qualification":False,
            "caution":"not a production load/SLA/DoS/security test; values depend on runner hardware"
        }))
        return 0
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)

if __name__=="__main__":
    raise SystemExit(main())
