#!/usr/bin/env python3
"""Benchmark ONLY authenticated localhost gateway HTTP+JSON; no production cloud.

Uses a fresh process-local test credential (never prints or writes credential).
Constructed requests. No genuine external customer load or Internet network.
"""
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import math
import os
import platform
import secrets
from statistics import mean
from threading import Thread
from time import perf_counter_ns

from model01_authenticated_gateway import create_server, load_original
from model01_reference_api import canonical, ORIGINAL_MODEL_SHA256


def benchmark():
    token=secrets.token_urlsafe(48)
    model,q=load_original()
    server=create_server(model,q,token,capacity=2000,refill_per_second=2000,concurrent=16)
    worker=Thread(target=server.serve_forever,daemon=True)
    worker.start()
    req=canonical({"schema":"model01-energy-source-aligned-v1",
                   "source_aligned_step_index":10500,
                   "last_145_energy_wh":[50.0+3.0*math.sin(j/10) for j in range(145)]}).encode()
    def once(i):
        conn=http.client.HTTPConnection("127.0.0.1",server.server_port,timeout=3)
        start=perf_counter_ns()
        try:
            conn.request("POST","/v1/predict",body=req,
                         headers={"Authorization":"Bearer "+token,
                                  "Content-Type":"application/json"})
            response=conn.getresponse()
            data=json.loads(response.read())
            if response.status!=200 or data.get("model_checkpoint_sha256")!=ORIGINAL_MODEL_SHA256:
                raise RuntimeError("failed authentication/model identity under measured load")
            return (perf_counter_ns()-start)/1e6
        finally:
            conn.close()
    def run(n,threads):
        for i in range(16):once(i)
        if threads==1:
            samples=[once(i) for i in range(n)]
        else:
            with ThreadPoolExecutor(max_workers=threads) as pool:
                samples=list(pool.map(once,range(n)))
        samples.sort()
        return {"n":n,"clients":threads,"median_ms":samples[(n-1)//2],
                "p95_ms":samples[math.ceil(.95*n)-1],
                "p99_ms":samples[math.ceil(.99*n)-1],
                "mean_ms":mean(samples),"max_ms":samples[-1],
                "invalid_replies":0}
    try:
        one=run(200,1)
        four=run(300,4)
        return {"schema":"model01-authenticated-loopback-benchmark-v1",
                "model_checkpoint_sha256":ORIGINAL_MODEL_SHA256,
                "host":platform.platform(),"arch":platform.machine(),
                "cpu_count_reported":os.cpu_count(),
                "python":platform.python_version(),
                "network":"LOCALHOST_ONLY_HTTP_NO_TLS",
                "auth":"EPHEMERAL_TEST_BEARER_NEVER_RECORDED",
                "source":"CONSTRUCTED_INPUT_ONLY",
                "single_client":one,"four_clients":four,
                "production_cloud_deployment":False,
                "real_customer_load":False,
                "scientific_model_qualification":False,
                "warnings":["latencies are VM/run-specific and not priced network cloud inference",
                            "no external transport, TLS, service reliability or token rotation measured"]}
    finally:
        server.shutdown();server.server_close();worker.join(timeout=3)


if __name__=="__main__":
    print(canonical(benchmark()))
