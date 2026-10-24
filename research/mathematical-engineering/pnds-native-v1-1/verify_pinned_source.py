#!/usr/bin/env python3
"""Pinned real-source versus PNDS finite-array math verification, zero training.

This is a *source-operator* check, not a whole-model/FP32/learned recall claim.
The actual original source is fetched at its immutable commit and git blob SHA
verified before import. Mathematical reference likewise pinned to PNDS commit.
"""
from __future__ import annotations
import hashlib
import math
import types
import urllib.request
from functools import lru_cache

import torch
from torch.nn import functional as F

SOURCE_COMMIT = "081e534a074001c6a3b2251b3ff31ad20ec4fca8"
SOURCE_GIT_BLOB = "ff7b7d3e6ad7ec5154560b56a14968d2c0472574"
SOURCE_URL = ("https://raw.githubusercontent.com/FPC-effortless/TAC-OSM/" + SOURCE_COMMIT
              + "/src/tac_osm/aligned_associative_tac.py")
PNDS_COMMIT = "552c62048640a34317d49b4ab2b8075a13a8db4a"
PNDS_GIT_BLOB = "fa8060f8ee0076b6bfa6357517ba53b4cddaa5ff"
PNDS_URL = ("https://raw.githubusercontent.com/FPC-effortless/pnds-lean/" + PNDS_COMMIT
            + "/foundation/v1/native_memory_math.py")


def load_trusted(repo_url, git_blob, name):
    with urllib.request.urlopen(repo_url, timeout=30) as handle:
        raw = handle.read()
    git_sha = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if git_sha != git_blob:
        raise RuntimeError(f"pin mismatch: {name} expected {git_blob}, got {git_sha}")
    mod = types.ModuleType(name)
    exec(compile(raw, repo_url, "exec"), mod.__dict__)
    return mod


@lru_cache(maxsize=1)
def modules():
    source = load_trusted(SOURCE_URL, SOURCE_GIT_BLOB, "pinned_source")
    reference = load_trusted(PNDS_URL, PNDS_GIT_BLOB, "pnds_math")
    return source, reference


def verify():
    torch.set_num_threads(1)
    source, mathref = modules()
    torch.manual_seed(20261024)
    checks = 0
    for dtype in (torch.float32, torch.float64):
        for batch in (1, 2, 8):
            m = torch.randn(batch,8,8,dtype=dtype) * .4
            k = F.normalize(torch.randn(batch,8,dtype=dtype), dim=-1)
            v = F.normalize(torch.randn(batch,8,dtype=dtype), dim=-1)
            alpha = torch.rand(batch,1,dtype=dtype) * .7 + .2
            beta = torch.rand(batch,1,dtype=dtype) * .8 + .1
            writable = torch.arange(batch).remainder(2).eq(0).unsqueeze(-1)
            for radius in (None, 4.0, .25):
                result = source.gated_write(m,k,v,alpha,beta,writable,radius)
                target = torch.tensor([mathref.write(
                    m[i].tolist(), k[i].tolist(), v[i].tolist(),
                    alpha[i].item(), beta[i].item(), writable[i].item(), radius
                ) for i in range(batch)],dtype=dtype)
                tol = 3e-6 if dtype == torch.float32 else 1e-11
                torch.testing.assert_close(result, target, rtol=tol, atol=tol)
                checks += 1
            read_only = source.gated_write(m,k,v,alpha,beta,torch.zeros_like(writable), .1)
            torch.testing.assert_close(read_only,m,rtol=0,atol=0)
            checks += 1

    # Exact coordinate sensitivity on *unprojected* source: ∂(Mq)_i/∂v_i
    # = beta * inner(k,q). This formula is not valid if projection activates.
    m = torch.randn(1,8,8,dtype=torch.float64) * .1
    k = F.normalize(torch.randn(1,8,dtype=torch.float64),dim=-1)
    q = F.normalize(torch.randn(1,8,dtype=torch.float64),dim=-1)
    v = torch.randn(1,8,dtype=torch.float64,requires_grad=True)
    alpha = torch.full((1,1),.91,dtype=torch.float64)
    beta = torch.full((1,1),.49,dtype=torch.float64)
    yes = torch.ones(1,1,dtype=torch.bool)
    u = source.gated_write(m,k,v,alpha,beta,yes,None)
    out = torch.bmm(u,q.unsqueeze(-1)).squeeze(-1)
    grad = torch.autograd.grad(out[0,0],v)[0]
    expected = beta[0,0] * (k*q).sum()
    torch.testing.assert_close(grad[0,0], expected,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(grad[0,1:], torch.zeros_like(grad[0,1:]),
                               rtol=0,atol=1e-12)
    checks += 1

    # Full original-source M/key/value/alpha/beta differential away from
    # projection kinks; this is software autograd correctness, NOT whole network.
    def operator(M, K, V, A, B):
        return source.gated_write(M,K,V,A,B,yes,None)
    args = (
        torch.randn(1,8,8,dtype=torch.float64,requires_grad=True)*.2,
        F.normalize(torch.randn(1,8,dtype=torch.float64),dim=-1).detach().requires_grad_(),
        torch.randn(1,8,dtype=torch.float64,requires_grad=True)*.2,
        torch.tensor([[.9]],dtype=torch.float64,requires_grad=True),
        torch.tensor([[.4]],dtype=torch.float64,requires_grad=True),
    )
    if not torch.autograd.gradcheck(operator,args,eps=1e-6,atol=1e-6,rtol=1e-4):
        raise RuntimeError("original operator double-precision autograd failed")
    checks += 1

    print({"status":"PASS_PINNED_OPERATOR_PARITY_ONLY","checks":checks,
           "actual_source_commit":SOURCE_COMMIT, "pnds_reference_commit":PNDS_COMMIT,
           "full_native_model_verified":False,"trained_memory_improved":False,
           "new_training_steps":0})


if __name__ == "__main__":
    verify()
