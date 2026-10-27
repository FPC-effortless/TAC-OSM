#!/usr/bin/env python3
"""Model01 authenticated loopback inference gateway for a future TLS-terminated cloud edge.

This is a development/test gateway, NOT a publicly deployed secure service.
No built-in TLS, tenant isolation, remote address trust, shared multi-replica
rate limit, production monitoring or billing. Never bind it to public networks.
"""
from __future__ import annotations
import argparse
import hashlib
import hmac
import os
import secrets
import socket
from http.server import ThreadingHTTPServer
from threading import BoundedSemaphore, Lock
from time import monotonic

from model01_reference_api import canonical, load_original, make_server, InputError

MAX_AUTH_HEADER = 256
MIN_KEY_BYTES = 32


class GatewayError(ValueError):
    pass


def key_from_environment():
    key = os.environ.get("MODEL01_API_KEY")
    if key is None:
        raise GatewayError("MODEL01_API_KEY is required (development gateway remains off)")
    if not isinstance(key, str) or not key.isascii() or not MIN_KEY_BYTES <= len(key.encode()) <= 128:
        raise GatewayError("MODEL01_API_KEY must be 32..128 ASCII bytes")
    if key.isspace() or key.strip() != key:
        raise GatewayError("invalid MODEL01_API_KEY whitespace")
    return key


class Bucket:
    """Process-local token bucket; not cross-instance or production rate protection."""
    def __init__(self, capacity=32, refill_per_second=16):
        if type(capacity) is not int or not 1 <= capacity <= 10000:
            raise GatewayError("invalid capacity")
        if type(refill_per_second) not in (int, float) or not 0 < refill_per_second <= 10000:
            raise GatewayError("invalid refill rate")
        self.capacity = capacity
        self.rate = refill_per_second
        self.tokens = float(capacity)
        self.last = monotonic()
        self.lock = Lock()

    def allow(self):
        with self.lock:
            now = monotonic()
            self.tokens = min(self.capacity, self.tokens + max(0, now - self.last) * self.rate)
            self.last = now
            if self.tokens < 1:
                return False
            self.tokens -= 1
            return True


def handler_for(model, interval_q, key, *, capacity=32, refill_per_second=16, concurrent=8):
    if not isinstance(key, str) or not key.isascii() or not MIN_KEY_BYTES <= len(key) <= 128:
        raise GatewayError("invalid required secret")
    if type(concurrent) is not int or not 1 <= concurrent <= 128:
        raise GatewayError("invalid concurrent request cap")
    base = make_server(model, interval_q)
    token_digest = hashlib.sha256(key.encode("ascii")).digest()
    bucket = Bucket(capacity, refill_per_second)
    inflight = BoundedSemaphore(concurrent)

    class AuthenticatedHandler(base):
        server_version = "Model01GatewayPrototype"
        sys_version = ""
        protocol_version = "HTTP/1.0"

        def _gate(self):
            # Never echo, print, persist, or interpolate an authorization header.
            self.close_connection = True
            self.connection.settimeout(4)
            if (self.headers.get("Transfer-Encoding") is not None
                or len(self.headers.get_all("Content-Length") or []) > 1):
                self.respond(400, {"error":"INVALID_FRAMING"})
                return False
            auths = self.headers.get_all("Authorization") or []
            if len(auths) != 1:
                self.respond(401, {"error":"UNAUTHORIZED"})
                return False
            header = auths[0]
            if not isinstance(header, str) or len(header) > MAX_AUTH_HEADER or not header.startswith("Bearer "):
                self.respond(401, {"error":"UNAUTHORIZED"})
                return False
            candidate = header[7:]
            if not candidate.isascii() or not hmac.compare_digest(
                hashlib.sha256(candidate.encode("ascii")).digest(), token_digest
            ):
                self.respond(401, {"error":"UNAUTHORIZED"})
                return False
            if not bucket.allow():
                self.respond(429, {"error":"RATE_LIMITED"})
                return False
            if not inflight.acquire(blocking=False):
                self.respond(503, {"error":"BUSY"})
                return False
            return True

        def do_POST(self):
            if not self._gate():
                return
            try:
                super().do_POST()
            except (TimeoutError, socket.timeout):
                # Broken/incomplete clients cannot keep holding a worker.
                self.close_connection = True
            finally:
                inflight.release()

        def do_GET(self):
            if not self._gate():
                return
            try:
                super().do_GET()
            finally:
                inflight.release()

    return AuthenticatedHandler


def create_server(model, interval_q, key, *, host="127.0.0.1", port=0,
                  capacity=32, refill_per_second=16, concurrent=8):
    if host != "127.0.0.1" or type(port) is not int or not 0 <= port <= 65535:
        raise GatewayError("only 127.0.0.1 loopback binding permitted")
    handler = handler_for(model, interval_q, key, capacity=capacity,
                          refill_per_second=refill_per_second, concurrent=concurrent)
    class DevServer(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 64
    return DevServer((host, port), handler)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8766)
    a=p.parse_args(argv)
    try:
        key=key_from_environment()
        model,q=load_original()
        if not 1 <= a.port <= 65535:
            raise GatewayError("invalid port")
        with create_server(model,q,key,port=a.port) as server:
            print(canonical({"status":"LOCAL_AUTHENTICATED_DEVELOPMENT_ONLY",
                             "host":"127.0.0.1","port":a.port,
                             "external_network":"DISABLED","tls":"NOT_PROVIDED",
                             "independent_model_qualification":"NOT_ESTABLISHED"}),flush=True)
            server.serve_forever(poll_interval=.1)
        return 0
    except (GatewayError,InputError,OSError) as exc:
        # Never include secret-bearing runtime values in messages.
        print(canonical({"status":"BLOCKED_OR_FAILED","reason":str(exc)}))
        return 2


if __name__=="__main__":
    raise SystemExit(main())
