#!/usr/bin/env python3
"""Authenticated, bounded Model01 *private ingress* serving prototype.

This is NOT a production-hosted service. Only binds 127.0.0.1, never accepts
TLS via this HTTP listener. A separately authenticated/TLS-terminating trusted
ingress is REQUIRED for any later cloud installation, as are network isolation,
load testing, key lifecycle, service monitoring and a reviewed deployment.
No unqualified user data may enter the inference path.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re
import secrets
import sys
import threading
import time

import model01_reference_api as reference

MAX_BODY = reference.MAX_HTTP_BODY
VALID_DIGEST = re.compile(r"^[0-9a-f]{64}$")
SERVER_VERSION = "Model01-Guarded-Prototype"

class PolicyError(ValueError):
    pass

def verify_configuration(key_sha256: str, max_per_minute: int, max_active: int) -> None:
    if not isinstance(key_sha256, str) or not VALID_DIGEST.fullmatch(key_sha256):
        raise PolicyError("MODEL01_KEY_SHA256 must contain exactly 64 lowercase hex digits")
    # Deny well-known misconfiguration values; these are not credentials.
    if key_sha256 == hashlib.sha256(b"").hexdigest() or key_sha256 == "0" * 64:
        raise PolicyError("empty/default API secret forbidden")
    if type(max_per_minute) is not int or not 1 <= max_per_minute <= 10000:
        raise PolicyError("request-per-minute budget outside 1..10000")
    if type(max_active) is not int or not 1 <= max_active <= 32:
        raise PolicyError("maximum active inferences outside 1..32")

class MinuteLimiter:
    """Bounded, lock-protected monotonic rolling window, global to this process."""
    def __init__(self, limit: int, clock=time.monotonic):
        self.limit = limit
        self.clock = clock
        self.accesses = deque()
        self.lock = threading.Lock()

    def admit(self):
        with self.lock:
            now = self.clock()
            while self.accesses and self.accesses[0] <= now - 60:
                self.accesses.popleft()
            if len(self.accesses) >= self.limit:
                return False
            self.accesses.append(now)
            return True

def authenticate(header, expected_hash):
    if not isinstance(header, str) or len(header) > 512 or not header.startswith("Bearer "):
        return False
    candidate = header[7:]
    if len(candidate) < 24 or len(candidate) > 200 or any(not ch.isascii() or ch.isspace() for ch in candidate):
        return False
    digest = hashlib.sha256(candidate.encode("ascii")).hexdigest()
    return hmac.compare_digest(digest, expected_hash)

def make_handler(model, calibration_q, key_sha256, limiter, active):
    class Handler(BaseHTTPRequestHandler):
        server_version = SERVER_VERSION
        sys_version = ""
        protocol_version = "HTTP/1.0"

        def log_message(self, *_):
            # Never log credentials, request or health payloads.
            pass

        def respond(self, status, content, retry_after=None):
            payload=reference.canonical(content).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            if status == 401:
                self.send_header("WWW-Authenticate", 'Bearer realm="model01"')
            if retry_after is not None:
                self.send_header("Retry-After", str(retry_after))
            self.end_headers()
            self.wfile.write(payload)

        def authorized(self):
            # Allow exactly one Authorization header, and no cookies/query
            # tokens or X-Forwarded-* trust. No public unauthenticated health.
            auth_headers = self.headers.get_all("Authorization", [])
            if len(auth_headers) != 1 or not authenticate(auth_headers[0], key_sha256):
                self.respond(401, {"error": "UNAUTHORIZED"})
                return False
            if not limiter.admit():
                self.respond(429, {"error": "RATE_LIMITED"}, retry_after=60)
                return False
            return True

        def do_GET(self):
            if self.path != "/health":
                self.respond(404, {"error": "NOT_FOUND"})
                return
            if not self.authorized():
                return
            self.respond(200, {"status": "PRIVATE_INGRESS_PROTOTYPE_ONLY",
                               "model_sha256": reference.ORIGINAL_MODEL_SHA256,
                               "independent_qualification": "NOT_QUALIFIED"})

        def do_POST(self):
            if self.path != "/v1/predict":
                self.respond(404, {"error": "NOT_FOUND"})
                return
            if not self.authorized():
                return
            # Explicitly reject Transfer-Encoding and ambiguous duplicate
            # length fields; read no body until all checks pass.
            lengths=self.headers.get_all("Content-Length", [])
            if (self.headers.get("Transfer-Encoding") is not None
                or len(lengths) != 1 or not lengths[0].isdecimal()
                or not 0 < int(lengths[0]) <= MAX_BODY):
                self.respond(413, {"error": "INVALID_BODY_LENGTH"})
                return
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "application/json":
                self.respond(415, {"error": "JSON_ONLY"})
                return
            if not active.acquire(blocking=False):
                self.respond(503, {"error": "BUSY"}, retry_after=1)
                return
            try:
                raw=self.rfile.read(int(lengths[0]))
                if len(raw) != int(lengths[0]):
                    raise reference.InputError("truncated body")
                obj=json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(reference.InputError("nonfinite JSON constant")),
                               parse_float=float)
                response=reference.predict(model, calibration_q, obj)
            except (ValueError, reference.InputError, UnicodeError, TypeError, json.JSONDecodeError):
                self.respond(400, {"error": "INVALID_INPUT"})
                return
            finally:
                active.release()
            self.respond(200, response)

    return Handler

def create_server(key_sha256, rate=600, concurrency=4, port=0):
    verify_configuration(key_sha256, rate, concurrency)
    model, q = reference.load_original()
    limiter = MinuteLimiter(rate)
    active = threading.BoundedSemaphore(concurrency)
    class BoundedThreadingHTTPServer(ThreadingHTTPServer):
        daemon_threads=True
        request_queue_size=16
        block_on_close=False
    return BoundedThreadingHTTPServer(("127.0.0.1", port),
                                     make_handler(model, q, key_sha256, limiter, active))

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--requests-per-minute", type=int, default=600)
    parser.add_argument("--max-active", type=int, default=4)
    args=parser.parse_args(argv)
    try:
        if not 1 <= args.port <= 65535:
            raise PolicyError("invalid loopback port")
        digest=os.environ.get("MODEL01_KEY_SHA256", "")
        # Deliberately no command-line key, token, or logged hash.
        with create_server(digest, args.requests_per_minute, args.max_active, args.port) as server:
            print(reference.canonical({"status":"PRIVATE_INGRESS_PROTOTYPE_ONLY",
                                       "host":"127.0.0.1", "port": args.port,
                                       "deployment": "NOT_PRODUCTION",
                                       "tls": "EXTERNAL_TRUSTED_INGRESS_REQUIRED",
                                       "warning": "DO_NOT_EXPOSE_OR_FORWARD_PUBLICLY"}),flush=True)
            server.serve_forever(poll_interval=.1)
        return 0
    except (OSError, PolicyError, reference.InputError) as error:
        print(reference.canonical({"status":"BLOCKED_OR_FAILED",
                                   "reason": str(error)}),file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
