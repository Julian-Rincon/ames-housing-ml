#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAVI Agent API — servidor de desarrollo local.

Traduce requests de `http.server` a eventos de Lambda Function URL (payload
format 2.0) e invoca `handler.handler` directamente, para poder probar la UI
y la API sin desplegar nada. Usa `SAVI_DATA_DIR` para que `savi_api.store`
lea los fixtures locales en vez de ir a S3.

Uso:
    SAVI_DATA_DIR=../out/fixture python api/local_server.py --port 8080
"""
from __future__ import annotations

import argparse
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qsl, urlsplit

API_DIR = Path(__file__).resolve().parent
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

import handler as lambda_handler  # noqa: E402  (tras ajustar sys.path)


def _build_event(
    method: str,
    raw_path: str,
    query: str,
    headers: dict,
    body_bytes: bytes,
) -> dict:
    query_params = dict(parse_qsl(query, keep_blank_values=True)) if query else None
    return {
        "version": "2.0",
        "rawPath": raw_path,
        "rawQueryString": query,
        "headers": {k.lower(): v for k, v in headers.items()},
        "queryStringParameters": query_params,
        "requestContext": {"http": {"method": method, "path": raw_path}},
        "body": body_bytes.decode("utf-8", errors="replace") if body_bytes else None,
        "isBase64Encoded": False,
    }


class LambdaProxyHandler(BaseHTTPRequestHandler):
    server_version = "SaviLocalServer/1.0"

    def _handle(self, method: str) -> None:
        split = urlsplit(self.path)
        content_length = int(self.headers.get("Content-Length", 0) or 0)
        body_bytes = self.rfile.read(content_length) if content_length else b""

        event = _build_event(method, split.path, split.query, dict(self.headers), body_bytes)
        response = lambda_handler.handler(event, context=None)

        status = response.get("statusCode", 200)
        headers = response.get("headers", {})
        body = response.get("body") or ""
        body_out = body.encode("utf-8") if isinstance(body, str) else body

        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body_out)))
        self.end_headers()
        if body_out:
            self.wfile.write(body_out)

    def do_GET(self) -> None:  # noqa: N802 - nombre requerido por BaseHTTPRequestHandler
        self._handle("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._handle("POST")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._handle("OPTIONS")

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003 - firma heredada
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Servidor de desarrollo local para SAVI Agent API")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), LambdaProxyHandler)
    print(f"SAVI local server escuchando en http://{args.host}:{args.port} (Ctrl+C para salir)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
