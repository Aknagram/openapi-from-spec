from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from openapi_from_spec.pipeline import Settings, convert_text

_MAX_BODY = 15_000_000


def _static(name: str) -> Path:
    return Path(__file__).resolve().parent / "static" / name


def _sample() -> Path:
    return Path(__file__).resolve().parents[2] / "examples" / "account-balances.txt"


def serve(settings: Settings, host: str = "127.0.0.1", port: int = 8090) -> None:
    server = make_server(settings, host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def make_server(settings: Settings, host: str, port: int) -> ThreadingHTTPServer:
    handler = _handler(settings)
    return ThreadingHTTPServer((host, port), handler)


def _handler(settings: Settings) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in {"/", "/index.html"}:
                self._bytes(200, "text/html; charset=utf-8", _static("index.html").read_bytes())
            elif path == "/health":
                self._json(200, {"status": "ok"})
            elif path == "/sample":
                sample = _sample()
                if not sample.is_file():
                    self._json(404, {"error": "пример не найден"})
                    return
                self._bytes(200, "text/plain; charset=utf-8", sample.read_bytes())
            else:
                self._json(404, {"error": "не найдено"})

        def do_POST(self) -> None:
            if self.path.split("?", 1)[0] != "/v1/convert":
                self._json(404, {"error": "не найдено"})
                return
            length = int(self.headers.get("Content-Length") or "0")
            if length > _MAX_BODY:
                self._json(413, {"error": "документ больше 15 МБ"})
                return
            raw = self.rfile.read(length)
            content_type = self.headers.get("Content-Type", "")
            title = settings.title
            try:
                if "application/json" in content_type:
                    payload = json.loads(raw.decode("utf-8"))
                    text = str(payload.get("text") or "")
                    if payload.get("title"):
                        title = str(payload["title"])
                else:
                    text = raw.decode("utf-8")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                self._json(400, {"error": "нужен UTF-8 текст или JSON с полем text"})
                return
            spec_settings = Settings(**{**settings.__dict__, "title": title})
            result = convert_text(text, spec_settings)
            self._json(
                200,
                {
                    "yaml": result.yaml_text,
                    "report": result.report,
                    "errors": result.errors,
                },
            )

        def log_message(self, fmt: str, *args: object) -> None:
            return

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._bytes(status, "application/json; charset=utf-8", body)

        def _bytes(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler
