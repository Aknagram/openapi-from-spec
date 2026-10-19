from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from openapi_from_spec.models import ConvertResult
from openapi_from_spec.pipeline import Settings, convert_text

_MAX_BODY = 15_000_000
_TEXT_SUFFIXES = {".md", ".txt", ".markdown"}
_FILE_FIELDS = {"file", "document"}


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


def parse_multipart(content_type: str, body: bytes) -> list[tuple[str, str | None, bytes]]:
    match = re.search(r'boundary="?([^";]+)"?', content_type, re.IGNORECASE)
    if not match:
        raise ValueError("нужен файл .md или .txt")
    marker = b"--" + match.group(1).encode("utf-8")
    parts: list[tuple[str, str | None, bytes]] = []
    for chunk in body.split(marker):
        chunk = chunk.strip(b"\r\n")
        if not chunk or chunk == b"--":
            continue
        if chunk.endswith(b"--"):
            chunk = chunk[:-2].strip(b"\r\n")
        header_blob, separator, payload = chunk.partition(b"\r\n\r\n")
        if not separator:
            header_blob, separator, payload = chunk.partition(b"\n\n")
        if not separator:
            continue
        if payload.endswith(b"\r\n"):
            payload = payload[:-2]
        headers = header_blob.decode("utf-8", errors="replace")
        disposition = ""
        for line in headers.splitlines():
            if line.lower().startswith("content-disposition:"):
                disposition = line.split(":", 1)[1].strip()
        name_match = re.search(r'name="([^"]*)"', disposition) or re.search(
            r"name=([^;\s]+)", disposition
        )
        file_match = re.search(r'filename="([^"]*)"', disposition) or re.search(
            r"filename=([^;\s]+)", disposition
        )
        name = name_match.group(1) if name_match else ""
        filename = file_match.group(1) if file_match else None
        parts.append((name, filename, payload))
    return parts


def read_multipart(content_type: str, body: bytes, title: str) -> tuple[str, str, str]:
    file_part: tuple[str | None, bytes] | None = None
    clarification = ""
    for name, filename, payload in parse_multipart(content_type, body):
        if name in _FILE_FIELDS and file_part is None:
            file_part = (filename, payload)
        elif name == "title" and payload.strip():
            title = payload.decode("utf-8")
        elif name == "clarification":
            clarification = payload.decode("utf-8")
    if file_part is None:
        raise ValueError("нужен файл .md или .txt")
    return _text_file(*file_part), title, clarification


def _text_file(filename: str | None, payload: bytes) -> str:
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix and suffix not in _TEXT_SUFFIXES:
            raise ValueError("нужен файл .md или .txt")
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("нужен файл .md или .txt в UTF-8") from exc


def dify_response(result: ConvertResult) -> tuple[str, bytes, list[tuple[str, str]]]:
    """Для chatflow: либо запрос уточнений, либо файл YAML. Оба ответа с кодом 200."""
    gaps = result.report.get("gaps") or []
    unresolved = result.report.get("unresolved") or []
    if gaps or unresolved or result.errors:
        questions: list[dict[str, str]] = []
        lines: list[str] = []
        for gap in gaps:
            if not isinstance(gap, dict):
                continue
            question = str(gap.get("question") or "").strip()
            if not question:
                continue
            operation = str(gap.get("operation") or "")
            field_name = str(gap.get("field") or "")
            questions.append({"operation": operation, "field": field_name, "question": question})
            prefix = f"{operation}.{field_name}" if operation and field_name else operation or field_name
            lines.append(f"- {prefix}: {question}" if prefix else f"- {question}")
        for item in unresolved:
            if not isinstance(item, dict):
                continue
            name = str(item.get("raw_name") or "метод")
            question = f"Не указан HTTP-метод для {name}."
            questions.append({"operation": name, "field": "", "question": question})
            lines.append(f"- {question}")
        for error in result.errors:
            question = str(error).strip()
            if not question:
                continue
            questions.append({"operation": "", "field": "", "question": question})
            lines.append(f"- {question}")
        message = "Нужны уточнения по ТЗ:\n" + "\n".join(lines)
        payload = {"result": "need_info", "message": message, "questions": questions}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return "application/json; charset=utf-8", body, []
    body = result.yaml_text.encode("utf-8")
    headers = [("Content-Disposition", 'attachment; filename="openapi.yaml"')]
    return "application/yaml; charset=utf-8", body, headers


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
            clarification = ""
            from_dify = "multipart/form-data" in content_type
            try:
                if from_dify:
                    text, title, clarification = read_multipart(content_type, raw, title)
                elif "application/json" in content_type:
                    payload = json.loads(raw.decode("utf-8"))
                    text = str(payload.get("text") or "")
                    if payload.get("title"):
                        title = str(payload["title"])
                    clarification = str(payload.get("clarification") or "")
                else:
                    text = raw.decode("utf-8")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                message = str(exc) if from_dify and str(exc) else "нужен UTF-8 текст или JSON с полем text"
                self._json(400, {"error": message})
                return
            if clarification.strip():
                text = text.rstrip() + "\n\n" + clarification.strip() + "\n"
            spec_settings = Settings(**{**settings.__dict__, "title": title})
            result = convert_text(text, spec_settings)
            if from_dify:
                response_type, body, extra = dify_response(result)
                self._bytes(200, response_type, body, extra)
                return
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

        def _bytes(
            self,
            status: int,
            content_type: str,
            body: bytes,
            extra: list[tuple[str, str]] | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            for key, value in extra or []:
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler
