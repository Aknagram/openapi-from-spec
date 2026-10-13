from __future__ import annotations

import json
import re

from openapi_from_spec.models import Gap, IRField, Operation, Segment
from openapi_from_spec.naming import kebab, parse_target
from openapi_from_spec.segment import _match_method
from openapi_from_spec.tables import rows_to_fields

_SECTIONS = (
    ("параметры запроса", "request"),
    ("входные параметры", "request"),
    ("параметры пути", "path"),
    ("заголовки", "header"),
    ("тело запроса", "body"),
    ("параметры ответа", "response"),
    ("выходные параметры", "response"),
    ("пример ответа", "example_response"),
    ("пример запроса", "example_request"),
    ("описание", "description"),
)


def _section_kind(line: str) -> str | None:
    folded = re.sub(r"\s+", " ", line.strip().lower()).rstrip(":")
    if not folded or len(line.strip()) > 80:
        return None
    for prefix, kind in _SECTIONS:
        if folded == prefix or folded.startswith(prefix + " "):
            return kind
    return None


def _decode_json(text: str) -> object | None:
    decoder = json.JSONDecoder()
    start = 0
    while True:
        brace = text.find("{", start)
        bracket = text.find("[", start)
        candidates = [item for item in (brace, bracket) if item >= 0]
        if not candidates:
            return None
        index = min(candidates)
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            start = index + 1
            continue
        return value


def _head_method(segment: Segment) -> tuple[str | None, str, str | None, dict[str, list[str]]]:
    method = segment.http_method
    raw_name = segment.raw_name
    explicit = segment.explicit_path
    query = dict(segment.query)
    for line in segment.text.splitlines()[:40]:
        matched = _match_method(line)
        if not matched:
            continue
        verb, token = matched
        name, path, found_query = parse_target(token)
        method = method or verb
        if name and not raw_name:
            raw_name = name
        explicit = explicit or path
        for key, values in found_query.items():
            query.setdefault(key, values)
        break
    return method, raw_name, explicit, query


def _place(location_hint: str, method: str | None, field: IRField, query: dict[str, list[str]], path_names: set[str]) -> str:
    if location_hint == "response":
        return "response"
    if location_hint == "path" or field.name in path_names:
        return "path"
    if location_hint == "header":
        return "header"
    if location_hint == "body":
        return "body"
    if field.name in query:
        return "query"
    if method in {"post", "put", "patch"}:
        return "body"
    if method in {"get", "head", "delete"}:
        return "query"
    return "unresolved"


def _path_names(explicit_path: str | None, raw_name: str) -> set[str]:
    names: set[str] = set()
    for source in (explicit_path, raw_name):
        if not source:
            continue
        for match in re.finditer(r"\{([^{}]+)\}", source):
            names.add(match.group(1))
    return names


def _summary_and_description(text: str) -> tuple[str, str]:
    summary = ""
    description: list[str] = []
    mode = "preamble"
    for line in text.splitlines():
        kind = _section_kind(line)
        if kind == "description":
            mode = "description"
            continue
        if kind is not None:
            if mode == "description":
                mode = "done"
            continue
        if mode == "done":
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if mode == "preamble":
            if _match_method(stripped):
                continue
            if summary == "":
                summary = stripped
            continue
        description.append(stripped)
    return summary, "\n".join(description).strip()


def extract_segment(segment: Segment) -> Operation:
    method, raw_name, explicit_path, query = _head_method(segment)
    summary, description = _summary_and_description(segment.text)
    if segment.title and not summary:
        summary = segment.title
    buckets: dict[str, list[str]] = {
        "request": [],
        "path": [],
        "header": [],
        "body": [],
        "response": [],
    }
    example_response = ""
    example_request = ""
    current: str | None = None
    capture = ""
    for line in segment.text.splitlines():
        kind = _section_kind(line)
        if kind is not None:
            current = kind
            continue
        if current in buckets:
            buckets[current].append(line)
        elif current == "example_response":
            example_response += line + "\n"
        elif current == "example_request":
            example_request += line + "\n"
        elif current == "description":
            capture += line + "\n"
    if capture.strip() and not description:
        description = capture.strip()

    path_names = _path_names(explicit_path, raw_name)
    fields: list[IRField] = []
    gaps: list[Gap] = []
    for hint, lines in buckets.items():
        for item in rows_to_fields(lines, hint):
            item.location = _place(hint, method, item, query, path_names)
            if item.name in query:
                item.examples = list(query[item.name])
            fields.append(item)
    for name, values in query.items():
        if any(item.name == name for item in fields):
            continue
        quote = next((line.strip() for line in segment.text.splitlines() if name in line), name)
        fields.append(
            IRField(name=name, location="query", examples=list(values), origin="extracted", quote=quote)
        )
        gaps.append(Gap("Параметр есть в строке запроса, но его нет в таблице.", field=name, quote=quote))

    for name in path_names:
        if any(item.name == name for item in fields):
            continue
        fields.append(
            IRField(
                name=name,
                location="path",
                required=True,
                origin="extracted",
                quote=explicit_path or raw_name,
                description="",
            )
        )

    if method is None:
        gaps.append(Gap("Во фрагменте нет HTTP-метода (GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS)."))
    if not raw_name:
        source = (segment.title or summary or "operation").strip()
        raw_name = kebab(source)
        name_origin = "transliterated"
        gaps.append(
            Gap(
                "В фрагменте нет латинского имени метода. Путь построен транслитерацией заголовка.",
                field=raw_name,
                quote=source,
            )
        )
    else:
        name_origin = "extracted"

    for item in fields:
        if item.location == "unresolved":
            gaps.append(
                Gap(
                    "Неясно, где передаётся поле: в query или в теле. HTTP-метод не указан.",
                    field=item.name,
                    quote=item.quote,
                )
            )
        if item.tz_type is None and item.location != "response":
            gaps.append(Gap("Не указан тип поля.", field=item.name, quote=item.quote))

    head = segment.text.strip().splitlines()
    return Operation(
        http_method=method,
        raw_name=raw_name,
        summary=summary,
        description=description,
        fields=fields,
        response_example=_decode_json(example_response),
        request_example=_decode_json(example_request),
        span=(segment.start, segment.end),
        source_head=(head[0] if head else "")[:160],
        gaps=gaps,
        explicit_path=explicit_path,
        name_origin=name_origin,
        repeated_query={key for key, values in query.items() if len(values) > 1},
    )


def needs_model(operation: Operation, text: str) -> bool:
    folded = text.lower()
    marker = any(
        key in folded
        for key in ("параметры запроса", "параметры ответа", "тело запроса", "пример ответа", "пример запроса")
    )
    parsed = any(item.origin == "extracted" and item.tz_type for item in operation.fields)
    parsed = parsed or operation.response_example is not None or operation.request_example is not None
    return marker and not parsed
