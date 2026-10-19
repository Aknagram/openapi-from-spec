from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import yaml

from openapi_from_spec.compile import compile_operations, dump_yaml
from openapi_from_spec.conventions import DEFAULT_PATH_PREFIX, ERROR_RESPONSES
from openapi_from_spec.extract import extract_segment, needs_model
from openapi_from_spec.llm import (
    AgentBridgeClient,
    DifyClient,
    LlmClient,
    VllmClient,
    apply_model_extract,
    is_agent_base_url,
    nest_dotted_fields,
    prompt_pair,
)
from openapi_from_spec.models import ConvertResult, Gap, IRField, Operation, Segment
from openapi_from_spec.naming import declared_placeholder_routes, example_targets, find_template_url, parse_target
from openapi_from_spec.segment import (
    extend_segment_tail,
    find_segments,
    normalize,
    pack_windows,
    render_window,
)
from openapi_from_spec.stitch import stitch_documents
from openapi_from_spec.validate import schema_errors

_VERBS = {"get", "post", "put", "patch", "delete", "head", "options"}
_GENERIC_FIELDS = {
    "id",
    "name",
    "description",
    "status",
    "type",
    "code",
    "user",
    "created_at",
    "updated_at",
}


@dataclass
class Settings:
    title: str = "API"
    version: str = "1.0.0"
    path_prefix: str = DEFAULT_PATH_PREFIX
    error_templates: bool = True
    server_url: str | None = None
    max_window_chars: int = 12000
    window_overlap: int = 800
    vllm_base_url: str | None = None
    vllm_model: str | None = None
    vllm_api_key: str | None = None
    dify_base_url: str | None = None
    dify_api_key: str | None = None
    dify_user: str = "openapi-from-spec"
    timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            title=os.environ.get("OPENAPI_TITLE") or "API",
            version=os.environ.get("OPENAPI_VERSION") or "1.0.0",
            path_prefix=os.environ.get("OPENAPI_PATH_PREFIX") or DEFAULT_PATH_PREFIX,
            server_url=os.environ.get("OPENAPI_SERVER_URL") or None,
            vllm_base_url=os.environ.get("OPENAPI_VLLM_BASE_URL") or None,
            vllm_model=os.environ.get("OPENAPI_VLLM_MODEL") or None,
            vllm_api_key=os.environ.get("OPENAPI_VLLM_API_KEY") or None,
            dify_base_url=os.environ.get("OPENAPI_DIFY_BASE_URL") or None,
            dify_api_key=os.environ.get("OPENAPI_DIFY_API_KEY") or None,
            dify_user=os.environ.get("OPENAPI_DIFY_USER") or "openapi-from-spec",
        )


class _Calls:
    def __init__(self, inner: LlmClient) -> None:
        self.inner = inner
        self.count = 0

    def complete_json(self, *, task: str, system: str, user: str) -> dict:
        self.count += 1
        return self.inner.complete_json(task=task, system=system, user=user)


def convert_text(text: str, settings: Settings | None = None, llm: LlmClient | None = None) -> ConvertResult:
    settings = settings or Settings.from_env()
    text = normalize(text)
    client = llm or _client_from_settings(settings)
    counted = _Calls(client) if client is not None else None
    if counted is not None and text.strip():
        return _convert_with_model(text, settings, counted)
    document_gaps: list[Gap] = []
    if not text.strip():
        document_gaps.append(Gap("Документ пуст."))
    else:
        document_gaps.append(
            Gap(
                "Документ на методы режет только модель. "
                "Задайте OPENAPI_DIFY_BASE_URL и OPENAPI_DIFY_API_KEY "
                "или OPENAPI_VLLM_BASE_URL и OPENAPI_VLLM_MODEL."
            )
        )
    spec, _notes = stitch_documents([], title=settings.title, version=settings.version)
    report = _report(text, [], [], document_gaps, [], settings, 0)
    return ConvertResult(yaml_text=dump_yaml(spec), spec=spec, report=report, errors=[])


def _client_from_settings(settings: Settings) -> LlmClient | None:
    if settings.dify_base_url and settings.dify_api_key:
        return DifyClient(
            settings.dify_base_url,
            settings.dify_api_key,
            settings.dify_user,
            settings.timeout_seconds,
        )
    if is_agent_base_url(settings.vllm_base_url):
        return AgentBridgeClient(timeout=max(settings.timeout_seconds, 1800.0))
    if settings.vllm_base_url and settings.vllm_model:
        return VllmClient(
            settings.vllm_base_url,
            settings.vllm_model,
            settings.vllm_api_key,
            settings.timeout_seconds,
        )
    return None


def _empty_segment_detail(errors: list[str]) -> str:
    if errors and all(_transport_error(item) for item in errors):
        return f"Модель недоступна: {errors[0]}"
    detail = "Модель не выделила ни одного фрагмента с дословными границами."
    if errors:
        return f"{detail} {errors[0]}"
    return detail


def _transport_error(message: str) -> bool:
    text = message.lower()
    marks = ("10061", "10054", "connection", "connect", "timed out", "timeout", "не установлено", "разорвал")
    return any(mark in text for mark in marks)


def _convert_with_model(text: str, settings: Settings, llm: _Calls) -> ConvertResult:
    document_gaps: list[Gap] = []
    segments, segment_errors = _segment_with_model(text, llm, settings)
    parts: list[dict[str, Any]] = []
    operations: list[Operation] = []
    long_operations: list[Operation] = []
    wrote_yaml = False
    for segment in segments:
        if _is_long_method(segment, settings):
            long_operations.append(_extract_long_segment(segment, llm, settings))
            continue
        document, gaps, found = _yaml_for_segment(segment, llm)
        document_gaps.extend(gaps)
        if document is not None:
            parts.append(document)
            wrote_yaml = True
        operations.extend(found)
    conventions = ["Границы фрагментов взяты из цитат модели и проверены по тексту окна."]
    if long_operations:
        compiled, notes = compile_operations(
            long_operations,
            title=settings.title,
            version=settings.version,
            path_prefix=settings.path_prefix,
            error_templates=settings.error_templates,
            server_url=settings.server_url,
        )
        parts.append(compiled)
        operations.extend(long_operations)
        conventions.append("Длинный метод собран извлечением по окнам и одним YAML.")
        conventions.extend(notes)
    if wrote_yaml:
        conventions.append("YAML каждого короткого фрагмента написала модель. Код сшил paths и одинаковые схемы.")
    spec, notes = stitch_documents(parts, title=settings.title, version=settings.version)
    if segments:
        _tidy_surface(spec, text)
        document_gaps.extend(_cover_unwritten_requests(text, spec, llm))
        finalize_spec(spec, text)
    else:
        detail = _empty_segment_detail(segment_errors)
        document_gaps.append(Gap(detail))
    document_gaps.extend(Gap(note) for note in notes)
    errors = schema_errors(spec)
    known = {gap.question for gap in document_gaps}
    for error in errors:
        if error not in known:
            document_gaps.append(Gap(error))
    report = _report(
        text,
        segments,
        operations,
        document_gaps,
        conventions,
        settings,
        llm.count,
        templates=[],
    )
    return ConvertResult(yaml_text=dump_yaml(spec), spec=spec, report=report, errors=errors)


def _is_long_method(segment: Segment, settings: Settings) -> bool:
    return bool(segment.http_method) and len(segment.text) > settings.max_window_chars


def _yaml_for_segment(segment: Segment, llm: _Calls) -> tuple[dict[str, Any] | None, list[Gap], list[Operation]]:
    system, user = prompt_pair("openapi", segment.text)
    try:
        payload = llm.complete_json(task="openapi", system=system, user=user)
    except Exception as exc:
        gap = Gap(f"Вызов модели для YAML не выполнен: {exc}")
        return None, [gap], [_model_operation(segment)]
    document, raw, errors = _read_yaml_payload(payload)
    if errors:
        system, user = prompt_pair("openapi", segment.text, previous=raw, errors=errors)
        try:
            payload = llm.complete_json(task="openapi", system=system, user=user)
        except Exception as exc:
            errors.append(f"Повторный вызов модели не выполнен: {exc}")
        else:
            repaired, _raw, repaired_errors = _read_yaml_payload(payload)
            if repaired is not None:
                document = repaired
            errors = repaired_errors
    if isinstance(document, dict):
        _align_document_with_source(document, segment.text)
    gaps = [Gap(item) for item in errors]
    return document, gaps, _operations_from_document(segment, document)


def _fragment_at(text: str, at: int) -> tuple[int, int]:
    """Окно непокрытого вызова начинается с его строки, а не с более раннего запроса под тем же заголовком."""
    start = at
    headings = list(re.finditer(r"(?m)^#{1,3} ", text[:start]))
    if headings:
        heading = headings[-1].start()
        between = text[heading:start]
        if _OWN_REQUEST.search(between) is None and not any(_table_request_rows(between)):
            start = heading
    end = len(text)
    following = re.search(r"(?m)^#{1,3} ", text[at + 1 :])
    if following:
        end = at + 1 + following.start()
    return start, _limit_to_own_request(text, start, end)


def _fragment_for_request(text: str, match: re.Match[str]) -> tuple[int, int]:
    return _fragment_at(text, match.start())


def _unwritten_calls(text: str) -> list[tuple[int, str, str]]:
    found: list[tuple[int, str, str]] = []
    for match in _OWN_REQUEST.finditer(text):
        path = _clean_request_path(_request_path(match.group(2)), text)
        if path:
            found.append((match.start(), match.group(1).lower(), path))
    for match in _PLACEHOLDER_REQUEST.finditer(text):
        tail = match.group(2).split("}", 1)[1]
        path = _clean_request_path("/" + tail.lstrip("/"), text)
        if path and path != "/":
            found.append((match.start(), match.group(1).lower(), path))
    found.extend(_table_request_rows(text))
    return found


def _cover_unwritten_requests(text: str, spec: dict[str, Any], llm: _Calls) -> list[Gap]:
    """Строка или ячейка вызова, которой нет в собранных paths, уходит модели отдельным фрагментом."""
    gaps: list[Gap] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for at, method, path in _unwritten_calls(text):
        if not path:
            continue
        key = (method, _call_parts(path))
        if key in seen or _route_covered(spec, method, path):
            continue
        seen.add(key)
        start, end = _fragment_at(text, at)
        if end <= start:
            continue
        segment = Segment(
            start=start,
            end=end,
            text=text[start:end],
            http_method=method,
            explicit_path=path,
            anchor="request-line",
        )
        document, segment_gaps, _operations = _yaml_for_segment(segment, llm)
        gaps.extend(segment_gaps)
        if not isinstance(document, dict):
            continue
        info = spec.get("info") if isinstance(spec.get("info"), dict) else {}
        merged, _notes = stitch_documents(
            [spec, document],
            title=str((info or {}).get("title") or "API"),
            version=str((info or {}).get("version") or "1.0.0"),
        )
        spec.clear()
        spec.update(merged)
    return gaps


def _read_yaml_payload(payload: object) -> tuple[dict[str, Any] | None, str, list[str]]:
    if not isinstance(payload, dict):
        return None, "", ["Модель не вернула объект с полем yaml."]
    raw = str(payload.get("yaml") or "")
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:yaml|yml)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    if not text.strip():
        return None, raw, ["Модель не вернула YAML."]
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError:
        try:
            document = yaml.safe_load(_quote_yaml_values(text))
        except yaml.YAMLError as exc:
            return None, raw, [f"YAML фрагмента не разобран: {exc}"]
    if not isinstance(document, dict):
        return None, raw, ["YAML фрагмента не объект."]
    _normalize_model_document(document)
    return document, raw, schema_errors(document)


_SUCCESS_HEADINGS = {"успех", "success", "ok"}


def _normalize_model_document(document: dict[str, Any]) -> None:
    paths = document.get("paths")
    if not isinstance(paths, dict):
        return
    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        for method, operation in methods.items():
            if str(method).startswith("x-") or not isinstance(operation, dict):
                continue
            operation_id = operation.get("operationId")
            if isinstance(operation_id, str):
                operation["operationId"] = _strip_path_param(operation_id, str(path), operation.get("parameters"))
            _replace_success_heading(operation)


def _strip_path_param(operation_id: str, path: str, parameters: object) -> str:
    names = [match for match in re.findall(r"\{([^{}]+)\}", path)]
    if isinstance(parameters, list):
        for item in parameters:
            if isinstance(item, dict) and item.get("in") == "path" and isinstance(item.get("name"), str):
                names.append(item["name"])
    for name in names:
        suffix = name[:1].upper() + name[1:]
        if suffix and operation_id.endswith(suffix) and operation_id != suffix:
            operation_id = operation_id[: -len(suffix)]
    return operation_id


def _quote_yaml_values(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^(\s*[^:#\n][^:\n]*:\s+)(.+)$", line)
        value = match.group(2) if match else ""
        if match and value[:1] not in {"'", '"', "|", ">", "{", "[", "*"} and (": " in value or "•" in value):
            lines.append(match.group(1) + "'" + value.replace("'", "''") + "'")
        else:
            lines.append(line)
    return "\n".join(lines)


def _route_parts(path: str) -> list[str]:
    parts = []
    for part in _path_segments(path):
        if part.lower().endswith(".json"):
            part = part[:-5]
        if part:
            parts.append(part)
    if len(parts) >= 2 and parts[0] == "api" and re.fullmatch(r"v\d+", parts[1]):
        return parts[2:]
    return parts


def _same_route(left: str, right: str) -> bool:
    one = _route_parts(left)
    other = _route_parts(right)
    if _segments_match(one, other):
        return True
    if (
        len(one) == len(other) + 1
        and (one[-1].startswith("{") or one[-1].isdigit())
        and _segments_match(one[:-1], other)
    ):
        return True
    if (
        len(other) == len(one) + 1
        and (other[-1].startswith("{") or other[-1].isdigit())
        and _segments_match(other[:-1], one)
    ):
        return True
    return False


def _route_covered(spec: dict[str, Any], method: str, path: str) -> bool:
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return False
    for existing, methods in paths.items():
        if not isinstance(methods, dict) or method not in methods:
            continue
        if _same_route(str(existing), path) or _example_version_of(path, str(existing)):
            return True
    return False


def _merge_missed_examples(spec: dict[str, Any], source: str) -> None:
    """Добавляет операцию, если модель пропустила строку примера `GET /path`."""
    missed = [
        segment
        for segment in find_segments(source)
        if segment.http_method and segment.explicit_path and segment.explicit_path.startswith("/")
        and not _route_covered(spec, segment.http_method, segment.explicit_path)
    ]
    if not missed:
        return
    info = spec.get("info") if isinstance(spec.get("info"), dict) else {}
    operations = [_finish_operation(segment, None, Settings()) for segment in missed]
    compiled, _notes = compile_operations(
        operations,
        title=str(info.get("title") or "API"),
        version=str(info.get("version") or "1.0.0"),
    )
    paths = spec.setdefault("paths", {})
    if not isinstance(paths, dict):
        return
    for path, methods in (compiled.get("paths") or {}).items():
        if not isinstance(methods, dict):
            continue
        for method, operation in methods.items():
            if str(method).startswith("x-") or not isinstance(operation, dict):
                continue
            if _route_covered(spec, str(method), str(path)):
                continue
            paths.setdefault(str(path), {})[str(method)] = operation
    schemas = ((compiled.get("components") or {}).get("schemas") or {})
    if not schemas:
        return
    components = spec.setdefault("components", {})
    if not isinstance(components, dict):
        return
    target = components.setdefault("schemas", {})
    if isinstance(target, dict):
        for name, schema in schemas.items():
            target.setdefault(name, schema)


def _polish_model_draft(spec: dict[str, Any], source: str) -> None:
    _unwrap_bracket_urls(spec)
    _drop_repeated_parameter_descriptions(spec)
    _fix_status_schema_refs(spec)
    _fill_empty_example_schemas(spec)
    _keep_evidenced_tags(spec, source)
    _apply_example_nullability(spec)
    _complete_named_statuses(spec, source)
    _fill_empty_success_bodies(spec, source)
    _fill_missing_request_bodies(spec, source)
    _add_curl_headers(spec, source)
    _drop_unnamed_templates(spec, source)
    _fill_property_descriptions(spec, source)
    _apply_example_nullability(spec)
    _unwrap_bracket_urls(spec)
    _drop_duplicate_example(spec)


def _unwrap_bracket_urls(node: object) -> None:
    if isinstance(node, dict):
        for key, value in list(node.items()):
            if isinstance(value, str):
                node[key] = re.sub(r"<(https?://[^>\s]+)>", r"\1", value)
            else:
                _unwrap_bracket_urls(value)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            if isinstance(value, str):
                node[index] = re.sub(r"<(https?://[^>\s]+)>", r"\1", value)
            else:
                _unwrap_bracket_urls(value)


def _source_marks_required(source: str, name: str) -> bool:
    for match in re.finditer(re.escape(name), source):
        window = source[max(0, match.start() - 160) : match.end() + 160].lower()
        if "обязатель" in window or re.search(r"\brequired\b", window):
            return True
    return False


def _relax_example_query_params(spec: dict[str, Any], source: str) -> None:
    names: set[str] = set()
    for match in re.finditer(r"\?([^\s'\"<>\\]+)", source):
        names.update(key for key, _value in parse_qsl(match.group(1), keep_blank_values=True) if key)
    if not names:
        return
    for operation in _operations(spec):
        parameters = operation.get("parameters")
        if not isinstance(parameters, list):
            continue
        for parameter in parameters:
            if not isinstance(parameter, dict) or parameter.get("in") != "query":
                continue
            name = str(parameter.get("name") or "")
            if name not in names or _source_marks_required(source, name):
                continue
            description = str(parameter.get("description") or "").lower()
            if "обязатель" in description:
                continue
            parameter.pop("required", None)


def _drop_repeated_parameter_descriptions(spec: dict[str, Any]) -> None:
    for operation in _operations(spec):
        parameters = operation.get("parameters")
        if not isinstance(parameters, list):
            continue
        descriptions = [
            str(item.get("description"))
            for item in parameters
            if isinstance(item, dict) and item.get("description")
        ]
        for parameter in parameters:
            if not isinstance(parameter, dict):
                continue
            description = str(parameter.get("description") or "")
            name = str(parameter.get("name") or "")
            if descriptions.count(description) > 1 and name.lower() not in description.lower():
                parameter.pop("description", None)


def _fix_status_schema_refs(spec: dict[str, Any]) -> None:
    schemas = ((spec.get("components") or {}).get("schemas") or {})
    for operation in _operations(spec):
        responses = operation.get("responses")
        if not isinstance(responses, dict):
            continue
        for code, response in responses.items():
            if not isinstance(response, dict):
                continue
            content = response.get("content")
            if not isinstance(content, dict):
                continue
            for media in content.values():
                if not isinstance(media, dict):
                    continue
                schema = media.get("schema")
                if not isinstance(schema, dict):
                    continue
                ref = schema.get("$ref")
                if not isinstance(ref, str):
                    continue
                name = ref.rsplit("/", 1)[-1]
                embedded = re.findall(r"(?<!\d)(\d{3})(?!\d)", name)
                if not embedded or str(code) in embedded:
                    continue
                target = schemas.get(name) if isinstance(schemas, dict) else None
                properties = target.get("properties") if isinstance(target, dict) else None
                media["schema"] = (
                    {"type": "object", "properties": properties}
                    if isinstance(properties, dict)
                    else {"type": "object", "properties": {"error": {"type": "string"}}}
                )


_FRAMING_HEADERS = {"host", "content-type", "content-length", "accept"}


def _add_example_headers(spec: dict[str, Any], source: str) -> None:
    """Заголовок из примера запроса: флаг -H или строка «Имя: значение» после адреса."""
    pattern = re.compile(
        r"\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+['\"]?\s*<?(https?://[^\s'\"<>]+)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(source):
        method = match.group(1).lower()
        parsed_path = urlsplit(match.group(2)).path or ""
        block = source[match.end() : match.end() + 500]
        headers = re.findall(r"-H\s+['\"]([^:'\"]+)\s*:", block)
        headers.extend(
            name
            for name in re.findall(r"(?m)^\s*([A-Za-z][A-Za-z0-9-]*)\s*:", block)
            if name.lower() not in _FRAMING_HEADERS
        )
        if not parsed_path or not headers:
            continue
        for path, methods in (spec.get("paths") or {}).items():
            if not isinstance(methods, dict) or not _same_route(str(path), parsed_path):
                continue
            operation = methods.get(method)
            if not isinstance(operation, dict):
                continue
            _append_headers(operation, headers)


def _append_headers(operation: dict[str, Any], headers: list[str]) -> None:
    parameters = operation.setdefault("parameters", [])
    if not isinstance(parameters, list):
        return
    present = {str(item.get("name")).lower() for item in parameters if isinstance(item, dict)}
    for header in headers:
        if header.lower() in present or header.lower() in _FRAMING_HEADERS:
            continue
        schema: dict[str, Any] = {"type": "string"}
        if "key" in header.lower():
            schema["format"] = "uuid"
        parameters.append({"name": header, "in": "header", "required": True, "schema": schema})
        present.add(header.lower())


def _add_prose_headers(spec: dict[str, Any], source: str) -> None:
    """Фраза «заголовок Имя обязателен» называет параметр заголовка ближайшего метода."""
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return
    named = re.compile(r"(?:заголовок|header)\s+[`'\"]?([A-Za-z][A-Za-z0-9-]*)[`'\"]?", re.IGNORECASE)
    request_pattern = re.compile(
        r"\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+['\"]?\s*<?(https?://[^\s)'\">]+|/[^\s]+)",
        re.IGNORECASE,
    )
    for match in named.finditer(source):
        header = match.group(1)
        if header.lower() in _FRAMING_HEADERS:
            continue
        request = None
        for seen in request_pattern.finditer(source[: match.start()]):
            request = seen
        if request is None:
            continue
        raw_path = request.group(2)
        path = _split_example_path(raw_path) if raw_path.lower().startswith("http") else raw_path.split("?")[0]
        operation = _best_operation(paths, request.group(1).lower(), path) if path else None
        if operation is None:
            continue
        sentence = source[match.start() : match.end() + 120]
        required = re.search(r"обязател|необходим|\brequired\b", sentence, re.IGNORECASE) is not None
        parameters = operation.setdefault("parameters", [])
        if not isinstance(parameters, list):
            continue
        current = next(
            (
                item
                for item in parameters
                if isinstance(item, dict) and str(item.get("name")).lower() == header.lower()
            ),
            None,
        )
        if current is None:
            parameters.append(
                {"name": header, "in": "header", "required": required, "schema": {"type": "string"}}
            )
        elif required:
            current["required"] = True


def _section_route(text: str) -> str:
    found: list[str] = []
    pattern = re.compile(
        r"https?://[^\s'\"<>]+|(?<![\w])/[A-Za-z0-9_{}.-]+(?:/[A-Za-z0-9_{}.-]+)+",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        token = match.group(0).rstrip(".,)")
        path = urlsplit(token).path if token.lower().startswith("http") else token.split("?", 1)[0]
        if path.endswith(".json"):
            path = path[: -len(".json")]
        if path.count("/") >= 1:
            found.append(path)
    return found[-1] if found else ""


def _computed_fields(source: str) -> list[tuple[str, str, str]]:
    """Поле, про которое сказано, что клиент его не задаёт: имя, фраза ошибки, путь раздела."""
    marker = re.compile(
        r"вычисляем|не переда[её]тся в запросах|не может быть (?:задано|установлено) клиентом",
        re.IGNORECASE,
    )
    named = re.compile(r"поле\s+\*?`?([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
    found: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for match in marker.finditer(source):
        window = source[max(0, match.start() - 160) : match.end() + 80]
        for name in named.findall(window):
            if name.lower() in seen:
                continue
            seen.add(name.lower())
            phrase = ""
            quoted = re.search(
                rf'["«]([^"\n]*\b{re.escape(name)}\b[^"\n]*не может быть (?:задано|установлено) клиентом[^"\n]*)',
                source,
                re.IGNORECASE,
            )
            if quoted:
                phrase = quoted.group(1)
            else:
                plain = re.search(
                    rf"поле\s+\*?{re.escape(name)}\*?[^.|\n]*не может быть (?:задано|установлено) клиентом",
                    source,
                    re.IGNORECASE,
                )
                if plain:
                    phrase = plain.group(0)
            phrase = re.sub(r"\s+", " ", phrase).strip(" \"'")
            before = source[max(0, match.start() - 1800) : match.start()]
            found.append((name, phrase, _section_route(before)))
    return found


def _note_computed_fields(spec: dict[str, Any], source: str) -> None:
    for name, phrase, route in _computed_fields(source):
        for path, methods in (spec.get("paths") or {}).items():
            if not isinstance(methods, dict):
                continue
            if route and not _same_route(str(path), route):
                continue
            for operation in methods.values():
                if not isinstance(operation, dict):
                    continue
                if not route:
                    schema = _request_schema(spec, operation)
                    properties = schema.get("properties") if isinstance(schema, dict) else None
                    if not isinstance(properties, dict) or name not in properties:
                        continue
                _drop_request_property(spec, operation, name)
                if not phrase:
                    continue
                responses = operation.setdefault("responses", {})
                if not isinstance(responses, dict):
                    continue
                response = responses.get("400")
                if not isinstance(response, dict):
                    responses["400"] = {"description": phrase}
                    continue
                if "$ref" in response:
                    continue
                description = str(response.get("description") or "").rstrip()
                if phrase not in description:
                    if description and description[-1] not in ".!?":
                        description += "."
                    response["description"] = (description + " " + phrase).strip()


def _drop_request_property(spec: dict[str, Any], operation: dict[str, Any], name: str) -> None:
    schema = _request_schema(spec, operation)
    if not isinstance(schema, dict):
        return
    properties = schema.get("properties")
    if isinstance(properties, dict):
        properties.pop(name, None)
    required = schema.get("required")
    if isinstance(required, list) and name in required:
        required.remove(name)


_AUTHOR_REMARK = re.compile(
    r"(указываем|опишем|описываем|по аналогии|логичнее было бы|для примера я|я специально)",
    re.IGNORECASE,
)


def _plain(text: str) -> str:
    return re.sub(r"[*_`]+", "", text).strip()


def _is_section_heading(text: str, source: str) -> bool:
    plain = _plain(text).strip()
    if not plain or len(plain) > 80:
        return False
    if plain.startswith("#"):
        body = re.sub(r"^#+\s*", "", plain).strip()
        if body and re.search(rf"(?m)^#+\s*{re.escape(body)}", source):
            return True
    if plain.endswith((".", "!", "?")):
        return False
    return re.search(
        rf"(?:^|\n)\s*(?:#+\s+|\*\*\s*)?{re.escape(plain)}\s*(?:\*\*)?\s*(?:$|\n)",
        source,
        re.IGNORECASE,
    ) is not None


def _sentence_after(source: str, heading: str) -> str:
    plain = _plain(heading)
    if not plain:
        return ""
    match = re.search(rf"(?:#+\s+|\*\*\s*)?{re.escape(plain)}(?:\*\*)?", source, re.IGNORECASE)
    if match is None:
        return ""
    lines = [
        line.strip()
        for line in source[match.end() : match.end() + 500].splitlines()
        if line.strip() and not line.strip().startswith("|")
    ]
    text = _plain(" ".join(lines))
    text = re.split(r"\bURL\s*:", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = re.split(r"разрешенные запросы", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
    if sentence and _AUTHOR_REMARK.search(sentence):
        rest = text[len(sentence) :].strip()
        sentence = re.split(r"(?<=[.!?])\s+", rest, maxsplit=1)[0].strip()
    if not sentence or _is_section_heading(sentence, source) or _AUTHOR_REMARK.search(sentence):
        return ""
    sentence = re.sub(r"^(?:в нашем случае|в данном случае)\s+", "", sentence, count=1, flags=re.IGNORECASE)
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
    return sentence[:160]


def _move_heading_success(spec: dict[str, Any], source: str) -> None:
    for operation in _operations(spec):
        responses = operation.get("responses")
        if not isinstance(responses, dict):
            continue
        success = responses.get("200")
        if not isinstance(success, dict) or not _is_section_heading(str(success.get("description") or ""), source):
            continue
        for code, response in list(responses.items()):
            if not str(code).startswith("2") or code == "200" or not isinstance(response, dict):
                continue
            if response.get("content") or not str(response.get("description") or "").startswith("Ответ"):
                continue
            content = success.get("content")
            if isinstance(content, dict):
                response["content"] = content
            response["description"] = "Успешный ответ"
            responses.pop("200", None)
            break


def _fill_empty_example_schemas(spec: dict[str, Any]) -> None:
    for operation in _operations(spec):
        responses = operation.get("responses")
        if not isinstance(responses, dict):
            continue
        for response in responses.values():
            content = response.get("content") if isinstance(response, dict) else None
            if not isinstance(content, dict):
                continue
            for media in content.values():
                if not isinstance(media, dict) or not isinstance(media.get("example"), (dict, list)):
                    continue
                schema = media.get("schema")
                if schema == {} or (isinstance(schema, dict) and not schema):
                    media["schema"] = _schema_from_example(media["example"])


def _schema_from_example(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return {"type": "object", "properties": {str(key): _schema_from_example(item) for key, item in value.items()}}
    if isinstance(value, list):
        return {"type": "array", "items": _schema_from_example(value[0]) if value else {}}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    return {"type": "string"}


def _replace_section_summaries(spec: dict[str, Any], source: str) -> None:
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return
    for methods in paths.values():
        if not isinstance(methods, dict):
            continue
        for operation in methods.values():
            if not isinstance(operation, dict):
                continue
            summary = str(operation.get("summary") or "")
            description = _plain(str(operation.get("description") or ""))
            first = ""
            if description and not _is_section_heading(description, source):
                first = description.split("\n", 1)[0].strip()
                if _AUTHOR_REMARK.search(first):
                    first = ""
            if _is_section_heading(summary, source):
                sentence = _sentence_after(source, summary) or first
                if sentence:
                    operation["summary"] = sentence[:160]
            elif description and _is_section_heading(description, source):
                operation.pop("description", None)
            responses = operation.get("responses")
            if not isinstance(responses, dict):
                continue
            for response in responses.values():
                if isinstance(response, dict) and _is_section_heading(str(response.get("description") or ""), source):
                    response["description"] = "Успешный ответ"


def _drop_author_remarks(spec: dict[str, Any]) -> None:
    """Убирает реплику автора шаблона. Она объясняет, как писать ТЗ, а не поведение метода."""
    for operation in _operations(spec):
        description = str(operation.get("description") or "")
        if description and _AUTHOR_REMARK.search(description):
            operation.pop("description", None)
        summary = str(operation.get("summary") or "")
        if summary and _AUTHOR_REMARK.search(summary):
            replacement = str(operation.get("description") or "").split(".", 1)[0].strip()
            if replacement and not _AUTHOR_REMARK.search(replacement):
                operation["summary"] = replacement[:160]
            else:
                operation.pop("summary", None)


def _operations(spec: dict[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return found
    for methods in paths.values():
        if not isinstance(methods, dict):
            continue
        for method, operation in methods.items():
            if not str(method).startswith("x-") and isinstance(operation, dict):
                found.append(operation)
    return found


def finalize_spec(spec: dict[str, Any], source: str) -> None:
    """Приводит YAML к синтаксису OpenAPI. Пустые подписи дописывает из фрагмента, готовые решения модели не заменяет."""
    _tidy_surface(spec, source)
    _brace_colon_paths(spec)
    _reuse_sibling_path_params(spec)
    _move_unresolved_path_params(spec)
    _restore_request_prefix(spec, source)
    _extend_unique_document_suffix(spec, source)
    _fold_json_suffix_paths(spec)
    _collapse_duplicate_routes(spec, source)
    _apply_document_templates(spec, source)
    _drop_ungrounded_servers(spec, source)
    _strip_repeated_server_prefix(spec)
    _align_operations_to_requests(spec, source)
    _apply_document_identity(spec, source)
    _polish_model_draft(spec, source)
    _apply_status_catalog(spec, source)
    _fold_example_versions(spec, source)
    _move_unresolved_path_params(spec)
    _strip_emphasis(spec)
    _drop_unused_schemas(spec)


def _brace_colon_paths(spec: dict[str, Any]) -> None:
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return
    rewritten: dict[str, Any] = {}
    for path, methods in paths.items():
        braced = re.sub(r":([A-Za-z][A-Za-z0-9]*)", r"{\1}", str(path))
        current = rewritten.setdefault(braced, {})
        if isinstance(methods, dict):
            current.update(methods)
            if isinstance(current, dict):
                for method, operation in list(current.items()):
                    if str(method).startswith("x-") or not isinstance(operation, dict):
                        continue
                    for name in re.findall(r"\{([^{}]+)\}", braced):
                        _ensure_path_param(operation, name, "")
    spec["paths"] = rewritten


def _method_headings(source: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for match in re.finditer(
        r"^#+\s*\d+(?:\.\d+)+\.?([A-Za-z][A-Za-z0-9]*)\s*[-–—]\s*(.+?)\s*$",
        source,
        re.MULTILINE,
    ):
        found[match.group(1).lower()] = match.group(2).strip()
    return found


def _move_unresolved_path_params(spec: dict[str, Any]) -> None:
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return
    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        names = set(re.findall(r"\{([^{}]+)\}", str(path)))
        for method, operation in methods.items():
            if str(method).startswith("x-") or not isinstance(operation, dict):
                continue
            parameters = operation.get("parameters")
            if not isinstance(parameters, list):
                continue
            for item in list(parameters):
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "")
                if item.get("in") == "query" and (name in names or _UUID.fullmatch(name)):
                    parameters.remove(item)
                    continue
                if item.get("in") != "path" or name in names:
                    continue
                if names:
                    description = str(item.get("description") or "").strip()
                    parameters.remove(item)
                    if description and len(names) == 1:
                        kept_name = next(iter(names))
                        for kept in parameters:
                            if (
                                isinstance(kept, dict)
                                and kept.get("in") == "path"
                                and kept.get("name") == kept_name
                                and not str(kept.get("description") or "").strip()
                            ):
                                kept["description"] = description
                else:
                    item["in"] = "query"


def _align_method_summaries(spec: dict[str, Any], source: str) -> None:
    headings = _method_headings(source)
    paths = spec.get("paths")
    if not headings or not isinstance(paths, dict):
        return
    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        for method, operation in methods.items():
            if str(method).startswith("x-") or not isinstance(operation, dict):
                continue
            summary = str(operation.get("summary") or "")
            named = re.search(r"([A-Za-z][A-Za-z0-9]+)\s*[-–—]", summary)
            if named is None:
                continue
            current = named.group(1).lower()
            best = _heading_for_path(str(path), str(operation.get("operationId") or ""), headings)
            if best and current != best[0] and not best[0].startswith(current):
                operation["summary"] = best[1]


def _heading_for_path(path: str, operation_id: str, headings: dict[str, str]) -> tuple[str, str] | None:
    tokens = {piece.lower() for piece in _path_segments(path) if not piece.startswith("{")}
    operation = re.sub(r"[^a-z0-9]", "", operation_id.lower())
    best: tuple[int, str, str] | None = None
    for name, phrase in headings.items():
        folded = name.lower()
        score = sum(3 for token in tokens if token and token in folded)
        if operation and (operation in folded or folded in operation):
            score += 5
        if best is None or score > best[0]:
            best = (score, folded, phrase)
    if best is None or best[0] < 5:
        return None
    return best[1], best[2]


def _format_tables(source: str) -> list[dict[str, str]]:
    tables: list[dict[str, str]] = []
    name_column: int | None = None
    format_column: int | None = None
    current: dict[str, str] = {}
    for line in source.splitlines():
        if "|" not in line:
            if current:
                tables.append(current)
                current = {}
            name_column = None
            format_column = None
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or re.fullmatch(r"[:\-\s]+", cells[0]):
            continue
        lowered = [cell.lower().replace(" ", "") for cell in cells]
        if any("наименован" in cell for cell in lowered) and any(
            cell.startswith("формат") or cell.startswith("тип") for cell in lowered
        ):
            if current:
                tables.append(current)
            current = {}
            name_column = next(index for index, cell in enumerate(lowered) if "наименован" in cell)
            format_column = next(
                index for index, cell in enumerate(lowered) if cell.startswith("формат") or cell.startswith("тип")
            )
            continue
        if name_column is None or format_column is None or format_column >= len(cells):
            continue
        name = re.sub(r"\s+", "", cells[name_column])
        if not name or name.lower() in {"наименование", "parameter"}:
            continue
        current[name] = re.sub(r"\s+", "", cells[format_column])
    if current:
        tables.append(current)
    return tables


def _schema_format_name(cell: str) -> str | None:
    if not re.fullmatch(r"[A-Z][A-Za-z0-9]+", cell):
        return None
    if cell.lower() in {"string", "integer", "boolean", "number", "object", "array"}:
        return None
    return cell


def _align_table_objects(spec: dict[str, Any], source: str) -> None:
    fields = {name: fmt for table in _format_tables(source) for name, fmt in table.items()}
    schema_fields = {name: fmt for name, fmt in fields.items() if _schema_format_name(fmt)}
    if not schema_fields:
        return
    renames = {f"{name}s": name for name in schema_fields if f"{name}s" not in schema_fields}
    _rename_schema_fields(spec, renames)
    _unwrap_schema_fields(spec, schema_fields)
    _drop_shape_examples(spec)


def _rename_schema_fields(node: object, renames: dict[str, str]) -> None:
    if isinstance(node, list):
        for child in node:
            _rename_schema_fields(child, renames)
        return
    if not isinstance(node, dict):
        return
    properties = node.get("properties")
    if isinstance(properties, dict):
        for old, new in renames.items():
            if old in properties and new not in properties:
                properties[new] = properties.pop(old)
        required = node.get("required")
        if isinstance(required, list):
            node["required"] = [renames.get(str(name), str(name)) for name in required]
    for child in node.values():
        _rename_schema_fields(child, renames)


def _unwrap_schema_fields(spec: dict[str, Any], fields: dict[str, str]) -> None:
    schemas = (spec.get("components") or {}).get("schemas") or {}

    def walk(node: object) -> None:
        if isinstance(node, list):
            for child in node:
                walk(child)
            return
        if not isinstance(node, dict):
            return
        properties = node.get("properties")
        if isinstance(properties, dict):
            for name, fmt in fields.items():
                schema_name = _schema_format_name(fmt)
                property_schema = properties.get(name)
                if schema_name is None or not isinstance(property_schema, dict) or schema_name not in schemas:
                    continue
                if property_schema.get("$ref") == f"#/components/schemas/{schema_name}":
                    continue
                properties[name] = {"$ref": f"#/components/schemas/{schema_name}"}
        for key, child in node.items():
            if key != "example":
                walk(child)

    walk(spec)


def _loose_json_objects(source: str) -> list[dict[str, Any]]:
    """Читает JSON примеров, даже если блоки разорваны и перед скобкой стоит запятая."""
    fences = re.findall(r"```[^\n]*\n(.*?)```", source, re.DOTALL)
    found: list[dict[str, Any]] = []
    buffer = ""
    for fence in fences:
        buffer = f"{buffer}\n{fence}".strip()
        payload = _loads_loose_object(buffer)
        if isinstance(payload, dict):
            found.append(payload)
            buffer = ""
        elif len(buffer) > 20000:
            buffer = ""
    if not found:
        payload = _loads_loose_object(source)
        if isinstance(payload, dict):
            found.append(payload)
    return found


def _wider_status_example(before: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Короткий объект у строки статуса не отбрасывает поля более полного примера в том же методе."""
    merged = payload
    for fence in re.findall(r"```[^\n]*\n(.*?)```", before, re.DOTALL):
        obj = _loads_loose_object(fence)
        if not isinstance(obj, dict) or not set(payload).issubset(obj) or len(obj) <= len(merged):
            continue
        wider = dict(obj)
        wider.update(payload)
        merged = wider
    return merged


def _loads_loose_object(text: str) -> dict[str, Any] | None:
    start = 0
    while True:
        begin = text.find("{", start)
        if begin < 0:
            break
        depth = 0
        end = None
        for index in range(begin, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None:
            break
        raw = _json_text(text[begin:end])
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            start = begin + 1
            continue
        if isinstance(payload, dict):
            return payload
        start = end
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return None
    raw = _json_text(match.group(0))
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _example_lists(source: str) -> dict[str, list[Any]]:
    found: dict[str, list[Any]] = {}

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, list):
                    found[str(key)] = value
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    for payload in _loose_json_objects(source):
        walk(payload)
    return found


def _example_scalars(source: str) -> dict[str, object]:
    found: dict[str, object] = {}

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if not isinstance(value, (dict, list)) and value is not None:
                    found[str(key)] = value
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    for payload in _loose_json_objects(source):
        walk(payload)
    return found


def _as_array(schema: dict[str, Any], item_format: str) -> dict[str, Any]:
    if schema.get("type") == "array":
        return schema
    description = schema.get("description")
    schema_name = _schema_format_name(re.sub(r"\s+", "", item_format)) if item_format else None
    if schema_name:
        items: dict[str, Any] = {"$ref": f"#/components/schemas/{schema_name}"}
    elif isinstance(schema.get("$ref"), str):
        items = {"$ref": schema["$ref"]}
    elif "string" in item_format.lower() or schema.get("type") == "string":
        items = {"type": "string"}
        if schema.get("format"):
            items["format"] = schema["format"]
    else:
        items = {key: value for key, value in schema.items() if key != "description"}
    result: dict[str, Any] = {"type": "array", "items": items}
    if isinstance(description, str) and description:
        result["description"] = description
    return result


def _prefer_example_arrays(spec: dict[str, Any], source: str) -> None:
    """Пример-массив задаёт имя и форму поля, если таблица описала только тип элемента."""
    arrays = _example_lists(source)
    if not arrays:
        return
    fields = {name: fmt for table in _format_tables(source) for name, fmt in table.items()}

    def walk(node: object) -> None:
        if isinstance(node, list):
            for child in node:
                walk(child)
            return
        if not isinstance(node, dict):
            return
        properties = node.get("properties")
        if isinstance(properties, dict):
            renamed: dict[str, str] = {}
            for name in list(properties):
                schema = properties.get(name)
                if not isinstance(schema, dict):
                    continue
                plural = f"{name}s"
                if plural in arrays and name not in arrays and plural not in properties:
                    properties[plural] = _as_array(schema, fields.get(name, ""))
                    properties.pop(name, None)
                    renamed[name] = plural
                elif name in arrays:
                    properties[name] = _as_array(schema, fields.get(name, ""))
            required = node.get("required")
            if renamed and isinstance(required, list):
                node["required"] = [renamed.get(str(item), str(item)) for item in required]
        for key, child in node.items():
            if key != "example":
                walk(child)

    walk(spec.get("components"))
    walk(spec.get("paths"))


def _drop_incorrect_request_examples(spec: dict[str, Any], source: str) -> None:
    """JSON под подписью «некорректный запрос» — это пример ошибки, а не тело успешного вызова."""
    bad = _incorrect_request_objects(source)
    if not bad:
        return
    for operation in _operations(spec):
        body = operation.get("requestBody")
        if not isinstance(body, dict):
            continue
        content = body.get("content")
        if not isinstance(content, dict):
            continue
        for media in content.values():
            example = media.get("example") if isinstance(media, dict) else None
            if isinstance(example, dict) and any(_same_payload(example, payload) for payload in bad):
                media.pop("example", None)


def _incorrect_request_objects(source: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if not re.search(r"некорректн", line, re.IGNORECASE):
            continue
        chunk = "\n".join(lines[index : index + 30])
        payload = _first_json_object(chunk)
        if isinstance(payload, dict) and set(payload) - {"error", "errorMessage", "messageOutput"}:
            found.append(payload)
    return found


def _first_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for index, char in enumerate(text[start:], start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return _loads_loose_object(text[start : index + 1])
    return None


def _same_payload(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    return all(str(left[key]) == str(right[key]) for key in left)


def _prefer_attached_objects(spec: dict[str, Any]) -> None:
    """Объект в примере не остаётся массивом, если пример разбирается."""

    def walk(node: object) -> None:
        if isinstance(node, list):
            for child in node:
                walk(child)
            return
        if not isinstance(node, dict):
            return
        example = node.get("example")
        schema = node.get("schema")
        if isinstance(example, dict) and isinstance(schema, dict):
            properties = schema.get("properties")
            if isinstance(properties, dict):
                for name, value in example.items():
                    current = properties.get(name)
                    if not isinstance(value, dict) or not isinstance(current, dict) or current.get("type") != "array":
                        continue
                    items = current.get("items")
                    if isinstance(items, dict) and items.get("type") == "object":
                        properties[name] = dict(items)
                    else:
                        properties[name] = _schema_from_example(value)
        for key, child in node.items():
            if key != "example":
                walk(child)

    walk(spec.get("paths"))
    walk(spec.get("components"))


def _drop_example_captions(spec: dict[str, Any]) -> None:
    for operation in _operations(spec):
        description = str(operation.get("description") or "").strip()
        if re.match(r"пример\s+некорректн", description, re.IGNORECASE):
            operation.pop("description", None)


def _note_example_uuid_formats(spec: dict[str, Any], source: str) -> None:
    scalars = _example_scalars(source)
    if not scalars:
        return
    formats = _format_by_name(source)

    def walk(node: object) -> None:
        if isinstance(node, list):
            for child in node:
                walk(child)
            return
        if not isinstance(node, dict):
            return
        properties = node.get("properties")
        if isinstance(properties, dict):
            for name, schema in properties.items():
                value = scalars.get(str(name))
                cell = formats.get(str(name), "")
                if (
                    isinstance(schema, dict)
                    and schema.get("type") == "string"
                    and not schema.get("format")
                    and isinstance(value, str)
                    and re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", value, re.I)
                    and (not cell or "uuid" in cell.lower())
                ):
                    schema["format"] = "uuid"
        for key, child in node.items():
            if key != "example":
                walk(child)

    walk(spec.get("components"))
    walk(spec.get("paths"))


def _attach_matching_examples(spec: dict[str, Any], source: str) -> None:
    payloads = _loose_json_objects(source)
    if not payloads:
        return
    for operation in _operations(spec):
        responses = operation.get("responses")
        if not isinstance(responses, dict):
            continue
        for response in responses.values():
            if not isinstance(response, dict):
                continue
            content = response.get("content")
            if not isinstance(content, dict):
                continue
            for media in content.values():
                if not isinstance(media, dict) or media.get("example") is not None:
                    continue
                schema = _resolve_schema(spec, media.get("schema"))
                properties = schema.get("properties") if isinstance(schema, dict) else None
                if not isinstance(properties, dict):
                    continue
                names = set(properties)
                names.update(f"{name}s" for name in properties)
                for payload in payloads:
                    if len(set(payload) & names) >= 2:
                        media["example"] = payload
                        break


def _format_by_name(source: str) -> dict[str, str]:
    return {name: fmt for table in _format_tables(source) for name, fmt in table.items()}


def _documented_formats(source: str, name: str) -> list[str]:
    """Форматы принадлежат параметру, который назван в той же фразе, а не любому имени выше по тексту."""
    owner = re.compile(
        rf"(?:параметр\w*|parameter)\s+['`\"]?{re.escape(name)}['`\"]?"
        rf"|формат\w*[^.\n]{{0,60}}['`\"]{re.escape(name)}['`\"]",
        re.IGNORECASE,
    )
    for match in owner.finditer(source):
        window = source[match.start() : match.end() + 700]
        tokens: list[str] = []
        for line in window.splitlines():
            stripped = line.strip()
            if tokens and (not stripped or stripped.startswith("#")):
                break
            for left, right in re.findall(r"'([^']+)'|`([^`]+)`", line):
                token = left or right
                if token.lower() == name.lower():
                    continue
                if re.search(r"y{2,4}", token, re.IGNORECASE) and token not in tokens:
                    tokens.append(token)
        if len(tokens) >= 2:
            return tokens
    return []


def _apply_parameter_formats(spec: dict[str, Any], source: str) -> None:
    for operation in _operations(spec):
        parameters = operation.get("parameters")
        if not isinstance(parameters, list):
            continue
        for parameter in parameters:
            if not isinstance(parameter, dict) or parameter.get("in") != "query":
                continue
            formats = _documented_formats(source, str(parameter.get("name") or ""))
            schema = parameter.get("schema")
            if not isinstance(schema, dict):
                continue
            if not any("T" not in item.upper() for item in formats) or not any("T" in item.upper() for item in formats):
                if schema.get("pattern") == r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2}(\.\d{1,3})?)?Z?)?$":
                    schema.pop("pattern", None)
                description = re.sub(r"\s+Форматы:.*$", "", str(parameter.get("description") or "")).strip()
                if description:
                    parameter["description"] = description
                elif parameter.get("description"):
                    parameter.pop("description", None)
                continue
            schema.pop("format", None)
            schema["type"] = schema.get("type") or "string"
            schema["pattern"] = r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2}(\.\d{1,3})?)?Z?)?$"
            extra = "Форматы: " + ", ".join(formats)
            description = str(parameter.get("description") or "")
            if extra not in description and not all(item in description for item in formats):
                description = (description + " " + extra).strip()
            parameter["description"] = _collapse_repeated_format_clause(description)
        _drop_repeated_format_sentences(operation)


def _collapse_repeated_format_clause(description: str) -> str:
    match = re.search(r"\s+Форматы:\s*(.+)$", description)
    if not match:
        return description
    tokens = [item.strip().rstrip(".") for item in match.group(1).split(",") if item.strip()]
    before = description[: match.start()]
    if tokens and all(token in before for token in tokens):
        return before.rstrip()
    return description


def _drop_repeated_format_sentences(operation: dict[str, Any]) -> None:
    description = str(operation.get("description") or "")
    parameters = [item for item in (operation.get("parameters") or []) if isinstance(item, dict)]
    if not description or not parameters:
        return
    kept: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", description):
        tokens = re.findall(r"y{2,4}[-A-Za-z:.]+", sentence, re.IGNORECASE)
        drop = False
        if "формат" in sentence.lower() and tokens:
            for parameter in parameters:
                name = str(parameter.get("name") or "")
                param_text = str(parameter.get("description") or "")
                if name and name in sentence and all(token in param_text for token in tokens):
                    drop = True
        if not drop:
            kept.append(sentence)
    if kept:
        operation["description"] = " ".join(kept).strip()
    else:
        operation.pop("description", None)


def _field_notes(source: str) -> dict[str, dict[str, str]]:
    """Пояснения полей по разделу таблицы, чтобы одно имя в двух моделях не смешивалось."""
    sections: dict[str, dict[str, str]] = {}
    owner = ""
    name_column: int | None = None
    note_column: int | None = None
    last_name = ""
    for line in source.splitlines():
        if line.startswith("#"):
            owner = line.strip().lstrip("#").strip().lower()
            name_column = None
            note_column = None
            last_name = ""
            continue
        if "|" not in line:
            if line.strip():
                name_column = None
                note_column = None
                last_name = ""
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or re.fullmatch(r"[:\-\s]+", cells[0] if cells else ""):
            continue
        lowered = [cell.lower().replace(" ", "").replace("-", "") for cell in cells]
        if any("наименован" in cell or cell in {"name", "field"} for cell in lowered) and any(
            word in cell for cell in lowered for word in ("обозначен", "описан", "примечан", "description")
        ):
            name_column = next(
                index for index, cell in enumerate(lowered) if "наименован" in cell or cell in {"name", "field"}
            )
            note_column = next(
                index
                for index, cell in enumerate(lowered)
                if any(word in cell for word in ("обозначен", "описан", "примечан", "description"))
            )
            continue
        if name_column is None or note_column is None or note_column >= len(cells):
            continue
        bucket = sections.setdefault(owner, {})
        name = re.sub(r"\s+", "", cells[name_column]) if name_column < len(cells) else ""
        note = re.sub(r"\s+", " ", cells[note_column]).strip()
        if not name:
            if last_name and note:
                bucket[last_name] = (bucket.get(last_name, "") + " " + note).strip()
            continue
        if name.lower() in {"наименование", "parameter"}:
            continue
        last_name = name
        if note:
            bucket[name] = note
    return sections


def _description_note(sections: dict[str, dict[str, str]], schema_name: str, field: str) -> str:
    note = _note_for(sections, schema_name, field)
    if note or not field.endswith("s") or field.endswith("ss") or len(field) <= 3:
        return note
    return _note_for(sections, schema_name, field[:-1])


def _note_for(sections: dict[str, dict[str, str]], schema_name: str, field: str) -> str:
    folded = schema_name.lower()
    for owner, notes in sections.items():
        if folded and folded in owner.replace(" ", "") and notes.get(field):
            return notes[field]
    matches = [notes[field] for notes in sections.values() if notes.get(field)]
    if len(set(matches)) == 1:
        return matches[0]
    return ""


def _fill_property_descriptions(spec: dict[str, Any], source: str) -> None:
    sections = _field_notes(source)
    if not sections:
        return

    def fill(schema_name: str, schema: object) -> None:
        if not isinstance(schema, dict):
            return
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return
        for name, prop in properties.items():
            if not isinstance(prop, dict):
                continue
            if "$ref" not in prop and not prop.get("description"):
                note = _description_note(sections, schema_name, str(name))
                if note:
                    prop["description"] = note
            fill(schema_name, prop.get("items"))

    schemas = (spec.get("components") or {}).get("schemas")
    if isinstance(schemas, dict):
        for schema_name, schema in schemas.items():
            fill(str(schema_name), schema)
    for operation in _operations(spec):
        fill("", _request_schema(spec, operation))
        responses = operation.get("responses")
        if not isinstance(responses, dict):
            continue
        for response in responses.values():
            content = response.get("content") if isinstance(response, dict) else None
            if not isinstance(content, dict):
                continue
            for media in content.values():
                if isinstance(media, dict):
                    fill("", _resolve_schema(spec, media.get("schema")))


def _mark_nullable_from_example(schema: dict[str, Any] | None, example: object, spec: dict[str, Any]) -> None:
    if not isinstance(schema, dict):
        return
    if isinstance(example, dict):
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            resolved = _resolve_schema(spec, schema)
            properties = resolved.get("properties") if isinstance(resolved, dict) else None
            schema = resolved if isinstance(resolved, dict) else schema
        if not isinstance(properties, dict):
            return
        for name, value in example.items():
            prop = properties.get(str(name))
            if not isinstance(prop, dict):
                continue
            if value is None:
                target = _resolve_schema(spec, prop) or prop
                if isinstance(target, dict):
                    target["nullable"] = True
            else:
                _mark_nullable_from_example(prop, value, spec)
    elif isinstance(example, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for value in example:
                if isinstance(value, dict):
                    _mark_nullable_from_example(items, value, spec)


def _apply_example_nullability(spec: dict[str, Any]) -> None:
    for operation in _operations(spec):
        for key in ("requestBody", "responses"):
            node = operation.get(key)
            if not isinstance(node, dict):
                continue
            contents = []
            if key == "requestBody":
                content = node.get("content")
                if isinstance(content, dict):
                    contents.extend(content.values())
            else:
                for response in node.values():
                    if isinstance(response, dict) and isinstance(response.get("content"), dict):
                        contents.extend(response["content"].values())
            for media in contents:
                if isinstance(media, dict) and media.get("example") is not None:
                    _mark_nullable_from_example(
                        media.get("schema") if isinstance(media.get("schema"), dict) else None,
                        media["example"],
                        spec,
                    )


def _first_heading(source: str) -> str:
    generic = {"общие положения", "история изменений", "запрос", "формат ответа", "обработка ошибок"}
    for line in source.splitlines():
        if not line.startswith("#"):
            continue
        text = line.strip().lstrip("#").strip()
        if text and text.lower() not in generic:
            return text[:180]
    return ""


def _document_title(source: str) -> str:
    generic = {"общие положения", "история изменений", "запрос", "формат ответа", "обработка ошибок"}
    for line in source.splitlines():
        if not line.startswith("#"):
            continue
        text = line.strip().lstrip("#").strip()
        if len(text) >= 12 and text.lower() not in generic:
            return text[:180]
    for line in source.splitlines():
        text = line.strip()
        if text and not text.startswith(("|", "-", "#", "```")):
            return text[:180]
    return ""


def _document_version(source: str) -> str:
    lines = source.splitlines()
    for index, line in enumerate(lines):
        lowered_line = line.lower()
        if "|" not in line or not any(word in lowered_line for word in ("версия", "version", "ревизия")):
            continue
        cells = [cell.strip().lower() for cell in line.strip().strip("|").split("|")]
        column = next(
            (
                number
                for number, cell in enumerate(cells)
                if any(word in cell for word in ("версия", "version", "ревизия"))
            ),
            None,
        )
        if column is None:
            continue
        for follow in lines[index + 1 : index + 6]:
            if "|" not in follow or re.fullmatch(r"[\s|:-]+", follow.strip()):
                continue
            values = [cell.strip() for cell in follow.strip().strip("|").split("|")]
            if column < len(values) and values[column] and not re.fullmatch(r":?-{3,}:?", values[column]):
                return values[column]
            break
    return ""


def _preamble_version(source: str) -> str:
    """Версия рядом с заголовком, до первой таблицы."""
    head = "\n".join(source.splitlines()[:15])
    head = head.split("\n|", 1)[0]
    match = re.search(r"(?im)^v(?:ersion)?\s+(\d+(?:\.\d+){0,2})\s*$", head)
    return match.group(1) if match else ""


def _apply_document_identity(spec: dict[str, Any], source: str) -> None:
    info = spec.get("info")
    if not isinstance(info, dict):
        return
    title = _first_heading(source) or _document_title(source)
    current = str(info.get("title") or "")
    if current in {"", "API"} and title:
        info["title"] = title
    current_version = str(info.get("version") or "")
    preamble = _preamble_version(source)
    if preamble and current_version != preamble and (
        current_version in {"", "1.0.0"} or current_version == _document_version(source)
    ):
        info["version"] = preamble
    elif current_version == "1.0.0":
        version = _document_version(source)
        if version:
            info["version"] = version


def _drop_shape_examples(spec: dict[str, Any]) -> None:
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return
    for methods in paths.values():
        if not isinstance(methods, dict):
            continue
        for method, operation in methods.items():
            if str(method).startswith("x-") or not isinstance(operation, dict):
                continue
            responses = operation.get("responses")
            if not isinstance(responses, dict):
                continue
            for response in responses.values():
                if not isinstance(response, dict):
                    continue
                content = response.get("content")
                if not isinstance(content, dict):
                    continue
                for media in content.values():
                    if not isinstance(media, dict) or not isinstance(media.get("example"), dict):
                        continue
                    schema = _resolve_schema(spec, media.get("schema"))
                    properties = schema.get("properties") if isinstance(schema, dict) else None
                    if not isinstance(properties, dict):
                        continue
                    example = media["example"]
                    if any(_example_contradicts(spec, properties.get(key), value) for key, value in example.items()):
                        media.pop("example", None)


def _example_contradicts(spec: dict[str, Any], schema: object, value: object) -> bool:
    if not isinstance(schema, dict):
        return True
    resolved = _resolve_schema(spec, schema) or schema
    if isinstance(value, list):
        return resolved.get("type") != "array" and "items" not in resolved
    return False


def _closed_status_codes(source: str) -> set[str] | None:
    """Таблица кодов закрывает список, только если в ней есть свои формулировки, а не стандартные английские причины."""
    table = _status_code_table(source)
    if not any(text and not _is_generic_status(text) for text in table.values()):
        return None
    return set(table)


def _status_code_table(source: str) -> dict[str, str]:
    found: dict[str, str] = {}
    code_column: int | None = None
    description_column: int | None = None
    for line in source.splitlines():
        if "|" not in line:
            code_column = None
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or re.fullmatch(r"[:\-\s]+", cells[0]):
            continue
        lowered = [cell.lower() for cell in cells]
        if any("status" in cell or "код ответа" in cell for cell in lowered):
            code_column = next(index for index, cell in enumerate(lowered) if "status" in cell or "код" in cell)
            description_column = next((index for index in range(len(cells)) if index != code_column), None)
            continue
        if code_column is None or code_column >= len(cells):
            continue
        code = cells[code_column]
        if not re.fullmatch(r"[1-5]\d{2}", code):
            continue
        description = ""
        if description_column is not None and description_column < len(cells):
            description = cells[description_column]
        found[code] = description
    return found


def _apply_status_catalog(spec: dict[str, Any], source: str) -> None:
    catalog = _closed_status_codes(source)
    descriptions = _status_code_table(source)
    paths = spec.get("paths")
    if not catalog or not isinstance(paths, dict):
        _drop_templates_outside_catalog(spec, source)
        return
    for methods in paths.values():
        if not isinstance(methods, dict):
            continue
        for method, operation in methods.items():
            if str(method).startswith("x-") or not isinstance(operation, dict):
                continue
            responses = operation.setdefault("responses", {})
            if not isinstance(responses, dict):
                continue
            for code in list(responses):
                response = responses.get(code)
                if str(code) in catalog or str(code) in {"200", "201", "204"}:
                    continue
                if isinstance(response, dict) and _response_outside_closed_list(spec, response):
                    responses.pop(code, None)
            for code, description in descriptions.items():
                current = responses.get(code)
                if isinstance(current, dict) and description and _is_template_response(spec, current):
                    responses[code] = {"description": description}
                    continue
                if code in responses or code in {"200", "201", "204"}:
                    continue
                responses[code] = {"description": description or f"Ответ {code}"}
    _drop_unused_templates(spec)


def _drop_templates_outside_catalog(spec: dict[str, Any], source: str) -> None:
    catalog = _closed_status_codes(source)
    paths = spec.get("paths")
    if catalog is None or not isinstance(paths, dict):
        return
    for methods in paths.values():
        if not isinstance(methods, dict):
            continue
        for method, operation in methods.items():
            if str(method).startswith("x-") or not isinstance(operation, dict):
                continue
            responses = operation.get("responses")
            if not isinstance(responses, dict):
                continue
            for code in list(responses):
                if str(code) in catalog or str(code) in {"200", "201", "204"}:
                    continue
                response = responses.get(code)
                if isinstance(response, dict) and _response_outside_closed_list(spec, response):
                    responses.pop(code, None)
    _drop_unused_templates(spec)


def _response_outside_closed_list(spec: dict[str, Any], response: dict[str, Any]) -> bool:
    if _is_template_response(spec, response):
        return True
    description = response.get("description")
    return isinstance(description, str) and bool(description.strip()) and _is_generic_status(description)


def _drop_unused_schemas(spec: dict[str, Any]) -> None:
    """Схема, на которую не ведёт ни один $ref, не остаётся в components."""
    components = spec.get("components")
    if not isinstance(components, dict):
        return
    schemas = components.get("schemas")
    if not isinstance(schemas, dict):
        return

    def refs_in(node: object, found: set[str]) -> None:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
                found.add(ref.rsplit("/", 1)[-1])
            for child in node.values():
                refs_in(child, found)
        elif isinstance(node, list):
            for child in node:
                refs_in(child, found)

    used: set[str] = set()
    for key, value in spec.items():
        if key == "components":
            continue
        refs_in(value, used)
    pending = list(used)
    while pending:
        name = pending.pop()
        before = len(used)
        refs_in(schemas.get(name), used)
        if len(used) != before:
            pending.extend(used - set(pending) - {name})
    for name in list(schemas):
        schema = schemas.get(name)
        if name in used or not isinstance(schema, dict) or schema.get("x-origin") != "skill-template":
            continue
        schemas.pop(name, None)
    if not schemas:
        components.pop("schemas", None)
    if not components:
        spec.pop("components", None)


def _drop_unused_templates(spec: dict[str, Any]) -> None:
    responses = (spec.get("components") or {}).get("responses")
    if not isinstance(responses, dict):
        return
    used: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/components/responses/"):
                used.add(ref.rsplit("/", 1)[-1])
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(spec.get("paths"))
    for name in list(responses):
        response = responses.get(name)
        if isinstance(response, dict) and response.get("x-origin") == "skill-template" and name not in used:
            responses.pop(name, None)
    if not responses:
        spec.get("components", {}).pop("responses", None)


def _reuse_sibling_path_params(spec: dict[str, Any]) -> None:
    """Число из примера не заводит второй путь `{id}`, если рядом уже назван параметр."""
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return
    alias: dict[str, str] = {}
    for path in paths:
        segments = _path_segments(str(path))
        if not segments or segments[-1] != "{id}":
            continue
        for other in paths:
            other_segments = _path_segments(str(other))
            if (
                str(other) != str(path)
                and len(other_segments) == len(segments)
                and other_segments[:-1] == segments[:-1]
                and other_segments[-1].startswith("{")
                and other_segments[-1] != "{id}"
            ):
                alias[str(path)] = str(other)
                break
    if not alias:
        return
    merged: dict[str, Any] = {}
    for path, methods in paths.items():
        target = alias.get(str(path), str(path))
        bucket = merged.setdefault(target, {})
        if not isinstance(methods, dict) or not isinstance(bucket, dict):
            continue
        parameter = _path_segments(target)[-1].strip("{}") if str(path) in alias else ""
        for method, operation in methods.items():
            if parameter and isinstance(operation, dict):
                for item in operation.get("parameters") or []:
                    if isinstance(item, dict) and item.get("in") == "path" and item.get("name") == "id":
                        item["name"] = parameter
            previous = bucket.get(method)
            if isinstance(previous, dict) and isinstance(operation, dict) and _richness(previous) >= _richness(operation):
                continue
            bucket[method] = operation
    spec["paths"] = merged


def _path_segments(path: str) -> list[str]:
    return [piece for piece in path.split("/") if piece]


def _segments_match(left: list[str], right: list[str]) -> bool:
    if len(left) != len(right):
        return False
    for one, other in zip(left, right):
        if one == other:
            continue
        one_param = one.startswith("{") and one.endswith("}")
        other_param = other.startswith("{") and other.endswith("}")
        if one_param and other_param:
            continue
        if (one_param and other and not other_param) or (other_param and one and not one_param):
            continue
        if one.isdigit() and other_param:
            continue
        if other.isdigit() and one_param:
            continue
        return False
    return True


def _prefix_before(operation_path: str, example_path: str) -> str | None:
    operation = _path_segments(operation_path)
    example = _path_segments(example_path)
    if not operation or not example:
        return None
    for start in range(len(example)):
        rest = example[start:]
        if _segments_match(operation, rest):
            return "/" + "/".join(example[:start]) if start else ""
        if len(rest) == len(operation) + 1 and rest[-1].isdigit() and _segments_match(operation, rest[:-1]):
            return "/" + "/".join(example[:start]) if start else ""
    return None


def _aligned_path(operation_path: str, example_path: str) -> str | None:
    operation = _path_segments(operation_path)
    example = _path_segments(example_path)
    if not operation or not example:
        return None
    for start in range(len(example)):
        rest = example[start:]
        if _segments_match(operation, rest):
            return "/" + "/".join(operation)
        if len(rest) == len(operation) + 1 and rest[-1].isdigit() and _segments_match(operation, rest[:-1]):
            return "/" + "/".join([*operation, "{id}"])
    return None


def _rewrite_paths(spec: dict[str, Any], source: str) -> None:
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return
    examples = [path for _method, _origin, path in example_targets(source)]
    prefixes = [
        prefix
        for operation_path in list(paths)
        for example in examples
        if (prefix := _prefix_before(str(operation_path), example))
    ]
    prefix = max(prefixes, key=len) if prefixes else ""
    rewritten: dict[str, dict[str, Any]] = {}
    for operation_path, methods in list(paths.items()):
        if not isinstance(methods, dict):
            rewritten[str(operation_path)] = methods
            continue
        aligned = str(operation_path)
        for example in examples:
            candidate = _aligned_path(str(operation_path), example)
            if candidate and candidate.count("{") > aligned.count("{"):
                aligned = candidate
        if prefix and aligned.startswith(prefix + "/"):
            aligned = aligned[len(prefix) :] or "/"
        for method, operation in methods.items():
            if not isinstance(operation, dict) or str(method).startswith("x-"):
                continue
            if "{id}" in aligned:
                _ensure_path_param(operation, "id", source)
            current = rewritten.setdefault(aligned, {})
            previous = current.get(method)
            if isinstance(previous, dict) and _richness(previous) >= _richness(operation):
                continue
            current[method] = operation
    spec["paths"] = rewritten


def _ensure_path_param(operation: dict[str, Any], name: str, source: str) -> None:
    parameters = operation.setdefault("parameters", [])
    if not isinstance(parameters, list):
        return
    if any(isinstance(item, dict) and item.get("in") == "path" and item.get("name") == name for item in parameters):
        return
    schema: dict[str, Any] = {"type": "string"}
    for field in _table_fields(source):
        if field["name"] == name and field["schema"].get("type"):
            schema = dict(field["schema"])
            break
    parameters.insert(
        0,
        {"name": name, "in": "path", "required": True, "schema": schema},
    )


def _richness(operation: dict[str, Any]) -> int:
    score = len(str(operation.get("description") or "")) // 80
    if operation.get("requestBody"):
        score += 5
    responses = operation.get("responses")
    if not isinstance(responses, dict):
        return score
    for code, response in responses.items():
        if not isinstance(response, dict):
            continue
        if "content" in response:
            score += 3
        if str(code) not in {"400", "401", "404", "500"}:
            score += 1
        if "$ref" not in response:
            score += 1
    return score


def _drop_unevidenced_operations(spec: dict[str, Any], source: str) -> None:
    evidence = [(method, path) for method, _origin, path in example_targets(source) if method and path]
    evidence.extend(
        (method, path)
        for method, _placeholder, path, _name in declared_placeholder_routes(source)
        if method and path
    )
    paths = spec.get("paths")
    if not evidence or not isinstance(paths, dict):
        return
    for path in list(paths):
        methods = paths.get(path)
        if not isinstance(methods, dict):
            continue
        for method in list(methods):
            if str(method).startswith("x-") or not isinstance(methods.get(method), dict):
                continue
            if any(
                method == seen_method and _prefix_before(str(path), seen_path) is not None
                for seen_method, seen_path in evidence
            ):
                continue
            methods.pop(method, None)
        if not any(not str(method).startswith("x-") for method in methods):
            paths.pop(path, None)


def _table_fields(source: str) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []
    required_column: int | None = None
    type_column: int | None = None
    description_column: int | None = None
    for line in source.splitlines():
        if "|" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or re.fullmatch(r"[:\-\s]+", cells[0]):
            continue
        lowered = [cell.lower().replace(" ", "") for cell in cells]
        if any("обязат" in cell for cell in lowered) or any(cell.startswith("тип") for cell in lowered):
            required_column = next((index for index, cell in enumerate(lowered) if "обязат" in cell), required_column)
            type_column = next((index for index, cell in enumerate(lowered) if cell.startswith("тип")), type_column)
            description_column = next(
                (index for index, cell in enumerate(lowered) if "описан" in cell),
                description_column,
            )
            continue
        if required_column is None and type_column is None:
            continue
        name = re.sub(r"\[\]$", "", re.sub(r"\s+", "", cells[0])).split(".")[-1]
        if not name or name.lower() in {"параметр", "parameter"}:
            continue
        required_cell = cells[required_column].strip() if required_column is not None and required_column < len(cells) else ""
        type_cell = cells[type_column] if type_column is not None and type_column < len(cells) else ""
        description = cells[description_column] if description_column is not None and description_column < len(cells) else ""
        methods = _method_cell(required_cell)
        fields.append(
            {
                "name": name,
                "methods": methods,
                "schema": _schema_from_type(type_cell, description),
            }
        )
    return fields


def _method_cell(cell: str) -> set[str] | None:
    parts = [part.strip().lower() for part in cell.split(",") if part.strip()]
    if not parts or any(part not in _VERBS for part in parts):
        return None
    return set(parts)


def _schema_from_type(cell: str, description: str) -> dict[str, Any]:
    kind = cell.lower().replace(" ", "")
    schema: dict[str, Any]
    if "bool" in kind:
        schema = {"type": "boolean"}
    elif kind.startswith("int") or kind in {"integer", "long"}:
        schema = {"type": "integer"}
    elif any(word in kind for word in ("float", "double", "decimal", "numeric", "number")):
        schema = {"type": "number"}
    elif "uuid" in kind:
        schema = {"type": "string", "format": "uuid"}
    elif any(word in kind for word in ("datetime", "timestamp", "date-time")):
        schema = {"type": "string", "format": "date-time"}
    else:
        schema = {"type": "string"}
    if description.strip():
        schema["description"] = description.strip()
    return schema


def _apply_obligation_marks(spec: dict[str, Any], source: str) -> None:
    """Ячейка Да/Yes/+ делает поле обязательным, Нет/No снимает обязательность."""
    from openapi_from_spec.tables import parse_required

    tables = _obligation_tables(source, parse_required)
    if not tables:
        return
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return
    examples = _request_example_keys(source)
    seen: set[int] = set()

    def visit(schema: object, allowed: set[str] | None) -> None:
        if not isinstance(schema, dict) or id(schema) in seen:
            return
        seen.add(id(schema))
        if "$ref" in schema:
            visit(_resolve_schema(spec, schema), allowed)
            return
        properties = schema.get("properties")
        if isinstance(properties, dict):
            marks = _best_obligation(set(properties), tables)
            if marks:
                required = [
                    item
                    for item in (schema.get("required") or [])
                    if not isinstance(item, str) or item not in marks or marks[item]
                ]
                for name, flag in marks.items():
                    if flag and name not in required and (allowed is None or name in allowed):
                        required.append(name)
                if required:
                    schema["required"] = required
                else:
                    schema.pop("required", None)
            for prop in properties.values():
                if isinstance(prop, dict):
                    visit(prop.get("items"), None)
                    visit(prop, None)

    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        for method, operation in methods.items():
            if str(method).startswith("x-") or not isinstance(operation, dict):
                continue
            allowed: set[str] | None = set()
            for (seen_method, seen_path), names in examples.items():
                if seen_method == str(method) and _prefix_before(str(path), seen_path) is not None:
                    allowed |= names
            if not allowed:
                allowed = None
            request = _request_schema(spec, operation)
            visit(request, allowed)
            responses = operation.get("responses")
            if not isinstance(responses, dict):
                continue
            for code, response in responses.items():
                if not str(code).startswith("2") or not isinstance(response, dict):
                    continue
                content = response.get("content")
                if not isinstance(content, dict):
                    continue
                for media in content.values():
                    if isinstance(media, dict):
                        visit(media.get("schema"), None)


def _obligation_tables(source: str, parse_required) -> list[dict[str, bool]]:
    tables: list[dict[str, bool]] = []
    current: dict[str, bool] = {}
    name_column = 0
    required_column: int | None = None
    for line in source.splitlines():
        if "|" not in line:
            if current:
                tables.append(current)
                current = {}
            required_column = None
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or re.fullmatch(r"[:\-\s]+", cells[0]):
            continue
        lowered = [cell.lower().replace(" ", "") for cell in cells]
        if any(cell == "required" or "обязат" in cell for cell in lowered):
            if current:
                tables.append(current)
                current = {}
            required_column = next(index for index, cell in enumerate(lowered) if cell == "required" or "обязат" in cell)
            name_column = next(
                (
                    index
                    for index, cell in enumerate(lowered)
                    if "наимен" in cell or cell in {"name", "field", "parameter", "параметр"}
                ),
                0,
            )
            continue
        if required_column is None or name_column >= len(cells) or required_column >= len(cells):
            continue
        name = re.sub(r"\s+", "", cells[name_column])
        if not name or name.lower() in {"наименование", "parameter", "параметр", "required"}:
            continue
        flag = parse_required(cells[required_column])
        if flag is None:
            continue
        current[name] = flag
    if current:
        tables.append(current)
    return tables


def _best_obligation(names: set[str], tables: list[dict[str, bool]]) -> dict[str, bool] | None:
    best: dict[str, bool] | None = None
    best_score = (-1, -1)
    for table in tables:
        shared = {name: flag for name, flag in table.items() if name in names}
        if not shared or (len(shared) < 2 and not names <= set(table)):
            continue
        score = (len(shared), -len(set(table) - names))
        if score > best_score:
            best = shared
            best_score = score
    return best


def _apply_request_required(spec: dict[str, Any], source: str) -> None:
    fields = [field for field in _table_fields(source) if field["methods"]]
    if not fields:
        return
    catalog = _table_fields(source)
    by_name = {field["name"]: field for field in fields}
    _remove_required_names(spec, set(by_name))
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return
    request_keys = _request_example_keys(source)
    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        for method, operation in methods.items():
            if str(method).startswith("x-") or not isinstance(operation, dict):
                continue
            schema = _request_schema(spec, operation)
            if not isinstance(schema, dict):
                continue
            properties = schema.setdefault("properties", {})
            if not isinstance(properties, dict):
                continue
            for group in _request_table_groups(source):
                shared = (set(properties) & set(group)) - _GENERIC_FIELDS
                if not shared:
                    continue
                for name, field in group.items():
                    if str(method) not in (field.get("methods") or ()):
                        continue
                    if name not in properties:
                        properties[name] = dict(field["schema"])
                    required = schema.setdefault("required", [])
                    if isinstance(required, list) and name not in required:
                        required.append(name)
            names: set[str] = set()
            for (seen_method, seen_path), seen_names in request_keys.items():
                if seen_method == str(method) and _prefix_before(str(path), seen_path) is not None:
                    names |= seen_names
            for name in names:
                field = next((item for item in catalog if item["name"] == name), None)
                if field is None or name in properties:
                    continue
                if _schema_shares_table(properties, source, name):
                    properties[name] = dict(field["schema"])
            if names:
                required = schema.get("required")
                if isinstance(required, list):
                    kept = [item for item in required if item in names]
                    if kept:
                        schema["required"] = kept
                    else:
                        schema.pop("required", None)


def _request_table_groups(source: str) -> list[dict[str, dict[str, Any]]]:
    """Поля одной таблицы параметров. Одинаковое имя в другой таблице сюда не попадает."""
    groups: list[dict[str, dict[str, Any]]] = []
    current: dict[str, dict[str, Any]] = {}
    required_column: int | None = None
    type_column: int | None = None
    description_column: int | None = None
    started = False
    for line in source.splitlines():
        if "|" not in line:
            if current:
                groups.append(current)
                current = {}
            started = False
            required_column = None
            type_column = None
            description_column = None
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or re.fullmatch(r"[:\-\s]+", cells[0]):
            continue
        lowered = [cell.lower().replace(" ", "") for cell in cells]
        if any("обязат" in cell for cell in lowered) or any(cell.startswith("тип") for cell in lowered):
            if current:
                groups.append(current)
                current = {}
            required_column = next((index for index, cell in enumerate(lowered) if "обязат" in cell), None)
            type_column = next((index for index, cell in enumerate(lowered) if cell.startswith("тип")), None)
            description_column = next((index for index, cell in enumerate(lowered) if "описан" in cell), None)
            started = True
            continue
        if not started:
            continue
        name = re.sub(r"\[\]$", "", re.sub(r"\s+", "", cells[0])).split(".")[-1]
        if not name or name.lower() in {"параметр", "parameter"}:
            continue
        required_cell = cells[required_column].strip() if required_column is not None and required_column < len(cells) else ""
        type_cell = cells[type_column] if type_column is not None and type_column < len(cells) else ""
        description = cells[description_column] if description_column is not None and description_column < len(cells) else ""
        current[name] = {
            "name": name,
            "methods": _method_cell(required_cell),
            "schema": _schema_from_type(type_cell, description),
        }
    if current:
        groups.append(current)
    return groups


def _schema_shares_table(properties: dict[str, Any], source: str, name: str) -> bool:
    tables = _field_tables(source)
    for table in tables:
        if name not in table:
            continue
        if set(properties) & set(table):
            return True
    return False


def _field_tables(source: str) -> list[set[str]]:
    tables: list[set[str]] = []
    current: set[str] = set()
    started = False
    for line in source.splitlines():
        if "|" not in line:
            if current:
                tables.append(current)
                current = set()
            started = False
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or re.fullmatch(r"[:\-\s]+", cells[0]):
            continue
        lowered = cells[0].lower().replace(" ", "")
        if "параметр" in lowered or any("обязат" in cell.lower().replace(" ", "") for cell in cells):
            if current:
                tables.append(current)
            current = set()
            started = True
            continue
        if not started:
            continue
        name = re.sub(r"\[\]$", "", re.sub(r"\s+", "", cells[0])).split(".")[-1]
        if name:
            current.add(name)
    if current:
        tables.append(current)
    return tables


def _remove_required_names(node: object, names: set[str]) -> None:
    if isinstance(node, list):
        for child in node:
            _remove_required_names(child, names)
        return
    if not isinstance(node, dict):
        return
    required = node.get("required")
    if isinstance(required, list) and ("properties" in node or node.get("type") == "object"):
        kept = [item for item in required if item not in names]
        if kept:
            node["required"] = kept
        else:
            node.pop("required", None)
    for key, child in node.items():
        if key != "example":
            _remove_required_names(child, names)


def _request_schema(spec: dict[str, Any], operation: dict[str, Any]) -> dict[str, Any] | None:
    body = operation.get("requestBody")
    if not isinstance(body, dict):
        return None
    content = body.get("content")
    if not isinstance(content, dict):
        return None
    for media in content.values():
        if not isinstance(media, dict):
            continue
        schema = media.get("schema")
        resolved = _resolve_schema(spec, schema)
        if isinstance(resolved, dict):
            return resolved
    return None


def _resolve_schema(spec: dict[str, Any], schema: object) -> dict[str, Any] | None:
    if not isinstance(schema, dict):
        return None
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
        name = ref.rsplit("/", 1)[-1]
        found = ((spec.get("components") or {}).get("schemas") or {}).get(name)
        return found if isinstance(found, dict) else None
    return schema


def _request_example_keys(source: str) -> dict[tuple[str, str], set[str]]:
    keys: dict[tuple[str, str], set[str]] = {}
    lines = source.splitlines()
    for index, line in enumerate(lines):
        match = re.search(r"\b(GET|POST|PUT|PATCH|DELETE)\s+(/[^\s]+)", line, re.IGNORECASE)
        if match is None:
            continue
        method = match.group(1).lower()
        path = match.group(2).split("?")[0]
        if path.endswith(".json"):
            path = path[: -len(".json")]
        before = "\n".join(lines[max(0, index - 6) : index])
        if re.search(r"некорректн", before, re.IGNORECASE):
            continue
        chunk: list[str] = []
        for follow in lines[index + 1 : index + 40]:
            if re.search(r"\b(GET|POST|PUT|PATCH|DELETE)\s+/", follow, re.IGNORECASE):
                break
            if re.search(r"пример ответа|\(\d{3}\s+[A-Za-z]|HTTP/1\.[01]\s+\d{3}", follow, re.IGNORECASE):
                break
            chunk.append(follow)
        for name in _json_keys("\n".join(chunk)):
            keys.setdefault((method, path.rstrip("/")), set()).add(name)
    return keys


def _json_keys(text: str) -> set[str]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return set()
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return set()
    if not isinstance(payload, dict) or set(payload) <= {"error", "errorMessage", "messageOutput"}:
        return set()
    return {str(key) for key in payload}


def _bind_status_bodies(spec: dict[str, Any], source: str) -> None:
    """Пример после строки статуса принадлежит этому коду, даже если модель перепутала тела."""
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return
    grouped: dict[tuple[str, str], list[tuple[str, dict[str, Any] | None, str]]] = {}
    for code, method, path, payload, reason in _status_blocks(source):
        grouped.setdefault((method, path), []).append((code, payload, reason))
    for (method, path), blocks in grouped.items():
        operation = _best_operation(paths, method, path)
        if operation is None:
            continue
        responses = operation.setdefault("responses", {})
        if not isinstance(responses, dict):
            continue
        named = {code for code, _payload, _reason in blocks}
        for code, payload, reason in blocks:
            response = responses.get(code)
            if not isinstance(response, dict) or "$ref" in response:
                response = {"description": reason or f"Ответ {code}"}
                responses[code] = response
            if payload is None:
                response.pop("content", None)
                current = str(response.get("description") or "")
                if reason and (re.search(r"[А-Яа-яЁё]", reason) or not re.search(r"[А-Яа-яЁё]", current)):
                    response["description"] = reason
                continue
            _replace_status_body(response, payload, reason, spec)
        success = {code for code in named if code.startswith("2")}
        if not success:
            continue
        for code in list(responses):
            if code in named or not str(code).startswith("2"):
                continue
            response = responses.get(code)
            if isinstance(response, dict) and "$ref" not in response:
                responses.pop(code, None)


def _status_blocks(source: str) -> list[tuple[str, str, str, dict[str, Any] | None, str]]:
    found: list[tuple[str, str, str, dict[str, Any] | None, str]] = []
    request_pattern = re.compile(
        r"\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+['\"]?\s*<?(https?://[^\s)'\">]+|/[^\s]+)",
        re.IGNORECASE,
    )
    seen: set[int] = set()
    for match in _STATUS_LINES.finditer(source):
        code, raw_reason = _status_match(match)
        if not code.isdigit() or not 100 <= int(code) <= 599 or match.start() in seen:
            continue
        seen.add(match.start())
        reason = _status_reason(raw_reason)
        request = None
        for seen_request in request_pattern.finditer(source[: match.start()]):
            request = seen_request
        if request is None:
            continue
        raw_path = request.group(2)
        path = _split_example_path(raw_path) if raw_path.lower().startswith("http") else raw_path.split("?")[0]
        if not path:
            continue
        after = source[match.end() : match.end() + 1800]
        stop = re.search(
            r"HTTP/1\.[01]\s+\d{3}|^[ \t]*#{0,6}[ \t]*\d{3}\s+[A-Za-z]|^#{1,3} ",
            after,
            re.IGNORECASE | re.MULTILINE,
        )
        chunk = after[: stop.start()] if stop else after
        method = request.group(1).lower()
        gap_start = 0
        for older in _STATUS_LINES.finditer(source[: match.start()]):
            gap_start = older.end()
        before = source[gap_start : match.start()]
        if code == "204" or re.search(
            r"тело ответа пуст|без тела|пустым телом|no content",
            f"{reason}\n{before}\n{chunk}",
            re.IGNORECASE,
        ):
            found.append((code, method, path, None, reason))
            continue
        payload = _loads_loose_object(chunk)
        if not isinstance(payload, dict):
            continue
        payload = _wider_status_example(before, payload)
        found.append((code, method, path, payload, reason))
    return found


_STATUS_LINES = re.compile(
    r"HTTP/1\.[01]\s+(\d{3})\s*([^\n]*)"
    r"|^[ \t]*#{0,6}[ \t]*(\d{3})\s+([^\n]+)"
    r"|Код HTTP:\s*(\d{3})"
    r"|\((\d{3})\s+([A-Za-z][^)\n]{0,80})\)",
    re.IGNORECASE | re.MULTILINE,
)
_REQUEST_LINE = re.compile(
    r"\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+['\"]?\s*<?(https?://[^\s)'\">]+|/[^\s]+)",
    re.IGNORECASE,
)
_ERROR_KEYS = {"error", "errormessage", "messageoutput", "messagetouser", "message", "activityid"}


def _status_match(match: re.Match[str]) -> tuple[str, str]:
    if match.group(1):
        return match.group(1), match.group(2) or ""
    if match.group(3):
        return match.group(3), match.group(4) or ""
    if match.group(5):
        return match.group(5), ""
    return match.group(6) or "", match.group(7) or ""


def _status_reason(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip(" -")
    parts = re.split(r"\s+[—–-]\s+", cleaned, maxsplit=1)
    if len(parts) == 2 and re.search(r"[А-Яа-яЁё]", parts[1]):
        cleaned = parts[1].strip(" -")
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _word_in_source(word: str, source: str) -> bool:
    if not word:
        return False
    return re.search(rf"(?<![\w]){re.escape(word.lower())}(?![\w])", source.lower()) is not None


def _keep_evidenced_tags(spec: dict[str, Any], source: str) -> None:
    """Тег остаётся, только если это слово есть в документе. Иначе берётся последний сегмент пути."""
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return
    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        segments = [piece for piece in str(path).split("/") if piece and not piece.startswith("{")]
        resource = segments[-1] if segments else ""
        for method, operation in methods.items():
            if str(method).startswith("x-") or not isinstance(operation, dict):
                continue
            tags = operation.get("tags")
            if not isinstance(tags, list):
                continue
            kept = [tag for tag in tags if isinstance(tag, str) and _word_in_source(tag, source)]
            if not kept and resource and _word_in_source(resource, source):
                kept = [resource]
            if kept:
                operation["tags"] = kept
            else:
                operation.pop("tags", None)


def _best_operation(paths: dict[str, Any], method: str, example_path: str) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_length = -1
    for path, methods in paths.items():
        if not isinstance(methods, dict) or not isinstance(methods.get(method), dict):
            continue
        if _prefix_before(str(path), example_path) is None and not _same_route(str(path), example_path):
            continue
        length = len(_path_segments(str(path)))
        if length > best_length:
            best = methods[method]
            best_length = length
    return best


def _replace_status_body(
    response: dict[str, Any], payload: dict[str, Any], reason: str, spec: dict[str, Any]
) -> None:
    content = response.setdefault("content", {})
    if not isinstance(content, dict):
        return
    media = content.setdefault("application/json", {})
    if not isinstance(media, dict):
        return
    example = media.get("example")
    schema = media.get("schema")
    if isinstance(schema, dict) and "$ref" in schema:
        resolved = _resolve_schema(spec, schema)
        covered = resolved.get("properties") if isinstance(resolved, dict) else None
        if isinstance(covered, dict) and set(payload) <= set(covered):
            media["example"] = payload
            _status_description(response, payload, reason)
            return
    if not isinstance(example, dict) or example != payload:
        media["example"] = payload
        schema = media.get("schema")
        properties = schema.get("properties") if isinstance(schema, dict) else None
        current = set(properties) if isinstance(properties, dict) else set()
        if not current or current.isdisjoint(set(payload)):
            media["schema"] = _schema_from_example(payload)
        elif isinstance(properties, dict):
            for name, schema_part in _schema_from_example(payload)["properties"].items():
                properties.setdefault(name, schema_part)
    _status_description(response, payload, reason)


def _message_field(payload: dict[str, Any]) -> str:
    ranked: list[tuple[int, str]] = []
    order = {"messageoutput": 0, "errormessage": 1, "messagetouser": 2, "message": 3}
    for key, value in payload.items():
        if not isinstance(value, str) or not value.strip():
            continue
        folded = re.sub(r"[_\s]+", "", str(key)).lower()
        if folded in order:
            ranked.append((order[folded], value.strip()))
    if not ranked:
        return ""
    ranked.sort()
    return ranked[0][1]


_GENERIC_STATUS = {
    "ok",
    "created",
    "accepted",
    "no content",
    "bad request",
    "unauthorized",
    "forbidden",
    "not found",
    "conflict",
    "unprocessable entity",
    "internal server error",
    "internal error",
    "service unavailable",
}


def _is_generic_status(text: str) -> bool:
    folded = re.sub(r"\s+", " ", text or "").strip().casefold()
    return not folded or folded.isdigit() or folded in _GENERIC_STATUS


def _complete_named_statuses(spec: dict[str, Any], source: str) -> None:
    """Подпись вроде Not Found заменяется объяснением фрагмента. Уже записанную фразу и пример не трогает."""
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return
    for code, method, path, payload, reason in _status_blocks(source):
        operation = _best_operation(paths, method, path)
        if operation is None:
            continue
        responses = operation.setdefault("responses", {})
        if not isinstance(responses, dict):
            continue
        phrase = _message_field(payload) if isinstance(payload, dict) else ""
        if (
            not phrase
            and isinstance(payload, dict)
            and isinstance(payload.get("error"), str)
            and payload["error"].strip()
            and (not reason or _is_generic_status(reason))
        ):
            phrase = payload["error"].strip()
        if not phrase and reason and not _is_generic_status(reason):
            phrase = reason
        response = responses.get(code)
        if isinstance(response, dict) and "$ref" in response:
            if not _is_template_response(spec, response) or not (phrase or isinstance(payload, dict)):
                continue
            response = {"description": phrase or f"Ответ {code}"}
            responses[code] = response
        elif not isinstance(response, dict):
            if not isinstance(payload, dict) and not phrase:
                continue
            response = {"description": phrase or f"Ответ {code}"}
            responses[code] = response
        elif phrase and _is_generic_status(str(response.get("description") or "")):
            response["description"] = phrase
        if isinstance(payload, dict):
            _attach_status_example(response, payload)
            _fill_empty_media_schema(response, payload)
    _drop_unused_templates(spec)


def _json_text(raw: str) -> str:
    lines = [line for line in raw.splitlines() if not re.fullmatch(r"`{3,}\w*", line.strip())]
    return re.sub(r",(\s*[}\]])", r"\1", "\n".join(lines))


def _fill_empty_media_schema(response: dict[str, Any], payload: dict[str, Any]) -> None:
    content = response.get("content")
    if not isinstance(content, dict):
        return
    for media in content.values():
        if isinstance(media, dict) and _schema_needs_example_shape(media.get("schema")):
            media["schema"] = _schema_from_example(payload)


def _schema_needs_example_shape(schema: object) -> bool:
    if schema is None or schema == {}:
        return True
    return isinstance(schema, dict) and "$ref" not in schema and schema.get("type") == "object" and not schema.get("properties")


def _error_payload(payload: dict[str, Any]) -> bool:
    keys = {re.sub(r"[_\s]+", "", str(key)).lower() for key in payload}
    return bool(keys) and keys <= _ERROR_KEYS


def _largest_data_object(text: str) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_span = 0
    start = 0
    while True:
        begin = text.find("{", start)
        if begin < 0:
            break
        depth = 0
        end = None
        for index in range(begin, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None:
            break
        raw = _json_text(text[begin:end])
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            start = begin + 1
            continue
        if isinstance(payload, dict) and not _error_payload(payload) and end - begin > best_span:
            best = payload
            best_span = end - begin
        start = end
    return best


def _example_path(raw: str) -> str:
    if raw.lower().startswith("http"):
        return _split_example_path(raw)
    path = raw.split("?")[0].rstrip("/")
    return path[: -len(".json")] if path.endswith(".json") else path


def _fill_empty_success_bodies(spec: dict[str, Any], source: str) -> None:
    """Пустой успешный ответ получает самый большой JSON своего запроса. Заполненную схему не заменяет."""
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return
    requests = list(_REQUEST_LINE.finditer(source))
    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        for method, operation in methods.items():
            if str(method).startswith("x-") or not isinstance(operation, dict):
                continue
            spans = []
            for index, match in enumerate(requests):
                if match.group(1).lower() != str(method):
                    continue
                example_path = _example_path(match.group(2))
                if _prefix_before(str(path), example_path) is None and not _same_route(str(path), example_path):
                    continue
                end = requests[index + 1].start() if index + 1 < len(requests) else len(source)
                region = source[match.end() : end]
                if str(method) in {"post", "put", "patch"}:
                    success = re.search(
                        r"HTTP/1\.[01]\s+2\d{2}|\(2\d{2}\s+[A-Za-z]|Код HTTP:\s*2\d{2}",
                        region,
                        re.IGNORECASE,
                    )
                    region = region[success.end() :] if success else ""
                if region:
                    spans.append(region)
            if not spans:
                continue
            payload = _largest_data_object(spans[-1])
            if not isinstance(payload, dict):
                continue
            responses = operation.get("responses")
            if not isinstance(responses, dict):
                continue
            response = next(
                (
                    responses.get(code)
                    for code in ("200", "201", "202")
                    if isinstance(responses.get(code), dict) and "$ref" not in responses.get(code)
                ),
                None,
            )
            if not isinstance(response, dict):
                continue
            content = response.get("content")
            if not isinstance(content, dict):
                if _response_has_body(response):
                    continue
                response["content"] = {
                    "application/json": {"schema": _schema_from_example(payload), "example": payload}
                }
                continue
            for media in content.values():
                if not isinstance(media, dict) or media.get("example") is not None:
                    continue
                if _schema_needs_example_shape(media.get("schema")):
                    media["schema"] = _schema_from_example(payload)
                    media["example"] = payload


def _response_has_body(response: dict[str, Any]) -> bool:
    content = response.get("content")
    if not isinstance(content, dict):
        return False
    return any(isinstance(media, dict) and (media.get("example") is not None or not _schema_needs_example_shape(media.get("schema"))) for media in content.values())


def _fill_missing_request_bodies(spec: dict[str, Any], source: str) -> None:
    """Тело POST, PUT и PATCH дописывается из JSON запроса, если модель его не записала."""
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return
    for match in _REQUEST_LINE.finditer(source):
        method = match.group(1).lower()
        if method not in {"post", "put", "patch"}:
            continue
        operation = _best_operation(paths, method, _example_path(match.group(2)))
        if operation is None or operation.get("requestBody"):
            continue
        nxt = _REQUEST_LINE.search(source, match.end())
        status = re.search(r"HTTP/1\.[01]\s+\d{3}|\(\d{3}\s+[A-Za-z]|Код HTTP:\s*\d{3}", source[match.end() :], re.IGNORECASE)
        end = match.end() + (status.start() if status else 1200)
        if nxt is not None:
            end = min(end, nxt.start())
        payload = _largest_data_object(source[match.end() : end])
        if not isinstance(payload, dict):
            continue
        operation["requestBody"] = {
            "required": True,
            "content": {"application/json": {"schema": _schema_from_example(payload), "example": payload}},
        }


def _add_curl_headers(spec: dict[str, Any], source: str) -> None:
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return
    for match in _REQUEST_LINE.finditer(source):
        method = match.group(1).lower()
        operation = _best_operation(paths, method, _example_path(match.group(2)))
        if operation is None:
            continue
        nxt = _REQUEST_LINE.search(source, match.end())
        end = nxt.start() if nxt else match.end() + 500
        window = source[match.end() : min(end, match.end() + 500)]
        parameters = operation.setdefault("parameters", [])
        if not isinstance(parameters, list):
            continue
        names = {item.get("name") for item in parameters if isinstance(item, dict)}
        for header in re.finditer(r"-H\s+(['\"])([^'\"]+)\1", window):
            raw_header = header.group(2)
            if ":" not in raw_header:
                continue
            name, value = raw_header.split(":", 1)
            name = name.strip()
            if not name or name in names:
                continue
            names.add(name)
            parameters.append(
                {
                    "name": name,
                    "in": "header",
                    "required": True,
                    "description": value.strip(),
                    "schema": {"type": "string"},
                }
            )


def _drop_unnamed_templates(spec: dict[str, Any], source: str) -> None:
    """Шаблонный код, которого нет во фрагменте, не остаётся в операции."""
    named = set(re.findall(r"\b([1-5]\d{2})\b", source))
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return
    for methods in paths.values():
        if not isinstance(methods, dict):
            continue
        for method, operation in methods.items():
            if str(method).startswith("x-") or not isinstance(operation, dict):
                continue
            responses = operation.get("responses")
            if not isinstance(responses, dict):
                continue
            for code in list(responses):
                response = responses.get(code)
                if str(code) in named or not isinstance(response, dict):
                    continue
                if _is_template_response(spec, response):
                    responses.pop(code, None)
    _drop_unused_templates(spec)


def _drop_duplicate_example(node: object) -> None:
    """В 3.0.3 рядом не держатся и `example`, и `examples`."""
    if isinstance(node, dict):
        examples = node.get("examples")
        if isinstance(examples, dict) and examples and "example" in node:
            node.pop("example", None)
        for child in node.values():
            _drop_duplicate_example(child)
    elif isinstance(node, list):
        for child in node:
            _drop_duplicate_example(child)


def _fold_json_suffix_paths(spec: dict[str, Any]) -> None:
    """`/orders.json` сливается с `/orders`, если оба описывают один ресурс."""
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return
    for path in list(paths):
        if not str(path).endswith(".json"):
            continue
        methods = paths.get(path)
        plain = str(path)[: -len(".json")] or "/"
        if not isinstance(methods, dict):
            continue
        if plain not in paths:
            paths[plain] = paths.pop(path)
            continue
        bucket = paths.get(plain)
        if not isinstance(bucket, dict):
            continue
        for method, operation in methods.items():
            previous = bucket.get(method)
            if previous is None or (isinstance(operation, dict) and _richness(operation) > _richness(previous)):
                bucket[method] = operation
        paths.pop(path, None)


_UUID = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_VERB_CELL = re.compile(r"^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)$", re.IGNORECASE)


def _identifier_words(source: str) -> set[str]:
    """Слова, которые документ уже написал в именах и в целых отрезках пути."""
    cache: dict[int, set[str]] = getattr(_identifier_words, "_cache", {})
    cached = cache.get(id(source))
    if cached is not None:
        return cached
    words: set[str] = set()
    for match in re.finditer(r"[A-Za-z][a-z0-9]*[A-Z][A-Za-z0-9]*", source):
        token = match.group(0)
        before = source[max(0, match.start() - 2) : match.start()]
        after = source[match.end() : match.end() + 2]
        if "-" in before or "-" in after:
            continue
        parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", token)
        for part in parts:
            if len(part) > 1:
                words.add(part.lower())
        words.add("".join(part.lower() for part in parts))
    for segment in re.findall(r"(?<=/)([A-Za-z][A-Za-z0-9]*)(?=/|[?#\s]|\}|$)", source):
        if len(segment) > 2:
            words.add(segment.lower())
    cache[id(source)] = words
    _identifier_words._cache = cache
    return words


def _join_wrapped_hyphen(value: str, source: str = "") -> str:
    """Пробел внутри адреса — перенос. Дефис остаётся между двумя целыми словами и склеивается, если разрезал одно имя."""
    words = _identifier_words(source)

    def replace(match: re.Match[str]) -> str:
        left, right = match.group(1), match.group(2)
        if left.lower() in words and right.lower() in words:
            return f"{left}-{right}"
        if (left + right).lower() in words:
            return left + right
        return f"{left}-{right}"

    text = re.sub(r"([A-Za-z0-9]+)-\s+([A-Za-z0-9]+)", replace, value)
    path, separator, query = text.partition("?")
    path = re.sub(r"\s+", "", path)
    return path + (separator + query if separator else "")


def _is_sample_segment(part: str) -> bool:
    return bool(
        part.isdigit()
        or _UUID.fullmatch(part)
        or (part.startswith("{") and part.endswith("}") and len(part) > 2)
    )


def _path_from_messy(value: str, source: str = "") -> str:
    """Путь без скобок markdown-ссылки и без переноса строки."""
    text = value.strip()
    if "](" in text:
        dest = text.split("](", 1)[1].split(")", 1)[0].strip()
        dest = re.sub(r"^(https?:)/(?=[^/])", r"\1//", dest, flags=re.IGNORECASE)
        text = dest or text.split("](", 1)[0].lstrip("[")
    text = _join_wrapped_hyphen(text, source).split("?")[0].rstrip("/")
    if re.match(r"https?://", text, re.IGNORECASE):
        text = urlsplit(text).path or "/"
    if text.startswith("{") and "}" in text:
        text = "/" + text.split("}", 1)[1].lstrip("/")
    if not text.startswith("/"):
        return ""
    return text or "/"


def _clean_request_path(path: str, source: str = "") -> str:
    cleaned = _path_from_messy(path, source)
    return cleaned if cleaned.startswith("/") else ""


def _table_request_rows(text: str) -> list[tuple[int, str, str]]:
    """Строка таблицы, в которой есть глагол и адрес, — один вызов."""
    found: list[tuple[int, str, str]] = []
    for match in re.finditer(r"(?m)^\|.+\|\s*$", text):
        cells = [cell.strip() for cell in match.group(0).strip().strip("|").split("|")]
        verb = next((cell for cell in cells if _VERB_CELL.fullmatch(cell)), "")
        if not verb:
            continue
        target = ""
        for cell in cells:
            joined = _join_wrapped_hyphen(cell, text)
            if re.search(r"(\{[^/{}]+\}/|https?://|^/)", joined):
                target = joined
                break
        path = _clean_request_path(target, text) if target else ""
        if path:
            found.append((match.start(), verb.lower(), path))
    return found


def _placeholder_name(source: str) -> str:
    counts: dict[str, int] = {}
    for match in re.finditer(r"(?<![\w/])\{([A-Za-z][A-Za-z0-9]*)\}/", source):
        name = match.group(1)
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return ""
    return max(counts, key=lambda name: (counts[name], name == "apiUrl"))


def _tidy_surface(spec: dict[str, Any], source: str) -> None:
    """Убирает markdown из путей и серверов и оставляет плейсхолдер хоста, если вызовы заданы им."""
    paths = spec.get("paths")
    if isinstance(paths, dict):
        renamed: dict[str, Any] = {}
        for path, methods in paths.items():
            cleaned = _path_from_messy(str(path), source) or str(path)
            bucket = renamed.setdefault(cleaned, {})
            if isinstance(methods, dict) and isinstance(bucket, dict):
                for method, operation in methods.items():
                    previous = bucket.get(method)
                    if isinstance(previous, dict) and isinstance(operation, dict):
                        _union_operation(previous, operation)
                    else:
                        bucket[method] = operation
        spec["paths"] = renamed
    servers = spec.get("servers")
    if not isinstance(servers, list):
        servers = []
    cleaned_servers: list[Any] = []
    for server in servers:
        if not isinstance(server, dict):
            continue
        url = str(server.get("url") or "")
        if "](" in url:
            dest = url.split("](", 1)[1].split(")", 1)[0].strip()
            dest = re.sub(r"^(https?:)/(?=[^/])", r"\1//", dest, flags=re.IGNORECASE)
            url = dest
        if "](" in url or not url:
            continue
        item = dict(server)
        item["url"] = url
        cleaned_servers.append(item)
    placeholder = _placeholder_name(source)
    if placeholder and any(re.match(r"\{[^{}]+\}", str(item.get("url") or "")) for item in cleaned_servers if isinstance(item, dict)):
        kept = [
            item
            for item in cleaned_servers
            if isinstance(item, dict) and not re.match(r"https?://", str(item.get("url") or ""), re.IGNORECASE)
        ]
        if kept:
            cleaned_servers = kept
    chosen: dict[str, Any] = {}
    for server in cleaned_servers:
        if not isinstance(server, dict):
            continue
        url = str(server.get("url") or "")
        key = url.casefold()
        current = chosen.get(key)
        if current is None:
            chosen[key] = server
            continue
        default = ""
        variables = server.get("variables")
        if isinstance(variables, dict):
            for value in variables.values():
                if isinstance(value, dict):
                    default = str(value.get("default") or "")
        if default.startswith("http"):
            continue
        chosen[key] = server
    if placeholder:
        folded = "{" + placeholder.casefold() + "}"
        for key, server in list(chosen.items()):
            url = str(server.get("url") or "")
            if url.casefold() == folded or url.casefold().startswith(folded + "/"):
                server = dict(server)
                server["url"] = "{" + placeholder + "}" + url[len(folded) :]
                variables = server.get("variables")
                if isinstance(variables, dict):
                    renamed_vars = {}
                    for name, value in variables.items():
                        renamed_vars[placeholder if name.casefold() == placeholder.casefold() else name] = value
                    server["variables"] = renamed_vars
                chosen[key] = server
    if placeholder:
        bare = "{" + placeholder + "}"
        if any(str(server.get("url") or "") == bare for server in chosen.values()):
            chosen = {
                key: server
                for key, server in chosen.items()
                if str(server.get("url") or "") == bare
                or not str(server.get("url") or "").casefold().startswith(bare.casefold() + "/")
            }
    if chosen:
        spec["servers"] = list(chosen.values())
    elif placeholder:
        spec["servers"] = [{"url": "{" + placeholder + "}"}]


def _extend_unique_document_suffix(spec: dict[str, Any], source: str) -> None:
    """Короткий путь удлиняется до единственного адреса из таблицы или плейсхолдера, если сервер этот префикс не содержит."""
    paths = spec.get("paths")
    servers = spec.get("servers") or []
    if not isinstance(paths, dict) or not servers:
        return
    mounts = [
        mount
        for server in servers
        if isinstance(server, dict) and (mount := _server_mount(str(server.get("url") or ""))) is not None
    ]
    documents = _placeholder_document_paths(source)
    renamed: dict[str, Any] = {}
    for operation_path, methods in paths.items():
        op_parts = [part for part in str(operation_path).split("/") if part]
        target = str(operation_path)
        shapes: dict[tuple[str, ...], str] = {}
        for document in documents:
            doc_parts = [part for part in document.split("/") if part]
            if len(doc_parts) <= len(op_parts):
                continue
            if [_sample_key(part) for part in doc_parts[-len(op_parts) :]] != [_sample_key(part) for part in op_parts]:
                continue
            prefix_parts = doc_parts[: -len(op_parts)]
            shapes.setdefault(tuple(_sample_key(part) for part in prefix_parts), "/" + "/".join(prefix_parts))
        if len(shapes) == 1:
            prefix = next(iter(shapes.values())).rstrip("/")
            if prefix and not any(mount == prefix or mount.endswith(prefix) for mount in mounts):
                target = f"{prefix}/{str(operation_path).lstrip('/')}"
        bucket = renamed.setdefault(target, {})
        if isinstance(methods, dict) and isinstance(bucket, dict):
            for method, operation in methods.items():
                previous = bucket.get(method)
                if isinstance(previous, dict) and isinstance(operation, dict):
                    _union_operation(previous, operation)
                else:
                    bucket[method] = operation
    spec["paths"] = renamed


def _sample_key(part: str) -> str:
    if _is_sample_segment(part):
        return "{}"
    if part.lower().endswith(".json"):
        part = part[:-5]
    return part


def _apply_document_templates(spec: dict[str, Any], source: str) -> None:
    """Образец-идентификатор в пути заменяется именем параметра из адреса документа."""
    paths = spec.get("paths")
    documents = _document_request_paths(source)
    if not isinstance(paths, dict) or not documents:
        return
    renamed: dict[str, Any] = {}
    for path, methods in paths.items():
        parts = [part for part in str(path).split("/") if part]
        target_parts = parts
        for document in documents:
            doc_parts = [part for part in document.split("/") if part]
            if len(doc_parts) != len(parts):
                continue
            if [_sample_key(part) for part in doc_parts] != [_sample_key(part) for part in parts]:
                continue
            rewritten = [
                doc if _is_sample_segment(current) and doc.startswith("{") and doc.endswith("}") else current
                for current, doc in zip(parts, doc_parts)
            ]
            if rewritten != parts and any(part.startswith("{") for part in rewritten):
                target_parts = rewritten
                break
        target = "/" + "/".join(target_parts) if target_parts else str(path)
        bucket = renamed.setdefault(target, {})
        if isinstance(methods, dict) and isinstance(bucket, dict):
            for method, operation in methods.items():
                previous = bucket.get(method)
                if isinstance(previous, dict) and isinstance(operation, dict):
                    _union_operation(previous, operation)
                else:
                    bucket[method] = operation
                if isinstance(bucket.get(method), dict):
                    for name in re.findall(r"\{([^{}]+)\}", target):
                        _ensure_path_param(bucket[method], name, source)
    spec["paths"] = renamed


_PLACEHOLDER_REQUEST = re.compile(
    r"(?m)^(?:curl\b[^\n]*-X\s+)?"
    r"(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+['\"]?\s*<?"
    r"(\{[^{}\s]+\}/[^\s>'\"\\?]+)",
    re.IGNORECASE,
)


def _server_mount(url: str) -> str | None:
    """Путь сервера. Пустая строка — хост без префикса. None — адрес, который нельзя разобрать."""
    cleaned = url.strip()
    if re.match(r"https?://", cleaned, re.IGNORECASE):
        return urlsplit(cleaned).path.rstrip("/")
    match = re.fullmatch(r"\{[^{}]+\}(/.*)?", cleaned)
    if match:
        return (match.group(1) or "").rstrip("/")
    return None


def _server_token(url: str) -> str:
    cleaned = url.strip()
    if re.match(r"https?://", cleaned, re.IGNORECASE):
        return urlsplit(cleaned).netloc
    match = re.search(r"\{([^{}]+)\}", cleaned)
    return match.group(1) if match else cleaned


def _drop_ungrounded_servers(spec: dict[str, Any], source: str) -> None:
    """Сервер, которого нет в документе, убирается, если рядом есть сервер из документа."""
    servers = spec.get("servers")
    if not isinstance(servers, list) or len(servers) < 2:
        return
    grounded: list[Any] = []
    for server in servers:
        if not isinstance(server, dict):
            continue
        token = _server_token(str(server.get("url") or ""))
        if token and token in source:
            grounded.append(server)
    if grounded and len(grounded) < len(servers):
        spec["servers"] = grounded


def _strip_repeated_server_prefix(spec: dict[str, Any]) -> None:
    """Путь не начинается с префикса, которым уже заканчивается каждый сервер."""
    servers = spec.get("servers")
    paths = spec.get("paths")
    if not isinstance(servers, list) or not servers or not isinstance(paths, dict):
        return
    mounts: list[str] = []
    for server in servers:
        if not isinstance(server, dict):
            return
        mount = _server_mount(str(server.get("url") or ""))
        if mount is None:
            return
        mounts.append(mount)
    prefix = mounts[0]
    if not prefix or any(item != prefix for item in mounts):
        return
    renamed: dict[str, Any] = {}
    for path, methods in paths.items():
        current = str(path)
        if current == prefix or current.startswith(prefix + "/"):
            current = current[len(prefix) :] or "/"
            if not current.startswith("/"):
                current = "/" + current
        bucket = renamed.setdefault(current, {})
        if isinstance(methods, dict) and isinstance(bucket, dict):
            for method, operation in methods.items():
                previous = bucket.get(method)
                if isinstance(previous, dict) and isinstance(operation, dict):
                    _union_operation(previous, operation)
                else:
                    bucket[method] = operation
    spec["paths"] = renamed


def _call_parts(path: str) -> tuple[str, ...]:
    parts = list(_route_parts(path.split("?")[0].rstrip("/")))
    if parts and parts[-1].lower().endswith(".json"):
        parts[-1] = parts[-1][:-5]
        if not parts[-1]:
            parts.pop()
    return tuple("{}" if _is_sample_segment(part) else part for part in parts)


def _documented_calls(source: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for _at, method, path in _unwritten_calls(source):
        if path:
            found.append((method, path))
    for method, _origin, path in example_targets(source):
        if method and path:
            cleaned = _clean_request_path(path, source)
            if cleaned:
                found.append((method.lower(), cleaned))
    return found


def _operation_part_sets(path: str, mounts: list[str]) -> set[tuple[str, ...]]:
    keys = {_call_parts(path)}
    for mount in mounts:
        if mount:
            keys.add(_call_parts(f"{mount.rstrip('/')}/{path.lstrip('/')}"))
    return keys


def _part_keys(path: str, mounts: list[str]) -> set[tuple[str, ...]]:
    return {key for key in _operation_part_sets(path, mounts) if key}


def _call_covers(call_path: str, keys: set[tuple[str, ...]]) -> bool:
    """Адрес покрывает путь операции, в том числе когда перед путём в адресе стоит префикс сервера."""
    parts = _call_parts(call_path)
    return any(parts == key or (len(parts) > len(key) and parts[-len(key) :] == key) for key in keys)


def _methods_for_route(path: str, calls: list[tuple[str, str]], mounts: list[str]) -> set[str] | None:
    keys = _part_keys(path, mounts)
    found = {method for method, call_path in calls if _call_covers(call_path, keys)}
    return found or None


def _unrequested_sibling(path: str, calls: list[tuple[str, str]], mounts: list[str]) -> bool:
    """Путь отличается от строки запроса только хвостовым параметром или его отсутствием."""
    for key in _part_keys(path, mounts):
        if key and key[-1] == "{}":
            parent = key[:-1]
            if any(_call_parts(call_path) == parent for _method, call_path in calls) and not any(
                _call_parts(call_path) == key for _method, call_path in calls
            ):
                return True
        if any(_call_parts(call_path) == key + ("{}",) for _method, call_path in calls):
            return True
    return False


def _sample_target(method: str, path: str, calls: list[tuple[str, str]], mounts: list[str]) -> str | None:
    """Строка с числом вместо параметра переносит операцию на путь с этим параметром."""
    keys = _operation_part_sets(path, mounts)
    names: list[str] = []
    for call_method, call_path in calls:
        if call_method != method:
            continue
        call = _call_parts(call_path)
        if not any(len(call) == len(key) + 1 and call[:-1] == key and call[-1] == "{}" for key in keys):
            continue
        raw_parts = list(_route_parts(call_path.split("?")[0]))
        raw = raw_parts[-1] if raw_parts else ""
        names.append(raw if raw.startswith("{") and raw.endswith("}") else "{id}")
    if not names:
        return None
    name = next((item for item in names if item != "{id}"), names[0])
    return path.rstrip("/") + "/" + name


def _align_operations_to_requests(spec: dict[str, Any], source: str) -> None:
    """Операция остаётся, если документ содержит её строку запроса. Число в пути становится параметром."""
    calls = _documented_calls(source)
    paths = spec.get("paths")
    if not calls or not isinstance(paths, dict):
        return
    mounts = [
        mount
        for server in (spec.get("servers") or [])
        if isinstance(server, dict) and (mount := _server_mount(str(server.get("url") or ""))) is not None
    ]
    for path, methods in list(paths.items()):
        if not isinstance(methods, dict):
            continue
        for method, operation in list(methods.items()):
            if str(method).startswith("x-") or not isinstance(operation, dict):
                continue
            if operation.get("x-method-origin") == "inferred" or str(method) in (
                _methods_for_route(str(path), calls, mounts) or set()
            ):
                continue
            target = _sample_target(str(method), str(path), calls, mounts)
            evidenced = _methods_for_route(str(path), calls, mounts) is not None
            if target is None or target == path:
                if evidenced or _unrequested_sibling(str(path), calls, mounts):
                    methods.pop(method, None)
                continue
            name = target.rsplit("/", 1)[-1].strip("{}")
            if name:
                _ensure_path_param(operation, name, source)
            bucket = paths.setdefault(target, {})
            if not isinstance(bucket, dict):
                methods.pop(method, None)
                continue
            current = bucket.get(method)
            if isinstance(current, dict):
                base, extra = (current, operation) if _richness(current) >= _richness(operation) else (operation, current)
                _union_operation(base, extra)
                bucket[method] = base
            else:
                bucket[method] = operation
            methods.pop(method, None)
        if not any(not str(method).startswith("x-") for method in methods):
            paths.pop(path, None)
    evidenced_verbs: set[str] = set()
    for path, methods in list(paths.items()):
        if not isinstance(methods, dict):
            continue
        for method in methods:
            if str(method).startswith("x-"):
                continue
            if _methods_for_route(str(path), calls, mounts):
                evidenced_verbs.add(str(method))
    for path, methods in list(paths.items()):
        if not isinstance(methods, dict):
            continue
        for method, operation in list(methods.items()):
            if str(method).startswith("x-") or not isinstance(operation, dict):
                continue
            if operation.get("x-method-origin") == "inferred" or str(method) not in evidenced_verbs:
                continue
            if _methods_for_route(str(path), calls, mounts):
                continue
            methods.pop(method, None)
        if not any(not str(method).startswith("x-") for method in methods):
            paths.pop(path, None)


def _version_alias(example_path: str, declared_path: str, *, brace_names: bool) -> bool:
    """Тот же вызов, если адрес отличается от таблицы только номером версии и образцом идентификатора."""
    example = [part for part in example_path.split("/") if part]
    declared = [part for part in declared_path.split("/") if part]
    if len(example) != len(declared) or not example:
        return False
    version_diffs = 0
    sample = False
    for left, right in zip(example, declared):
        if left == right:
            continue
        if re.fullmatch(r"v\d+", left) and re.fullmatch(r"v\d+", right):
            version_diffs += 1
            continue
        braced = right.startswith("{") and right.endswith("}")
        concrete = bool(left.isdigit() or _UUID.fullmatch(left))
        renamed = brace_names and left.startswith("{") and left.endswith("}") and left != right
        if braced and (concrete or renamed):
            sample = True
            continue
        return False
    return version_diffs == 1 and sample


def _example_version_of(example_path: str, declared_path: str) -> bool:
    return _version_alias(example_path, declared_path, brace_names=False)


def _fold_example_versions(spec: dict[str, Any], source: str) -> None:
    """Пример со старым номером версии не остаётся вторым путём, если таблица вызова назвала другой."""
    paths = spec.get("paths")
    declarations = _table_request_rows(source)
    if not isinstance(paths, dict) or not declarations:
        return
    declared_keys = {(method, path) for _at, method, path in declarations}
    for path, methods in list(paths.items()):
        if not isinstance(methods, dict):
            continue
        for method, operation in list(methods.items()):
            if str(method).startswith("x-") or not isinstance(operation, dict):
                continue
            if (str(method), str(path)) in declared_keys:
                continue
            targets = [
                declared
                for _at, declared_method, declared in declarations
                if declared_method == str(method)
                and _version_alias(str(path), declared, brace_names=True)
            ]
            if len(targets) != 1 or targets[0] == path:
                continue
            target = targets[0]
            bucket = paths.setdefault(target, {})
            if isinstance(bucket, dict) and method not in bucket:
                bucket[method] = operation
                for name in re.findall(r"\{([^{}]+)\}", target):
                    _ensure_path_param(operation, name, source)
            methods.pop(method, None)
        if not any(not str(name).startswith("x-") for name in methods):
            paths.pop(path, None)


def _collapse_duplicate_routes(spec: dict[str, Any], source: str) -> None:
    """Два пути одного ресурса сходятся в тот, который вместе с сервером даёт адрес из документа."""
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return
    groups: dict[tuple[str, ...], list[str]] = {}
    for path in list(paths):
        groups.setdefault(_call_parts(str(path)), []).append(str(path))
    document_paths = _document_request_paths(source)
    for aliases in groups.values():
        if len(aliases) < 2:
            continue
        target = max(aliases, key=lambda path: _path_rank(spec, path, document_paths))
        bucket = paths.get(target)
        if not isinstance(bucket, dict):
            continue
        for alias in aliases:
            if alias == target or alias not in paths:
                continue
            methods = paths.get(alias)
            if isinstance(methods, dict):
                for method, operation in methods.items():
                    if not isinstance(operation, dict):
                        continue
                    current = bucket.get(method)
                    if not isinstance(current, dict):
                        bucket[method] = operation
                        continue
                    _union_operation(current, operation)
            paths.pop(alias, None)
    _drop_doubling_servers(spec, document_paths)


def _placeholder_document_paths(source: str) -> list[str]:
    """Адреса, записанные после плейсхолдера хоста или в ячейке таблицы вызовов."""
    found: list[str] = []
    for match in _PLACEHOLDER_REQUEST.finditer(source):
        tail = match.group(2).split("}", 1)[1]
        path = _clean_request_path("/" + tail.lstrip("/"), source)
        if path:
            found.append(path)
    for _at, _method, path in _table_request_rows(source):
        if path:
            found.append(path)
    return found


def _document_request_paths(source: str) -> list[str]:
    found: list[str] = []
    for _method, path in _documented_calls(source):
        if path:
            found.append(path)
    for _method, _origin, path in example_targets(source):
        cleaned = _clean_request_path(path, source) if path else ""
        if cleaned:
            found.append(cleaned)
    return found


def _http_server_path(server: object) -> str | None:
    if not isinstance(server, dict):
        return None
    url = str(server.get("url") or "")
    if not re.match(r"https?://", url, re.IGNORECASE):
        return None
    return urlsplit(url).path.rstrip("/")


def _join_server_path(server_path: str, operation_path: str) -> str:
    if not server_path:
        return operation_path if operation_path.startswith("/") else "/" + operation_path
    return server_path.rstrip("/") + "/" + operation_path.lstrip("/")


def _path_is_doubled(path: str) -> bool:
    parts = _path_segments(path)
    for index in range(len(parts) - 3):
        if (
            parts[index] == parts[index + 2] == "api"
            and re.fullmatch(r"v\d+", parts[index + 1] or "")
            and parts[index + 1] == parts[index + 3]
        ):
            return True
    return False


def _path_matches_document(composed: str, document_paths: list[str]) -> bool:
    folded = composed.rstrip("/")
    for path in document_paths:
        other = path.rstrip("/")
        if folded == other or folded.endswith(other) or other.endswith(folded):
            return True
    return False


def _path_rank(spec: dict[str, Any], path: str, document_paths: list[str]) -> tuple[int, int, int]:
    exact = 1 if any(_path_segments(path) == _path_segments(item) for item in document_paths) else 0
    usable = 0
    for server in spec.get("servers") or []:
        server_path = _http_server_path(server)
        if server_path is None:
            continue
        composed = _join_server_path(server_path, path)
        if _path_is_doubled(composed):
            continue
        if _path_matches_document(composed, document_paths):
            usable = 1
            break
    braced = 1 if "{" in path else 0
    return (usable, braced, exact, len(path))


def _union_operation(base: dict[str, Any], incoming: dict[str, Any]) -> None:
    if not str(base.get("description") or "") and incoming.get("description"):
        base["description"] = incoming["description"]
    if not str(base.get("summary") or "") and incoming.get("summary"):
        base["summary"] = incoming["summary"]
    if not base.get("requestBody") and incoming.get("requestBody"):
        base["requestBody"] = incoming["requestBody"]
    incoming_params = [item for item in (incoming.get("parameters") or []) if isinstance(item, dict)]
    if incoming_params:
        current = base.setdefault("parameters", [])
        if isinstance(current, list):
            seen = {(str(item.get("in")), str(item.get("name"))) for item in current if isinstance(item, dict)}
            for item in incoming_params:
                key = (str(item.get("in")), str(item.get("name")))
                if key not in seen:
                    current.append(item)
                    seen.add(key)
    incoming_responses = incoming.get("responses")
    if not isinstance(incoming_responses, dict):
        return
    responses = base.setdefault("responses", {})
    if not isinstance(responses, dict):
        return
    for code, response in incoming_responses.items():
        if not isinstance(response, dict):
            continue
        current = responses.get(code)
        if not isinstance(current, dict) or len(json.dumps(response, ensure_ascii=False, default=str)) > len(
            json.dumps(current, ensure_ascii=False, default=str)
        ):
            responses[code] = response


def _drop_doubling_servers(spec: dict[str, Any], document_paths: list[str]) -> None:
    servers = spec.get("servers")
    paths = spec.get("paths")
    if not isinstance(servers, list) or len(servers) < 2 or not isinstance(paths, dict) or not document_paths:
        return
    useful: list[Any] = []
    for server in servers:
        server_path = _http_server_path(server)
        if server_path is None:
            useful.append(server)
            continue
        matched = False
        for path in paths:
            composed = _join_server_path(server_path, str(path))
            if not _path_is_doubled(composed) and _path_matches_document(composed, document_paths):
                matched = True
                break
        if matched:
            useful.append(server)
    if useful and len(useful) < len(servers):
        spec["servers"] = useful


def _restore_request_prefix(spec: dict[str, Any], source: str) -> None:
    """Путь дополняется префиксом из строки запроса, если сервер этот префикс ещё не содержит."""
    paths = spec.get("paths")
    server_paths = []
    for item in spec.get("servers") or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "")
        if not re.match(r"https?://", url, re.IGNORECASE):
            continue
        server_paths.append(urlsplit(url).path.rstrip("/"))
    if not isinstance(paths, dict) or not server_paths:
        return
    grouped: dict[str, set[str]] = {}
    for method, _origin, example in example_targets(source):
        if not method:
            continue
        for operation_path, methods in paths.items():
            if not isinstance(methods, dict) or method not in methods:
                continue
            prefix = _prefix_before(str(operation_path), example)
            if not prefix or prefix == "/":
                continue
            grouped.setdefault(str(operation_path), set()).add(prefix.rstrip("/"))
    renamed: dict[str, Any] = {}
    for operation_path, methods in paths.items():
        prefixes = grouped.get(str(operation_path), set())
        target = str(operation_path)
        if len(prefixes) == 1:
            prefix = next(iter(prefixes))
            if not any(path == prefix or path.endswith(prefix) for path in server_paths):
                candidate = f"{prefix}/{str(operation_path).lstrip('/')}"
                if candidate not in paths:
                    target = candidate
        bucket = renamed.setdefault(target, {})
        if isinstance(methods, dict) and isinstance(bucket, dict):
            for name, operation in methods.items():
                bucket.setdefault(name, operation)
    spec["paths"] = renamed


def _attach_status_example(response: dict[str, Any], payload: dict[str, Any]) -> None:
    content = response.setdefault("content", {})
    if not isinstance(content, dict):
        return
    media = content.setdefault("application/json", {})
    if not isinstance(media, dict) or media.get("example") is not None:
        return
    media["example"] = payload


def _status_description(response: dict[str, Any], payload: dict[str, Any], reason: str) -> None:
    user_message = _message_field(payload)
    if user_message:
        response["description"] = user_message
        return
    values = " ".join(str(value) for value in payload.values() if isinstance(value, str))
    description = str(response.get("description") or "")
    if description and description not in values:
        message = payload.get("error")
        if isinstance(message, str) and message in values:
            response["description"] = message
        elif reason:
            response["description"] = reason
    elif not description and reason:
        response["description"] = reason


def _add_named_status_responses(spec: dict[str, Any], source: str) -> None:
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return
    for code, method, path, description in _named_statuses(source):
        operation = _operation_for(paths, method, path)
        if operation is None:
            continue
        responses = operation.setdefault("responses", {})
        if not isinstance(responses, dict):
            continue
        current = responses.get(code)
        if current is not None and not _is_template_response(spec, current):
            continue
        response: dict[str, Any] = {"description": description or f"Ответ {code}"}
        if description:
            response["content"] = {
                "application/json": {
                    "schema": {"type": "object", "properties": {"error": {"type": "string"}}},
                    "example": {"error": description},
                }
            }
        responses[code] = response


def _named_statuses(source: str) -> list[tuple[str, str, str, str]]:
    found: list[tuple[str, str, str, str]] = []
    pattern = re.compile(r"\((\d{3})\s+[A-Za-z][^)\n]{0,60}\)|HTTP/1\.[01]\s+(\d{3})\b|Код HTTP:\s*(\d{3})")
    for match in pattern.finditer(source):
        code = next(group for group in match.groups() if group)
        before = source[max(0, match.start() - 900) : match.start()]
        request = None
        for seen in re.finditer(
            r"\b(GET|POST|PUT|PATCH|DELETE)\s+['\"]?\s*<?(https?://[^\s)'\">]+|/[^\s]+)",
            before,
            re.IGNORECASE,
        ):
            request = seen
        if request is None:
            continue
        after = source[match.end() : match.end() + 700]
        message = re.search(r'"messageOutput"\s*:\s*"([^"]+)"', after)
        if message is None:
            message = re.search(r'"(?:error|errorMessage)"\s*:\s*"([^"]+)"', after)
        description = message.group(1) if message else ""
        if not description:
            description = _status_phrases(source).get(code, "")
        raw_path = request.group(2)
        if raw_path.lower().startswith("http"):
            target = _split_example_path(raw_path)
            path = target or ""
        else:
            path = raw_path.split("?")[0].rstrip("/")
            if path.endswith(".json"):
                path = path[: -len(".json")]
        if not path:
            continue
        found.append((code, request.group(1).lower(), path, description))
    return found


def _split_example_path(url: str) -> str:
    from openapi_from_spec.naming import example_targets

    targets = example_targets(f"GET {url}")
    for _method, _origin, path in targets:
        if path:
            return path
    return ""


def _is_template_response(spec: dict[str, Any], response: dict[str, Any]) -> bool:
    ref = response.get("$ref")
    if not isinstance(ref, str):
        return False
    name = ref.rsplit("/", 1)[-1]
    target = ((spec.get("components") or {}).get("responses") or {}).get(name)
    return isinstance(target, dict) and target.get("x-origin") == "skill-template"


def _operation_for(paths: dict[str, Any], method: str, example_path: str) -> dict[str, Any] | None:
    for path, methods in paths.items():
        if not isinstance(methods, dict) or not isinstance(methods.get(method), dict):
            continue
        if _prefix_before(str(path), example_path) is not None:
            return methods[method]
    return None


def _choose_server(spec: dict[str, Any], source: str) -> None:
    paths = spec.get("paths")
    if not isinstance(paths, dict) or not paths:
        return
    placeholders = [
        item
        for item in declared_placeholder_routes(source)
        if item[0]
        and any(
            item[0] in methods and _prefix_before(str(path), item[2]) is not None
            for path, methods in paths.items()
            if isinstance(methods, dict)
        )
    ]
    if placeholders:
        placeholder = placeholders[0][1]
        name = placeholder.strip("{}")
        spec["servers"] = [{"url": placeholder, "variables": {name: {"default": "/"}}}]
        return
    best: tuple[tuple[int, int, str], str] | None = None
    for method, origin, path in example_targets(source):
        if not origin:
            continue
        for operation_path, methods in paths.items():
            if not isinstance(methods, dict):
                continue
            for operation_method in methods:
                if str(operation_method).startswith("x-"):
                    continue
                if method and method != str(operation_method):
                    continue
                prefix = _prefix_before(str(operation_path), path)
                if prefix is None:
                    continue
                url = origin + prefix if prefix else origin
                rank = (len(_path_segments(str(operation_path))), -len(prefix), url)
                if best is None or rank > best[0]:
                    best = (rank, url)
    if best is None:
        return
    spec["servers"] = [{"url": best[1]}]


def _strip_emphasis(node: object) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "description" and isinstance(value, str):
                node[key] = re.sub(r"\*([^*\n]+)\*", r"\1", value)
            else:
                _strip_emphasis(value)
    elif isinstance(node, list):
        for child in node:
            _strip_emphasis(child)


def _align_document_with_source(document: dict[str, Any], source: str) -> None:
    empty = _empty_required_names(source)
    if empty:
        _drop_required_names(document, empty)
    _apply_status_phrases(document, source)
    _apply_formats(document, source)
    _apply_example_server(document, source)
    _drop_contradictory_error_examples(document)


def _empty_required_names(source: str) -> set[str]:
    names: set[str] = set()
    column: int | None = None
    for line in source.splitlines():
        if "|" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        header = next(
            (index for index, cell in enumerate(cells) if "обязат" in cell.lower().replace(" ", "")),
            None,
        )
        if header is not None:
            column = header
            continue
        if column is None or not cells or column >= len(cells):
            continue
        if re.fullmatch(r"[:\-\s]+", cells[0]):
            continue
        if cells[column].strip():
            continue
        name = re.sub(r"\[\]$", "", re.sub(r"\s+", "", cells[0]))
        if name and "." not in name:
            names.add(name)
    return names


def _drop_required_names(node: object, empty: set[str]) -> None:
    if isinstance(node, list):
        for child in node:
            _drop_required_names(child, empty)
        return
    if not isinstance(node, dict):
        return
    required = node.get("required")
    if isinstance(required, list) and ("properties" in node or node.get("type") == "object"):
        kept = [name for name in required if name not in empty]
        if kept:
            node["required"] = kept
        else:
            node.pop("required", None)
    for child in node.values():
        _drop_required_names(child, empty)


def _status_phrases(source: str) -> dict[str, str]:
    phrases: dict[str, str] = {}
    for match in re.finditer(r"Код HTTP:\s*(\d{3})", source):
        before = source[: match.start()]
        heading = before.rfind("## ")
        if heading < 0:
            continue
        lines: list[str] = []
        for line in before[heading:match.start()].splitlines()[1:]:
            text = line.strip()
            if not text or text.startswith("#") or text.startswith("```"):
                continue
            if text.lower().startswith("тело "):
                continue
            lines.append(text)
        if lines:
            phrases[match.group(1)] = " ".join(lines)
    return phrases


def _apply_example_server(document: dict[str, Any], source: str) -> None:
    found = find_template_url(source)
    paths = document.get("paths")
    if found is None or not isinstance(paths, dict):
        return
    _method, origin, full_path = found
    for path in paths:
        if not isinstance(path, str):
            continue
        if path == full_path:
            document["servers"] = [{"url": origin}]
            return
        if full_path.endswith(path):
            document["servers"] = [{"url": origin + full_path[: -len(path)]}]
            return


def _formats_from_source(source: str) -> dict[str, str]:
    found: dict[str, str] = {}
    column: int | None = None
    for line in source.splitlines():
        if "|" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and any("тип" in cell.lower() for cell in cells) and not re.fullmatch(r"[:\-\s]+", cells[0]):
            column = next(index for index, cell in enumerate(cells) if "тип" in cell.lower())
            continue
        if column is None or column >= len(cells) or not cells[0] or re.fullmatch(r"[:\-\s]+", cells[0]):
            continue
        leaf = re.sub(r"\[\]$", "", re.sub(r"\s+", "", cells[0])).split(".")[-1]
        kind = cells[column].lower().replace(" ", "")
        if "url" in kind:
            found[leaf] = "uri"
        elif "timestamp" in kind or "datetime" in kind or "date-time" in kind:
            found[leaf] = "date-time"
    return found


def _apply_formats(document: dict[str, Any], source: str) -> None:
    formats = _formats_from_source(source)

    def walk(node: object) -> None:
        if isinstance(node, list):
            for child in node:
                walk(child)
            return
        if not isinstance(node, dict):
            return
        properties = node.get("properties")
        if isinstance(properties, dict):
            for name, schema in properties.items():
                if isinstance(schema, dict) and schema.get("type") == "string" and name in formats:
                    schema["format"] = formats[name]
        for child in node.values():
            walk(child)

    walk(document)


def _apply_status_phrases(document: dict[str, Any], source: str) -> None:
    phrases = _status_phrases(source)
    paths = document.get("paths")
    if not phrases or not isinstance(paths, dict):
        return
    for methods in paths.values():
        if not isinstance(methods, dict):
            continue
        for method, operation in methods.items():
            if str(method).startswith("x-") or not isinstance(operation, dict):
                continue
            responses = operation.get("responses")
            if not isinstance(responses, dict):
                continue
            for code, response in responses.items():
                phrase = phrases.get(str(code))
                if phrase and isinstance(response, dict) and "$ref" not in response:
                    response["description"] = phrase


def _drop_contradictory_error_examples(document: dict[str, Any]) -> None:
    paths = document.get("paths")
    if not isinstance(paths, dict):
        return
    for methods in paths.values():
        if not isinstance(methods, dict):
            continue
        for method, operation in methods.items():
            if str(method).startswith("x-") or not isinstance(operation, dict):
                continue
            responses = operation.get("responses")
            if not isinstance(responses, dict):
                continue
            for response in responses.values():
                if isinstance(response, dict):
                    _strip_error_example(response, str(response.get("description") or ""))


def _strip_error_example(node: object, description: str) -> None:
    if not isinstance(node, dict):
        return
    example = node.get("example")
    if isinstance(example, dict):
        message = example.get("errorMessage")
        if isinstance(message, str) and message.strip() and message.strip() not in description:
            node.pop("example", None)
    for key, child in list(node.items()):
        if key == "example":
            continue
        if key == "errorMessage" and isinstance(child, dict):
            message = child.get("example")
            if isinstance(message, str) and message.strip() and message.strip() not in description:
                child.pop("example", None)
        _strip_error_example(child, description)


def _replace_success_heading(operation: dict[str, Any]) -> None:
    summary = operation.get("summary")
    responses = operation.get("responses")
    if not isinstance(summary, str) or not summary.strip() or not isinstance(responses, dict):
        return
    success = responses.get("200")
    if success is None:
        success = responses.get(200)
    if not isinstance(success, dict):
        return
    if str(success.get("description") or "").strip().lower() in _SUCCESS_HEADINGS:
        success["description"] = summary.strip()


def _operations_from_document(segment: Segment, document: dict[str, Any] | None) -> list[Operation]:
    found: list[Operation] = []
    paths = document.get("paths") if isinstance(document, dict) else None
    if isinstance(paths, dict):
        for path, methods in paths.items():
            if not isinstance(methods, dict):
                continue
            for method, body in methods.items():
                if str(method).startswith("x-") or not isinstance(body, dict):
                    continue
                operation_id = body.get("operationId")
                found.append(
                    _model_operation(
                        segment,
                        http_method=str(method).lower(),
                        raw_name=segment.raw_name or segment.title or str(operation_id or path),
                        emit_path=str(path),
                        emit_operation_id=str(operation_id) if isinstance(operation_id, str) else None,
                    )
                )
    if not found:
        found.append(_model_operation(segment, gaps=[Gap("Фрагмент не дал операции в paths.")]))
    return found


def _model_operation(
    segment: Segment,
    *,
    http_method: str | None = None,
    raw_name: str = "",
    emit_path: str | None = None,
    emit_operation_id: str | None = None,
    gaps: list[Gap] | None = None,
) -> Operation:
    return Operation(
        http_method=http_method if http_method is not None else segment.http_method,
        raw_name=raw_name or segment.raw_name or segment.title or "фрагмент",
        summary=segment.title,
        description="",
        fields=[],
        span=(segment.start, segment.end),
        source_head=segment.text[:240],
        gaps=list(gaps or []),
        name_origin="model",
        emit_path=emit_path,
        emit_operation_id=emit_operation_id,
    )


def _finish_operation(segment: Segment, llm: _Calls | None, settings: Settings) -> Operation:
    operation = extract_segment(segment)
    if not needs_model(operation, segment.text):
        return operation
    if llm is None:
        operation.gaps.append(
            Gap("Таблицы фрагмента не разобраны. Подключите vLLM, чтобы извлечь поля по скиллу.")
        )
        return operation
    for chunk in _chunks(segment.text, settings):
        system, user = prompt_pair("extract", chunk)
        try:
            payload = llm.complete_json(task="extract", system=system, user=user)
        except Exception as exc:
            operation.gaps.append(Gap(f"Вызов модели для фрагмента не выполнен: {exc}"))
            continue
        apply_model_extract(operation, payload, chunk)
    return operation


def _chunks(text: str, settings: Settings) -> list[str]:
    if len(text) <= settings.max_window_chars:
        return [text]
    return [
        render_window(text, start, end)
        for start, end in pack_windows(text, settings.max_window_chars, settings.window_overlap)
    ]


def _extract_long_segment(segment: Segment, llm: _Calls, settings: Settings) -> Operation:
    operation = extract_segment(segment)
    # Таблицы этого шаблона не размечены знакомыми заголовками, поэтому поля берёт модель.
    operation.fields = []
    operation.gaps = []
    operation.request_example = None
    operation.response_example = None
    if segment.title:
        operation.summary = segment.title
    _bind_fragment_request(operation, segment.text)
    if not operation.explicit_path:
        _apply_template_url(operation, segment.text)
    card = _method_card(segment)
    for start, end in pack_windows(segment.text, settings.max_window_chars, settings.window_overlap):
        shown = render_window(segment.text, start, end)
        system, user = prompt_pair("extract", shown, card=card)
        try:
            payload = llm.complete_json(task="extract", system=system, user=user)
        except Exception as exc:
            operation.gaps.append(Gap(f"Вызов модели для фрагмента не выполнен: {exc}"))
            continue
        apply_model_extract(operation, payload, shown)
    _bind_fragment_request(operation, segment.text)
    if not operation.explicit_path:
        _apply_template_url(operation, segment.text)
    operation.fields = nest_dotted_fields(operation.fields, operation.gaps)
    return operation


def _bind_fragment_request(operation: Operation, text: str) -> None:
    """Путь берётся из строки запроса фрагмента. Пример с образцом идентификатора её не заменяет."""
    calls = [(method, path) for _at, method, path in _unwritten_calls(text)]
    method = operation.http_method
    own = [path for verb, path in calls if method is None or verb == method]
    if method is None and len(calls) == 1:
        operation.http_method = calls[0][0]
        own = [calls[0][1]]
    braced = [path for path in own if "{" in path]
    chosen = braced[0] if len(braced) == 1 else own[0] if len(own) == 1 else None
    if chosen is None:
        return
    operation.explicit_path = chosen
    pieces = [piece for piece in chosen.split("/") if piece and not (piece.startswith("{") and piece.endswith("}"))]
    if pieces and (not operation.raw_name or "/" in operation.raw_name or "{" in operation.raw_name):
        operation.raw_name = pieces[-1]
        operation.name_origin = "extracted"
    if _placeholder_name(text):
        operation.server_url = None


def _apply_template_url(operation: Operation, text: str) -> None:
    found = find_template_url(text)
    if found is None:
        return
    method, server, path = found
    operation.http_method = method
    operation.server_url = server
    operation.explicit_path = path
    pieces = [piece for piece in path.split("/") if piece and not (piece.startswith("{") and piece.endswith("}"))]
    if pieces:
        operation.raw_name = pieces[-1]
        operation.name_origin = "extracted"
    for match in re.finditer(r"\{([^{}]+)\}", path):
        name = match.group(1)
        if any(field.name == name and field.location == "path" for field in operation.fields):
            continue
        operation.fields.append(
            IRField(
                name=name,
                location="path",
                required=True,
                origin="extracted",
                quote=path if path in text else name,
            )
        )


def _missed_request_segments(text: str, segments: list[Segment]) -> list[Segment]:
    """Фрагмент для строки curl или полного адреса, которую модель не включила ни в один метод."""
    missed: list[Segment] = []
    seen: set[tuple[str, str]] = set()
    for method, path, start, end in _request_line_spans(text):
        key = (method, path)
        if key in seen or any(_segment_covers_request(segment, method, path) for segment in segments):
            continue
        seen.add(key)
        fragment_start, fragment_end = _request_fragment(text, start, end)
        raw_name, explicit_path, query = parse_target(path)
        missed.append(
            Segment(
                start=fragment_start,
                end=fragment_end,
                text=text[fragment_start:fragment_end],
                http_method=method,
                raw_name=raw_name,
                explicit_path=explicit_path or path,
                query=query,
                anchor="request-line",
            )
        )
    return missed


_OWN_REQUEST = re.compile(
    r"(?m)^(?:curl\b[^\n]*-X\s+)?"
    r"(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+['\"]?\s*<?"
    r"(https?://[^\s>'\"\\]+|/[^\s>'\"\\]+)",
    re.IGNORECASE,
)


def _request_path(raw: str) -> str:
    raw = raw.rstrip(".,;")
    if raw.lower().startswith("http"):
        return _split_example_path(raw)
    path = raw.split("?")[0]
    return path if path.startswith("/") else ""


def _owner_request(text: str) -> tuple[str, str] | None:
    match = _OWN_REQUEST.search(text)
    if match is None:
        return None
    path = _request_path(match.group(2))
    if not path:
        return None
    return match.group(1).lower(), path


def _limit_to_own_request(text: str, start: int, end: int) -> int:
    """Конец фрагмента — перед строкой запроса другого метода или другого маршрута."""
    owned_method = ""
    owned_path = ""
    owned_at = -1
    for match in _OWN_REQUEST.finditer(text):
        at = match.start()
        if at < start or at >= end:
            continue
        path = _request_path(match.group(2))
        method = match.group(1).lower()
        if not path:
            continue
        if owned_at < 0:
            owned_method, owned_path, owned_at = method, path, at
            continue
        if method == owned_method and _same_route(path, owned_path):
            continue
        heading = start
        for mark in re.finditer(r"(?m)^#{1,2} ", text[owned_at:at]):
            heading = owned_at + mark.start()
        if heading > owned_at:
            return heading
        line = text.rfind("\n", start, at)
        return start if line < 0 else line + 1
    return end


def _request_line_spans(text: str) -> list[tuple[str, str, int, int]]:
    found: list[tuple[str, str, int, int]] = []
    patterns = (
        re.compile(
            r"curl\b.*?-X\s+(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b.*?<?(https?://[^\s>'\"\\]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?m)^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+['\"]?<?(https?://[^\s>'\"\\]+)",
            re.IGNORECASE,
        ),
    )
    for pattern in patterns:
        for match in pattern.finditer(text):
            target = _split_example_path(match.group(2))
            if target:
                found.append((match.group(1).lower(), target, match.start(), match.end()))
    return found


def _request_fragment(text: str, match_start: int, match_end: int) -> tuple[int, int]:
    headings = list(re.finditer(r"(?m)^#{1,3} ", text[:match_start]))
    start = headings[-1].start() if headings else match_start
    following = re.search(r"(?m)^#{1,3} ", text[match_end:])
    end = match_end + following.start() if following else len(text)
    return start, end


def _segment_covers_request(segment: Segment, method: str, path: str) -> bool:
    owner = _owner_request(segment.text)
    if owner is not None and owner[0] != method:
        return False
    if segment.http_method and segment.http_method != method:
        return False
    if path and path in segment.text:
        return True
    route = owner[1] if owner is not None else segment.explicit_path
    return bool(route and _same_route(route, path))


def _segment_own_call(segment: Segment) -> tuple[str, str] | None:
    if segment.http_method and segment.explicit_path:
        return segment.http_method, segment.explicit_path
    for _at, method, path in _unwritten_calls(segment.text):
        if segment.http_method is None or method == segment.http_method:
            return method, path
    if segment.http_method:
        return segment.http_method, ""
    return None


def _call_continues(own: tuple[str, str] | None, method: str, path: str) -> bool:
    if own is None or method != own[0]:
        return False
    own_path = own[1]
    if not own_path:
        return True
    if _same_route(path, own_path) or _example_version_of(path, own_path):
        return True
    own_parts = [part for part in own_path.split("/") if part]
    path_parts = [part for part in path.split("/") if part]
    return bool(path_parts and len(own_parts) >= len(path_parts) and _segments_match(own_parts[-len(path_parts) :], path_parts))


def _method_heading_at(chunk: str) -> int | None:
    match = re.search(
        r"(?m)^#{1,3}\s+\S.*\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b",
        chunk,
    )
    return match.start() if match else None


def _has_foreign_call(segment: Segment, chunk: str) -> bool:
    if _method_heading_at(chunk) is not None:
        return True
    own = _segment_own_call(segment)
    return any(not _call_continues(own, method, path) for _at, method, path in _unwritten_calls(chunk))


def _attach_gap_before_foreign(text: str, segment: Segment, boundary: int) -> None:
    if not segment.http_method or boundary <= segment.end:
        return
    gap = text[segment.end : boundary]
    cut = len(gap)
    heading = _method_heading_at(gap)
    if heading is not None:
        cut = min(cut, heading)
    own = _segment_own_call(segment)
    for at, method, path in _unwritten_calls(gap):
        if not _call_continues(own, method, path):
            cut = min(cut, at)
            break
    if cut <= 0 or not gap[:cut].strip():
        return
    segment.end += cut
    segment.text = text[segment.start : segment.end]


def _absorb_continuations(text: str, segments: list[Segment]) -> list[Segment]:
    """Хвост без новой строки запроса остаётся у предыдущего метода, в том числе за границей цитаты."""
    ordered = sorted(segments, key=lambda item: (item.start, item.end))
    if not ordered:
        return []
    kept: list[Segment] = [ordered[0]]
    for segment in ordered[1:]:
        previous = kept[-1]
        gap = text[previous.end : segment.start]
        if previous.http_method and not _has_foreign_call(previous, gap) and not _has_foreign_call(previous, segment.text):
            previous.end = max(previous.end, segment.end)
            previous.text = text[previous.start : previous.end]
            continue
        _attach_gap_before_foreign(text, previous, segment.start)
        kept.append(segment)
    _attach_gap_before_foreign(text, kept[-1], len(text))
    return kept


def _segment_with_model(text: str, llm: _Calls, settings: Settings) -> tuple[list[Segment], list[str]]:
    found: list[Segment] = []
    failures: list[str] = []
    open_segment: Segment | None = None
    for start, end in pack_windows(text, settings.max_window_chars, settings.window_overlap):
        shown = render_window(text, start, end)
        card = _method_card(open_segment) if open_segment is not None else ""
        system, user = prompt_pair("segment", shown, start=start, end=end, total=len(text), card=card)
        payload, error = _complete_segment(llm, system, user)
        if error and _window_has_method(shown):
            payload, error = _complete_segment(llm, system, user)
        if error:
            failures.append(error)
            continue
        accepted, missed = _accept_segments(text, start, end, payload)
        if not accepted and _window_has_method(shown):
            payload, error = _complete_segment(llm, system, user)
            if error:
                failures.append(error)
            else:
                accepted, missed = _accept_segments(text, start, end, payload)
        continuation = payload.get("continues_previous") if isinstance(payload, dict) else None
        if open_segment is not None and isinstance(continuation, dict):
            span = _quote_span(text, start, end, continuation)
            if span is not None and span[1] > open_segment.end:
                open_segment.end = extend_segment_tail(text, span[1], end)
                open_segment.end = _limit_to_own_request(text, open_segment.start, open_segment.end)
                open_segment.text = text[open_segment.start : open_segment.end]
        if missed and not accepted:
            quote = str(missed[0].get("start_quote") or "").strip()
            if quote:
                failures.append(f"Цитата границы не найдена в окне: {quote[:120]}")
        for segment in accepted:
            found.append(segment)
            open_segment = segment
    if found:
        for segment in found:
            limited = _limit_to_own_request(text, segment.start, segment.end)
            if limited < segment.end:
                segment.end = limited
                segment.text = text[segment.start : segment.end]
        found = _absorb_continuations(text, found)
        found.extend(_missed_request_segments(text, found))
    return _drop_contained(found), failures


def _complete_segment(llm: _Calls, system: str, user: str) -> tuple[dict[str, Any], str]:
    try:
        payload = llm.complete_json(task="segment", system=system, user=user)
    except Exception as exc:
        return {}, str(exc)
    if not isinstance(payload, dict):
        return {}, ""
    return payload, ""


def _accept_segments(
    text: str, start: int, end: int, payload: dict[str, Any]
) -> tuple[list[Segment], list[dict[str, Any]]]:
    accepted: list[Segment] = []
    missed: list[dict[str, Any]] = []
    for item in payload.get("segments") or []:
        if not isinstance(item, dict):
            continue
        segment = _segment_from_item(text, start, end, item)
        if segment is None:
            missed.append(item)
            continue
        accepted.append(segment)
    return accepted, missed


def _window_has_method(shown: str) -> bool:
    if not re.search(r"\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b", shown, re.IGNORECASE):
        return False
    folded = shown.lower()
    return "curl" in folded or "/" in shown or "http" in folded


def _method_card(segment: Segment) -> str:
    found = find_template_url(segment.text)
    if found is None:
        return ""
    method, server, path = found
    match = re.search(
        rf"\b{method}\s+{re.escape(server + path)}",
        segment.text,
        re.IGNORECASE,
    )
    if match is None:
        return ""
    return segment.text[match.start() : match.end()]


def _quote_span(text: str, start: int, end: int, item: dict) -> tuple[int, int] | None:
    start_quote = str(item.get("start_quote") or "")
    end_quote = str(item.get("end_quote") or "")
    body = text[start:end]
    begin = _locate_quote(body, start_quote)
    if begin is None:
        return None
    finish = _locate_quote(body, end_quote, begin[1])
    if finish is None:
        return None
    abs_start = text.rfind("\n", 0, start + begin[0]) + 1
    abs_end = text.find("\n", start + finish[1])
    if abs_end < 0:
        abs_end = len(text)
    if abs_end <= abs_start:
        return None
    return abs_start, abs_end


def _locate_quote(body: str, quote: str, offset: int = 0) -> tuple[int, int] | None:
    """Находит цитату в окне. Пробелы и скобки вокруг URL не обязаны совпасть символ в символ."""
    if not quote:
        return None
    exact = body.find(quote, offset)
    if exact >= 0:
        return exact, exact + len(quote)
    pieces = [piece for piece in re.split(r"\s+", quote.strip()) if piece]
    if len(pieces) < 2 and len(quote.strip()) < 12:
        return None
    parts: list[str] = []
    for piece in pieces:
        cleaned = piece.replace("«", '"').replace("»", '"').replace("“", '"').replace("”", '"')
        url = re.fullmatch(r"[\"']?<?(https?://[^\s>'\"<>]+)>?[\"']?", cleaned)
        if url:
            parts.append(r"\"?<?" + re.escape(url.group(1)) + r">?\"?")
        else:
            parts.append(re.escape(cleaned.strip("'\"")))
    match = re.search(r"\s+".join(parts), body[offset:])
    if match is None:
        return None
    return offset + match.start(), offset + match.end()


def _segment_from_item(text: str, start: int, end: int, item: object) -> Segment | None:
    if not isinstance(item, dict):
        return None
    span = _quote_span(text, start, end, item)
    if span is None:
        return None
    abs_start, abs_end = span
    abs_end = extend_segment_tail(text, abs_end, end)
    abs_end = _limit_to_own_request(text, abs_start, abs_end)
    chunk = text[abs_start:abs_end]
    method = str(item.get("http_method") or "").lower()
    owner = _owner_request(chunk)
    if owner is not None:
        method = owner[0]
    elif method not in _VERBS or not re.search(rf"\b{method}\b", chunk, re.IGNORECASE):
        method = ""
    raw_name = str(item.get("raw_name") or "")
    explicit = None
    query: dict[str, list[str]] = {}
    if raw_name and raw_name in chunk:
        raw_name, explicit, query = parse_target(raw_name)
    else:
        raw_name = ""
    title = str(item.get("title") or "").strip()
    if title and title not in chunk:
        title = ""
    return Segment(
        start=abs_start,
        end=abs_end,
        text=chunk,
        http_method=method or None,
        raw_name=raw_name,
        explicit_path=explicit,
        query=query,
        title=title,
        anchor="model",
    )


def _drop_contained(segments: list[Segment]) -> list[Segment]:
    ordered = sorted(segments, key=lambda item: (item.start, -(item.end - item.start)))
    kept: list[Segment] = []
    for segment in ordered:
        if any(other.start <= segment.start and other.end >= segment.end for other in kept):
            continue
        kept.append(segment)
    return sorted(kept, key=lambda item: item.start)


def _report(
    text: str,
    segments: list[Segment],
    operations: list[Operation],
    document_gaps: list[Gap],
    conventions: list[str],
    settings: Settings,
    model_calls: int,
    templates: list[str] | None = None,
) -> dict:
    def gap_row(gap: Gap, raw_name: str = "") -> dict[str, str]:
        return {"operation": raw_name, "field": gap.field, "question": gap.question, "quote": gap.quote}

    flat = [gap_row(gap) for gap in document_gaps]
    published = []
    unresolved = []
    for operation in operations:
        row = {
            "raw_name": operation.raw_name,
            "http_method": operation.http_method,
            "path": operation.emit_path,
            "operation_id": operation.emit_operation_id,
            "name_origin": operation.name_origin,
            "source_head": operation.source_head,
            "span": list(operation.span),
        }
        if operation.http_method and operation.emit_path:
            published.append(row)
        else:
            unresolved.append(row)
        flat.extend(gap_row(gap, operation.raw_name) for gap in operation.gaps)
    preamble = text[: segments[0].start].strip() if segments else ""
    return {
        "openapi": "3.0.3",
        "title": settings.title,
        "model_calls": model_calls,
        "conventions": conventions,
        "templates": list(ERROR_RESPONSES) if templates is None and settings.error_templates else list(templates or []),
        "operations": published,
        "unresolved": unresolved,
        "gaps": flat,
        "preamble": preamble[:1000],
    }
