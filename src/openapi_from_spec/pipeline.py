from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import yaml

from openapi_from_spec.compile import compile_operations, dump_yaml
from openapi_from_spec.conventions import DEFAULT_PATH_PREFIX, ERROR_RESPONSES
from openapi_from_spec.extract import extract_segment, needs_model
from openapi_from_spec.llm import (
    AgentBridgeClient,
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
from openapi_from_spec.validate import schema_errors, validate_document

_VERBS = {"get", "post", "put", "patch", "delete", "head", "options"}


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
        segments: list[Segment] = []
    else:
        segments = find_segments(text)
        if not segments:
            document_gaps.append(
                Gap(
                    "В документе не нашлись границы методов. Подключите vLLM: модель получит окна текста и скилл сегментации."
                )
            )

    operations = [_finish_operation(segment, counted, settings) for segment in segments]
    spec, conventions = compile_operations(
        operations,
        title=settings.title,
        version=settings.version,
        path_prefix=settings.path_prefix,
        error_templates=settings.error_templates,
        server_url=settings.server_url,
    )
    errors = validate_document(spec, operations)
    report = _report(text, segments, operations, document_gaps, conventions, settings, counted.count if counted else 0)
    return ConvertResult(yaml_text=dump_yaml(spec), spec=spec, report=report, errors=errors)


def _client_from_settings(settings: Settings) -> LlmClient | None:
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


def _convert_with_model(text: str, settings: Settings, llm: _Calls) -> ConvertResult:
    document_gaps: list[Gap] = []
    segments, segment_errors = _segment_with_model(text, llm, settings)
    if not segments:
        detail = "Модель не выделила ни одного фрагмента с дословными границами."
        if segment_errors:
            detail = f"{detail} {segment_errors[0]}"
        document_gaps.append(Gap(detail))
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
    finalize_spec(spec, text)
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


def finalize_spec(spec: dict[str, Any], source: str) -> None:
    """Выравнивает сшитый документ по строкам запроса, таблице и кодам примера."""
    _brace_colon_paths(spec)
    _ensure_declared_operations(spec, source)
    _rewrite_paths(spec, source)
    _drop_unevidenced_operations(spec, source)
    _move_unresolved_path_params(spec)
    _align_method_summaries(spec, source)
    _apply_request_required(spec, source)
    _add_named_status_responses(spec, source)
    _align_table_objects(spec, source)
    _apply_status_catalog(spec, source)
    _choose_server(spec, source)
    _strip_emphasis(spec)


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


def _ensure_declared_operations(spec: dict[str, Any], source: str) -> None:
    routes = [(method, path, name) for method, _placeholder, path, name in declared_placeholder_routes(source) if method and path]
    if not routes:
        return
    paths = spec.setdefault("paths", {})
    if not isinstance(paths, dict):
        return
    headings = _method_headings(source)
    seen: set[tuple[str, str]] = set()
    for method, path, name in routes:
        key = (method, path)
        if key in seen:
            continue
        seen.add(key)
        methods = paths.setdefault(path, {})
        if not isinstance(methods, dict) or isinstance(methods.get(method), dict):
            continue
        operation_id = name or _operation_id_from_path(method, path)
        summary = headings.get(operation_id.lower(), "")
        operation: dict[str, Any] = {
            "operationId": operation_id,
            "responses": {"200": {"description": summary or operation_id}},
        }
        if summary:
            operation["summary"] = summary
        for param in re.findall(r"\{([^{}]+)\}", path):
            _ensure_path_param(operation, param, source)
        methods[method] = operation


def _operation_id_from_path(method: str, path: str) -> str:
    parts = [re.sub(r"[^A-Za-z0-9]", "", piece) for piece in _path_segments(path) if not piece.startswith("{")]
    tail = "".join(piece[:1].upper() + piece[1:] for piece in parts if piece)
    return method.lower() + (tail or "Root")


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
            for item in parameters:
                if isinstance(item, dict) and item.get("in") == "path" and item.get("name") not in names:
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
    table = _status_code_table(source)
    match = re.search(r"следующ\w*\s+HTTP-код", source, re.IGNORECASE)
    if match is None and not table:
        return None
    codes = set(table)
    if match is not None:
        codes.update(re.findall(r"\b([1-5]\d{2})\b", source[match.start() :]))
    return codes or None


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
                if isinstance(response, dict) and _is_template_response(spec, response):
                    responses.pop(code, None)
            for code, description in descriptions.items():
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
                if isinstance(response, dict) and _is_template_response(spec, response):
                    responses.pop(code, None)
    _drop_unused_templates(spec)


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
            for name, field in by_name.items():
                if str(method) in field["methods"] and name in properties:
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
                    properties[name] = field["schema"]


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
    _apply_template_url(operation, segment.text)
    operation.fields = nest_dotted_fields(operation.fields, operation.gaps)
    return operation


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


def _segment_with_model(text: str, llm: _Calls, settings: Settings) -> tuple[list[Segment], list[str]]:
    found: list[Segment] = []
    failures: list[str] = []
    open_segment: Segment | None = None
    for start, end in pack_windows(text, settings.max_window_chars, settings.window_overlap):
        shown = render_window(text, start, end)
        card = _method_card(open_segment) if open_segment is not None else ""
        system, user = prompt_pair("segment", shown, start=start, end=end, total=len(text), card=card)
        try:
            payload = llm.complete_json(task="segment", system=system, user=user)
        except Exception as exc:
            failures.append(str(exc))
            continue
        if not isinstance(payload, dict):
            continue
        continuation = payload.get("continues_previous")
        if open_segment is not None and isinstance(continuation, dict):
            span = _quote_span(text, start, end, continuation)
            if span is not None and span[1] > open_segment.end:
                open_segment.end = extend_segment_tail(text, span[1], end)
                open_segment.text = text[open_segment.start : open_segment.end]
        for item in payload.get("segments") or []:
            segment = _segment_from_item(text, start, end, item)
            if segment is None:
                continue
            found.append(segment)
            open_segment = segment
    return _drop_contained(found), failures


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
    begin = body.find(start_quote)
    if begin < 0 or not start_quote:
        return None
    finish = body.find(end_quote, begin + len(start_quote))
    if finish < 0 or not end_quote:
        return None
    abs_start = text.rfind("\n", 0, start + begin) + 1
    abs_end = text.find("\n", start + finish + len(end_quote))
    if abs_end < 0:
        abs_end = len(text)
    if abs_end <= abs_start:
        return None
    return abs_start, abs_end


def _segment_from_item(text: str, start: int, end: int, item: object) -> Segment | None:
    if not isinstance(item, dict):
        return None
    span = _quote_span(text, start, end, item)
    if span is None:
        return None
    abs_start, abs_end = span
    abs_end = extend_segment_tail(text, abs_end, end)
    chunk = text[abs_start:abs_end]
    method = str(item.get("http_method") or "").lower()
    if method not in _VERBS or not re.search(rf"\b{method}\b", chunk, re.IGNORECASE):
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
