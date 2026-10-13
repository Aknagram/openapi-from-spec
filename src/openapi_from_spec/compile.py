from __future__ import annotations

import json
from typing import Any

import yaml

from openapi_from_spec.conventions import (
    DEFAULT_PATH_PREFIX,
    ERROR_RESPONSES,
    ERROR_SCHEMA_NAME,
    OPENAPI_VERSION,
    TYPE_MAP,
)
from openapi_from_spec.models import Gap, IRField, Operation
from openapi_from_spec.naming import kebab, pascal, singular

_ARRAY_TYPES = {"array", "list", "set", "массив", "список", "array [<objects>]"}


class _Dumper(yaml.SafeDumper):
    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        super().increase_indent(flow, False)


def dump_yaml(spec: dict[str, Any]) -> str:
    return yaml.dump(
        spec,
        Dumper=_Dumper,
        allow_unicode=True,
        sort_keys=False,
        width=88,
        default_flow_style=False,
    )


class Registry:
    def __init__(self) -> None:
        self.schemas: dict[str, dict[str, Any]] = {}
        self._shapes: dict[str, str] = {}

    def hoist(self, base: str, schema: dict[str, Any]) -> str:
        shape = json.dumps(_shape(schema), ensure_ascii=False, sort_keys=True)
        existing = self._shapes.get(shape)
        if existing:
            return existing
        name = base if base and base[0].isalpha() else f"Schema{base}"
        unique = name
        number = 2
        while unique in self.schemas:
            unique = f"{name}{number}"
            number += 1
        self.schemas[unique] = schema
        self._shapes[shape] = unique
        return unique


def _shape(schema: dict[str, Any]) -> Any:
    if "$ref" in schema:
        return {"$ref": schema["$ref"]}
    kept: dict[str, Any] = {}
    for key, value in schema.items():
        if key in {"description", "example", "x-origin"}:
            continue
        if key == "properties" and isinstance(value, dict):
            kept[key] = {name: _shape(item) for name, item in value.items()}
        elif key == "items" and isinstance(value, dict):
            kept[key] = _shape(value)
        else:
            kept[key] = value
    return kept


def _is_array(field: IRField) -> bool:
    return (field.tz_type or "").strip().lower() in _ARRAY_TYPES


def _string_items(field: IRField) -> bool:
    return bool(field.examples) and all(isinstance(item, str) for item in field.examples)


def schema_for(field: IRField, registry: Registry, gaps: list[Gap], conventions: list[str]) -> dict[str, Any]:
    if field.children and _is_array(field):
        item = _object_schema(field.children, registry, gaps, conventions)
        name = registry.hoist(pascal(singular(field.name)), item)
        schema: dict[str, Any] = {"type": "array", "items": {"$ref": f"#/components/schemas/{name}"}}
    elif field.children:
        nested = _object_schema(field.children, registry, gaps, conventions)
        if len(field.children) >= 2:
            name = registry.hoist(pascal(field.name), nested)
            schema = {"$ref": f"#/components/schemas/{name}"}
        else:
            schema = nested
    else:
        schema = _scalar_schema(field, gaps, conventions)
    if field.description and "$ref" not in schema and "description" not in schema:
        schema["description"] = field.description
    return schema


def _scalar_schema(field: IRField, gaps: list[Gap], conventions: list[str]) -> dict[str, Any]:
    key = (field.tz_type or "").strip().lower()
    if key in _ARRAY_TYPES:
        if _string_items(field):
            items: dict[str, Any] = {"type": "string"}
            if field.max_length:
                items["maxLength"] = field.max_length
                _note(conventions, "Максимальная длина списка перенесена на строковый элемент.")
            return _with_description({"type": "array", "items": items}, field)
        gaps.append(Gap("Не указан тип элемента списка.", field=field.name, quote=field.quote))
        schema = {"type": "array", "items": {}}
        if field.max_length:
            schema["x-tz-max-length"] = field.max_length
        return _with_description(schema, field)
    if not key or key not in TYPE_MAP:
        if field.location == "response" or key:
            gaps.append(
                Gap(
                    "Тип поля не сопоставлен со словарём OpenAPI.",
                    field=field.name,
                    quote=field.quote or (field.tz_type or ""),
                )
            )
        schema = {}
        if field.tz_type:
            schema["x-tz-type"] = field.tz_type
        return _with_description(schema, field)
    kind, fmt = TYPE_MAP[key]
    if kind == "object":
        schema = {"type": "object"}
    else:
        schema = {"type": kind}
        if fmt:
            schema["format"] = fmt
    if field.max_length and schema.get("type") == "string":
        schema["maxLength"] = field.max_length
    elif field.max_length and schema.get("type") == "integer":
        schema["x-tz-max-length"] = field.max_length
    return _with_description(schema, field)


def _with_description(schema: dict[str, Any], field: IRField) -> dict[str, Any]:
    if field.description:
        schema["description"] = field.description
    return schema


def _object_schema(
    fields: list[IRField],
    registry: Registry,
    gaps: list[Gap],
    conventions: list[str],
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for item in fields:
        properties[item.name] = schema_for(item, registry, gaps, conventions)
        if item.required is True:
            required.append(item.name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _note(conventions: list[str], text: str) -> None:
    if text not in conventions:
        conventions.append(text)


def _path_for(operation: Operation, prefix: str, conventions: list[str]) -> str:
    if operation.explicit_path:
        parts = []
        for piece in operation.explicit_path.split("/"):
            if not piece:
                continue
            if piece.startswith("{") and piece.endswith("}"):
                parts.append(piece)
            else:
                parts.append(kebab(piece))
        return "/" + "/".join(parts)
    _note(conventions, f"К имени метода добавлен префикс {prefix}. В ТЗ полного пути не было.")
    slug = kebab(operation.raw_name)
    base = prefix.rstrip("/")
    return f"{base}/{slug}" if base else f"/{slug}"


def _parameter(operation: Operation, field: IRField, registry: Registry, gaps: list[Gap], conventions: list[str]) -> dict[str, Any]:
    schema = schema_for(field, registry, gaps, conventions)
    schema.pop("description", None)
    body: dict[str, Any] = {"name": field.name, "in": field.location, "schema": schema}
    if field.required is True or field.location == "path":
        body["required"] = True
    elif field.required is False:
        body["required"] = False
    if field.description:
        body["description"] = field.description
    if field.examples:
        body["example"] = field.examples if schema.get("type") == "array" or len(field.examples) > 1 else field.examples[0]
    if field.location == "query" and field.name in operation.repeated_query and schema.get("type") == "array":
        body["style"] = "form"
        body["explode"] = True
    return body


def _resolve(schema: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    ref = schema.get("$ref")
    if not ref:
        return schema
    name = ref.rsplit("/", 1)[-1]
    return ((spec.get("components") or {}).get("schemas") or {}).get(name, {})


def extra_example_fields(example: Any, schema: dict[str, Any], spec: dict[str, Any], prefix: str, gaps: list[Gap]) -> None:
    schema = _resolve(schema, spec)
    if isinstance(example, dict) and schema.get("type") == "object":
        props = schema.get("properties") or {}
        for key, value in example.items():
            if key not in props:
                gaps.append(Gap(f"Пример содержит поле {prefix}{key}, которого нет в таблице.", field=str(key)))
            else:
                extra_example_fields(value, props[key], spec, f"{prefix}{key}.", gaps)
    elif isinstance(example, list) and schema.get("type") == "array":
        for item in example:
            extra_example_fields(item, schema.get("items") or {}, spec, prefix, gaps)


def compile_operations(
    operations: list[Operation],
    *,
    title: str,
    version: str,
    path_prefix: str = DEFAULT_PATH_PREFIX,
    error_templates: bool = True,
    server_url: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    registry = Registry()
    conventions: list[str] = []
    paths: dict[str, dict[str, Any]] = {}
    used_ids: set[str] = set()
    used_routes: set[tuple[str, str]] = set()
    used_template = False

    for operation in operations:
        if not operation.http_method:
            continue
        gaps = operation.gaps
        base_path = _path_for(operation, path_prefix, conventions)
        path = base_path
        suffix = 2
        while (operation.http_method, path) in used_routes:
            path = f"{base_path}-{suffix}"
            suffix += 1
            gaps.append(Gap("Путь повторялся и получил числовой суффикс.", field=path))
        used_routes.add((operation.http_method, path))
        operation_id = f"{operation.http_method}{pascal(operation.raw_name)}"
        base_id = operation_id
        number = 2
        while operation_id in used_ids:
            operation_id = f"{base_id}{number}"
            number += 1
        used_ids.add(operation_id)
        operation.emit_path = path
        operation.emit_operation_id = operation_id

        parameters = [
            _parameter(operation, field, registry, gaps, conventions)
            for field in operation.fields
            if field.location in {"query", "path", "header"}
        ]
        body_fields = [field for field in operation.fields if field.location == "body"]
        response_fields = [field for field in operation.fields if field.location == "response"]
        op_body: dict[str, Any] = {"operationId": operation_id}
        if operation.summary:
            op_body["summary"] = operation.summary
        if operation.description:
            op_body["description"] = operation.description
        if parameters:
            op_body["parameters"] = parameters
        if body_fields:
            body_schema = _object_schema(body_fields, registry, gaps, conventions)
            body_name = registry.hoist(pascal(operation.raw_name) + "Request", body_schema)
            content: dict[str, Any] = {"schema": {"$ref": f"#/components/schemas/{body_name}"}}
            if operation.request_example is not None:
                content["example"] = operation.request_example
            request_body: dict[str, Any] = {"content": {"application/json": content}}
            if any(field.required is True for field in body_fields):
                request_body["required"] = True
            op_body["requestBody"] = request_body
            _note(conventions, "Тело запроса записано как application/json.")

        responses: dict[str, Any] = {}
        success = operation.status_responses.get("200") or operation.summary or "Успешный ответ"
        if response_fields:
            response_schema = _object_schema(response_fields, registry, gaps, conventions)
            response_name = registry.hoist(pascal(operation.raw_name) + "Response", response_schema)
            content = {"schema": {"$ref": f"#/components/schemas/{response_name}"}}
            if operation.response_example is not None:
                content["example"] = operation.response_example
            responses["200"] = {
                "description": success,
                "content": {"application/json": content},
            }
            _note(conventions, "Успешный ответ записан как application/json.")
        elif operation.response_example is not None:
            responses["200"] = {
                "description": success,
                "content": {"application/json": {"schema": {}, "example": operation.response_example}},
            }
        else:
            responses["200"] = {"description": success}
        documented = {code: text for code, text in operation.status_responses.items() if code != "200"}
        if documented:
            for code, description in documented.items():
                responses[code] = {"description": description}
            _note(conventions, "Коды ответов взяты из текста метода, шаблон поверх них не ставился.")
        elif error_templates:
            used_template = True
            for code, description in ERROR_RESPONSES.items():
                responses[code] = {
                    "description": description,
                    "x-origin": "skill-template",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": f"#/components/schemas/{ERROR_SCHEMA_NAME}"}
                        }
                    },
                }
        op_body["responses"] = responses
        paths.setdefault(path, {})[operation.http_method] = op_body

    if used_template:
        registry.schemas[ERROR_SCHEMA_NAME] = {
            "type": "object",
            "x-origin": "skill-template",
            "description": "Шаблон ошибки. В ТЗ этого тела не было.",
            "properties": {
                "code": {"type": "string"},
                "message": {"type": "string"},
            },
        }
        _note(conventions, "Ответы 400, 401, 404 и 500 добавлены шаблоном скилла, это не поля ТЗ.")

    spec: dict[str, Any] = {
        "openapi": OPENAPI_VERSION,
        "info": {"title": title, "version": version},
        "paths": paths,
    }
    servers: list[dict[str, str]] = []
    if server_url:
        servers.append({"url": server_url})
    for operation in operations:
        if operation.server_url and not any(item["url"] == operation.server_url for item in servers):
            servers.append({"url": operation.server_url})
    if servers:
        spec["servers"] = servers
    if registry.schemas:
        spec["components"] = {"schemas": registry.schemas}

    for operation in operations:
        if operation.response_example is None or not operation.emit_path or not operation.http_method:
            continue
        response = (
            spec["paths"]
            .get(operation.emit_path, {})
            .get(operation.http_method, {})
            .get("responses", {})
            .get("200", {})
        )
        content = ((response.get("content") or {}).get("application/json") or {})
        if "schema" in content:
            extra_example_fields(operation.response_example, content["schema"], spec, "", operation.gaps)
    return spec, conventions
