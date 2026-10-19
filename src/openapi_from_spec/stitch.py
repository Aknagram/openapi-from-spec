from __future__ import annotations

import datetime as dt
import json
from typing import Any


def _json_default(value: Any) -> str:
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def stitch_documents(parts: list[dict[str, Any]], *, title: str, version: str) -> tuple[dict[str, Any], list[str]]:
    """Складывает YAML фрагментов. Одинаковая форма схемы получает один $ref."""
    notes: list[str] = []
    paths: dict[str, dict[str, Any]] = {}
    schemas: dict[str, Any] = {}
    responses: dict[str, Any] = {}
    shape_owner: dict[str, str] = {}
    response_owner: dict[str, str] = {}
    seen_routes: set[tuple[str, str]] = set()
    seen_ids: set[str] = set()
    servers: list[Any] = []
    for part in parts:
        copied = json.loads(json.dumps(part, ensure_ascii=False, default=_json_default))
        rename = _schema_renames(copied, schemas, shape_owner, notes)
        _apply_renames(copied, rename, "#/components/schemas/")
        response_rename = _response_renames(copied, responses, response_owner, notes)
        _apply_renames(copied, response_rename, "#/components/responses/")
        _take_schemas(copied, rename, schemas)
        _take_responses(copied, response_rename, responses)
        _take_paths(copied, paths, schemas, seen_routes, seen_ids, notes)
        for server in copied.get("servers") or []:
            if server not in servers:
                servers.append(server)
    spec: dict[str, Any] = {
        "openapi": "3.0.3",
        "info": {"title": title, "version": str(version)},
        "paths": paths,
    }
    if servers:
        spec["servers"] = servers
    components: dict[str, Any] = {}
    if schemas:
        components["schemas"] = schemas
    if responses:
        components["responses"] = responses
    if components:
        spec["components"] = components
    return spec, notes


def _schema_renames(
    document: dict[str, Any],
    schemas: dict[str, Any],
    shape_owner: dict[str, str],
    notes: list[str],
) -> dict[str, str]:
    rename: dict[str, str] = {}
    incoming = ((document.get("components") or {}).get("schemas") or {})
    for name, schema in incoming.items():
        if not isinstance(schema, dict):
            continue
        shape = json.dumps(_shape(schema), ensure_ascii=False, sort_keys=True)
        owner = shape_owner.get(shape)
        if owner:
            rename[name] = owner
            continue
        target = name
        if target in schemas or target in rename.values():
            target = _fresh(name, set(schemas) | set(rename.values()))
            notes.append(f"Схема {name} повторилась с другой формой и записана как {target}.")
        shape_owner[shape] = target
        rename[name] = target
    return rename


def _response_renames(
    document: dict[str, Any],
    responses: dict[str, Any],
    shape_owner: dict[str, str],
    notes: list[str],
) -> dict[str, str]:
    rename: dict[str, str] = {}
    incoming = ((document.get("components") or {}).get("responses") or {})
    for name, response in incoming.items():
        if not isinstance(response, dict):
            continue
        shape = json.dumps(response, ensure_ascii=False, sort_keys=True)
        owner = shape_owner.get(shape)
        if owner:
            rename[name] = owner
            continue
        target = name
        if target in responses or target in rename.values():
            target = _fresh(name, set(responses) | set(rename.values()))
            notes.append(f"Ответ {name} повторился с другой формой и записан как {target}.")
        shape_owner[shape] = target
        rename[name] = target
    return rename


def _take_responses(document: dict[str, Any], rename: dict[str, str], responses: dict[str, Any]) -> None:
    incoming = ((document.get("components") or {}).get("responses") or {})
    for name, response in incoming.items():
        target = rename.get(name, name)
        if target not in responses and isinstance(response, dict):
            responses[target] = response


def _take_schemas(document: dict[str, Any], rename: dict[str, str], schemas: dict[str, Any]) -> None:
    incoming = ((document.get("components") or {}).get("schemas") or {})
    for name, schema in incoming.items():
        target = rename.get(name, name)
        if target not in schemas and isinstance(schema, dict):
            schemas[target] = schema


def _take_paths(
    document: dict[str, Any],
    paths: dict[str, dict[str, Any]],
    schemas: dict[str, Any],
    seen_routes: set[tuple[str, str]],
    seen_ids: set[str],
    notes: list[str],
) -> None:
    incoming = document.get("paths")
    if not isinstance(incoming, dict):
        if incoming is not None:
            notes.append("Фрагмент без объекта paths пропущен при сшивке.")
        return
    for path, methods in incoming.items():
        if not isinstance(methods, dict):
            notes.append(f"Путь {path} пропущен: значение не объект.")
            continue
        for method, operation in methods.items():
            if str(method).startswith("x-") or not isinstance(operation, dict):
                continue
            route = (str(method), str(path))
            if route in seen_routes:
                label = f"{str(method).upper()} {path}"
                _merge_operation(paths[str(path)][str(method)], operation, schemas, notes, label)
                notes.append(f"Повтор {label} объединён с первым фрагментом.")
                continue
            operation_id = operation.get("operationId")
            if isinstance(operation_id, str) and operation_id in seen_ids:
                fresh = _fresh(operation_id, seen_ids)
                notes.append(f"operationId {operation_id} повторён и записан как {fresh}.")
                operation["operationId"] = fresh
                operation_id = fresh
            if isinstance(operation_id, str):
                seen_ids.add(operation_id)
            paths.setdefault(str(path), {})[str(method)] = operation
            seen_routes.add(route)


def _merge_operation(
    base: dict[str, Any],
    incoming: dict[str, Any],
    schemas: dict[str, Any],
    notes: list[str],
    label: str,
) -> None:
    _merge_parameters(base, incoming, notes, label)
    _merge_responses(base, incoming, schemas, notes, label)


def _merge_parameters(base: dict[str, Any], incoming: dict[str, Any], notes: list[str], label: str) -> None:
    incoming_params = [item for item in (incoming.get("parameters") or []) if isinstance(item, dict)]
    if not incoming_params:
        return
    current = base.setdefault("parameters", [])
    if not isinstance(current, list):
        return
    seen = {
        (str(item.get("in")), str(item.get("name"))): item
        for item in current
        if isinstance(item, dict)
    }
    for item in incoming_params:
        key = (str(item.get("in")), str(item.get("name")))
        previous = seen.get(key)
        if previous is None:
            current.append(item)
            seen[key] = item
            continue
        _note_type_conflict(previous.get("schema"), item.get("schema"), notes, f"{label} {key[1]}")


def _merge_responses(
    base: dict[str, Any],
    incoming: dict[str, Any],
    schemas: dict[str, Any],
    notes: list[str],
    label: str,
) -> None:
    current = base.setdefault("responses", {})
    if not isinstance(current, dict):
        return
    incoming_responses = incoming.get("responses")
    if not isinstance(incoming_responses, dict):
        return
    for code, response in incoming_responses.items():
        if not isinstance(response, dict):
            continue
        previous = current.get(code)
        if not isinstance(previous, dict):
            current[code] = response
            continue
        left = _json_schema(previous)
        right = _json_schema(response)
        if left is None and right is not None:
            previous["content"] = response.get("content")
            continue
        if isinstance(left, dict) and isinstance(right, dict):
            _merge_schema(left, right, schemas, notes, f"{label} {code}")


def _json_schema(response: dict[str, Any]) -> dict[str, Any] | None:
    content = response.get("content")
    if not isinstance(content, dict):
        return None
    body = content.get("application/json")
    if not isinstance(body, dict):
        return None
    schema = body.get("schema")
    return schema if isinstance(schema, dict) else None


def _merge_schema(
    left: dict[str, Any],
    right: dict[str, Any],
    schemas: dict[str, Any],
    notes: list[str],
    label: str,
) -> None:
    left_schema = _deref(left, schemas)
    right_schema = _deref(right, schemas)
    left_type = left_schema.get("type")
    right_type = right_schema.get("type")
    if left_type and right_type and left_type != right_type:
        notes.append(f"Конфликт типа {label}: {left_type} и {right_type}.")
        return
    if not left_type and right_type:
        left_schema["type"] = right_type
        if "format" in right_schema and "format" not in left_schema:
            left_schema["format"] = right_schema["format"]
    if "properties" in left_schema or "properties" in right_schema or left_schema.get("type") == "object":
        props = left_schema.setdefault("properties", {})
        if isinstance(props, dict):
            for name, schema in (right_schema.get("properties") or {}).items():
                if not isinstance(schema, dict):
                    continue
                if name not in props:
                    props[name] = json.loads(json.dumps(schema))
                elif isinstance(props[name], dict):
                    _merge_schema(props[name], schema, schemas, notes, f"{label}.{name}")
    if left_schema.get("type") == "array" and isinstance(right_schema.get("items"), dict):
        items = left_schema.setdefault("items", {})
        if isinstance(items, dict):
            _merge_schema(items, right_schema["items"], schemas, notes, f"{label}[]")


def _note_type_conflict(left: object, right: object, notes: list[str], label: str) -> None:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return
    left_type = left.get("type")
    right_type = right.get("type")
    if left_type and right_type and left_type != right_type:
        notes.append(f"Конфликт типа {label}: {left_type} и {right_type}.")


def _deref(schema: dict[str, Any], schemas: dict[str, Any]) -> dict[str, Any]:
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
        found = schemas.get(ref.rsplit("/", 1)[-1])
        if isinstance(found, dict):
            return found
    return schema


def _apply_renames(node: Any, rename: dict[str, str], prefix: str) -> None:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith(prefix):
            name = ref.rsplit("/", 1)[-1]
            if name in rename:
                node["$ref"] = prefix + rename[name]
        for value in node.values():
            _apply_renames(value, rename, prefix)
    elif isinstance(node, list):
        for value in node:
            _apply_renames(value, rename, prefix)


def _shape(schema: dict[str, Any]) -> Any:
    if "$ref" in schema:
        return {"$ref": schema["$ref"]}
    kept: dict[str, Any] = {}
    for key, value in schema.items():
        if key in {"description", "example"}:
            continue
        if key == "properties" and isinstance(value, dict):
            kept[key] = {name: _shape(item) if isinstance(item, dict) else item for name, item in value.items()}
        elif key == "items" and isinstance(value, dict):
            kept[key] = _shape(value)
        else:
            kept[key] = value
    return kept


def _fresh(name: str, taken: set[str]) -> str:
    number = 2
    candidate = f"{name}{number}"
    while candidate in taken:
        number += 1
        candidate = f"{name}{number}"
    return candidate
