from __future__ import annotations

from openapi_from_spec.models import Operation


def validate_document(spec: dict, operations: list[Operation]) -> list[str]:
    errors: list[str] = []
    if spec.get("openapi") != "3.0.3":
        errors.append("Поле openapi должно быть равно 3.0.3.")
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        errors.append("В спецификации нет объекта paths.")
        paths = {}
    schemas = ((spec.get("components") or {}).get("schemas") or {})
    for ref in _refs(spec):
        if not ref.startswith("#/components/schemas/"):
            errors.append(f"Ссылка {ref} выходит за components/schemas.")
            continue
        name = ref.rsplit("/", 1)[-1]
        if name not in schemas:
            errors.append(f"Схема {ref} не объявлена.")
    for path, methods in paths.items():
        if not isinstance(methods, dict):
            errors.append(f"Путь {path} описан не объектом.")
            continue
        for method, operation in methods.items():
            if method.startswith("x-") or not isinstance(operation, dict):
                continue
            if "operationId" not in operation or "responses" not in operation:
                errors.append(f"У {method.upper()} {path} нет operationId или responses.")
    for operation in operations:
        if operation.http_method is None:
            if not operation.gaps:
                errors.append(f"Фрагмент «{operation.raw_name}» не попал в paths и не содержит вопроса.")
            continue
        if not operation.emit_path or not operation.emit_operation_id:
            errors.append(f"Метод {operation.http_method} {operation.raw_name} не записан в paths.")
    return errors


def schema_errors(spec: dict) -> list[str]:
    """Проверяет собранный документ схемой OpenAPI 3.0.3."""
    try:
        from openapi_spec_validator import validate
    except ImportError:
        return ["Пакет openapi-spec-validator не установлен."]
    errors: list[str] = []
    try:
        validate(spec)
    except Exception as exc:
        errors.append(str(exc))
    errors.extend(_missing_array_items(spec))
    return errors


def _missing_array_items(spec: dict) -> list[str]:
    found: list[str] = []

    def walk(node: object, name: str) -> None:
        if isinstance(node, list):
            for child in node:
                walk(child, name)
            return
        if not isinstance(node, dict):
            return
        if node.get("type") == "array" and "items" not in node and "$ref" not in node:
            found.append(f"У массива {name} нет items.")
        properties = node.get("properties")
        if isinstance(properties, dict):
            for key, child in properties.items():
                walk(child, key)
        items = node.get("items")
        if isinstance(items, dict):
            walk(items, name)
        for key, child in node.items():
            if key in {"properties", "items", "example"}:
                continue
            walk(child, name)

    walk(spec, "schema")
    return found


def _refs(node: object) -> list[str]:
    found: list[str] = []

    def walk(value: object) -> None:
        if isinstance(value, dict):
            ref = value.get("$ref")
            if isinstance(ref, str):
                found.append(ref)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(node)
    return found
