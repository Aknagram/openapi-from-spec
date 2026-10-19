"""Прогон синтетических ТЗ. Не часть пакета: только проверка универсальности."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openapi_from_spec.pipeline import Settings, convert_text  # noqa: E402

DOCS = ROOT / "probe" / "docs"
OUT = ROOT / "out" / "probe"


def operations(spec: dict) -> list[tuple[str, str, dict]]:
    found = []
    for path, methods in (spec.get("paths") or {}).items():
        if not isinstance(methods, dict):
            continue
        for method, operation in methods.items():
            if str(method).startswith("x-") or not isinstance(operation, dict):
                continue
            found.append((str(method).lower(), str(path), operation))
    return found


def find_op(spec: dict, method: str, suffix: str) -> dict | None:
    for verb, path, operation in operations(spec):
        if verb == method and (path.rstrip("/") == suffix.rstrip("/") or path.rstrip("/").endswith(suffix.rstrip("/"))):
            return operation
    return None


def response(operation: dict | None, code: str) -> dict | None:
    if operation is None:
        return None
    responses = operation.get("responses") or {}
    value = responses.get(code) or responses.get(int(code))
    return value if isinstance(value, dict) else None


def schema_props(spec: dict, operation: dict | None, where: str) -> dict:
    if operation is None:
        return {}
    node = operation.get("requestBody") if where == "request" else None
    if where != "request":
        media = ((response(operation, where) or {}).get("content") or {}).get("application/json") or {}
        node = media
    if where == "request":
        content = (operation.get("requestBody") or {}).get("content") or {}
        node = content.get("application/json") or {}
    schema = node.get("schema") if isinstance(node, dict) else None
    return _props(spec, schema)


def _props(spec: dict, schema: object) -> dict:
    if not isinstance(schema, dict):
        return {}
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/"):
        cursor: object = spec
        for part in ref[2:].split("/"):
            cursor = cursor.get(part) if isinstance(cursor, dict) else None
        return _props(spec, cursor)
    props = schema.get("properties")
    return props if isinstance(props, dict) else {}


def text_of(node: object) -> str:
    return json.dumps(node, ensure_ascii=False)


def check(name: str, spec: dict, report: dict) -> list[str]:
    problems: list[str] = []
    ops = operations(spec)
    if report.get("gaps"):
        questions = [item.get("question", "") for item in report["gaps"]]
        if any("Модель недоступна" in question or "только модель" in question for question in questions):
            problems.append("модель не разобрала документ: " + " | ".join(questions))
            return problems

    def need(method: str, suffix: str) -> dict | None:
        operation = find_op(spec, method, suffix)
        if operation is None:
            problems.append(f"нет {method.upper()} {suffix}; есть {[(m, p) for m, p, _ in ops]}")
        return operation

    if name.startswith("01"):
        get = need("get", "/courier/orders")
        post = need("post", "/courier/orders")
        extra = [f"{m.upper()} {p}" for m, p, _ in ops if m in {"put", "delete"}]
        if extra:
            problems.append("лишние операции из списка глаголов: " + ", ".join(extra))
        props = schema_props(spec, get, "200")
        orders = props.get("orders")
        if not isinstance(orders, dict) or orders.get("type") != "array":
            problems.append("orders не массив")
        body = schema_props(spec, post, "request")
        if "address" not in (body.get("required") or _required_parent(spec, post)):
            if "address" not in _request_required(spec, post):
                problems.append("address не обязателен в теле заказа")
        weight = body.get("weightKg") if isinstance(body, dict) else None
        if isinstance(weight, dict) and weight.get("nullable") is not True:
            problems.append("weightKg из примера null не nullable")
    elif name.startswith("02"):
        need("get", "/clinic/slots")
        need("post", "/clinic/slots")
        servers = spec.get("servers") or []
        if not any("{apiUrl}" in str(item.get("url")) or "apiUrl" in text_of(item) for item in servers if isinstance(item, dict)):
            problems.append(f"сервер не плейсхолдер apiUrl: {servers}")
        for verb, path, operation in ops:
            codes = set((operation.get("responses") or {}))
            for forbidden in ("401", "404", "500"):
                if forbidden in codes:
                    problems.append(f"{verb} {path} содержит {forbidden} вне закрытого списка")
            for required_code in ("403", "503"):
                if required_code not in codes:
                    problems.append(f"{verb} {path} без {required_code}")
    elif name.startswith("03"):
        get = need("get", "/readings")
        props = schema_props(spec, get, "200")
        readings = props.get("readings")
        if not isinstance(readings, dict) or readings.get("type") != "array":
            problems.append("readings не массив")
        elif readings.get("description") != "Показание счётчика":
            problems.append(f"описание readings: {readings.get('description')!r}")
        date = _param(get, "date")
        if date is None:
            problems.append("нет query date")
        elif (date.get("schema") or {}).get("format"):
            problems.append("у date оставлен один format при двух форматах")
        conflict = response(get, "409")
        if conflict is None:
            problems.append("нет ответа 409")
        elif "Показание уже есть" not in text_of(conflict):
            problems.append("409 без текста messageToUser")
        servers = text_of(spec.get("servers"))
        if "grid.example" not in servers:
            problems.append(f"сервер не grid.example: {servers}")
    elif name.startswith("04"):
        post = need("post", "/parking/sessions")
        delete = need("delete", "/parking/sessions/{sessionId}")
        body = schema_props(spec, post, "request")
        if "amount" in body:
            problems.append("вычисляемое amount попало в запрос")
        if "plate" not in _request_required(spec, post):
            problems.append("plate не обязателен")
        gone = response(delete, "204")
        if gone is None:
            problems.append("нет 204")
        elif gone.get("content"):
            problems.append("у 204 есть тело, хотя тело ответа пустое")
    elif name.startswith("05"):
        post = need("post", "/library/loans")
        example = _request_example(post)
        if example == {"readerId": ""}:
            problems.append("в запрос попал некорректный пример")
        required = _request_required(spec, post)
        if not {"readerId", "bookId"} <= required:
            problems.append(f"обязательные поля запроса: {sorted(required)}")
        error = response(post, "422")
        if error is None or "задолженность" not in text_of(error):
            problems.append("422 без фразы про задолженность")
    elif name.startswith("06"):
        need("get", "/stations")
        observations = need("get", "/observations")
        if "meteo.example" not in text_of(spec.get("servers")):
            problems.append(f"сервер не meteo.example: {spec.get('servers')}")
        header = _param(observations, "X-Api-Key")
        if header is None or header.get("in") != "header":
            problems.append("X-Api-Key не header")
        denied = response(observations, "401")
        if denied is None or "Нужен ключ станции" not in text_of(denied):
            problems.append("401 без messageOutput")
        props = schema_props(spec, observations, "200")
        if not isinstance(props.get("observations"), dict) or props["observations"].get("type") != "array":
            problems.append("observations не массив")
    elif name.startswith("07"):
        patch = need("patch", "/warehouse/moves/{moveId}")
        if spec.get("info", {}).get("version") != "4":
            problems.append(f"версия {spec.get('info', {}).get('version')!r}")
        if "Склад" not in str(spec.get("info", {}).get("title")):
            problems.append(f"заголовок {spec.get('info', {}).get('title')!r}")
        props = schema_props(spec, patch, "200")
        comment = props.get("comment")
        if isinstance(comment, dict) and comment.get("nullable") is not True:
            problems.append("comment из null не nullable")
        if "destination" not in _request_required(spec, patch):
            problems.append("destination не обязателен")
    elif name.startswith("08"):
        post = need("post", "/refunds")
        if "tickets.example" not in text_of(spec.get("servers")):
            problems.append(f"сервер не tickets.example: {spec.get('servers')}")
        bad = response(post, "400")
        if bad is None:
            problems.append("нет 400")
        elif bad.get("content"):
            problems.append("у 400 есть тело, хотя оно пустое")
        error = response(post, "422")
        if error is None or "Возврат уже оформлен" not in text_of(error):
            problems.append("422 без errorMessage")
        if not {"ticket", "reason"} <= _request_required(spec, post):
            problems.append(f"обязательные поля: {sorted(_request_required(spec, post))}")
    elif name.startswith("09"):
        get = need("get", "/canteen/menu")
        post = need("post", "/canteen/menu/items")
        props = schema_props(spec, get, "200")
        items = props.get("items")
        if not isinstance(items, dict) or items.get("type") != "array":
            problems.append("items не массив")
        else:
            item_props = _props(spec, items.get("items"))
            name_prop = item_props.get("name") if item_props else props.get("name")
            if not isinstance(name_prop, dict) or name_prop.get("description") != "Название блюда":
                problems.append(f"описание name: {None if not isinstance(name_prop, dict) else name_prop.get('description')!r}")
        if not {"name", "price"} <= _request_required(spec, post):
            problems.append(f"Yes/No не стали required: {sorted(_request_required(spec, post))}")
    elif name.startswith("10"):
        get = need("get", "/archive/cards/{cardId}")
        delete = need("delete", "/archive/cards/{cardId}")
        summary = str((get or {}).get("summary") or "")
        for forbidden in ("шаг", "Описываем", "указываем", "логичнее"):
            if forbidden.lower() in summary.lower():
                problems.append(f"summary взято из ремарки: {summary!r}")
        if get and "карточку" not in summary.lower() and "карточку" not in str(get.get("description") or "").lower():
            problems.append(f"summary без фразы метода: {summary!r}")
        gone = response(delete, "204")
        if gone is None:
            problems.append("нет 204 у удаления")
        elif gone.get("content"):
            problems.append("у 204 есть тело")
        if "archive.example" not in text_of(spec.get("servers")):
            problems.append(f"сервер не archive.example: {spec.get('servers')}")
    return problems


def _required_parent(spec: dict, operation: dict | None) -> list:
    return []


def _request_required(spec: dict, operation: dict | None) -> set[str]:
    if operation is None:
        return set()
    content = ((operation.get("requestBody") or {}).get("content") or {}).get("application/json") or {}
    schema = content.get("schema")
    if isinstance(schema, dict) and isinstance(schema.get("$ref"), str):
        cursor: object = spec
        for part in schema["$ref"][2:].split("/"):
            cursor = cursor.get(part) if isinstance(cursor, dict) else None
        schema = cursor
    required = schema.get("required") if isinstance(schema, dict) else None
    return {str(item) for item in required} if isinstance(required, list) else set()


def _request_example(operation: dict | None) -> object:
    if operation is None:
        return None
    content = ((operation.get("requestBody") or {}).get("content") or {}).get("application/json") or {}
    return content.get("example")


def _param(operation: dict | None, name: str) -> dict | None:
    if operation is None:
        return None
    for item in operation.get("parameters") or []:
        if isinstance(item, dict) and str(item.get("name")).lower() == name.lower():
            return item
    return None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        vllm_base_url="http://127.0.0.1:11434/api",
        vllm_model="qwen3.8:27b",
        timeout_seconds=180,
    )
    lines: list[str] = []
    failed = 0
    for path in sorted(DOCS.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        try:
            result = convert_text(text, settings)
        except Exception as exc:  # noqa: BLE001 — прогон должен сохранить причину и идти дальше
            failed += 1
            message = f"{path.name}: ИСКЛЮЧЕНИЕ {type(exc).__name__}: {exc}"
            lines.append(message)
            if "10061" in str(exc) or "Connection" in type(exc).__name__ or "отказ" in str(exc).lower():
                lines.append("Модель недоступна, остальные документы не запускались.")
                break
            lines.append(traceback.format_exc())
            continue
        (OUT / f"{path.stem}.yaml").write_text(result.yaml_text, encoding="utf-8")
        (OUT / f"{path.stem}.report.json").write_text(
            json.dumps(result.report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        problems = check(path.name, result.spec, result.report)
        if result.errors:
            problems.append("ошибки схемы: " + "; ".join(result.errors[:4]))
        if any("Модель недоступна" in item or "только модель" in item for item in problems):
            lines.append(f"{path.name}: модель недоступна")
            lines.extend(f"  - {item}" for item in problems)
            lines.append("Остальные документы не запускались.")
            break
        calls = result.report.get("model_calls")
        if problems:
            failed += 1
            lines.append(f"{path.name}: {len(problems)} проблем, вызовов {calls}")
            lines.extend(f"  - {item}" for item in problems)
        else:
            lines.append(f"{path.name}: ок, вызовов {calls}, операций {len(operations(result.spec))}")
        (OUT / "score.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(lines[-1] if not problems else lines[-1 - len(problems)], flush=True)
    (OUT / "score.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
