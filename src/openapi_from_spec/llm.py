from __future__ import annotations

import json
import re
import time
from typing import Any, Protocol
from urllib.parse import urlparse

from openapi_from_spec.models import Gap, IRField, Operation
from openapi_from_spec.skill_loader import system_prompt

_LOCATIONS = {"query", "path", "header", "cookie", "body", "response"}
_VERBS = {"get", "post", "put", "patch", "delete", "head", "options"}


class LlmClient(Protocol):
    def complete_json(self, *, task: str, system: str, user: str) -> dict[str, Any]:
        """Возвращает JSON одной задачи скилла."""


def user_prompt(
    task: str,
    fragment: str,
    *,
    start: int = 0,
    end: int = 0,
    total: int = 0,
    previous: str = "",
    errors: list[str] | None = None,
    card: str = "",
) -> str:
    prefix = ""
    if card:
        prefix = (
            "Уже открытый метод. Строка ниже — дословная цитата из предыдущих окон, не из этого окна:\n"
            f"{card}\n\n"
        )
    if task == "segment":
        return (
            prefix
            + f"Окно символов {start}–{end} из документа длиной {total}. "
            + "В запросе нет остальных частей документа.\n\n"
            + fragment
        )
    if task == "openapi" and (previous or errors):
        details = "\n".join(errors or [])
        return (
            "Фрагмент ТЗ:\n\n"
            f"{fragment}\n\n"
            "Проверка OpenAPI 3.0.3 не пройдена:\n"
            f"{details}\n\n"
            "Предыдущий YAML:\n"
            f"{previous}\n\n"
            "Верни исправленный YAML только для этого фрагмента."
        )
    if task == "openapi":
        return (
            "Ниже один фрагмент ТЗ. Верни YAML OpenAPI 3.0.3 только для его операций.\n\n"
            f"{fragment}"
        )
    return prefix + "Ниже один фрагмент ТЗ. Извлеки из него одну операцию.\n\n" + fragment


def parse_json_content(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        finish = text.rfind("}")
        if start < 0 or finish <= start:
            raise
        data = json.loads(text[start : finish + 1])
    if not isinstance(data, dict):
        raise ValueError("модель вернула не JSON-объект")
    return data


def _post_model(client, url: str, headers: dict[str, str], payload: dict[str, Any]):
    """Повторяет запрос, если хост модели на мгновение не принимает соединение."""
    import httpx

    delay = 1.0
    for attempt in range(3):
        try:
            return client.post(url, headers=headers, json=payload)
        except httpx.TransportError:
            if attempt == 2:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("модель недоступна")


def ollama_chat_url(url: str | None) -> str | None:
    """Локальный Ollama: POST /api/chat, а не OpenAI /v1/chat/completions."""
    parsed = urlparse((url or "").strip())
    if not parsed.scheme or not parsed.netloc:
        return None
    path = parsed.path.rstrip("/")
    if parsed.port != 11434 and path not in {"/api", "/api/chat"}:
        return None
    return f"{parsed.scheme}://{parsed.netloc}/api/chat"


class VllmClient:
    """Клиент OpenAI-совместимого vLLM. Скилл передаётся в каждом запросе, сеть модели не нужна."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def complete_json(self, *, task: str, system: str, user: str) -> dict[str, Any]:
        import httpx

        ollama = ollama_chat_url(self.base_url)
        url = ollama or (self.base_url + "/chat/completions")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if ollama:
            payload = {"model": self.model, "stream": False, "think": False, "messages": messages}
            extra = {"format": "json"} if task != "openapi" else {}
        else:
            payload = {"model": self.model, "temperature": 0, "messages": messages}
            extra = {"response_format": {"type": "json_object"}} if task != "openapi" else {}
        with httpx.Client(timeout=self.timeout) as client:
            response = _post_model(client, url, headers, {**payload, **extra} if extra else payload)
            if response.status_code == 400 and extra:
                response = _post_model(client, url, headers, payload)
            if response.status_code >= 400:
                raise RuntimeError(f"модель вернула HTTP {response.status_code}")
            body = response.json()
        content = _chat_content(body)
        if task == "openapi":
            return {"yaml": content}
        return parse_json_content(content)


def _chat_content(body: dict[str, Any]) -> str:
    message = body.get("message")
    if isinstance(message, dict) and "content" in message:
        return str(message.get("content") or "")
    return str(body["choices"][0]["message"]["content"])


def prompt_pair(task: str, fragment: str, **meta: Any) -> tuple[str, str]:
    return system_prompt(task), user_prompt(task, fragment, **meta)


def apply_model_extract(operation: Operation, payload: dict[str, Any], source: str) -> None:
    if not isinstance(payload, dict):
        operation.gaps.append(Gap("Модель вернула не объект JSON."))
        return
    method = str(payload.get("http_method") or "").lower()
    if operation.http_method is None and method in _VERBS and re.search(rf"\b{method}\b", source, re.IGNORECASE):
        operation.http_method = method
    raw_name = payload.get("raw_name")
    if (
        operation.name_origin != "extracted"
        and isinstance(raw_name, str)
        and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", raw_name)
        and raw_name in source
    ):
        operation.raw_name = raw_name
        operation.name_origin = "model"
    if not operation.summary and isinstance(payload.get("summary"), str) and payload["summary"] in source:
        operation.summary = payload["summary"]
    if not operation.description and isinstance(payload.get("description"), str) and payload["description"] in source:
        operation.description = payload["description"]

    known = {field.name: field for field in operation.fields}
    for key, fallback in (("parameters", "query"), ("body_fields", "body"), ("response_fields", "response")):
        for raw in payload.get(key) or []:
            field = _field_from_model(raw, source, fallback)
            if field is None:
                continue
            current = known.get(field.name)
            if current is not None:
                if current.tz_type is None and field.tz_type:
                    current.tz_type = field.tz_type
                    current.required = current.required if current.required is not None else field.required
                    current.description = current.description or field.description
                    current.quote = field.quote or current.quote
                continue
            operation.fields.append(field)
            known[field.name] = field
    for item in payload.get("responses") or []:
        _take_status(operation, item, source)
    example = payload.get("example")
    if operation.response_example is None and example is not None and _grounded(example, source):
        operation.response_example = example
    for item in (payload.get("gaps") or [])[:30]:
        if isinstance(item, dict) and str(item.get("question") or "").strip():
            operation.gaps.append(
                Gap(
                    str(item["question"]).strip(),
                    field=str(item.get("field") or ""),
                    quote=str(item.get("quote") or ""),
                )
            )
    _rehome(operation)
    _place_unmarked_request_fields(operation, source)


def _table_places(source: str) -> dict[str, str]:
    """Имя из строки таблицы и расположение, если ячейка его назвала. Пустая строка — расположения нет."""
    found: dict[str, str] = {}
    for line in source.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2 or set(cells[0]) <= {"-", ":"}:
            continue
        name = cells[0].strip("{} ")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", name):
            continue
        place = ""
        for cell in cells[1:]:
            token = cell.casefold()
            if token in {"query", "строка", "строка запроса"}:
                place = "query"
            elif token in {"header", "заголовок"}:
                place = "header"
            elif token in {"path", "путь"}:
                place = "path"
            elif token in {"body", "тело", "тело запроса"}:
                place = "body"
        found[name] = place
    return found


def _place_unmarked_request_fields(operation: Operation, source: str) -> None:
    """У POST, PUT и PATCH поле таблицы без расположения — тело. У GET такое поле не остаётся телом."""
    places = _table_places(source)
    for field in operation.fields:
        if field.name not in places:
            continue
        named = places[field.name]
        if named:
            if field.location in {"query", "unresolved", "body"}:
                field.location = named
            continue
        if operation.http_method in {"post", "put", "patch"} and field.location in {"query", "unresolved"}:
            field.location = "body"
        elif operation.http_method in {"get", "head", "delete"} and field.location == "body":
            field.location = "response"


def _rehome(operation: Operation) -> None:
    if operation.http_method is None:
        return
    for field in operation.fields:
        if field.location != "unresolved":
            continue
        field.location = "body" if operation.http_method in {"post", "put", "patch"} else "query"


_DOTTED_TOKEN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\[\])?(?:\.[A-Za-z_][A-Za-z0-9_]*(?:\[\])?)+"
    r"|[A-Za-z_][A-Za-z0-9_]*\[\]"
)
_DOTTED_PART = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)(\[\])?")


def align_field_name(name: str, quote: str) -> str:
    """Ячейка `author.id` задаёт имя свойства `id`, но в поле хранится полный путь."""
    for match in _DOTTED_TOKEN.finditer(quote):
        token = match.group(0)
        parts = _DOTTED_PART.findall(token)
        last = parts[-1][0] if parts else token
        if name == token or name == last:
            return token
    return name


def nest_dotted_fields(fields: list[IRField], gaps: list[Gap]) -> list[IRField]:
    """Собирает `author.id` и `ingredients[].name` во вложенные свойства."""
    result: list[IRField] = []
    tops: dict[str, IRField] = {}
    child_index: dict[int, dict[str, IRField]] = {}

    def lookup_of(node: IRField) -> dict[str, IRField]:
        found = child_index.get(id(node))
        if found is None:
            found = {item.name: item for item in node.children}
            child_index[id(node)] = found
        return found

    for field in fields:
        parts = _path_parts(field.name)
        if not parts:
            result.append(field)
            tops[field.name] = field
            continue
        bucket = result
        lookup = tops
        for depth, (part_name, is_array) in enumerate(parts):
            leaf = depth == len(parts) - 1
            node = lookup.get(part_name)
            if node is None:
                if leaf:
                    tz_type = field.tz_type or ("array" if is_array else None)
                else:
                    tz_type = "array" if is_array else "object"
                node = IRField(
                    name=part_name,
                    tz_type=tz_type,
                    max_length=field.max_length if leaf else None,
                    required=field.required if leaf else None,
                    description=field.description if leaf else "",
                    quote=field.quote,
                    origin=field.origin,
                    location=field.location,
                    children=list(field.children) if leaf else [],
                )
                bucket.append(node)
                lookup[part_name] = node
            elif leaf:
                _fill_leaf(node, field, gaps)
            elif is_array and not _is_array_type(node.tz_type):
                if node.tz_type and node.tz_type.strip().lower() not in {"object", "<object>"}:
                    gaps.append(
                        Gap(
                            f"Конфликт типа {part_name}: {node.tz_type} и array.",
                            field=part_name,
                            quote=field.quote,
                        )
                    )
                else:
                    node.tz_type = "array"
            if not leaf:
                bucket = node.children
                lookup = lookup_of(node)
    return result


def _path_parts(token: str) -> list[tuple[str, bool]] | None:
    if not _DOTTED_TOKEN.fullmatch(token):
        return None
    return [(name, bool(marker)) for name, marker in _DOTTED_PART.findall(token)]


def _is_array_type(tz_type: str | None) -> bool:
    return (tz_type or "").strip().lower() in {"array", "list", "set", "массив", "список", "array [<objects>]"}


def _fill_leaf(node: IRField, field: IRField, gaps: list[Gap]) -> None:
    if field.tz_type and node.tz_type and field.tz_type != node.tz_type and not node.children:
        gaps.append(
            Gap(
                f"Конфликт типа {node.name}: {node.tz_type} и {field.tz_type}.",
                field=node.name,
                quote=field.quote,
            )
        )
        return
    if not node.tz_type and field.tz_type:
        node.tz_type = field.tz_type
    if node.required is None:
        node.required = field.required
    if not node.description:
        node.description = field.description
    if field.quote:
        node.quote = field.quote


def _take_status(operation: Operation, item: object, source: str) -> None:
    if not isinstance(item, dict):
        return
    code = str(item.get("code") or "").strip()
    description = str(item.get("description") or "").strip()
    quote = str(item.get("quote") or "")
    if not code.isdigit() or code not in source:
        return
    if description and description not in source:
        return
    if quote and quote not in source:
        return
    if code not in operation.status_responses and description:
        operation.status_responses[code] = description


def _field_from_model(raw: Any, source: str, fallback: str) -> IRField | None:
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "").strip()
    quote = str(raw.get("quote") or "")
    if not name or quote not in source:
        return None
    name = align_field_name(name, quote)
    if name not in quote:
        return None
    location = str(raw.get("location") or fallback).lower()
    if location not in _LOCATIONS:
        location = fallback
    children = []
    for child in raw.get("children") or []:
        parsed = _field_from_model(child, source, "response" if location == "response" else location)
        if parsed is not None:
            children.append(parsed)
    max_length = raw.get("max_length")
    required = raw.get("required")
    return IRField(
        name=name,
        tz_type=str(raw["tz_type"]).strip() if raw.get("tz_type") else None,
        max_length=max_length if isinstance(max_length, int) else None,
        required=required if isinstance(required, bool) else None,
        description=str(raw.get("description") or ""),
        quote=quote,
        origin="model",
        location=location,
        children=children,
    )


def _grounded(example: Any, source: str) -> bool:
    def walk(value: Any) -> bool:
        if isinstance(value, dict):
            return all(str(key) in source and walk(item) for key, item in value.items())
        if isinstance(value, list):
            return all(walk(item) for item in value)
        if isinstance(value, str):
            return value in source
        if isinstance(value, bool) or value is None:
            return True
        if isinstance(value, (int, float)):
            return str(value) in source
        return False

    return walk(example)
