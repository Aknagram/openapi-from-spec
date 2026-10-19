from __future__ import annotations

import re
from urllib.parse import urlsplit

_TRANSLIT = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "i",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "sch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}


def transliterate(text: str) -> str:
    parts: list[str] = []
    for char in text:
        low = char.lower()
        if low in _TRANSLIT:
            mapped = _TRANSLIT[low]
            parts.append(mapped.upper() if char.isupper() and mapped else mapped)
        else:
            parts.append(char)
    return "".join(parts)


def kebab(text: str) -> str:
    value = transliterate(text.strip())
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", value)
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1-\2", value)
    value = value.replace("_", "-").replace(" ", "-")
    value = re.sub(r"[^A-Za-z0-9-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-").lower()
    return value or "operation"


def pascal(text: str) -> str:
    body = kebab(text)
    return "".join(part.capitalize() for part in body.split("-") if part) or "Operation"


def singular(name: str) -> str:
    if len(name) > 3 and name.endswith("ies"):
        return name[:-3] + "y"
    if len(name) > 3 and name.endswith("s") and not name.endswith("ss"):
        return name[:-1]
    return name


def parse_target(token: str | None) -> tuple[str, str | None, dict[str, list[str]]]:
    if not token:
        return "", None, {}
    cleaned = token.strip().strip("`'\"").rstrip(".,;")
    path, _, query = cleaned.partition("?")
    path = path.rstrip(".,;")
    pairs: dict[str, list[str]] = {}
    if query:
        from urllib.parse import parse_qs

        for key, values in parse_qs(query, keep_blank_values=True).items():
            pairs[key] = values
    if path.startswith("/"):
        pieces = [piece for piece in path.split("/") if piece]
        last = pieces[-1] if pieces else "operation"
        if last.startswith("{") and last.endswith("}"):
            last = last[1:-1]
        return last, path, pairs
    return path, None, pairs


def looks_like_url(token: str) -> bool:
    return "://" in token or token.lower().startswith("http")


_VERBS = "GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS"
_TEMPLATE_URL = re.compile(
    rf"(?:-X\s+)?\b({_VERBS})\s+['\"]?\s*<?(https?://[^\s)'\">]+)",
    re.IGNORECASE,
)
_RELATIVE_REQUEST = re.compile(
    rf"\b({_VERBS})\s+(/[^\s?#]+)",
    re.IGNORECASE,
)
_ABSOLUTE_URL = re.compile(r"https?://[^\s)'\">]+", re.IGNORECASE)


def find_template_url(text: str) -> tuple[str, str, str] | None:
    """Глагол, сервер и путь из полного адреса. Адрес с {параметром} важнее примера того же метода."""
    found: list[tuple[str, str, str, bool]] = []
    for match in _TEMPLATE_URL.finditer(text):
        url = match.group(2).rstrip(".,;")
        parsed = urlsplit(url)
        if not parsed.netloc or not parsed.path:
            continue
        found.append(
            (
                match.group(1).lower(),
                f"{parsed.scheme}://{parsed.netloc}",
                parsed.path,
                "{" in url,
            )
        )
    if not found:
        return None
    owner = found[0][0]
    same_method = [item for item in found if item[0] == owner]
    method, server, path, _templated = next((item for item in same_method if item[3]), same_method[0])
    return method, server, path


def example_targets(text: str) -> list[tuple[str | None, str | None, str]]:
    """Глагол или None, origin или None и путь из строк запроса и абсолютных адресов."""
    found: list[tuple[str | None, str | None, str]] = []
    for match in _TEMPLATE_URL.finditer(text):
        target = _split_http_url(match.group(2))
        if target is not None:
            origin, path = target
            found.append((match.group(1).lower(), origin, path))
    for match in _RELATIVE_REQUEST.finditer(text):
        path = _clean_path(match.group(2))
        if path and not path.startswith("//"):
            found.append((match.group(1).lower(), None, path))
    for match in _ABSOLUTE_URL.finditer(text):
        target = _split_http_url(match.group(0))
        if target is not None:
            origin, path = target
            found.append((None, origin, path))
    return found


def _split_http_url(url: str) -> tuple[str, str] | None:
    parsed = urlsplit(url.rstrip(".,;"))
    if not parsed.netloc or not parsed.path or parsed.path == "/":
        return None
    return f"{parsed.scheme}://{parsed.netloc}", _clean_path(parsed.path)


def declared_placeholder_routes(text: str) -> list[tuple[str, str, str, str]]:
    """Глагол, плейсхолдер, путь и имя метода из таблицы или строки `{apiUrl}/...`."""
    found: list[tuple[str, str, str, str]] = []
    verbs = {item.lower() for item in _VERBS.split("|")}
    for line in text.splitlines():
        if "|" not in line or "{" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        verb = next((cell.lower() for cell in cells if cell.lower() in verbs), "")
        name = next((cell for cell in cells if re.search(r"[A-Za-z]{3,}", cell) and cell.lower() not in verbs), "")
        for cell in cells:
            route = _placeholder_route(_squash_url(cell))
            if route is None:
                continue
            placeholder, path = route
            found.append((verb, placeholder, path, _squash_name(name)))
    for match in re.finditer(
        rf"\b({_VERBS})\s+(\{{[A-Za-z][A-Za-z0-9]*\}}/?\S+)",
        text,
        re.IGNORECASE,
    ):
        route = _placeholder_route(_squash_url(match.group(2)))
        if route is None:
            continue
        placeholder, path = route
        found.append((match.group(1).lower(), placeholder, path, ""))
    return found


def _squash_url(cell: str) -> str:
    compact = re.sub(r"\s+", "", cell)

    def braces(match: re.Match[str]) -> str:
        return "{" + match.group(1).replace("-", "") + "}"

    compact = re.sub(r"\{([^{}]+)\}", braces, compact)
    return re.sub(r":([A-Za-z][A-Za-z0-9]*)", r"{\1}", compact)


def _squash_name(name: str) -> str:
    compact = re.sub(r"\s+", "", name)
    compact = re.sub(r"-([A-Za-z])", r"\1", compact)
    return compact if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", compact) else ""


def _placeholder_route(cell: str) -> tuple[str, str] | None:
    match = re.search(r"(\{[A-Za-z][A-Za-z0-9]*\})(/?[A-Za-z0-9{}_./~-]+)", cell)
    if match is None:
        return None
    path = match.group(2)
    if not path.startswith("/"):
        path = "/" + path
    path = _clean_path(path.split("?")[0])
    if path == "/":
        return None
    return match.group(1), path


def _clean_path(path: str) -> str:
    path = path.split("?")[0].rstrip(".,;")
    if path.endswith(".json"):
        path = path[: -len(".json")]
    if not path.startswith("/"):
        path = "/" + path
    return path.rstrip("/") or "/"
