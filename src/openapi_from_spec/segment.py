from __future__ import annotations

import re

from openapi_from_spec.models import Segment
from openapi_from_spec.naming import looks_like_url, parse_target
from openapi_from_spec.tables import is_header, split_columns

VERBS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS")
_VERB = "|".join(VERBS)
_METHOD_LINE = re.compile(rf"^({_VERB})\s+(\S+)", re.IGNORECASE)
_SHORT_LABEL = re.compile(
    rf"^(?:метод|method)\s*[:—\-]\s*({_VERB})\b(?:\s+(\S+))?",
    re.IGNORECASE,
)
_HTTP_LABEL = re.compile(
    rf"^(?:HTTP[-\s]?метод|тип\s+запроса)\s*[:—\-]\s*({_VERB})\b(?:\s+(\S+))?",
    re.IGNORECASE,
)
_INLINE = re.compile(
    rf"(?:^|\s)({_VERB})\s+([A-Za-z/{{][^\s,;)]*)",
    re.IGNORECASE,
)
_NUMBERED = re.compile(r"^(?:#{1,6}\s+)?\d+(?:\.\d+)*\.?\s+\S")
_MARKDOWN = re.compile(r"^#{1,6}\s+\S")

_KEYWORDS = (
    "описание",
    "параметры запроса",
    "параметры ответа",
    "пример ответа",
    "пример запроса",
    "тело запроса",
    "параметры пути",
    "заголовки",
)


_IDENT_PIECE = re.compile(r"[A-Za-z0-9_.\[\]]+")


def normalize(text: str) -> str:
    if text.startswith("\ufeff"):
        text = text[1:]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _repair_table_identifiers(text)


def _repair_table_identifiers(text: str) -> str:
    """Склеивает один пробел внутри идентификатора в ячейке markdown-таблицы."""
    lines = []
    for line in text.split("\n"):
        if line.count("|") < 2:
            lines.append(line)
            continue
        lines.append("|".join(_repair_identifier_cell(cell) for cell in line.split("|")))
    return "\n".join(lines)


def _repair_identifier_cell(cell: str) -> str:
    stripped = cell.strip()
    if stripped.count(" ") != 1:
        return cell
    left, right = stripped.split(" ")
    if not left or not right:
        return cell
    if _IDENT_PIECE.fullmatch(left) and _IDENT_PIECE.fullmatch(right):
        return cell.replace(stripped, left + right, 1)
    return cell


def _norm_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip().lower()).rstrip(":")


_TAIL_HEADING = re.compile(
    r"^(?:#{1,6}\s+)?(?:"
    r"для запроса|для тела ответа|для ответа|маппинг|"
    r"параметры запроса|параметры ответа|параметры пути|"
    r"пример ответа|пример запроса|query-parameters|"
    r"тело запроса|тело ответа"
    r")\b",
    re.IGNORECASE,
)
_METHOD_TITLE = re.compile(
    rf"^(?:метод|method)\s+({_VERB})\b(?:\s+(\S+))?",
    re.IGNORECASE,
)


def extend_segment_tail(text: str, end: int, window_end: int) -> int:
    """Таблица полей того же метода в этом окне входит в фрагмент, даже если цитата оборвалась раньше."""
    if end >= window_end:
        return end
    offset = end
    saw_tail = False
    for line in text[end:window_end].split("\n"):
        raw = line.strip()
        if raw and _is_new_method_line(raw):
            return offset if saw_tail else end
        if raw and (_TAIL_HEADING.match(raw) or _is_mapping_header(raw)):
            saw_tail = True
        offset += len(line) + 1
    return window_end if saw_tail else end


def _is_new_method_line(line: str) -> bool:
    stripped = re.sub(r"^#{1,6}\s+", "", line.strip())
    if _match_method(stripped):
        return True
    match = _METHOD_TITLE.match(stripped)
    if not match:
        return False
    token = match.group(2)
    return not (token and looks_like_url(token))


def _is_mapping_header(line: str) -> bool:
    if line.count("|") < 3:
        return False
    folded = line.lower()
    has_name = "параметр" in folded or "название" in folded
    return has_name and "тип" in folded


def is_keyword_line(line: str) -> bool:
    folded = _norm_line(line)
    if not folded or len(line.strip()) > 80:
        return False
    return any(folded == key or folded.startswith(key + " ") for key in _KEYWORDS)


def _match_method(line: str) -> tuple[str, str | None] | None:
    raw = line.strip()
    for pattern in (_METHOD_LINE, _HTTP_LABEL, _SHORT_LABEL):
        match = pattern.match(raw)
        if not match:
            continue
        token = match.group(2)
        if token and looks_like_url(token):
            return None
        return match.group(1).lower(), token
    return None


def _match_inline(line: str) -> tuple[str, str] | None:
    raw = line.strip()
    if _match_method(raw):
        return None
    match = _INLINE.search(raw)
    if not match:
        return None
    token = match.group(2)
    if looks_like_url(token):
        return None
    return match.group(1).lower(), token


class _Line:
    def __init__(self, text: str, start: int, end: int) -> None:
        self.text = text
        self.start = start
        self.end = end


def _lines(text: str) -> list[_Line]:
    rows: list[_Line] = []
    offset = 0
    for piece in text.split("\n"):
        rows.append(_Line(piece, offset, offset + len(piece)))
        offset += len(piece) + 1
    return rows


def _next_content(lines: list[_Line], index: int) -> int | None:
    cursor = index + 1
    while cursor < len(lines):
        if lines[cursor].text.strip():
            return cursor
        cursor += 1
    return None


def _is_numbered(line: str) -> bool:
    stripped = line.strip()
    return bool(_NUMBERED.match(stripped) or _MARKDOWN.match(stripped))


def _keyword_before_next_heading(lines: list[_Line], index: int) -> bool:
    cursor = index + 1
    while cursor < len(lines):
        if _is_numbered(lines[cursor].text) and cursor != index:
            return False
        if is_keyword_line(lines[cursor].text):
            return True
        cursor += 1
    return False


def _is_title(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 160:
        return False
    if is_keyword_line(stripped) or _match_method(stripped):
        return False
    if stripped.startswith(("└", "├", "|", "{", "}", "[", "]")):
        return False
    if "\t" in line or len(split_columns(stripped)) >= 3:
        return False
    if stripped.endswith((".", ";", "…")):
        return False
    return True


def _prev_content(lines: list[_Line], index: int) -> int | None:
    cursor = index - 1
    while cursor >= 0:
        if lines[cursor].text.strip():
            return cursor
        cursor -= 1
    return None


def _title_anchor(lines: list[_Line], index: int) -> bool:
    if not _is_title(lines[index].text):
        return False
    if _is_numbered(lines[index].text) and _keyword_before_next_heading(lines, index):
        return True
    nxt = _next_content(lines, index)
    if nxt is None or not is_keyword_line(lines[nxt].text):
        return False
    prev = _prev_content(lines, index)
    if prev is not None and _match_method(lines[prev].text):
        return False
    return True


class _Anchor:
    def __init__(self, line: int, kind: str, method: str | None, token: str | None) -> None:
        self.line = line
        self.kind = kind
        self.method = method
        self.token = token


def _catalog_anchors(lines: list[_Line]) -> list[_Anchor]:
    header_at: int | None = None
    method_col = path_col = -1
    anchors: list[_Anchor] = []
    for index, line in enumerate(lines):
        cells = split_columns(line.text)
        if len(cells) < 2:
            continue
        folded = [cell.lower() for cell in cells]
        if any("метод" in cell or cell == "method" for cell in folded) and any(
            word in cell for cell in folded for word in ("путь", "url", "uri", "path", "endpoint")
        ):
            header_at = index
            method_col = next(i for i, cell in enumerate(folded) if "метод" in cell or cell == "method")
            path_col = next(
                i
                for i, cell in enumerate(folded)
                if any(word in cell for word in ("путь", "url", "uri", "path", "endpoint"))
            )
            continue
        if header_at is None or is_header(cells):
            continue
        if index == header_at + 1 and set(cells) <= {"---"}:
            continue
        if method_col >= len(cells):
            continue
        method_match = re.fullmatch(rf"({_VERB})", cells[method_col].strip(), re.IGNORECASE)
        if not method_match:
            continue
        token = cells[path_col].strip() if path_col < len(cells) else None
        anchors.append(_Anchor(index, "catalog", method_match.group(1).lower(), token))
    return anchors


def find_segments(text: str) -> list[Segment]:
    text = normalize(text)
    lines = _lines(text)
    line_methods = []
    inline_methods = []
    for index, line in enumerate(lines):
        matched = _match_method(line.text)
        if matched:
            line_methods.append(_Anchor(index, "method", matched[0], matched[1]))
            continue
        inline = _match_inline(line.text)
        if inline:
            inline_methods.append(_Anchor(index, "inline", inline[0], inline[1]))

    method_anchors = line_methods or inline_methods
    titles = [
        _Anchor(index, "title", None, None)
        for index, _line in enumerate(lines)
        if _title_anchor(lines, index)
    ]
    anchors = method_anchors + titles + _catalog_anchors(lines)
    anchors.sort(key=lambda item: (item.line, 0 if item.kind != "title" else 1))
    chosen: list[_Anchor] = []
    seen: set[int] = set()
    for anchor in anchors:
        if anchor.line in seen:
            continue
        seen.add(anchor.line)
        chosen.append(anchor)
    if not chosen:
        return []

    starts = []
    for position, anchor in enumerate(chosen):
        previous = chosen[position - 1].line if position else -1
        line_no = anchor.line
        if anchor.kind == "inline":
            while line_no - 1 > previous and lines[line_no - 1].text.strip():
                line_no -= 1
        starts.append(line_no)

    segments: list[Segment] = []
    for position, anchor in enumerate(chosen):
        start_line = starts[position]
        end_line = starts[position + 1] if position + 1 < len(starts) else len(lines)
        if end_line <= start_line:
            continue
        start = lines[start_line].start
        end = lines[end_line - 1].end
        chunk = text[start:end]
        raw_name, explicit_path, query = parse_target(anchor.token)
        title = lines[anchor.line].text.strip()
        if anchor.kind == "title":
            title = re.sub(r"^#{1,6}\s+", "", title)
            title = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", title)
        segments.append(
            Segment(
                start=start,
                end=end,
                text=chunk,
                http_method=anchor.method,
                raw_name=raw_name,
                explicit_path=explicit_path,
                query=query,
                title=title if anchor.kind == "title" else "",
                anchor=anchor.kind,
            )
        )
    return _prefer_longer(segments)


def _prefer_longer(segments: list[Segment]) -> list[Segment]:
    best: dict[tuple[str, str], Segment] = {}
    order: list[tuple[str, str]] = []
    untitled: list[Segment] = []
    for segment in segments:
        if not segment.http_method or not segment.raw_name:
            untitled.append(segment)
            continue
        key = (segment.http_method, segment.raw_name.lower())
        current = best.get(key)
        if current is None:
            best[key] = segment
            order.append(key)
        elif len(segment.text) > len(current.text):
            best[key] = segment
    ordered = [best[key] for key in order]
    return sorted(untitled + ordered, key=lambda item: item.start)


def pack_windows(text: str, limit: int, overlap: int = 400) -> list[tuple[int, int]]:
    """Режет текст окнами по границам строк. Конец каждого следующего окна дальше предыдущего."""
    if limit < 200:
        raise ValueError("окно модели должно быть не короче 200 символов")
    text = normalize(text)
    if not text:
        return []
    if len(text) <= limit:
        return [(0, len(text))]
    lines = _lines(text)
    windows: list[tuple[int, int]] = []
    start_i = 0
    guard = 0
    while start_i < len(lines) and guard < len(lines) + 5:
        guard += 1
        header_i: int | None = None
        end_i = start_i
        while end_i < len(lines):
            if is_header(split_columns(lines[end_i].text)):
                header_i = end_i
            if end_i > start_i and lines[end_i].end - lines[start_i].start > limit:
                break
            end_i += 1
        if end_i == start_i:
            end_i = start_i + 1
        last = end_i - 1
        windows.append((lines[start_i].start, lines[last].end))
        if end_i >= len(lines):
            break
        next_i = end_i
        if header_i is not None and start_i < header_i < end_i:
            next_i = header_i
        elif overlap > 0:
            back = last
            cut = lines[last].end
            while back > start_i and cut - lines[back].start < overlap:
                back -= 1
            next_i = max(start_i + 1, back)
        if next_i <= start_i:
            next_i = min(end_i, start_i + 1)
        start_i = next_i
    return _split_long_slices(windows, limit)


def _split_long_slices(windows: list[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
    pieces: list[tuple[int, int]] = []
    for start, end in windows:
        cursor = start
        while cursor < end:
            pieces.append((cursor, min(end, cursor + limit)))
            cursor += limit
    return pieces


def render_window(text: str, start: int, end: int) -> str:
    """Добавляет шапку таблицы, если окно начинается с середины её строк."""
    chunk = text[start:end]
    if any(is_header(split_columns(line)) for line in chunk.splitlines()):
        return chunk
    prefix = _table_header_before(text, start)
    if not prefix:
        return chunk
    return prefix + "\n" + chunk


def _table_header_before(text: str, start: int) -> str | None:
    earlier = text[:start]
    lines = earlier.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    header: str | None = None
    for line in lines:
        if not line.strip():
            header = None
            continue
        if is_header(split_columns(line)):
            header = line
    return header
