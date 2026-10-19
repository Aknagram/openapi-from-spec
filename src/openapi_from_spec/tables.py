from __future__ import annotations

import re

from openapi_from_spec.models import IRField

_TREE = re.compile(r"^(?:(?:└|├)\s*[─\-–]?\s*)+")


def split_columns(line: str) -> list[str]:
    raw = line.strip()
    if not raw:
        return []
    if "\t" in line:
        return [cell.strip() for cell in line.strip().split("\t")]
    if raw.startswith("|") and raw.count("|") >= 2:
        return [cell.strip() for cell in raw.strip("|").split("|")]
    return [cell.strip() for cell in re.split(r"\s{2,}", raw) if cell.strip() != ""]


def column_kind(cell: str) -> str | None:
    name = re.sub(r"\s+", " ", cell.strip().lower())
    if "тип" in name or name == "type":
        if ":" in name or len(name) > 30:
            return None
        return "type"
    if "длин" in name:
        return "len"
    if "обязат" in name:
        return "required"
    if "описан" in name:
        return "desc"
    if name.startswith("парам") or "поле" in name or "наимен" in name or name in {"name", "field"}:
        return "name"
    return None


def is_header(cells: list[str]) -> bool:
    kinds = [column_kind(cell) for cell in cells]
    named = [kind for kind in kinds if kind]
    return "type" in named and len(named) >= 2


def parse_required(cell: str) -> bool | None:
    value = cell.strip().lower()
    if value in {"", "-", "—", "–", "n/a"}:
        return None
    if value in {"да", "yes", "true", "y", "+", "м", "m", "обязательный", "обязательно"}:
        return True
    if value in {"нет", "no", "false", "n", "необязательный", "необязательно"}:
        return False
    if value == "о" and cell.strip() == "О":
        return True
    if value == "н" and cell.strip() == "Н":
        return False
    if cell.strip() == "O":
        return False
    return None


def parse_length(cell: str) -> int | None:
    digits = re.sub(r"[^\d]", "", cell or "")
    if not digits:
        return None
    return int(digits)


def parse_name(cell: str) -> tuple[int, str]:
    depth = 0
    rest = cell.strip()
    while True:
        match = _TREE.match(rest)
        if not match:
            break
        depth += max(1, rest[: match.end()].count("└") + rest[: match.end()].count("├"))
        rest = rest[match.end() :]
    rest = rest.strip()
    return depth, rest


def header_index(cells: list[str]) -> dict[str, int]:
    index: dict[str, int] = {}
    for position, cell in enumerate(cells):
        kind = column_kind(cell)
        if kind and kind not in index:
            index[kind] = position
    if "name" not in index:
        index["name"] = 0
    return index


def guess_index(width: int) -> dict[str, int]:
    if width >= 5:
        return {"name": 0, "type": 1, "len": 2, "required": 3, "desc": 4}
    if width == 4:
        return {"name": 0, "type": 1, "len": 2, "desc": 3}
    if width == 3:
        return {"name": 0, "type": 1, "desc": 2}
    return {"name": 0, "desc": 1 if width > 1 else 0}


def cell(cells: list[str], index: dict[str, int], kind: str) -> str:
    position = index.get(kind)
    if position is None or position >= len(cells):
        return ""
    return cells[position]


def nest_rows(rows: list[tuple[int, IRField]]) -> list[IRField]:
    roots: list[IRField] = []
    stack: list[tuple[int, IRField]] = []
    for depth, item in rows:
        while stack and stack[-1][0] >= depth:
            stack.pop()
        if depth > 0 and not stack:
            depth = 0
        if stack:
            stack[-1][1].children.append(item)
        else:
            roots.append(item)
        stack.append((depth, item))
    return roots


def rows_to_fields(lines: list[str], location: str) -> list[IRField]:
    index: dict[str, int] | None = None
    pending: list[tuple[int, IRField]] = []
    for line in lines:
        cells = split_columns(line)
        if len(cells) < 2:
            continue
        if is_header(cells):
            index = header_index(cells)
            continue
        if set(cells) <= {"---", ":---", "---:"}:
            continue
        used = index or guess_index(len(cells))
        depth, name = parse_name(cell(cells, used, "name"))
        if not name or name.lower() in {"параметр", "parameter", "название", "наименование"}:
            continue
        required = parse_required(cell(cells, used, "required")) if "required" in used else None
        pending.append(
            (
                depth,
                IRField(
                    name=name,
                    tz_type=cell(cells, used, "type") or None,
                    max_length=parse_length(cell(cells, used, "len")),
                    required=required,
                    description=cell(cells, used, "desc"),
                    quote=line.strip(),
                    origin="extracted",
                    location=location,
                ),
            )
        )
    return nest_rows(pending)
