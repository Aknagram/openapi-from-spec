from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Gap:
    question: str
    field: str = ""
    quote: str = ""


@dataclass
class IRField:
    name: str
    tz_type: str | None = None
    max_length: int | None = None
    required: bool | None = None
    description: str = ""
    quote: str = ""
    origin: str = "extracted"
    location: str = "query"
    children: list[IRField] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)


@dataclass
class Segment:
    start: int
    end: int
    text: str
    http_method: str | None = None
    raw_name: str = ""
    explicit_path: str | None = None
    query: dict[str, list[str]] = field(default_factory=dict)
    title: str = ""
    anchor: str = "method"


@dataclass
class Operation:
    http_method: str | None
    raw_name: str
    summary: str
    description: str
    fields: list[IRField]
    response_example: Any = None
    request_example: Any = None
    span: tuple[int, int] = (0, 0)
    source_head: str = ""
    gaps: list[Gap] = field(default_factory=list)
    explicit_path: str | None = None
    server_url: str | None = None
    status_responses: dict[str, str] = field(default_factory=dict)
    name_origin: str = "extracted"
    repeated_query: set[str] = field(default_factory=set)
    emit_path: str | None = None
    emit_operation_id: str | None = None


@dataclass
class ConvertResult:
    yaml_text: str
    spec: dict[str, Any]
    report: dict[str, Any]
    errors: list[str]
