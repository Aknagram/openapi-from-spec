from __future__ import annotations

import os
from pathlib import Path

_TASKS = {
    "segment": (
        "Выполни только задачу segment из скилла. "
        "Игнорируй задачи extract и openapi. Ответ — один JSON-объект без Markdown."
    ),
    "extract": (
        "Выполни только задачу extract из скилла. "
        "Игнорируй задачи segment и openapi. Ответ — один JSON-объект без Markdown."
    ),
    "openapi": (
        "Выполни только задачу openapi из скилла. "
        "Игнорируй задачи segment и extract. Ответ — YAML OpenAPI 3.0.3 без пояснений вокруг."
    ),
}


def skill_dir() -> Path:
    override = os.environ.get("OPENAPI_SKILL_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / "skill"


def load_skill_markdown() -> str:
    path = skill_dir() / "SKILL.md"
    return path.read_text(encoding="utf-8")


def strip_frontmatter(text: str) -> str:
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    if end == -1:
        return text
    return text[end + 4 :].lstrip("\n")


def system_prompt(task: str) -> str:
    if task not in _TASKS:
        raise ValueError(f"неизвестная задача скилла: {task}")
    body = strip_frontmatter(load_skill_markdown()).rstrip()
    if task == "openapi":
        rules = (skill_dir() / "knowledge" / "openapi-3.0.3.md").read_text(encoding="utf-8").strip()
        body = f"{body}\n\n{rules}"
    return body + "\n\n## Текущая задача\n\n" + _TASKS[task] + "\n"
