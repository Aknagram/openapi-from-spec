"""Прогон трёх длинных ТЗ. Методы длиннее окна, хвост поля должен доехать до YAML."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openapi_from_spec.pipeline import Settings, convert_text  # noqa: E402

DOCS = ROOT / "probe" / "docs"
OUT = ROOT / "out" / "probe"

EXPECT = {
    "long-01-insurance.md": {
        "title": "Страховые заявления",
        "version": "3",
        "ops": [("post", "/claims"), ("get", "/claims/")],
        "needles": ["policyNumber", "settlementStamp", "payoutLedger", "Полис не действует", "Заявление не найдено"],
    },
    "long-02-freight.md": {
        "title": "Грузовые перевозки",
        "version": "2",
        "ops": [("post", "/shipments"), ("patch", "/shipments/")],
        "needles": ["cargoName", "dispatchStamp", "routeRevision", "Накладная уже есть"],
    },
    "long-03-personnel.md": {
        "title": "Кадровый архив",
        "version": "5",
        "ops": [("post", "/dossiers"), ("get", "/dossiers/")],
        "needles": ["employeeNumber", "dossierSeal", "archiveBox", "Табельный номер уже занят"],
    },
}


def operations(spec: dict) -> list[tuple[str, str]]:
    found = []
    for path, methods in (spec.get("paths") or {}).items():
        if not isinstance(methods, dict):
            continue
        for method, operation in methods.items():
            if str(method).startswith("x-") or not isinstance(operation, dict):
                continue
            found.append((str(method).lower(), str(path)))
    return found


def check(name: str, spec: dict, report: dict, yaml_text: str) -> list[str]:
    problems: list[str] = []
    questions = [item.get("question", "") for item in report.get("gaps") or []]
    if any("Модель недоступна" in question or "только модель" in question for question in questions):
        return ["модель не разобрала документ: " + " | ".join(questions)]
    expected = EXPECT[name]
    if expected["title"] not in str(spec.get("info", {}).get("title")):
        problems.append(f"заголовок {spec.get('info', {}).get('title')!r}")
    if str(spec.get("info", {}).get("version")) != expected["version"]:
        problems.append(f"версия {spec.get('info', {}).get('version')!r}")
    ops = operations(spec)
    for method, suffix in expected["ops"]:
        if not any(verb == method and suffix in path for verb, path in ops):
            problems.append(f"нет {method.upper()} {suffix}; есть {ops}")
    for needle in expected["needles"]:
        if needle not in yaml_text:
            problems.append(f"в YAML нет «{needle}»")
    long_note = any("Длинный метод" in item for item in report.get("conventions") or [])
    if not long_note:
        problems.append("документ не пошёл по пути длинного метода")
    if len(ops) > len(expected["ops"]):
        problems.append(f"лишние операции: {ops}")
    return problems


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        vllm_base_url="http://127.0.0.1:11434/api",
        vllm_model="qwen3.8:27b",
        timeout_seconds=240,
    )
    lines: list[str] = []
    failed = 0
    for path in sorted(DOCS.glob("long-*.md")):
        text = path.read_text(encoding="utf-8")
        try:
            result = convert_text(text, settings)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            lines.append(f"{path.name}: ИСКЛЮЧЕНИЕ {type(exc).__name__}: {exc}")
            lines.append(traceback.format_exc())
            break
        (OUT / f"{path.stem}.yaml").write_text(result.yaml_text, encoding="utf-8")
        (OUT / f"{path.stem}.report.json").write_text(
            json.dumps(result.report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        problems = check(path.name, result.spec, result.report, result.yaml_text)
        if result.errors:
            problems.append("ошибки схемы: " + "; ".join(result.errors[:4]))
        calls = result.report.get("model_calls")
        if any("Модель недоступна" in item or "только модель" in item for item in problems):
            lines.append(f"{path.name}: модель недоступна, вызовов {calls}")
            lines.extend(f"  - {item}" for item in problems)
            failed += 1
            break
        if problems:
            failed += 1
            lines.append(f"{path.name}: {len(problems)} проблем, вызовов {calls}")
            lines.extend(f"  - {item}" for item in problems)
        else:
            lines.append(f"{path.name}: ок, вызовов {calls}, операций {len(operations(result.spec))}")
        (OUT / "long-score.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(lines[-1], flush=True)
    (OUT / "long-score.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
