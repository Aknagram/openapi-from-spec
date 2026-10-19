"""Прогон десяти длинных ТЗ. При обрыве связи останавливается и пишет, на каком документе."""

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
STOP = OUT / "wide-STOP.txt"
SCORE = OUT / "wide-score.txt"

ORDER = [f"wide-{number:02d}" for number in range(1, 11)]


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


def param_names(operation: dict, location: str) -> list[str]:
    names = []
    for item in operation.get("parameters") or []:
        if isinstance(item, dict) and item.get("in") == location:
            names.append(str(item.get("name")))
    return names


def codes(operation: dict) -> set[str]:
    return {str(code) for code in (operation.get("responses") or {})}


def find(ops, method: str, suffix: str):
    for verb, path, operation in ops:
        if verb == method and path.rstrip("/").endswith(suffix.rstrip("/")):
            return path, operation
    return None, None


def check(name: str, spec: dict, report: dict, yaml_text: str) -> list[str]:
    problems: list[str] = []
    questions = [item.get("question", "") for item in report.get("gaps") or []]
    if any("Модель недоступна" in question or "только модель" in question for question in questions):
        return ["модель не разобрала документ: " + " | ".join(questions[:3])]
    info = spec.get("info") or {}
    ops = operations(spec)
    long_note = any("Длинный метод" in item for item in report.get("conventions") or [])
    if not long_note:
        problems.append("документ не пошёл по пути длинного метода")
    servers = [item.get("url") for item in spec.get("servers") or [] if isinstance(item, dict)]

    if name.startswith("wide-01"):
        problems += _expect(info, "Портовый реестр", "4.2", ops, [("post", "/harbor/v1/calls")], yaml_text, ["berthSeal", "Причал занят"])
        problems += _servers(servers, "{apiUrl}")
    elif name.startswith("wide-02"):
        problems += _expect(
            info, "Лабораторные пробы", "2.1", ops,
            [("post", "/lab/v1/samples"), ("get", "/lab/v1/samples/{sampleId}")],
            yaml_text, ["sampleBarcode", "resultStamp"],
        )
        if len(ops) != 2:
            problems.append(f"ожидались 2 операции, есть {[(m, p) for m, p, _ in ops]}")
    elif name.startswith("wide-03"):
        problems += _expect(info, "Договоры аренды", "6", ops, [("get", "/lease/v1/contracts/{contractId}")], yaml_text, ["handoverSeal"])
        _path, operation = find(ops, "get", "/contracts/{contractId}")
        names = param_names(operation or {}, "path")
        if "agreementId" in names:
            problems.append(f"лишний path-параметр agreementId: {names}")
        if "contractId" not in names:
            problems.append(f"нет path-параметра contractId: {names}")
    elif name.startswith("wide-04"):
        problems += _expect(info, "Торги", "3", ops, [("post", "/auction/v2/lots/{lotId}/close")], yaml_text, ["gavelSeal"])
        if any("/v1/" in path for _m, path, _op in ops):
            problems.append(f"пример v1 остался вторым путём: {[(m, p) for m, p, _ in ops]}")
    elif name.startswith("wide-05"):
        problems += _expect(
            info, "Аптечный склад", "1.4", ops,
            [("put", "/pharmacy/v1/stores/{storeId}/shelves"), ("get", "/pharmacy/v1/stores/{storeId}/batch-line")],
            yaml_text, ["shelfSeal", "batchSeal"],
        )
        if any("shel-ves" in path or path.endswith("/shelves") is False and "shel" in path for _m, path, _op in ops):
            problems.append(f"полка не склеилась: {[(m, p) for m, p, _ in ops]}")
    elif name.startswith("wide-06"):
        problems += _expect(info, "Судебные дела", "8", ops, [("post", "/court/v1/filings")], yaml_text, ["filingSeal"])
        extra = [(m, p) for m, p, _ in ops if not (m == "post" and p.endswith("/filings"))]
        if extra:
            problems.append(f"глаголы без адреса стали вызовами: {extra}")
    elif name.startswith("wide-07"):
        problems += _expect(info, "Музейный фонд", "2", ops, [("get", "/museum/v1/exhibits/{exhibitId}")], yaml_text, ["exhibitSeal"])
        if any("museum.example.test" in str(item) for item in servers):
            problems.append(f"пример стал вторым сервером: {servers}")
        if any("edd5b4df" in path for _m, path, _op in ops):
            problems.append("образец UUID остался отрезком пути")
    elif name.startswith("wide-08"):
        problems += _expect(info, "Учёт электроэнергии", "5.0", ops, [("get", "/grid/v1/meters/{meterId}/readings")], yaml_text, ["readingSeal", "meterPoint", "X-Trace"])
        _path, operation = find(ops, "get", "/readings")
        if "meterPoint" not in param_names(operation or {}, "query"):
            problems.append(f"нет query meterPoint: {param_names(operation or {}, 'query')}")
        if "X-Trace" not in param_names(operation or {}, "header"):
            problems.append(f"нет заголовка X-Trace: {param_names(operation or {}, 'header')}")
        if "array" not in yaml_text:
            problems.append("readings не описан как массив")
    elif name.startswith("wide-09"):
        problems += _expect(info, "Выпуски газеты", "9", ops, [("get", "/press/v1/issues/{issueId}")], yaml_text, ["issueSeal"])
        if any(path.endswith(".json") for _m, path, _op in ops):
            problems.append(f"суффикс .json остался путём: {[(m, p) for m, p, _ in ops]}")
    elif name.startswith("wide-10"):
        problems += _expect(info, "Протоколы осмотра", "7", ops, [("post", "/clinic/v2/visits")], yaml_text, ["protocolStamp", "policyNumber"])
        _path, operation = find(ops, "post", "/visits")
        present = codes(operation or {})
        if "422" not in present:
            problems.append(f"нет 422: {sorted(present)}")
        if "400" not in present:
            problems.append(f"нет 400: {sorted(present)}")
        if "500" in present:
            problems.append("добавлен код 500, которого в методе нет")
    return problems


def _expect(info, title, version, ops, expected, yaml_text, needles) -> list[str]:
    problems = []
    if title not in str(info.get("title")):
        problems.append(f"заголовок {info.get('title')!r}")
    if str(info.get("version")) != version:
        problems.append(f"версия {info.get('version')!r}, ожидалась {version}")
    for method, suffix in expected:
        if not any(verb == method and path.rstrip("/").endswith(suffix.rstrip("/")) for verb, path, _op in ops):
            problems.append(f"нет {method.upper()} {suffix}; есть {[(m, p) for m, p, _ in ops]}")
    for needle in needles:
        if needle not in yaml_text:
            problems.append(f"в YAML нет «{needle}»")
    return problems


def _servers(servers: list, token: str) -> list[str]:
    if servers != [token]:
        return [f"серверы {servers}, ожидался [{token}]"]
    return []


def model_down(problems: list[str], report: dict) -> bool:
    text = " ".join(problems)
    questions = " ".join(item.get("question", "") for item in report.get("gaps") or [])
    blob = (text + " " + questions).lower()
    marks = ("модель недоступна", "10061", "10054", "connection", "timed out", "timeout", "разорвал")
    return any(mark in blob for mark in marks)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if STOP.exists():
        STOP.unlink()
    settings = Settings(
        vllm_base_url="http://127.0.0.1:11434/api",
        vllm_model="qwen3.8:27b",
        timeout_seconds=300,
    )
    paths = []
    for prefix in ORDER:
        found = sorted(DOCS.glob(prefix + "-*.md"))
        if not found:
            print("НЕТ ФАЙЛА", prefix, flush=True)
            return 2
        paths.append(found[0])
    lines: list[str] = []
    for index, path in enumerate(paths, start=1):
        print(f"START {index}/10 {path.name}", flush=True)
        text = path.read_text(encoding="utf-8")
        try:
            result = convert_text(text, settings)
        except Exception as exc:
            lines.append(f"{path.name}: ИСКЛЮЧЕНИЕ {type(exc).__name__}: {exc}")
            SCORE.write_text("\n".join(lines) + "\n", encoding="utf-8")
            STOP.write_text(
                f"Обрыв на документе {index}/10 {path.name} до записи YAML.\n{type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
            print("STOP", path.name, flush=True)
            traceback.print_exc()
            return 2
        (OUT / f"{path.stem}.yaml").write_text(result.yaml_text, encoding="utf-8")
        (OUT / f"{path.stem}.report.json").write_text(
            json.dumps(result.report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        problems = check(path.name, result.spec, result.report, result.yaml_text)
        if result.errors:
            problems.append("ошибки схемы: " + "; ".join(result.errors[:6]))
        calls = result.report.get("model_calls")
        ops = operations(result.spec)
        summary = f"{path.name}: вызовов {calls}, операций {[(m, p) for m, p, _ in ops]}, версия {result.spec.get('info', {}).get('version')}"
        if model_down(problems, result.report):
            lines.append(summary)
            lines.extend(f"  - {item}" for item in problems)
            SCORE.write_text("\n".join(lines) + "\n", encoding="utf-8")
            nxt = paths[index].name if index < len(paths) else "все документы обработаны"
            STOP.write_text(
                f"Модель недоступна на документе {index}/10 {path.name}.\n"
                f"Следующий не начат: {nxt}.\n",
                encoding="utf-8",
            )
            print("STOP", path.name, flush=True)
            return 2
        if problems:
            lines.append(summary + f", проблем {len(problems)}")
            lines.extend(f"  - {item}" for item in problems)
        else:
            lines.append(summary + ", ок")
        SCORE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(summary, flush=True)
    print("ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
