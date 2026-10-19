from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openapi_from_spec.api import serve
from openapi_from_spec.pipeline import Settings, convert_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Собрать OpenAPI 3.0.3 YAML из текстового ТЗ REST API."
    )
    parser.add_argument("document", nargs="?", help="Файл ТЗ в UTF-8. Без файла читается stdin.")
    parser.add_argument("-o", "--output", help="Куда записать YAML. По умолчанию stdout.")
    parser.add_argument("--report", help="Куда записать JSON-отчёт о пробелах и соглашениях.")
    parser.add_argument("--title")
    parser.add_argument("--api-version")
    parser.add_argument("--path-prefix")
    parser.add_argument("--server-url")
    parser.add_argument(
        "--vllm-base-url",
        help="База /v1 совместимого сервера. Значение agent ждёт ответ в out/agent-bridge.",
    )
    parser.add_argument("--vllm-model")
    parser.add_argument("--dify-base-url", help="База Dify. Клиент дописывает /v1/chat-messages.")
    parser.add_argument("--dify-api-key", help="API-ключ приложения Dify с моделью.")
    parser.add_argument("--no-error-templates", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Ненулевой код, если остались вопросы по ТЗ.")
    parser.add_argument("--serve", action="store_true", help="Локальная страница сборки.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args(argv)
    settings = _settings(args)
    if args.serve:
        print(f"http://{args.host}:{args.port}", file=sys.stderr)
        serve(settings, args.host, args.port)
        return 0
    if args.document:
        text = Path(args.document).read_text(encoding="utf-8")
    elif sys.stdin.isatty():
        parser.print_help()
        return 1
    else:
        text = sys.stdin.read()
    result = convert_text(text, settings)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(result.yaml_text, encoding="utf-8")
    else:
        sys.stdout.write(result.yaml_text)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(result.report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(
        f"Операций: {len(result.report['operations'])}. "
        f"Без метода: {len(result.report['unresolved'])}. "
        f"Вопросов: {len(result.report['gaps'])}. "
        f"Вызовов модели: {result.report['model_calls']}.",
        file=sys.stderr,
    )
    if result.errors:
        for error in result.errors:
            print(error, file=sys.stderr)
        return 2
    if args.strict and result.report["gaps"]:
        return 3
    return 0


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    if args.title:
        settings.title = args.title
    if args.api_version:
        settings.version = args.api_version
    if args.path_prefix:
        settings.path_prefix = args.path_prefix
    if args.server_url:
        settings.server_url = args.server_url
    if args.vllm_base_url:
        settings.vllm_base_url = args.vllm_base_url
    if args.vllm_model:
        settings.vllm_model = args.vllm_model
    if args.dify_base_url:
        settings.dify_base_url = args.dify_base_url
    if args.dify_api_key:
        settings.dify_api_key = args.dify_api_key
    if args.no_error_templates:
        settings.error_templates = False
    return settings


if __name__ == "__main__":
    raise SystemExit(main())
