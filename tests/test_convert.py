from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

from openapi_from_spec.api import make_server
from openapi_from_spec.llm import DifyClient, _chat_content, dify_v1_base, ollama_chat_url
from openapi_from_spec.compile import compile_operations, dump_yaml
from openapi_from_spec.pipeline import Settings, _client_from_settings, _finish_operation, _report, convert_text
from openapi_from_spec.validate import validate_document
from openapi_from_spec.segment import normalize, pack_windows, render_window
from openapi_from_spec.skill_loader import load_skill_markdown, skill_dir, strip_frontmatter, system_prompt
from openapi_from_spec.stitch import stitch_documents

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = (ROOT / "examples" / "account-balances.txt").read_text(encoding="utf-8")


def compile_rules(text: str, settings: Settings | None = None):
    """Прямой разбор известных таблиц. Сервис так документ больше не режет."""
    from openapi_from_spec.models import ConvertResult
    from openapi_from_spec.segment import find_segments

    settings = settings or Settings()
    text = normalize(text)
    segments = find_segments(text)
    operations = [_finish_operation(segment, None, settings) for segment in segments]
    spec, conventions = compile_operations(
        operations,
        title=settings.title,
        version=settings.version,
        path_prefix=settings.path_prefix,
        error_templates=settings.error_templates,
        server_url=settings.server_url,
    )
    errors = validate_document(spec, operations)
    report = _report(text, segments, operations, [], conventions, settings, 0)
    return ConvertResult(yaml_text=dump_yaml(spec), spec=spec, report=report, errors=errors)


class Spy:
    def __init__(self, payload=None, handler=None):
        self.payload = payload or {}
        self.handler = handler
        self.calls = []

    def complete_json(self, *, task, system, user):
        self.calls.append({"task": task, "system": system, "user": user})
        if self.handler:
            return self.handler(task, user)
        return self.payload


class ConvertTests(unittest.TestCase):
    def test_account_balances_compiles_without_model(self) -> None:
        result = compile_rules(SAMPLE, Settings(title="Счета"))
        self.assertEqual(result.errors, [])
        self.assertEqual(result.report["model_calls"], 0)
        spec = yaml.safe_load(result.yaml_text)
        self.assertEqual(spec["openapi"], "3.0.3")
        self.assertEqual(spec["info"]["title"], "Счета")
        operation = spec["paths"]["/api/v1/account-balances"]["get"]
        self.assertEqual(operation["operationId"], "getAccountBalances")
        self.assertIn("текущем остатке", operation["description"])
        parameter = operation["parameters"][0]
        self.assertEqual(parameter["name"], "accounts")
        self.assertTrue(parameter["required"])
        self.assertEqual(parameter["schema"]["type"], "array")
        self.assertEqual(parameter["schema"]["items"], {"type": "string", "maxLength": 64})
        self.assertEqual(parameter["example"], ["acc1", "acc2"])
        schemas = spec["components"]["schemas"]
        balance = schemas["Balance"]["properties"]
        self.assertEqual(balance["account"]["maxLength"], 64)
        self.assertEqual(balance["balanceAmount"]["format"], "int64")
        self.assertEqual(balance["balanceAmount"]["x-tz-max-length"], 18)
        response = schemas["AccountBalancesResponse"]["properties"]
        self.assertIn("$ref", response["balances"]["items"])
        self.assertEqual(response["totalBalance"]["format"], "int64")
        example = operation["responses"]["200"]["content"]["application/json"]["example"]
        self.assertEqual(example["totalBalance"], 4700000000)
        self.assertEqual(operation["responses"]["400"]["x-origin"], "skill-template")
        self.assertNotIn("401", json.dumps(result.report["gaps"]))
        self.assertNotIn("currency", result.yaml_text)

    def test_second_method_without_verb_stays_unresolved(self) -> None:
        extra = """

Создание перевода

Описание
Создаёт перевод между счетами.

Параметры запроса

Параметр\tТип данных\tМаксимальная длина\tОбязательность\tОписание
amount\tLong\t12\tДа\tСумма
"""
        result = compile_rules(SAMPLE + extra, Settings())
        self.assertEqual(result.errors, [])
        self.assertIn("/api/v1/account-balances", result.spec["paths"])
        self.assertEqual(list(result.spec["paths"]), ["/api/v1/account-balances"])
        self.assertEqual(len(result.report["unresolved"]), 1)
        questions = " ".join(gap["question"] for gap in result.report["gaps"])
        self.assertIn("нет HTTP-метода", questions)

    def test_inline_methods_split_without_headings(self) -> None:
        text = (
            "Получение баланса\n"
            "Клиент вызывает GET balance?id=1 и получает amount.\n\n"
            "Создание платежа\n"
            "Клиент вызывает POST payment и передаёт amount.\n"
        )
        result = compile_rules(text, Settings())
        methods = sorted(path + " " + method for path, item in result.spec["paths"].items() for method in item)
        self.assertEqual(methods, ["/api/v1/balance get", "/api/v1/payment post"])

    def test_model_quote_must_exist(self) -> None:
        text = "GET limits\n\nПараметры запроса\n- id, тип String, длина 8, обязательный\n"
        yaml_text = """
openapi: 3.0.3
info: {title: Limits, version: "1.0.0"}
paths:
  /limits:
    get:
      operationId: getLimits
      responses:
        "200":
          description: ok
"""

        def handler(task, user):
            if task == "segment":
                return {
                    "segments": [
                        {
                            "start_quote": "этой цитаты нет",
                            "end_quote": "и этой тоже нет",
                            "http_method": "GET",
                            "raw_name": "limits",
                        },
                        {
                            "start_quote": "GET limits",
                            "end_quote": "обязательный",
                            "http_method": "GET",
                            "raw_name": "limits",
                        },
                    ]
                }
            self.assertEqual(task, "openapi")
            self.assertIn("GET limits", user)
            self.assertNotIn("этой цитаты нет", user)
            return {"yaml": yaml_text}

        result = convert_text(text, Settings(title="Limits"), Spy(handler=handler))
        self.assertEqual(result.errors, [])
        self.assertEqual(list(result.spec["paths"]), ["/limits"])
        self.assertEqual(result.spec["paths"]["/limits"]["get"]["operationId"], "getLimits")

    def test_claim_fragment_yaml_is_stitched_without_table_parser(self) -> None:
        other = "Другой раздел про счета и остатки.\n\n"
        claim = """## 1. Регистрация кредитной заявки физического лица (CR-050)

| URL                | {base_url}/claim/individual   | {base_url}/claim/individual                             | {base_url}/claim/individual   |
|--------------------|-------------------------------|---------------------------------------------------------|-------------------------------|
| Входящие параметры | Входящие параметры            | Входящие параметры                                      | Входящие параметры            |
| Название           | Тип                           | Описание                                                | Признак                       |
| header             | object                        | Реквизиты поставщика                                    | M                             |
| type               | String(1)                     | Идентификатор типа поставщика                           | M                             |
| head               | String(3)                     | Головной код организации                                | M                             |
"""
        yaml_text = """
openapi: 3.0.3
info: {title: Claims, version: "1.0.0"}
servers:
  - url: "{base_url}"
    variables:
      base_url:
        default: /
paths:
  /claim/individual:
    post:
      operationId: registerIndividualClaim
      x-method-origin: inferred
      responses:
        "200":
          description: ok
"""

        def handler(task, user):
            if task == "segment":
                return {
                    "segments": [
                        {
                            "start_quote": "## 1. Регистрация кредитной заявки физического лица (CR-050)",
                            "end_quote": "Головной код организации",
                            "title": "CR-050",
                            "http_method": None,
                            "raw_name": None,
                        }
                    ]
                }
            self.assertEqual(task, "openapi")
            self.assertIn("CR-050", user)
            self.assertIn("{base_url}/claim/individual", user)
            self.assertNotIn("Другой раздел", user)
            return {"yaml": yaml_text}

        result = convert_text(other + claim, Settings(title="Claims"), Spy(handler=handler))
        self.assertEqual(result.errors, [])
        self.assertIn("/claim/individual", result.spec["paths"])
        operation = result.spec["paths"]["/claim/individual"]["post"]
        self.assertEqual(operation["x-method-origin"], "inferred")
        self.assertEqual(result.spec["servers"][0]["url"], "{base_url}")

    def test_long_document_requests_do_not_contain_full_text(self) -> None:
        sections = [
            f"## Раздел {index}\n\nТекст раздела {index}. " + ("слово " * 400) for index in range(6)
        ]
        text = "\n\n".join(sections)
        empty = """
openapi: 3.0.3
info: {title: API, version: "1.0.0"}
paths: {}
"""

        def handler(task, user):
            self.assertLess(len(user), len(text))
            if task == "segment":
                self.assertIn("Окно символов", user)
                for index in range(6):
                    heading = f"## Раздел {index}"
                    if heading in user and f"Текст раздела {index}." in user:
                        return {
                            "segments": [
                                {
                                    "start_quote": heading,
                                    "end_quote": f"Текст раздела {index}.",
                                    "title": heading,
                                    "http_method": None,
                                    "raw_name": None,
                                }
                            ]
                        }
                return {"segments": []}
            self.assertEqual(task, "openapi")
            return {"yaml": empty}

        result = convert_text(text, Settings(max_window_chars=1800, window_overlap=80), Spy(handler=handler))
        self.assertGreaterEqual(result.report["model_calls"], 2)
        self.assertEqual(result.errors, [])

    def test_invalid_yaml_is_retried_once(self) -> None:
        text = "GET limits\n\nПараметры запроса\n- id, тип String, длина 8, обязательный\n"
        bad = """
openapi: 3.0.3
info: {title: Limits, version: "1.0.0"}
paths:
  /limits:
    get:
      operationId: getLimits
      responses:
        "200":
          description: ok
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/Missing"
"""
        openapi_calls = {"count": 0}

        def handler(task, user):
            if task == "segment":
                return {
                    "segments": [
                        {
                            "start_quote": "GET limits",
                            "end_quote": "обязательный",
                            "http_method": "GET",
                            "raw_name": "limits",
                        }
                    ]
                }
            openapi_calls["count"] += 1
            if openapi_calls["count"] == 1:
                self.assertNotIn("Проверка OpenAPI", user)
            else:
                self.assertIn("Проверка OpenAPI", user)
                self.assertIn("GET limits", user)
                self.assertIn("Missing", user)
            return {"yaml": bad}

        result = convert_text(text, Settings(title="Limits"), Spy(handler=handler))
        self.assertEqual(openapi_calls["count"], 2)
        self.assertTrue(result.errors)
        self.assertIn("/limits", result.yaml_text)
        questions = " ".join(gap["question"] for gap in result.report["gaps"])
        self.assertIn("Missing", questions)

    def test_stitch_reuses_same_shape_and_renames_conflict(self) -> None:
        item = {"type": "object", "properties": {"id": {"type": "string"}}}
        other = {"type": "object", "properties": {"name": {"type": "string"}}}

        def part(path: str, method: str, operation_id: str, schema: dict) -> dict:
            return {
                "paths": {
                    path: {
                        method: {
                            "operationId": operation_id,
                            "responses": {
                                "200": {
                                    "description": "ok",
                                    "content": {
                                        "application/json": {"schema": {"$ref": "#/components/schemas/Item"}}
                                    },
                                }
                            },
                        }
                    }
                },
                "components": {"schemas": {"Item": schema}},
            }

        spec, notes = stitch_documents(
            [
                part("/a", "get", "getA", item),
                part("/b", "get", "getB", item),
                part("/c", "post", "getA", other),
            ],
            title="T",
            version="1",
        )
        self.assertEqual(set(spec["components"]["schemas"]), {"Item", "Item2"})
        same = spec["paths"]["/b"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
        renamed = spec["paths"]["/c"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
        self.assertEqual(same, {"$ref": "#/components/schemas/Item"})
        self.assertEqual(renamed, {"$ref": "#/components/schemas/Item2"})
        self.assertEqual(spec["paths"]["/c"]["post"]["operationId"], "getA2")
        self.assertTrue(any("Item2" in note for note in notes))

    def test_stitch_keeps_shared_responses(self) -> None:
        def part(path: str, operation_id: str, not_found: str) -> dict:
            return {
                "openapi": "3.0.3",
                "info": {"title": "RecipeGA", "version": "1.0.0"},
                "paths": {
                    path: {
                        "get": {
                            "operationId": operation_id,
                            "parameters": [
                                {
                                    "name": path.strip("/").split("/")[-1].strip("{}"),
                                    "in": "path",
                                    "required": True,
                                    "schema": {"type": "string"},
                                }
                            ],
                            "responses": {
                                "400": {"$ref": "#/components/responses/BadRequest"},
                                "404": {"$ref": "#/components/responses/NotFound"},
                            },
                        }
                    }
                },
                "components": {
                    "schemas": {
                        "ErrorResponse": {
                            "type": "object",
                            "properties": {"errorMessage": {"type": "string"}},
                        }
                    },
                    "responses": {
                        "BadRequest": {
                            "description": "Некорректный запрос",
                            "x-origin": "skill-template",
                        },
                        "NotFound": {
                            "description": not_found,
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                    },
                },
            }

        spec, notes = stitch_documents(
            [
                part("/recipes/{recipeId}", "getRecipeById", "Рецепт не найден"),
                part("/authors/{authorId}", "getAuthorById", "Автор не найден"),
            ],
            title="RecipeGA",
            version="1.0.0",
        )
        responses = spec["components"]["responses"]
        self.assertIn("BadRequest", responses)
        self.assertIn("NotFound", responses)
        self.assertIn("NotFound2", responses)
        self.assertNotIn("BadRequest2", responses)
        recipe = spec["paths"]["/recipes/{recipeId}"]["get"]["responses"]
        author = spec["paths"]["/authors/{authorId}"]["get"]["responses"]
        self.assertEqual(recipe["400"], {"$ref": "#/components/responses/BadRequest"})
        self.assertEqual(recipe["404"], {"$ref": "#/components/responses/NotFound"})
        self.assertEqual(author["404"]["$ref"], "#/components/responses/NotFound2")
        self.assertTrue(any("NotFound2" in note for note in notes))
        from openapi_from_spec.validate import schema_errors

        self.assertEqual(schema_errors(spec), [])

    def test_early_end_quote_keeps_mapping_table(self) -> None:
        text = """## Метод GET /recipes/{recipeId}

Пример ответа

Код HTTP: 200

## Для тела ответа

| Параметр в JSON | Описание | Обязательность | Тип данных |
| title | Название | Да | string |

## Метод POST /recipes

Создать рецепт
"""
        empty = """
openapi: 3.0.3
info: {title: API, version: "1.0.0"}
paths: {}
"""

        def handler(task, user):
            if task == "segment":
                return {
                    "segments": [
                        {
                            "start_quote": "## Метод GET /recipes/{recipeId}",
                            "end_quote": "Код HTTP: 200",
                            "title": "получить рецепт по ID",
                            "http_method": "GET",
                            "raw_name": "recipes",
                        }
                    ]
                }
            self.assertEqual(task, "openapi")
            self.assertIn("Параметр в JSON", user)
            self.assertIn("title", user)
            self.assertNotIn("Создать рецепт", user)
            return {"yaml": empty}

        result = convert_text(text, Settings(title="RecipeGA"), Spy(handler=handler))
        self.assertEqual(result.errors, [])
        self.assertEqual(result.report["unresolved"][0]["span"][1], text.find("## Метод POST"))

    def test_model_segments_only_a_window(self) -> None:
        text = "Абзац без таблицы и без метода. " * 80
        empty = """
openapi: 3.0.3
info: {title: API, version: "1.0.0"}
paths: {}
"""

        def handler(task, user):
            self.assertLess(len(user), len(text))
            if task == "openapi":
                return {"yaml": empty}
            self.assertEqual(task, "segment")
            self.assertLess(len(user), 900)
            self.assertIn("Окно символов", user)
            start = text.find("Абзац")
            return {
                "segments": [
                    {
                        "start_quote": text[start : start + 20],
                        "end_quote": "без метода.",
                        "title": "",
                        "http_method": "GET",
                        "raw_name": "missing",
                    }
                ]
            }

        result = convert_text(text, Settings(max_window_chars=500, window_overlap=0), Spy(handler=handler))
        self.assertGreaterEqual(result.report["model_calls"], 1)
        self.assertEqual(result.spec["paths"], {})

    def test_large_catalog_does_not_call_model(self) -> None:
        blocks = []
        for index in range(25):
            blocks.append(
                f"GET item{index}\n\nОписание\nМетод номер {index}.\n\n"
                "Параметры запроса\n\n"
                "Параметр\tТип данных\tМаксимальная длина\tОбязательность\tОписание\n"
                "id\tString\t8\tДа\tИдентификатор\n"
            )
        result = compile_rules("\n".join(blocks), Settings())
        self.assertEqual(result.errors, [])
        self.assertEqual(len(result.spec["paths"]), 25)
        self.assertEqual(result.report["model_calls"], 0)

    def test_post_body(self) -> None:
        text = (
            "POST payments\n\nОписание\nСоздаёт платёж.\n\n"
            "Параметры запроса\n\n"
            "Параметр\tТип данных\tМаксимальная длина\tОбязательность\tОписание\n"
            "amount\tLong\t18\tДа\tСумма\n"
            "currency\tString\t3\tДа\tВалюта\n\n"
            "Параметры ответа\n\n"
            "Парамет\tТип данных\tМаксимальная длина\tОписание\n"
            "paymentId\tString\t36\tИдентификатор платежа\n"
        )
        result = compile_rules(text, Settings())
        operation = result.spec["paths"]["/api/v1/payments"]["post"]
        self.assertTrue(operation["requestBody"]["required"])
        name = operation["requestBody"]["content"]["application/json"]["schema"]["$ref"].split("/")[-1]
        required = result.spec["components"]["schemas"][name]["required"]
        self.assertEqual(required, ["amount", "currency"])
        self.assertIn("paymentId", result.yaml_text)

    def test_windows_cover_text_and_keep_table_header(self) -> None:
        preamble = "Введение. " * 30
        header = "Параметр\tТип данных\tМаксимальная длина\tОписание"
        rows = [f"field{index}\tString\t8\tПоле {index}" for index in range(12)]
        text = preamble + "\n" + header + "\n" + "\n".join(rows)
        windows = pack_windows(text, 280, overlap=40)
        self.assertGreater(len(windows), 1)
        self.assertEqual(windows[0][0], 0)
        self.assertEqual(windows[-1][1], len(text))
        for previous, current in zip(windows, windows[1:]):
            self.assertGreater(current[1], previous[1])
        late = next(start for start, _end in windows if start > text.find(header))
        shown = render_window(text, late, text.find("field11"))
        self.assertIn("Параметр", shown)
        self.assertIn("field11", text)

    def test_skill_has_no_links_and_openapi_rules(self) -> None:
        text = load_skill_markdown()
        self.assertNotIn("http://", text)
        self.assertNotIn("https://", text)
        prompt = system_prompt("extract")
        self.assertNotIn("name: openapi-from-spec", prompt)
        self.assertIn("YAML", prompt)
        self.assertIn("Задача extract", strip_frontmatter(text))
        mapping = (skill_dir() / "knowledge" / "type-mapping.md").read_text(encoding="utf-8")
        for word in ("List", "String", "Long", "Array"):
            self.assertIn(word, mapping)
        rules = (skill_dir() / "knowledge" / "openapi-3.0.3.md").read_text(encoding="utf-8")
        self.assertNotIn("http://", rules)
        self.assertNotIn("https://", rules)
        self.assertIn("3.0.3", rules)
        prompt = system_prompt("openapi")
        self.assertIn("Задача openapi", prompt)
        self.assertIn("$ref", prompt)
        self.assertIn("x-method-origin", prompt)
        self.assertIn("skill-template", prompt)
        self.assertIn("Таблица полей важнее примера", prompt)
        self.assertIn("camelCase", prompt)
        self.assertIn("getRecipesRecipeId", prompt)
        self.assertIn("есть `items`", strip_frontmatter(text))
        self.assertIn("Что чем является", strip_frontmatter(text))
        self.assertIn("400, 401, 404, 500", strip_frontmatter(text))
        self.assertIn("В `responses` только коды", strip_frontmatter(text))
        self.assertNotIn("http://", prompt)
        self.assertNotIn("https://", prompt)


    def test_identifier_cell_keeps_a_single_space_only_for_prose(self) -> None:
        text = (
            "| author.published At | Дата публикации рецепта |\n"
            "| ingredients[].i d | UUID продукта |\n"
            "| description | может быть пустым |\n"
        )
        repaired = normalize(text)
        self.assertIn("author.publishedAt", repaired)
        self.assertIn("ingredients[].id", repaired)
        self.assertIn("Дата публикации рецепта", repaired)
        self.assertIn("может быть пустым", repaired)

    def test_stitch_merges_same_route_and_reports_type_conflict(self) -> None:
        def part(schema: dict, *extra_parameters: dict, extra_responses: dict | None = None) -> dict:
            responses = {
                "200": {
                    "description": "ok",
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Recipe"}}},
                }
            }
            responses.update(extra_responses or {})
            parameters = [
                {"name": "recipeId", "in": "path", "required": True, "schema": {"type": "string", "format": "uuid"}}
            ]
            parameters.extend(extra_parameters)
            return {
                "paths": {
                    "/public/v1/recipes/{recipeId}": {
                        "get": {
                            "operationId": "getRecipes",
                            "parameters": parameters,
                            "responses": responses,
                        }
                    }
                },
                "components": {"schemas": {"Recipe": schema}},
            }

        spec, notes = stitch_documents(
            [
                part({"type": "object", "properties": {"id": {"type": "string"}}}, extra_responses={"404": {"description": "Рецепт не найден"}}),
                part(
                    {"type": "object", "properties": {"id": {"type": "integer"}, "title": {"type": "string"}}},
                    {"name": "verbose", "in": "query", "schema": {"type": "boolean"}},
                    extra_responses={"400": {"description": "Неверный формат id"}},
                ),
            ],
            title="Recipes",
            version="1",
        )
        operation = spec["paths"]["/public/v1/recipes/{recipeId}"]["get"]
        names = {(item["in"], item["name"]) for item in operation["parameters"]}
        self.assertIn(("query", "verbose"), names)
        self.assertEqual(operation["responses"]["404"]["description"], "Рецепт не найден")
        self.assertEqual(operation["responses"]["400"]["description"], "Неверный формат id")
        properties = spec["components"]["schemas"]["Recipe"]["properties"]
        self.assertEqual(properties["id"]["type"], "string")
        self.assertEqual(properties["title"]["type"], "string")
        self.assertTrue(any("Конфликт типа" in note for note in notes))
        self.assertTrue(any("объединён" in note for note in notes))

    def test_long_recipe_method_becomes_one_operation(self) -> None:
        document = (ROOT / "docling" / "GetAnalyst_REST_API_metod_Zapolnennyi_774_shablon_zadachi.md").read_text(
            encoding="utf-8"
        )
        prepared = normalize(document)
        head = prepared.find("Метод GET /recipes/{recipeId}")
        table = prepared.find("Параметр в JSON")
        self.assertGreater(table, head)
        window = max((table - head) // 2, 400)
        fields = (
            ("id", "uuid", True),
            ("title", "string", True),
            ("description", "string", False),
            ("cookingTime", "integer", True),
            ("calories", "number", True),
            ("proteins", "number", True),
            ("fats", "number", True),
            ("carbs", "number", True),
            ("image", "string (url)", False),
            ("author.id", "uuid", True),
            ("author.name", "string", True),
            ("author.avatar", "string (url)", False),
            ("author.publishedAt", "string (timestamp)", True),
            ("category", "string", True),
            ("ingredients[].id", "uuid", True),
            ("ingredients[].name", "string", True),
            ("ingredients[].quantity", "integer", True),
            ("steps[].step", "integer", True),
            ("steps[].text", "string", True),
            ("steps[].image", "string (url)", False),
            ("is_favorited", "boolean", True),
            ("tags[]", "array", False),
        )

        def window_text(user: str) -> str:
            for marker in (
                "В запросе нет остальных частей документа.",
                "Извлеки из него одну операцию.",
            ):
                index = user.rfind(marker)
                if index >= 0:
                    return user[index + len(marker) :].lstrip("\n")
            return user

        def edges(fragment: str) -> tuple[str, str] | None:
            lines = [line.strip() for line in fragment.splitlines() if len(line.strip()) >= 8]
            if lines and "параметр" in lines[0].lower() and "тип" in lines[0].lower():
                lines = lines[1:]
            if len(lines) < 2:
                return None
            start = lines[0]
            begin = fragment.find(start)
            for line in reversed(lines[1:]):
                if fragment.find(line, begin + len(start)) >= 0:
                    return start, line
            return None

        def cell(fragment: str, name: str) -> str | None:
            for line in fragment.splitlines():
                stripped = line.strip()
                if not stripped.startswith("|"):
                    continue
                cells = [item.strip() for item in stripped.strip("|").split("|")]
                if cells and cells[0] == name:
                    return cells[0]
            return None

        def handler(task, user):
            fragment = window_text(user)
            if task == "openapi":
                return {"yaml": "openapi: 3.0.3\ninfo: {title: API, version: \"1.0.0\"}\npaths: {}\n"}
            if task == "segment":
                body: dict = {"segments": [], "continues_previous": None}
                if "Метод GET /recipes/{recipeId}" in fragment:
                    pair = edges(fragment[fragment.find("Метод GET /recipes/{recipeId}") :])
                    if pair:
                        body["segments"].append(
                            {
                                "start_quote": "Метод GET /recipes/{recipeId}",
                                "end_quote": pair[1],
                                "title": "получить рецепт по ID",
                                "http_method": "GET",
                                "raw_name": "recipes",
                            }
                        )
                elif (pair := edges(fragment)):
                    body["continues_previous"] = {"start_quote": pair[0], "end_quote": pair[1]}
                return body
            response_fields = []
            for name, tz_type, required in fields:
                quote = cell(fragment, name)
                if quote is None:
                    continue
                response_fields.append(
                    {
                        "name": name,
                        "location": "response",
                        "tz_type": tz_type,
                        "required": required,
                        "description": name,
                        "quote": quote,
                        "children": [],
                    }
                )
            parameters = []
            if "{recipeId}" in fragment:
                parameters.append(
                    {
                        "name": "recipeId",
                        "location": "path",
                        "tz_type": "UUID",
                        "required": True,
                        "description": "Уникальный идентификатор рецепта",
                        "quote": "{recipeId}",
                        "children": [],
                    }
                )
            responses = []
            if "Рецепт не найден" in fragment and "404" in fragment:
                responses.append({"code": "404", "description": "Рецепт не найден", "quote": "Код HTTP: 404"})
            if "не соответствует типу данных UUID" in fragment and "400" in fragment:
                responses.append(
                    {
                        "code": "400",
                        "description": "не соответствует типу данных UUID",
                        "quote": "Код HTTP: 400",
                    }
                )
            if "Сервис недоступен, ошибка на сервере" in fragment and "503" in fragment:
                responses.append(
                    {
                        "code": "503",
                        "description": "Сервис недоступен, ошибка на сервере",
                        "quote": "Код HTTP: 503",
                    }
                )
            gaps = []
            if "Жанр указан некорректно" in fragment:
                gaps.append(
                    {
                        "field": "400",
                        "question": "Пример 400 говорит про жанр, хотя текст описывает формат UUID",
                        "quote": "Жанр указан некорректно",
                    }
                )
            if "28 }, 29 {" in fragment:
                gaps.append(
                    {
                        "field": "200",
                        "question": "Пример 200 обрезан и содержит номера строк",
                        "quote": "28 }, 29 {",
                    }
                )
            return {
                "http_method": "GET" if "GET" in fragment else None,
                "raw_name": "recipes" if "recipes" in fragment else "",
                "summary": "",
                "description": "",
                "parameters": parameters,
                "body_fields": [],
                "response_fields": response_fields,
                "example": None,
                "responses": responses,
                "gaps": gaps,
            }

        spy = Spy(handler=handler)
        result = convert_text(
            document,
            Settings(title="RecipeGA", max_window_chars=window, window_overlap=0),
            spy,
        )
        tasks = [call["task"] for call in spy.calls]
        self.assertIn("extract", tasks)
        self.assertNotIn("openapi", tasks)
        self.assertEqual(result.errors, [])
        self.assertEqual(list(result.spec["paths"]), ["/public/v1/recipes/{recipeId}"])
        operation = result.spec["paths"]["/public/v1/recipes/{recipeId}"]["get"]
        self.assertEqual(operation["operationId"], "getRecipes")
        self.assertEqual(result.spec["servers"], [{"url": "https://api.recipe-ga.com"}])
        parameter = operation["parameters"][0]
        self.assertEqual(parameter["name"], "recipeId")
        self.assertEqual(parameter["schema"]["format"], "uuid")
        schemas = result.spec["components"]["schemas"]
        response = schemas["RecipesResponse"]["properties"]
        self.assertEqual(response["author"]["$ref"], "#/components/schemas/Author")
        self.assertEqual(response["ingredients"]["type"], "array")
        self.assertEqual(response["steps"]["type"], "array")
        self.assertEqual(response["is_favorited"]["type"], "boolean")
        self.assertEqual(response["tags"]["type"], "array")
        self.assertIn("publishedAt", schemas["Author"]["properties"])
        self.assertEqual(schemas["Author"]["properties"]["publishedAt"]["format"], "date-time")
        self.assertIn("name", schemas["Ingredient"]["properties"])
        self.assertIn("text", schemas["Step"]["properties"])
        self.assertEqual(operation["responses"]["404"]["description"], "Рецепт не найден")
        self.assertIn("UUID", operation["responses"]["400"]["description"])
        self.assertIn("503", operation["responses"])
        self.assertNotIn("401", operation["responses"])
        self.assertNotIn("500", operation["responses"])
        self.assertNotIn("x-origin", operation["responses"]["400"])
        questions = " ".join(gap["question"] for gap in result.report["gaps"])
        self.assertIn("жанр", questions)
        self.assertIn("номера строк", questions)

    def test_model_yaml_drops_path_param_and_success_heading(self) -> None:
        from openapi_from_spec.pipeline import _read_yaml_payload
        from openapi_from_spec.validate import schema_errors

        document, _raw, errors = _read_yaml_payload(
            {
                "yaml": """
openapi: 3.0.3
info: {title: RecipeGA, version: 1.0.0}
paths:
  /recipes/{recipeId}:
    get:
      operationId: getRecipesRecipeId
      summary: получить рецепт по ID
      parameters:
        - name: recipeId
          in: path
          required: true
          schema: {type: string, format: uuid}
      responses:
        "200":
          description: Успех
          content:
            application/json:
              schema: {$ref: '#/components/schemas/Recipe'}
components:
  schemas:
    Recipe:
      type: object
      properties:
        tags: {type: array, description: Список тегов}
"""
            }
        )
        operation = document["paths"]["/recipes/{recipeId}"]["get"]
        self.assertEqual(operation["operationId"], "getRecipes")
        self.assertEqual(operation["responses"]["200"]["description"], "получить рецепт по ID")
        self.assertTrue(any("tags" in item and "items" in item for item in errors))
        recipe = document["components"]["schemas"]["Recipe"]
        recipe["properties"]["tags"]["items"] = {"type": "string"}
        self.assertEqual(schema_errors(document), [])

    def test_empty_required_cell_and_contradictory_example_are_removed(self) -> None:
        from openapi_from_spec.pipeline import _align_document_with_source

        source = """
## Для тела ответа

| Параметр в JSON | Описание | Обязате льность | Тип данных |
| id | UUID | Да | uuid |
| author | | | <object> |
| author.id | UUID автора | Да | uuid |

| ingredients | Список | | array |
| steps | Шаги | | array |
"""
        document = {
            "paths": {
                "/recipes/{recipeId}": {
                    "get": {
                        "responses": {
                            "400": {
                                "description": "формат id не соответствует UUID",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "errorMessage": {
                                                    "type": "string",
                                                    "example": "Жанр указан некорректно",
                                                }
                                            },
                                        }
                                    }
                                },
                            },
                            "404": {
                                "description": "Рецепт не найден по переданному id.",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "errorMessage": {
                                                    "type": "string",
                                                    "example": "Рецепт не найден",
                                                }
                                            },
                                        }
                                    }
                                },
                            },
                        }
                    }
                }
            },
            "components": {
                "schemas": {
                    "Recipe": {
                        "type": "object",
                        "required": ["id", "author", "ingredients", "steps"],
                        "properties": {},
                    },
                    "Author": {
                        "type": "object",
                        "required": ["id"],
                        "properties": {},
                    },
                }
            },
        }
        _align_document_with_source(document, source)
        self.assertEqual(document["components"]["schemas"]["Recipe"]["required"], ["id"])
        self.assertEqual(document["components"]["schemas"]["Author"]["required"], ["id"])
        bad = document["paths"]["/recipes/{recipeId}"]["get"]["responses"]["400"]
        message = bad["content"]["application/json"]["schema"]["properties"]["errorMessage"]
        self.assertNotIn("example", message)
        found = document["paths"]["/recipes/{recipeId}"]["get"]["responses"]["404"]
        kept = found["content"]["application/json"]["schema"]["properties"]["errorMessage"]
        self.assertEqual(kept["example"], "Рецепт не найден")

    def test_status_phrase_replaces_contradictory_description(self) -> None:
        from openapi_from_spec.pipeline import _align_document_with_source

        source = """
## Пример ответа - Неверный формат запроса

Если в запросе неверно указан формат id - не соответствует типу данных UUID.

Код HTTP: 400
"""
        document = {
            "paths": {
                "/recipes/{recipeId}": {
                    "get": {
                        "responses": {
                            "400": {"description": "Жанр указан некорректно"},
                            "200": {"description": "Получить рецепт по ID"},
                        }
                    }
                }
            }
        }
        _align_document_with_source(document, source)
        responses = document["paths"]["/recipes/{recipeId}"]["get"]["responses"]
        self.assertIn("UUID", responses["400"]["description"])
        self.assertEqual(responses["200"]["description"], "Получить рецепт по ID")

    def test_timestamp_cell_sets_date_time_format(self) -> None:
        from openapi_from_spec.pipeline import _align_document_with_source

        source = """
| Параметр | Тип данных |
| author.published At | string (timestamp) |
| image | string (url) |
"""
        document = {
            "components": {
                "schemas": {
                    "Author": {"type": "object", "properties": {"publishedAt": {"type": "string"}}},
                    "Recipe": {"type": "object", "properties": {"image": {"type": "string"}}},
                }
            }
        }
        _align_document_with_source(document, source)
        self.assertEqual(document["components"]["schemas"]["Author"]["properties"]["publishedAt"]["format"], "date-time")
        self.assertEqual(document["components"]["schemas"]["Recipe"]["properties"]["image"]["format"], "uri")

    def test_example_url_fills_server_when_path_is_short(self) -> None:
        from openapi_from_spec.pipeline import _align_document_with_source

        source = "1 GET https://api.recipe-ga.com/public/v1/recipes/{recipeId}\n"
        document = {
            "paths": {"/recipes/{recipeId}": {"get": {"responses": {"200": {"description": "ok"}}}}},
            "servers": [{"url": "{base_url}", "variables": {"base_url": {"default": "/"}}}],
        }
        _align_document_with_source(document, source)
        self.assertEqual(document["servers"], [{"url": "https://api.recipe-ga.com/public/v1"}])

    def test_finalize_drops_verb_list_and_keeps_example_shape(self) -> None:
        from openapi_from_spec.pipeline import finalize_spec

        source = """
URL: https://XXXXX/api/v1/currencies.json
Разрешенные запросы: только GET

| Параметр | Тип | Обязательность в запросах | Описание |
| --- | --- | --- | --- |
| id | int | post, put | Идентификатор |
| code | string | post | Код |

GET /api/v1/currencies

Модель users. https://XXXXX/api/v1/users.json
Разрешенные запросы: GET, POST, PUT, DELETE

| Параметр | Тип | Обязательность в запросах | Описание |
| --- | --- | --- | --- |
| last_name | string | post | Фамилия |
| status | int |  | Статус |

POST /api/v1/users HTTP/1.1

{
  "last_name": "Сидоров",
  "status": 5
}

Пример ответа (400 Bad Request)

{
  "error": "Отсутствует обязательный параметр - last_name"
}

PUT /api/v1/transactions/1 HTTP/1.1

{
  "sum": 0,
  "description": "Перевод"
}

| Параметр | Тип | Обязательность в запросах | Описание |
| --- | --- | --- | --- |
| sum | float | post | Сумма |
| description | string |  | Описание |

Пример ответа (422 Unprocessable Entity)

{
  "error": "Сумма должна быть больше 0"
}
"""
        spec = {
            "openapi": "3.0.3",
            "info": {"title": "Fintech", "version": "1.0.0"},
            "paths": {
                "/currencies": {
                    "get": {
                        "operationId": "getCurrencies",
                        "responses": {"200": {"description": "Список валют"}},
                    }
                },
                "/users": {
                    "get": {"operationId": "getUsers", "responses": {"200": {"description": "Список"}}},
                    "post": {
                        "operationId": "createUser",
                        "responses": {"200": {"description": "Создан"}},
                    },
                    "delete": {"operationId": "deleteUser", "responses": {"200": {"description": "Удалён"}}},
                },
                "/api/v1/users": {
                    "post": {
                        "operationId": "postUsers",
                        "description": "Ошибка *status* в запросе",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/CreateUserRequest"
                                    }
                                }
                            }
                        },
                        "responses": {
                            "201": {"description": "Создан"},
                            "400": {"description": "Не хватает поля"},
                        },
                    }
                },
                "/api/v1/transactions": {
                    "put": {
                        "operationId": "putTransactions",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/UpdateTransactionRequest"}
                                }
                            }
                        },
                        "responses": {"200": {"description": "Изменена"}},
                    }
                },
            },
            "components": {
                "schemas": {
                    "Currency": {
                        "type": "object",
                        "required": ["id", "code"],
                        "properties": {
                            "id": {"type": "integer"},
                            "code": {"type": "string"},
                        },
                    },
                    "CreateUserRequest": {
                        "type": "object",
                        "properties": {"last_name": {"type": "string"}},
                    },
                    "UpdateTransactionRequest": {
                        "type": "object",
                        "required": ["sum"],
                        "properties": {"sum": {"type": "number"}},
                    },
                }
            },
            "servers": [{"url": "{base_url}", "variables": {"base_url": {"default": "/"}}}],
        }
        finalize_spec(spec, source)
        self.assertEqual(spec["servers"], [{"url": "{base_url}", "variables": {"base_url": {"default": "/"}}}])
        self.assertEqual(set(spec["paths"]["/api/v1/users"]), {"post"})
        self.assertNotIn("*", spec["paths"]["/api/v1/users"]["post"]["description"])
        self.assertEqual(spec["components"]["schemas"]["Currency"]["required"], ["id", "code"])
        unprocessable = spec["paths"]["/api/v1/transactions/{id}"]["put"]["responses"]["422"]
        self.assertEqual(unprocessable["content"]["application/json"]["example"]["error"], "Сумма должна быть больше 0")

    def test_missed_example_lines_are_added_to_the_model_spec(self) -> None:
        from openapi_from_spec.pipeline import finalize_spec

        source = (ROOT / "docling" / "habr-shablon-tz-rest-api.md").read_text(encoding="utf-8")
        spec = {
            "openapi": "3.0.3",
            "info": {"title": "Fintech", "version": "1.0.0"},
            "paths": {
                "/accounts": {
                    "post": {
                        "operationId": "createAccounts",
                        "summary": "Создать модель accounts",
                        "description": "Счёт хранит баланс клиента.",
                        "responses": {"200": {"description": "Создать модель accounts"}},
                    }
                },
                "/transactions/{id}": {
                    "put": {
                        "operationId": "updateTransactions",
                        "responses": {"422": {"description": "Сумма должна быть больше 0"}},
                    }
                },
            },
        }
        finalize_spec(spec, source)
        self.assertEqual(set(spec["paths"]), {"/accounts", "/transactions/{id}"})
        self.assertEqual(spec["paths"]["/accounts"]["post"]["summary"], "Создать модель accounts")

    def test_model_draft_drops_instruction_and_false_required(self) -> None:
        from openapi_from_spec.pipeline import finalize_spec

        source = """
curl -X GET "<https://meowgram.ru/api/v1/cats/1/posts?page=1&pageSize=10>" \\
     -H "Authorization: Bearer ACCESS_TOKEN"
"imageUrl": "<https://example.com/images/cat1.jpg>"
"""
        spec = {
            "openapi": "3.0.3",
            "info": {"title": "Meowgram", "version": "1.0.0"},
            "paths": {
                "/cats/{catId}/posts": {
                    "get": {
                        "summary": "Получение постов котика",
                        "description": "Указываем: логичнее было бы назвать /my/posts",
                        "parameters": [
                            {
                                "name": "page",
                                "in": "query",
                                "required": True,
                                "description": "Для каждого параметра указываем описание",
                                "schema": {"type": "integer"},
                            },
                            {
                                "name": "pageSize",
                                "in": "query",
                                "required": True,
                                "description": "Для каждого параметра указываем описание",
                                "schema": {"type": "integer"},
                            },
                            {
                                "name": "catId",
                                "in": "path",
                                "required": True,
                                "description": "Указываем идентификатор",
                                "schema": {"type": "integer"},
                            },
                        ],
                        "responses": {
                            "200": {
                                "description": "ok",
                                "content": {
                                    "application/json": {
                                        "example": {"imageUrl": "<https://example.com/images/cat1.jpg>"}
                                    }
                                },
                            },
                            "400": {
                                "description": "Ошибка",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/Error401"}}
                                },
                            },
                        },
                    }
                }
            },
            "components": {
                "schemas": {"Error401": {"type": "object", "properties": {"error": {"type": "string"}}}}
            },
            "servers": [{"url": "https://meowgram.ru/api/v1"}],
        }
        finalize_spec(spec, source)
        operation = spec["paths"]["/cats/{catId}/posts"]["get"]
        self.assertIn("указываем", str(operation.get("description") or "").lower())
        page = next(item for item in operation["parameters"] if item["name"] == "page")
        self.assertIn("required", page)
        self.assertIn("Authorization", [item["name"] for item in operation["parameters"] if item.get("in") == "header"])
        example = operation["responses"]["200"]["content"]["application/json"]["example"]
        self.assertEqual(example["imageUrl"], "https://example.com/images/cat1.jpg")
        schema = operation["responses"]["400"]["content"]["application/json"]["schema"]
        self.assertNotIn("$ref", schema)
        self.assertIn("error", schema["properties"])

    def test_incorrect_request_example_does_not_replace_the_table(self) -> None:
        from openapi_from_spec.pipeline import finalize_spec

        source = """
**Создать модель accounts**

Модель accounts описывает банковские счета пользователей.

| Параметр | Тип | Обязательность в запросах | Описание |
| --- | --- | --- | --- |
| user_id | int | post | Ссылка на пользователя |
| acc_number | string | post | Номер счета |
| currency_id | int | post | Ссылка на валюту |
| is_default | bool |  | Основной счет |

*Пример некорректного POST-запроса*

POST /api/v1/accounts HTTP/1.1

{
  "acc_number": "40817810099910004314",
  "currency_id": 1,
  "is_default": false
}

*Пример ответа на POST-запрос *(201 Created)

{
  "data": {"id": 3, "email": null}
}
"""
        spec = {
            "openapi": "3.0.3",
            "info": {"title": "Fintech", "version": "1.0.0"},
            "paths": {
                "/accounts": {
                    "post": {
                        "description": "Пример некорректного POST-запроса на создание счета.",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "acc_number": {"type": "string"},
                                            "currency_id": {"type": "integer"},
                                            "is_default": {"type": "boolean"},
                                        },
                                    },
                                    "example": {
                                        "acc_number": "40817810099910004314",
                                        "currency_id": 1,
                                        "is_default": False,
                                    },
                                }
                            }
                        },
                        "responses": {
                            "201": {
                                "description": "Создан",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "data": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "id": {"type": "integer"},
                                                            "email": {"type": "string"},
                                                        },
                                                    },
                                                }
                                            },
                                        },
                                        "example": {"data": {"id": 3, "email": None}},
                                    }
                                },
                            }
                        },
                    }
                }
            },
        }
        finalize_spec(spec, source)
        media = spec["paths"]["/accounts"]["post"]["requestBody"]["content"]["application/json"]
        self.assertIn("example", media)
        self.assertNotIn("user_id", media["schema"]["properties"])
        data = spec["paths"]["/accounts"]["post"]["responses"]["201"]["content"]["application/json"]
        self.assertEqual(data["schema"]["properties"]["data"]["type"], "array")

    def test_status_body_stays_with_its_code_and_missed_curl_is_a_fragment(self) -> None:
        from openapi_from_spec.models import Segment
        from openapi_from_spec.pipeline import _bind_status_bodies, _missed_request_segments

        source = """
### Публикация нового поста котика

curl -X POST '<https://meowgram.ru/api/v1/posts>'

HTTP/1.1 201 Created
{"post_id": 123, "text": "кот"}

HTTP/1.1 400 Bad Request
{"error": "Invalid request", "messageOutput": "Некорректный формат запроса"}
"""
        spec = {
            "paths": {
                "/posts": {
                    "post": {
                        "responses": {
                            "200": {
                                "description": "Создание",
                                "content": {"application/json": {"schema": {"type": "object"}, "example": {"error": "x"}}},
                            },
                            "201": {
                                "description": "Некорректный формат запроса",
                                "content": {
                                    "application/json": {
                                        "schema": {"type": "object", "properties": {"error": {"type": "string"}}},
                                        "example": {"error": "Некорректный формат запроса"},
                                    }
                                },
                            },
                            "400": {"description": "Некорректный формат запроса"},
                        }
                    }
                }
            }
        }
        _bind_status_bodies(spec, source)
        responses = spec["paths"]["/posts"]["post"]["responses"]
        self.assertNotIn("200", responses)
        created = responses["201"]["content"]["application/json"]
        self.assertEqual(created["example"]["post_id"], 123)
        self.assertIn("post_id", created["schema"]["properties"])
        self.assertEqual(responses["201"]["description"], "Created")
        self.assertEqual(
            responses["400"]["content"]["application/json"]["example"]["messageOutput"],
            "Некорректный формат запроса",
        )
        text = (ROOT / "docling" / "habr-magnit-meowgram.md").read_text(encoding="utf-8")
        post_at = text.find("### Публикация нового поста котика")
        covered = Segment(
            start=post_at,
            end=len(text),
            text=text[post_at:],
            http_method="post",
            explicit_path="/posts",
        )
        missed = _missed_request_segments(text, [covered])
        self.assertTrue(any(item.http_method == "get" and "cats/1/posts" in item.text for item in missed))
        self.assertFalse(any(item.http_method == "post" for item in missed))

    def test_short_path_does_not_steal_longer_url_prefix(self) -> None:
        from openapi_from_spec.pipeline import finalize_spec

        source = """
curl -X GET "<https://meowgram.ru/api/v1/cats/1/posts?page=1>"
HTTP/1.1 401 Unauthorized
{"error": "Signature has expired", "messageOutput": "Требуется аутентификация"}

curl -X POST '<https://meowgram.ru/api/v1/posts>'
"""
        spec = {
            "paths": {
                "/cats/{catId}/posts": {
                    "get": {
                        "operationId": "getCatsPosts",
                        "responses": {"401": {"$ref": "#/components/responses/Unauthorized"}},
                    }
                },
                "/posts": {
                    "post": {"operationId": "createPost", "responses": {"201": {"description": "Создан"}}}
                },
            },
            "components": {
                "responses": {"Unauthorized": {"description": "Требуется аутентификация", "x-origin": "skill-template"}}
            },
            "servers": [{"url": "{base_url}", "variables": {"base_url": {"default": "/"}}}],
        }
        finalize_spec(spec, source)
        self.assertEqual(spec["servers"], [{"url": "{base_url}", "variables": {"base_url": {"default": "/"}}}])
        unauthorized = spec["paths"]["/cats/{catId}/posts"]["get"]["responses"]["401"]
        self.assertIn("Требуется аутентификация", json.dumps(unauthorized, ensure_ascii=False))

    def test_without_model_document_is_not_cut(self) -> None:
        text = (ROOT / "docling" / "habr-shablon-tz-rest-api.md").read_text(encoding="utf-8")
        result = convert_text(text, Settings(title="Fintech"), None)
        self.assertEqual(result.spec["paths"], {})
        self.assertEqual(result.report["model_calls"], 0)
        self.assertIn("только модель", result.report["gaps"][0]["question"])

    def test_unreachable_model_is_not_an_empty_segmentation(self) -> None:
        def fail(task, user):
            raise ConnectionRefusedError(10061, "Подключение не установлено")

        result = convert_text("GET /orders\n", Settings(), Spy(handler=fail))
        question = result.report["gaps"][0]["question"]
        self.assertIn("Модель недоступна", question)
        self.assertNotIn("дословными границами", question)

    def test_empty_model_segments_do_not_invent_methods(self) -> None:
        text = (ROOT / "docling" / "habr-magnit-meowgram.md").read_text(encoding="utf-8")
        result = convert_text(text, Settings(title="Meowgram"), Spy({"segments": []}))
        self.assertEqual(result.spec["paths"], {})
        self.assertIn("дословными границами", result.report["gaps"][0]["question"])

    def test_empty_model_without_urls_keeps_the_boundary_gap(self) -> None:
        result = convert_text("Описание сервиса без методов.", Settings(), Spy({"segments": []}))
        spec = yaml.safe_load(result.yaml_text)
        self.assertEqual(spec["paths"], {})
        self.assertIn("дословными границами", result.report["gaps"][0]["question"])

    def test_quote_span_matches_url_inside_angle_brackets(self) -> None:
        from openapi_from_spec.pipeline import _quote_span

        text = 'curl -X GET "<https://meowgram.ru/api/v1/cats/1/posts?page=1>"\n'
        span = _quote_span(
            text,
            0,
            len(text),
            {
                "start_quote": "curl  -X   GET",
                "end_quote": '"https://meowgram.ru/api/v1/cats/1/posts?page=1"',
            },
        )
        self.assertEqual(span, (0, text.index("\n")))

    def test_table_object_wins_over_example_array_and_closed_codes(self) -> None:
        from openapi_from_spec.pipeline import finalize_spec

        source = """
| Наименование | Обозначение | Формат | Обязательность | Примечание |
| --- | --- | --- | --- | --- |
| classRecord | Объект класса | ClassRecord | + | |
| groupRecords | Объект группы | GroupRecord | + | |

Сервис использует следующие HTTP-коды для информирования о статусе обработки запроса:

400 Bad Request - некорректный формат параметра date

500 Internal Server Error - внутренняя ошибка сервера.
"""
        spec = {
            "paths": {
                "/api/v1/GetAllRecords": {
                    "get": {
                        "operationId": "getAllRecords",
                        "responses": {
                            "200": {
                                "description": "Набор записей",
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/schemas/GetAllRecordsResponse"},
                                        "example": {
                                            "classRecords": [{"code": "FNS"}],
                                            "groupRecords": [{"id": "0659191f-39c6-41b1-b126-62ec5d32e6db"}],
                                        },
                                    }
                                },
                            },
                            "400": {"description": "некорректный формат параметра date"},
                            "401": {"$ref": "#/components/responses/Unauthorized"},
                            "404": {"$ref": "#/components/responses/NotFound"},
                            "500": {"description": "внутренняя ошибка сервера"},
                        },
                    }
                }
            },
            "components": {
                "schemas": {
                    "GetAllRecordsResponse": {
                        "type": "object",
                        "required": ["classRecords", "groupRecords"],
                        "properties": {
                            "classRecords": {
                                "type": "array",
                                "items": {"$ref": "#/components/schemas/ClassRecord"},
                            },
                            "groupRecords": {
                                "type": "array",
                                "items": {"$ref": "#/components/schemas/GroupRecord"},
                            },
                        },
                    },
                    "ClassRecord": {"type": "object", "properties": {"code": {"type": "string"}}},
                    "GroupRecord": {"type": "object", "properties": {"id": {"type": "string"}}},
                },
                "responses": {
                    "Unauthorized": {"description": "Требуется аутентификация", "x-origin": "skill-template"},
                    "NotFound": {"description": "Ресурс не найден", "x-origin": "skill-template"},
                },
            },
        }
        finalize_spec(spec, source)
        body = spec["components"]["schemas"]["GetAllRecordsResponse"]
        self.assertIn("classRecords", body["properties"])
        self.assertNotIn("classRecord", body["properties"])
        self.assertEqual(body["required"], ["classRecords", "groupRecords"])
        responses = spec["paths"]["/api/v1/GetAllRecords"]["get"]["responses"]
        self.assertEqual(responses["400"]["description"], "некорректный формат параметра date")
        self.assertNotIn("401", responses)
        self.assertNotIn("404", responses)

    def test_json_example_array_overrides_table_object(self) -> None:
        from openapi_from_spec.pipeline import finalize_spec

        source = (ROOT / "docling" / "api_proxy_esnsi_m4d_esia.md").read_text(encoding="utf-8")
        spec = {
            "openapi": "3.0.3",
            "info": {"title": "API", "version": "1.0.0"},
            "paths": {
                "/GetAllRecords": {
                    "get": {
                        "operationId": "getAllRecords",
                        "tags": ["Classifier", "NotInDocument"],
                        "parameters": [
                            {
                                "name": "date",
                                "in": "query",
                                "required": False,
                                "description": "Возвращает набор записей справочника. Поддерживаемые форматы: yyyy-MM-dd, yyyy-MM-ddTHH:mm, yyyy-MM-ddTHH:mm:ss, yyyy-MM-ddTHH:mm:ss.sss, yyyy-MM-ddTHH:mm:ss.sssZ Форматы: yyyy-MM-dd, yyyy-MM-ddTHH:mm, yyyy-MM-ddTHH:mm:ss, yyyy-MM-ddTHH:mm:ss.sss, yyyy-MM-ddTHH:mm:ss.sssZ",
                                "schema": {"type": "string", "format": "date-time"},
                            }
                        ],
                        "responses": {
                            "200": {
                                "description": "Набор записей",
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/schemas/GetAllRecordsResponse"}
                                    }
                                },
                            },
                            "400": {"description": "Тело ответа пустое."},
                            "500": {
                                "description": "внутренняя ошибка сервера.",
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/schemas/InternalErrorResponse"}
                                    }
                                },
                            },
                        },
                    }
                }
            },
            "components": {
                "schemas": {
                    "GetAllRecordsResponse": {
                        "type": "object",
                        "required": ["date", "classRecord", "groupRecords"],
                        "properties": {
                            "date": {"type": "string", "format": "date-time"},
                            "classRecord": {"$ref": "#/components/schemas/ClassRecord"},
                            "groupRecords": {"$ref": "#/components/schemas/GroupRecord"},
                        },
                    },
                    "ClassRecord": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"},
                            "name": {"type": "string"},
                            "revoked": {"type": "string", "format": "date-time"},
                            "pwrRead": {
                                "type": "string",
                                "description": "Право получения доверенности с данным полномочием",
                            },
                            "pwrIss": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "GroupRecord": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "format": "uuid"},
                            "parent": {"type": "string", "format": "uuid"},
                        },
                    },
                    "InternalErrorResponse": {
                        "type": "object",
                        "properties": {
                            "activityId": {"type": "string"},
                            "messageToUser": {"type": "string"},
                        },
                    },
                }
            },
        }
        finalize_spec(spec, source)
        body = spec["components"]["schemas"]["GetAllRecordsResponse"]
        self.assertIn("classRecord", body["properties"])
        self.assertNotIn("classRecords", body["properties"])
        self.assertEqual(body["properties"]["groupRecords"]["$ref"], "#/components/schemas/GroupRecord")

    def test_placeholder_table_keeps_method_and_status_catalog(self) -> None:
        from openapi_from_spec.pipeline import _read_yaml_payload, finalize_spec

        source = """
## 4.1.getToken - получить токен

| Тип запроса | Наименование метода | URL | Синхронность |
| --- | --- | --- | --- |
| POST | getToken | {apiUrl}/open-api-authentication/v1/token | Да |

## 4.12.getOpenClient - получить открытую информацию о клиенте

| Тип запроса | Наименование метода | URL | Синхронность |
| --- | --- | --- | --- |
| GET | getOpenClient | {apiUrl}/open-api-application/v2/applications/client/open | Да |

## 6.1.2.Расшифровка кодов ответа

| HTTPS Status Code | Описание |
| --- | --- |
| 200 | OK |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 503 | Service Unavailable |

GET https://open-api.example/open-api-application/v2/applications/client/open?applicationId=1
"""
        spec = {
            "paths": {
                "/open-api-application/v2/applications/client/open": {
                    "get": {
                        "operationId": "getOpen",
                        "summary": "4.11.updateClient - добавить данные",
                        "parameters": [
                            {
                                "name": "applicationId",
                                "in": "path",
                                "required": True,
                                "schema": {"type": "string"},
                            }
                        ],
                        "responses": {
                            "200": {"description": "Открытые данные"},
                            "500": {"$ref": "#/components/responses/InternalError"},
                        },
                    }
                }
            },
            "components": {
                "responses": {
                    "InternalError": {"description": "Внутренняя ошибка сервиса", "x-origin": "skill-template"}
                }
            },
        }
        finalize_spec(spec, source)
        self.assertEqual([item["url"] for item in spec["servers"]], ["{apiUrl}"])
        operation = spec["paths"]["/open-api-application/v2/applications/client/open"]["get"]
        self.assertEqual(operation["summary"], "4.11.updateClient - добавить данные")
        self.assertEqual(operation["parameters"][0]["in"], "query")
        self.assertNotIn("403", operation["responses"])
        self.assertNotIn("500", operation["responses"])
        document, _raw, errors = _read_yaml_payload(
            {"yaml": "openapi: 3.0.3\ninfo:\n  title: T\n  version: '1'\npaths: {}\ncomponents:\n  schemas:\n    Offer:\n      description: Статус предложения: значение справочника\n"}
        )
        self.assertIsNotNone(document)
        self.assertFalse(any("не разобран" in item for item in errors))

    def test_yes_no_header_and_empty_204_follow_the_document(self) -> None:
        from openapi_from_spec.pipeline import finalize_spec

        menu = """
POST /canteen/menu/items

| Name | Description | Type | Required |
| --- | --- | --- | --- |
| name | Название блюда | string | Yes |
| price | Цена порции | number | No |

{"name":"Борщ","price":120}
"""
        menu_spec = {
            "paths": {
                "/canteen/menu/items": {
                    "post": {
                        "responses": {"201": {"description": "Добавлено"}},
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["price"],
                                        "properties": {
                                            "name": {"type": "string"},
                                            "price": {"type": "number"},
                                        },
                                    }
                                }
                            }
                        },
                    }
                }
            }
        }
        finalize_spec(menu_spec, menu)
        body = menu_spec["paths"]["/canteen/menu/items"]["post"]["requestBody"]["content"]["application/json"]
        self.assertEqual(body["schema"]["required"], ["price"])
        self.assertEqual(body["schema"]["properties"]["name"]["description"], "Название блюда")

        parking = """
POST /parking/sessions
Поле amount вычисляется на сервере и не передаётся в запросах.
{"plate":"A123AA","zone":"center"}

DELETE /parking/sessions/{sessionId}
Закрывает сессию. Тело ответа пустое.
204 No Content
"""
        parking_spec = {
            "paths": {
                "/parking/sessions": {
                    "post": {
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "plate": {"type": "string"},
                                            "zone": {"type": "string"},
                                            "amount": {"type": "number"},
                                        },
                                    }
                                }
                            }
                        },
                        "responses": {"201": {"description": "Открыта"}},
                    }
                },
                "/parking/sessions/{sessionId}": {
                    "delete": {
                        "responses": {
                            "204": {
                                "description": "Закрыта",
                                "content": {"application/json": {"schema": {"type": "object"}}},
                            }
                        }
                    }
                },
            }
        }
        finalize_spec(parking_spec, parking)
        posted = parking_spec["paths"]["/parking/sessions"]["post"]["requestBody"]["content"]["application/json"]
        self.assertIn("amount", posted["schema"]["properties"])
        self.assertIn("content", parking_spec["paths"]["/parking/sessions/{sessionId}"]["delete"]["responses"]["204"])

        weather = """
GET https://meteo.example/v1/stations/{stationId}/observations
Заголовок X-Api-Key обязателен.
"""
        weather_spec = {
            "paths": {
                "/stations/{stationId}/observations": {
                    "get": {"responses": {"200": {"description": "Наблюдения"}}}
                }
            }
        }
        finalize_spec(weather_spec, weather)
        parameters = weather_spec["paths"]["/stations/{stationId}/observations"]["get"]["parameters"]
        self.assertEqual([item["name"] for item in parameters], ["stationId"])

    def test_catalog_formats_title_and_example_id_stay_with_their_owner(self) -> None:
        from openapi_from_spec.pipeline import finalize_spec

        clinic = """
# Запись к врачу

## Коды ответа

| Код ответа | Описание |
| --- | --- |
| 400 | Неверная дата приёма |
| 403 | Нет права на запись |
"""
        clinic_spec = {
            "info": {"title": "API", "version": "1.0.0"},
            "paths": {
                "/clinic/slots": {
                    "get": {
                        "responses": {
                            "400": {"$ref": "#/components/responses/BadRequest"},
                            "403": {"description": "Нет права на запись"},
                        }
                    }
                }
            },
            "components": {
                "responses": {"BadRequest": {"description": "Некорректный запрос", "x-origin": "skill-template"}}
            },
        }
        finalize_spec(clinic_spec, clinic)
        self.assertEqual(clinic_spec["info"]["title"], "Запись к врачу")
        self.assertEqual(
            clinic_spec["paths"]["/clinic/slots"]["get"]["responses"]["400"]["description"],
            "Неверная дата приёма",
        )
        self.assertEqual(
            clinic_spec["paths"]["/clinic/slots"]["get"]["responses"]["403"]["description"],
            "Нет права на запись",
        )

        meters = """
curl -X GET "<https://grid.example/meters/v2/readings?account=100>"
Параметр date необязательный. Поддерживаемые форматы параметра 'date':
- 'yyyy-MM-dd'
- 'yyyy-MM-ddTHH:mm:ssZ'
"""
        meters_spec = {
            "paths": {
                "/meters/v2/readings": {
                    "get": {
                        "parameters": [
                            {
                                "name": "account",
                                "in": "query",
                                "schema": {"type": "integer", "pattern": r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2}(\.\d{1,3})?)?Z?)?$"},
                                "description": "Идентификатор счёта Форматы: yyyy-MM-dd, yyyy-MM-ddTHH:mm:ssZ",
                            },
                            {"name": "date", "in": "query", "schema": {"type": "string", "format": "date-time"}},
                        ],
                        "responses": {"200": {"description": "Показания"}},
                    }
                }
            }
        }
        finalize_spec(meters_spec, meters)
        parameters = meters_spec["paths"]["/meters/v2/readings"]["get"]["parameters"]
        account = next(item for item in parameters if item["name"] == "account")
        date = next(item for item in parameters if item["name"] == "date")
        self.assertIn("pattern", account["schema"])
        self.assertEqual(date["schema"].get("format"), "date-time")

        library = {
            "info": {"title": "Книга не выдаётся", "version": "1.0.0"},
            "paths": {"/library/loans": {"post": {"responses": {"201": {"description": "Создано"}}}}},
        }
        finalize_spec(library, "# Выдача книг\n\nPOST /library/loans\n\n## Книга не выдаётся\n")
        self.assertEqual(library["info"]["title"], "Книга не выдаётся")

        archive = {
            "paths": {
                "/archive/cards/{cardId}": {
                    "get": {
                        "parameters": [{"name": "cardId", "in": "path", "required": True, "schema": {"type": "string"}}],
                        "responses": {"200": {"description": "Карточка"}},
                    }
                },
                "/archive/cards/{id}": {
                    "delete": {
                        "parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}],
                        "responses": {"204": {"description": "Удалено"}},
                    }
                },
            }
        }
        finalize_spec(archive, "GET /archive/cards/{cardId}\n\ncurl -X DELETE '<https://archive.example/archive/cards/15>'\n")
        self.assertNotIn("/archive/cards/{id}", archive["paths"])
        deleted = archive["paths"]["/archive/cards/{cardId}"]["delete"]
        self.assertEqual(deleted["parameters"][0]["name"], "cardId")
        self.assertIn("204", deleted["responses"])


class OllamaClientTests(unittest.TestCase):
    def test_local_chat_url_and_message_content(self) -> None:
        self.assertEqual(ollama_chat_url("http://127.0.0.1:11434/api"), "http://127.0.0.1:11434/api/chat")
        self.assertEqual(ollama_chat_url("http://127.0.0.1:11434"), "http://127.0.0.1:11434/api/chat")
        self.assertIsNone(ollama_chat_url("https://api.z.ai/api/paas/v4"))
        self.assertEqual(_chat_content({"message": {"role": "assistant", "content": "ready"}}), "ready")


class ApiTests(unittest.TestCase):
    def test_convert_over_http(self) -> None:
        server = make_server(Settings(title="Счета"), "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        try:
            page = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode("utf-8")
            self.assertIn("OpenAPI 3.0.3", page)
            self.assertIn("Приложить .md", page)
            self.assertIn('for="file"', page)
            self.assertIn('accept=".md,.markdown,.txt,text/markdown,text/plain"', page)
            sample = urllib.request.urlopen(f"http://127.0.0.1:{port}/sample").read().decode("utf-8")
            self.assertIn("accountBalances", sample)
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/convert",
                data=json.dumps({"text": SAMPLE, "title": "Счета"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            payload = json.loads(urllib.request.urlopen(request).read().decode("utf-8"))
            self.assertEqual(payload["errors"], [])
            self.assertEqual(yaml.safe_load(payload["yaml"])["paths"], {})
            self.assertIn("только модель", payload["report"]["gaps"][0]["question"])
            self.assertEqual(payload["report"]["model_calls"], 0)
        finally:
            server.shutdown()
            server.server_close()

    def test_multipart_returns_yaml_file_or_questions(self) -> None:
        server = make_server(Settings(title="Счета"), "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        try:
            ready = _post_file(port, "account-balances.txt", SAMPLE.encode("utf-8"), title="Счета")
            self.assertIn("application/json", ready.headers.get("Content-Type", ""))
            ready_payload = json.loads(ready.body.decode("utf-8"))
            self.assertEqual(ready_payload["result"], "need_info")
            self.assertIn("только модель", ready_payload["message"])

            questions = _post_file(port, "gap.txt", "Описание сервиса без методов.".encode("utf-8"))
            self.assertIn("application/json", questions.headers.get("Content-Type", ""))
            payload = json.loads(questions.body.decode("utf-8"))
            self.assertEqual(payload["result"], "need_info")
            self.assertNotIn("yaml", payload)
            self.assertTrue(payload["message"].startswith("Нужны уточнения по ТЗ:"))
            self.assertTrue(payload["questions"])

            with self.assertRaises(urllib.error.HTTPError) as caught:
                _post_file(port, "spec.pdf", b"%PDF")
            self.assertEqual(caught.exception.code, 400)
            caught.exception.close()
        finally:
            server.shutdown()
            server.server_close()


class _Response:
    def __init__(self, headers, body: bytes) -> None:
        self.headers = headers
        self.body = body


def _post_file(port: int, filename: str, payload: bytes, title: str = "") -> _Response:
    boundary = "----spec"
    parts = [
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            "Content-Type: text/plain\r\n\r\n"
        ).encode("utf-8")
        + payload
        + b"\r\n"
    ]
    if title:
        parts.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="title"\r\n\r\n{title}\r\n'
            ).encode("utf-8")
        )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/convert",
        data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request) as response:
        return _Response(response.headers, response.read())


class DifyClientTests(unittest.TestCase):
    def test_dify_chatflow_forwards_file_and_comment(self) -> None:
        chat = yaml.safe_load((ROOT / "dify" / "tz-to-openapi.yml").read_text(encoding="utf-8"))
        self.assertEqual(chat["version"], "0.6.0")
        self.assertEqual(chat["kind"], "app")
        self.assertEqual(chat["app"]["mode"], "advanced-chat")
        self.assertEqual(chat["app"]["name"], "ТЗ → OpenAPI")
        self.assertEqual(chat["workflow"]["conversation_variables"], [])
        self.assertEqual(chat["workflow"]["environment_variables"][0]["name"], "OPENAPI_SERVICE_URL")
        nodes = {node["id"]: node["data"] for node in chat["workflow"]["graph"]["nodes"]}
        self.assertEqual(set(nodes), {"start", "http", "answer"})
        fields = {item["key"]: item for item in nodes["http"]["body"]["data"]}
        self.assertEqual(fields["file"]["file"], ["sys", "files"])
        self.assertEqual(fields["clarification"]["value"], "{{#sys.query#}}")
        self.assertIn("{{#http.body#}}", nodes["answer"]["answer"])

    def test_base_url_gains_v1(self) -> None:
        self.assertEqual(dify_v1_base("https://dify.example"), "https://dify.example/v1")
        self.assertEqual(dify_v1_base("https://dify.example/v1/"), "https://dify.example/v1")
        self.assertEqual(
            dify_v1_base("https://dify.example/v1/chat-messages"),
            "https://dify.example/v1",
        )

    def test_dify_client_is_preferred_over_vllm(self) -> None:
        client = _client_from_settings(
            Settings(
                dify_base_url="https://dify.example",
                dify_api_key="app-key",
                vllm_base_url="http://127.0.0.1:11434/api",
                vllm_model="qwen",
            )
        )
        self.assertIsInstance(client, DifyClient)
        self.assertEqual(client.base_url, "https://dify.example/v1")

    def test_streaming_chat_does_not_send_model_name(self) -> None:
        seen: list[tuple[str, dict]] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or "0")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                seen.append((self.path, payload))
                query = payload.get("query")
                if query == "fail":
                    events = [json.dumps({"event": "error", "message": "сломанный узел"})]
                elif query == "yaml":
                    events = [json.dumps({"event": "message", "answer": "openapi: 3.0.3\n"})]
                else:
                    events = [
                        json.dumps({"event": "message", "answer": '{"ok":'}),
                        json.dumps({"event": "agent_message", "answer": " true}"}),
                    ]
                encoded = "".join(f"data: {event}\n\n" for event in events).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, fmt: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        try:
            client = DifyClient(f"http://127.0.0.1:{port}", "secret-key", timeout=10)
            self.assertEqual(
                client.complete_json(task="extract", system="system-skill", user="fragment"),
                {"ok": True},
            )
            path, payload = seen[0]
            self.assertEqual(path, "/v1/chat-messages")
            self.assertNotIn("model", payload)
            self.assertEqual(payload["inputs"], {"system": "system-skill"})
            self.assertEqual(payload["query"], "fragment")
            self.assertEqual(payload["response_mode"], "streaming")
            self.assertEqual(payload["conversation_id"], "")
            self.assertEqual(payload["user"], "openapi-from-spec")
            self.assertEqual(
                client.complete_json(task="openapi", system="system-skill", user="yaml"),
                {"yaml": "openapi: 3.0.3"},
            )
            with self.assertRaises(RuntimeError):
                client.complete_json(task="extract", system="system-skill", user="fail")
        finally:
            server.shutdown()
            server.server_close()


class RequestBoundaryTests(unittest.TestCase):
    def test_quote_past_the_next_method_keeps_two_operations(self) -> None:
        from openapi_from_spec.models import Operation
        from openapi_from_spec.naming import find_template_url
        from openapi_from_spec.pipeline import (
            _apply_template_url,
            _limit_to_own_request,
            _segment_with_model,
        )
        from openapi_from_spec.skill_loader import load_skill_markdown

        text = """# Док

## Первый

Создаёт запись.

POST https://example.test/v1/items

### Для запроса

| name | значение |

Полис не действует

## Второй

Возвращает запись.

GET https://example.test/v1/items/{itemId}
"""

        class _OneSegment:
            def complete_json(self, task: str, system: str, user: str) -> dict:
                return {
                    "segments": [
                        {
                            "start_quote": "## Первый",
                            "end_quote": "GET https://example.test/v1/items/{itemId}",
                            "title": "Первый",
                            "http_method": "GET",
                            "raw_name": "/v1/items/{itemId}",
                        }
                    ],
                    "continues_previous": None,
                }

        self.assertLess(_limit_to_own_request(text, 0, len(text)), text.find("GET https://"))
        self.assertEqual(find_template_url(text)[:1], ("post",))
        self.assertEqual(find_template_url(text)[2], "/v1/items")
        same_method = (
            "GET https://example.test/v1/items/5\n\nGET https://example.test/v1/items/{itemId}\n"
        )
        self.assertEqual(_limit_to_own_request(same_method, 0, len(same_method)), len(same_method))
        self.assertEqual(find_template_url(same_method)[2], "/v1/items/{itemId}")

        segments, failures = _segment_with_model(text, _OneSegment(), Settings())
        self.assertEqual(failures, [])
        self.assertEqual([segment.http_method for segment in segments], ["post", "get"])
        operations: list[Operation] = []
        for segment in segments:
            operation = Operation(
                http_method=segment.http_method,
                raw_name=segment.raw_name or "items",
                summary=segment.title,
                description="",
                fields=[],
            )
            _apply_template_url(operation, segment.text)
            operations.append(operation)
        spec, _conventions = compile_operations(operations, title="Док", version="1")
        paths = spec["paths"]
        self.assertIn("post", paths["/v1/items"])
        self.assertIn("get", paths["/v1/items/{itemId}"])
        self.assertFalse(any(str(path).endswith("-2") for path in paths))
        self.assertIn("Полис не действует", segments[0].text)
        self.assertNotIn("{itemId}", segments[0].text)
        skill = load_skill_markdown()
        self.assertIn("Продолжение обрывается перед следующим вызовом.", skill)
        self.assertNotIn("http://", skill)
        self.assertNotIn("https://", skill)

    def test_short_status_line_keeps_fields_from_the_fenced_example(self) -> None:
        from openapi_from_spec.pipeline import _bind_status_bodies

        source = """## Регистрация

POST https://example.test/v1/items

### Пример ответа

```json
{
  "claimId": "CL-100",
  "status": "registered",
  "settlementStamp": null
}
```

HTTP/1.1 201 Created
{"claimId": "CL-100", "status": "registered"}
"""
        spec = {
            "openapi": "3.0.3",
            "paths": {"/v1/items": {"post": {"responses": {"201": {"description": "Created"}}}}},
        }
        _bind_status_bodies(spec, source)
        media = spec["paths"]["/v1/items"]["post"]["responses"]["201"]["content"]["application/json"]
        self.assertIn("settlementStamp", media["example"])
        self.assertIsNone(media["example"]["settlementStamp"])
        self.assertIn("settlementStamp", media["schema"]["properties"])

    def test_gaps_are_filled_without_replacing_a_finished_phrase(self) -> None:
        import datetime

        from openapi_from_spec.pipeline import finalize_spec
        from openapi_from_spec.stitch import stitch_documents

        stitched, _notes = stitch_documents(
            [
                {
                    "paths": {
                        "/clinic/slots": {
                            "post": {
                                "requestBody": {
                                    "content": {
                                        "application/json": {
                                            "example": {"day": datetime.date(2026, 4, 1)}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            ],
            title="API",
            version="1.0.0",
        )
        self.assertEqual(
            stitched["paths"]["/clinic/slots"]["post"]["requestBody"]["content"]["application/json"]["example"]["day"],
            "2026-04-01",
        )

        kept = {
            "info": {"title": "Своё имя", "version": "9"},
            "paths": {
                "/archive/cards/{cardId}": {
                    "get": {"responses": {"200": {"description": "Карточка"}}},
                    "delete": {"responses": {"204": {"description": "Удаляет карточку"}}},
                }
            },
        }
        finalize_spec(kept, "# Архив документов\n\n| Версия |\n| --- |\n| 4 |\n")
        self.assertEqual(kept["info"]["title"], "Своё имя")
        self.assertEqual(kept["info"]["version"], "9")

        source = """# Складские перемещения

| Дата | Версия |
| --- | --- |
| 01.03.2026 | 4 |

POST https://example.test/v1/claims

HTTP/1.1 422 Unprocessable Entity
{"errorMessage": "Полис не действует"}

## 409 Conflict - перевозка уже создана.

```json
{"messageToUser": "Накладная уже есть"}
```

## Формат ответа

| Наименование | Обозначение |
| --- | --- |
| reading | Показание счётчика |
"""
        spec = {
            "info": {"title": "API", "version": "1.0.0"},
            "paths": {
                "/archive/cards/{cardId}": {
                    "get": {"responses": {"200": {"description": "Карточка"}}}
                },
                "/archive/cards/{id}": {
                    "delete": {
                        "parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}],
                        "responses": {"204": {"description": "Удаляет карточку"}},
                    }
                },
                "/v1/claims": {
                    "post": {
                        "responses": {
                            "422": {"description": "Unprocessable Entity"},
                            "409": {"description": "перевозка уже создана."},
                        }
                    }
                },
                "/meters/v2/readings": {
                    "get": {
                        "responses": {
                            "200": {
                                "description": "Показания",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "readings": {"type": "array", "items": {"type": "object"}},
                                                "name": {"type": "string", "description": "уже есть"},
                                            },
                                        }
                                    }
                                },
                            }
                        }
                    }
                },
            },
        }
        finalize_spec(spec, source)
        self.assertEqual(spec["info"]["title"], "Складские перемещения")
        self.assertEqual(spec["info"]["version"], "4")
        self.assertNotIn("/archive/cards/{id}", spec["paths"])
        deleted = spec["paths"]["/archive/cards/{cardId}"]["delete"]
        self.assertEqual(deleted["parameters"][0]["name"], "cardId")
        claim = spec["paths"]["/v1/claims"]["post"]["responses"]
        self.assertEqual(claim["422"]["description"], "Полис не действует")
        self.assertEqual(claim["409"]["description"], "перевозка уже создана.")
        self.assertEqual(claim["409"]["content"]["application/json"]["example"]["messageToUser"], "Накладная уже есть")
        props = spec["paths"]["/meters/v2/readings"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]["properties"]
        self.assertEqual(props["readings"]["description"], "Показание счётчика")
        self.assertEqual(props["name"]["description"], "уже есть")

    def test_fragment_example_fills_a_gap_and_keeps_a_finished_phrase(self) -> None:
        from openapi_from_spec.pipeline import finalize_spec

        source = """
POST /api/v1/users HTTP/1.1
{"last_name":"Сидоров"}

Пример ответа (201 Created)
{"id": 3}

curl -X GET "https://notes.example/api/v1/notes" \\
     -H "Authorization: Bearer TOKEN"

HTTP/1.1 200 OK
{"items": [1]}
"""
        spec = {
            "servers": [{"url": "https://notes.example"}],
            "paths": {
                "/users": {
                    "post": {
                        "responses": {
                            "201": {"description": "Пользователь создан"},
                            "401": {"$ref": "#/components/responses/Unauthorized"},
                        }
                    }
                },
                "/notes": {
                    "get": {
                        "responses": {
                            "200": {
                                "description": "OK",
                                "content": {"application/json": {"schema": {"type": "object"}}},
                            },
                            "404": {"$ref": "#/components/responses/NotFound"},
                        }
                    }
                },
            },
            "components": {
                "responses": {
                    "Unauthorized": {"description": "Требуется аутентификация", "x-origin": "skill-template"},
                    "NotFound": {"description": "Ресурс не найден", "x-origin": "skill-template"},
                }
            },
        }
        finalize_spec(spec, source)
        created = spec["paths"]["/api/v1/users"]["post"]
        self.assertEqual(created["responses"]["201"]["description"], "Пользователь создан")
        self.assertEqual(created["responses"]["201"]["content"]["application/json"]["example"], {"id": 3})
        self.assertEqual(created["requestBody"]["content"]["application/json"]["example"], {"last_name": "Сидоров"})
        self.assertNotIn("401", created["responses"])
        notes = spec["paths"]["/api/v1/notes"]["get"]
        self.assertEqual(notes["responses"]["200"]["content"]["application/json"]["example"], {"items": [1]})
        self.assertNotIn("404", notes["responses"])
        self.assertIn("Authorization", {item["name"] for item in notes["parameters"]})

    def test_one_call_keeps_a_single_path_and_its_statuses(self) -> None:
        from openapi_from_spec.pipeline import _OWN_REQUEST, _fragment_for_request, finalize_spec

        source = """
GET https://kpmp.nalog.gov.ru/authority-classifier-idmz/api/v1/GetAllRecords

400 Bad Request - неверный формат параметра date. Тело ответа пустое.
"""
        spec = {
            "servers": [
                {"url": "https://kpmp.nalog.gov.ru/authority-classifier-idmz/api/v1"},
                {"url": "https://kpmp.nalog.gov.ru/authority-classifier-idmz"},
            ],
            "paths": {
                "/GetAllRecords": {"get": {"responses": {"200": {"description": "Набор записей"}}}},
                "/api/v1/GetAllRecords": {
                    "get": {
                        "responses": {
                            "200": {"description": "Набор записей"},
                            "400": {"description": "неверный формат параметра date"},
                        }
                    }
                },
            },
        }
        finalize_spec(spec, source)
        self.assertEqual(len(spec["paths"]), 1)
        operation = next(iter(spec["paths"].values()))["get"]
        self.assertIn("200", operation["responses"])
        self.assertIn("400", operation["responses"])
        self.assertEqual(len(spec["servers"]), 1)
        server_path = spec["servers"][0]["url"].split("://", 1)[1].split("/", 1)[-1]
        path = next(iter(spec["paths"]))
        joined = "/" + server_path.strip("/") + "/" + path.lstrip("/")
        self.assertNotIn("/api/v1/api/v1", joined)
        self.assertTrue(joined.endswith("/authority-classifier-idmz/api/v1/GetAllRecords"))

        listed = {
            "servers": [{"url": "https://bank.example/api/v1"}],
            "paths": {
                "/transactions": {"get": {"responses": {"200": {"description": "Список"}}}},
                "/transactions/{id}": {"put": {"responses": {"422": {"description": "Сумма должна быть больше 0"}}}},
            },
        }
        finalize_spec(
            listed,
            "GET /api/v1/transactions\n\nPUT /api/v1/transactions/1\n\n{\"error\":\"Сумма должна быть больше 0\"}\n",
        )
        self.assertEqual(set(listed["paths"]), {"/transactions", "/transactions/{id}"})

        text = (ROOT / "docling" / "habr-shablon-tz-rest-api.md").read_text(encoding="utf-8")
        match = next(item for item in _OWN_REQUEST.finditer(text) if "transactions/1" in item.group(0))
        start, end = _fragment_for_request(text, match)
        chunk = text[start:end]
        self.assertIn("transactions/1", chunk)
        self.assertIn("Сумма должна быть больше 0", chunk)
        self.assertNotIn("POST /api/v1/users", chunk)

    def test_table_call_keeps_one_path_and_document_version(self) -> None:
        from openapi_from_spec.pipeline import finalize_spec

        source = """
## OpenAPI 2.0: Взаимодействие с системой «ВсегдаДа»

v 10.5.0

## История изменений

| Версия | Дата |
| --- | --- |
| 0.1 | 04.06.2020 |

## 4.1.getToken - получить токен

| Тип запроса | Наименование метода | URL | Синхронность |
| --- | --- | --- | --- |
| POST | getToken | {apiUrl}/open-api-authentication/v1/token | Да |

## 4.9.putBasket - отправить корзину

| Тип запроса | Наименование метода | URL | Синхронность |
| --- | --- | --- | --- |
| PUT | putBasket | {apiUrl}/open-api-application/v2/applications/{appId}/bas- ket | Да |

PUT {apiUrl}/open-api-application/v2/applications/edd5b4df-eb28-483e-b33a-9ec4dafe41ef/basket
"""
        spec = {
            "openapi": "3.0.3",
            "info": {"title": "OpenAPI 2.0: Взаимодействие с системой «ВсегдаДа»", "version": "0.1"},
            "servers": [
                {"url": "{apiUrl}", "variables": {"apiUrl": {"default": "/"}}},
                {"url": "{apiURL}", "variables": {"apiURL": {"default": "/"}}},
                {"url": "https://open-api-application-v2.test02.rancher-test.vsegda.da/open-api-application/v2"},
            ],
            "paths": {
                "/getToken": {"post": {"operationId": "getToken", "responses": {"200": {"description": "Токен"}}}},
                "/open-api-authentication/v1/token": {
                    "post": {"operationId": "postToken", "responses": {"200": {"description": "Токен доступа"}}}
                },
                "/open-api-application/v2/applications/edd5b4df-eb28-483e-b33a-9ec4dafe41ef/basket": {
                    "put": {"responses": {"200": {"description": "Корзина"}}}
                },
                "/open-api-application/v2/applications/d1341eef-98e9-4fb1-9eb4-13f048f40035/status](https:/open-api-application-v2.test02.rancher-test.vsegda.da/open-api-application/v2/applications/{applicationId}/status": {
                    "get": {"responses": {"200": {"description": "Статус"}}}
                },
            },
        }
        finalize_spec(spec, source)
        self.assertEqual(spec["info"]["version"], "10.5.0")
        self.assertEqual([item["url"] for item in spec["servers"]], ["{apiUrl}"])
        self.assertNotIn("/getToken", spec["paths"])
        self.assertIn("/open-api-authentication/v1/token", spec["paths"])
        basket = next(path for path in spec["paths"] if path.endswith("/basket"))
        self.assertNotIn("edd5b4df", basket)
        self.assertIn("{", basket)
        status = next(path for path in spec["paths"] if path.endswith("/status"))
        self.assertNotIn("](", status)
        self.assertIn("{applicationId}", status)

    def test_wrapped_path_alias_and_example_version_stay_one_call(self) -> None:
        from openapi_from_spec.pipeline import _table_request_rows, finalize_spec

        source = """
## 4.8.getUrl - ссылка на заявку

| Тип запроса | Наименование метода | URL | Синхронность |
| --- | --- | --- | --- |
| GET | getUrl | {apiUrl}/open-api-application/v2/applications/{appId}/url | Да |

## 4.26.cancel - отменить заявку

| Тип запроса | Наименование метода | URL | Синхронность |
| --- | --- | --- | --- |
| POST | cancel | {apiUrl}/open-api-application/v2/applications/{appId}/cancel | Да |

POST {apiUrl}/open-api-application/v1/applications/04355f53-bb60-43e8-aadd-beced847fb55/cancel

## 4.38.getCreditLineLimitByCustomer

| Тип запроса | Наименование метода | URL | Синхронность |
| --- | --- | --- | --- |
| GET | getCreditLineLimitByCustomer | {apiUrl}/open-api-application/v2/ credit- line/{customerId}/limit-info | SYNC |
| GET | getCreditLineLimitByShortIdent | {apiUrl}/open-api-application/v2/credit- line/limit-info | SYNC |

## Заказы двух версий

| Тип запроса | Наименование метода | URL | Синхронность |
| --- | --- | --- | --- |
| GET | listV1 | {apiUrl}/v1/orders/{id} | Да |
| GET | listV2 | {apiUrl}/v2/orders/{id} | Да |
"""
        rows = [(method, path) for _at, method, path in _table_request_rows(source)]
        self.assertIn(("get", "/open-api-application/v2/credit-line/{customerId}/limit-info"), rows)
        self.assertIn(("get", "/open-api-application/v2/credit-line/limit-info"), rows)
        self.assertIn(("post", "/open-api-application/v2/applications/{appId}/cancel"), rows)
        spec = {
            "openapi": "3.0.3",
            "info": {"title": "ВсегдаДа", "version": "10.5.0"},
            "servers": [{"url": "{apiUrl}"}],
            "paths": {
                "/open-api-application/v2/applications/{appId}/url": {
                    "get": {
                        "parameters": [
                            {
                                "name": "appId",
                                "in": "path",
                                "required": True,
                                "description": "ИД заявки",
                                "schema": {"type": "string"},
                            },
                            {
                                "name": "applicationId",
                                "in": "path",
                                "required": True,
                                "description": "ИД заявки ВсегдаДа",
                                "schema": {"type": "string"},
                            },
                        ],
                        "responses": {"200": {"description": "Ссылка"}},
                    }
                },
                "/open-api-application/v2/applications/{appId}/cancel": {
                    "post": {"responses": {"200": {"description": "Отменена"}}}
                },
                "/open-api-application/v1/applications/04355f53-bb60-43e8-aadd-beced847fb55/cancel": {
                    "post": {"responses": {"200": {"description": "Отменена примером"}}}
                },
                "/v1/orders/{id}": {"get": {"responses": {"200": {"description": "Версия 1"}}}},
                "/v2/orders/{id}": {"get": {"responses": {"200": {"description": "Версия 2"}}}},
            },
        }
        finalize_spec(spec, source)
        url = spec["paths"]["/open-api-application/v2/applications/{appId}/url"]["get"]
        self.assertEqual([item["name"] for item in url["parameters"] if item["in"] == "path"], ["appId"])
        self.assertNotIn("/open-api-application/v1/applications/04355f53-bb60-43e8-aadd-beced847fb55/cancel", spec["paths"])
        self.assertIn("/open-api-application/v2/applications/{appId}/cancel", spec["paths"])
        self.assertIn("/v1/orders/{id}", spec["paths"])
        self.assertIn("/v2/orders/{id}", spec["paths"])

    def test_long_tail_version_server_and_success_code(self) -> None:
        from openapi_from_spec.compile import compile_operations
        from openapi_from_spec.llm import apply_model_extract
        from openapi_from_spec.models import IRField, Operation, Segment
        from openapi_from_spec.pipeline import _absorb_continuations, finalize_spec

        source = """# Портовый реестр

v 6

Короткое введение.

## История изменений

| Версия | Дата |
| --- | --- |
| 0.1 | 01.02.2020 |

POST {apiUrl}/harbor/v1/calls
"""
        spec = {
            "openapi": "3.0.3",
            "info": {"title": "Портовый реестр", "version": "0.1"},
            "paths": {"/harbor/v1/calls": {"post": {"responses": {"201": {"description": "Создано"}}}}},
        }
        finalize_spec(spec, source)
        self.assertEqual(spec["info"]["version"], "6")
        self.assertEqual([item["url"] for item in spec["servers"]], ["{apiUrl}"])

        operation = Operation(
            http_method="post",
            raw_name="calls",
            summary="регистрация",
            description="",
            fields=[IRField(name="berthSeal", location="response", tz_type="string")],
            explicit_path="/harbor/v1/calls",
            status_responses={"201": "Карточка создана", "409": "Причал занят"},
        )
        compiled, _notes = compile_operations([operation], title="Портовый реестр", version="6", error_templates=True)
        responses = compiled["paths"]["/harbor/v1/calls"]["post"]["responses"]
        self.assertIn("201", responses)
        self.assertNotIn("200", responses)
        self.assertNotIn("500", responses)
        self.assertIn("schema", responses["201"]["content"]["application/json"])

        only_ok = Operation(
            http_method="get",
            raw_name="contracts",
            summary="договор",
            description="",
            fields=[IRField(name="handoverSeal", location="response", tz_type="string")],
            explicit_path="/lease/v1/contracts/{contractId}",
            status_responses={"200": "Карточка договора"},
        )
        listed, _notes = compile_operations([only_ok], title="Договоры", version="6", error_templates=True)
        codes = set(listed["paths"]["/lease/v1/contracts/{contractId}"]["get"]["responses"])
        self.assertEqual(codes, {"200"})

        line = "POST {apiUrl}/lab/v1/samples\n"
        text = line + "\n| sampleBarcode | Штрихкод | string | Да |\n"
        head = Segment(start=0, end=len(line), text=line, http_method="post", explicit_path="/lab/v1/samples")
        absorbed = _absorb_continuations(text, [head])
        self.assertEqual(len(absorbed), 1)
        self.assertIn("sampleBarcode", absorbed[0].text)

        placed = Operation(
            http_method="post",
            raw_name="calls",
            summary="",
            description="",
            fields=[],
            explicit_path="/harbor/v1/calls",
        )
        apply_model_extract(
            placed,
            {"parameters": [{"name": "vesselName", "location": "query", "tz_type": "string", "quote": "vesselName"}]},
            "| vesselName | Название судна | string | Да |",
        )
        self.assertEqual(placed.fields[0].location, "body")

        from openapi_from_spec.pipeline import _bind_fragment_request

        museum = Operation(
            http_method="get",
            raw_name="{gateway}/museum/v1/exhibits/{exhibitId}",
            summary="",
            description="",
            fields=[],
        )
        fragment = (
            "GET {gateway}/museum/v1/exhibits/{exhibitId}\n"
            "GET https://museum.example.test/museum/v1/exhibits/edd5b4df-eb28-483e-b33a-9ec4dafe41ef\n"
        )
        _bind_fragment_request(museum, fragment)
        self.assertEqual(museum.explicit_path, "/museum/v1/exhibits/{exhibitId}")
        self.assertIsNone(museum.server_url)

        gotten = Operation(
            http_method="get",
            raw_name="handoverSeal",
            summary="",
            description="",
            fields=[IRField(name="handoverSeal", location="body", tz_type="string", quote="handoverSeal")],
        )
        apply_model_extract(gotten, {"parameters": []}, "| handoverSeal | Печать передачи | string | Да |")
        gotten.fields.append(IRField(name="handoverSeal", location="body", tz_type="string", quote="handoverSeal"))
        from openapi_from_spec.llm import _place_unmarked_request_fields

        _place_unmarked_request_fields(gotten, "| handoverSeal | Печать передачи | string | Да |")
        self.assertEqual(gotten.fields[-1].location, "response")


if __name__ == "__main__":
    unittest.main()
