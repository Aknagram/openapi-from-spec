from __future__ import annotations

import json
import threading
import unittest
import urllib.request
from pathlib import Path

import yaml

from openapi_from_spec.api import make_server
from openapi_from_spec.llm import _chat_content, ollama_chat_url
from openapi_from_spec.pipeline import Settings, convert_text
from openapi_from_spec.segment import normalize, pack_windows, render_window
from openapi_from_spec.skill_loader import load_skill_markdown, skill_dir, strip_frontmatter, system_prompt
from openapi_from_spec.stitch import stitch_documents

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = (ROOT / "examples" / "account-balances.txt").read_text(encoding="utf-8")


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
        result = convert_text(SAMPLE, Settings(title="Счета"), None)
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
        result = convert_text(SAMPLE + extra, Settings(), None)
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
        result = convert_text(text, Settings(), None)
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
        result = convert_text("\n".join(blocks), Settings(), None)
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
        result = convert_text(text, Settings(), None)
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
        self.assertIn("errorMessage", prompt)
        self.assertIn("Для тела ответа", prompt)
        self.assertIn("camelCase", prompt)
        self.assertIn("getRecipesRecipeId", prompt)
        self.assertIn("есть `items`", strip_frontmatter(text))
        self.assertIn("не слово «Успех»", strip_frontmatter(text))
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
        self.assertEqual(spec["servers"], [{"url": "https://XXXXX/api/v1"}])
        self.assertEqual(set(spec["paths"]), {"/currencies", "/users", "/transactions/{id}"})
        self.assertEqual(set(spec["paths"]["/users"]), {"post"})
        self.assertEqual(spec["paths"]["/users"]["post"]["operationId"], "postUsers")
        self.assertNotIn("*", spec["paths"]["/users"]["post"]["description"])
        self.assertNotIn("required", spec["components"]["schemas"]["Currency"])
        user = spec["components"]["schemas"]["CreateUserRequest"]
        self.assertEqual(user["required"], ["last_name"])
        self.assertEqual(user["properties"]["status"]["type"], "integer")
        transaction = spec["paths"]["/transactions/{id}"]["put"]
        self.assertEqual(transaction["parameters"][0]["name"], "id")
        self.assertEqual(transaction["responses"]["422"]["description"], "Сумма должна быть больше 0")
        updated = spec["components"]["schemas"]["UpdateTransactionRequest"]
        self.assertIn("description", updated["properties"])
        self.assertNotIn("required", updated)

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
        self.assertEqual(spec["servers"], [{"url": "https://meowgram.ru/api/v1"}])
        self.assertEqual(
            spec["paths"]["/cats/{catId}/posts"]["get"]["responses"]["401"]["description"],
            "Требуется аутентификация",
        )

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
        self.assertEqual(
            body["properties"]["classRecord"],
            {"$ref": "#/components/schemas/ClassRecord"},
        )
        self.assertEqual(
            body["properties"]["groupRecords"],
            {"$ref": "#/components/schemas/GroupRecord"},
        )
        self.assertNotIn("classRecords", body["properties"])
        self.assertEqual(body["required"], ["classRecord", "groupRecords"])
        media = spec["paths"]["/api/v1/GetAllRecords"]["get"]["responses"]["200"]["content"]["application/json"]
        self.assertNotIn("example", media)
        responses = spec["paths"]["/api/v1/GetAllRecords"]["get"]["responses"]
        self.assertNotIn("401", responses)
        self.assertNotIn("404", responses)
        self.assertIn("400", responses)
        self.assertIn("500", responses)
        self.assertNotIn("responses", spec["components"])

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
        self.assertEqual(spec["servers"][0]["url"], "{apiUrl}")
        self.assertIn("post", spec["paths"]["/open-api-authentication/v1/token"])
        self.assertEqual(
            spec["paths"]["/open-api-authentication/v1/token"]["post"]["summary"],
            "получить токен",
        )
        operation = spec["paths"]["/open-api-application/v2/applications/client/open"]["get"]
        self.assertEqual(operation["summary"], "получить открытую информацию о клиенте")
        self.assertEqual(operation["parameters"][0]["in"], "query")
        self.assertEqual(operation["responses"]["403"]["description"], "Forbidden")
        self.assertEqual(operation["responses"]["503"]["description"], "Service Unavailable")
        self.assertNotIn("500", operation["responses"])
        document, _raw, errors = _read_yaml_payload(
            {"yaml": "openapi: 3.0.3\ninfo:\n  title: T\n  version: '1'\npaths: {}\ncomponents:\n  schemas:\n    Offer:\n      description: Статус предложения: значение справочника\n"}
        )
        self.assertIsNotNone(document)
        self.assertFalse(any("не разобран" in item for item in errors))

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
            self.assertIn("/api/v1/account-balances", payload["yaml"])
            self.assertEqual(payload["report"]["model_calls"], 0)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
