from __future__ import annotations

OPENAPI_VERSION = "3.0.3"
DEFAULT_PATH_PREFIX = "/api/v1"

# Ключ — как в ТЗ, в нижнем регистре. Компилятор не угадывает тип вне этой таблицы.
TYPE_MAP: dict[str, tuple[str, str | None]] = {
    "string": ("string", None),
    "text": ("string", None),
    "строка": ("string", None),
    "string (url)": ("string", "uri"),
    "string (timestamp)": ("string", "date-time"),
    "<object>": ("object", None),
    "array [<objects>]": ("array", None),
    "uuid": ("string", "uuid"),
    "date": ("string", "date"),
    "дата": ("string", "date"),
    "datetime": ("string", "date-time"),
    "date-time": ("string", "date-time"),
    "дата и время": ("string", "date-time"),
    "boolean": ("boolean", None),
    "bool": ("boolean", None),
    "логический": ("boolean", None),
    "булево": ("boolean", None),
    "integer": ("integer", None),
    "int": ("integer", None),
    "int32": ("integer", "int32"),
    "целое": ("integer", None),
    "long": ("integer", "int64"),
    "int64": ("integer", "int64"),
    "number": ("number", None),
    "decimal": ("number", None),
    "double": ("number", "double"),
    "float": ("number", "float"),
    "число": ("number", None),
    "object": ("object", None),
    "объект": ("object", None),
    "array": ("array", None),
    "list": ("array", None),
    "set": ("array", None),
    "массив": ("array", None),
    "список": ("array", None),
    "file": ("string", "binary"),
    "binary": ("string", "binary"),
    "файл": ("string", "binary"),
}

ERROR_RESPONSES: dict[str, str] = {
    "400": "Некорректный запрос",
    "401": "Требуется аутентификация",
    "404": "Ресурс не найден",
    "500": "Внутренняя ошибка сервиса",
}

ERROR_SCHEMA_NAME = "ErrorResponse"
