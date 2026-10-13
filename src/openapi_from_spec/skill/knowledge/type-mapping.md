# Типы ТЗ, которые понимает компилятор

Модель копирует тип как в таблице. Ниже перевод, который делает код. Неизвестное слово типом OpenAPI не становится.

| Как в ТЗ | OpenAPI |
| --- | --- |
| String, Text, Строка | string |
| String (url) | string, format uri |
| String (timestamp) | string, format date-time |
| UUID | string, format uuid |
| Date, Дата | string, format date |
| DateTime, Date-Time, Дата и время | string, format date-time |
| Boolean, Bool, Логический, Булево | boolean |
| Integer, Int, Целое | integer |
| Int32 | integer, format int32 |
| Long, Int64 | integer, format int64 |
| Number, Decimal, Число | number |
| Double | number, format double |
| Float | number, format float |
| Object, Объект | object |
| Array, List, Set, Массив, Список | array |
| File, Binary, Файл | string, format binary |

Колонка «Максимальная длина» у строки — `maxLength`. У целого — `x-tz-max-length`. У списка строк — `maxLength` элемента, если пример показывает строки.

Обязательность: «Да», «М», «+» — required true; «Нет», «Н» — false. Пустая ячейка — вопрос, не догадка. Латинская «O» в чужих таблицах означает optional и даёт false; кириллическая «О» означает обязательный и даёт true.
