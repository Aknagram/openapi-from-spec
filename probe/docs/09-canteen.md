# Меню столовой

GET /canteen/menu

Возвращает блюда, которые подают сегодня.

| Name | Description | Type | Required |
| --- | --- | --- | --- |
| name | Название блюда | string | Yes |
| price | Цена порции | number | Yes |
| spicy | Признак острого блюда | boolean | No |

Пример ответа (200 OK)

```json
{"items":[{"name":"Борщ","price":120,"spicy":false}]}
```

POST /canteen/menu/items

Добавляет блюдо в меню на сегодня.

| Name | Description | Type | Required |
| --- | --- | --- | --- |
| name | Название блюда | string | Yes |
| price | Цена порции | number | Yes |

Пример запроса:

```json
{"name":"Борщ","price":120}
```

Пример ответа (201 Created)

```json
{"name":"Борщ","price":120}
```
