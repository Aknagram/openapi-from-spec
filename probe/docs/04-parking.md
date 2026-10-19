# Городская парковка

POST /parking/sessions

Открывает парковочную сессию. Поле amount вычисляется на сервере и не передаётся в запросах.

| Наименование | Обозначение | Формат | Обязательность |
| --- | --- | --- | --- |
| plate | Номер автомобиля | string | Да |
| zone | Зона парковки | string | Да |
| amount | Сумма к оплате | number | Нет |

Пример запроса:

```json
{"plate":"A123AA","zone":"center"}
```

Пример ответа (201 Created)

```json
{"sessionId":"s-1","plate":"A123AA"}
```

DELETE /parking/sessions/{sessionId}

Закрывает сессию. Тело ответа пустое.

204 No Content
