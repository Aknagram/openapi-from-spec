# Наблюдения метеостанций

GET https://meteo.example/v1/stations

Возвращает список станций региона.

```json
{"stations":[{"stationId":"msk-1","name":"Центр"}]}
```

GET https://meteo.example/v1/stations/{stationId}/observations

Возвращает наблюдения одной станции. Заголовок X-Api-Key обязателен.

HTTP/1.1 200 OK

```json
{"observations":[{"tempC":-3}]}
```

HTTP/1.1 401 Unauthorized

```json
{"error":"Signature has expired","messageOutput":"Нужен ключ станции"}
```
