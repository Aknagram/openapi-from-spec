"""Три длинных ТЗ. Каждый метод длиннее окна сборки, хвост таблицы стоит после 12000 символов."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "probe" / "docs"
TARGET = 15500


def table(rows: list[tuple[str, str, str, str, str]]) -> str:
    lines = [
        "| Наименование | Обозначение | Формат | Обязательность | Примечание |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, title, fmt, required, note in rows:
        lines.append(f"| {name} | {title} | {fmt} | {required} | {note} |")
    return "\n".join(lines)


def rules(topic: str, count: int) -> str:
    paragraphs = []
    for number in range(1, count + 1):
        paragraphs.append(
            f"{number}. При обработке «{topic}» правило {number} применяется целиком. "
            f"Оператор сверяет пакет документов, фиксирует расхождение в журнале смены "
            f"и не подменяет значение соседним полем. Повтор той же проверки в следующем окне "
            f"не создаёт новую операцию: это продолжение того же метода, пункт {number}."
        )
    return "\n\n".join(paragraphs)


def method(
    heading: str,
    verb: str,
    url: str,
    summary: str,
    request_rows: list[tuple[str, str, str, str, str]],
    response_rows: list[tuple[str, str, str, str, str]],
    example: str,
    statuses: str,
    topic: str,
) -> str:
    body = [
        f"## {heading}",
        "",
        summary,
        "",
        f"{verb} {url}",
        "",
        "### Параметры запроса",
        "",
        "Параметры пути и строки перечислены в адресе метода. Отдельных query-полей сверх адреса нет.",
        "",
        "### Для запроса",
        "",
        table(request_rows),
        "",
        "### Правила метода",
        "",
        rules(topic, 8),
        "",
    ]
    text = "\n".join(body)
    extra = 1
    while len(text) < TARGET:
        text += (
            f"\n\n{extra + 8}. Дополнительная проверка «{topic}», лист {extra}. "
            "Сотрудник сравнивает оригинал и скан, записывает расхождение отдельной строкой журнала "
            "и оставляет метод тем же: новая строка запроса здесь не начинается."
        )
        extra += 1
    tail = "\n".join(
        [
            "",
            "### Для тела ответа",
            "",
            table(response_rows),
            "",
            "### Пример ответа",
            "",
            "```json",
            example,
            "```",
            "",
            statuses.strip(),
            "",
        ]
    )
    return text + "\n" + tail


def insurance() -> str:
    request = [
        ("policyNumber", "Номер полиса", "string", "Да", "Номер договора страхования, как в полисе."),
        ("insuredName", "Страхователь", "string", "Да", "ФИО страхователя полностью."),
        ("lossDate", "Дата события", "string", "Да", "Календарная дата страхового случая."),
        ("lossAddress", "Адрес события", "string", "Да", "Адрес, где зафиксирован ущерб."),
        ("damageKind", "Вид ущерба", "string", "Да", "Краткий код вида повреждения."),
        ("vehicleVin", "VIN", "string", "Нет", "Идентификатор транспортного средства, если случай автомобильный."),
        ("bankAccount", "Счёт выплаты", "string", "Да", "Расчётный счёт для перечисления."),
        ("contactPhone", "Телефон", "string", "Да", "Телефон для связи по заявлению."),
    ]
    response = [
        ("claimId", "Номер заявления", "string", "Да", "Идентификатор, присвоенный при регистрации."),
        ("status", "Статус", "string", "Да", "Текущий статус рассмотрения."),
        ("registeredAt", "Момент регистрации", "string", "Да", "Дата и время приёма заявления."),
        ("settlementStamp", "Отметка урегулирования", "string", "Нет", "Служебная отметка конца карточки заявления."),
    ]
    example = (
        '{\n  "claimId": "CL-100",\n  "status": "registered",\n'
        '  "registeredAt": "2026-04-02T10:00:00Z",\n  "settlementStamp": null\n}'
    )
    statuses = """
HTTP/1.1 201 Created
{"claimId": "CL-100", "status": "registered"}

HTTP/1.1 422 Unprocessable Entity
{"errorMessage": "Полис не действует на дату события"}
"""
    create = method(
        "Регистрация страхового заявления",
        "POST",
        "https://insurance.example/v1/claims",
        "Регистрирует заявление о страховом случае и возвращает номер рассмотрения.",
        request,
        response,
        example,
        statuses,
        "страховое заявление",
    )
    get_request = [
        ("view", "Режим карточки", "string", "Нет", "Если указан full, ответ включает журнал."),
    ]
    get_response = [
        ("claimId", "Номер заявления", "string", "Да", "Тот же номер, что при регистрации."),
        ("status", "Статус", "string", "Да", "Статус на момент запроса."),
        ("payoutLedger", "Журнал выплаты", "string", "Нет", "Последняя строка журнала выплаты в конце карточки."),
    ]
    read = method(
        "Карточка страхового заявления",
        "GET",
        "https://insurance.example/v1/claims/{claimId}",
        "Возвращает карточку заявления со статусом и журналом выплаты.",
        get_request,
        get_response,
        '{\n  "claimId": "CL-100",\n  "status": "registered",\n  "payoutLedger": "ожидает счёт"\n}',
        'HTTP/1.1 404 Not Found\n{"errorMessage": "Заявление не найдено"}',
        "карточка заявления",
    )
    return (
        "# Страховые заявления\n\n"
        "Сервис принимает и показывает заявления по полисам.\n\n"
        "## История изменений\n\n"
        "| Дата | Автор | Версия | Изменения |\n| --- | --- | --- | --- |\n"
        "| 02.04.2026 | Орлова | 3 | Длинные карточки заявления |\n\n"
        + create
        + "\n"
        + read
    )


def freight() -> str:
    request = [
        ("cargoName", "Наименование груза", "string", "Да", "Как в транспортной накладной."),
        ("weightKg", "Масса", "number", "Да", "Масса брутто в килограммах."),
        ("places", "Число мест", "integer", "Да", "Количество грузовых мест."),
        ("senderName", "Отправитель", "string", "Да", "Название организации отправителя."),
        ("receiverName", "Получатель", "string", "Да", "Название организации получателя."),
        ("originCity", "Город отправления", "string", "Да", "Город склада погрузки."),
        ("destinationCity", "Город назначения", "string", "Да", "Город склада выгрузки."),
        ("temperatureMode", "Температурный режим", "string", "Нет", "Пусто, если режим обычный."),
    ]
    response = [
        ("shipmentId", "Номер перевозки", "string", "Да", "Идентификатор созданной перевозки."),
        ("routeCode", "Код маршрута", "string", "Да", "Присвоенный код плеча."),
        ("dispatchStamp", "Отметка отгрузки", "string", "Нет", "Хвостовая отметка диспетчера."),
    ]
    create = method(
        "Создание перевозки",
        "POST",
        "https://freight.example/v1/shipments",
        "Создаёт перевозку и возвращает номер маршрута.",
        request,
        response,
        '{\n  "shipmentId": "SH-7",\n  "routeCode": "MSK-KZN",\n  "dispatchStamp": null\n}',
        '## 409 Conflict - перевозка с такой накладной уже создана.\n\n'
        '```json\n{"messageToUser": "Накладная уже есть"}\n```',
        "перевозка груза",
    )
    patch_request = [
        ("comment", "Комментарий диспетчера", "string", "Нет", "Можно передать null, чтобы очистить."),
        ("destinationCity", "Город назначения", "string", "Да", "Новый город выгрузки."),
    ]
    patch_response = [
        ("shipmentId", "Номер перевозки", "string", "Да", "Номер не меняется."),
        ("destinationCity", "Город назначения", "string", "Да", "Город после изменения."),
        ("routeRevision", "Ревизия маршрута", "string", "Да", "Хвостовой номер ревизии маршрута."),
    ]
    update = method(
        "Изменение плеча перевозки",
        "PATCH",
        "https://freight.example/v1/shipments/{shipmentId}",
        "Меняет город назначения у уже созданной перевозки.",
        patch_request,
        patch_response,
        '{\n  "shipmentId": "SH-7",\n  "destinationCity": "Казань",\n  "routeRevision": "r-4"\n}',
        "400 Bad Request - город назначения пустой. Тело ответа пустое.",
        "изменение плеча",
    )
    return (
        "# Грузовые перевозки\n\n"
        "Диспетчерская создаёт и правит плечи маршрута.\n\n"
        "## История изменений\n\n"
        "| Дата | Автор | Версия | Изменения |\n| --- | --- | --- | --- |\n"
        "| 11.03.2026 | Белов | 2 | Длинные карточки перевозки |\n\n"
        + create
        + "\n"
        + update
    )


def personnel() -> str:
    request = [
        ("employeeNumber", "Табельный номер", "string", "Да", "Номер в кадровом учёте."),
        ("lastName", "Фамилия", "string", "Да", "Фамилия работника."),
        ("firstName", "Имя", "string", "Да", "Имя работника."),
        ("departmentCode", "Подразделение", "string", "Да", "Код подразделения."),
        ("positionName", "Должность", "string", "Да", "Название должности по штатному расписанию."),
        ("hiredOn", "Дата приёма", "string", "Да", "Дата приказа о приёме."),
        ("email", "Почта", "string", "Нет", "Рабочая почта, может отсутствовать."),
    ]
    response = [
        ("dossierId", "Номер досье", "string", "Да", "Идентификатор личного дела."),
        ("employeeNumber", "Табельный номер", "string", "Да", "Тот же табельный номер."),
        ("dossierSeal", "Печать досье", "string", "Нет", "Хвостовая отметка архива кадров."),
    ]
    create = method(
        "Открытие личного дела",
        "POST",
        "https://personnel.example/hr/dossiers",
        "Открывает личное дело работника и возвращает номер досье.",
        request,
        response,
        '{\n  "dossierId": "DS-3",\n  "employeeNumber": "E-440",\n  "dossierSeal": null\n}',
        'Код HTTP: 422\n\n{"errorMessage": "Табельный номер уже занят"}',
        "личное дело",
    )
    get_response = [
        ("dossierId", "Номер досье", "string", "Да", "Номер запрошенного дела."),
        ("positionName", "Должность", "string", "Да", "Текущая должность."),
        ("archiveBox", "Короб архива", "string", "Нет", "Хвостовой номер короба, где лежит дело."),
    ]
    read = method(
        "Чтение личного дела",
        "GET",
        "https://personnel.example/hr/dossiers/{dossierId}",
        "Возвращает личное дело работника по номеру досье.",
        [("includeOrders", "Включать приказы", "boolean", "Нет", "Если да, в конец карточки добавляются приказы.")],
        get_response,
        '{\n  "dossierId": "DS-3",\n  "positionName": "Архивариус",\n  "archiveBox": "K-18"\n}',
        "204 No Content\n\nТело ответа пустое, если дело ещё не заведено в архив и карточки нет.",
        "чтение досье",
    )
    return (
        "# Кадровый архив\n\n"
        "Архив хранит личные дела и не смешивает их с приказами других работников.\n\n"
        "## История изменений\n\n"
        "| Дата | Автор | Версия | Изменения |\n| --- | --- | --- | --- |\n"
        "| 20.01.2026 | Сомова | 5 | Длинные досье |\n\n"
        + create
        + "\n"
        + read
    )


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    files = {
        "long-01-insurance.md": insurance(),
        "long-02-freight.md": freight(),
        "long-03-personnel.md": personnel(),
    }
    for name, text in files.items():
        path = DOCS / name
        path.write_text(text, encoding="utf-8")
        print(f"{name}: {len(text)} символов")


if __name__ == "__main__":
    main()
