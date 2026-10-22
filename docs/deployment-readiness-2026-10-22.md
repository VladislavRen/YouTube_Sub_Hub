# Проверка библиотек и Google-доступа — 22 октября 2026

> Исторический снимок до исправлений. Текущее состояние описано в
> [результатах исправлений](deployment-fixes-2026-10-22.md); версии и результаты
> ниже сохранены как исходные доказательства аудита.

**Вывод:** исходники можно передать для ознакомления и настройки, но текущую
сборку нельзя считать проверенным актуальным релизом для запуска у коллеги.
Нужны обновление уязвимых зависимостей с проверкой совместимости и новая OAuth
авторизация на стороне коллеги. Для свежих входных данных нужна пустая библиотека.

Проверены `pyproject.toml`, весь `uv.lock`, `docker-compose.yml`, инструкции,
OAuth helper и пути получения/кэширования YouTube-данных. Это аудит зависимостей
и готовности к передаче; реализация и версии пакетов в Git не изменены.

## Основные находки

| ID | Приоритет | Доказательство и влияние | Действие / статус |
| --- | --- | --- | --- |
| DEP-01 | P1 | `pip-audit` обнаружил известные advisories у 14 пакетов стандартной установки + dev на macOS. Есть `yt-dlp`, `starlette`, `urllib3`, `cryptography`, `httplib2`; сведения ниже. | Блокер для рекомендации текущей сборки как безопасного актуального релиза. Обновить согласованный граф зависимостей, повторить аудит и тесты. Эксплуатируемость каждой CVE в приложении отдельно не доказана. |
| DEP-02 | P2 | В исходном README установка через `uv pip install -e .` обходила `uv.lock`. В `.venv311` 27 установленных версий отличались от lock. | Инструкция исправлена на `uv sync --locked`. Новые версии не протестированы этим исправлением. |
| DEP-03 | P2 | `yt-dlp` закреплён на 2026.3.17; актуальная версия 2026.10.21. Нет `yt-dlp-ejs`, необходимого вместе с JS runtime для современных YouTube challenges. | Для media fallback подготовить обновление `yt-dlp[default]`, добавить runtime в системные требования и проверить реальную загрузку. |
| GOOGLE-01 | P1 | Попытка обновить локальные credentials через официальный Google OAuth endpoint завершилась `invalid_grant: Bad Request`; вызов `channels.list` не состоялся. | Блокер официального доступа с текущим токеном. Требуется повторный интерактивный вход; коллеге — собственный token. Причина отзыва/истечения не установлена. |
| GOOGLE-02 | P1 | `captions.download` требует прав редактировать видео. `data_api_only` по умолчанию отключает неофициальный transcript provider. | Блокер сценария «любой чужой публичный ролик через официальный API». Проверять собственное видео с captions; scope не расширяет права на чужой контент. |
| GOOGLE-03 | P2 | `src/yt_sub_hub/youtube_data_api.py:67` передаёт требуемые scopes в `Credentials.from_authorized_user_file`, после чего `has_scopes` проверяет уже эти значения. На синтетическом read-only token проверка force-ssl вернула True. | Проверять сохранённые/реально выданные scopes; текущая локальная проверка не доказывает разрешение Google. Подлинный серверный доступ не обходится. |
| GOOGLE-04 | P2 | `src/yt_sub_hub/youtube_discovery.py:18` считает `search.list` как 100 общих units, а текущая документация Google описывает отдельный bucket: 100 вызовов/сутки по 1 unit. | Follow-up: разнести бюджет поиска и прочих методов. Текущий отчёт `quota_usage` не отражает фактическую модель Google. Просто заменить 100 на 1 недостаточно. |
| DATA-01 | P2 | `pipeline.py:_resolve_or_load_transcript` возвращает кэш до запроса. `process --force` меняет поведение повторного анализа, но не инвалидирует transcript. Staged pipeline также использует cache hits. | Для передачи использовать новую пустую `library`. Для долгой эксплуатации реализовать явный refresh и правила хранения API Data. |
| OPS-01 | P2 | `neo4j:5-community` — плавающий tag; порты 7474/7687 опубликованы на всех интерфейсах. | Follow-up: зафиксировать проверенный patch/digest и ограничить локальные порты loopback. Python driver 6.x совместим с Neo4j 5.x; сам разный major не является дефектом. |

Источники для требований: [Google captions.download](https://developers.google.com/youtube/v3/docs/captions/download),
[актуальные квоты](https://developers.google.com/youtube/v3/determine_quota_cost),
[yt-dlp EJS](https://github.com/yt-dlp/yt-dlp/wiki/EJS),
[совместимость Neo4j](https://neo4j.com/docs/python-manual/current/upgrade/).
Приоритеты — оценка риска передачи проекта, а не присвоенный каждой уязвимости CVSS.

## Актуальность прямых зависимостей

Версии получены из JSON API PyPI при проверке; «актуальная» означает последний
опубликованный stable release, а не подтверждённую совместимость с приложением.
Все перечисленные текущие releases допускают Python 3.11 по `Requires-Python`.

| Библиотека | В `uv.lock` | Актуальная в PyPI |
| --- | --- | --- |
| [fastapi](https://pypi.org/project/fastapi/) | 0.136.1 | 0.159.1 |
| [uvicorn](https://pypi.org/project/uvicorn/) | 0.46.0 | 0.54.0 |
| [typer](https://pypi.org/project/typer/) | 0.25.1 | 0.27.3 |
| [youtube-transcript-api](https://pypi.org/project/youtube-transcript-api/) | 1.2.4 | 1.2.4 |
| [yt-dlp](https://pypi.org/project/yt-dlp/) | 2026.3.17 | 2026.10.21 |
| [openai-whisper](https://pypi.org/project/openai-whisper/) | 20250625 | 20250625 |
| [openai](https://pypi.org/project/openai/) | 2.36.0 | 3.36.1 |
| [httpx](https://pypi.org/project/httpx/) | 0.28.1 | 0.28.1 |
| [ddgs](https://pypi.org/project/ddgs/) | 9.14.4 | 9.16.0 |
| [neo4j](https://pypi.org/project/neo4j/) | 6.2.0 | 6.4.1 |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | 1.2.2 | 1.2.4 |
| [google-api-python-client](https://pypi.org/project/google-api-python-client/) | 2.196.0 | 2.201.0 |
| [google-auth-httplib2](https://pypi.org/project/google-auth-httplib2/) | 0.4.0 | 0.4.4 |
| [google-auth-oauthlib](https://pypi.org/project/google-auth-oauthlib/) | 1.4.0 | 1.5.0 |
| [pytest](https://pypi.org/project/pytest/) — extra dev | 9.0.3 | 9.1.1 |
| [beeai-framework](https://pypi.org/project/beeai-framework/) — optional extra | 0.1.80 | 0.1.85 |

Важные транзитивные версии: `google-auth` 2.52.0 → 2.62.0,
`torch` 2.12.0 → 2.14.1, `starlette` 1.0.0 → 1.9.0,
`pydantic` 2.13.4 → 2.14.1, `urllib3` 2.7.0 → 2.8.0.
Major upgrade OpenAI SDK 2 → 3 требует отдельной проверки LM Studio provider и
benchmark; не следует автоматически принимать все latest releases.

## Результат security scanner

`pip-audit 2.10.1`, источник advisories PyPI:

- Весь lock: **137 registry package pins**, включая optional beeai и платформенные
  зависимости; **56 уникальных пар package/advisory ID в 19 пакетах**.
- Стандартная установка + extra dev на macOS/Python 3.11: **88 пакетов**;
  **30 уникальных пар package/advisory ID в 14 пакетах**.
- Сырой scanner сообщил 89 и 49 записей соответственно, но часть ID повторяется.
  В числах выше дубли одного advisory ID внутри пакета удалены.
- Совпадение с advisory подтверждено scanner по версии; это не означает,
  что все уязвимые функции вызываются данным сервисом. Например, некоторые
  advisories yt-dlp относятся к `--exec` и внешним downloaders, а OAuthlib — к
  серверной проверке PKCE. Они не приравниваются к доказанному взлому приложения.

Минимальные версии ниже взяты из advisories; это ориентиры для resolver, а не
инструкция устанавливать каждый пакет отдельно. После обновления нужен новый audit.

| Пакет в стандартной установке | Закреплён | Уникальных advisories | Версия, покрывающая указанные исправления |
| --- | --- | --- | --- |
| anyio | 4.13.0 | 2 | 4.14.2 |
| certifi | 2026.4.22 | 1 | 2026.10.19 |
| cryptography | 48.0.0 | 4 | 50.0.0 |
| fsspec | 2026.4.0 | 1 | 2026.6.0 |
| h2 | 4.3.0 | 1 | 4.4.1 |
| hpack | 4.1.0 | 1 | 4.2.0 |
| httplib2 | 0.31.2 | 1 | 0.32.0 |
| oauthlib | 3.3.1 | 1 | 4.0.0 |
| pyasn1 | 0.6.3 | 3 | 0.6.4 |
| setuptools | 81.0.0 | 1 | 83.0.0 |
| starlette | 1.0.0 | 5 | 1.3.1 |
| torch | 2.12.0 | 1 | 2.13.0 |
| urllib3 | 2.7.0 | 3 | 2.8.0 |
| yt-dlp | 2026.3.17 | 5 | 2026.7.4 |

Только в расширенном lock дополнительно затронуты `aiohttp`, `json-repair`,
`litellm`, `multidict`, `pydantic-settings`. Полные ID, aliases и fix versions,
включая эти пакеты, сохранены в [машиночитаемом отчёте](dependency-audit-2026-10-22.json).
Примеры первичных advisories:
[yt-dlp filename sanitization](https://github.com/yt-dlp/yt-dlp/security/advisories/GHSA-c6mh-fpjc-4pr3),
[Starlette form limits](https://github.com/Kludex/starlette/security/advisories/GHSA-82w8-qh3p-5jfq).

Для повторения стандартной проверки без установки пакетов scanner-ом:

```bash
uv export --locked --extra dev --no-emit-project --no-hashes -o /tmp/yt-sub-hub-requirements.txt
uvx --python 3.11 pip-audit -r /tmp/yt-sub-hub-requirements.txt --no-deps --disable-pip
```

Полный scan построен из всех registry entries `uv.lock` без environment markers,
поэтому охватывает и пакеты, не устанавливаемые на текущей ОС. Он не проверяет
встроенные native libraries, содержимое Docker image или LM Studio models.

## Google и актуальность данных

В локальном JSON указан правильный scope `youtube.force-ssl`, тип клиента —
`installed`. Однако реальное обновление credentials отвергнуто Google с
`invalid_grant`. Токен не изменялся, интерактивный вход не запускался. Статус
Testing/In production, включённые API и реальные квоты проекта через Cloud Console
не проверены. Истечение access token само по себе нормально; проблема установлена
именно ответом на попытку refresh.

Для External + Testing Google обычно выдаёт refresh token на 7 дней. Для
ограниченного личного использования есть исключение из обязательной OAuth
верификации; оно не отменяет правила YouTube. Коллеге рекомендуется отдельный
проект, Desktop OAuth client и вход своим аккаунтом.
[Refresh token expiration](https://developers.google.com/identity/protocols/oauth2#expiration),
[исключения из верификации](https://support.google.com/cloud/answer/13464323?hl=en).

Не найдено механизма планового refresh/delete сохранённых YouTube API Data.
Google устанавливает правила актуализации и удаления, в том числе 30-дневный
срок для определённых категорий данных; точное применение зависит от типа данных
и авторизации. Media fallback также не получает разрешение YouTube только за счёт
OAuth или установки переменной `MEDIA_DOWNLOAD=enabled`.
[Developer Policies, III.E](https://developers.google.com/youtube/terms/developer-policies).

## Проверки и ограничения

- `uv lock --check`: успешно, manifest-lock drift не обнаружен.
- `uv pip check` в `.venv311` и затем в `.venv`: успешно; это проверка declared
  version constraints, а не отсутствия уязвимостей.
- Существующее `.venv311`: **650 passed**; оно отличается от lock-файла.
- `uv run --locked --extra dev python -m pytest -q`: **649 passed, 1 failed**.
  Упал `test_benchmark_aggregates_runs_and_recommends_two` (`tests/test_lm_benchmark.py:207`).
  В сохранённом benchmark report baseline успел обработать 9/12 chunks за лимит
  10 секунд; рекомендация корректно откатилась к concurrency=1 вместо ожидаемых 2.
  Это не подтверждает полную совместимость lock-сборки; причина замедления отдельно
  не установлена. Отдельный повтор этого теста также завершился падением за
  10.17 секунд (`safe_default=1`, ожидалось 2); проблему нельзя списать на единичный сбой.
- `uv run --locked --extra dev python -m compileall -q src tests`: успешно.
- Проверены локальные ссылки документации, согласованность итогов JSON audit и
  `git diff --check`: ошибок не обнаружено.
- `.venv` синхронизировано с существующим lock + dev для проверки; `.venv311`
  оставлено без обновления. `pyproject.toml`, `uv.lock` и код не изменены.
- По проверке Git нет tracked-файлов, попадающих под текущие ignore rules.
  Это не полный secret scan истории. Локальные OAuth files и `.env` не включены
  в отчёт и не предназначены для передачи.
- Полный live pipeline с Google, LM Studio и Neo4j не подтверждён из-за ошибки
  OAuth; получение чужих captions и media downloads не выполнялось.
- На машине коллеги и на других ОС тестирование не проводилось. Docker image,
  native binaries и модельные веса не проходили vulnerability scan.

## Следующий шаг перед передачей рабочего релиза

Обновить граф зависимостей и включить EJS при сохранении media fallback, получить
чистый/разобранный audit и воспроизводимые тесты; исправить проверку OAuth scope
и учёт квот. Затем коллеге выполнить [инструкцию развёртывания](self-hosting.md)
со своими credentials, моделями и новой пустой библиотекой, проверить captions
своего видео и один полный pipeline. Для длительной эксплуатации дополнительно
нужны refresh/delete сохранённых API Data.
