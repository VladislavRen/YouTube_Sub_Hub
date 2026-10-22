# Развёртывание на компьютере коллеги

Обновлено 22 октября 2026 года. Зависимости и сценарии обновления исправлены;
проверки описаны в [отчёте об исправлениях](deployment-fixes-2026-10-22.md) и
[результате сквозного прогона](handoff-validation-2026-10-22.md).
Команды ниже воспроизводят проверенный lock-файл. Google credentials, модели
и доступность внешних сервисов нужно настроить для каждой установки.

## 1. Что передавать

Передавайте исходники через Git вместе с `pyproject.toml`, `uv.lock`,
`.env.example`, `docker-compose.yml` и документацией. Не копируйте целиком рабочую
папку: в ней могут находиться `.env`, `token.json`, `client_secret*.json`, cookies,
виртуальные окружения, `library/`, базы данных и персональные результаты обработки.
На новой машине создайте собственную библиотеку и новый том Neo4j.

## 2. Окружение

Нужны Git, uv, Python 3.11, Docker с Compose, LM Studio и системные `ffmpeg`/`ffprobe`.
Текущий `doctor --strict` проверяет ffmpeg и Neo4j даже при использовании только captions.
Compose запускает только Neo4j; Python-сервис и LM Studio запускаются отдельно.

Из корня свежего checkout в macOS/Linux:

```bash
uv sync --locked --extra dev --python 3.11
. .venv/bin/activate
umask 077
cp .env.example .env
```

Инструкция рассчитана на macOS/Linux. Для Windows используйте Linux-окружение
WSL2: блокировки файлов библиотеки требуют POSIX `flock`. Нативный Windows
этой сборкой не поддерживается и не тестировался.

В `.env` задайте собственный `YT_SUB_HUB_NEO4J_PASSWORD`. Запустите сервер LM Studio
на `http://localhost:1234/v1`. Получите доступные идентификаторы моделей:

```bash
curl http://localhost:1234/v1/models
```

Установите `YT_SUB_HUB_QWEN_MODEL` и `YT_SUB_HUB_GEMMA_MODEL` в реальные ID доступных
локальных моделей с поддержкой нужного JSON-вывода. Можно использовать одну модель
в обеих настройках. Значения в `.env.example` — примеры исходной машины.
В примере обе роли анализа и перевод используют `google/gemma-4-e4b`.
Перевод включён: строгая проверка качества API/очереди требует русские транскрипт,
анализ и заметку. Настройте `YT_SUB_HUB_TRANSLATION_MODEL` на доступную модель.
Ошибки записи Neo4j прерывают обработку (`YT_SUB_HUB_GRAPH_BEST_EFFORT=false`).
Для vision нужна отдельно настроенная модель.

```bash
docker compose up -d neo4j
```

Compose фиксирует Neo4j 5.26.32 Community по digest и публикует порты только на
`127.0.0.1:7474` и `127.0.0.1:7687`. Без собственного пароля запуск отклоняется.
Смена пароля в `.env` не меняет пароль уже инициализированной базы.

## 3. Google: собственный проект и собственная авторизация

1. В [Google Cloud Console](https://console.cloud.google.com/) создайте проект для
   своей установки и включите **YouTube Data API v3**.
2. В **Google Auth Platform → Branding** заполните имя приложения и контактные данные.
3. В **Audience** выберите External для обычного Google-аккаунта. В режиме Testing
   добавьте свой email в Test users. Internal подходит только для пользователей
   соответствующей организации Google Workspace / Cloud Identity.
4. В **Data Access** добавьте `https://www.googleapis.com/auth/youtube.force-ssl`.
5. В **Clients** создайте клиент типа **Desktop app** и скачайте JSON. Сохраните его
   как `client_secrets.json` в корне проекта.
6. С активированным Python-окружением выполните на машине с браузером:

```bash
yt-sub-hub-auth --timeout 1800
chmod 600 client_secrets.json token.json
```

Войдите своим Google-аккаунтом и разрешите запрошенный доступ. Для локального CLI
используется browser OAuth с loopback callback на случайном локальном порту.
Этот helper не реализует авторизацию на удалённом сервере без браузера.
Тип клиента Desktop соответствует [официальной инструкции Google](https://developers.google.com/youtube/v3/quickstart/python)
и [desktop OAuth flow](https://developers.google.com/identity/protocols/oauth2/native-app).

Helper ищет client JSON в текущем каталоге, читает путь токена из
`YT_SUB_HUB_YOUTUBE_API_TOKEN_FILE` в `.env` (по умолчанию `token.json`).
Пути можно явно указать через `--client-secrets PATH --token-file PATH`;
последний должен совпадать с настройкой сервиса. Сохранение атомарное, с правами
`0600` на POSIX; при отказе Google существующий токен сохраняется. Проверяются
фактически выданные scopes. Старый `python youtube_auth_test.py` также работает.

`--no-browser` выводит ссылку для браузера на этой же машине. Оставляйте helper
работать до сообщения `OAuth credentials saved securely`. Если после Google
открывается `localhost — ERR_CONNECTION_REFUSED`, обработчик мог завершиться
по тайм-ауту: запустите команду заново и используйте новую ссылку. Старая ссылка
привязана к уже завершённой сессии.

Google API учитывает HTTP CONNECT / SOCKS-прокси из переменных окружения и
системных настроек macOS, включая исключения proxy bypass. Поддержка PySocks
включена в зависимости. Прокси с URL `https://` (TLS до самого прокси) этим
транспортом не поддерживаются: используйте поддерживаемый маршрут. HTTPS
соединение с Google при этом сохраняет проверку сертификата.

### Что именно разрешает Google

- `youtube.force-ssl` подходит для `captions.list` / `captions.download` и уже
  используется кодом. Это широкое разрешение; заменять его на `youtube.readonly`
  для скачивания captions нельзя.
- Для `captions.download` аккаунту нужны права редактировать конкретное видео.
  Просмотр публичного ролика не даёт таких прав. Поэтому сначала проверяйте
  собственное видео с субтитрами. [Требования метода](https://developers.google.com/youtube/v3/docs/captions/download).
- Публичные метаданные Google допускает получать с API key, но текущий код сервиса
  реализует официальный доступ через OAuth token. API key не заменяет этот файл
  без доработки. [Google quickstart](https://developers.google.com/youtube/v3/quickstart/python).
- Для ограниченного личного использования Google предусматривает исключение из
  обязательной верификации; для публичного приложения с чувствительными scopes
  требования другие. [Исключения](https://support.google.com/cloud/answer/13464323?hl=en),
  [верификация](https://support.google.com/cloud/answer/13464321).
- В External + Testing refresh token обычно истекает через 7 дней. После перехода
  в In production заново авторизуйтесь; это не гарантирует бессрочный токен:
  возможны отзыв доступа и ограничения администратора.
  [Сроки токенов](https://developers.google.com/identity/protocols/oauth2#expiration).

### Квоты и неофициальная загрузка

На дату проверки Google указывает отдельный лимит по умолчанию 100 вызовов
`search.list` в сутки и 10 000 units для остальных методов, кроме отдельно
лимитируемого `videos.insert`. `captions.list` стоит 50 units, скачивание одного
трека — 200 units. Фактические квоты своего проекта смотрите в Cloud Console.
Discovery теперь ведёт раздельные бюджеты: до 100 search-вызовов и
`--max-quota-units` для остальных методов (по умолчанию 900) на один discovery run.
SQLite ledger сохраняет расходы между стадиями, параллельными воркерами и retries;
неудавшиеся запросы также резервируют бюджет. Это локальный предел запуска,
а не остаток суточной квоты всего Google-проекта: новый run не восстанавливает
реальную квоту Google. Остаток проекта проверяйте в Cloud Console.
[Квоты](https://developers.google.com/youtube/v3/determine_quota_cost),
[скачивание captions](https://developers.google.com/youtube/v3/docs/captions/download).

Начните с `YT_SUB_HUB_MEDIA_DOWNLOAD=disabled` и `YT_SUB_HUB_ACCEPTANCE_MODE=data_api_only`.
Этот режим не использует неофициальный transcript fallback. Включение `full_processing`
и media download — отдельный выбор оператора, а не разрешение Google на скачивание:
политики API ограничивают хранение аудиовизуальных копий без предварительного
письменного разрешения YouTube. [Политики YouTube](https://developers.google.com/youtube/terms/developer-policies).

Технически для современного yt-dlp также нужны совместимые EJS scripts и JS runtime.
Совместимый `yt-dlp-ejs` уже включён через `yt-dlp[default]`. Установите поддерживаемый
runtime: Deno используется по умолчанию; для Node.js >=22 задайте
`YT_SUB_HUB_YTDLP_JS_RUNTIME=node`. При включённом media fallback `doctor`
проверяет наличие EJS и исполняемого файла runtime. Версию runtime сверяйте
самостоятельно с [инструкцией yt-dlp](https://github.com/yt-dlp/yt-dlp/wiki/EJS).

## 4. Проверка и получение свежих данных

```bash
yt-sub-hub doctor --strict
yt-sub-hub captions-probe "https://www.youtube.com/watch?v=VIDEO_ID" --json
```

Замените `VIDEO_ID` на ID своего видео с субтитрами. У `captions-probe` нужно
прочитать результат JSON, включая `summary.can_run_data_api_only`: нулевой exit code
сам по себе не означает успешного скачивания captions. `doctor` проверяет метаданные,
но не доказывает наличие прав на captions конкретного ролика.

Для первого запуска используйте новую пустую библиотеку. Например, укажите в `.env`
`YT_SUB_HUB_LIBRARY_DIR=library/colleague-fresh` и убедитесь, что этот подкаталог ещё
не существует. Затем:

```bash
yt-sub-hub process "https://www.youtube.com/watch?v=VIDEO_ID"
uvicorn yt_sub_hub.api:create_app --factory --host 127.0.0.1 --port 8000
```

API-документация: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
Для чужого публичного ролика с доступными субтитрами можно явно разрешить
неофициальный transcript fallback, сохранив запрет загрузки аудио/видео:

```bash
yt-sub-hub process "https://www.youtube.com/watch?v=VIDEO_ID" --acceptance-mode full_processing
# Или отправить в запущенный API:
curl -X POST http://127.0.0.1:8000/jobs \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://www.youtube.com/watch?v=VIDEO_ID","acceptance_mode":"full_processing","quality_mode":"strict"}'
```

При `YT_SUB_HUB_MEDIA_DOWNLOAD=disabled` отсутствие доступных субтитров завершит
обработку ошибкой без скачивания медиа. Оставьте глобальный acceptance-профиль
официальным: `doctor --strict` пока требует media download для глобального
`full_processing`, тогда как явный режим отдельного задания разрешает только
доступные fallback-провайдеры. HTTP 200 при создании задания означает постановку
в очередь; результат проверяйте через `GET /jobs/{job_id}`, `/quality` и `/artifacts`.
Статус с предупреждением требует чтения причины: в strict-режиме предупреждения
review или продуктовой оценки качества останавливают запись графа.
Готовность результата проверяйте также по `product_quality.production_ready` в `/quality`.

Qwen с включёнными рассуждениями на исходной машине не уложился в 120 секунд.
Теперь `doctor` проверяет законченный JSON-ответ, а не только HTTP 200;
модель, возвращающая один reasoning или обрезанный ответ, проверку не проходит.

Чтобы заново получить метаданные и субтитры уже обработанного видео:

```bash
yt-sub-hub process "https://www.youtube.com/watch?v=VIDEO_ID" --refresh
```

`--refresh` получает исходные данные заново и пересоздаёт зависящие от них
переводы, анализ и граф. Ошибка загрузки не приводит к использованию старого
транскрипта, в том числе после перезапуска. Заметки перед обновлением сохраняются
как `note.before-refresh.<timestamp>.md` / `note.source.before-refresh.<timestamp>.md`.
`--force` повторяет анализ и может использовать ещё свежий кэш.

Метаданные и исходные транскрипты старше 30 дней не используются при обработке
(CLI и staged pipeline). Возраст определяется временем записи файла; перенос
готового кэша с изменением времени файлов может исказить возраст. Для коллеги
поэтому создавайте пустую библиотеку.

Для удаления просроченных исходных файлов сначала посмотрите список:

```bash
yt-sub-hub cache-cleanup
# После проверки списка:
yt-sub-hub cache-cleanup --apply
```

Команда удаляет только `metadata.json`, `transcript.raw.json` и
`transcript.normalized.json` в валидных каталогах видео. Заметки, резервные копии,
производные результаты, discovery-кэш, медиа и данные Neo4j сохраняются.
Успешная синхронизация обновлённого видео атомарно заменяет основные связи графа,
сохраняя отдельные research/discovery-связи; при
отключённой или неуспешной синхронизации удалённый граф может остаться прежним.
Обработка по возрасту и очистка исходников не заменяют политику хранения всех
данных: для длительной эксплуатации определите обновление/удаление остальных
категорий и расписание обслуживания. Google требует обновлять или удалять
определённые категории сохранённых данных через 30 дней.
[Правила хранения API Data](https://developers.google.com/youtube/terms/developer-policies#handling-youtube-data-and-content).

## 5. Дополнительные режимы

Для загрузки медиа и распознавания в своём профиле задайте:

```dotenv
YT_SUB_HUB_MEDIA_DOWNLOAD=enabled
YT_SUB_HUB_ACCEPTANCE_MODE=full_processing
YT_SUB_HUB_YTDLP_JS_RUNTIME=node
YT_SUB_HUB_WHISPER_BACKEND=openai-whisper-cpu
YT_SUB_HUB_WHISPER_MODEL=base
YT_SUB_HUB_VISION_MODEL=google/gemma-4-e4b
```

Нужны FFmpeg, выбранный JS runtime и загруженная vision-модель в LM Studio.
Обычная обработка сначала использует доступные субтитры; Whisper включается при
их отсутствии. `base` экономит ресурсы, но допускает ошибки имён и таймкодов.
Проверенный дополнительный backend на Apple Silicon — `whisper-cpp-metal`;
бинарник и модель настраиваются через `setup-whisper-cpp --help`.

После обработки своего видео:

```bash
yt-sub-hub analyze-frame VIDEO_ID --at 30
yt-sub-hub research-video VIDEO_ID --max-sources 8
yt-sub-hub review-research VIDEO_ID --json
yt-sub-hub research-sync-neo4j VIDEO_ID --json
yt-sub-hub discover-youtube VIDEO_ID --limit 20 --max-quota-units 12
yt-sub-hub review-discovery VIDEO_ID --json
yt-sub-hub discovery-sync-neo4j VIDEO_ID --json
```

Для research нужны готовые русские core-артефакты. Он собирает внешний контекст
и проверяет происхождение цитат; **истинность тезисов автоматически не проверяется**.
Непроверенные по смыслу находки имеют `accepted=false`; `summary_ru=null` означает,
что переведённое резюме не создавалось, а исходная выдержка лежит в `excerpt_original`.
После изменения заметки/анализа или неуспешного повтора нужно заново выполнить
`research-video`, затем discovery: старые результаты не проходят проверку хешей/run ID.

Бюджет `--max-quota-units 12` ограничивает общий счётчик методов Data API в этом
запуске; `search.list` учитывается отдельно. В проверенном примере использованы
12 search-вызовов и 12 общих единиц. Это не оценка остатка дневной квоты проекта.
Discovery не запускает обработку найденных видео автоматически. `needs_review`
и отношение `unclear` требуют просмотра; поисковый запрос сам по себе не доказывает
поддержку или опровержение тезиса.

Vision сохраняет изображение и ответ для сверки. Мелкий текст, числа и детали
интерфейса могут быть распознаны неверно даже при успешном HTTP 200.
[Результаты живой проверки и ограничения](optional-modes-validation-2026-10-22.md).

## 6. Типовые ошибки

| Ошибка | Что проверить |
| --- | --- |
| `invalid_grant` | Заново пройти OAuth своим аккаунтом; проверить Testing, отзыв доступа и политику организации. Причину нельзя установить только по этому коду. |
| `access_denied` / приложение в Testing | Email в Test users и ограничения администратора Workspace. |
| `accessNotConfigured` | Включена ли YouTube Data API v3 именно в проекте OAuth client. |
| `captions_forbidden` / 403 | Права аккаунта на видео и выданный scope; повторная установка библиотек прав не добавляет. |
| `quotaExceeded` | Квоты проекта и раздельные лимиты search/остальных методов. |
| `lm_studio_model:… missing` | ID модели в `.env`, загрузка модели и доступность локального сервера. |
| `localhost — ERR_CONNECTION_REFUSED` после входа | Перезапустить `yt-sub-hub-auth --timeout 1800`, открыть новую ссылку на той же машине. |
| Старые результаты после `--force` | Используйте `--refresh` для повторного получения исходных данных. |
| `Video … is busy` | Дождаться завершения активного воркера этого видео и повторить обновление/очистку. |
