# HW03/HW04 – eda_cli: мини-EDA для CSV + HTTP-сервис качества данных

CLI-приложение для базового анализа CSV-файлов и HTTP-сервис для оценки качества данных.
Используется в рамках Семинаров 03-04 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports --sep , --encoding utf-8 --max-hist-columns 6 --top-k-categories 5 --title "EDA-отчёт"
```

Параметры:

- `--out-dir` – каталог для отчёта (по умолчанию `reports`)
- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).
- `--max-hist-columns` – максимум числовых колонок для гистограмм (по умолчанию `6`)
- `--top-k-categories` – сколько top-значений выводить для категориальных признаков (по умолчанию `5`)
- `--title` – заголовок отчёта (по умолчанию `"EDA-отчёт"`)

В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

## HTTP-сервис (HW04)

### Запуск сервиса

```bash
uv run uvicorn eda_cli.api:app --reload --port 8000
```

## Доступные эндпоинты

### GET /health - проверка работоспособности

```bash
curl -X GET "http://localhost:8000/health"
```

### POST /quality - оценка качества по агрегированным признакам

```bash
curl -X POST "http://localhost:8000/quality" \
  -H "Content-Type: application/json" \
  -d '{
    "n_rows": 1000,
    "n_cols": 10,
    "max_missing_share": 0.1,
    "numeric_cols": 5,
    "categorical_cols": 5
  }'
```

### POST /quality-from-csv - оценка качества по CSV-файлу

```bash
curl -X POST "http://localhost:8000/quality-from-csv" \
  -F "file=@data/example.csv"
```

### POST /quality-flags-from-csv - ПОЛНЫЙ НАБОР ФЛАГОВ КАЧЕСТВА

```bash
curl -X POST "http://localhost:8000/quality-flags-from-csv" \
  -F "file=@data/example.csv"
```


## Тесты

```bash
uv run pytest -q
```
