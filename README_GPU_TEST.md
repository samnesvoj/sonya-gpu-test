# SONYA GPU Test — Vast.ai Quick Start

Минимальная автономная папка для GPU-теста 3 видео.
Не нужен полный проект. Только эта папка.

---

## Структура

```
SONYA-GPU-TEST/
  benchmark_runner.py       ← главный скрипт
  requirements_min.txt      ← только YOLO/видео (быстро)
  requirements_full.txt     ← + Whisper/аудио (для hook/story)
  README_GPU_TEST.md        ← этот файл
  scripts/
    hook_mode_v1.py
    story_mode_v1.py
    trailer_mode_v3.py
    modes_scoring.py
    educational_mode_v5.py
    asr.py
    utils.py
  data/input/               ← сюда кладёшь видео
  outputs/                  ← сюда идут результаты
```

---

## 1. Развернуть на Vast.ai

```bash
# Клонировать или загрузить архив
git clone <repo> && cd SONYA-GPU-TEST
# ИЛИ распаковать: unzip sonya-gpu-test.zip && cd SONYA-GPU-TEST

# День 1 — быстрый старт (только YOLO, без Whisper):
pip install -r requirements_min.txt

# День 2 — с реальными hook/story/educational (нужен Whisper):
pip install -r requirements_full.txt
```

---

## 2. Положить видео

```bash
# Загрузить через Jupyter → Upload, или через wget/yt-dlp:
yt-dlp -o "data/input/%(title)s.%(ext)s" "https://youtube.com/..."

# Структура:
data/input/
  video_01.mp4
  video_02.mp4
  video_03.mp4
```

---

## 3. Запустить тесты

### Быстрый тест (только JSON, без нарезки):

```bash
python benchmark_runner.py \
  --videos data/input \
  --modes hook story trailer_preview \
  --model yolov8n \
  --whisper-model tiny \
  --output outputs/test_today \
  --skip-export
```

### С нарезкой клипов (Whisper small — лучше качество):

```bash
python benchmark_runner.py \
  --videos data/input \
  --modes hook story trailer_preview \
  --model yolov8n \
  --whisper-model small \
  --output outputs/test_today
```

### Все режимы:

```bash
python benchmark_runner.py \
  --videos data/input \
  --modes hook story viral educational trailer_preview \
  --model yolov8n \
  --whisper-model base \
  --output outputs/all_modes
```

> **Важно:** `trailer_preview` должен идти **последним** — он использует
> результаты hook/story/viral/educational из текущего прогона.

---

## 4. Забрать результаты

```bash
zip -r outputs_test_today.zip outputs/test_today
# Скачать через Jupyter → File browser → правая кнопка → Download
```

---

## 5. Что появится в outputs/

```
outputs/test_today/
  summary.csv                     ← главная таблица всех прогонов
  progress_report.txt             ← статус по уровням 🔴🟡🟢🔵
  runtime_breakdown.csv           ← время по этапам
  failure_breakdown_template.csv  ← заполни после просмотра клипов
  model_comparison.csv            ← сравнение yolov8n vs yolov8s

  video_01__hook__yolov8n__abc123/
    input_metadata.json           ← длина, fps, разрешение
    runtime_metrics.json          ← время, GPU, stub/real, adapter_error
    candidates.json               ← все найденные моменты
    ranking.json                  ← отсортированные
    export_decision.json          ← auto_export / manual_review / reject
    boundary_diagnostics.json     ← качество обрезки
    human_review_template.json    ← ЗАПОЛНИ ВРУЧНУЮ после просмотра!
    output.mp4                    ← итоговый клип (если не --skip-export)
```

---

## 6. Ключи CLI

| Ключ | По умолчанию | Описание |
|---|---|---|
| `--videos` | `data/input` | Папка с mp4/mov/avi |
| `--modes` | `hook story trailer_preview` | Список режимов |
| `--model` | `yolov8n` | YOLO: n (быстро) / s / m / x (точно) |
| `--output` | `outputs/test_today` | Папка результатов |
| `--skip-export` | выключено | Только JSON, без нарезки mp4 |
| `--whisper-model` | `base` | tiny/base/small/medium/large |

---

## 7. Реальные режимы vs заглушки

Все режимы **автоматически** пробуют подключить реальный модуль из `scripts/`.
Если реальный модуль падает → fallback в заглушку + `adapter_error` в `runtime_metrics.json`.

| В метриках | Значение |
|---|---|
| `"stub": false` | реальный режим отработал |
| `"stub": true` | заглушка (нет Whisper или ошибка) |
| `"adapter_error": "..."` | почему упал реальный режим |

---

## 8. Ручная оценка (после просмотра output.mp4)

Открой `human_review_template.json` в папке прогона и заполни:

```json
{
  "human_rating": 4,
  "mode_match": true,
  "top3_contains_good": true,
  "boundary_ok": false,
  "would_publish": false,
  "failure_reason": "boundary",
  "notes": "момент хороший, но обрезает середину фразы"
}
```

**Варианты `failure_reason`:**
`proposer` · `ranking` · `boundary` · `quality_gate` · `transition` · `assembly` · `runtime` · `bad_input`

---

## 9. Порядок тестирования режимов

```
1. hook           ← независимый, первый
2. story          ← независимый
3. viral          ← независимый, без ASR
4. educational    ← независимый
5. trailer_preview ← зависит от 1–4, всегда последний
```

---

## 10. MVP-нормы (цели для первого этапа)

| Режим | top3_contains_good | boundary_ok |
|---|---|---|
| Hook | ≥ 70% | ≥ 75% |
| Story | ≥ 60% | ≥ 75% |
| Trailer/Preview | ≥ 55% | ≥ 75% |
| Viral | ≥ 65% | ≥ 75% |
| Educational | ≥ 60% | ≥ 75% |
