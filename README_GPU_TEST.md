# SONYA GPU Test — Vast.ai Quick Start

Минимальная автономная папка для GPU-теста.
Не нужен полный проект. Только эта папка.

---

## Структура

```
sonya-gpu-test-upload/
  benchmark_runner.py       ← главный скрипт
  requirements_min.txt      ← только YOLO/видео (быстро)
  requirements_full.txt     ← + Whisper/аудио
  README_GPU_TEST.md        ← этот файл
  scripts/
    hook_mode_v1.py
    story_mode_v1.py
    trailer_mode_v3.py
    modes_scoring.py
    educational_mode_v5.py
    audio_cache.py          ← shared audio cache (WAV extraction)
    asr.py
    utils.py
  data/input/               ← сюда кладёшь видео (НЕ коммитить)
  outputs/                  ← сюда идут результаты (НЕ коммитить)
```

> **Не коммитить:** `data/input/`, `outputs/`, `*.mp4`, `*.zip`, `*.pt`, `__pycache__/`

---

## 1. Развернуть на Vast.ai

```bash
git clone <repo> && cd sonya-gpu-test-upload
pip install -r requirements_full.txt
```

---

## 2. Положить видео

```bash
mkdir -p data/input
yt-dlp -o "data/input/%(title)s.%(ext)s" "https://youtube.com/..."
# или загрузить через Jupyter Upload
```

---

## 3. Запустить тесты

### Benchmark без экспорта (быстро, только JSON):

```bash
python benchmark_runner.py \
  --videos data/input \
  --modes hook story viral educational trailer_preview \
  --model yolov8n \
  --whisper-model tiny \
  --output outputs/test_today \
  --skip-export \
  --strict-real
```

### Production review-split export:

```bash
python benchmark_runner.py \
  --videos data/input \
  --modes hook story viral educational trailer_preview \
  --model yolov8n \
  --whisper-model tiny \
  --output outputs/prod_today \
  --strict-real \
  --export-policy review-split
```

### Resume (пропустить YOLO/ASR из кэша):

```bash
python benchmark_runner.py \
  --videos data/input \
  --modes hook story viral educational trailer_preview \
  --model yolov8n \
  --whisper-model tiny \
  --output outputs/prod_today \
  --strict-real \
  --export-policy review-split \
  --resume
```

> **Важно:** `trailer_preview` должен идти **последним** — он использует
> результаты hook/story/viral/educational из текущего прогона.

---

## 4. Export Policies

| `--export-policy` | Поведение |
|---|---|
| `all` | Backward compat. Экспортирует все non-reject кандидаты. Дефолт. |
| `auto-only` | Экспортирует только `auto_export`. `manual_review` — нет. |
| `review-split` | Production: `auto_export` → `exports/auto/`, `manual_review` → `exports/review/` |
| `none` | Ничего не экспортировать. |
| `--skip-export` | Переопределяет любой `--export-policy`. Нет экспорта совсем. |

### Пример review-split output tree:

```
outputs/prod_today/
  exports/
    auto/
      educational/
        <run_id>.mp4        ← auto_export clip
    review/
      hook/
        <run_id>.mp4        ← manual_review clip
      story/
        <run_id>.mp4
      viral/
        <run_id>.mp4
      trailer_preview/
        <run_id>.mp4
```

---

## 5. Структура outputs/

```
outputs/<run>/
  run_manifest.json               ← git commit, gpu, timing, export summary
  exports_manifest.json           ← все экспортированные клипы по bucket
  summary.csv                     ← главная таблица + export_policy/bucket/path
  progress_report.txt             ← статус по уровням + export policy summary
  runtime_breakdown.csv           ← время по этапам
  failure_breakdown_template.csv  ← заполни после просмотра

  shared/
    base_analysis.json            ← YOLO base analysis (один раз на видео)
    asr_segments.json             ← Whisper ASR (один раз на видео)
    input_metadata.json           ← metadata видео
    audio_cache_manifest.json     ← audio WAV cache diagnostics

  exports/                        ← aggregated clips по bucket
    auto/
      <mode>/
        <run_id>.mp4
    review/
      <mode>/
        <run_id>.mp4

  <video>__<mode>__<model>__<id>/
    candidates.json
    ranking.json
    export_decision.json          ← decision + export_policy + bucket + path
    base_analysis.json            ← копия YOLO (для backward compat)
    pipeline_trace.json           ← diagnostics: base_analysis + audio_cache
    runtime_metrics.json          ← timing + export fields
    human_review_template.json    ← заполни вручную
    output.mp4                    ← legacy copy клипа (если экспортировался)
    error_traceback.txt           ← если реальный mode упал
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
| `--strict-real` | выключено | Exit 1 если хоть один режим = stub |
| `--export-policy` | `all` | all / auto-only / review-split / none |
| `--resume` | выключено | Переиспользовать shared/base_analysis + asr_segments |

---

## 7. Long-video safe mode

На видео >10 min YOLO автоматически переключается в `sparse_long_video`:

| Длина | Режим | Интервал | Max кадров |
|---|---|---|---|
| ≤600s | standard | 2.0s | 300 |
| 600–1200s | sparse_long_video | max(5, dur/300) | 300 |
| >1200s | sparse_long_video | max(5, dur/200) | 200 |

Metadata через ffprobe (fallback: OpenCV).
Если OpenCV не читает кадры → ffmpeg fallback (sparse jpg extraction).

---

## 8. Audio Cache

Viral и educational используют shared audio cache:
- ffmpeg извлекает mono 16kHz WAV один раз (`$TMPDIR/sonya_audio_cache/<hash>.wav`)
- librosa загружает WAV в RAM один раз за процесс
- Окна = numpy slice — нет повторных `librosa.load(mp4, offset, duration)`
- Нет PySoundFile / audioread warnings на mp4

Диагностика: `shared/audio_cache_manifest.json` и `pipeline_trace.json → audio_cache`.

---

## 9. --resume

Если `shared/base_analysis.json` и `shared/asr_segments.json` уже есть в `--output` папке:

```
Resume: using cached shared/base_analysis.json
Resume: using cached shared/asr_segments.json
```

YOLO и ASR не пересчитываются. Полезно при повторных прогонах на той же папке.
Если JSON битый — пересчитает автоматически.

---

## 10. Реальные режимы vs заглушки

| В метриках | Значение |
|---|---|
| `"stub": false` | реальный режим отработал |
| `"stub": true` + `error_traceback.txt` | реальный mode упал |
| `--strict-real` | exit 1 если stub=true хоть у одного режима |

---

## 11. Ручная оценка (после просмотра output.mp4)

Открой `human_review_template.json` и заполни:

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

## 12. Порядок режимов

```
1. hook           ← независимый
2. story          ← независимый
3. viral          ← независимый, audio cache
4. educational    ← независимый, audio cache
5. trailer_preview ← зависит от 1–4, ВСЕГДА ПОСЛЕДНИЙ
```

---

## 13. MVP-нормы

| Режим | top3_contains_good | boundary_ok |
|---|---|---|
| Hook | ≥ 70% | ≥ 75% |
| Story | ≥ 60% | ≥ 75% |
| Trailer/Preview | ≥ 55% | ≥ 75% |
| Viral | ≥ 65% | ≥ 75% |
| Educational | ≥ 60% | ≥ 75% |
