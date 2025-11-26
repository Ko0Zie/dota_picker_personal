# Dota2 Draft AI Assistant

Локальная ML-система для рекомендаций героев и item билдов на основе данных OpenDota. Архитектура рассчитана на запуск внутри VSCode (Python 3.10+) без интеграции в клиент Dota2.

⚠️ **Важно:** инструмент предназначен только для внешнего использования (второй монитор/CLI). Никаких хуков, инжектов или автоматизации клиента. Valve запрещает подобные модификации.

## Репозиторий

```
.
├── build_dataset.py      # формирование табличного датасета и фичей
├── fetch_opendota.py     # загрузка героев, матчей, itemPopularity c rate-limit контролем
├── item_module.py        # условные рекомендации айтемов + fallback к meta
├── predict_cli.py        # CLI для top-N героев, item build и объяснений
├── train_model.py        # обучение LightGBM с time split и метриками
├── utils.py              # вспомогательные функции и константы
├── data/
│   ├── raw/              # реальные выгрузки OpenDota
│   └── sample/           # мини-набор для smoke-тестов
├── models/               # сохраняемые бустеры и метаданные
├── examples/             # сценарии драфтов и player profile пример
└── docs/architecture.md  # high-level схема пайплайна
```

## Быстрый старт (VSCode)

1. **Создать виртуальное окружение**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Скачать данные**
   ```bash
   python fetch_opendota.py --num-matches 5000 --patch 7.36c
   ```
   Скрипт сохранит `heroes.json`, `hero_stats.json`, `item_popularity.json`, `matches/matches_*.jsonl`.

3. **Построить датасет и фичи**
   ```bash
   python build_dataset.py --raw-dir data/raw --processed-dir data/processed
   ```
   На выходе: `drafts_dataset.parquet/csv`, `hero_stats.json`, `synergy_stats.json`, `counter_stats.json`, `item_stats.json`, `feature_manifest.json`.

4. **Обучить модель**
   ```bash
   python train_model.py --processed-dir data/processed --models-dir models
   ```
   Скрипт сохранит `models/hero_rec_model.txt`, `model_metadata.json`, `training_metrics.json` (metrics: AUC, logloss, Top-1/Top-K).

5. **Запустить CLI (пример)**
   ```bash
   python predict_cli.py \
     --role carry \
     --enemies "Invoker,Lion,Queen of Pain" \
     --allies "Bane,Vengeful Spirit" \
     --bans "Juggernaut,Sven" \
     --patch 7.36c \
     --show-items \
     --player-profile examples/player_profile.example.json
   ```

   Пример вывода (с тестовыми данными):
   ```
   Rank Hero           Win Prob Confidence Reasons
   1    Anti-Mage      0.572    0.80       + synergy (0.58) with Bane, Vengeful Spirit; Global winrate 0.54
   2    Shadow Fiend   0.533    0.77       + counters (0.51) vs Invoker, Lion; Global winrate 0.51
   ...
   Anti-Mage item plan:
     - battle_fury @ 15m (confidence 0.08, match_item_timings)
     - manta @ 25m (confidence 0.07, match_item_timings)
   ```

## Что внутри модели

- **Фичи (обязательные по ТЗ)**: `candidate_hero_id`, `role`, `patch`, `hero_global_winrate`, `hero_games_count`, `pairwise_synergy_score`, `counter_score`, `avg_item_timing_first/second`, `skill_bracket`, `bans_count`, `n_allies_known`, `n_enemies_known`.
- **Pairwise synergy & counter**: реализовано в `build_dataset.py` через отдельный агрегатор по всем матчам (улучшение из чек-листа уже внедрено).
- **Item module**: `item_module.py` строит условные рекомендации по статистике покупок; если данных мало — fallback на `itemPopularity` с заниженным confidence.
- **Explainability**: CLI сообщает краткие правила (`synergy`, `counter`, `global winrate`). Дополнительно можно вызвать `--explain-top` для SHAP-подобного вывода (потребуется `shap`).

## Валидация и метрики

После `train_model.py` см. `models/training_metrics.json`:
- `val_auc`, `val_logloss`
- `val_top1`, `val_topK`

Для кастомного uplift (личные матчи):
1. Сформируйте матч-лист игрока через `fetch_opendota.py --steam-id ...` (можно расширить).
2. Соберите hold-out датасет.
3. Прогоните модель для каждого драфта, измерьте фактический winrate top-1 / top-3 vs baseline (random/meta). Инструкция описана в README, секция “Custom uplift”.

## Примеры сценариев

Файл `examples/scenarios.json` содержит несколько готовых драфтов. Запуск:
```bash
python predict_cli.py --scenario carry_vs_magic --show-items
```

## Персонализация и расширения

- **Player priors**: `predict_cli.py --player-profile my_profile.json` (см. `examples/player_profile.example.json`). Профиль может хранить любимые герои, avoid-лист, весовые коэффициенты на уровне ролей.
- **Регулярный retrain**: повторяйте связку `fetch -> build_dataset -> train_model`. Скрипты пере-пишут артефакты; держите версии по patch-id в отдельных подпапках.
- **Patch-awareness**: патч сохраняется как категориальный признак; можно добавить target encoding по winrate в `build_dataset.py`.
- **RAG для coach notes**:
  1. Сформируйте текстовые заметки / гайды.
  2. Прогоните через embeddings (например, `sentence-transformers`) и сохраните в Milvus/Pinecone/FAISS.
  3. В `predict_cli.py` добавьте блок, который по top hero достаёт релевантные заметки и печатает подсказку.

## Check-list улучшений

- [x] Pairwise synergy агрегатор (средний winrate по каждой паре союзников).
- [x] Counter score против известных врагов.
- [x] Conditional item рекомендации + fallback к meta.
- [ ] Добавить персональные фичи (например, winrate игрока на герое, lane preferences).
- [ ] Авто-переобучение по расписанию (cron + data snapshot).
- [ ] Векторное хранилище заметок (RAG) для объяснений.
- [ ] Расширенный item-модуль с учётом enemy lineups (item winrate vs конкретный враг).

## Скрипты валидации/отчёта

Для быстрого подсчёта метрик:
```bash
python train_model.py --processed-dir data/processed --models-dir models --topk 3
python - <<'PY'
import json
metrics = json.load(open('models/training_metrics.json'))
print(metrics)
PY
```

SHAP визуализации (один пример):
```bash
python predict_cli.py --scenario mid_vs_push --explain-top
```

## Частые вопросы

- **Нету Parquet?** Пайплайн сохраняет и CSV. `train_model.py` автоматически выберет CSV, если Parquet отсутствует.
- **Сколько времени занимает прогноз?** < 5 секунд на ноутбуке после загрузки модели (топ-N до 10 героев).
- **Нужно ли публиковать данные?** Нет, используйте локально; не делитесь личными профилями без разрешения.

## Безопасность и легальность

- Не внедрять скрипт в клиент.
- Не хранить/публиковать приватные данные игроков.
- Следить за условиями Valve/Steam.