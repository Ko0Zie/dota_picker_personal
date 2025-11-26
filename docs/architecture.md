## System Architecture

The local assistant is structured as a reproducible, file based pipeline that can be run end to end or stepwise inside VSCode.

### Data layer
- `fetch_opendota.py` downloads raw payloads from OpenDota (`heroes`, `itemPopularity`, `publicMatches`, optional `players/<id>/matches`) and stores them as JSON/JSONL under `data/raw/`.
- Raw data is versioned by patch and timestamp so the system can be quickly re-trained as soon as new patch data is available.

### Feature & dataset layer
- `build_dataset.py` reads raw JSON/JSONL inputs, converts them into a candidate-per-hero table, and writes `data/processed/drafts_dataset.parquet`.
- Features include hero level stats (global winrate, games count), player context (role, skill bracket, n allies known, bans count), and interaction signals (pairwise synergy with allies, counter score vs. each known enemy). Average item timings and patch target encoding are also produced in this step.

### Modeling layer
- `train_model.py` splits the dataset chronologically by `match_patch`/`start_time` and trains a LightGBM ranking/classification model to estimate `win_prob`.
- Model artifacts (LightGBM Booster, feature manifest, label encoders) are stored in `models/` for reproducibility.
- A separate `item_module.py` handles conditional item recommendation by aggregating purchase stats from raw data and exposing a callable interface that predict_cli can reuse.

### Serving / CLI layer
- `predict_cli.py` loads the trained model, feature manifest, hero metadata, and item module outputs to produce a ranked list of candidate heroes for the provided draft context.
- The CLI reports top-N heroes with `win_prob`, confidence (sample count for the slice), recommended item builds with timing, and brief textual explanations referencing synergy/counter scores or meta priors.
- Optional SHAP explanations (if `shap` is installed) can attribute the most influential features for a single recommendation.

### Vector notes (optional)
- Coach / analyst notes can be embedded (sentence-transformers) and stored in a vector DB (Milvus/Pinecone/FAISS). The CLI can fetch relevant notes for the chosen hero to enrich explanations. Hooks for this are provided but kept optional to preserve the offline workflow.
