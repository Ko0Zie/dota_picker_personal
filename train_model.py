import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from utils import (
    MODEL_FEATURES,
    MODELS_DIR,
    PROCESSED_DIR,
    ensure_dir,
    get_logger,
    load_feature_manifest,
    topk_accuracy,
)


logger = get_logger("train_model")


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    if dataset_path.suffix == ".csv":
        return pd.read_csv(dataset_path)
    if dataset_path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(dataset_path)
    raise ValueError(f"Unsupported dataset format: {dataset_path}")


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM model for hero recommendation")
    parser.add_argument("--processed-dir", type=str, default=str(PROCESSED_DIR))
    parser.add_argument("--dataset-path", type=str, help="Override dataset path")
    parser.add_argument("--models-dir", type=str, default=str(MODELS_DIR))
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=64)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
    else:
        parquet_path = processed_dir / "drafts_dataset.parquet"
        csv_path = processed_dir / "drafts_dataset.csv"
        dataset_path = parquet_path if parquet_path.exists() else csv_path
    models_dir = Path(args.models_dir)
    ensure_dir(models_dir)

    df = load_dataset(dataset_path)
    if df.empty:
        raise ValueError("Dataset is empty.")
    feature_names = load_feature_manifest(processed_dir / "feature_manifest.json")
    features = feature_names or MODEL_FEATURES

    df = df.sort_values("match_start_time").reset_index(drop=True)
    df = df.dropna(subset=["win"])
    df = df.fillna(0)

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    categorical_cols = ["candidate_hero_id", "role", "patch"]
    train_set = lgb.Dataset(
        train_df[features],
        label=train_df["win"],
        categorical_feature=categorical_cols,
        free_raw_data=False,
    )
    val_set = lgb.Dataset(
        val_df[features],
        label=val_df["win"],
        reference=train_set,
        categorical_feature=categorical_cols,
        free_raw_data=False,
    )

    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 3,
        "min_data_in_leaf": 50,
        "verbosity": -1,
    }

    booster = lgb.train(
        params,
        train_set,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        num_boost_round=2000,
        early_stopping_rounds=100,
    )

    val_preds = booster.predict(val_df[features], num_iteration=booster.best_iteration)
    auc = roc_auc_score(val_df["win"], val_preds)
    ll = log_loss(val_df["win"], val_preds, eps=1e-6)
    topk = topk_accuracy(val_df["win"], val_preds, val_df["match_id"], k=args.topk)
    top1 = topk_accuracy(val_df["win"], val_preds, val_df["match_id"], k=1)

    metrics = {
        "val_auc": float(auc),
        "val_logloss": float(ll),
        "val_top1": float(top1),
        f"val_top{args.topk}": float(topk),
    }
    logger.info("Validation metrics: %s", metrics)

    model_path = models_dir / "hero_rec_model.txt"
    booster.save_model(model_path)
    metadata = {
        "feature_names": features,
        "categorical_features": categorical_cols,
        "params": params,
        "best_iteration": booster.best_iteration,
        "metrics": metrics,
        "dataset_path": str(dataset_path),
    }
    with open(models_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(models_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Model saved to %s", model_path)


if __name__ == "__main__":
    main()
