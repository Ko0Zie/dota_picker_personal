import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
CONFIGS_DIR = ROOT_DIR / "configs"

SUPPORTED_ROLES = ["carry", "mid", "offlane", "support", "hard_support"]
MODEL_FEATURES = [
    "candidate_hero_id",
    "role",
    "patch",
    "hero_global_winrate",
    "hero_games_count",
    "pairwise_synergy_score",
    "counter_score",
    "avg_item_timing_first",
    "avg_item_timing_second",
    "skill_bracket",
    "bans_count",
    "n_allies_known",
    "n_enemies_known",
]


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(name)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(payload: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_jsonl(path: Path) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_jsonl(records: Sequence[dict], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec))
            f.write("\n")


def normalize_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def build_hero_lookup(heroes: Sequence[dict]) -> Tuple[Dict[str, dict], Dict[int, dict]]:
    by_name = {}
    by_id = {}
    for hero in heroes:
        localized = normalize_name(hero.get("localized_name", ""))
        full = normalize_name(hero.get("name", ""))
        hero_id = hero["id"]
        by_id[hero_id] = hero
        by_name[localized] = hero
        by_name[full] = hero
    return by_name, by_id


def hero_id_to_name(hero_id: int, hero_by_id: Dict[int, dict]) -> str:
    hero = hero_by_id.get(hero_id)
    if not hero:
        return f"Hero {hero_id}"
    return hero.get("localized_name") or hero.get("name") or f"Hero {hero_id}"


def hero_identifier_to_id(identifier: str, hero_lookup: Dict[str, dict]) -> Optional[int]:
    key = normalize_name(identifier)
    hero = hero_lookup.get(key)
    if hero:
        return hero["id"]
    if identifier.isdigit():
        return int(identifier)
    return None


def aggregate_pairwise_score(
    subject_id: int,
    other_ids: Iterable[int],
    table: Dict[str, dict],
) -> Tuple[float, int]:
    if not other_ids:
        return 0.0, 0
    total_score = 0.0
    total_samples = 0
    for other_id in other_ids:
        key = f"{subject_id}:{other_id}"
        entry = table.get(key) or table.get(f"{other_id}:{subject_id}")
        if not entry:
            continue
        games = entry.get("games", 0)
        if games == 0:
            continue
        win_rate = entry.get("wins", 0) / games
        total_score += win_rate
        total_samples += games
    if total_samples == 0:
        return 0.0, 0
    return total_score / max(1, len(list(other_ids))), total_samples


def load_feature_artifacts(processed_dir: Path = PROCESSED_DIR) -> dict:
    artifacts = {}
    for name in ["hero_stats", "synergy_stats", "counter_stats", "item_stats"]:
        path = processed_dir / f"{name}.json"
        if path.exists():
            artifacts[name] = load_json(path)
    return artifacts


def role_from_player(player: dict) -> str:
    lane_role = player.get("lane_role")
    if lane_role == 1:
        return "hard_support"
    if lane_role == 2:
        return "support"
    if lane_role == 3:
        return "offlane"
    if lane_role == 4:
        return "mid"
    if lane_role == 5:
        return "carry"
    lane = player.get("lane")
    if lane == 1:
        return "safe_lane"
    if lane == 2:
        return "mid"
    if lane == 3:
        return "offlane"
    return "unknown"


def hero_roles(hero: dict) -> List[str]:
    roles = hero.get("roles") or []
    return [normalize_name(r) for r in roles]


def safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def load_config(path: Optional[Path] = None) -> dict:
    if not path:
        default_path = CONFIGS_DIR / "settings.yaml"
        return load_yaml(default_path) if default_path.exists() else {}
    return load_yaml(path)


def load_yaml(path: Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_feature_manifest(feature_names: Sequence[str], path: Path) -> None:
    payload = {"feature_names": list(feature_names)}
    save_json(payload, path)


def load_feature_manifest(path: Path) -> List[str]:
    data = load_json(path)
    return data.get("feature_names", [])


def topk_accuracy(
    y_true: Sequence[int], y_pred: Sequence[float], groups: Sequence[Any], k: int = 3
) -> float:
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": groups})
    correct = 0
    total = 0
    for _, group_df in df.groupby("group"):
        total += 1
        preds = group_df.sort_values("y_pred", ascending=False).head(k)
        if preds["y_true"].max() > 0:
            correct += 1
    return correct / total if total else 0.0


def compute_confidence(sample_count: int) -> float:
    if sample_count <= 0:
        return 0.05
    capped = min(sample_count, 2000)
    return round(0.05 + (capped / 2000) * 0.95, 3)


def filter_candidates(
    heroes: Sequence[dict],
    role: str,
    banned_ids: Sequence[int],
    taken_ids: Sequence[int],
) -> List[dict]:
    candidates = []
    ban_set = set(banned_ids)
    taken_set = set(taken_ids)
    role_norm = normalize_name(role)
    for hero in heroes:
        if hero["id"] in ban_set or hero["id"] in taken_set:
            continue
        roles = hero_roles(hero)
        if roles and role_norm not in roles:
            continue
        candidates.append(hero)
    return candidates


def slice_item_timings(purchase_log: List[dict]) -> Tuple[Optional[float], Optional[float]]:
    if not purchase_log:
        return None, None
    sorted_log = sorted(
        [item for item in purchase_log if item.get("time") is not None],
        key=lambda x: x["time"],
    )
    first = sorted_log[0]["time"] if sorted_log else None
    second = sorted_log[1]["time"] if len(sorted_log) > 1 else None
    return first, second
