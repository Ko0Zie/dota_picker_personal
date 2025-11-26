import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import lightgbm as lgb
import pandas as pd
from rich.console import Console
from rich.table import Table

from item_module import ItemRecommender
from utils import (
    MODEL_FEATURES,
    MODELS_DIR,
    PROCESSED_DIR,
    RAW_DATA_DIR,
    aggregate_pairwise_score,
    build_hero_lookup,
    compute_confidence,
    filter_candidates,
    get_logger,
    hero_id_to_name,
    hero_identifier_to_id,
    load_feature_manifest,
    load_json,
    normalize_name,
)


console = Console()
logger = get_logger("predict_cli")


def parse_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        return value
    parts = [chunk.strip() for chunk in value.split(",")]
    return [p for p in parts if p]


def resolve_path(primary: Path, fallback: Path) -> Path:
    if primary.exists():
        return primary
    if fallback.exists():
        console.print(f"[yellow]Using fallback data from {fallback}[/yellow]")
        return fallback
    raise FileNotFoundError(f"Missing required file: {primary}")


def parse_hero_inputs(items: List[str], hero_lookup: Dict[str, dict]) -> List[int]:
    ids = []
    for item in items:
        hero_id = hero_identifier_to_id(item, hero_lookup)
        if hero_id is None:
            console.print(f"[red]Unknown hero: {item}[/red]")
            continue
        ids.append(hero_id)
    return ids


def load_player_profile(path: Optional[Path]) -> dict:
    if not path or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_preferences(score: float, hero_id: int, profile: dict) -> float:
    boost = 0.0
    preferred = profile.get("preferred_heroes", [])
    avoid = profile.get("avoid_heroes", [])
    if preferred and hero_id in preferred:
        boost += 0.02
    if avoid and hero_id in avoid:
        boost -= 0.02
    role_weights = profile.get("role_weights", {})
    if role_weights:
        boost += role_weights.get(str(hero_id), 0.0)
    return score + boost


def build_features_for_candidate(
    hero_id: int,
    role: str,
    patch: str,
    hero_stats: Dict[int, dict],
    synergy: dict,
    counters: dict,
    allies: List[int],
    enemies: List[int],
    bans_count: int,
    skill_bracket: int,
) -> dict:
    hero_entry = hero_stats.get(hero_id, {})
    hero_global_winrate = hero_entry.get("hero_global_winrate", hero_entry.get("wins", 0) / hero_entry.get("games", 1) if hero_entry.get("games") else 0.5)
    hero_games = hero_entry.get("games", 0)
    synergy_score, synergy_samples = aggregate_pairwise_score(hero_id, allies, synergy)
    counter_score, counter_samples = aggregate_pairwise_score(hero_id, enemies, counters)
    features = {
        "candidate_hero_id": str(hero_id),
        "role": role,
        "patch": patch,
        "hero_global_winrate": hero_global_winrate,
        "hero_games_count": hero_games,
        "pairwise_synergy_score": synergy_score,
        "counter_score": counter_score,
        "avg_item_timing_first": hero_entry.get("avg_item_timing_first", 0.0),
        "avg_item_timing_second": hero_entry.get("avg_item_timing_second", 0.0),
        "skill_bracket": skill_bracket,
        "bans_count": bans_count,
        "n_allies_known": len(allies),
        "n_enemies_known": len(enemies),
        "synergy_samples": synergy_samples,
        "counter_samples": counter_samples,
    }
    return features


def explain_candidate(candidate: dict, hero_name: str, allies: List[int], enemies: List[int], hero_name_lookup: Dict[int, str]) -> List[str]:
    reasons = []
    synergy_score = candidate["pairwise_synergy_score"]
    counter_score = candidate["counter_score"]
    if allies and synergy_score:
        ally_names = ", ".join(hero_name_lookup[a] for a in allies if a in hero_name_lookup)
        reasons.append(f"+ synergy ({synergy_score:.2f}) with {ally_names}")
    if enemies and counter_score:
        enemy_names = ", ".join(hero_name_lookup[e] for e in enemies if e in hero_name_lookup)
        reasons.append(f"+ counters ({counter_score:.2f}) vs {enemy_names}")
    if candidate["hero_global_winrate"]:
        reasons.append(f"Global winrate {candidate['hero_global_winrate']:.2f}")
    if not reasons:
        reasons.append("Data-driven pick based on historical win rate.")
    return reasons[:2]


def load_scenario(name: str, scenarios_path: Path) -> dict:
    scenarios = load_json(scenarios_path)
    for scenario in scenarios:
        if scenario["name"] == name:
            return scenario
    raise ValueError(f"Scenario {name} not found")


def main():
    parser = argparse.ArgumentParser(description="CLI for Dota hero recommendation")
    parser.add_argument("--role", required=False, help="carry/mid/offlane/support/hard_support")
    parser.add_argument("--enemies", help="Comma separated enemy heroes")
    parser.add_argument("--allies", help="Known ally heroes")
    parser.add_argument("--bans", help="Banned heroes")
    parser.add_argument("--patch", default="latest")
    parser.add_argument("--skill", type=int, default=4, help="Skill bracket (1-8)")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--player-profile", type=str, help="Path to JSON with preferences")
    parser.add_argument("--scenario", help="Load inputs from examples/scenarios.json")
    parser.add_argument("--show-items", action="store_true", help="Print item build suggestions")
    parser.add_argument("--explain-top", action="store_true", help="Compute SHAP style explanations for top pick")
    args = parser.parse_args()

    scenarios_path = Path("examples/scenarios.json")
    if args.scenario and scenarios_path.exists():
        scenario = load_scenario(args.scenario, scenarios_path)
        args.role = scenario.get("role", args.role)
        args.enemies = ",".join(scenario.get("enemy_picks", []))
        args.allies = ",".join(scenario.get("ally_picks", []))
        args.bans = ",".join(scenario.get("bans", []))
        args.patch = scenario.get("patch", args.patch)

    role = normalize_name(args.role or "carry")
    enemies_input = parse_list(args.enemies)
    allies_input = parse_list(args.allies)
    bans_input = parse_list(args.bans)

    heroes_path = resolve_path(RAW_DATA_DIR / "heroes.json", Path("data/sample/raw/heroes.json"))
    heroes = load_json(heroes_path)
    hero_lookup_by_name, hero_lookup_by_id = build_hero_lookup(heroes)
    hero_name_lookup = {hid: hero_id_to_name(hid, hero_lookup_by_id) for hid in hero_lookup_by_id}

    enemy_ids = parse_hero_inputs(enemies_input, hero_lookup_by_name)
    ally_ids = parse_hero_inputs(allies_input, hero_lookup_by_name)
    ban_ids = parse_hero_inputs(bans_input, hero_lookup_by_name)
    bans_count = len(ban_ids)
    taken_ids = enemy_ids + ally_ids

    processed_dir = PROCESSED_DIR if (PROCESSED_DIR / "hero_stats.json").exists() else Path("data/sample/processed")
    hero_stats_path = processed_dir / "hero_stats.json"
    synergy_path = processed_dir / "synergy_stats.json"
    counter_path = processed_dir / "counter_stats.json"

    hero_stats_raw = load_json(hero_stats_path)
    hero_stats = {int(k): v for k, v in hero_stats_raw.items()}
    synergy = load_json(synergy_path)
    counters = load_json(counter_path)

    item_recommender = ItemRecommender(
        item_stats_path=processed_dir / "item_stats.json",
        hero_item_popularity_path=RAW_DATA_DIR / "item_popularity.json",
    )

    models_dir = MODELS_DIR
    model_path = models_dir / "hero_rec_model.txt"
    if not model_path.exists():
        model_path = Path("models/hero_rec_model.txt")
    if not model_path.exists():
        raise FileNotFoundError("Trained model not found. Please run train_model.py first.")
    booster = lgb.Booster(model_file=str(model_path))
    feature_names = load_feature_manifest(processed_dir / "feature_manifest.json")
    features = feature_names or MODEL_FEATURES

    candidates = filter_candidates(heroes, role, ban_ids, taken_ids)
    if not candidates:
        console.print("[red]No candidates available. Check inputs or run build_dataset first.[/red]")
        return

    rows = []
    for hero in candidates:
        candidate_features = build_features_for_candidate(
            hero["id"],
            role,
            args.patch,
            hero_stats,
            synergy,
            counters,
            ally_ids,
            enemy_ids,
            bans_count,
            args.skill,
        )
        rows.append(candidate_features)
    df = pd.DataFrame(rows)
    missing_cols = [c for c in features if c not in df.columns]
    for col in missing_cols:
        df[col] = 0

    predictions = booster.predict(df[features])
    profile = load_player_profile(Path(args.player_profile)) if args.player_profile else {}
    df["win_prob"] = predictions
    df["adjusted_prob"] = [
        apply_preferences(prob, int(row["candidate_hero_id"]), profile) for prob, row in zip(predictions, rows)
    ]
    df["confidence"] = [
        compute_confidence(hero_stats.get(int(row["candidate_hero_id"]), {}).get("games", 0))
        for row in rows
    ]
    df["hero_id"] = [int(row["candidate_hero_id"]) for row in rows]
    df["hero_name"] = [hero_name_lookup.get(hid, str(hid)) for hid in df["hero_id"]]
    df["reasons"] = [
        explain_candidate(row, name, ally_ids, enemy_ids, hero_name_lookup)
        for row, name in zip(rows, df["hero_name"])
    ]

    df = df.sort_values("adjusted_prob", ascending=False).head(args.top_n)

    table = Table(show_lines=True)
    table.add_column("Rank", justify="right")
    table.add_column("Hero")
    table.add_column("Win Prob", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Reasons")
    for idx, row in enumerate(df.itertuples(), start=1):
        table.add_row(
            str(idx),
            row.hero_name,
            f"{row.adjusted_prob:.3f}",
            f"{row.confidence:.2f}",
            "; ".join(row.reasons),
        )
    console.print(table)

    if args.show_items:
        for row in df.itertuples():
            recs = item_recommender.recommend(row.hero_id, enemy_ids, role, top_n=4)
            if not recs:
                continue
            console.print(f"[bold]{row.hero_name}[/bold] item plan:")
            for rec in recs:
                console.print(
                    f" - {rec['item']} @ {rec['avg_time'] // 60:.0f}m (confidence {rec['confidence']:.2f}, {rec['source']})"
                )

    if args.explain_top:
        try:
            import shap
        except ImportError:
            console.print("[yellow]Install shap to enable detailed explanations.[/yellow]")
        else:
            top_row = df.iloc[0]
            sample = pd.DataFrame([top_row[features]])
            explainer = shap.TreeExplainer(booster)
            shap_values = explainer.shap_values(sample)
            feature_importances = sorted(
                zip(features, shap_values[0][0]),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:5]
            console.print(f"[green]Top feature contributions for {top_row.hero_name}:[/green]")
            for feature, value in feature_importances:
                console.print(f" - {feature}: {value:+.4f}")


if __name__ == "__main__":
    main()
