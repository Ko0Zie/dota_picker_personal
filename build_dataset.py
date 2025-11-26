import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import (
    MODEL_FEATURES,
    PROCESSED_DIR,
    RAW_DATA_DIR,
    aggregate_pairwise_score,
    ensure_dir,
    get_logger,
    load_json,
    load_jsonl,
    save_feature_manifest,
    save_json,
    safe_mean,
    role_from_player,
)


logger = get_logger("build_dataset")


def normalize_role(role: str) -> str:
    mapping = {
        "safe_lane": "carry",
        "unknown": "support",
        "jungle": "support",
    }
    return mapping.get(role, role)


class FeatureAggregator:
    def __init__(self):
        self.hero_totals: Dict[int, Dict[str, int]] = defaultdict(
            lambda: {"wins": 0, "games": 0}
        )
        self.synergy: Dict[str, Dict[str, int]] = defaultdict(lambda: {"wins": 0, "games": 0})
        self.counter: Dict[str, Dict[str, int]] = defaultdict(lambda: {"wins": 0, "games": 0})
        self.item_timings: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    def ingest_match(self, match: dict) -> None:
        players = match.get("players", [])
        radiant_ids = [p.get("hero_id") for p in players if p.get("isRadiant")]
        dire_ids = [p.get("hero_id") for p in players if not p.get("isRadiant")]
        for player in players:
            hero_id = player.get("hero_id")
            if not hero_id:
                continue
            is_radiant = player.get("isRadiant")
            did_win = bool(match.get("radiant_win")) == bool(is_radiant)
            self.hero_totals[hero_id]["games"] += 1
            if did_win:
                self.hero_totals[hero_id]["wins"] += 1
            allies = radiant_ids if is_radiant else dire_ids
            enemies = dire_ids if is_radiant else radiant_ids
            self._update_interactions(hero_id, allies, did_win, table=self.synergy, mirror=True)
            self._update_interactions(hero_id, enemies, did_win, table=self.counter, mirror=False)
            self._update_items(hero_id, player.get("purchase_log", []))

    def _update_interactions(
        self,
        hero_id: int,
        others: Iterable[int],
        did_win: bool,
        table: Dict[str, Dict[str, int]],
        mirror: bool,
    ) -> None:
        for other_id in others:
            if not other_id or other_id == hero_id:
                continue
            key = f"{hero_id}:{other_id}"
            table[key]["games"] += 1
            if did_win:
                table[key]["wins"] += 1
            if mirror:
                mirrored = f"{other_id}:{hero_id}"
                table[mirrored]["games"] += 1
                if did_win:
                    table[mirrored]["wins"] += 1

    def _update_items(self, hero_id: int, purchase_log: List[dict]) -> None:
        for event in purchase_log:
            item = event.get("key")
            time_val = event.get("time")
            if not item or time_val is None or time_val < 0:
                continue
            self.item_timings[hero_id][item].append(time_val)


def compute_item_features(aggregator: FeatureAggregator) -> Dict[int, Dict[str, float]]:
    features = {}
    for hero_id, items in aggregator.item_timings.items():
        all_first = []
        all_second = []
        for times in items.values():
            sorted_times = sorted(times)
            if sorted_times:
                all_first.append(sorted_times[0])
            if len(sorted_times) > 1:
                all_second.append(sorted_times[1])
        features[hero_id] = {
            "avg_item_timing_first": safe_mean(all_first),
            "avg_item_timing_second": safe_mean(all_second),
        }
    return features


def build_rows(
    matches: List[dict],
    hero_stats_map: Dict[int, dict],
    aggregator: FeatureAggregator,
) -> pd.DataFrame:
    rows = []
    synergy_table = aggregator.synergy
    counter_table = aggregator.counter
    item_features = compute_item_features(aggregator)
    for match in tqdm(matches, desc="rows"):
        match_id = match.get("match_id")
        patch = str(match.get("patch"))
        skill = match.get("skill") or 0
        start_time = match.get("start_time")
        picks_bans = match.get("picks_bans") or []
        bans_count = sum(1 for pb in picks_bans if not pb.get("is_pick"))
        players = match.get("players", [])
        radiant_ids = [p.get("hero_id") for p in players if p.get("isRadiant")]
        dire_ids = [p.get("hero_id") for p in players if not p.get("isRadiant")]
        for idx, player in enumerate(players):
            hero_id = player.get("hero_id")
            if not hero_id:
                continue
            is_radiant = player.get("isRadiant")
            did_win = bool(match.get("radiant_win")) == bool(is_radiant)
            allies = radiant_ids if is_radiant else dire_ids
            enemies = dire_ids if is_radiant else radiant_ids
            synergy_score, _ = aggregate_pairwise_score(hero_id, [a for a in allies if a != hero_id], synergy_table)
            counter_score, _ = aggregate_pairwise_score(hero_id, enemies, counter_table)
            hero_stats = hero_stats_map.get(hero_id, {})
            hero_games = hero_stats.get("games", aggregator.hero_totals[hero_id]["games"])
            hero_wins = hero_stats.get("wins", aggregator.hero_totals[hero_id]["wins"])
            global_wr = hero_wins / hero_games if hero_games else 0.5
            item_feat = item_features.get(hero_id, {})
            role = normalize_role(role_from_player(player))
            row = {
                "match_id": match_id,
                "sample_idx": f"{match_id}_{idx}",
                "candidate_hero_id": str(hero_id),
                "role": role,
                "patch": patch,
                "hero_global_winrate": global_wr,
                "hero_games_count": hero_games,
                "pairwise_synergy_score": synergy_score,
                "counter_score": counter_score,
                "avg_item_timing_first": item_feat.get("avg_item_timing_first", 0.0),
                "avg_item_timing_second": item_feat.get("avg_item_timing_second", 0.0),
                "skill_bracket": skill,
                "bans_count": bans_count,
                "n_allies_known": len([a for a in allies if a]),
                "n_enemies_known": len([e for e in enemies if e]),
                "win": int(did_win),
                "team_is_radiant": int(bool(is_radiant)),
                "match_start_time": start_time,
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    return df


def load_matches(matches_dir: Path) -> List[dict]:
    records = []
    for path in sorted(matches_dir.glob("*.jsonl")):
        logger.info("Loading %s", path.name)
        records.extend(load_jsonl(path))
    return records


def ingest_hero_stats(hero_stats_path: Path) -> Dict[int, dict]:
    stats = load_json(hero_stats_path) if hero_stats_path.exists() else []
    mapping = {}
    for hero in stats:
        hero_id = hero.get("id") or hero.get("hero_id")
        if not hero_id:
            continue
        wins = hero.get("pro_win") or hero.get("turbo_win") or hero.get("1_win")
        picks = hero.get("pro_pick") or hero.get("turbo_pick") or hero.get("1_pick")
        mapping[int(hero_id)] = {
            "wins": wins or 0,
            "games": picks or 0,
        }
    return mapping


def persist_artifacts(
    df: pd.DataFrame,
    aggregator: FeatureAggregator,
    processed_dir: Path,
) -> None:
    ensure_dir(processed_dir)
    dataset_path = processed_dir / "drafts_dataset.parquet"
    df.to_parquet(dataset_path, index=False)
    df.to_csv(processed_dir / "drafts_dataset.csv", index=False)
    save_feature_manifest(MODEL_FEATURES, processed_dir / "feature_manifest.json")
    item_feature_map = compute_item_features(aggregator)
    hero_stats_payload = {}
    for hero_id, stats in aggregator.hero_totals.items():
        games = stats["games"]
        wins = stats["wins"]
        hero_stats_payload[str(hero_id)] = {
            "wins": wins,
            "games": games,
            "hero_global_winrate": wins / games if games else 0.5,
            "avg_item_timing_first": item_feature_map.get(hero_id, {}).get("avg_item_timing_first", 0.0),
            "avg_item_timing_second": item_feature_map.get(hero_id, {}).get("avg_item_timing_second", 0.0),
        }
    save_json(hero_stats_payload, processed_dir / "hero_stats.json")
    save_json(
        aggregator.synergy,
        processed_dir / "synergy_stats.json",
    )
    save_json(
        aggregator.counter,
        processed_dir / "counter_stats.json",
    )
    item_payload = {}
    for hero_id, items in aggregator.item_timings.items():
        item_payload[str(hero_id)] = {
            "items": {
                item: {
                    "count": len(times),
                    "avg_time": float(np.mean(times)),
                }
                for item, times in items.items()
            },
            "total_matches": aggregator.hero_totals[hero_id]["games"],
        }
    save_json(item_payload, processed_dir / "item_stats.json")
    logger.info("Artifacts persisted under %s", processed_dir)


def main():
    parser = argparse.ArgumentParser(description="Build ML dataset from raw OpenDota matches")
    parser.add_argument("--raw-dir", type=str, default=str(RAW_DATA_DIR))
    parser.add_argument("--processed-dir", type=str, default=str(PROCESSED_DIR))
    parser.add_argument("--matches-subdir", type=str, default="matches")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    matches_dir = raw_dir / args.matches_subdir
    ensure_dir(processed_dir)

    hero_stats_map = ingest_hero_stats(raw_dir / "hero_stats.json")
    matches = load_matches(matches_dir)
    logger.info("Loaded %s matches", len(matches))

    aggregator = FeatureAggregator()
    for match in tqdm(matches, desc="aggregate"):
        aggregator.ingest_match(match)

    df = build_rows(matches, hero_stats_map, aggregator)
    persist_artifacts(df, aggregator, processed_dir)


if __name__ == "__main__":
    main()
