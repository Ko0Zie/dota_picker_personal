from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from utils import RAW_DATA_DIR, PROCESSED_DIR, load_json


class ItemRecommender:
    """Lightweight conditional item recommender backed by aggregated stats."""

    def __init__(
        self,
        item_stats_path: Path = PROCESSED_DIR / "item_stats.json",
        hero_item_popularity_path: Path = RAW_DATA_DIR / "item_popularity.json",
    ):
        sample_item_pop = Path("data/sample/raw/item_popularity.json")
        self.item_stats_path = item_stats_path
        self.hero_item_popularity_path = hero_item_popularity_path
        self.hero_items = load_json(item_stats_path) if item_stats_path.exists() else {}
        if hero_item_popularity_path.exists():
            self.meta_popularity = load_json(hero_item_popularity_path)
        elif sample_item_pop.exists():
            self.meta_popularity = load_json(sample_item_pop)
        else:
            self.meta_popularity = {}

    def recommend(
        self,
        hero_id: int,
        enemy_ids: Optional[Sequence[int]] = None,
        role: Optional[str] = None,
        top_n: int = 5,
    ) -> List[Dict[str, float]]:
        entry = self.hero_items.get(str(hero_id))
        if entry:
            ranked = sorted(
                entry["items"].items(),
                key=lambda kv: kv[1]["count"],
                reverse=True,
            )
            payload = [
                {
                    "item": item_name,
                    "avg_time": round(item_stats.get("avg_time", 0.0), 1),
                    "confidence": min(1.0, item_stats.get("count", 0) / max(1, entry["total_matches"])),
                    "source": "match_item_timings",
                }
                for item_name, item_stats in ranked[:top_n]
            ]
            if payload:
                return payload
        return self._fallback(hero_id, top_n=top_n)

    def _fallback(self, hero_id: int, top_n: int = 5) -> List[Dict[str, float]]:
        meta = self.meta_popularity.get(str(hero_id))
        if not meta:
            return []
        sections = []
        for bucket in ["start_game_items", "early_game_items", "mid_game_items", "late_game_items"]:
            bucket_items = meta.get(bucket) or {}
            for item_name, stats in bucket_items.items():
                sections.append(
                    (
                        item_name,
                        stats.get("winrate", 0),
                        stats.get("games", 0),
                        stats.get("avg_time", 0.0),
                        bucket,
                    )
                )
        sections.sort(key=lambda row: row[1], reverse=True)
        payload = []
        for item_name, winrate, games, avg_time, bucket in sections[:top_n]:
            payload.append(
                {
                    "item": item_name,
                    "avg_time": avg_time or 0.0,
                    "confidence": min(1.0, games / 1000) if games else 0.2,
                    "source": f"meta_{bucket}",
                }
            )
        return payload


def main():
    recommender = ItemRecommender()
    hero_id = 1
    recs = recommender.recommend(hero_id, top_n=3)
    for rec in recs:
        print(rec)


if __name__ == "__main__":
    main()
