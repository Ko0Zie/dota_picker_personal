import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

from utils import RAW_DATA_DIR, ensure_dir, get_logger, save_json, save_jsonl


API_BASE = "https://api.opendota.com/api"
RATE_LIMIT_SLEEP = 1.1  # seconds
logger = get_logger("fetch_opendota")


def setup_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fetch_json(session: requests.Session, endpoint: str, params: Optional[dict] = None):
    url = f"{API_BASE}/{endpoint}"
    response = session.get(url, params=params, timeout=30)
    if response.status_code == 429:
        logger.warning("Hit rate limit, sleeping for %.1fs", RATE_LIMIT_SLEEP)
        time.sleep(RATE_LIMIT_SLEEP)
        response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_heroes(session: requests.Session, output_dir: Path) -> Path:
    logger.info("Fetching hero metadata")
    heroes = fetch_json(session, "heroes")
    out_path = output_dir / "heroes.json"
    save_json(heroes, out_path)
    return out_path


def fetch_hero_stats(session: requests.Session, output_dir: Path) -> Path:
    logger.info("Fetching hero stats (global winrates)")
    stats = fetch_json(session, "heroStats")
    out_path = output_dir / "hero_stats.json"
    save_json(stats, out_path)
    return out_path


def fetch_item_popularity(session: requests.Session, hero_ids: List[int], output_dir: Path) -> Path:
    logger.info("Fetching per-hero item popularity (this can take a while)")
    results = {}
    for hero_id in tqdm(hero_ids, desc="itemPopularity"):
        try:
            payload = fetch_json(session, f"heroes/{hero_id}/itemPopularity")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch itemPopularity for hero %s: %s", hero_id, exc)
            continue
        results[str(hero_id)] = payload
        time.sleep(0.5)
    out_path = output_dir / "item_popularity.json"
    save_json(results, out_path)
    return out_path


def fetch_match_ids(
    session: requests.Session,
    patch: Optional[str],
    min_mmr: Optional[int],
    num_matches: int,
) -> List[int]:
    logger.info("Fetching match ids via publicMatches (target=%s)", num_matches)
    collected = []
    cursor = None
    while len(collected) < num_matches:
        params = {"mmr_descending": True}
        if cursor:
            params["less_than_match_id"] = cursor
        if patch:
            params["patch"] = patch
        if min_mmr:
            params["min_mmr"] = min_mmr
        batch = fetch_json(session, "publicMatches", params=params)
        if not batch:
            break
        for match in batch:
            match_id = match.get("match_id")
            if match_id:
                collected.append(match_id)
        cursor = batch[-1].get("match_id")
        logger.info("Collected %s/%s match ids", len(collected), num_matches)
        time.sleep(RATE_LIMIT_SLEEP)
    return collected[:num_matches]


def fetch_matches(
    session: requests.Session,
    match_ids: List[int],
    output_dir: Path,
) -> Path:
    ensure_dir(output_dir)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"matches_{timestamp}.jsonl"
    records = []
    for match_id in tqdm(match_ids, desc="matches"):
        try:
            data = fetch_json(session, f"matches/{match_id}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch match %s: %s", match_id, exc)
            continue
        records.append(data)
        time.sleep(RATE_LIMIT_SLEEP)
    save_jsonl(records, out_path)
    logger.info("Saved %s matches to %s", len(records), out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download Dota data from OpenDota")
    parser.add_argument("--num-matches", type=int, default=2000)
    parser.add_argument("--patch", type=str, help="Patch id filter (e.g. 7.36c)")
    parser.add_argument("--min-mmr", type=int, help="Optional mmr threshold")
    parser.add_argument("--output-dir", type=str, default=str(RAW_DATA_DIR))
    parser.add_argument("--skip-item-popularity", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    session = setup_session()

    heroes_path = fetch_heroes(session, output_dir)
    hero_stats_path = fetch_hero_stats(session, output_dir)
    heroes = fetch_json(session, "heroes")
    hero_ids = [hero["id"] for hero in heroes]

    if not args.skip_item_popularity:
        fetch_item_popularity(session, hero_ids, output_dir)

    match_ids = fetch_match_ids(session, args.patch, args.min_mmr, args.num_matches)
    matches_path = fetch_matches(session, match_ids, output_dir / "matches")

    metadata = {
        "heroes_path": str(heroes_path),
        "hero_stats_path": str(hero_stats_path),
        "matches_path": str(matches_path),
        "patch_filter": args.patch,
        "min_mmr": args.min_mmr,
        "num_matches": len(match_ids),
        "generated_utc": datetime.utcnow().isoformat(),
    }
    save_json(metadata, output_dir / "fetch_metadata.json")
    logger.info("Done. Metadata saved to %s", output_dir / "fetch_metadata.json")


if __name__ == "__main__":
    main()
