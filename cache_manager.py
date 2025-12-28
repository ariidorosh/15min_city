from __future__ import annotations

import glob
import json
import os
import re
import shutil
from typing import Dict, List, Optional, Tuple

from logger_config import logger
from paths import (
    EXTRA_SEARCH_DIRS,
    DIR_META,
    DIR_BOUNDARIES,
    DIR_GRAPHS,
    DIR_POI_ALL,
)
from utils import ensure_dir, norm_name, tokens

CITIES_CACHE_FILE = os.path.join(DIR_META, "cities.json")


def locate_and_deploy_cached_file_debug(
    expected_path: str,
    suffix: str,
    place: str,
    required_tokens: Optional[List[str]] = None,
    extra_search_dirs: Optional[List[str]] = None,
) -> Tuple[Optional[str], Dict[str, object]]:
    """
    Шукає кеш-файл з суфіксом `suffix`:
    - якщо expected_path існує -> повертає його
    - інакше шукає в директорії expected_path та EXTRA_SEARCH_DIRS
    - якщо знаходить — копіює в expected_path (якщо вдається), або повертає знайдений шлях

    Повертає: (path_to_read, info)
    """
    info: Dict[str, object] = {
        "expected_path": expected_path,
        "used_path": None,
        "found": False,
        "action": "not_found",
        "candidates": [],
    }

    if os.path.exists(expected_path):
        info.update({"used_path": expected_path, "found": True, "action": "exists"})
        return expected_path, info

    place_tokens = tokens(place)
    required = [t.lower() for t in (required_tokens or []) if t]

    search_dirs = [os.path.dirname(expected_path)] + (extra_search_dirs or EXTRA_SEARCH_DIRS)

    scored: List[Tuple[int, str]] = []
    for base in search_dirs:
        pattern = os.path.join(base, f"*{suffix}")
        for p in glob.glob(pattern):
            fname = norm_name(os.path.basename(p))
            score = sum(1 for t in place_tokens if t and t in fname)

            if required and not all(rt in fname for rt in required):
                continue

            scored.append((score, p))

    scored.sort(key=lambda x: (-x[0], x[1]))
    info["candidates"] = [p for _, p in scored]

    if not scored or scored[0][0] <= 0:
        return None, info

    best_score, best = scored[0]
    try:
        ensure_dir(os.path.dirname(expected_path))
        shutil.copy2(best, expected_path)
        logger.info("Deployed external cache %s -> %s (score=%d)", best, expected_path, best_score)
        info.update({"used_path": expected_path, "found": True, "action": "copied"})
        return expected_path, info
    except Exception as e:
        logger.warning("Failed to copy external cache %s -> %s: %s", best, expected_path, e)
        info.update({"used_path": best, "found": True, "action": "fallback_read"})
        return best, info


def load_cached_cities(cache_file: str = CITIES_CACHE_FILE) -> List[str]:
    """
    Читає список міст зі службового cities.json.
    Повертає [] якщо файла нема або він пошкоджений.
    """
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return (json.load(f) or {}).get("cities", []) or []
        except Exception:
            return []
    return []


def save_cached_cities(cities: List[str], cache_file: str = CITIES_CACHE_FILE) -> None:
    """Записує cities.json (безпечно створює DIR_META)."""
    ensure_dir(os.path.dirname(cache_file))
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"cities": list(cities or [])}, f, ensure_ascii=False, indent=2)


def _unsafename_to_place(safe_base: str) -> str:
    # “Зворотнє” від safe_name() не 1-в-1, тож робимо простий human-friendly варіант.
    return (safe_base or "").replace("_", " ").strip()


def discover_cached_cities() -> List[str]:
    """
    Автоматично знаходить міста, для яких вже є кеш-файли:
    - *_poi_all.geojson
    - *_boundary.geojson
    - *.graphml
    """
    found = set()

    for p in glob.glob(os.path.join(DIR_POI_ALL, "*_poi_all.geojson")):
        base = os.path.basename(p)
        place_safe = re.sub(r"_poi_all\.geojson$", "", base)
        found.add(_unsafename_to_place(place_safe))

    for p in glob.glob(os.path.join(DIR_BOUNDARIES, "*_boundary.geojson")):
        base = os.path.basename(p)
        place_safe = re.sub(r"_boundary\.geojson$", "", base)
        found.add(_unsafename_to_place(place_safe))

    for p in glob.glob(os.path.join(DIR_GRAPHS, "*.graphml")):
        base = os.path.basename(p)
        place_safe = re.sub(r"\.graphml$", "", base)
        found.add(_unsafename_to_place(place_safe))

    return sorted({c for c in found if c})
