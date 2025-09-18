import os
import json
import hashlib
import glob
import shutil
import re
from typing import Dict, List, Iterable, Tuple, Optional

import osmnx as ox
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from config import POI_CATEGORIES
from logger_config import logger

# =========================
# Директорії даних
# =========================
DATA_ROOT = "data"
DIR_POI_ALL = os.path.join(DATA_ROOT, "poi", "all")
DIR_LEVELS = os.path.join(DATA_ROOT, "poi", "levels")
DIR_BOUNDARIES = os.path.join(DATA_ROOT, "boundaries")
DIR_CACHE = os.path.join(DATA_ROOT, "cache")
EXTRA_SEARCH_DIRS = [os.getcwd(), "/mnt/data"]

for d in (DIR_POI_ALL, DIR_LEVELS, DIR_BOUNDARIES, DIR_CACHE):
    os.makedirs(d, exist_ok=True)

# =========================
# Налаштування OSMnx
# =========================
ox.settings.use_cache = True
ox.settings.cache_folder = DIR_CACHE
ox.settings.log_console = False


# =========================
# Утиліти
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_name(s: str) -> str:
    return s.lower().replace(",", "").replace(" ", "_")


def _norm_name(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-zа-яіїєґ0-9]+", "_", s)
    return s.strip("_")


def _paths(place: str) -> Tuple[str, str]:
    safe = _safe_name(place)
    return (
        os.path.join(DIR_BOUNDARIES, f"{safe}_boundary.geojson"),
        os.path.join(DIR_POI_ALL, f"{safe}_poi_all.geojson"),
    )


def _locate_and_deploy_cached_file_debug(expected_path: str, suffix: str, place: str, required_tokens: Optional[List[str]] = None):
    info: Dict[str, object] = {
        "expected_path": expected_path,
        "used_path": None,
        "found": False,
        "action": "not_found",
        "candidates": []
    }
    if os.path.exists(expected_path):
        info.update({"used_path": expected_path, "found": True, "action": "exists"})
        return expected_path, info

    place_tokens = re.findall(r"[\w\u0400-\u04FF]+", place.lower())
    required = [t for t in (required_tokens or []) if t]

    search_bases = [os.path.dirname(expected_path)] + EXTRA_SEARCH_DIRS
    scored: List[Tuple[int, str]] = []
    for base in search_bases:
        try:
            pattern = os.path.join(base, f"*{suffix}")
            for p in glob.glob(pattern):
                fname_raw = os.path.basename(p).lower()
                fname = _norm_name(fname_raw)
                score = sum(1 for t in place_tokens if t and t in fname)
                # --- строгий фільтр по required-токенах
                if required and not all(t in fname for t in required):
                    continue
                scored.append((score, p))
        except Exception:
            logger.debug("_locate: skip base %s due to error", base)

    scored.sort(key=lambda x: (-x[0], x[1]))
    info["candidates"] = [p for _, p in scored]

    if not scored or scored[0][0] <= 0:
        return None, info

    best = scored[0][1]
    try:
        ensure_dir(os.path.dirname(expected_path))
        shutil.copy2(best, expected_path)
        logger.info("Deployed external cache file %s -> %s (score=%d)", best, expected_path, scored[0][0])
        info.update({"used_path": expected_path, "found": True, "action": "copied"})
        return expected_path, info
    except Exception as e:
        logger.warning("Failed to copy external cache %s -> %s: %s", best, expected_path, e)
        info.update({"used_path": best, "found": True, "action": "fallback_read"})
        return best, info


def _to_polygon(place: str) -> BaseGeometry:
    gdf = ox.geocode_to_gdf(place)  # OSMnx кешує
    return gdf.geometry.unary_union


def _build_tags_dict(categories: Dict[str, Iterable[str]]) -> Dict[str, List[str]]:
    tags = {}
    for key, vals in categories.items():
        uniq = sorted({str(v).strip() for v in vals if v is not None and str(v).strip()})
        if uniq:
            tags[key] = uniq
    return tags


def _representative_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(
        lambda geom: geom.representative_point() if not isinstance(geom, Point) else geom
    )
    return gdf


def _infer_primary_tag_columns(gdf: gpd.GeoDataFrame, keys_priority: List[str]) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["poi_key"] = pd.NA
    gdf["poi_value"] = pd.NA
    for k in keys_priority:
        if k in gdf.columns:
            mask = gdf["poi_key"].isna() & gdf[k].notna()
            gdf.loc[mask, "poi_key"] = k
            gdf.loc[mask, "poi_value"] = gdf.loc[mask, k].astype(str)
    return gdf


def _drop_dups(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    subset_cols = [c for c in ["osmid", "name"] if c in gdf.columns]
    return gdf.drop_duplicates(subset=subset_cols + ["geometry"]) if subset_cols else gdf.drop_duplicates("geometry")


def _to_crs_4326(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    try:
        return g.to_crs(4326)
    except Exception:
        g = g.copy()
        g.set_crs(4326, inplace=True)
        return g


def _tags_signature(tags: Dict[str, List[str]]) -> str:
    norm = {k: sorted({str(v).strip() for v in vals if v is not None and str(v).strip()})
            for k, vals in tags.items()}
    payload = json.dumps(dict(sorted(norm.items())), ensure_ascii=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]


# ============================================================
# 1) Перший запуск: завантаження з OSM + збереження
# ============================================================
def fetch_all_poi_from_osm(place: str,
                           categories: Dict[str, List[str]] = POI_CATEGORIES) -> gpd.GeoDataFrame:
    polygon = _to_polygon(place)
    tags = _build_tags_dict(categories)
    logger.info("fetch_all_poi_from_osm: features_from_polygon для '%s'", place)
    g = ox.features_from_polygon(polygon, tags)

    if g.empty:
        logger.info("fetch_all_poi_from_osm: порожньо для '%s'", place)
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    g = _to_crs_4326(g)
    g = _representative_points(g)
    g = _infer_primary_tag_columns(g, list(categories.keys()))
    g = _drop_dups(g)

    for col in ["name", "brand", "addr:street", "addr:housenumber"]:
        if col in g.columns:
            g[col] = g[col].astype(str)

    logger.info("fetch_all_poi_from_osm: отримано %d об'єктів для '%s'", len(g), place)
    return g


def save_boundary(place: str) -> Optional[str]:
    boundary_path, _ = _paths(place)
    try:
        boundary = ox.geocode_to_gdf(place).to_crs(4326)
        boundary.to_file(boundary_path, driver="GeoJSON")
        logger.info("boundary saved: %s", boundary_path)
        return boundary_path
    except Exception as e:
        logger.warning("boundary save failed for %s: %s", place, e)
        return None


def save_all_poi(gdf: gpd.GeoDataFrame, place: str, split_by_topkey: bool = False) -> None:
    _, poi_all_path = _paths(place)

    if gdf.empty:
        logger.warning("save_all_poi: empty -> skip save for %s", place)
        return

    gdf.reset_index(drop=True).to_file(poi_all_path, driver="GeoJSON")
    logger.info("saved %s (%d features)", poi_all_path, len(gdf))

    if split_by_topkey and "poi_key" in gdf.columns:
        safe = _safe_name(place)
        for k in sorted(gdf["poi_key"].dropna().unique()):
            part = gdf[gdf["poi_key"] == k].reset_index(drop=True)
            pth = os.path.join(DIR_LEVELS, f"{safe}_poi_{k}.geojson")
            part.to_file(pth, driver="GeoJSON")
            logger.info("saved %s (%d features)", pth, len(part))


# ============================================================
# 2) Повторний запуск: читання з кешу (debug, з фолбеком без pyogrio)
# ============================================================
def load_boundary_from_cache_debug(place: str, required_tokens: Optional[List[str]] = None):
    boundary_path, _ = _paths(place)
    located, info = _locate_and_deploy_cached_file_debug(boundary_path, "_boundary.geojson", place, required_tokens=required_tokens)
    if located is None:
        return None, info
    try:
        try:
            g = gpd.read_file(located, engine="pyogrio")
        except Exception:
            g = gpd.read_file(located)
        logger.info("loaded boundary from cache: %s", located)
        return g.to_crs(4326), info
    except Exception as e:
        logger.warning("boundary read failed for %s: %s", place, e)
        return None, info


def load_all_poi_from_cache_debug(place: str, required_tokens: Optional[List[str]] = None):
    _, poi_all_path = _paths(place)
    located, info = _locate_and_deploy_cached_file_debug(poi_all_path, "_poi_all.geojson", place, required_tokens=required_tokens)
    if located is None:
        return None, info
    try:
        try:
            g = gpd.read_file(located, engine="pyogrio")
        except Exception:
            g = gpd.read_file(located)
        logger.info("loaded all-poi from cache: %s", located)
        return _to_crs_4326(g), info
    except Exception as e:
        logger.warning("poi read failed for %s: %s", place, e)
        return None, info


def load_boundary_from_cache(place: str) -> Optional[gpd.GeoDataFrame]:
    g, _ = load_boundary_from_cache_debug(place)
    return g


def load_all_poi_from_cache(place: str) -> Optional[gpd.GeoDataFrame]:
    g, _ = load_all_poi_from_cache_debug(place)
    return g


# ============================================================
# 3) Оркестратор з описом джерела
# ============================================================
def get_poi_with_source(place: str,
                        split_by_topkey: bool = False,
                        force_refresh: bool = False,
                        required_tokens: Optional[List[str]] = None):
    boundary_path, poi_all_path = _paths(place)
    info = {"expected_path": poi_all_path, "used_path": None, "candidates": []}

    if not force_refresh:
        cached, cinfo = load_all_poi_from_cache_debug(place, required_tokens=required_tokens)
        info.update({"used_path": cinfo.get("used_path"), "candidates": cinfo.get("candidates", [])})
        if cached is not None:
            info["action"] = "cache"
            return cached, info

    # кеш не знайшли — качаємо та зберігаємо
    save_boundary(place)  # ок, якщо не збережеться
    gdf = fetch_all_poi_from_osm(place, POI_CATEGORIES)
    try:
        save_all_poi(gdf, place, split_by_topkey=split_by_topkey)
    finally:
        info["used_path"] = poi_all_path
        info["action"] = "osm"
    return gdf, info
