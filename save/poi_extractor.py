# poi_extractor.py
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import geopandas as gpd
import osmnx as ox

from config import POI_CATEGORIES
from logger_config import logger
from cache_manager import locate_and_deploy_cached_file_debug
from paths import DIR_BOUNDARIES, DIR_POI_ALL, DIR_POI_LEVELS, ensure_data_dirs
from utils import safe_name
from osmnx_client import ensure_osmnx_settings, fetch_all_poi_from_osm, to_crs_4326
from types import SourceInfo


# ============================================================
# Paths & IO helpers
# ============================================================
def _paths(place: str) -> Tuple[str, str]:
    safe = safe_name(place)
    boundary_path = os.path.join(DIR_BOUNDARIES, f"{safe}_boundary.geojson")
    poi_all_path = os.path.join(DIR_POI_ALL, f"{safe}_poi_all.geojson")
    return boundary_path, poi_all_path


def _read_geo(path: str) -> gpd.GeoDataFrame:
    try:
        return gpd.read_file(path, engine="pyogrio")
    except Exception:
        return gpd.read_file(path)


# ============================================================
# Boundary
# ============================================================
def save_boundary(place: str) -> Optional[str]:
    """Зберігає boundary (GeoJSON)."""
    ensure_data_dirs()
    ensure_osmnx_settings()

    boundary_path, _ = _paths(place)
    try:
        boundary = ox.geocode_to_gdf(place).to_crs(4326)
        boundary.to_file(boundary_path, driver="GeoJSON")
        logger.info("boundary saved: %s", boundary_path)
        return boundary_path
    except Exception as e:
        logger.warning("boundary save failed for %s: %s", place, e)
        return None


def load_boundary_from_cache_debug(
    place: str,
    required_tokens: Optional[List[str]] = None,
) -> Tuple[Optional[gpd.GeoDataFrame], SourceInfo]:
    ensure_data_dirs()
    boundary_path, _ = _paths(place)

    located, cache_info = locate_and_deploy_cached_file_debug(
        expected_path=boundary_path,
        suffix="_boundary.geojson",
        place=place,
        required_tokens=required_tokens,
    )

    info: SourceInfo = {
        "place": place,
        "expected_path": boundary_path,
        "used_path": cache_info.get("used_path"),
        "candidates": cache_info.get("candidates", []),
        "cache_action": cache_info.get("action"),
        "source": "cache",
    }

    if located is None or not os.path.exists(located):
        return None, info

    try:
        g = _read_geo(located)
        logger.info("loaded boundary from cache: %s", located)
        return g.to_crs(4326), info
    except Exception as e:
        logger.warning("boundary read failed for %s: %s", place, e)
        return None, info


def load_boundary_from_cache(place: str) -> Optional[gpd.GeoDataFrame]:
    g, _ = load_boundary_from_cache_debug(place)
    return g


# ============================================================
# POI cache
# ============================================================
def save_all_poi(
    gdf: gpd.GeoDataFrame,
    place: str,
    split_by_topkey: bool = False,
) -> None:
    """Зберігає всі POI одним GeoJSON (+ опційно розбиває по poi_key)."""
    ensure_data_dirs()
    _, poi_all_path = _paths(place)

    if gdf.empty:
        logger.warning("save_all_poi: empty -> skip save for %s", place)
        return

    gdf.reset_index(drop=True).to_file(poi_all_path, driver="GeoJSON")
    logger.info("saved %s (%d features)", poi_all_path, len(gdf))

    if split_by_topkey and "poi_key" in gdf.columns:
        safe = safe_name(place)
        for k in sorted(gdf["poi_key"].dropna().unique()):
            part = gdf[gdf["poi_key"] == k].reset_index(drop=True)
            pth = os.path.join(DIR_POI_LEVELS, f"{safe}_poi_{k}.geojson")
            part.to_file(pth, driver="GeoJSON")
            logger.info("saved %s (%d features)", pth, len(part))


def load_all_poi_from_cache_debug(
    place: str,
    required_tokens: Optional[List[str]] = None,
) -> Tuple[Optional[gpd.GeoDataFrame], SourceInfo]:
    ensure_data_dirs()
    _, poi_all_path = _paths(place)

    located, cache_info = locate_and_deploy_cached_file_debug(
        expected_path=poi_all_path,
        suffix="_poi_all.geojson",
        place=place,
        required_tokens=required_tokens,
    )

    info: SourceInfo = {
        "place": place,
        "expected_path": poi_all_path,
        "used_path": cache_info.get("used_path"),
        "candidates": cache_info.get("candidates", []),
        "cache_action": cache_info.get("action"),
        "source": "cache",
    }

    if located is None or not os.path.exists(located):
        return None, info

    try:
        g = _read_geo(located)
        logger.info("loaded all-poi from cache: %s", located)
        return to_crs_4326(g), info
    except Exception as e:
        logger.warning("poi read failed for %s: %s", place, e)
        return None, info


def load_all_poi_from_cache(place: str) -> Optional[gpd.GeoDataFrame]:
    g, _ = load_all_poi_from_cache_debug(place)
    return g


# ============================================================
# Public API
# ============================================================
def get_poi_with_source(
    place: str,
    split_by_topkey: bool = False,
    force_refresh: bool = False,
    required_tokens: Optional[List[str]] = None,
) -> Tuple[gpd.GeoDataFrame, SourceInfo]:
    """
    Повертає (gdf, SourceInfo)
    """
    ensure_data_dirs()
    _, poi_all_path = _paths(place)

    # ---------- пробуємо кеш ----------
    if not force_refresh:
        cached, info = load_all_poi_from_cache_debug(
            place,
            required_tokens=required_tokens,
        )
        if cached is not None:
            return cached, info

    # ---------- тягнемо з OSM ----------
    ensure_osmnx_settings()

    # boundary — best effort
    save_boundary(place)

    gdf = fetch_all_poi_from_osm(place, POI_CATEGORIES)

    try:
        save_all_poi(gdf, place, split_by_topkey=split_by_topkey)
    except Exception as e:
        logger.warning("poi save failed for %s: %s", place, e)

    info: SourceInfo = {
        "place": place,
        "source": "osm",
        "expected_path": poi_all_path,
        "used_path": poi_all_path,
        "candidates": [],
    }

    return gdf, info
