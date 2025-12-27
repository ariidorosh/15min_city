# poi_extractor.py
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import osmnx as ox

from cache_manager import locate_and_deploy_cached_file_debug
from config import POI_CATEGORIES
from logger_config import logger
from osmnx_client import ensure_osmnx_settings, fetch_all_poi_from_osm, to_crs_4326
from paths import DIR_BOUNDARIES, DIR_POI_ALL, DIR_POI_LEVELS, ensure_data_dirs
from utils import safe_name


def _paths(place: str) -> Tuple[str, str]:
    safe = safe_name(place)
    boundary_path = os.path.join(DIR_BOUNDARIES, f"{safe}_boundary.geojson")
    poi_all_path = os.path.join(DIR_POI_ALL, f"{safe}_poi_all.geojson")
    return boundary_path, poi_all_path


def _read_geo(path: str) -> gpd.GeoDataFrame:
    """Читання GeoJSON з fallback, якщо pyogrio недоступний."""
    try:
        return gpd.read_file(path, engine="pyogrio")
    except Exception:
        return gpd.read_file(path)


def save_boundary(place: str) -> Optional[str]:
    """Качає boundary з OSM і зберігає в кеш (GeoJSON)."""
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


def save_all_poi(gdf: gpd.GeoDataFrame, place: str, split_by_topkey: bool = False) -> Optional[str]:
    """Зберігає всі POI одним GeoJSON (+ опційно розбиває по poi_key)."""
    ensure_data_dirs()
    _, poi_all_path = _paths(place)

    if gdf is None or gdf.empty:
        logger.warning("save_all_poi: empty -> skip save for %s", place)
        return None

    gdf.reset_index(drop=True).to_file(poi_all_path, driver="GeoJSON")
    logger.info("saved %s (%d features)", poi_all_path, len(gdf))

    if split_by_topkey and "poi_key" in gdf.columns:
        safe = safe_name(place)
        for k in sorted(gdf["poi_key"].dropna().unique()):
            part = gdf[gdf["poi_key"] == k].reset_index(drop=True)
            pth = os.path.join(DIR_POI_LEVELS, f"{safe}_poi_{k}.geojson")
            part.to_file(pth, driver="GeoJSON")
            logger.info("saved %s (%d features)", pth, len(part))

    return poi_all_path


def load_boundary_from_cache_debug(
    place: str,
    required_tokens: Optional[List[str]] = None,
) -> Tuple[Optional[gpd.GeoDataFrame], Dict[str, object]]:
    """Тільки кеш. Якщо кешу нема — повертає (None, info)."""
    ensure_data_dirs()
    boundary_path, _ = _paths(place)

    located, cinfo = locate_and_deploy_cached_file_debug(
        expected_path=boundary_path,
        suffix="_boundary.geojson",
        place=place,
        required_tokens=required_tokens,
    )

    info: Dict[str, object] = {
        "place": place,
        "source": "cache",
        "cache_action": cinfo.get("action"),
        "expected_path": cinfo.get("expected_path", boundary_path),
        "used_path": cinfo.get("used_path"),
        "candidates": cinfo.get("candidates", []),
    }

    if not located or not os.path.exists(located):
        return None, info

    try:
        g = _read_geo(located)
        logger.info("loaded boundary from cache: %s", located)
        return to_crs_4326(g), info
    except Exception as e:
        logger.warning("boundary read failed for %s: %s", place, e)
        return None, info


def load_all_poi_from_cache_debug(
    place: str,
    required_tokens: Optional[List[str]] = None,
) -> Tuple[Optional[gpd.GeoDataFrame], Dict[str, object]]:
    """Тільки кеш. Якщо кешу нема — повертає (None, info)."""
    ensure_data_dirs()
    _, poi_all_path = _paths(place)

    located, cinfo = locate_and_deploy_cached_file_debug(
        expected_path=poi_all_path,
        suffix="_poi_all.geojson",
        place=place,
        required_tokens=required_tokens,
    )

    info: Dict[str, object] = {
        "place": place,
        "source": "cache",
        "cache_action": cinfo.get("action"),
        "expected_path": cinfo.get("expected_path", poi_all_path),
        "used_path": cinfo.get("used_path"),
        "candidates": cinfo.get("candidates", []),
    }

    if not located or not os.path.exists(located):
        return None, info

    try:
        g = _read_geo(located)
        logger.info("loaded all-poi from cache: %s", located)
        return to_crs_4326(g), info
    except Exception as e:
        logger.warning("poi read failed for %s: %s", place, e)
        return None, info


def load_boundary_from_cache(place: str) -> Optional[gpd.GeoDataFrame]:
    g, _ = load_boundary_from_cache_debug(place)
    return g


def load_all_poi_from_cache(place: str) -> Optional[gpd.GeoDataFrame]:
    g, _ = load_all_poi_from_cache_debug(place)
    return g


def get_poi_with_source(
    place: str,
    *,
    split_by_topkey: bool = False,
    force_refresh: bool = False,
    required_tokens: Optional[List[str]] = None,
) -> Tuple[gpd.GeoDataFrame, Dict[str, object]]:
    """POI для міста: кеш або OSM."""
    ensure_data_dirs()
    _, poi_all_path = _paths(place)

    cache_info: Optional[Dict[str, object]] = None
    if not force_refresh:
        cached, cinfo = load_all_poi_from_cache_debug(place, required_tokens=required_tokens)
        cache_info = cinfo
        if cached is not None:
            return cached, cinfo

    # Кеш не спрацював — тягнемо з OSM
    ensure_osmnx_settings()
    save_boundary(place)  # ок, якщо не збережеться

    gdf = fetch_all_poi_from_osm(place, POI_CATEGORIES)
    saved_path = save_all_poi(gdf, place, split_by_topkey=split_by_topkey) or poi_all_path

    info: Dict[str, object] = {
        "place": place,
        "source": "osm",
        "expected_path": poi_all_path,
        "used_path": saved_path,
        "candidates": (cache_info or {}).get("candidates", []),
        "cache_action": (cache_info or {}).get("cache_action"),
        "force_refresh": force_refresh,
    }

    return gdf, info
