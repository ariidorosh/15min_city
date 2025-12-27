# osmnx_client.py
from __future__ import annotations

from typing import Dict, Iterable, List

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from config import POI_CATEGORIES
from logger_config import logger
from paths import DIR_CACHE


_OSMNX_CONFIGURED = False


def ensure_osmnx_settings() -> None:
    """Лінива конфігурація OSMnx: викликати перед будь-яким ox.*."""
    global _OSMNX_CONFIGURED
    if _OSMNX_CONFIGURED:
        return

    ox.settings.use_cache = True
    ox.settings.cache_folder = DIR_CACHE
    ox.settings.log_console = False

    _OSMNX_CONFIGURED = True


def to_polygon(place: str) -> BaseGeometry:
    ensure_osmnx_settings()
    gdf = ox.geocode_to_gdf(place)
    return gdf.geometry.unary_union


def build_tags_dict(categories: Dict[str, Iterable[str]]) -> Dict[str, List[str]]:
    tags: Dict[str, List[str]] = {}
    for key, vals in categories.items():
        uniq = sorted({str(v).strip() for v in vals if v is not None and str(v).strip()})
        if uniq:
            tags[key] = uniq
    return tags


def to_crs_4326(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.copy()
        gdf.set_crs(4326, inplace=True)
        return gdf
    return gdf.to_crs(4326)


def representative_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(
        lambda geom: geom.representative_point() if not isinstance(geom, Point) else geom
    )
    return gdf


def infer_primary_tag_columns(gdf: gpd.GeoDataFrame, keys_priority: List[str]) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["poi_key"] = pd.NA
    gdf["poi_value"] = pd.NA

    for k in keys_priority:
        if k in gdf.columns:
            mask = gdf["poi_key"].isna() & gdf[k].notna()
            gdf.loc[mask, "poi_key"] = k
            gdf.loc[mask, "poi_value"] = gdf.loc[mask, k].astype(str)

    return gdf


def drop_dups(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    subset_cols = [c for c in ["osmid", "name"] if c in gdf.columns]
    if subset_cols:
        return gdf.drop_duplicates(subset=subset_cols + ["geometry"])
    return gdf.drop_duplicates("geometry")


def fetch_all_poi_from_osm(place: str, categories: Dict[str, List[str]] = POI_CATEGORIES) -> gpd.GeoDataFrame:
    ensure_osmnx_settings()

    polygon = to_polygon(place)
    tags = build_tags_dict(categories)

    logger.info("fetch_all_poi_from_osm: features_from_polygon для '%s'", place)
    gdf = ox.features_from_polygon(polygon, tags)

    if gdf.empty:
        logger.info("fetch_all_poi_from_osm: порожньо для '%s'", place)
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    gdf = to_crs_4326(gdf)
    gdf = representative_points(gdf)
    gdf = infer_primary_tag_columns(gdf, list(categories.keys()))
    gdf = drop_dups(gdf)

    for col in ["name", "brand", "addr:street", "addr:housenumber"]:
        if col in gdf.columns:
            gdf[col] = gdf[col].astype(str)

    logger.info("fetch_all_poi_from_osm: отримано %d об'єктів для '%s'", len(gdf), place)
    return gdf
