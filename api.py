# api.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import networkx as nx

from graph_builder import get_city_graph_with_source
from poi_extractor import get_poi_with_source, load_boundary_from_cache_debug
from path_finder import compute_isochrone, find_shortest_path
from app_types import Algorithm, LatLon, SourceInfo


# ============================================================
# Graph
# ============================================================
def load_city_graph(
    city: str,
    *,
    network_type: str = "walk",
    required_tokens: Optional[List[str]] = None,
    force_refresh: bool = False,
) -> Tuple[nx.MultiDiGraph, SourceInfo]:
    """Завантажує граф міста (кеш або OSM). UI має працювати тільки з цією функцією."""
    G, info = get_city_graph_with_source(
        city_name=city,
        network_type=network_type,
        required_tokens=required_tokens,
        force_refresh=force_refresh,
    )
    return G, _normalize_info(info)


# ============================================================
# POI
# ============================================================
def load_poi(
    city: str,
    *,
    required_tokens: Optional[List[str]] = None,
    force_refresh: bool = False,
    split_by_topkey: bool = False,
) -> Tuple[gpd.GeoDataFrame, SourceInfo]:
    """Завантажує POI для міста (кеш або OSM)."""
    gdf, info = get_poi_with_source(
        place=city,
        required_tokens=required_tokens,
        force_refresh=force_refresh,
        split_by_topkey=split_by_topkey,
    )
    return gdf, _normalize_info(info)


def load_city_boundary(
    city: str,
    *,
    required_tokens: Optional[List[str]] = None,
) -> Tuple[Optional[gpd.GeoDataFrame], SourceInfo]:
    """Завантажує boundary міста з кешу. Boundary з OSM тут НЕ качаємо — свідомо."""
    gdf, info = load_boundary_from_cache_debug(city, required_tokens=required_tokens)
    return gdf, _normalize_info(info)


# ============================================================
# Routing
# ============================================================
def build_route(
    G: nx.MultiDiGraph,
    *,
    start: LatLon,
    end: LatLon,
    algorithm: Algorithm = "dijkstra",
    weight: str = "length",
    use_undirected: bool = True,
):
    return find_shortest_path(
        G,
        start_latlon=start,
        end_latlon=end,
        algorithm=algorithm,
        weight=weight,
        use_undirected=use_undirected,
    )


def build_isochrone(
    G: nx.MultiDiGraph,
    *,
    center: LatLon,
    cutoff: float,
    weight: str = "length",
):
    """Побудова ізохрони."""
    return compute_isochrone(
        G,
        center_latlon=center,
        cutoff=cutoff,
        weight=weight,
    )


# ============================================================
# Internal helpers
# ============================================================
def _normalize_info(info: Dict[str, Any]) -> SourceInfo:
    """Приводить будь-який внутрішній info до єдиного контракту SourceInfo."""
    out: SourceInfo = {}

    # 1) Новий формат (graph_builder/poi_extractor вже так повертають)
    if "place" in info:
        out["place"] = str(info["place"])
    if "source" in info:
        out["source"] = info["source"]  # type: ignore
    if "cache_action" in info and info.get("cache_action") is not None:
        out["cache_action"] = info["cache_action"]  # type: ignore
    if "expected_path" in info and info.get("expected_path"):
        out["expected_path"] = str(info["expected_path"])
    if "used_path" in info and info.get("used_path"):
        out["used_path"] = str(info["used_path"])
    if "candidates" in info and info.get("candidates"):
        out["candidates"] = list(info["candidates"])  # type: ignore

    if out.get("source"):
        return out

    # 2) Легасі fallback: якщо десь ще прилітає "action"
    action = info.get("action")
    if action in ("cache", "osm"):
        out["source"] = action  # type: ignore
    elif action in ("exists", "copied", "fallback_read", "not_found"):
        out["source"] = "cache"
        out["cache_action"] = action  # type: ignore
    else:
        out["source"] = "osm"

    if "place" in info:
        out["place"] = str(info["place"])
    if "expected_path" in info and info.get("expected_path"):
        out["expected_path"] = str(info["expected_path"])
    if "used_path" in info and info.get("used_path"):
        out["used_path"] = str(info["used_path"])
    if "candidates" in info and info.get("candidates"):
        out["candidates"] = list(info["candidates"])  # type: ignore

    return out
