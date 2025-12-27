# api.py
from __future__ import annotations

from typing import List, Optional, Tuple

import geopandas as gpd
import networkx as nx

from graph_builder import get_city_graph_with_source
from poi_extractor import get_poi_with_source, load_boundary_from_cache_debug
from path_finder import compute_isochrone, find_shortest_path
from types import Algorithm, LatLon, SourceInfo


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
    """
    Завантажує граф міста (кеш або OSM).
    UI працює тільки з цією функцією.
    """
    G, info = get_city_graph_with_source(
        city_name=city,
        network_type=network_type,
        required_tokens=required_tokens,
        force_refresh=force_refresh,
    )
    return G, _normalize_info(info, place_fallback=city)


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
    """
    Завантажує POI для міста.
    """
    gdf, info = get_poi_with_source(
        place=city,
        required_tokens=required_tokens,
        force_refresh=force_refresh,
        split_by_topkey=split_by_topkey,
    )
    return gdf, _normalize_info(info, place_fallback=city)


def load_city_boundary(
    city: str,
    *,
    required_tokens: Optional[List[str]] = None,
) -> Tuple[Optional[gpd.GeoDataFrame], SourceInfo]:
    """
    Завантажує boundary міста з кешу.
    Boundary з OSM тут НЕ качаємо — це свідомо.
    """
    gdf, info = load_boundary_from_cache_debug(
        city,
        required_tokens=required_tokens,
    )
    return gdf, _normalize_info(info, place_fallback=city)


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
):
    """
    Побудова маршруту між двома точками.
    """
    return find_shortest_path(
        G,
        start_latlon=start,
        end_latlon=end,
        algorithm=algorithm,
        weight=weight,
    )


def build_isochrone(
    G: nx.MultiDiGraph,
    *,
    center: LatLon,
    cutoff: float,
    weight: str = "length",
):
    """
    Побудова ізохрони.
    """
    return compute_isochrone(
        G,
        center_latlon=center,
        cutoff=cutoff,
        weight=weight,
    )


# ============================================================
# Internal helpers
# ============================================================
def _normalize_info(info: object, *, place_fallback: str) -> SourceInfo:
    """
    Тонкий нормалізатор: переважно no-op.
    Потрібен як запобіжник, якщо десь ще повернеться старий dict-формат.
    """
    if isinstance(info, dict):
        out: SourceInfo = dict(info)  # type: ignore

        # гарантуємо, що place є завжди
        out.setdefault("place", place_fallback)

        # якщо ще десь залишилось старе поле action -> мапимо
        if "source" not in out and out.get("action") in ("cache", "osm"):
            out["source"] = out["action"]  # type: ignore

        # старий cache_action міг приїхати як action (exists/copied/...)
        if "cache_action" not in out and out.get("action") in ("exists", "copied", "fallback_read", "not_found"):
            out["cache_action"] = out["action"]  # type: ignore

        return out

    # якщо раптом прилетіло щось дивне — повертаємо мінімальний контракт
    return {"place": place_fallback, "source": "osm"}
